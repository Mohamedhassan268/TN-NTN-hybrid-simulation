"""Canonical matched-scenario generator for the revision-v2 study.

One trajectory contains the complete decision context and realized link outcomes
for all five candidate networks.  Geometry and weather are shared across Ku, Ka,
and S campaigns; only the LEO ``TechConfig`` changes by band.  Generated rows are
therefore paired by ``(trajectory_id, step_index, network_type)``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import numpy as np
import pandas as pd

from noise_models.constants import C_LIGHT
from noise_models.link import simulate_trace
from noise_models.technologies import get_tech_config

from .features import AREAS, NETWORK_TYPES

DATASET_VERSION = "2.0.0"
DECISION_INTERVAL_S = 10.0
EPISODE_STEPS = 60
EARTH_RADIUS_KM = 6371.0
EARTH_MU_M3_S2 = 3.986004418e14
MIN_VISIBLE_ELEV_DEG = 10.0
BANDS = ("ku", "ka", "s")
BAND_TO_LEO_CONFIG = {"ku": "LEO", "ka": "LEO_Ka", "s": "LEO_S"}
NETWORK_TO_CONFIG = {
    "HAPS": "HAPS",
    "NR_5G": "5G_NR",
    "SAT (LEO)": "LEO",
    "UAV": "UAV",
    "WiFi": "WiFi6_5GHz",
}

# Availability is contextual; geometric visibility is applied separately.
AREA_NETWORKS = {
    "Desert": {"HAPS", "SAT (LEO)", "UAV"},
    "Highway": {"HAPS", "NR_5G", "SAT (LEO)", "UAV"},
    "Indoor": {"NR_5G", "SAT (LEO)", "WiFi"},
    "Maritime": {"HAPS", "SAT (LEO)", "UAV"},
    "Rural": {"HAPS", "NR_5G", "SAT (LEO)", "UAV"},
    "Urban": set(NETWORK_TYPES),
}

# Ten-second transition probabilities. Mean dwell time is about 11--17 minutes.
AREA_TRANSITION = {
    "Indoor": {"Indoor": 0.990, "Urban": 0.010},
    "Urban": {"Urban": 0.985, "Indoor": 0.005, "Highway": 0.010},
    "Highway": {"Highway": 0.985, "Urban": 0.010, "Rural": 0.005},
    "Rural": {"Rural": 0.985, "Highway": 0.005, "Desert": 0.005, "Maritime": 0.005},
    "Desert": {"Desert": 0.990, "Rural": 0.010},
    "Maritime": {"Maritime": 0.990, "Rural": 0.010},
}

GEOMETRY_REGIMES = {
    "nominal": {"altitude_km": (500.0, 600.0), "peak_elevation_deg": (30.0, 90.0)},
    "low_shell": {"altitude_km": (350.0, 450.0), "peak_elevation_deg": (30.0, 90.0)},
    "high_shell": {"altitude_km": (900.0, 1200.0), "peak_elevation_deg": (30.0, 90.0)},
    "low_elevation": {"altitude_km": (500.0, 600.0), "peak_elevation_deg": (10.0, 30.0)},
}

OUTCOME_MAP = {
    "RSSI_dBm": "rssi_dbm",
    "SNR_dB": "snr_db",
    "SINR_dB": "sinr_db",
    "BER": "ber",
    "BLER": "bler",
    "Throughput_Mbps": "throughput_mbps",
    "Propagation_Delay_ms": "propagation_delay_ms",
    "Latency_ms": "latency_ms",
    "Packet_Loss_pct": "packet_loss_pct",
    "Spectral_Efficiency_bps_hz": "spectral_efficiency_bps_hz",
    "Link_Quality_Index": "link_quality_index",
    "MCS_Order": "mcs_order",
    "Code_Rate": "code_rate",
    "Spectral_Efficiency_Saturated": "spectral_efficiency_saturated",
}


@dataclass(frozen=True)
class ScenarioSpec:
    trajectory_id: str
    scenario_id: str
    ue_id: str
    split: str
    seed: int
    geometry_regime: str = "nominal"
    n_steps: int = EPISODE_STEPS
    decision_interval_s: float = DECISION_INTERVAL_S


@dataclass(frozen=True)
class EpisodeScenario:
    """In-memory, one-band view of one canonical trajectory."""

    trajectory_id: str
    scenario_id: str
    band: str
    frame: pd.DataFrame

    @property
    def n_steps(self) -> int:
        return int(self.frame["step_index"].nunique())


def split_for_index(index: int) -> str:
    if index < 800:
        return "train"
    if index < 1000:
        return "validation"
    if index < 1200:
        return "test"
    raise ValueError("canonical dataset contains exactly 1,200 trajectories")


def canonical_specs(total: int = 1200, base_seed: int = 20260912) -> list[ScenarioSpec]:
    if not 1 <= total <= 1200:
        raise ValueError("total must be between 1 and 1200")
    return [
        ScenarioSpec(
            trajectory_id=f"trajectory_v2_{i:04d}",
            scenario_id=f"scenario_v2_{i:04d}",
            ue_id=f"ue_v2_{i:04d}",
            split=split_for_index(i),
            seed=base_seed + i,
        )
        for i in range(total)
    ]


def _bounded_walk(rng, n, low, high, step_fraction=0.025):
    values = np.empty(n, dtype=float)
    values[0] = rng.uniform(low, high)
    sigma = (high - low) * step_fraction
    for i in range(1, n):
        values[i] = np.clip(values[i - 1] + rng.normal(0.0, sigma), low, high)
    return values


def _rain_series(rng, n):
    rain = np.zeros(n, dtype=float)
    active = False
    rate = 0.0
    for i in range(n):
        if not active and rng.random() < 0.025:
            active, rate = True, rng.uniform(2.0, 50.0)
        elif active and rng.random() < 0.12:
            active, rate = False, 0.0
        if active:
            rate = float(np.clip(rate + rng.normal(0.0, 1.0), 0.1, 50.0))
            rain[i] = rate
    return rain


def sample_area_sequence(rng: np.random.Generator, n_steps: int) -> list[str]:
    sequence = [str(rng.choice(AREAS))]
    for _ in range(1, n_steps):
        transitions = AREA_TRANSITION[sequence[-1]]
        sequence.append(str(rng.choice(list(transitions), p=list(transitions.values()))))
    return sequence


def _peak_central_angle_rad(altitude_km: float, peak_elevation_deg: float) -> float:
    elevation = np.deg2rad(peak_elevation_deg)
    orbital_radius = EARTH_RADIUS_KM + altitude_km
    slant = -EARTH_RADIUS_KM * np.sin(elevation) + np.sqrt(
        (EARTH_RADIUS_KM * np.sin(elevation)) ** 2
        + orbital_radius**2
        - EARTH_RADIUS_KM**2
    )
    cos_psi = (
        EARTH_RADIUS_KM**2 + orbital_radius**2 - slant**2
    ) / (2.0 * EARTH_RADIUS_KM * orbital_radius)
    return float(np.arccos(np.clip(cos_psi, -1.0, 1.0)))


def spherical_leo_pass(
    rng: np.random.Generator,
    n_steps: int,
    decision_interval_s: float,
    regime: str = "nominal",
) -> dict[str, np.ndarray | float]:
    """Generate a spherical-Earth pass centered near the ten-minute episode."""
    params = GEOMETRY_REGIMES[regime]
    altitude_km = float(rng.uniform(*params["altitude_km"]))
    peak_elevation = float(rng.uniform(*params["peak_elevation_deg"]))
    psi_min = _peak_central_angle_rad(altitude_km, peak_elevation)
    orbital_radius_m = (EARTH_RADIUS_KM + altitude_km) * 1000.0
    omega = np.sqrt(EARTH_MU_M3_S2 / orbital_radius_m**3)
    times = np.arange(n_steps, dtype=float) * decision_interval_s
    closest_time = rng.uniform(times[0], times[-1])
    along_track = omega * (times - closest_time)
    central_angle = np.sqrt(psi_min**2 + along_track**2)
    orbital_radius_km = EARTH_RADIUS_KM + altitude_km
    slant_km = np.sqrt(
        EARTH_RADIUS_KM**2
        + orbital_radius_km**2
        - 2.0 * EARTH_RADIUS_KM * orbital_radius_km * np.cos(central_angle)
    )
    elevation = np.degrees(np.arctan2(
        orbital_radius_km * np.cos(central_angle) - EARTH_RADIUS_KM,
        orbital_radius_km * np.sin(central_angle),
    ))
    visible = elevation >= MIN_VISIBLE_ELEV_DEG
    radial_velocity_ms = np.gradient(slant_km * 1000.0, decision_interval_s)
    return {
        "altitude_km": altitude_km,
        "distance_km": slant_km,
        "elevation_deg": elevation,
        "visible": visible,
        "radial_velocity_ms": radial_velocity_ms,
    }


def _geometry(spec: ScenarioSpec, rng: np.random.Generator) -> tuple[list[str], dict]:
    n = spec.n_steps
    area = sample_area_sequence(rng, n)
    rain = _rain_series(rng, n)
    nr_distance = _bounded_walk(rng, n, 0.03, 3.0)
    nr_d2d_m = np.maximum(np.sqrt((nr_distance * 1000.0) ** 2 - 23.5**2), 10.0)
    nr_los_probability = np.minimum(
        18.0 / nr_d2d_m + np.exp(-nr_d2d_m / 63.0) * (1.0 - 18.0 / nr_d2d_m),
        1.0,
    )
    nr_los = rng.random(n) < nr_los_probability

    wifi_distance = _bounded_walk(rng, n, 0.001, 0.08)
    wifi_walls = rng.choice([0.0, 0.0, 12.0], size=n)

    uav_altitude = _bounded_walk(rng, n, 50.0, 300.0)
    uav_horizontal = _bounded_walk(rng, n, 0.05, 5.0)
    uav_distance = np.sqrt(uav_horizontal**2 + (uav_altitude / 1000.0) ** 2)
    uav_elevation = np.degrees(np.arctan2(uav_altitude / 1000.0, uav_horizontal))
    uav_los_probability = 1.0 / (1.0 + 9.61 * np.exp(-0.16 * (uav_elevation - 9.61)))

    haps_horizontal = _bounded_walk(rng, n, 5.0, 200.0)
    haps_distance = np.sqrt(haps_horizontal**2 + 20.0**2)
    haps_elevation = np.degrees(np.arctan2(20.0, haps_horizontal))

    leo = spherical_leo_pass(
        rng, n, spec.decision_interval_s, regime=spec.geometry_regime
    )
    speed = _bounded_walk(rng, n, 1.0, 20.0)
    return area, {
        "NR_5G": {
            "distance_km": nr_distance,
            "elevation_deg": np.zeros(n),
            "altitude_m": np.full(n, 1.5),
            "speed_ms": speed,
            "rain_rate_mmhr": rain,
            "los": nr_los,
            "visible": np.ones(n, dtype=bool),
            "wall_loss_db": np.zeros(n),
            "clutter_loss_db": np.zeros(n),
        },
        "WiFi": {
            "distance_km": wifi_distance,
            "elevation_deg": np.zeros(n),
            "altitude_m": np.full(n, 1.5),
            "speed_ms": np.minimum(speed, 2.0),
            "rain_rate_mmhr": np.zeros(n),
            "los": wifi_walls == 0,
            "visible": np.ones(n, dtype=bool),
            "wall_loss_db": wifi_walls,
            "clutter_loss_db": np.zeros(n),
        },
        "UAV": {
            "distance_km": uav_distance,
            "elevation_deg": uav_elevation,
            "altitude_m": uav_altitude,
            "speed_ms": speed,
            "rain_rate_mmhr": rain,
            "los": rng.random(n) < uav_los_probability,
            "visible": np.ones(n, dtype=bool),
            "wall_loss_db": np.zeros(n),
            "clutter_loss_db": np.zeros(n),
        },
        "HAPS": {
            "distance_km": haps_distance,
            "elevation_deg": haps_elevation,
            "altitude_m": np.full(n, 20_000.0),
            "speed_ms": speed,
            "rain_rate_mmhr": rain,
            "los": np.ones(n, dtype=bool),
            "visible": haps_elevation >= 5.0,
            "wall_loss_db": np.zeros(n),
            "clutter_loss_db": np.zeros(n),
        },
        "SAT (LEO)": {
            "distance_km": leo["distance_km"],
            "elevation_deg": leo["elevation_deg"],
            "altitude_m": np.full(n, float(leo["altitude_km"]) * 1000.0),
            "speed_ms": np.full(n, 7_500.0),
            "rain_rate_mmhr": rain,
            "los": np.ones(n, dtype=bool),
            "visible": leo["visible"],
            "radial_velocity_ms": leo["radial_velocity_ms"],
            "wall_loss_db": np.zeros(n),
            "clutter_loss_db": np.zeros(n),
        },
    }


def _config_for(network: str, band: str, n_steps: int):
    name = BAND_TO_LEO_CONFIG[band] if network == "SAT (LEO)" else NETWORK_TO_CONFIG[network]
    return replace(
        get_tech_config(name), n_steps=n_steps, step_s=DECISION_INTERVAL_S
    )


def generate_scenario(spec: ScenarioSpec, bands=BANDS) -> pd.DataFrame:
    """Generate one fully paired trajectory for all requested bands."""
    rng = np.random.default_rng(spec.seed)
    areas, geometry = _geometry(spec, rng)
    frames = []
    for band_index, band in enumerate(bands):
        if band not in BANDS:
            raise ValueError(f"unknown band: {band}")
        results = {}
        available_by_network = {}
        for network_index, network in enumerate(NETWORK_TYPES):
            geo = geometry[network]
            cfg = _config_for(network, band, spec.n_steps)
            radial_velocity = geo.get("radial_velocity_ms", np.zeros(spec.n_steps))
            doppler = -np.asarray(radial_velocity) * cfg.carrier_freq_hz / C_LIGHT
            if network != "SAT (LEO)":
                doppler = np.asarray(geo["speed_ms"]) * cfg.carrier_freq_hz / C_LIGHT
            # TN/HAPS/UAV conditions are identical across bands. Only the LEO
            # trace gets a band-specific random stream and physical config.
            paired_band_index = band_index if network == "SAT (LEO)" else 0
            trace_rng = np.random.default_rng(
                np.random.SeedSequence([spec.seed, paired_band_index, network_index])
            )
            scenario = {
                **geo,
                "doppler_char_hz": doppler,
                "platform_altitude_km": np.asarray(geo["altitude_m"]) / 1000.0,
            }
            results[network] = simulate_trace(scenario, cfg, trace_rng)
            available_by_network[network] = np.array([
                network in AREA_NETWORKS[area] and bool(geo["visible"][step])
                for step, area in enumerate(areas)
            ])

        masks = np.array([
            sum(
                (1 << network_index)
                for network_index, network in enumerate(NETWORK_TYPES)
                if available_by_network[network][step]
            )
            for step in range(spec.n_steps)
        ], dtype=np.int16)

        rows = []
        for network in NETWORK_TYPES:
            geo = geometry[network]
            trace = results[network]
            for step in range(spec.n_steps):
                available = bool(available_by_network[network][step])
                row = {
                    "dataset_version": DATASET_VERSION,
                    "scenario_id": spec.scenario_id,
                    "trajectory_id": spec.trajectory_id,
                    "ue_id": spec.ue_id,
                    "split": spec.split,
                    "step_index": step,
                    "time_s": step * spec.decision_interval_s,
                    "decision_interval_s": spec.decision_interval_s,
                    "band": band,
                    "geometry_regime": spec.geometry_regime,
                    "area": areas[step],
                    "network_type": network,
                    "available_mask": int(masks[step]),
                    "area_allowed": network in AREA_NETWORKS[areas[step]],
                    "visible": bool(geo["visible"][step]),
                    "available": available,
                    "distance_km": float(geo["distance_km"][step]),
                    "elevation_deg": float(geo["elevation_deg"][step]),
                    "altitude_m": float(geo["altitude_m"][step]),
                    "speed_mps": float(geo["speed_ms"][step]),
                    "doppler_hz": float(trace["Doppler_Hz"][step]),
                    "rain_rate_mm_h": float(geo["rain_rate_mmhr"][step]),
                }
                for source, target in OUTCOME_MAP.items():
                    value = trace[source][step]
                    row[target] = value.item() if hasattr(value, "item") else value
                    if not available:
                        row[target] = None
                if available:
                    row["log_throughput_mbps"] = float(np.log1p(row["throughput_mbps"]))
                    row["log_ber"] = float(-np.log10(row["ber"] + 1e-12))
                    row["log_packet_loss_pct"] = float(np.log1p(row["packet_loss_pct"]))
                else:
                    row["log_throughput_mbps"] = None
                    row["log_ber"] = None
                    row["log_packet_loss_pct"] = None
                rows.append(row)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def iter_scenarios(specs: list[ScenarioSpec], bands=BANDS) -> Iterator[pd.DataFrame]:
    for spec in specs:
        yield generate_scenario(spec, bands=bands)


def schema_definition() -> dict:
    return {
        "dataset_version": DATASET_VERSION,
        "primary_key": ["trajectory_id", "step_index", "band", "network_type"],
        "observation_order": {
            "offline": ["area_onehot[6]", "network[5].[available,rssi_norm,sinr_norm]"],
            "online": [
                "area_onehot[6]",
                "network[5].[available,rssi_norm,sinr_norm]",
                "previous_network_onehot[5]",
            ],
        },
        "units": {
            "time_s": "s",
            "decision_interval_s": "s",
            "distance_km": "km",
            "elevation_deg": "degree",
            "altitude_m": "m",
            "speed_mps": "m/s",
            "doppler_hz": "Hz",
            "rain_rate_mm_h": "mm/h",
            "rssi_dbm": "dBm",
            "snr_db": "dB",
            "sinr_db": "dB",
            "throughput_mbps": "Mbit/s",
            "propagation_delay_ms": "ms",
            "latency_ms": "ms",
            "packet_loss_pct": "%",
            "spectral_efficiency_bps_hz": "bit/s/Hz",
        },
    }
