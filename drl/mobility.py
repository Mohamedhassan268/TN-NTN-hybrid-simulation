"""Shared-mobility model for Phase 2+: generates ONE UE's geometry trajectory that is
shared across all candidate networks in an episode, so the agent can compare networks
at the same simulated moment/place instead of drawing each technology's KPIs from an
unrelated independent scenario (the gap documented in status.md).

Two families of per-technology geometry:
- TN (NR_5G, WiFi) and UAV: distance_km (/altitude_m,speed_ms for UAV) is a bounded
  random walk within ranges fit from the CSV per network_type - the UE moves around
  within a persistent local cell/AP/drone-relay, never fully out of range.
- NTN (SAT (LEO), HAPS): distance_km is DERIVED from an analytic elevation-angle pass
  profile (rise -> peak -> set, inverting noise_models/channels/pathloss.elevation_deg's
  flat-earth model: distance_km = platform_altitude_km / sin(elevation)). Below a
  minimum visible elevation the platform is out of view for that step - this produces
  a genuine, learnable "pass ending forces a handover" dynamic instead of a KPI that
  merely wanders. LEO passes are short and cyclic (satellite comes and goes); HAPS is
  modeled as a persistent regional platform (long passes, rare/no gaps) - both use the
  same generator with different period/peak-elevation calibration, reflecting their
  real operational difference (station-keeping vs. orbiting).

Area also evolves over the episode via a small hand-specified Markov chain (see
AREA_TRANSITION) instead of being fixed for the whole episode, so Available_Networks
changes over time and long-horizon handover reasoning has an actual reason to exist.

This does NOT model real orbital/ephemeris mechanics (no TLE data, no ground-track
geometry) - it is an analytic approximation sufficient to make handover cost, network
comparison, and forced-handover-on-pass-end meaningful without a full orbital simulator.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from noise_models.technologies import PLATFORM_ALTITUDE_KM

NETWORK_TO_TECH_CONFIG = {
    "NR_5G": "5G_NR",
    "WiFi": "WiFi6_5GHz",
    "SAT (LEO)": "LEO",
    "HAPS": "HAPS",
    "UAV": "UAV",
}

NTN_NETWORKS = {"SAT (LEO)", "HAPS"}
MIN_VISIBLE_ELEV_DEG = 10.0

# (peak_elev_range_deg, pass_len_range_steps, gap_len_range_steps)
_PASS_PARAMS = {
    "SAT (LEO)": {"peak_range": (15.0, 85.0), "pass_len_range": (8, 25), "gap_len_range": (5, 30)},
    "HAPS": {"peak_range": (25.0, 80.0), "pass_len_range": (40, 90), "gap_len_range": (0, 3)},
}

AREA_TRANSITION = {
    "Indoor":   {"Indoor": 0.90, "Urban": 0.10},
    "Urban":    {"Urban": 0.85, "Indoor": 0.05, "Highway": 0.10},
    "Highway":  {"Highway": 0.85, "Urban": 0.10, "Rural": 0.05},
    "Rural":    {"Rural": 0.85, "Highway": 0.05, "Desert": 0.05, "Maritime": 0.05},
    "Desert":   {"Desert": 0.90, "Rural": 0.10},
    "Maritime": {"Maritime": 0.90, "Rural": 0.10},
}


@dataclass
class GeometryRange:
    low: float
    high: float
    median: float


def fit_geometry_ranges(df: pd.DataFrame, low_pct: float = 1.0, high_pct: float = 99.0) -> dict:
    """Per network_type: distance_km / rain_rate_mmhr / doppler_hz / RSSI / SINR ranges,
    plus altitude_m / speed_ms for UAV. Returns {network_type: {field: GeometryRange}}."""
    ranges = {}
    for network, group in df.groupby("network_type"):
        fields = {}
        for col in ["distance_km", "Rain_Rate_mmhr", "Doppler_Hz", "RSSI_dBm", "SINR_dB"]:
            low, high = np.percentile(group[col], [low_pct, high_pct])
            fields[col] = GeometryRange(float(low), float(high), float(group[col].median()))
        if network == "UAV":
            for col in ["altitude_m", "speed_ms"]:
                low, high = np.percentile(group[col], [low_pct, high_pct])
                fields[col] = GeometryRange(float(low), float(high), float(group[col].median()))
        ranges[network] = fields
    return ranges


def sample_area_sequence(rng: np.random.Generator, n_steps: int, initial_area: str | None = None) -> list[str]:
    areas = list(AREA_TRANSITION.keys())
    if initial_area is None:
        initial_area = rng.choice(areas)
    seq = [initial_area]
    for _ in range(1, n_steps):
        trans = AREA_TRANSITION[seq[-1]]
        seq.append(rng.choice(list(trans.keys()), p=list(trans.values())))
    return seq


def _bounded_walk(rng: np.random.Generator, n_steps: int, low: float, high: float, step_frac: float = 0.03) -> np.ndarray:
    span = max(high - low, 1e-9)
    values = np.empty(n_steps)
    values[0] = rng.uniform(low, high)
    for t in range(1, n_steps):
        proposal = values[t - 1] + rng.normal(0.0, span * step_frac)
        values[t] = np.clip(proposal, low, high)
    return values


def _generate_pass_elevation(rng: np.random.Generator, n_steps: int, peak_range, pass_len_range, gap_len_range):
    """Sequence of rise-peak-set passes separated by out-of-view gaps. Returns
    (elevation_deg[n_steps], visible[n_steps] bool)."""
    elev = np.zeros(n_steps)
    visible = np.zeros(n_steps, dtype=bool)
    t = 0
    while t < n_steps:
        gap = min(int(rng.integers(gap_len_range[0], gap_len_range[1] + 1)), n_steps - t)
        t += gap
        if t >= n_steps:
            break
        pass_len = min(int(rng.integers(pass_len_range[0], pass_len_range[1] + 1)), n_steps - t)
        peak = rng.uniform(*peak_range)
        for i in range(pass_len):
            phase = i / max(pass_len - 1, 1)
            e = peak * np.sin(np.pi * phase)
            elev[t + i] = e
            visible[t + i] = e >= MIN_VISIBLE_ELEV_DEG
        t += pass_len
    return elev, visible


def _ntn_distance_from_elevation(elev_deg: np.ndarray, platform_altitude_km: float) -> np.ndarray:
    safe_elev = np.clip(elev_deg, 1.0, 90.0)
    return platform_altitude_km / np.sin(np.radians(safe_elev))


def generate_episode_trajectories(n_steps: int, geometry_ranges: dict, rng: np.random.Generator) -> dict:
    """Generates a trajectory for EVERY network type (not just those available in the
    initial Area) since Area now changes over the episode and any network may become
    relevant at some step. Returns {network_type: {"distance_km": array, "rain_rate_mmhr":
    array, "doppler_hz": float, "platform_altitude_km": float, "visible": array[bool],
    "altitude_m": array|None, "speed_ms": array|None}}."""
    trajectories = {}
    for network, fields in geometry_ranges.items():
        rain_r = fields["Rain_Rate_mmhr"]
        platform_altitude_km = PLATFORM_ALTITUDE_KM[NETWORK_TO_TECH_CONFIG[network]]

        if network in NTN_NETWORKS:
            params = _PASS_PARAMS[network]
            elev, visible = _generate_pass_elevation(
                rng, n_steps, params["peak_range"], params["pass_len_range"], params["gap_len_range"]
            )
            distance_km = _ntn_distance_from_elevation(elev, platform_altitude_km)
        else:
            dist_r = fields["distance_km"]
            distance_km = _bounded_walk(rng, n_steps, dist_r.low, dist_r.high)
            visible = np.ones(n_steps, dtype=bool)

        traj = {
            "distance_km": distance_km,
            "rain_rate_mmhr": _bounded_walk(rng, n_steps, rain_r.low, rain_r.high),
            "doppler_hz": fields["Doppler_Hz"].median,
            "platform_altitude_km": platform_altitude_km,
            "visible": visible,
        }
        if network == "UAV":
            traj["altitude_m"] = _bounded_walk(rng, n_steps, fields["altitude_m"].low, fields["altitude_m"].high)
            traj["speed_ms"] = _bounded_walk(rng, n_steps, fields["speed_ms"].low, fields["speed_ms"].high)
        else:
            traj["altitude_m"] = None
            traj["speed_ms"] = None
        trajectories[network] = traj
    return trajectories
