"""Large-scale propagation models used by the revision-v2 simulator.

The functions deliberately separate implemented standard-model components from
the surrounding custom simulator. Distances are metres inside 3GPP equations and
frequencies are GHz where those equations define them.

Implemented components:
* 3GPP TR 38.901 v17, Table 7.4.1-1: UMa LOS/NLOS path loss.
* 3GPP TR 36.777 v15, Table B-2: UMa-AV LOS/high-altitude NLOS terms.
* 3GPP TR 38.811 v15: free-space/basic NTN path-loss component.
* Wi-Fi: explicitly non-3GPP log-distance indoor model.

This is not a claim that the complete simulator is 3GPP compliant. Validity
domains and deviations are recorded in docs/standards_conformance_matrix.md.
"""
from __future__ import annotations

import numpy as np

from ..constants import C_LIGHT, fspl_db


def _array(value) -> np.ndarray:
    return np.asarray(value, dtype=float)


def fspl_path_loss_db(distance_km, freq_hz: float) -> np.ndarray:
    """Free-space path loss with a one-metre numerical floor."""
    return fspl_db(_array(distance_km) * 1000.0, freq_hz)


def uma_path_loss_db(
    distance_km,
    freq_hz: float,
    *,
    los: bool | np.ndarray = True,
    h_bs_m: float = 25.0,
    h_ut_m: float = 1.5,
) -> np.ndarray:
    """Selected 3GPP TR 38.901 UMa LOS/NLOS path-loss equations."""
    fc_ghz = float(freq_hz) / 1e9
    d_3d = np.maximum(_array(distance_km) * 1000.0, 10.0)
    height_delta = max(h_bs_m - h_ut_m, 0.0)
    d_2d = np.sqrt(np.maximum(d_3d**2 - height_delta**2, 10.0**2))
    h_bs_eff = max(h_bs_m - 1.0, 1e-6)
    h_ut_eff = max(h_ut_m - 1.0, 1e-6)
    d_bp = 4.0 * h_bs_eff * h_ut_eff * freq_hz / C_LIGHT
    pl_los_1 = 28.0 + 22.0 * np.log10(d_3d) + 20.0 * np.log10(fc_ghz)
    pl_los_2 = (
        28.0
        + 40.0 * np.log10(d_3d)
        + 20.0 * np.log10(fc_ghz)
        - 9.0 * np.log10(d_bp**2 + height_delta**2)
    )
    pl_los = np.where(d_2d <= d_bp, pl_los_1, pl_los_2)
    pl_nlos_prime = (
        13.54
        + 39.08 * np.log10(d_3d)
        + 20.0 * np.log10(fc_ghz)
        - 0.6 * (h_ut_m - 1.5)
    )
    pl_nlos = np.maximum(pl_los, pl_nlos_prime)
    return np.where(np.asarray(los, dtype=bool), pl_los, pl_nlos)


def uma_av_path_loss_db(
    distance_km,
    freq_hz: float,
    *,
    altitude_m,
    los: bool | np.ndarray = True,
    h_bs_m: float = 25.0,
) -> np.ndarray:
    """Selected UMa-AV terms from 3GPP TR 36.777 Table B-2."""
    fc_ghz = float(freq_hz) / 1e9
    h_ut = _array(altitude_m)
    d_3d = np.maximum(_array(distance_km) * 1000.0, 10.0)
    pl_los = 28.0 + 22.0 * np.log10(d_3d) + 20.0 * np.log10(fc_ghz)
    pl_nlos_high = (
        -17.5
        + (46.0 - 7.0 * np.log10(np.maximum(h_ut, 22.5))) * np.log10(d_3d)
        + 20.0 * np.log10(40.0 * np.pi * fc_ghz / 3.0)
    )
    pl_nlos_low = uma_path_loss_db(
        distance_km, freq_hz, los=False, h_bs_m=h_bs_m, h_ut_m=1.5
    )
    pl_nlos = np.where(h_ut > 22.5, np.maximum(pl_los, pl_nlos_high), pl_nlos_low)
    return np.where(np.asarray(los, dtype=bool), pl_los, pl_nlos)


def ntn_basic_path_loss_db(distance_km, freq_hz: float, *, clutter_loss_db=0.0) -> np.ndarray:
    """TR 38.811 basic NTN loss component: FSPL plus scenario clutter loss."""
    return fspl_path_loss_db(distance_km, freq_hz) + _array(clutter_loss_db)


def wifi_indoor_path_loss_db(
    distance_km,
    freq_hz: float,
    *,
    path_loss_exponent: float = 3.0,
    reference_distance_m: float = 1.0,
    wall_loss_db=0.0,
) -> np.ndarray:
    """Log-distance indoor model, explicitly outside the 3GPP claim boundary."""
    d_m = np.maximum(_array(distance_km) * 1000.0, reference_distance_m)
    pl_ref = fspl_db(reference_distance_m, freq_hz)
    return (
        pl_ref
        + 10.0 * path_loss_exponent * np.log10(d_m / reference_distance_m)
        + _array(wall_loss_db)
    )


def path_loss_db(
    distance_km,
    freq_hz: float,
    *,
    tech_name: str | None = None,
    los: bool | np.ndarray = True,
    altitude_m=None,
    clutter_loss_db=0.0,
    wall_loss_db=0.0,
) -> np.ndarray:
    """Technology-aware dispatcher with an FSPL-compatible legacy default."""
    if tech_name == "5G_NR":
        return uma_path_loss_db(distance_km, freq_hz, los=los)
    if tech_name == "UAV":
        if altitude_m is None:
            raise ValueError("UAV path loss requires altitude_m")
        return uma_av_path_loss_db(distance_km, freq_hz, altitude_m=altitude_m, los=los)
    if tech_name and (tech_name.startswith("LEO") or tech_name == "HAPS"):
        return ntn_basic_path_loss_db(distance_km, freq_hz, clutter_loss_db=clutter_loss_db)
    if tech_name and tech_name.startswith("WiFi"):
        return wifi_indoor_path_loss_db(distance_km, freq_hz, wall_loss_db=wall_loss_db)
    return fspl_path_loss_db(distance_km, freq_hz)


def shadowing_db_series(
    n_steps: int,
    step_s: float,
    sigma_db: float,
    velocity_ms: float,
    decorr_distance_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gudmundson-style, distance-correlated log-normal shadowing in dB."""
    if sigma_db <= 0:
        return np.zeros(n_steps)
    dstep = max(float(np.asarray(velocity_ms).mean()), 0.1) * step_s
    rho = float(np.clip(np.exp(-dstep / max(decorr_distance_m, 1.0)), 0.0, 0.999))
    w = rng.standard_normal(n_steps)
    x = np.empty(n_steps)
    x[0] = w[0]
    scale = np.sqrt(max(1 - rho**2, 1e-9))
    for i in range(1, n_steps):
        x[i] = rho * x[i - 1] + scale * w[i]
    return x * sigma_db


def elevation_deg(distance_km, platform_altitude_km: float, min_deg: float = 5.0):
    """Legacy elevation helper retained for old data; v2 persists spherical geometry."""
    distance = _array(distance_km)
    ratio = np.clip(platform_altitude_km / np.maximum(distance, 1e-6), 0.0, 1.0)
    result = np.maximum(np.degrees(np.arcsin(ratio)), min_deg)
    return float(result) if result.ndim == 0 else result
