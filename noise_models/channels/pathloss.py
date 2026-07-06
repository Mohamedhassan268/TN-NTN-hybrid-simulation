"""Large-scale path loss and correlated log-normal shadowing (Gudmundson model)."""
import numpy as np
from ..constants import fspl_db


def path_loss_db(distance_km: float, freq_hz: float) -> float:
    return float(fspl_db(distance_km * 1000.0, freq_hz))


def shadowing_db_series(n_steps: int, step_s: float, sigma_db: float, velocity_ms: float,
                         decorr_distance_m: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean log-normal shadowing (dB), AR(1)-correlated via a Gudmundson decorrelation distance."""
    if sigma_db <= 0:
        return np.zeros(n_steps)
    dstep = max(velocity_ms, 0.1) * step_s
    rho = float(np.clip(np.exp(-dstep / max(decorr_distance_m, 1.0)), 0.0, 0.999))
    w = rng.standard_normal(n_steps)
    x = np.empty(n_steps)
    x[0] = w[0]
    scale = np.sqrt(max(1 - rho ** 2, 1e-9))
    for i in range(1, n_steps):
        x[i] = rho * x[i - 1] + scale * w[i]
    return x * sigma_db


def elevation_deg(distance_km: float, platform_altitude_km: float, min_deg: float = 5.0) -> float:
    """Crude flat-earth elevation angle estimate from ground distance and platform altitude."""
    ratio = np.clip(platform_altitude_km / max(distance_km, 1e-6), 0.0, 1.0)
    return float(max(np.degrees(np.arcsin(ratio)), min_deg))
