"""Thermal noise floor (kTB)."""
from ..constants import thermal_noise_dbm


def noise_floor_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    return thermal_noise_dbm(bandwidth_hz, noise_figure_db)
