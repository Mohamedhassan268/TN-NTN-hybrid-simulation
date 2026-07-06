"""Physical constants and small shared helper functions used across channel models."""
import numpy as np

C_LIGHT = 299_792_458.0        # m/s
K_BOLTZMANN = 1.380649e-23     # J/K
T0_KELVIN = 290.0              # standard noise temperature reference


def thermal_noise_dbm(bandwidth_hz: float, noise_figure_db: float = 0.0) -> float:
    """Thermal noise floor N = kTB, in dBm, plus receiver noise figure."""
    n_watts = K_BOLTZMANN * T0_KELVIN * bandwidth_hz
    n_dbm = 10 * np.log10(n_watts / 1e-3)
    return n_dbm + noise_figure_db


def fspl_db(distance_m: np.ndarray, freq_hz: float) -> np.ndarray:
    """Free-space path loss in dB. distance_m must be > 0."""
    d = np.maximum(distance_m, 1.0)
    return 20 * np.log10(d) + 20 * np.log10(freq_hz) - 147.55


def shannon_capacity_bps_hz(sinr_db: np.ndarray) -> np.ndarray:
    """Shannon spectral efficiency bound (bps/Hz) from SINR in dB."""
    sinr_lin = db_to_lin(sinr_db)
    return np.log2(1.0 + np.maximum(sinr_lin, 0.0))


def db_to_lin(x_db):
    return 10.0 ** (np.asarray(x_db) / 10.0)


def lin_to_db(x_lin):
    return 10.0 * np.log10(np.maximum(np.asarray(x_lin), 1e-30))
