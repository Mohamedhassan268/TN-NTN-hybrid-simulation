"""Time-correlated small-scale multipath fading (Rayleigh / Rician), Clarke/Jakes-shaped."""
import numpy as np
from scipy.special import j0


def _ar1_gaussian(n_steps: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    rho = float(np.clip(rho, -0.999, 0.999))
    w = rng.standard_normal(n_steps)
    x = np.empty(n_steps)
    x[0] = w[0]
    scale = np.sqrt(max(1 - rho ** 2, 1e-9))
    for i in range(1, n_steps):
        x[i] = rho * x[i - 1] + scale * w[i]
    return x


def fading_gain_db(n_steps: int, step_s: float, fd_hz: float, fading_type: str,
                    k_db: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-mean-power fading gain in dB, time-correlated with the Clarke autocorrelation
    shape J0(2*pi*fd*dt) implied by Doppler spread fd."""
    fd = max(abs(fd_hz), 0.1)
    rho = float(np.clip(j0(2 * np.pi * fd * step_s), -0.99, 0.99))
    i = _ar1_gaussian(n_steps, rho, rng) / np.sqrt(2)
    q = _ar1_gaussian(n_steps, rho, rng) / np.sqrt(2)
    if fading_type == "rician":
        k_lin = 10 ** (k_db / 10)
        los = np.sqrt(k_lin / (k_lin + 1))
        scat_scale = np.sqrt(1 / (k_lin + 1))
        h = los + scat_scale * (i + 1j * q)
    else:
        h = i + 1j * q
    power = np.abs(h) ** 2
    return 10 * np.log10(np.maximum(power, 1e-6))
