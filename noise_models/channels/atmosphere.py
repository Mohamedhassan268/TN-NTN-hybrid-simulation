"""Atmospheric impairments: ITU-R P.618-style rain attenuation, gaseous absorption (P.676),
and tropospheric/ionospheric scintillation."""
import numpy as np

# Simplified ITU-R P.618 specific-attenuation coefficients (k, alpha), horizontal
# polarization, indexed by frequency in GHz. Linearly interpolated between anchors.
_RAIN_COEFF = {
    2.0: (0.0000650, 1.121),
    2.1: (0.0000722, 1.129),
    3.5: (0.00088, 1.223),
    5.0: (0.00175, 1.310),
    6.4: (0.00301, 1.340),
    12.0: (0.0188, 1.217),
}


def _interp_coeff(freq_ghz: float):
    freqs = sorted(_RAIN_COEFF.keys())
    if freq_ghz <= freqs[0]:
        return _RAIN_COEFF[freqs[0]]
    if freq_ghz >= freqs[-1]:
        return _RAIN_COEFF[freqs[-1]]
    for f0, f1 in zip(freqs, freqs[1:]):
        if f0 <= freq_ghz <= f1:
            k0, a0 = _RAIN_COEFF[f0]
            k1, a1 = _RAIN_COEFF[f1]
            w = (freq_ghz - f0) / (f1 - f0)
            return k0 + (k1 - k0) * w, a0 + (a1 - a0) * w
    return _RAIN_COEFF[freqs[-1]]


def rain_attenuation_db_series(n_steps: int, step_s: float, rain_rate_mmhr: float,
                                freq_hz: float, elevation_deg: float,
                                rng: np.random.Generator) -> np.ndarray:
    if rain_rate_mmhr <= 0:
        return np.zeros(n_steps)
    freq_ghz = freq_hz / 1e9
    k, alpha = _interp_coeff(freq_ghz)
    gamma_r = k * rain_rate_mmhr ** alpha  # specific attenuation, dB/km
    elevation_rad = np.deg2rad(max(elevation_deg, 5.0))
    l_eff = min(35 * np.exp(-0.015 * rain_rate_mmhr) / np.sin(elevation_rad), 30.0)
    mean_atten = gamma_r * l_eff
    jitter = rng.normal(0, 0.05 * max(mean_atten, 0.1), n_steps)
    if n_steps > 20:
        kernel = np.ones(20) / 20
        jitter = np.convolve(jitter, kernel, mode="same")
    return np.maximum(mean_atten + jitter, 0.0)


def gaseous_attenuation_db(freq_hz: float, elevation_deg: float) -> float:
    freq_ghz = freq_hz / 1e9
    zenith_atten = 0.03 + 0.006 * max(freq_ghz - 2, 0)
    elevation_rad = np.deg2rad(max(elevation_deg, 5.0))
    return float(zenith_atten / np.sin(elevation_rad))


def scintillation_db_series(n_steps: int, sigma_db: float, rng: np.random.Generator) -> np.ndarray:
    if sigma_db <= 0:
        return np.zeros(n_steps)
    rho = 0.9
    w = rng.standard_normal(n_steps)
    x = np.empty(n_steps)
    x[0] = w[0]
    scale = np.sqrt(1 - rho ** 2)
    for i in range(1, n_steps):
        x[i] = rho * x[i - 1] + scale * w[i]
    return x * sigma_db
