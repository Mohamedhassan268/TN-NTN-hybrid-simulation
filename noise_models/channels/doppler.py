"""Time-varying Doppler shift. Each clean row's Doppler_Hz value is treated as the
characteristic (t=0) Doppler for that UE's geometry; it evolves over the 6 s trace
with a smoothed jitter, plus a slow drift toward zero for LEO passes (Doppler rate)."""
import numpy as np


def doppler_series_hz(n_steps: int, step_s: float, fd_char_hz: float,
                       tech_name: str, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n_steps) * step_s
    jitter_sigma = max(abs(fd_char_hz) * 0.02, 0.5)
    jitter = rng.normal(0, jitter_sigma, n_steps)
    if n_steps > 5:
        kernel = np.ones(5) / 5
        jitter = np.convolve(jitter, kernel, mode="same")
    if tech_name == "LEO":
        rate_hz_per_s = -np.sign(fd_char_hz) * min(abs(fd_char_hz) / 1200.0, 5.0)
        drift = rate_hz_per_s * t
    else:
        drift = 0.0
    return fd_char_hz + drift + jitter
