"""Co-/adjacent-channel interference (SNR -> SINR gap) and WiFi contention/collision bursts."""
import numpy as np


def sinr_gap_db(n_steps: int, interference_margin_db: float, rng: np.random.Generator,
                smooth_steps: int = 15) -> np.ndarray:
    """Time-varying, non-negative dB gap subtracted from SNR to obtain SINR."""
    variation = rng.normal(0, max(interference_margin_db * 0.3, 0.3), n_steps)
    if n_steps > smooth_steps:
        kernel = np.ones(smooth_steps) / smooth_steps
        variation = np.convolve(variation, kernel, mode="same")
    return np.maximum(interference_margin_db + variation, 0.0)


def contention_burst_loss_db(n_steps: int, rng: np.random.Generator, prob: float = 0.04,
                              extra_loss_db: float = 10.0, burst_len: int = 4) -> np.ndarray:
    """CSMA collision/contention bursts (WiFi only): occasional short bursts of extra loss."""
    loss = np.zeros(n_steps)
    starts = np.where(rng.random(n_steps) < prob)[0]
    for s in starts:
        e = min(s + burst_len, n_steps)
        loss[s:e] = np.maximum(loss[s:e], extra_loss_db)
    return loss
