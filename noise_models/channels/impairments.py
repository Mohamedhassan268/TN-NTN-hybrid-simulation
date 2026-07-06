"""Hardware impairments: EVM-limited SINR ceiling (phase noise, ADC quantization, PA nonlinearity
lumped into one effective ceiling, combined with the channel SINR as a parallel noise source)."""
import numpy as np


def apply_hw_ceiling(sinr_db, ceiling_db: float):
    sinr_lin = 10 ** (np.asarray(sinr_db) / 10)
    ceil_lin = 10 ** (ceiling_db / 10)
    combined_lin = 1.0 / (1.0 / np.maximum(sinr_lin, 1e-6) + 1.0 / ceil_lin)
    return 10 * np.log10(np.maximum(combined_lin, 1e-12))
