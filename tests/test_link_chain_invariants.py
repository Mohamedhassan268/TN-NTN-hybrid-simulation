from __future__ import annotations

import numpy as np

from noise_models.channels.impairments import apply_hw_ceiling
from noise_models.channels.interference import sinr_gap_db


def test_interference_degradation_is_never_negative():
    gap = sinr_gap_db(200, 0.0, np.random.default_rng(12))
    assert (gap >= 0.0).all()


def test_hardware_impairment_combines_noise_powers_in_linear_domain():
    channel_db, hardware_db = 10.0, 20.0
    expected_linear = 1.0 / (1.0 / (10 ** (channel_db / 10)) + 1.0 / (10 ** (hardware_db / 10)))
    actual = float(apply_hw_ceiling(channel_db, hardware_db))
    assert np.isclose(actual, 10 * np.log10(expected_linear))


def test_log_ber_definition_is_exact(canonical_fixture):
    frame, *_ = canonical_fixture
    available = frame[frame["available"]]
    expected = -np.log10(available["ber"].to_numpy() + 1e-12)
    assert np.allclose(available["log_ber"].to_numpy(), expected)
