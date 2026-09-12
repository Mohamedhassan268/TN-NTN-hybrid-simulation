from __future__ import annotations

import numpy as np

from noise_models.channels.pathloss import (
    fspl_path_loss_db,
    ntn_basic_path_loss_db,
    uma_av_path_loss_db,
    uma_path_loss_db,
    wifi_indoor_path_loss_db,
)


def test_tr_38901_uma_los_reference_vector():
    # TR 38.901 Table 7.4.1-1, first-slope UMa LOS equation.
    expected_db = 28.0 + 22.0 * np.log10(100.0) + 20.0 * np.log10(3.5)
    actual_db = float(uma_path_loss_db(0.1, 3.5e9, los=True))
    assert abs(actual_db - expected_db) <= 0.1


def test_tr_38901_uma_nlos_is_maximum_of_los_and_nlos_prime():
    los_db = float(uma_path_loss_db(0.5, 3.5e9, los=True))
    nlos_prime_db = 13.54 + 39.08 * np.log10(500.0) + 20 * np.log10(3.5)
    actual_db = float(uma_path_loss_db(0.5, 3.5e9, los=False))
    assert abs(actual_db - max(los_db, nlos_prime_db)) <= 0.1


def test_tr_36777_uma_av_los_reference_vector():
    expected_db = 28.0 + 22.0 * np.log10(1000.0) + 20.0 * np.log10(2.0)
    actual_db = float(uma_av_path_loss_db(1.0, 2.0e9, altitude_m=100.0, los=True))
    assert abs(actual_db - expected_db) <= 0.1


def test_tr_38811_basic_ntn_component_is_fspl_plus_declared_clutter():
    expected_db = float(fspl_path_loss_db(600.0, 12.0e9)) + 2.5
    actual_db = float(ntn_basic_path_loss_db(600.0, 12.0e9, clutter_loss_db=2.5))
    assert abs(actual_db - expected_db) <= 1e-9


def test_wifi_model_is_explicitly_separate_log_distance_model():
    near = float(wifi_indoor_path_loss_db(0.001, 5.0e9))
    far = float(wifi_indoor_path_loss_db(0.01, 5.0e9))
    assert abs((far - near) - 30.0) <= 1e-9
