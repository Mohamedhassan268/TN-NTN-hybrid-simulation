from __future__ import annotations

import pandas as pd

from drl.canonical_data import assert_no_split_overlap, validate_canonical
from drl.scenarios import ScenarioSpec, generate_scenario


def test_schema_counts_nulls_and_split_isolation(canonical_fixture):
    frame, *_ = canonical_fixture
    report = validate_canonical(frame)
    assert report.valid, report.errors
    assert_no_split_overlap(frame)
    counts = frame.groupby(["trajectory_id", "step_index", "band"]).size()
    assert counts.eq(5).all()
    unavailable = frame[~frame["available"]]
    assert unavailable[["rssi_dbm", "sinr_db", "throughput_mbps"]].isna().all().all()


def test_generation_is_reproducible_from_manifest_seed():
    spec = ScenarioSpec("repeat", "repeat_scenario", "repeat_ue", "train", 991)
    first = generate_scenario(spec)
    second = generate_scenario(spec)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_only_leo_outcomes_change_between_band_configurations(canonical_fixture):
    frame, *_ = canonical_fixture
    trajectory = frame[frame["trajectory_id"] == "train_0000"]
    tn = trajectory[trajectory["network_type"] != "SAT (LEO)"]
    comparison = tn.pivot_table(
        index=["step_index", "network_type"], columns="band", values="sinr_db", dropna=False
    )
    assert comparison["ku"].equals(comparison["ka"])
    assert comparison["ku"].equals(comparison["s"])


def test_validator_rejects_inconsistent_mask_and_time(canonical_fixture):
    frame, *_ = canonical_fixture
    broken_mask = frame.copy()
    broken_mask.loc[0, "available_mask"] = 0
    assert not validate_canonical(broken_mask).valid
    broken_time = frame.copy()
    broken_time.loc[0, "time_s"] = 999.0
    assert not validate_canonical(broken_time).valid
