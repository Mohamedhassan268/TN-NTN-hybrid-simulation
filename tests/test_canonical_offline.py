from __future__ import annotations

import numpy as np

from drl.canonical_offline import build_logged_transitions, seeded_trajectory_partition


def test_cql_partition_is_trajectory_grouped_and_seeded():
    ids = [f"t{i}" for i in range(10)]
    train, validation = seeded_trajectory_partition(ids, seed=4, train_fraction=0.8)
    assert len(train) == 8
    assert len(validation) == 2
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(ids)
    assert (train, validation) == seeded_trajectory_partition(ids, seed=4, train_fraction=0.8)


def test_cql_observations_are_21_values(canonical_fixture):
    frame, _, _, params = canonical_fixture
    train = frame[(frame["split"] == "train") & (frame["band"] == "ku")]
    logged = build_logged_transitions(train, params, seed=9)
    assert logged["observations"].shape == (60, 21)
    assert logged["actions"].shape == (60,)
    assert logged["terminals"].sum() == 1
    assert np.isfinite(logged["rewards"]).all()
