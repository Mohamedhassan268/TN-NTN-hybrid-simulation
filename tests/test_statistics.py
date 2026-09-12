from __future__ import annotations

import pandas as pd

from drl.canonical_evaluation import exact_sign_flip_p, hierarchical_paired_bootstrap


def test_exact_sign_flip_uses_seed_level_differences():
    assert exact_sign_flip_p([1.0] * 10) == 2.0 / 1024.0


def test_hierarchical_bootstrap_accepts_seeded_policy_and_shared_baseline():
    rows = []
    for seed in range(3):
        for trajectory in range(4):
            rows.append({"policy": "ppo", "train_seed": seed, "trajectory_id": trajectory, "score": 2 + seed})
    for trajectory in range(4):
        rows.append({"policy": "baseline", "train_seed": -1, "trajectory_id": trajectory, "score": 1.0})
    result = hierarchical_paired_bootstrap(
        pd.DataFrame(rows), "ppo", "baseline", "score", n_boot=100, seed=7
    )
    assert result["n_training_seeds"] == 3
    assert result["n_trajectories"] == 4
    assert result["mean_paired_difference"] == 2.0
