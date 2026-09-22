"""Paired evaluation and seed-aware inference for revision v2."""
from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np
import pandas as pd


class SB3Policy:
    def __init__(self, model):
        self.model = model

    def predict(self, obs, env=None):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


class D3rlpyPolicy:
    """Adapter that exposes only the first 21 documented CQL features."""

    def __init__(self, model):
        self.model = model

    def predict(self, obs, env=None):
        action = self.model.predict(np.asarray(obs[:21], dtype=np.float32)[None, :])
        return int(action[0])


def rollout_trajectory(env, policy, trajectory_id: str) -> dict:
    if hasattr(policy, "reset"):
        policy.reset()
    obs, reset_info = env.reset(options={"trajectory_id": trajectory_id})
    rows = []
    while True:
        action = policy.predict(obs, env=env)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append({"training_reward": reward, **info})
        if terminated or truncated:
            break
    steps = pd.DataFrame(rows)
    valid = steps[~steps["invalid_action"]]
    return {
        "trajectory_id": trajectory_id,
        "scenario_id": reset_info["scenario_id"],
        "band": reset_info["band"],
        "steps": len(steps),
        "training_reward": float(steps["training_reward"].sum()),
        "reference_utility": float(steps["reference_utility"].sum()),
        "raw_qoe": float(valid["raw_qoe"].sum()),
        "switches": int(steps["switched"].sum()),
        "switch_rate": float(steps["switched"].mean()),
        "invalid_count": int(steps["invalid_action"].sum()),
        "invalid_rate": float(steps["invalid_action"].mean()),
        "throughput_mbps": float(valid["throughput_mbps"].mean()) if len(valid) else np.nan,
        "latency_ms": float(valid["latency_ms"].mean()) if len(valid) else np.nan,
        "ber": float(valid["ber"].mean()) if len(valid) else np.nan,
        "bler": float(valid["bler"].mean()) if len(valid) else np.nan,
        "packet_loss_pct": float(valid["packet_loss_pct"].mean()) if len(valid) else np.nan,
    }


def evaluate_policy(env, policy_factory: Callable[[], object], trajectory_ids=None) -> pd.DataFrame:
    ids = tuple(trajectory_ids or env.bank.trajectory_ids)
    rows = []
    for trajectory_id in ids:
        rows.append(rollout_trajectory(env, policy_factory(), trajectory_id))
    return pd.DataFrame(rows)


def hierarchical_paired_bootstrap(
    frame: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    metric: str,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Resample training seeds first and shared trajectories second."""
    subset = frame[frame["policy"].isin([policy_a, policy_b])]
    learned = subset[subset["train_seed"] >= 0]
    seeds = np.sort(learned["train_seed"].unique())
    if len(seeds) < 2:
        raise ValueError("hierarchical inference requires at least two training seeds")
    trajectories = np.sort(subset["trajectory_id"].unique())
    lookup = subset.set_index(["policy", "train_seed", "trajectory_id"])[metric]

    def policy_matrix(policy: str) -> np.ndarray:
        """Return seed-by-trajectory values, broadcasting a shared baseline."""
        values = np.empty((len(seeds), len(trajectories)), dtype=float)
        for seed_index, train_seed in enumerate(seeds):
            for trajectory_index, trajectory_id in enumerate(trajectories):
                key = (policy, train_seed, trajectory_id)
                baseline_key = (policy, -1, trajectory_id)
                if key in lookup.index:
                    values[seed_index, trajectory_index] = float(lookup.loc[key])
                elif baseline_key in lookup.index:
                    values[seed_index, trajectory_index] = float(lookup.loc[baseline_key])
                else:
                    raise KeyError(f"missing paired value for {key}")
        return values

    # A bootstrap replicate is the mean of every selected seed/trajectory pair.
    # Keeping only draw counts yields the identical statistic while avoiding the
    # previous Python-level loop over n_boot * seeds * trajectories.
    differences = policy_matrix(policy_a) - policy_matrix(policy_b)
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(0, len(seeds), size=(n_boot, len(seeds)))
    trajectory_draws = rng.integers(0, len(trajectories), size=(n_boot, len(trajectories)))
    replicate_index = np.arange(n_boot)[:, None]
    seed_counts = np.zeros((n_boot, len(seeds)), dtype=float)
    trajectory_counts = np.zeros((n_boot, len(trajectories)), dtype=float)
    np.add.at(seed_counts, (replicate_index, seed_draws), 1.0)
    np.add.at(trajectory_counts, (replicate_index, trajectory_draws), 1.0)
    estimates = ((seed_counts @ differences) * trajectory_counts).sum(axis=1)
    estimates /= len(seeds) * len(trajectories)
    observed = differences.mean(axis=1)
    standardizer = observed.std(ddof=1)
    effect_size = float(observed.mean() / standardizer) if standardizer > 0 else np.nan
    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "metric": metric,
        "n_training_seeds": len(seeds),
        "n_trajectories": len(trajectories),
        "mean_paired_difference": float(observed.mean()),
        "ci_low": float(np.percentile(estimates, 2.5)),
        "ci_high": float(np.percentile(estimates, 97.5)),
        "seed_standardized_effect": effect_size,
        "exact_sign_flip_p": exact_sign_flip_p(observed),
    }


def exact_sign_flip_p(seed_differences: np.ndarray) -> float:
    differences = np.asarray(seed_differences, dtype=float)
    observed = abs(float(differences.mean()))
    signs = np.asarray(list(product([-1.0, 1.0], repeat=len(differences))))
    permuted = np.abs((signs * differences).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))
