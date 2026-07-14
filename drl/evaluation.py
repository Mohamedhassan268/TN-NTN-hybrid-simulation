"""Reusable evaluation harness (Phase 4): multi-seed training + many-episode rollout
evaluation with bootstrap confidence intervals, shared by the baseline comparison
(Phase 5) and the reward/handover-cost sensitivity sweep (Phase 6). Replaces the
single-run, single-seed, 5-episode evaluation baked directly into scripts/train_ppo.py.

Any policy object with a `.predict(obs) -> int action` method can be evaluated here:
SB3Policy / D3RLPolicy adapt stable_baselines3 and d3rlpy models to that interface;
rule-based baselines (drl/baselines.py) implement it directly.
"""
from typing import Callable

import numpy as np
import pandas as pd


class SB3Policy:
    def __init__(self, model):
        self.model = model

    def predict(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


class D3RLPolicy:
    def __init__(self, algo):
        self.algo = algo

    def predict(self, obs: np.ndarray) -> int:
        return int(self.algo.predict(obs[None, :])[0])


def rollout_episode(env, policy, seed: int) -> dict:
    if hasattr(policy, "reset"):
        policy.reset()
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    switches = 0
    invalid_count = 0
    prev_network = None
    steps = 0
    for _ in range(env._episode_steps):
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if info.get("invalid_action"):
            invalid_count += 1
        else:
            network = info.get("network_type")
            if prev_network is not None and network != prev_network:
                switches += 1
            prev_network = network
        if terminated or truncated:
            break
    return {
        "total_reward": total_reward,
        "switches": switches,
        "switch_rate": switches / steps if steps else 0.0,
        "invalid_count": invalid_count,
        "outage_rate": invalid_count / steps if steps else 0.0,
        "steps": steps,
    }


def evaluate_policy(env_factory: Callable[[], object], policy, n_episodes: int = 200, base_seed: int = 10_000,
                     progress_label: str | None = None, progress_every: int = 20) -> pd.DataFrame:
    rows = []
    for ep in range(n_episodes):
        rows.append(rollout_episode(env_factory(), policy, seed=base_seed + ep))
        if progress_label and (ep + 1) % progress_every == 0:
            print(f"    {progress_label}: {ep + 1}/{n_episodes} episodes", flush=True)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boot_means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(values.mean()), float(lo), float(hi)


def summarize_eval(eval_df: pd.DataFrame, metrics=("total_reward", "switch_rate", "outage_rate")) -> dict:
    return {m: dict(zip(("mean", "ci_low", "ci_high"), bootstrap_ci(eval_df[m].values))) for m in metrics}


def multi_seed_evaluate(env_factory: Callable[[], object], policy_factory: Callable[[int], object],
                         seeds, n_episodes_per_seed: int = 200) -> pd.DataFrame:
    """policy_factory(seed) -> trained policy for that seed. Returns one row per
    (seed, episode) so downstream code can compute either per-seed or pooled CIs."""
    frames = []
    for seed in seeds:
        policy = policy_factory(seed)
        eval_df = evaluate_policy(env_factory, policy, n_episodes=n_episodes_per_seed, base_seed=20_000 + seed * 1000)
        eval_df["train_seed"] = seed
        frames.append(eval_df)
    return pd.concat(frames, ignore_index=True)
