"""Trajectory-grouped offline data construction for revision-v2 CQL."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import NETWORK_TYPES, build_canonical_context
from .reference_utility import PROFILES, compute_qoe


def seeded_trajectory_partition(
    trajectory_ids, seed: int, train_fraction: float = 0.9
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition trajectory identities only; rows from one trajectory never cross."""
    ids = np.asarray(sorted(set(trajectory_ids)), dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = ids[rng.permutation(len(ids))]
    cut = int(np.floor(len(ids) * train_fraction))
    if not 1 <= cut < len(ids):
        raise ValueError("trajectory partition requires at least two trajectories")
    return tuple(shuffled[:cut]), tuple(shuffled[cut:])


def build_logged_transitions(frame: pd.DataFrame, params: dict, seed: int) -> dict:
    """Create one logged random-valid action per decision epoch.

    CQL deliberately receives the documented 21-value current context and does
    not receive a previous-network indicator. Rewards are current-step QoE, while
    the online evaluation environment remains responsible for handover accounting.
    """
    rng = np.random.default_rng(seed)
    observations, actions, rewards, terminals = [], [], [], []
    ordered = frame.sort_values(["trajectory_id", "step_index", "network_type"])
    for _, trajectory in ordered.groupby("trajectory_id", sort=True):
        step_groups = list(trajectory.groupby("step_index", sort=True))
        for step_position, (_, candidates) in enumerate(step_groups):
            observations.append(build_canonical_context(candidates, params, include_previous=False))
            valid = candidates[candidates["available"]]
            if valid.empty:
                # Canonical generation is designed to prevent this; fail loudly if it occurs.
                raise ValueError("offline trajectory contains a step with no valid action")
            chosen = valid.iloc[int(rng.integers(len(valid)))]
            actions.append(NETWORK_TYPES.index(str(chosen["network_type"])))
            rewards.append(float(compute_qoe(chosen.to_frame().T, params, PROFILES["balanced"]).iloc[0]))
            terminals.append(float(step_position == len(step_groups) - 1))
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.float32),
    }
