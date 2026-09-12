from __future__ import annotations

import numpy as np
import pandas as pd

from drl.canonical_baselines import PrecomputedPolicy, exact_oracle_actions
from drl.canonical_env import CanonicalNetworkSelectionEnv
from drl.canonical_evaluation import rollout_trajectory
from drl.features import NETWORK_TYPES
from drl.reference_utility import PROFILES, compute_qoe


def _env(canonical_fixture):
    _, data_path, params_path, _ = canonical_fixture
    return CanonicalNetworkSelectionEnv(data_path, params_path, split="test", band="ku")


def test_observation_dimensions_and_deterministic_playback(canonical_fixture):
    env = _env(canonical_fixture)
    trajectory_id = env.bank.trajectory_ids[0]
    first, _ = env.reset(options={"trajectory_id": trajectory_id})
    second, _ = env.reset(options={"trajectory_id": trajectory_id})
    assert first.shape == (26,)
    assert np.array_equal(first, second)
    assert env.observation_space.contains(first)


def test_invalid_action_advances_time_and_preserves_previous_valid_connection(canonical_fixture):
    env = _env(canonical_fixture)
    obs, _ = env.reset(options={"trajectory_id": env.bank.trajectory_ids[0]})
    first_frame = env.current_candidates.set_index("network_type")
    valid_action = next(i for i, n in enumerate(NETWORK_TYPES) if first_frame.loc[n, "available"])
    obs, _, done, _, valid_info = env.step(valid_action)
    assert not done and not valid_info["invalid_action"]

    unavailable_action = None
    for _ in range(59):
        frame = env.current_candidates.set_index("network_type")
        unavailable_action = next(
            (i for i, n in enumerate(NETWORK_TYPES) if not frame.loc[n, "available"]), None
        )
        if unavailable_action is not None:
            break
        obs, _, done, _, _ = env.step(valid_action)
        if done:
            break
    assert unavailable_action is not None
    before = env._previous_action
    before_t = env._t
    obs, reward, _, _, info = env.step(unavailable_action)
    assert reward == -1.0
    assert info["invalid_action"]
    assert env._t == before_t + 1
    assert env._previous_action == before
    assert obs[-5 + before] == 1.0


def test_handover_is_charged_once(canonical_fixture):
    env = _env(canonical_fixture)
    env.reset(options={"trajectory_id": env.bank.trajectory_ids[0]})
    frame = env.current_candidates.set_index("network_type")
    first = next(i for i, n in enumerate(NETWORK_TYPES) if frame.loc[n, "available"])
    _, reward_1, _, _, info_1 = env.step(first)
    frame = env.current_candidates.set_index("network_type")
    alternatives = [i for i, n in enumerate(NETWORK_TYPES) if i != first and frame.loc[n, "available"]]
    if not alternatives:
        # Deterministically advance until two candidates are available.
        while not alternatives:
            env.step(first)
            frame = env.current_candidates.set_index("network_type")
            alternatives = [i for i, n in enumerate(NETWORK_TYPES) if i != first and frame.loc[n, "available"]]
    second = alternatives[0]
    _, reward_2, _, _, info_2 = env.step(second)
    assert not info_1["switched"]
    assert info_2["switched"]
    assert np.isclose(reward_2, info_2["raw_qoe"] - env.handover_penalty)


def test_exact_oracle_dominates_fixed_valid_action(canonical_fixture):
    env = _env(canonical_fixture)
    trajectory_id = env.bank.trajectory_ids[0]
    env.reset(options={"trajectory_id": trajectory_id})
    actions = exact_oracle_actions(env._episode.frame, env.reference_params)
    oracle = rollout_trajectory(env, PrecomputedPolicy(actions), trajectory_id)
    fixed = None
    for action in range(len(NETWORK_TYPES)):
        result = rollout_trajectory(env, PrecomputedPolicy([action] * 60), trajectory_id)
        fixed = result if fixed is None or result["reference_utility"] > fixed["reference_utility"] else fixed
    assert oracle["reference_utility"] + 1e-10 >= fixed["reference_utility"]


def test_reward_terms_are_monotone(canonical_fixture):
    _, _, _, params = canonical_fixture
    base = pd.DataFrame({
        "throughput_mbps": [10.0], "latency_ms": [100.0],
        "packet_loss_pct": [2.0], "ber": [1e-3],
    })
    base_qoe = float(compute_qoe(base, params, PROFILES["balanced"]).iloc[0])
    for column, value in [
        ("throughput_mbps", 20.0), ("latency_ms", 50.0),
        ("packet_loss_pct", 1.0), ("ber", 1e-4),
    ]:
        improved = base.copy()
        improved.loc[0, column] = value
        assert float(compute_qoe(improved, params, PROFILES["balanced"]).iloc[0]) >= base_qoe
