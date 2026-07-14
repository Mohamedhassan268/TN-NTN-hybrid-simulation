"""Classical telecom handover baselines (Phase 5), compared against the RL agents
via the same evaluation harness (drl/evaluation.py). These are what a networking-paper
reviewer expects instead of (or alongside) an offline-RL algorithm zoo:
- Random: zero-knowledge floor.
- Greedy: always switch to the instantaneously-best available network (the classic
  ping-pong strawman handover cost is meant to discourage).
- Hysteresis + time-to-trigger (3GPP A3-style): only switch if a candidate beats the
  current network by a margin, sustained for several consecutive steps.
- Oracle: upper bound. Plans a full-episode action sequence via dynamic programming
  over a reward matrix built from the SAME deterministic geometry trajectory
  (distance/Area/visibility, fixed once the episode seed is drawn) but an
  INDEPENDENT fading-noise draw used only for planning/estimation, not for scoring.
  The realized reward is then measured by actually stepping the env with that action
  plan - since NetworkSelectionEnv computes traces for every available network every
  step regardless of the chosen action, its fading-noise draw sequence is the same for
  every policy evaluated at a given seed, so oracle vs. RL vs. baselines are compared
  on identical realized conditions. The oracle is therefore "best plan given known
  geometry and typical fading," not a literal cheat using the exact future noise.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .features import NETWORK_TYPES, parse_observation
from .mobility import NETWORK_TO_TECH_CONFIG
from .reward import compute_reward

from noise_models.technologies import get_tech_config
from noise_models.link import simulate_trace


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.rng = np.random.default_rng(self._seed)

    def predict(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(len(NETWORK_TYPES)))


class GreedyPolicy:
    """Myopic: always pick the available network with the best current SINR."""

    def predict(self, obs: np.ndarray) -> int:
        parsed = parse_observation(obs)
        avail_idx = np.flatnonzero(parsed["available"] > 0.5)
        if len(avail_idx) == 0:
            return 0
        return int(avail_idx[np.argmax(parsed["sinr_norm"][avail_idx])])


class HysteresisPolicy:
    """3GPP A3-style: switch only if a candidate exceeds the current network's SINR
    by `margin` for `ttt_steps` consecutive steps. Forces an immediate switch if the
    current network drops out of availability."""

    def __init__(self, margin: float = 0.05, ttt_steps: int = 3):
        self.margin = margin
        self.ttt_steps = ttt_steps
        self.reset()

    def reset(self):
        self.current = None
        self.candidate = None
        self.candidate_count = 0

    def predict(self, obs: np.ndarray) -> int:
        parsed = parse_observation(obs)
        avail_idx = np.flatnonzero(parsed["available"] > 0.5)
        if len(avail_idx) == 0:
            return 0
        if self.current is None or self.current not in avail_idx:
            best = int(avail_idx[np.argmax(parsed["sinr_norm"][avail_idx])])
            self.current, self.candidate, self.candidate_count = best, None, 0
            return self.current

        current_sinr = parsed["sinr_norm"][self.current]
        others = avail_idx[avail_idx != self.current]
        if len(others) == 0:
            return self.current
        best_other = int(others[np.argmax(parsed["sinr_norm"][others])])
        best_other_sinr = parsed["sinr_norm"][best_other]

        if best_other_sinr > current_sinr + self.margin:
            if self.candidate == best_other:
                self.candidate_count += 1
            else:
                self.candidate, self.candidate_count = best_other, 1
            if self.candidate_count >= self.ttt_steps:
                self.current, self.candidate, self.candidate_count = self.candidate, None, 0
        else:
            self.candidate, self.candidate_count = None, 0
        return self.current


class PrecomputedPolicy:
    """Plays back a precomputed action sequence (used by the oracle)."""

    def __init__(self, actions: list[int]):
        self.actions = actions
        self.i = 0

    def reset(self):
        self.i = 0

    def predict(self, obs: np.ndarray) -> int:
        action = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return action


def _oracle_reward_matrix(env, reward_params, reward_weights, plan_rng: np.random.Generator) -> np.ndarray:
    n = env._episode_steps
    matrix = np.full((n, len(NETWORK_TYPES)), np.nan)
    for t in range(n):
        area = env._area_seq[t]
        area_ok = env._area_networks[area]
        for ai, network in enumerate(NETWORK_TYPES):
            traj = env._trajectories[network]
            if network not in area_ok or not traj["visible"][t]:
                continue
            tech_name = NETWORK_TO_TECH_CONFIG[network]
            cfg = replace(get_tech_config(tech_name), n_steps=1)
            scenario = {
                "distance_km": float(traj["distance_km"][t]),
                "rain_rate_mmhr": float(traj["rain_rate_mmhr"][t]),
                "doppler_char_hz": float(traj["doppler_hz"]),
                "platform_altitude_km": float(traj["platform_altitude_km"]),
            }
            if traj["altitude_m"] is not None:
                scenario["altitude_m"] = float(traj["altitude_m"][t])
                scenario["speed_ms"] = float(traj["speed_ms"][t])
            trace = simulate_trace(scenario, cfg, plan_rng)
            row = pd.DataFrame([{
                "Log_Throughput_Mbps": np.log1p(trace["Throughput_Mbps"][0]),
                "Latency_ms": trace["Latency_ms"][0],
                "Log_Packet_Loss_pct": np.log1p(trace["Packet_Loss_pct"][0]),
                "Log_BER": -np.log10(trace["BER"][0] + 1e-12),
            }])
            matrix[t, ai] = compute_reward(row, reward_params, reward_weights).iloc[0]
    return matrix


def _solve_oracle_actions(matrix: np.ndarray, handover_penalty: float) -> list[int]:
    n, k = matrix.shape
    neg = -1e9
    dp = np.where(np.isnan(matrix[0]), neg, matrix[0])
    back = np.full((n, k), -1, dtype=int)
    for t in range(1, n):
        prev_dp = dp
        dp = np.full(k, neg)
        for a in range(k):
            if np.isnan(matrix[t, a]):
                continue
            costs = prev_dp - np.where(np.arange(k) == a, 0.0, handover_penalty)
            best_prev = int(np.argmax(costs))
            val = costs[best_prev] + matrix[t, a]
            dp[a] = val
            back[t, a] = best_prev
    last = int(np.argmax(dp))
    if dp[last] <= neg / 2:
        return []
    actions = [last]
    for t in range(n - 1, 0, -1):
        last = int(back[t, last])
        actions.append(last)
    actions.reverse()
    return actions


def make_oracle_policy(env, seed: int, reward_params: dict, reward_weights, handover_penalty: float):
    """Call AFTER env.reset(seed=seed) so env._trajectories/_area_seq are populated
    for this episode. Returns a PrecomputedPolicy, or None if no valid plan exists
    (e.g. every network unavailable throughout - should not happen in practice)."""
    plan_rng = np.random.default_rng(seed * 7919 + 1)
    matrix = _oracle_reward_matrix(env, reward_params, reward_weights, plan_rng)
    actions = _solve_oracle_actions(matrix, handover_penalty)
    return PrecomputedPolicy(actions) if actions else None
