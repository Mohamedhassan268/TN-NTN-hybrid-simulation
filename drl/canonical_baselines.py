"""Revision-v2 baselines, including an exact realized finite-horizon oracle."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import NETWORK_TYPES, parse_observation
from .reference_utility import PROFILES, REFERENCE_HANDOVER_PENALTY, compute_qoe


class RandomAllPolicy:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def predict(self, obs, env=None):
        return int(self.rng.integers(len(NETWORK_TYPES)))


class RandomValidPolicy(RandomAllPolicy):
    def predict(self, obs, env=None):
        available = np.flatnonzero(parse_observation(obs)["available"] > 0.5)
        return int(self.rng.choice(available)) if len(available) else 0


class SinrGreedyPolicy:
    def predict(self, obs, env=None):
        parsed = parse_observation(obs)
        available = np.flatnonzero(parsed["available"] > 0.5)
        if not len(available):
            return 0
        return int(available[np.argmax(parsed["sinr_norm"][available])])


class CurrentQoeGreedyPolicy:
    """Privileged current-step baseline; sees realized current reward, not the future."""

    def predict(self, obs, env=None):
        if env is None:
            raise ValueError("CurrentQoeGreedyPolicy requires the evaluation environment")
        candidates = env.current_candidates
        candidates = candidates[candidates["available"]].copy()
        if candidates.empty:
            return 0
        candidates["qoe"] = compute_qoe(candidates, env.reference_params, PROFILES["balanced"])
        network = str(candidates.loc[candidates["qoe"].idxmax(), "network_type"])
        return NETWORK_TYPES.index(network)


class HysteresisDbPolicy:
    """A3-inspired SINR margin and time-to-trigger rule in physical dB units."""

    def __init__(self, margin_db: float, ttt_steps: int):
        self.margin_db = float(margin_db)
        self.ttt_steps = int(ttt_steps)
        self.reset()

    def reset(self):
        self.current = None
        self.candidate = None
        self.candidate_count = 0

    def predict(self, obs, env=None):
        if env is None:
            raise ValueError("HysteresisDbPolicy requires the evaluation environment")
        frame = env.current_candidates.set_index("network_type")
        available = [i for i, n in enumerate(NETWORK_TYPES) if bool(frame.loc[n, "available"])]
        if not available:
            return 0
        if self.current not in available:
            self.current = max(available, key=lambda i: float(frame.loc[NETWORK_TYPES[i], "sinr_db"]))
            self.candidate = None
            self.candidate_count = 0
            return self.current
        alternatives = [i for i in available if i != self.current]
        if not alternatives:
            return self.current
        best = max(alternatives, key=lambda i: float(frame.loc[NETWORK_TYPES[i], "sinr_db"]))
        current_sinr = float(frame.loc[NETWORK_TYPES[self.current], "sinr_db"])
        candidate_sinr = float(frame.loc[NETWORK_TYPES[best], "sinr_db"])
        if candidate_sinr > current_sinr + self.margin_db:
            if self.candidate == best:
                self.candidate_count += 1
            else:
                self.candidate, self.candidate_count = best, 1
            if self.candidate_count >= self.ttt_steps:
                self.current = best
                self.candidate = None
                self.candidate_count = 0
        else:
            self.candidate = None
            self.candidate_count = 0
        return self.current


class PrecomputedPolicy:
    def __init__(self, actions):
        self.actions = list(map(int, actions))
        self.reset()

    def reset(self):
        self.index = 0

    def predict(self, obs, env=None):
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return action


def exact_oracle_actions(episode_frame: pd.DataFrame, params: dict, penalty=REFERENCE_HANDOVER_PENALTY):
    """Solve the realized finite-horizon network sequence exactly by dynamic programming."""
    n_steps = int(episode_frame["step_index"].max()) + 1
    n_actions = len(NETWORK_TYPES)
    matrix = np.full((n_steps, n_actions), -np.inf)
    for step, group in episode_frame.groupby("step_index"):
        valid = group[group["available"]].copy()
        if valid.empty:
            continue
        valid["qoe"] = compute_qoe(valid, params, PROFILES["balanced"])
        for row in valid.itertuples():
            matrix[int(step), NETWORK_TYPES.index(row.network_type)] = float(row.qoe)
    dp = np.full_like(matrix, -np.inf)
    back = np.full((n_steps, n_actions), -1, dtype=int)
    dp[0] = matrix[0]
    for step in range(1, n_steps):
        for action in range(n_actions):
            if not np.isfinite(matrix[step, action]):
                continue
            transitions = dp[step - 1] - penalty * (np.arange(n_actions) != action)
            previous = int(np.argmax(transitions))
            dp[step, action] = matrix[step, action] + transitions[previous]
            back[step, action] = previous
    final = int(np.argmax(dp[-1]))
    if not np.isfinite(dp[-1, final]):
        raise RuntimeError("scenario has no valid finite-horizon action sequence")
    actions = [final]
    for step in range(n_steps - 1, 0, -1):
        actions.append(int(back[step, actions[-1]]))
    return list(reversed(actions))


def make_exact_oracle(env):
    if env._episode is None:
        raise RuntimeError("reset the environment before constructing its oracle")
    return PrecomputedPolicy(exact_oracle_actions(env._episode.frame, env.reference_params))
