"""Phase 3 Gym(nasium) environment: sequential, CONDITION-AWARE network selection with
handover cost and Area transitions.

Built on the shared-mobility model (drl/mobility.py). Unlike the earlier (Phase 2 v1)
design, the observation now includes live pre-decision measurables (RSSI/SINR) for
every candidate network at the current step, not just Area/availability context. This
is what makes the problem genuinely condition-reactive: outcomes (throughput, latency,
BER, packet loss) remain reward-only and never enter the observation - so this does
NOT leak the reward's own inputs, it exposes exactly what a real UE could measure
before deciding (current channel quality of each candidate), consistent with the
no-leakage discipline established in Phase 1 for network_type/KPI columns.

Episode = one UE trajectory across n_steps decision epochs; Area evolves via a Markov
chain (drl/mobility.py:AREA_TRANSITION) so Available_Networks changes over time. At
each step the agent picks a network_type; reward = Phase-1 composite QoE for that
instant's outcome KPIs, minus a flat handover penalty if it switched since the last
step. A network is choosable only if it's both in the current Area's Available_Networks
AND geometrically visible (NTN pass-in-view); picking anything else is invalid.
"""
import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .features import AREAS, NETWORK_TYPES
from .mobility import NETWORK_TO_TECH_CONFIG, fit_geometry_ranges, generate_episode_trajectories, sample_area_sequence
from .reward import RewardWeights, compute_reward

from noise_models.technologies import get_tech_config
from noise_models.link import simulate_trace

INVALID_ACTION_PENALTY = -1.0
HANDOVER_PENALTY = 0.15  # flat penalty in reward units (~half the typical best-vs-worst
                          # network reward gap per Area, see status.md) - discourages
                          # ping-pong without blocking necessary switches. Configurable.


def _load_area_networks(df: pd.DataFrame) -> dict:
    pairs = df[["Area", "Available_Networks"]].drop_duplicates()
    return {row["Area"]: set(row["Available_Networks"].split(",")) for _, row in pairs.iterrows()}


class NetworkSelectionEnv(gym.Env):
    def __init__(self, csv_path: str | Path, reward_params_path: str | Path,
                 episode_steps: int = 60, handover_penalty: float = HANDOVER_PENALTY,
                 reward_weights: RewardWeights = RewardWeights(), seed: int | None = None):
        super().__init__()
        df = pd.read_csv(csv_path)
        self._area_networks = _load_area_networks(df)
        self._geometry_ranges = fit_geometry_ranges(df)
        self._reward_params = json.loads(Path(reward_params_path).read_text())
        self._reward_weights = reward_weights
        self._episode_steps = episode_steps
        self._handover_penalty = handover_penalty

        self.action_space = spaces.Discrete(len(NETWORK_TYPES))
        # obs = Area one-hot + per-network [available, RSSI_norm, SINR_norm] + prev-action one-hot
        obs_dim = len(AREAS) + 3 * len(NETWORK_TYPES) + len(NETWORK_TYPES)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._trajectories = None
        self._area_seq = None
        self._t = 0
        self._prev_action = None
        self._current_available = None
        self._current_traces = None

    def _norm(self, value: float, network: str, field: str) -> float:
        r = self._geometry_ranges[network][field]
        span = max(r.high - r.low, 1e-9)
        return float(np.clip((value - r.low) / span, 0.0, 1.0))

    def _refresh_step_data(self):
        """Computes, for the CURRENT self._t, which networks are choosable and their
        pre-decision RSSI/SINR measurements (available candidates only)."""
        area = self._area_seq[self._t]
        area_ok = self._area_networks[area]
        available = set()
        traces = {}
        for network in NETWORK_TYPES:
            traj = self._trajectories[network]
            if network not in area_ok or not traj["visible"][self._t]:
                continue
            tech_name = NETWORK_TO_TECH_CONFIG[network]
            cfg = replace(get_tech_config(tech_name), n_steps=1)
            scenario = {
                "distance_km": float(traj["distance_km"][self._t]),
                "rain_rate_mmhr": float(traj["rain_rate_mmhr"][self._t]),
                "doppler_char_hz": float(traj["doppler_hz"]),
                "platform_altitude_km": float(traj["platform_altitude_km"]),
            }
            if traj["altitude_m"] is not None:
                scenario["altitude_m"] = float(traj["altitude_m"][self._t])
                scenario["speed_ms"] = float(traj["speed_ms"][self._t])
            trace = simulate_trace(scenario, cfg, self._rng)
            available.add(network)
            traces[network] = trace
        self._current_available = available
        self._current_traces = traces

    def _make_obs(self) -> np.ndarray:
        # on the terminal step self._t == episode_steps (one past the last valid
        # index) - clamp so the final returned obs still reflects the last real step.
        area = self._area_seq[min(self._t, self._episode_steps - 1)]
        area_onehot = np.array([1.0 if a == area else 0.0 for a in AREAS], dtype=np.float32)

        per_network = []
        for network in NETWORK_TYPES:
            if network in self._current_available:
                trace = self._current_traces[network]
                rssi_norm = self._norm(float(trace["RSSI_dBm"][0]), network, "RSSI_dBm")
                sinr_norm = self._norm(float(trace["SINR_dB"][0]), network, "SINR_dB")
                per_network.extend([1.0, rssi_norm, sinr_norm])
            else:
                per_network.extend([0.0, 0.0, 0.0])
        per_network = np.array(per_network, dtype=np.float32)

        prev_onehot = np.zeros(len(NETWORK_TYPES), dtype=np.float32)
        if self._prev_action is not None:
            prev_onehot[self._prev_action] = 1.0

        return np.concatenate([area_onehot, per_network, prev_onehot])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._trajectories = generate_episode_trajectories(self._episode_steps, self._geometry_ranges, self._rng)
        self._area_seq = sample_area_sequence(self._rng, self._episode_steps)
        self._t = 0
        self._prev_action = None
        self._refresh_step_data()
        return self._make_obs(), {}

    def step(self, action: int):
        network = NETWORK_TYPES[action]
        if network not in self._current_available:
            reward = INVALID_ACTION_PENALTY
            info = {"invalid_action": True, "area": self._area_seq[self._t]}
        else:
            trace = self._current_traces[network]
            row = pd.DataFrame([{
                "Log_Throughput_Mbps": np.log1p(trace["Throughput_Mbps"][0]),
                "Latency_ms": trace["Latency_ms"][0],
                "Log_Packet_Loss_pct": np.log1p(trace["Packet_Loss_pct"][0]),
                "Log_BER": -np.log10(trace["BER"][0] + 1e-12),
            }])
            reward = float(compute_reward(row, self._reward_params, self._reward_weights).iloc[0])
            if self._prev_action is not None and self._prev_action != action:
                reward -= self._handover_penalty
            info = {"invalid_action": False, "area": self._area_seq[self._t], "network_type": network}
            self._prev_action = action

        self._t += 1
        terminated = self._t >= self._episode_steps
        if not terminated:
            self._refresh_step_data()
        return self._make_obs(), reward, terminated, False, info
