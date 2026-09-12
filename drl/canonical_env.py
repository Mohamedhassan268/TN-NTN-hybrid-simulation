"""Gymnasium environment over immutable revision-v2 scenario banks."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from .canonical_data import load_canonical, validate_canonical
from .features import AREAS, NETWORK_TYPES
from .reference_utility import (
    PROFILES,
    REFERENCE_HANDOVER_PENALTY,
    UtilityWeights,
    compute_qoe,
    load_reference_params,
    normalize_observation,
)
from .scenarios import DATASET_VERSION, EpisodeScenario

INVALID_ACTION_PENALTY = -1.0


class CanonicalScenarioBank:
    def __init__(self, data_path: str | Path, split: str, band: str):
        frame = load_canonical(data_path, split=split, band=band)
        report = validate_canonical(frame)
        if not report.valid:
            raise ValueError("invalid canonical dataset: " + "; ".join(report.errors))
        self.split = split
        self.band = band
        self.frame = frame
        self.trajectory_ids = tuple(sorted(frame["trajectory_id"].unique()))
        self._episodes = {
            trajectory_id: EpisodeScenario(
                trajectory_id=trajectory_id,
                scenario_id=str(group["scenario_id"].iloc[0]),
                band=band,
                frame=group.sort_values(["step_index", "network_type"]).reset_index(drop=True),
            )
            for trajectory_id, group in frame.groupby("trajectory_id", sort=False)
        }

    def get(self, trajectory_id: str) -> EpisodeScenario:
        return self._episodes[trajectory_id]


class CanonicalNetworkSelectionEnv(gym.Env):
    """Five-action network selector with fixed, paired realized conditions.

    Invalid actions consume a decision epoch, receive -1, and leave the previous
    valid connection unchanged. The observation contains current measurements only;
    the full future frame is private to evaluation/oracle code.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path: str | Path | None,
        reference_params_path: str | Path | None,
        *,
        split: str = "train",
        band: str = "ku",
        handover_penalty: float = REFERENCE_HANDOVER_PENALTY,
        reward_weights: UtilityWeights = PROFILES["balanced"],
        seed: int | None = None,
        bank: CanonicalScenarioBank | None = None,
        reference_params: dict | None = None,
    ):
        super().__init__()
        if bank is None and data_path is None:
            raise ValueError("data_path is required when bank is not supplied")
        if reference_params is None and reference_params_path is None:
            raise ValueError("reference_params_path is required when params are not supplied")
        self.bank = bank or CanonicalScenarioBank(data_path, split, band)
        self.reference_params = reference_params or load_reference_params(reference_params_path)
        self.reward_weights = reward_weights
        self.reward_weights.validate()
        self.handover_penalty = float(handover_penalty)
        self._rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(len(NETWORK_TYPES))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(AREAS) + 3 * len(NETWORK_TYPES) + len(NETWORK_TYPES),),
            dtype=np.float32,
        )
        self._episode: EpisodeScenario | None = None
        self._steps: dict[int, pd.DataFrame] = {}
        self._t = 0
        self._previous_action: int | None = None

    @property
    def trajectory_id(self) -> str:
        if self._episode is None:
            raise RuntimeError("environment has not been reset")
        return self._episode.trajectory_id

    @property
    def episode_steps(self) -> int:
        if self._episode is None:
            return 60
        return self._episode.n_steps

    def _current_frame(self) -> pd.DataFrame:
        return self._steps[min(self._t, self.episode_steps - 1)]

    @property
    def current_candidates(self) -> pd.DataFrame:
        """Current realized candidates for privileged evaluation baselines only."""
        return self._current_frame().copy()

    def _observation(self) -> np.ndarray:
        frame = self._current_frame().set_index("network_type")
        area = str(frame["area"].iloc[0])
        values: list[float] = [1.0 if item == area else 0.0 for item in AREAS]
        for network in NETWORK_TYPES:
            row = frame.loc[network]
            if bool(row["available"]):
                values.extend([
                    1.0,
                    normalize_observation(float(row["rssi_dbm"]), "rssi", self.reference_params),
                    normalize_observation(float(row["sinr_db"]), "sinr", self.reference_params),
                ])
            else:
                values.extend([0.0, 0.0, 0.0])
        values.extend([
            1.0 if self._previous_action == index else 0.0
            for index in range(len(NETWORK_TYPES))
        ])
        return np.asarray(values, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        trajectory_id = options.get("trajectory_id")
        if trajectory_id is None:
            trajectory_id = str(self._rng.choice(self.bank.trajectory_ids))
        self._episode = self.bank.get(trajectory_id)
        self._steps = {
            int(step): group.copy()
            for step, group in self._episode.frame.groupby("step_index", sort=True)
        }
        self._t = 0
        self._previous_action = None
        return self._observation(), {
            "trajectory_id": self.trajectory_id,
            "scenario_id": self._episode.scenario_id,
            "band": self._episode.band,
            "dataset_version": DATASET_VERSION,
        }

    def step(self, action: int):
        action = int(action)
        network = NETWORK_TYPES[action]
        current = self._current_frame().set_index("network_type")
        row = current.loc[network]
        switched = False
        if not bool(row["available"]):
            reward = INVALID_ACTION_PENALTY
            info = {
                "invalid_action": True,
                "switched": False,
                "trajectory_id": self.trajectory_id,
                "step_index": self._t,
                "network_type": network,
                "reference_utility": INVALID_ACTION_PENALTY,
                "raw_qoe": np.nan,
            }
        else:
            selected = row.to_frame().T
            training_qoe = float(compute_qoe(selected, self.reference_params, self.reward_weights).iloc[0])
            reference_qoe = float(compute_qoe(selected, self.reference_params, PROFILES["balanced"]).iloc[0])
            switched = self._previous_action is not None and action != self._previous_action
            reward = training_qoe - (self.handover_penalty if switched else 0.0)
            reference_utility = reference_qoe - (
                REFERENCE_HANDOVER_PENALTY if switched else 0.0
            )
            info = {
                "invalid_action": False,
                "switched": switched,
                "trajectory_id": self.trajectory_id,
                "step_index": self._t,
                "network_type": network,
                "raw_qoe": reference_qoe,
                "reference_utility": reference_utility,
                "throughput_mbps": float(row["throughput_mbps"]),
                "latency_ms": float(row["latency_ms"]),
                "ber": float(row["ber"]),
                "bler": float(row["bler"]),
                "packet_loss_pct": float(row["packet_loss_pct"]),
            }
            self._previous_action = action
        self._t += 1
        terminated = self._t >= self.episode_steps
        return self._observation(), float(reward), terminated, False, info
