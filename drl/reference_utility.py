"""Fixed, training-only utility and observation normalization for revision v2."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UtilityWeights:
    throughput: float = 1.0 / 3.0
    latency: float = 1.0 / 3.0
    reliability: float = 1.0 / 3.0

    def validate(self) -> None:
        values = (self.throughput, self.latency, self.reliability)
        if any(v < 0 for v in values) or not np.isclose(sum(values), 1.0):
            raise ValueError("utility weights must be non-negative and sum to one")


REFERENCE_HANDOVER_PENALTY = 0.15
PROFILES = {
    "balanced": UtilityWeights(),
    "throughput": UtilityWeights(0.6, 0.2, 0.2),
    "latency": UtilityWeights(0.2, 0.6, 0.2),
    "reliability": UtilityWeights(0.2, 0.2, 0.6),
}


def _transformed(frame: pd.DataFrame) -> dict[str, pd.Series]:
    throughput = pd.to_numeric(frame["throughput_mbps"], errors="raise")
    latency = pd.to_numeric(frame["latency_ms"], errors="raise")
    packet_loss = pd.to_numeric(frame["packet_loss_pct"], errors="raise")
    ber = pd.to_numeric(frame["ber"], errors="raise")
    return {
        "throughput": np.log1p(throughput),
        "latency": np.log1p(latency),
        "packet_loss": np.log1p(packet_loss),
        "ber": -np.log10(ber + 1e-12),
    }


def fit_reference_params(
    training_frame: pd.DataFrame, low_percentile: float = 1.0, high_percentile: float = 99.0
) -> dict:
    available = training_frame[training_frame["available"]].dropna(
        subset=["rssi_dbm", "sinr_db", "throughput_mbps", "latency_ms", "packet_loss_pct", "ber"]
    )
    transformed = _transformed(available)
    utility = {
        name: {
            "low": float(np.percentile(values, low_percentile)),
            "high": float(np.percentile(values, high_percentile)),
        }
        for name, values in transformed.items()
    }
    observation = {
        name: {
            "low": float(np.percentile(available[column], low_percentile)),
            "high": float(np.percentile(available[column], high_percentile)),
        }
        for name, column in (("rssi", "rssi_dbm"), ("sinr", "sinr_db"))
    }
    return {
        "fit_split": "train",
        "fit_bands": ["ku", "ka", "s"],
        "percentiles": [low_percentile, high_percentile],
        "utility": utility,
        "observation": observation,
        "reference_weights": asdict(PROFILES["balanced"]),
        "reference_handover_penalty": REFERENCE_HANDOVER_PENALTY,
    }


def _normalize(values, bounds: dict, lower_is_better: bool = False):
    span = max(bounds["high"] - bounds["low"], 1e-12)
    normalized = np.clip((values - bounds["low"]) / span, 0.0, 1.0)
    return 1.0 - normalized if lower_is_better else normalized


def compute_qoe(
    frame: pd.DataFrame, params: dict, weights: UtilityWeights = PROFILES["balanced"]
) -> pd.Series:
    weights.validate()
    values = _transformed(frame)
    bounds = params["utility"]
    throughput = _normalize(values["throughput"], bounds["throughput"])
    latency = _normalize(values["latency"], bounds["latency"], lower_is_better=True)
    packet_loss = _normalize(values["packet_loss"], bounds["packet_loss"], lower_is_better=True)
    ber = _normalize(values["ber"], bounds["ber"])
    reliability = 0.5 * packet_loss + 0.5 * ber
    return (
        weights.throughput * throughput
        + weights.latency * latency
        + weights.reliability * reliability
    )


def normalize_observation(value: float, name: str, params: dict) -> float:
    return float(_normalize(np.asarray(value), params["observation"][name]))


def save_reference_params(params: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(params, indent=2), encoding="utf-8")


def load_reference_params(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
