"""Phase-1 composite QoE reward for the network-selection bandit/offline-RL agent.

R = w_tp*throughput_term + w_lat*latency_term + w_rel*(0.5*packet_loss_term + 0.5*ber_term)

All sub-terms are min-max normalized to [0, 1] with higher = better, using
normalization bounds fit on the 1st-99th percentile of the training data (see
fit_normalization) so a handful of outlier rows can't blow out the scale.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

_SOURCE_COLUMNS = {
    "tp": "Log_Throughput_Mbps",
    "lat": "Latency_ms",
    "pl": "Log_Packet_Loss_pct",
    "ber": "Log_BER",
}
_LOWER_IS_BETTER = {"tp": False, "lat": True, "pl": True, "ber": False}


@dataclass
class RewardWeights:
    throughput: float = 1.0 / 3
    latency: float = 1.0 / 3
    reliability: float = 1.0 / 3


def _transform(df: pd.DataFrame, term: str) -> pd.Series:
    values = df[_SOURCE_COLUMNS[term]]
    return np.log1p(values) if term == "lat" else values


def fit_normalization(df: pd.DataFrame, low_pct: float = 1.0, high_pct: float = 99.0) -> dict:
    params = {}
    for term in _SOURCE_COLUMNS:
        values = _transform(df, term)
        low, high = np.percentile(values, [low_pct, high_pct])
        params[term] = {"low": float(low), "high": float(high)}
    return params


def _normalized_term(df: pd.DataFrame, term: str, params: dict) -> pd.Series:
    values = _transform(df, term)
    low, high = params[term]["low"], params[term]["high"]
    norm = (values - low) / (high - low)
    norm = norm.clip(0.0, 1.0)
    return 1.0 - norm if _LOWER_IS_BETTER[term] else norm


def compute_reward(df: pd.DataFrame, params: dict, weights: RewardWeights = RewardWeights()) -> pd.Series:
    tp = _normalized_term(df, "tp", params)
    lat = _normalized_term(df, "lat", params)
    pl = _normalized_term(df, "pl", params)
    ber = _normalized_term(df, "ber", params)
    reliability = 0.5 * pl + 0.5 * ber
    return weights.throughput * tp + weights.latency * lat + weights.reliability * reliability
