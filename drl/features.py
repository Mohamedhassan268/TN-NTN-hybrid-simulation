"""State/action encoding for the Phase-1 network-selection bandit.

State = context known BEFORE a network is picked: Area (one-hot) + Available_Networks
(multi-hot over the 5 network types). KPI columns (SNR, distance_km, latency, etc.) are
NOT part of the state: they are entirely determined by which network a row represents
(e.g. distance_km is ~800-1900km for every SAT (LEO) row, <2km for every NR_5G row), so
they are outcomes of the action, not pre-decision context. They feed the reward only
(see drl/reward.py). network_type is the action and is excluded from state to avoid
label leakage.
"""
import numpy as np
import pandas as pd

NETWORK_TYPES = ["HAPS", "NR_5G", "SAT (LEO)", "UAV", "WiFi"]
AREAS = ["Desert", "Highway", "Indoor", "Maritime", "Rural", "Urban"]

ACTION_TO_NETWORK = {i: n for i, n in enumerate(NETWORK_TYPES)}
NETWORK_TO_ACTION = {n: i for i, n in enumerate(NETWORK_TYPES)}


def build_state(df: pd.DataFrame) -> np.ndarray:
    area_onehot = pd.get_dummies(df["Area"]).reindex(columns=AREAS, fill_value=0)
    avail_multihot = pd.DataFrame(
        {n: df["Available_Networks"].str.contains(n, regex=False).astype(int) for n in NETWORK_TYPES}
    )
    state = pd.concat([area_onehot, avail_multihot], axis=1)
    return state.to_numpy(dtype=np.float32)


def build_actions(df: pd.DataFrame) -> np.ndarray:
    return df["network_type"].map(NETWORK_TO_ACTION).to_numpy(dtype=np.int64)


def parse_observation(obs: np.ndarray) -> dict:
    """Decodes the Phase-3 NetworkSelectionEnv observation layout: Area one-hot +
    per-network [available, RSSI_norm, SINR_norm] + prev-action one-hot (see drl/env.py)."""
    n_areas = len(AREAS)
    n_net = len(NETWORK_TYPES)
    area_onehot = obs[:n_areas]
    per_network = obs[n_areas:n_areas + 3 * n_net].reshape(n_net, 3)
    prev_onehot = obs[n_areas + 3 * n_net:]
    return {
        "area_onehot": area_onehot,
        "available": per_network[:, 0],
        "rssi_norm": per_network[:, 1],
        "sinr_norm": per_network[:, 2],
        "prev_onehot": prev_onehot,
    }


def build_canonical_context(frame: pd.DataFrame, params: dict, include_previous: bool = False) -> np.ndarray:
    """Build the documented v2 21- or 26-value observation from one scenario step."""
    from .reference_utility import normalize_observation

    indexed = frame.set_index("network_type")
    area = str(indexed["area"].iloc[0])
    values = [1.0 if candidate == area else 0.0 for candidate in AREAS]
    for network in NETWORK_TYPES:
        row = indexed.loc[network]
        if bool(row["available"]):
            values.extend([
                1.0,
                normalize_observation(float(row["rssi_dbm"]), "rssi", params),
                normalize_observation(float(row["sinr_db"]), "sinr", params),
            ])
        else:
            values.extend([0.0, 0.0, 0.0])
    if include_previous:
        values.extend([0.0] * len(NETWORK_TYPES))
    return np.asarray(values, dtype=np.float32)
