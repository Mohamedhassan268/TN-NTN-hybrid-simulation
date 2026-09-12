"""Federated classifiers for the separate Package C quantum extension.

This module deliberately has no dependency on the DRL agents or manuscript.
PennyLane is optional and imported only when a hybrid QNN is requested.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from drl.canonical_data import load_canonical
from drl.features import NETWORK_TYPES, build_canonical_context
from drl.reference_utility import PROFILES, compute_qoe, load_reference_params


@dataclass(frozen=True)
class FederatedConfig:
    topology: str = "fedavg"  # centralized, fedavg, fedprox, hierarchical
    model_kind: str = "classical"  # classical or hybrid_qnn
    latent_dim: int = 6
    qnn_condition: str = "ideal"  # ideal, finite_shot, noisy
    rounds: int = 30
    local_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 1e-3
    proximal_mu: float = 0.01
    finite_shots: int = 1024

    def validate(self) -> None:
        if self.topology not in {"centralized", "fedavg", "fedprox", "hierarchical"}:
            raise ValueError("unsupported topology")
        if self.model_kind not in {"classical", "hybrid_qnn"}:
            raise ValueError("unsupported model kind")
        if not 4 <= self.latent_dim <= 8:
            raise ValueError("hybrid latent dimension/qubit count must be 4 through 8")
        if self.qnn_condition not in {"ideal", "finite_shot", "noisy"}:
            raise ValueError("unsupported QNN condition")


class ClassicalSelector(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(21, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 5),
        )

    def forward(self, features):
        return self.net(features)


class HybridQNNSelector(nn.Module):
    def __init__(self, latent_dim: int, condition: str, shots: int):
        super().__init__()
        try:
            import pennylane as qml
        except ImportError as error:
            raise RuntimeError(
                "hybrid_qnn requires the optional PennyLane dependency; install it in the Package C environment"
            ) from error
        self.encoder = nn.Sequential(nn.Linear(21, latent_dim), nn.Tanh())
        device_name = "default.mixed" if condition == "noisy" else "default.qubit"
        device_shots = shots if condition == "finite_shot" else None
        device = qml.device(device_name, wires=latent_dim, shots=device_shots)

        @qml.qnode(device, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(latent_dim), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(latent_dim))
            if condition == "noisy":
                for wire in range(latent_dim):
                    qml.DepolarizingChannel(0.01, wires=wire)
            return [qml.expval(qml.PauliZ(wire)) for wire in range(latent_dim)]

        layer = qml.qnn.TorchLayer(circuit, {"weights": (2, latent_dim)})
        self.net = nn.Sequential(self.encoder, layer, nn.Linear(latent_dim, 5))

    def forward(self, features):
        return self.net(features)


def make_model(config: FederatedConfig) -> nn.Module:
    config.validate()
    if config.model_kind == "classical":
        return ClassicalSelector()
    return HybridQNNSelector(config.latent_dim, config.qnn_condition, config.finite_shots)


def load_samples(records_path, params_path, split: str, bands=("ku", "ka", "s")) -> pd.DataFrame:
    """Build 21-feature, current-context network-selection classification records."""
    frame = load_canonical(records_path, split=split)
    params = load_reference_params(params_path)
    rows = []
    for (trajectory_id, step, band), candidates in frame.groupby(
        ["trajectory_id", "step_index", "band"], sort=False
    ):
        if band not in bands:
            continue
        valid = candidates[candidates["available"]].copy()
        if valid.empty:
            continue
        valid["qoe"] = compute_qoe(valid, params, PROFILES["balanced"])
        winner = str(valid.loc[valid["qoe"].idxmax(), "network_type"])
        rows.append({
            "trajectory_id": trajectory_id,
            "band": band,
            "area": str(candidates["area"].iloc[0]),
            "features": build_canonical_context(candidates, params, include_previous=False),
            "label": NETWORK_TYPES.index(winner),
        })
    return pd.DataFrame(rows)


def make_clients(samples: pd.DataFrame, seed: int, trajectories_per_client: int = 40):
    """Assign whole trajectories to clients and area-derived hierarchy groups."""
    rng = np.random.default_rng(seed)
    trajectory_ids = np.asarray(sorted(samples["trajectory_id"].unique()), dtype=object)
    trajectory_ids = trajectory_ids[rng.permutation(len(trajectory_ids))]
    clients = []
    for index in range(0, len(trajectory_ids), trajectories_per_client):
        ids = trajectory_ids[index:index + trajectories_per_client]
        group = samples[samples["trajectory_id"].isin(ids)].copy()
        cluster = str(group["area"].mode().iloc[0])
        clients.append((cluster, group))
    return clients


def _loader(frame: pd.DataFrame, batch_size: int, shuffle: bool):
    features = np.stack(frame["features"].to_numpy()).astype(np.float32)
    labels = frame["label"].to_numpy(dtype=np.int64)
    return DataLoader(TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)), batch_size=batch_size, shuffle=shuffle)


def _state_bytes(model: nn.Module) -> int:
    return sum(value.numel() * value.element_size() for value in model.state_dict().values())


def _average(states: Iterable[tuple[dict, int]]) -> dict:
    states = list(states)
    total = sum(weight for _, weight in states)
    result = {}
    for key in states[0][0]:
        result[key] = sum(state[key] * (weight / total) for state, weight in states)
    return result


def _local_fit(model, data, config: FederatedConfig, anchor_state=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(config.local_epochs):
        for features, labels in _loader(data, config.batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels)
            if anchor_state is not None:
                proximal = sum(
                    ((parameter - anchor_state[name]) ** 2).sum()
                    for name, parameter in model.named_parameters()
                )
                loss = loss + 0.5 * config.proximal_mu * proximal
            loss.backward()
            optimizer.step()


def evaluate(model, samples: pd.DataFrame, tier_dropout_index: int | None = None) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    model.eval()
    features = np.stack(samples["features"].to_numpy()).astype(np.float32)
    if tier_dropout_index is not None:
        start = 6 + 3 * tier_dropout_index
        features[:, start:start + 3] = 0.0
    labels = samples["label"].to_numpy(dtype=np.int64)
    with torch.no_grad():
        pred = model(torch.from_numpy(features)).argmax(dim=1).numpy()
    return {
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
    }


def run_federated(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, config: FederatedConfig, seed: int):
    """Run a classical baseline or selected hybrid topology; return round history and final metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    config.validate()
    model = make_model(config)
    clients = make_clients(train, seed)
    started = time.perf_counter()
    history = []
    update_bytes = _state_bytes(model)
    for round_index in range(1, config.rounds + 1):
        if config.topology == "centralized":
            _local_fit(model, train, config)
            transmitted = 0
        else:
            global_state = deepcopy(model.state_dict())
            client_states = []
            for cluster, data in clients:
                local = make_model(config)
                local.load_state_dict(global_state)
                proximal_anchor = (
                    {name: parameter.detach().clone() for name, parameter in local.named_parameters()}
                    if config.topology == "fedprox" else None
                )
                _local_fit(
                    local, data, config, anchor_state=proximal_anchor,
                )
                client_states.append((cluster, deepcopy(local.state_dict()), len(data)))
            if config.topology == "hierarchical":
                cluster_states = []
                for cluster in sorted({cluster for cluster, _, _ in client_states}):
                    members = [(state, weight) for member, state, weight in client_states if member == cluster]
                    cluster_states.append((_average(members), sum(weight for _, weight in members)))
                model.load_state_dict(_average(cluster_states))
            else:
                model.load_state_dict(_average((state, weight) for _, state, weight in client_states))
            transmitted = update_bytes * len(client_states)
        valid = evaluate(model, validation)
        history.append({
            "round": round_index, "validation_macro_f1": valid["macro_f1"],
            "validation_balanced_accuracy": valid["balanced_accuracy"],
            "model_update_bytes": transmitted, "simulated_elapsed_s": time.perf_counter() - started,
        })
    final = evaluate(model, test)
    final.update({
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_update_bytes": update_bytes,
        "simulated_wall_clock_s": time.perf_counter() - started,
    })
    for index, tier in enumerate(NETWORK_TYPES):
        dropped = evaluate(model, test, tier_dropout_index=index)
        final[f"tier_dropout_{tier}_macro_f1"] = dropped["macro_f1"]
    return model, pd.DataFrame(history), final
