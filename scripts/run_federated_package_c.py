"""Result-gated Package C runner; never contributes outputs to the DRL manuscript."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from federated.network_selection import FederatedConfig, load_samples, run_federated

ROOT = Path(__file__).resolve().parent.parent


def _samples():
    records = ROOT / "data" / "canonical_v2" / "records"
    params = ROOT / "data" / "canonical_v2" / "reference_params.json"
    return (
        load_samples(records, params, "train"),
        load_samples(records, params, "validation"),
        load_samples(records, params, "test"),
    )


def run_classical(config_path: Path, output: Path):
    study = json.loads(config_path.read_text(encoding="utf-8"))
    train, validation, test = _samples()
    rows = []
    for topology in study["topologies"]:
        for seed in study["classical_seeds"]:
            _, history, final = run_federated(
                train, validation, test, FederatedConfig(topology=topology), seed
            )
            history.assign(topology=topology, seed=seed).to_csv(
                output / f"history_{topology}_seed_{seed}.csv", index=False
            )
            rows.append({
                "topology": topology, "seed": seed,
                "validation_macro_f1": history.iloc[-1]["validation_macro_f1"],
                "validation_balanced_accuracy": history.iloc[-1]["validation_balanced_accuracy"],
                **final,
            })
    result = pd.DataFrame(rows)
    result.to_csv(output / "classical_results.csv", index=False)
    summary = result.groupby("topology", as_index=False).mean(numeric_only=True)
    winner = summary.sort_values(
        ["validation_macro_f1", "validation_balanced_accuracy"], ascending=False
    ).iloc[0]
    (output / "selected_topology.json").write_text(json.dumps({
        "status": "selected_from_validation_protocol", "topology": winner["topology"],
        "warning": "Do not run the hybrid replacement until this topology is independently reviewed.",
    }, indent=2), encoding="utf-8")


def run_hybrid(config_path: Path, output: Path, latent_dim: int, condition: str):
    """Run only after a classical topology was selected on validation data."""
    study = json.loads(config_path.read_text(encoding="utf-8"))
    selected_path = output / "selected_topology.json"
    if not selected_path.exists():
        raise SystemExit("run classical baselines and review selected_topology.json before hybrid replacement")
    topology = json.loads(selected_path.read_text(encoding="utf-8"))["topology"]
    train, validation, test = _samples()
    rows = []
    for seed in study["hybrid_seeds"]:
        _, history, final = run_federated(
            train, validation, test,
            FederatedConfig(
                topology=topology, model_kind="hybrid_qnn", latent_dim=latent_dim,
                qnn_condition=condition,
            ), seed,
        )
        history.assign(topology=topology, seed=seed, latent_dim=latent_dim, condition=condition).to_csv(
            output / f"hybrid_history_{condition}_{latent_dim}_seed_{seed}.csv", index=False
        )
        rows.append({"topology": topology, "seed": seed, "latent_dim": latent_dim, "condition": condition, **final})
    pd.DataFrame(rows).to_csv(output / f"hybrid_results_{condition}_{latent_dim}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "federated_package_c.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "package_c")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--condition", choices=["ideal", "finite_shot", "noisy"], default="ideal")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.hybrid:
        run_hybrid(args.config, args.output, args.latent_dim, args.condition)
    else:
        run_classical(args.config, args.output)
