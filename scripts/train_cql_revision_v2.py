"""Train independent 21-feature Discrete CQL models on trajectory-group splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset

from drl.canonical_data import load_canonical
from drl.canonical_offline import build_logged_transitions, seeded_trajectory_partition
from drl.experiment_manifest import checkpoint_metadata, load_manifest, validate_or_write_metadata
from drl.reference_utility import load_reference_params

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "experiments" / "revision_v2.json"
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "revision_v2"


def train_one(manifest: dict, band: str, seed: int, artifacts: Path) -> None:
    run_dir = artifacts / "cql" / band / f"seed_{seed}"
    model_path = run_dir / "model.d3"
    metadata_path = run_dir / "metadata.json"
    metadata = checkpoint_metadata(
        manifest, phase="cql", algorithm="cql", band=band, train_seed=seed,
        result_path=str(model_path.relative_to(ROOT)),
    )
    reusable = validate_or_write_metadata(metadata_path, metadata)
    if model_path.exists():
        if not reusable:
            raise RuntimeError(f"checkpoint exists without matching metadata: {model_path}")
        print(f"reuse verified checkpoint {model_path}")
        return

    cfg = manifest["cql"]
    frame = load_canonical(ROOT / manifest["dataset_path"], split="train", band=band)
    params = load_reference_params(ROOT / manifest["reference_params_path"])
    train_ids, internal_validation_ids = seeded_trajectory_partition(
        frame["trajectory_id"].unique(), seed, cfg["trajectory_train_fraction"]
    )
    split_record = {
        "seed": seed,
        "partition_source": "canonical training split only",
        "train_trajectory_ids": train_ids,
        "internal_validation_trajectory_ids": internal_validation_ids,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectory_partition.json").write_text(
        json.dumps(split_record, indent=2), encoding="utf-8"
    )
    transitions = build_logged_transitions(
        frame[frame["trajectory_id"].isin(train_ids)], params, seed
    )
    dataset = MDPDataset(**transitions, action_size=5)
    algorithm = DiscreteCQLConfig(
        batch_size=cfg["batch_size"], learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"], target_update_interval=cfg["target_update_interval"],
        alpha=cfg["alpha"],
    ).create(device=False)
    history = algorithm.fit(
        dataset, n_steps=cfg["n_steps"], n_steps_per_epoch=cfg["n_steps_per_epoch"],
        experiment_name=f"cql_{band}_seed_{seed}", with_timestamp=False,
        show_progress=True, save_interval=1_000_000,
    )
    rows = [{"step": step, **metrics} for step, metrics in history]
    pd.DataFrame(rows).to_csv(run_dir / "training_metrics.csv", index=False)
    algorithm.save(str(model_path))
    print(f"saved {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--band", choices=["ku", "ka", "s", "all"], default="all")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    bands = manifest["bands"] if args.band == "all" else [args.band]
    seeds = manifest["cql"]["training_seeds"] if args.seed is None else [args.seed]
    for band in bands:
        for seed in seeds:
            train_one(manifest, band, seed, args.artifacts)
