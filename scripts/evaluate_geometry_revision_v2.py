"""Evaluate every core seed against nominal and frozen geometry shifts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.canonical_baselines import make_exact_oracle
from drl.canonical_env import CanonicalNetworkSelectionEnv
from drl.canonical_evaluation import SB3Policy, evaluate_policy, rollout_trajectory
from drl.experiment_manifest import configuration_hash, load_manifest
from drl.features import NETWORK_TYPES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "experiments" / "revision_v2.json"
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "revision_v2"


def _difficulty(frame: pd.DataFrame) -> dict:
    by_step = frame.groupby(["trajectory_id", "step_index", "band"])
    valid_count = by_step["available"].sum()
    result = {
        "mean_valid_network_count": float(valid_count.mean()),
        "no_valid_network_rate": float((valid_count == 0).mean()),
    }
    for network in NETWORK_TYPES:
        network_frame = frame[frame["network_type"] == network]
        key = network.lower().replace(" ", "_").replace("(", "").replace(")", "")
        result[f"availability_{key}"] = float(network_frame["available"].mean())
    leo = frame[frame["network_type"] == "SAT (LEO)"]
    result["leo_visibility_outage_rate"] = float((~leo["visible"]).mean())
    return result


def _oracle_results(env) -> pd.DataFrame:
    rows = []
    for trajectory_id in env.bank.trajectory_ids:
        env.reset(options={"trajectory_id": trajectory_id})
        rows.append(rollout_trajectory(env, make_exact_oracle(env), trajectory_id))
    return pd.DataFrame(rows)


def evaluate(manifest_path: Path, artifacts: Path, suite_root: Path) -> None:
    manifest = load_manifest(manifest_path)
    expected_hash = configuration_hash(manifest)
    params_path = ROOT / manifest["reference_params_path"]
    all_rows = []
    difficulty_rows = []
    for regime in manifest["geometry_regimes"]:
        data_path = (
            ROOT / manifest["dataset_path"]
            if regime == "nominal" else suite_root / regime / "records.parquet"
        )
        for band in manifest["bands"]:
            env = CanonicalNetworkSelectionEnv(data_path, params_path, split="test", band=band)
            difficulty = _difficulty(env.bank.frame)
            difficulty_rows.append({"geometry_regime": regime, "band": band, **difficulty})
            oracle = _oracle_results(env).set_index("trajectory_id")
            for algorithm in manifest["core"]["algorithms"]:
                for seed in manifest["core"]["training_seeds"]:
                    run_dir = artifacts / "core" / band / algorithm / f"seed_{seed}"
                    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
                    if metadata["configuration_sha256"] != expected_hash:
                        raise RuntimeError(f"mismatched checkpoint metadata: {run_dir}")
                    model_type = PPO if algorithm == "ppo" else DQN
                    model = model_type.load(run_dir / "model.zip")
                    result = evaluate_policy(env, lambda model=model: SB3Policy(model))
                    result["oracle_reward"] = result["trajectory_id"].map(oracle["reference_utility"])
                    result["oracle_normalized_regret"] = (
                        result["oracle_reward"] - result["reference_utility"]
                    ) / np.maximum(np.abs(result["oracle_reward"]), 1e-12)
                    result["geometry_regime"] = regime
                    result["policy"] = algorithm
                    result["train_seed"] = seed
                    all_rows.append(result)
    output = artifacts / "geometry"
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(output / "paired_geometry_results.csv", index=False)
    pd.DataFrame(difficulty_rows).to_csv(output / "geometry_difficulty.csv", index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--suite-root", type=Path, default=ROOT / "data" / "canonical_v2" / "geometry_suites")
    args = parser.parse_args()
    evaluate(args.manifest, args.artifacts, args.suite_root)
