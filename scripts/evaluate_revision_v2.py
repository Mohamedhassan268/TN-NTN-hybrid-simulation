"""Paired baseline/model evaluation on canonical validation and test trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import d3rlpy
from stable_baselines3 import DQN, PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.canonical_baselines import (
    CurrentQoeGreedyPolicy, HysteresisDbPolicy, RandomAllPolicy, RandomValidPolicy,
    SinrGreedyPolicy, make_exact_oracle,
)
from drl.canonical_env import CanonicalNetworkSelectionEnv
from drl.canonical_evaluation import D3rlpyPolicy, SB3Policy, evaluate_policy, rollout_trajectory
from drl.experiment_manifest import configuration_hash, load_manifest

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "experiments" / "revision_v2.json"
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "revision_v2"


def env_for(manifest, split, band):
    return CanonicalNetworkSelectionEnv(
        ROOT / manifest["dataset_path"], ROOT / manifest["reference_params_path"],
        split=split, band=band,
    )


def tune_hysteresis(manifest, band):
    env = env_for(manifest, "validation", band)
    rows = []
    grid = manifest["hysteresis_validation_grid"]
    for margin in grid["margin_db"]:
        for ttt in grid["ttt_steps"]:
            result = evaluate_policy(env, lambda m=margin, t=ttt: HysteresisDbPolicy(m, t))
            rows.append({
                "band": band, "margin_db": margin, "ttt_steps": ttt,
                "reference_utility": result["reference_utility"].mean(),
                "switch_rate": result["switch_rate"].mean(),
            })
    frame = pd.DataFrame(rows).sort_values(
        ["reference_utility", "switch_rate"], ascending=[False, True]
    )
    return frame, frame.iloc[0].to_dict()


def evaluate_oracle(env):
    rows = []
    for trajectory_id in env.bank.trajectory_ids:
        env.reset(options={"trajectory_id": trajectory_id})
        rows.append(rollout_trajectory(env, make_exact_oracle(env), trajectory_id))
    return pd.DataFrame(rows)


def load_model(artifacts, algorithm, band, seed):
    path = artifacts / "core" / band / algorithm / f"seed_{seed}" / "model.zip"
    if not path.exists():
        raise FileNotFoundError(path)
    return (PPO if algorithm == "ppo" else DQN).load(path)


def main(manifest_path: Path, artifacts: Path):
    manifest = load_manifest(manifest_path)
    expected_hash = configuration_hash(manifest)
    all_results = []
    tuning_results = []
    selected = {}
    for band in manifest["bands"]:
        tuning, winner = tune_hysteresis(manifest, band)
        tuning_results.append(tuning)
        selected[band] = {"margin_db": winner["margin_db"], "ttt_steps": int(winner["ttt_steps"])}
        env = env_for(manifest, "test", band)
        baseline_factories = {
            "random_all": lambda: RandomAllPolicy(0),
            "random_valid": lambda: RandomValidPolicy(0),
            "sinr_greedy": SinrGreedyPolicy,
            "current_qoe_greedy": CurrentQoeGreedyPolicy,
            "hysteresis": lambda w=winner: HysteresisDbPolicy(w["margin_db"], int(w["ttt_steps"])),
        }
        for name, factory in baseline_factories.items():
            result = evaluate_policy(env, factory)
            result["policy"] = name
            result["train_seed"] = -1
            all_results.append(result)
        oracle = evaluate_oracle(env)
        oracle["policy"] = "oracle"
        oracle["train_seed"] = -1
        all_results.append(oracle)
        for algorithm in manifest["core"]["algorithms"]:
            for seed in manifest["core"]["training_seeds"]:
                metadata_path = artifacts / "core" / band / algorithm / f"seed_{seed}" / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata["configuration_sha256"] != expected_hash:
                    raise RuntimeError(f"mismatched checkpoint metadata: {metadata_path}")
                model = load_model(artifacts, algorithm, band, seed)
                result = evaluate_policy(env, lambda model=model: SB3Policy(model))
                result["policy"] = algorithm
                result["train_seed"] = seed
                all_results.append(result)
        for seed in manifest["cql"]["training_seeds"]:
            run_dir = artifacts / "cql" / band / f"seed_{seed}"
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            if metadata["configuration_sha256"] != expected_hash:
                raise RuntimeError(f"mismatched checkpoint metadata: {run_dir}")
            model = d3rlpy.load_learnable(str(run_dir / "model.d3"), device=False)
            result = evaluate_policy(env, lambda model=model: D3rlpyPolicy(model))
            result["policy"] = "cql"
            result["train_seed"] = seed
            all_results.append(result)

    output = artifacts / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(tuning_results, ignore_index=True).to_csv(output / "hysteresis_validation_grid.csv", index=False)
    (output / "hysteresis_selected.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    full = pd.concat(all_results, ignore_index=True)
    oracle = full[full["policy"] == "oracle"][["band", "trajectory_id", "reference_utility"]].rename(
        columns={"reference_utility": "oracle_reference_utility"}
    )
    compared = full[full["policy"] != "oracle"].merge(
        oracle, on=["band", "trajectory_id"], validate="many_to_one"
    )
    violations = compared[compared["reference_utility"] > compared["oracle_reference_utility"] + 1e-10]
    if not violations.empty:
        raise RuntimeError(
            f"oracle dominance failed for {len(violations)} paired policy trajectories"
        )
    full["oracle_reference_utility"] = full.set_index(["band", "trajectory_id"]).index.map(
        oracle.set_index(["band", "trajectory_id"])["oracle_reference_utility"]
    )
    full["oracle_normalized_regret"] = (
        full["oracle_reference_utility"] - full["reference_utility"]
    ) / full["oracle_reference_utility"].abs().clip(lower=1e-12)
    full.to_csv(output / "paired_test_results.csv", index=False)
    summary = full.groupby(["band", "policy", "train_seed"], as_index=False).mean(numeric_only=True)
    summary.to_csv(output / "seed_level_summary.csv", index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    main(args.manifest, args.artifacts)
