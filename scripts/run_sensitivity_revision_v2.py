"""Five-seed PPO handover/objective sensitivity campaign and raw-KPI export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.canonical_env import CanonicalNetworkSelectionEnv, CanonicalScenarioBank
from drl.canonical_evaluation import SB3Policy, evaluate_policy
from drl.experiment_manifest import checkpoint_metadata, load_manifest, validate_or_write_metadata
from drl.reference_utility import load_reference_params
from scripts.train_revision_v2 import MetricsCallback, make_shared_factory

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "experiments" / "revision_v2.json"
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "revision_v2"


def conditions(manifest):
    result = [
        (f"lambda_{penalty:g}", "balanced", float(penalty))
        for penalty in manifest["sweeps"]["handover_penalties"]
    ]
    result.extend(
        (f"objective_{profile}", profile, 0.15)
        for profile in manifest["sweeps"]["reward_profiles"]
        if profile != "balanced"
    )
    return result


def train_and_evaluate(manifest, artifacts, band, condition_id, profile, penalty, seed):
    run_dir = artifacts / "sensitivity" / band / condition_id / f"seed_{seed}"
    model_path = run_dir / "model.zip"
    metadata = checkpoint_metadata(
        manifest, phase="sensitivity", algorithm="ppo", band=band,
        condition=condition_id, reward_profile=profile,
        handover_penalty=penalty, train_seed=seed, result_path=str(model_path.relative_to(ROOT)),
    )
    reusable = validate_or_write_metadata(run_dir / "metadata.json", metadata)
    if model_path.exists():
        if not reusable:
            raise RuntimeError(f"unverified sweep checkpoint: {model_path}")
        model = PPO.load(model_path)
    else:
        params = load_reference_params(ROOT / manifest["reference_params_path"])
        bank = CanonicalScenarioBank(ROOT / manifest["dataset_path"], "train", band)
        vec = make_vec_env(
            make_shared_factory(bank, params, penalty, profile),
            n_envs=manifest["core"]["n_envs"], seed=seed,
        )
        cfg = manifest["core"]["ppo"]
        model = PPO(
            "MlpPolicy", vec, seed=seed, verbose=1,
            learning_rate=cfg["learning_rate"], n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"], gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_range"],
            ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"],
            policy_kwargs={"net_arch": {"pi": cfg["policy_network"], "vf": cfg["value_network"]}},
        )
        model.learn(
            total_timesteps=manifest["sweeps"]["total_timesteps"],
            callback=MetricsCallback(run_dir / "training_metrics.csv"),
        )
        model.save(model_path)
    env = CanonicalNetworkSelectionEnv(
        ROOT / manifest["dataset_path"], ROOT / manifest["reference_params_path"],
        split="test", band=band, handover_penalty=penalty,
    )
    result = evaluate_policy(env, lambda model=model: SB3Policy(model))
    result["condition"] = condition_id
    result["reward_profile"] = profile
    result["handover_penalty"] = penalty
    result["train_seed"] = seed
    result.to_csv(run_dir / "paired_test_results.csv", index=False)


def aggregate(artifacts: Path):
    paths = list((artifacts / "sensitivity").glob("*/*/seed_*/paired_test_results.csv"))
    if not paths:
        return
    result = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    output = artifacts / "sensitivity"
    result.to_csv(output / "paired_sensitivity_results.csv", index=False)
    result.groupby(
        ["band", "condition", "reward_profile", "handover_penalty", "train_seed"], as_index=False
    ).mean(numeric_only=True).to_csv(output / "seed_level_sensitivity.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--band", choices=["ku", "ka", "s", "all"], default="all")
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    selected_conditions = [c for c in conditions(manifest) if args.condition in (None, c[0])]
    if args.condition and not selected_conditions:
        raise SystemExit(f"unknown condition: {args.condition}")
    bands = manifest["bands"] if args.band == "all" else [args.band]
    seeds = manifest["sweeps"]["training_seeds"] if args.seed is None else [args.seed]
    for band in bands:
        for condition_id, profile, penalty in selected_conditions:
            for seed in seeds:
                train_and_evaluate(manifest, args.artifacts, band, condition_id, profile, penalty, seed)
    aggregate(args.artifacts)
