"""Three-seed PPO critic-capacity investigation on validation scenarios."""
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


def train_variant(manifest, artifacts, band, variant_name, seed):
    run_dir = artifacts / "development" / "ppo_critic" / band / variant_name / f"seed_{seed}"
    model_path = run_dir / "model.zip"
    metadata = checkpoint_metadata(
        manifest, phase="critic_development", algorithm="ppo", band=band,
        variant=variant_name, train_seed=seed, result_path=str(model_path.relative_to(ROOT)),
    )
    reusable = validate_or_write_metadata(run_dir / "metadata.json", metadata)
    if model_path.exists():
        if not reusable:
            raise RuntimeError(f"unverified development checkpoint: {model_path}")
        return PPO.load(model_path)
    params = load_reference_params(ROOT / manifest["reference_params_path"])
    bank = CanonicalScenarioBank(ROOT / manifest["dataset_path"], "train", band)
    factory = make_shared_factory(bank, params)
    vec_env = make_vec_env(factory, n_envs=manifest["core"]["n_envs"], seed=seed)
    base = manifest["core"]["ppo"]
    variant = manifest["critic_development"]["variants"][variant_name]
    model = PPO(
        "MlpPolicy", vec_env, seed=seed, verbose=1,
        learning_rate=base["learning_rate"], n_steps=base["n_steps"],
        batch_size=base["batch_size"], gamma=base["gamma"],
        gae_lambda=base["gae_lambda"], clip_range=base["clip_range"],
        ent_coef=base["ent_coef"], vf_coef=variant["vf_coef"],
        policy_kwargs={"net_arch": {
            "pi": variant["policy_network"], "vf": variant["value_network"]
        }},
    )
    model.learn(
        total_timesteps=manifest["core"]["total_timesteps"],
        callback=MetricsCallback(run_dir / "training_metrics.csv"),
    )
    model.save(model_path)
    return model


def run(manifest_path: Path, artifacts: Path):
    manifest = load_manifest(manifest_path)
    rows = []
    for band in manifest["bands"]:
        validation_env = CanonicalNetworkSelectionEnv(
            ROOT / manifest["dataset_path"], ROOT / manifest["reference_params_path"],
            split="validation", band=band,
        )
        for variant in manifest["critic_development"]["variants"]:
            for seed in manifest["critic_development"]["training_seeds"]:
                model = train_variant(manifest, artifacts, band, variant, seed)
                result = evaluate_policy(validation_env, lambda model=model: SB3Policy(model))
                metrics_path = artifacts / "development" / "ppo_critic" / band / variant / f"seed_{seed}" / "training_metrics.csv"
                metrics = pd.read_csv(metrics_path)
                explained = metrics.get("train/explained_variance", pd.Series(dtype=float)).dropna()
                rows.append({
                    "band": band, "variant": variant, "train_seed": seed,
                    "validation_reference_utility": result["reference_utility"].mean(),
                    "validation_raw_qoe": result["raw_qoe"].mean(),
                    "final_explained_variance": explained.iloc[-1] if len(explained) else float("nan"),
                })
    results = pd.DataFrame(rows)
    summary = results.groupby("variant", as_index=False).agg(
        validation_reference_utility=("validation_reference_utility", "mean"),
        explained_variance=("final_explained_variance", "mean"),
    )
    best_utility = summary["validation_reference_utility"].max()
    eligible = summary[summary["validation_reference_utility"] >= best_utility - 0.1]
    winner = eligible.sort_values(
        ["explained_variance", "validation_reference_utility"], ascending=False
    ).iloc[0]
    selection = {
        "status": "frozen", "selected_variant": winner["variant"],
        "selection_rule": manifest["critic_development"]["selection"],
        "validation_reference_utility": winner["validation_reference_utility"],
        "explained_variance": winner["explained_variance"],
        "configuration": manifest["critic_development"]["variants"][winner["variant"]],
    }
    output = artifacts / "development" / "ppo_critic"
    results.to_csv(output / "validation_results.csv", index=False)
    summary.to_csv(output / "variant_summary.csv", index=False)
    (output / "selected_configuration.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    run(args.manifest, args.artifacts)
