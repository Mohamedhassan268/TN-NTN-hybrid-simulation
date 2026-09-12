"""Hash-guarded PPO/DQN training for the canonical revision-v2 campaign."""
from __future__ import annotations

import argparse
import csv
from functools import partial
import json
from pathlib import Path
import sys

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.canonical_env import CanonicalNetworkSelectionEnv, CanonicalScenarioBank
from drl.experiment_manifest import checkpoint_metadata, load_manifest, validate_or_write_metadata
from drl.reference_utility import PROFILES, load_reference_params

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "experiments" / "revision_v2.json"
ARTIFACT_ROOT = ROOT / "artifacts" / "revision_v2"


class MetricsCallback(BaseCallback):
    """Persist SB3 logger values needed for critic/training-stability analysis."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.rows = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        values = self.model.logger.name_to_value
        row = {"timesteps": self.num_timesteps}
        for key in (
            "rollout/ep_rew_mean", "rollout/ep_len_mean", "train/value_loss",
            "train/explained_variance", "train/entropy_loss", "train/approx_kl",
            "train/clip_fraction", "train/loss",
        ):
            if key in values:
                row[key] = float(values[key])
        self.rows.append(row)

    def _on_training_end(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in self.rows for key in row})
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)


def make_shared_factory(bank, params, handover_penalty=0.15, profile="balanced"):
    return partial(
        CanonicalNetworkSelectionEnv,
        None,
        None,
        split=bank.split,
        band=bank.band,
        bank=bank,
        reference_params=params,
        handover_penalty=handover_penalty,
        reward_weights=PROFILES[profile],
    )


def verify_frozen_critic(manifest, output_root: Path) -> None:
    """Prevent core PPO training before the validation-only critic decision is frozen."""
    selected_path = output_root / "development" / "ppo_critic" / "selected_configuration.json"
    if not selected_path.exists():
        raise RuntimeError(
            "core PPO is blocked until critic development produces selected_configuration.json"
        )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("status") != "frozen":
        raise RuntimeError("critic configuration is not frozen")
    expected = manifest["core"]["ppo"]
    selected_cfg = selected["configuration"]
    for key, expected_value in {
        "policy_network": expected["policy_network"],
        "value_network": expected["value_network"],
        "vf_coef": expected["vf_coef"],
    }.items():
        if selected_cfg.get(key) != expected_value:
            raise RuntimeError(
                f"frozen critic {key} differs from core manifest; update and re-hash the manifest before training"
            )


def train_one(manifest, algorithm: str, band: str, seed: int, output_root: Path):
    dataset = ROOT / manifest["dataset_path"]
    params_path = ROOT / manifest["reference_params_path"]
    bank = CanonicalScenarioBank(dataset, "train", band)
    params = load_reference_params(params_path)
    core = manifest["core"]
    if algorithm == "ppo":
        verify_frozen_critic(manifest, output_root)
    run_dir = output_root / "core" / band / algorithm / f"seed_{seed}"
    model_path = run_dir / "model.zip"
    metadata_path = run_dir / "metadata.json"
    metadata = checkpoint_metadata(
        manifest, phase="core", algorithm=algorithm, band=band, train_seed=seed,
        result_path=str(model_path.relative_to(ROOT)),
    )
    reusable = validate_or_write_metadata(metadata_path, metadata)
    if model_path.exists():
        if not reusable:
            raise RuntimeError(f"checkpoint exists without matching prior metadata: {model_path}")
        print(f"reuse verified checkpoint {model_path}")
        return
    factory = make_shared_factory(bank, params)
    vec_env = make_vec_env(factory, n_envs=core["n_envs"], seed=seed)
    callback = MetricsCallback(run_dir / "training_metrics.csv")
    if algorithm == "ppo":
        cfg = core["ppo"]
        model = PPO(
            "MlpPolicy", vec_env, seed=seed, verbose=1,
            learning_rate=cfg["learning_rate"], n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"], gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_range"],
            ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"],
            policy_kwargs={"net_arch": {"pi": cfg["policy_network"], "vf": cfg["value_network"]}},
        )
    elif algorithm == "dqn":
        cfg = core["dqn"]
        model = DQN(
            "MlpPolicy", vec_env, seed=seed, verbose=1,
            learning_rate=cfg["learning_rate"], buffer_size=cfg["buffer_size"],
            learning_starts=cfg["learning_starts"], batch_size=cfg["batch_size"],
            train_freq=cfg["train_freq"], target_update_interval=cfg["target_update_interval"],
            gamma=cfg["gamma"], policy_kwargs={"net_arch": cfg["network"]},
        )
    else:
        raise ValueError("algorithm must be ppo or dqn")
    model.learn(total_timesteps=core["total_timesteps"], callback=callback)
    model.save(model_path)
    print(f"saved {model_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--algorithm", choices=["ppo", "dqn", "all"], default="all")
    parser.add_argument("--band", choices=["ku", "ka", "s", "all"], default="all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path, default=ARTIFACT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest = load_manifest(args.manifest)
    algorithms = manifest["core"]["algorithms"] if args.algorithm == "all" else [args.algorithm]
    bands = manifest["bands"] if args.band == "all" else [args.band]
    seeds = manifest["core"]["training_seeds"] if args.seed is None else [args.seed]
    for band in bands:
        for algorithm in algorithms:
            for seed in seeds:
                train_one(manifest, algorithm, band, seed, args.output_root)
