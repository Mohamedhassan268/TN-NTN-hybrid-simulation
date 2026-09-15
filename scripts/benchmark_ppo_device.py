"""Benchmark matched PPO training on CPU and CUDA and enforce the 4x adoption rule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.canonical_env import CanonicalScenarioBank
from drl.device import device_provenance, meets_speedup_threshold, resolve_device
from drl.experiment_manifest import configuration_hash, load_manifest
from drl.reference_utility import load_reference_params
from scripts.train_revision_v2 import make_shared_factory


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "experiments" / "revision_v2.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "revision_v2" / "device_benchmark" / "benchmark.json"
THRESHOLD = 4.0


def run_once(manifest: dict, device: str, timesteps: int, seed: int) -> dict:
    resolved = resolve_device(device)
    bank = CanonicalScenarioBank(ROOT / manifest["dataset_path"], "train", "ku")
    params = load_reference_params(ROOT / manifest["reference_params_path"])
    vec_env = make_vec_env(make_shared_factory(bank, params), n_envs=manifest["core"]["n_envs"], seed=seed)
    cfg = manifest["core"]["ppo"]
    if resolved == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    model = PPO(
        "MlpPolicy", vec_env, seed=seed, verbose=0, device=resolved,
        learning_rate=cfg["learning_rate"], n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"], gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"],
        policy_kwargs={"net_arch": {"pi": cfg["policy_network"], "vf": cfg["value_network"]}},
    )
    model.learn(total_timesteps=timesteps)
    wall_seconds = perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated()) if resolved == "cuda" else 0
    vec_env.close()
    return {
        "device": device_provenance(device, resolved),
        "timesteps": timesteps,
        "seed": seed,
        "n_envs": manifest["core"]["n_envs"],
        "wall_seconds": wall_seconds,
        "fps": timesteps / wall_seconds,
        "peak_gpu_memory_bytes": peak_memory,
    }


def benchmark(manifest: dict, timesteps: int = 20_480, seed: int = 999) -> dict:
    cpu = run_once(manifest, "cpu", timesteps, seed)
    result = {
        "status": "cpu_retained",
        "required_speedup": THRESHOLD,
        "configuration_sha256": configuration_hash(manifest),
        "cpu": cpu,
    }
    try:
        cuda = run_once(manifest, "cuda", timesteps, seed)
    except Exception as exc:  # CUDA unavailability/compatibility is a valid retention result.
        result["reason"] = f"CUDA benchmark unavailable: {type(exc).__name__}: {exc}"
        return result
    speedup = cpu["wall_seconds"] / cuda["wall_seconds"]
    result["cuda"] = cuda
    result["speedup"] = speedup
    if meets_speedup_threshold(cpu["wall_seconds"], cuda["wall_seconds"], THRESHOLD):
        result["status"] = "cuda_adopted"
        result["reason"] = "CUDA met the required 4x end-to-end speed-up."
    else:
        result["reason"] = "CUDA did not meet the required 4x end-to-end speed-up."
    return result


def main(manifest_path: Path, output_path: Path, timesteps: int, seed: int) -> dict:
    result = benchmark(load_manifest(manifest_path), timesteps, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timesteps", type=int, default=20_480)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()
    main(args.manifest, args.output, args.timesteps, args.seed)
