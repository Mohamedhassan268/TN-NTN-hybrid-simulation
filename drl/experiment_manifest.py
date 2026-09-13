"""Configuration hashing and checkpoint metadata for revision-v2 experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import d3rlpy
import gymnasium
import numpy
import pandas
import scipy
import stable_baselines3
import torch

ROOT = Path(__file__).resolve().parent.parent


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def configuration_hash(manifest: dict) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "d3rlpy": d3rlpy.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_provenance(manifest: dict) -> dict:
    """Bind an experiment to immutable dataset/split/normalization artifacts."""
    records = ROOT / manifest["dataset_path"]
    dataset_root = records.parent
    manifest_path = dataset_root / "manifest.json"
    splits_path = dataset_root / "splits.json"
    params_path = ROOT / manifest["reference_params_path"]
    return {
        "dataset_version": manifest["dataset_version"],
        "records_path": str(records.relative_to(ROOT)),
        "dataset_manifest_sha256": _sha256_file(manifest_path),
        "splits_sha256": _sha256_file(splits_path),
        "reference_params_sha256": _sha256_file(params_path),
    }


def checkpoint_metadata(manifest: dict, **run) -> dict:
    return {
        "configuration_sha256": configuration_hash(manifest),
        "dataset": dataset_provenance(manifest),
        "runtime": runtime_versions(),
        **run,
    }


def validate_or_write_metadata(path: str | Path, metadata: dict) -> bool:
    """Return True for a reusable exact match; write metadata for a new run."""
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError(f"stale or mismatched checkpoint metadata: {path}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return False
