from __future__ import annotations

from pathlib import Path

from drl.experiment_manifest import checkpoint_metadata, load_manifest
from drl.device import meets_speedup_threshold, resolve_device


ROOT = Path(__file__).resolve().parent.parent


def test_checkpoint_manifest_binds_dataset_splits_dependencies_and_result_path():
    manifest = load_manifest(ROOT / "experiments" / "revision_v2.json")
    metadata = checkpoint_metadata(
        manifest, phase="core", algorithm="ppo", band="ku", train_seed=0,
        result_path="artifacts/revision_v2/core/ku/ppo/seed_0/model.zip",
        device={"requested": "cpu", "resolved": "cpu", "torch": "test"},
    )
    assert metadata["configuration_sha256"]
    assert metadata["runtime"]["d3rlpy"]
    assert metadata["runtime"]["torch"]
    assert metadata["device"]["resolved"] == "cpu"
    assert metadata["dataset"]["dataset_version"] == "2.0.0"
    assert metadata["dataset"]["splits_sha256"]
    assert metadata["result_path"].endswith("model.zip")


def test_cpu_device_is_explicit_and_4x_threshold_is_strict():
    assert resolve_device("cpu") == "cpu"
    assert meets_speedup_threshold(40.0, 10.0)
    assert not meets_speedup_threshold(39.99, 10.0)
