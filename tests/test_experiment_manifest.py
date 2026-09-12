from __future__ import annotations

from pathlib import Path

from drl.experiment_manifest import checkpoint_metadata, load_manifest


ROOT = Path(__file__).resolve().parent.parent


def test_checkpoint_manifest_binds_dataset_splits_dependencies_and_result_path():
    manifest = load_manifest(ROOT / "experiments" / "revision_v2.json")
    metadata = checkpoint_metadata(
        manifest, phase="core", algorithm="ppo", band="ku", train_seed=0,
        result_path="artifacts/revision_v2/core/ku/ppo/seed_0/model.zip",
    )
    assert metadata["configuration_sha256"]
    assert metadata["runtime"]["d3rlpy"]
    assert metadata["dataset"]["dataset_version"] == "2.0.0"
    assert metadata["dataset"]["splits_sha256"]
    assert metadata["result_path"].endswith("model.zip")
