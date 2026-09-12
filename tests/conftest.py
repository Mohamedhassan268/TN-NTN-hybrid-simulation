from __future__ import annotations

import json

import pandas as pd
import pytest

from drl.reference_utility import fit_reference_params
from drl.scenarios import ScenarioSpec, generate_scenario


@pytest.fixture(scope="session")
def canonical_fixture(tmp_path_factory):
    """Small immutable bank with one trajectory in each logical split."""
    root = tmp_path_factory.mktemp("canonical-v2")
    specs = [
        ScenarioSpec("train_0000", "scenario_train", "ue_train", "train", 101),
        ScenarioSpec("validation_0000", "scenario_validation", "ue_validation", "validation", 202),
        ScenarioSpec("test_0000", "scenario_test", "ue_test", "test", 303),
    ]
    frame = pd.concat([generate_scenario(spec) for spec in specs], ignore_index=True)
    data_path = root / "records.parquet"
    frame.to_parquet(data_path, index=False)
    params = fit_reference_params(frame[frame["split"] == "train"])
    params_path = root / "reference_params.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return frame, data_path, params_path, params
