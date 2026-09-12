from __future__ import annotations

import json
from pathlib import Path

from scripts.run_physical_conformance import run


ROOT = Path(__file__).resolve().parent.parent


def test_every_selected_3gpp_component_has_vectors_and_passes(tmp_path):
    matrix = json.loads((ROOT / "standards" / "conformance_matrix.json").read_text(encoding="utf-8"))
    vector_ids = {
        item["id"]
        for item in json.loads((ROOT / "standards" / "reference_vectors.json").read_text(encoding="utf-8"))
    }
    assert {entry["standard"] for entry in matrix} == {
        "3GPP TR 38.901", "3GPP TR 36.777", "3GPP TR 38.811"
    }
    for entry in matrix:
        assert entry["version"]
        assert entry["clause_or_table"]
        assert entry["equation"]
        assert entry["parameter_domain"]
        assert entry["deviation"]
        assert set(entry["reference_vectors"]) <= vector_ids
    result = run(ROOT / "standards" / "reference_vectors.json", tmp_path / "results.csv")
    assert result["source_pass"].all()
