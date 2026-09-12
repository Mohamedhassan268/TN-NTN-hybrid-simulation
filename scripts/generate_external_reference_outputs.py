"""Generate frozen cross-check values by executing pinned upstream simulators.

Run this in an environment where OpenNTN is integrated as
``sionna.phy.channel.tr38811``. LLSim5G is imported from its pinned checkout.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLS = ROOT / "standards" / "external_tools.json"
DEFAULT_OUTPUT = ROOT / "standards" / "external_reference_outputs.csv"


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def require_pinned_checkout(name: str, path: Path, expected: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {name} checkout: {path}")
    actual = git_head(path)
    if actual != expected:
        raise SystemExit(f"{name} checkout is {actual}; expected pinned commit {expected}")


def llsim5g_values(root: Path) -> dict[str, float]:
    sys.path.insert(0, str(root))
    from channel_models.path_loss_models_tr_38_901 import uma_path_loss

    height_delta_m = 25.0 - 1.5
    return {
        "38901_uma_los_100m_3p5ghz": float(
            uma_path_loss(math.sqrt(100.0**2 - height_delta_m**2), 100.0, 1.5, 25.0, 3.5, True)
        ),
        "38901_uma_nlos_500m_3p5ghz": float(
            uma_path_loss(math.sqrt(500.0**2 - height_delta_m**2), 500.0, 1.5, 25.0, 3.5, False)
        ),
    }


def openntn_value() -> float:
    try:
        import tensorflow as tf
        from types import SimpleNamespace
        from sionna.phy.channel.tr38811 import utils
    except ImportError as exc:
        raise SystemExit(
            "OpenNTN is not integrated into Sionna; use its documented installer "
            "inside the isolated validation environment"
        ) from exc

    # OpenNTN uses metres and GHz in equation 6.6-1. Setting sigmaSF=0
    # isolates the deterministic basic-loss component. NLOS clutter is the
    # explicit 2.5 dB used by the study's reference vector.
    scenario = SimpleNamespace(
        _distance_3d=tf.constant([600_000.0], tf.float32),
        _carrier_frequency=tf.constant(12e9, tf.float32),
        _elevation_angle=10.0,
        los=tf.constant([False]),
        _params_nlos={
            "CL_10": tf.constant(2.5, tf.float32),
            "sigmaSF_10": tf.constant(0.0, tf.float32),
        },
        _params_los={"sigmaSF_10": tf.constant(0.0, tf.float32)},
    )
    utils.compute_pathloss_basic(scenario)
    return float(scenario._pl_b.numpy()[0])


def generate(tools_path: Path, llsim_root: Path, openntn_root: Path, output_path: Path) -> list[dict[str, object]]:
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    require_pinned_checkout("LLSim5G", llsim_root, tools["LLSim5G"]["commit"])
    require_pinned_checkout("OpenNTN", openntn_root, tools["OpenNTN"]["commit"])

    command = (
        "python scripts/generate_external_reference_outputs.py "
        "--llsim-root tmp/vendor/LLSim5G --openntn-root tmp/vendor/OpenNTN"
    )
    values = llsim5g_values(llsim_root)
    values["38811_basic_600km_12ghz_plus_clutter"] = openntn_value()
    rows = [
        {
            "id": vector_id,
            "external_db": value,
            "source": "LLSim5G",
            "source_version": tools["LLSim5G"]["version"],
            "source_commit": tools["LLSim5G"]["commit"],
            "status": "validated",
            "command_or_notebook": command,
            "notes": "Direct call to upstream TR 38.901 UMa path-loss implementation.",
        }
        for vector_id, value in values.items()
        if vector_id.startswith("38901_")
    ]
    rows.append({
        "id": "36777_uma_av_los_1km_2ghz",
        "external_db": "",
        "source": "",
        "source_version": "",
        "source_commit": "",
        "status": "not_applicable",
        "command_or_notebook": "",
        "notes": (
            "LLSim5G's A2G implementation cites Khawaja et al. (2019), not "
            "3GPP TR 36.777; the deterministic standard vector remains required."
        ),
    })
    rows.append({
        "id": "38811_basic_600km_12ghz_plus_clutter",
        "external_db": values["38811_basic_600km_12ghz_plus_clutter"],
        "source": "OpenNTN",
        "source_version": tools["OpenNTN"]["version"],
        "source_commit": tools["OpenNTN"]["commit"],
        "status": "validated",
        "command_or_notebook": command,
        "notes": "Upstream equation 6.6-1 with sigmaSF=0 and declared 2.5 dB NLOS clutter.",
    })
    rows.sort(key=lambda row: str(row["id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--llsim-root", type=Path, required=True)
    parser.add_argument("--openntn-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generated = generate(args.tools, args.llsim_root, args.openntn_root, args.output)
    for row in generated:
        print(row)
