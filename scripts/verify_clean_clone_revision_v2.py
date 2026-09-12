"""Clean-clone verification entry point for a release candidate checkout."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def run(checkout: Path) -> None:
    commands = [
        [sys.executable, "-m", "pytest", "tests", "-q"],
        [sys.executable, "scripts/run_physical_conformance.py"],
        [sys.executable, "scripts/release_gate_revision_v2.py"],
    ]
    for command in commands:
        completed = subprocess.run(command, cwd=checkout, check=False)
        if completed.returncode:
            raise SystemExit(f"clean-clone verification failed: {' '.join(command)}")
    print("clean-clone verification passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path, help="fresh clone with dependencies installed")
    args = parser.parse_args()
    run(args.checkout.resolve())
