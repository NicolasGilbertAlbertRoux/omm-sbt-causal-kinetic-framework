#!/usr/bin/env python3
"""Run all public OMM-SBT validation scripts."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/sample_outputs"))
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "experiments/run_simulation.py", "--output", str(args.output_dir / "simulation.csv")])
    run([sys.executable, "validation/independent_reproduction.py", "--output", str(args.output_dir / "independent_reproduction.csv")])
    run([sys.executable, "validation/invariance_scan.py", "--output", str(args.output_dir / "invariance_scan.csv")])
    run([sys.executable, "validation/continuity_analysis.py", "--output", str(args.output_dir / "continuity_analysis.csv")])


if __name__ == "__main__":
    main()
