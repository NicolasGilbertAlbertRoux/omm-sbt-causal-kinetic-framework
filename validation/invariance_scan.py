#!/usr/bin/env python3
"""Compact invariance scan for OMM-SBT."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import argparse
from pathlib import Path

import pandas as pd

from omm_sbt.analysis.invariance import invariance_scores
from omm_sbt.core.fields import initialize_state
from omm_sbt.models.parameters import ModelParameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/sample_outputs/invariance_scan.csv"))
    parser.add_argument("--size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    families = ["compact", "dipole", "ring", "mixed"]
    rows = []
    for idx, family in enumerate(families, start=1):
        state = initialize_state(size=args.size, family=family, seed=2000 + idx)
        scores = invariance_scores(state, ModelParameters())
        rows.append({"family": family, **scores})

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    summary = pd.DataFrame(
        [
            {"test": col, "score_mean": float(df[col].mean()), "score_min": float(df[col].min())}
            for col in df.columns
            if col != "family"
        ]
    )
    summary_path = args.output.with_name("invariance_scan_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("=== Invariance scan ===")
    print(df.to_string(index=False))
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"[OK] wrote {args.output}")
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
