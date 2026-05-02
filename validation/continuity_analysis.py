#!/usr/bin/env python3
"""Generalized continuity analysis for OMM-SBT."""

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

from omm_sbt.analysis.metrics import generalized_continuity_score
from omm_sbt.core.evolution import run_evolution
from omm_sbt.core.fields import initialize_state
from omm_sbt.models.parameters import ModelParameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/sample_outputs/continuity_analysis.csv"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, family in enumerate(["compact", "dipole", "ring", "mixed"], start=1):
        state0 = initialize_state(size=args.size, family=family, seed=3000 + idx)
        states, _ = run_evolution(state0, ModelParameters(), steps=args.steps)
        result = generalized_continuity_score(states[-2], states[-1])
        rows.append({"family": family, **result})

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
