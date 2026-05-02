#!/usr/bin/env python3
"""Run a single OMM-SBT simulation and export summary data."""

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

from omm_sbt.core.evolution import run_evolution
from omm_sbt.core.fields import initialize_state
from omm_sbt.models.parameters import ModelParameters
from omm_sbt.analysis.metrics import sector_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--family", type=str, default="mixed", choices=["compact", "dipole", "ring", "mixed"])
    parser.add_argument("--seed", type=int, default=115)
    parser.add_argument("--output", type=Path, default=Path("results/sample_outputs/simulation.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    state0 = initialize_state(size=args.size, family=args.family, seed=args.seed)
    params = ModelParameters()
    states, _ = run_evolution(state0, params, steps=args.steps)
    scores = sector_scores(states)

    df = pd.DataFrame([{**vars(args), **scores}])
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
