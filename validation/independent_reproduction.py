#!/usr/bin/env python3
"""Independent compact reproduction protocol for OMM-SBT.

This script intentionally uses only the packaged core implementation and a
small grid of families/parameters. It is the clean public analogue of the V115
independent reproduction stage.
"""

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

from omm_sbt.analysis.metrics import sector_scores
from omm_sbt.core.evolution import run_evolution
from omm_sbt.core.fields import initialize_state
from omm_sbt.models.parameters import ModelParameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/sample_outputs/independent_reproduction.csv"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()


def status(score: float) -> str:
    if score >= 0.70:
        return "derived_candidate"
    if score >= 0.40:
        return "robust"
    return "weak"


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    families = ["compact", "dipole", "ring", "mixed"]
    params_grid = [
        ModelParameters(D=0.12, gamma=0.010, lam=0.025, mu=0.010, alpha_phase=0.04, beta_vorticity=0.025, kappa_kinetic=0.08, feedback=0.020),
        ModelParameters(D=0.16, gamma=0.008, lam=0.030, mu=0.020, alpha_phase=0.06, beta_vorticity=0.040, kappa_kinetic=0.12, feedback=0.040),
        ModelParameters(D=0.20, gamma=0.006, lam=0.035, mu=0.035, alpha_phase=0.08, beta_vorticity=0.055, kappa_kinetic=0.16, feedback=0.060),
    ]

    rows = []
    case_id = 0
    for family in families:
        for params in params_grid:
            case_id += 1
            state0 = initialize_state(size=args.size, family=family, seed=1000 + case_id)
            states, _ = run_evolution(state0, params, steps=args.steps)
            scores = sector_scores(states)
            rows.append(
                {
                    "case_id": case_id,
                    "family": family,
                    **params.__dict__,
                    **scores,
                }
            )

    cases = pd.DataFrame(rows)
    cases.to_csv(args.output, index=False)

    sector_cols = ["schrodinger", "string_m", "dirac", "maxwell", "einstein", "thermo", "hawking", "radiation", "continuity"]
    summary = pd.DataFrame(
        [
            {
                "axis": col,
                "score_mean": float(cases[col].mean()),
                "score_min": float(cases[col].min()),
                "status": status(float(cases[col].mean())),
                "derived_rate": float((cases[col] >= 0.70).mean()),
                "robust_rate": float((cases[col] >= 0.40).mean()),
            }
            for col in sector_cols
        ]
    )

    summary_path = args.output.with_name("independent_reproduction_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("=== Independent reproduction cases ===")
    print(cases.to_string(index=False))
    print("\n=== Independent reproduction summary ===")
    print(summary.to_string(index=False))
    print(f"[OK] wrote {args.output}")
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
