"""Model parameters for the OMM-SBT reference implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParameters:
    """Numerical parameters for the coupled OMM-SBT update."""

    D: float = 0.16
    gamma: float = 0.008
    lam: float = 0.030
    mu: float = 0.020
    alpha_phase: float = 0.060
    beta_vorticity: float = 0.040
    kappa_kinetic: float = 0.120
    delta_kinetic: float = 0.018
    feedback: float = 0.040
    flux_push: float = 0.026
    vorticity_push: float = 0.016
    horizon_gate: float = 0.34
