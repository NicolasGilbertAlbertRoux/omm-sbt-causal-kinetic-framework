"""Evolution equations for the OMM-SBT causal-kinetic framework."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omm_sbt.core.fields import FieldState, derived_fields
from omm_sbt.core.operators import grad_magnitude, gradient, laplacian, normalize_l2
from omm_sbt.models.parameters import ModelParameters


@dataclass(frozen=True)
class StepDiagnostics:
    """Diagnostic fields generated during one update step."""

    pre_normalized_X: np.ndarray
    boundary: np.ndarray
    radiation: np.ndarray


def evolve_once(state: FieldState, params: ModelParameters) -> tuple[FieldState, StepDiagnostics]:
    """Perform one coupled OMM-SBT update.

    The update implements:
    - mantle diffusion/damping/saturation,
    - phase diffusion/vorticity coupling,
    - kinetic feedback from flux, vorticity, horizon boundary and radiation,
    - global finite-substrate normalization.
    """

    obs = derived_fields(state)
    X, phi, K = state.X, state.phi, state.K

    finite_correction = 2.0 * params.mu * (np.sum(X * X) - 1.0) * X

    pre_X = (
        X
        + params.D * laplacian(X)
        - params.gamma * X
        - params.lam * X**3
        - finite_correction
        + params.flux_push * np.tanh(obs.flux)
        - params.vorticity_push * np.sign(X) * np.tanh(np.abs(obs.vorticity))
    )

    grad_x, grad_y = gradient(X)
    phi_next = (
        phi
        + params.alpha_phase * laplacian(phi)
        + params.beta_vorticity * obs.vorticity
        + 0.010 * (grad_x * obs.Jx + grad_y * obs.Jy)
    )

    kinetic_production = (
        0.045 * grad_magnitude(X) ** 2
        + 0.040 * np.abs(obs.vorticity)
        + 0.040 * obs.flux
        + 0.060 * obs.boundary.astype(float)
        + params.feedback * obs.radiation
    )

    K_next = K + params.kappa_kinetic * laplacian(K) - params.delta_kinetic * K + kinetic_production
    K_next = np.clip(K_next, 1e-9, None)

    X_pre = normalize_l2(pre_X)
    obs_pre = derived_fields(FieldState(X_pre, phi_next, K_next))
    gate = np.where(obs_pre.horizon, params.horizon_gate, 1.0)
    X_next = normalize_l2(X + gate * (X_pre - X))

    return FieldState(X_next, phi_next, K_next), StepDiagnostics(
        pre_normalized_X=pre_X,
        boundary=obs.boundary,
        radiation=obs.radiation,
    )


def run_evolution(
    initial_state: FieldState,
    params: ModelParameters,
    steps: int = 100,
) -> tuple[list[FieldState], list[StepDiagnostics]]:
    """Run several update steps and return all states and diagnostics."""

    states = [initial_state]
    diagnostics: list[StepDiagnostics] = []
    state = initial_state
    for _ in range(steps):
        state, diag = evolve_once(state, params)
        states.append(state)
        diagnostics.append(diag)
    return states, diagnostics
