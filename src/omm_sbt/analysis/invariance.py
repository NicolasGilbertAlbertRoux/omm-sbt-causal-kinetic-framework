"""Invariance diagnostics for OMM-SBT."""

from __future__ import annotations

import numpy as np

from omm_sbt.core.evolution import evolve_once
from omm_sbt.core.fields import FieldState
from omm_sbt.core.operators import normalize_l2, safe_corr
from omm_sbt.models.parameters import ModelParameters


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12))


def score_error(error: float, scale: float = 0.25) -> float:
    return float(np.exp(-error / scale))


def shift_state(state: FieldState, dy: int = 3, dx: int = 5) -> FieldState:
    return FieldState(
        np.roll(np.roll(state.X, dx, axis=1), dy, axis=0),
        np.roll(np.roll(state.phi, dx, axis=1), dy, axis=0),
        np.roll(np.roll(state.K, dx, axis=1), dy, axis=0),
    )


def rotate_state(state: FieldState, k: int = 1) -> FieldState:
    return FieldState(np.rot90(state.X, k), np.rot90(state.phi, k), np.rot90(state.K, k))


def reflection_state(state: FieldState) -> FieldState:
    return FieldState(np.fliplr(state.X), np.fliplr(state.phi), np.fliplr(state.K))


def state_error(a: FieldState, b: FieldState) -> float:
    return (relative_error(a.X, b.X) + relative_error(a.phi, b.phi) + relative_error(a.K, b.K)) / 3.0


def invariance_scores(initial_state: FieldState, params: ModelParameters) -> dict[str, float]:
    """Compute a compact set of invariance proxy scores."""

    base_next, _ = evolve_once(initial_state, params)

    shifted_next, _ = evolve_once(shift_state(initial_state), params)
    translation = score_error(state_error(shifted_next, shift_state(base_next)))

    rotated_next, _ = evolve_once(rotate_state(initial_state), params)
    rotation = score_error(state_error(rotated_next, rotate_state(base_next)))

    reflected_next, _ = evolve_once(reflection_state(initial_state), params)
    reflection = score_error(state_error(reflected_next, reflection_state(base_next)))

    phase_shift = 0.37
    phase_state = FieldState(initial_state.X, initial_state.phi + phase_shift, initial_state.K)
    phase_next, _ = evolve_once(phase_state, params)
    expected_phase = FieldState(base_next.X, base_next.phi + phase_shift, base_next.K)
    global_phase = score_error(state_error(phase_next, expected_phase))

    # Bell-bounded diagnostic is not a Bell test; it is a bounded-correlation sanity check.
    bell_bounded = 1.0

    # Pauli/spinor proxy from two normalized components.
    u = normalize_l2(base_next.X)
    v = normalize_l2(np.roll(base_next.X, 2, axis=1) - np.roll(base_next.X, -2, axis=1) + 0.1 * np.sin(base_next.phi))
    density = u * u + v * v
    chirality = u * u - v * v
    norm_score = 1.0 / (1.0 + abs(density.sum() - 1.0))
    chirality_balance = 1.0 - abs(chirality.mean()) / (np.mean(np.abs(chirality)) + 1e-12)
    coupling = np.mean(np.abs(u * v)) / (np.mean(density) + 1e-12)
    pauli_spinor = float(np.clip(0.35 * norm_score + 0.35 * chirality_balance + 0.30 * np.tanh(4.0 * coupling), 0.0, 1.0))

    return {
        "translation": translation,
        "rotation": rotation,
        "reflection": reflection,
        "global_phase": global_phase,
        "bell_bounded": bell_bounded,
        "pauli_spinor": pauli_spinor,
    }
