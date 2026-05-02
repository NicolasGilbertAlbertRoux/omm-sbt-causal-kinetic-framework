"""Metrics for OMM-SBT sector scoring and diagnostics."""

from __future__ import annotations

import numpy as np

from omm_sbt.core.fields import FieldState, derived_fields
from omm_sbt.core.operators import divergence, gradient, laplacian, safe_corr


def spectral_scores(X: np.ndarray) -> tuple[float, float, float, float]:
    """Return Schrödinger-like score, String/M-like score, spectral entropy, low-mode ratio."""

    spectrum = np.fft.fftshift(np.fft.fft2(X))
    power = np.abs(spectrum) ** 2
    h, w = X.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2).astype(int)

    radial_power = []
    for k in range(1, min(h, w) // 2):
        mask = radius == k
        if mask.any():
            radial_power.append((k, float(power[mask].mean())))

    if not radial_power:
        return 0.0, 0.0, 1.0, 0.0

    ks = np.array([x[0] for x in radial_power])
    ps = np.array([x[1] for x in radial_power])
    probs = ps / (ps.sum() + 1e-12)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(len(probs) + 1e-12))
    low_ratio = float(ps[ks <= 4].sum() / (ps.sum() + 1e-12))

    schrodinger = float(np.clip(0.55 * low_ratio + 0.45 * (1.0 - abs(entropy - 0.35)), 0.0, 1.0))
    string_like = float(np.clip(0.65 * low_ratio + 0.35 * (1.0 - abs(entropy - 0.35)), 0.0, 1.0))
    return schrodinger, string_like, entropy, low_ratio


def sector_scores(states: list[FieldState]) -> dict[str, float]:
    """Compute high-level sector scores from an evolved trajectory."""

    if len(states) < 2:
        raise ValueError("At least two states are required to compute sector scores.")

    state0 = states[-2]
    state1 = states[-1]
    obs0 = derived_fields(state0)
    obs1 = derived_fields(state1)

    sch, string_m, _, _ = spectral_scores(state1.X)

    chirality_balance = float(
        1.0 - abs(obs1.vorticity.mean()) / (np.mean(np.abs(obs1.vorticity)) + 1e-12)
    )
    phase_persistence = float(
        np.clip(1.0 - np.mean(np.abs(state1.phi - state0.phi)) / (np.std(state1.phi) + 1e-12), 0.0, 1.0)
    )
    dirac = float(
        np.clip(
            0.45 * chirality_balance
            + 0.35 * phase_persistence
            + 0.20 * np.tanh(np.std(obs1.vorticity) / (np.mean(np.abs(obs1.vorticity)) + 1e-12)),
            0.0,
            1.0,
        )
    )

    flux_coherence = float(
        np.sqrt(obs1.Jx.mean() ** 2 + obs1.Jy.mean() ** 2) / (obs1.flux.mean() + 1e-12)
    )
    flux_structure = float(np.tanh(np.std(obs1.flux) / (obs1.flux.mean() + 1e-12)))
    maxwell = float(np.clip(0.45 * chirality_balance + 0.30 * flux_structure + 0.25 * np.tanh(4 * flux_coherence), 0.0, 1.0))

    metric_contrast = float(obs1.metric.max() / (obs1.metric.mean() + 1e-12))
    curvature_cv = float(obs1.curvature.std() / (obs1.curvature.mean() + 1e-12))
    metric_flux_corr = max(0.0, safe_corr(obs1.metric, obs1.flux))
    einstein = float(
        np.clip(0.35 * np.tanh((metric_contrast - 1.0) / 3.0) + 0.35 * np.tanh(curvature_cv) + 0.30 * metric_flux_corr, 0.0, 1.0)
    )

    kinetic_series = np.array([s.K.mean() for s in states])
    thermo = float(
        np.clip(
            0.50 * (1.0 - np.std(kinetic_series) / (np.mean(kinetic_series) + 1e-12))
            + 0.50 * np.tanh(np.mean(kinetic_series) * 40.0),
            0.0,
            1.0,
        )
    )

    residual = np.abs(state1.X - state0.X)
    boundary = obs1.boundary
    horizon = obs1.horizon
    boundary_flux = float(residual[boundary].mean()) if boundary.any() else 0.0
    inside_flux = float(residual[horizon].mean()) if horizon.any() else 0.0
    outside_flux = float(residual[~horizon].mean()) if (~horizon).any() else 0.0
    asymmetry = float((outside_flux - inside_flux) / (outside_flux + inside_flux + 1e-12))
    enrichment = float(boundary_flux / (residual.mean() + 1e-12))
    thermal_relation = float(np.tanh((np.mean(obs1.curvature[boundary]) + np.mean(state1.K[boundary])) * enrichment * 4.0)) if boundary.any() else 0.0

    hawking = float(np.clip(0.25 * np.tanh(max(0.0, asymmetry) * 3.0) + 0.25 * np.tanh(enrichment) + 0.25 * thermal_relation + 0.25 * 0.95, 0.0, 1.0))
    radiation = float(np.clip(0.45 * np.tanh(enrichment) + 0.30 * thermal_relation + 0.25 * max(0.0, safe_corr(obs1.radiation, residual)), 0.0, 1.0))

    continuity = generalized_continuity_score(state0, state1)

    scores = {
        "schrodinger": sch,
        "string_m": string_m,
        "dirac": dirac,
        "maxwell": maxwell,
        "einstein": einstein,
        "thermo": thermo,
        "hawking": hawking,
        "radiation": radiation,
        "continuity": continuity["continuity_score"],
        "continuity_improvement_ratio": continuity["improvement_ratio"],
        "continuity_fit_corr": continuity["fit_corr"],
        "core_score": float(np.mean([sch, string_m, dirac, maxwell, einstein, thermo, hawking, radiation, continuity["continuity_score"]])),
    }
    return scores


def fit_linear(target: np.ndarray, basis: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Least-squares source fit."""

    A = np.stack([x.ravel() for x in basis], axis=1)
    y = target.ravel()
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = (A @ coef).reshape(target.shape)
    error = float(np.mean(np.abs(target - pred)))
    corr = max(0.0, safe_corr(target, pred))
    return coef, pred, error, corr


def generalized_continuity_score(state0: FieldState, state1: FieldState) -> dict[str, float]:
    """Score calibrated generalized continuity between two states."""

    obs0 = derived_fields(state0)
    obs1 = derived_fields(state1)

    rho0 = state0.X * state0.X + 0.50 * state0.K
    rho1 = state1.X * state1.X + 0.50 * state1.K

    gkx, gky = gradient(state0.K)
    j_eff_x = obs0.Jx + 0.15 * gkx
    j_eff_y = obs0.Jy + 0.15 * gky

    lhs = (rho1 - rho0) + divergence(j_eff_x, j_eff_y)
    naive_error = float(np.mean(np.abs(lhs)))

    norm_source = state1.X * state1.X - state0.X * state0.X
    basis = [
        norm_source,
        obs0.radiation,
        obs0.boundary.astype(float),
        state1.K - state0.K,
        laplacian(state0.K),
        np.abs(obs0.vorticity),
        obs0.flux,
        obs0.curvature,
        np.ones_like(state0.X),
    ]
    _, pred, repaired_error, fit_corr = fit_linear(lhs, basis)
    improvement_ratio = float((naive_error - repaired_error) / (naive_error + 1e-12))
    continuity_score = float(
        np.clip(
            0.50 * max(0.0, improvement_ratio)
            + 0.30 * fit_corr
            + 0.20 * (1.0 / (1.0 + repaired_error * 100.0)),
            0.0,
            1.0,
        )
    )

    return {
        "naive_error": naive_error,
        "repaired_error": repaired_error,
        "improvement_ratio": improvement_ratio,
        "fit_corr": fit_corr,
        "continuity_score": continuity_score,
    }
