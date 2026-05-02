"""Field construction, initialization, and derived OMM-SBT quantities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import curl_2d, gradient, laplacian, normalize_l2


Array = np.ndarray


@dataclass(frozen=True)
class FieldState:
    """Full OMM-SBT state.

    X: oscillatory mantle amplitude.
    phi: phase/orientation field.
    K: substrate kinetic field.
    """

    X: Array
    phi: Array
    K: Array


@dataclass(frozen=True)
class DerivedFields:
    """Derived OMM-SBT quantities."""

    density: Array
    Jx: Array
    Jy: Array
    flux: Array
    vorticity: Array
    curvature: Array
    metric: Array
    horizon: Array
    boundary: Array
    radiation: Array


def initialize_state(size: int = 64, family: str = "mixed", seed: int | None = None) -> FieldState:
    """Initialize an OMM-SBT state.

    Families:
    - compact: centralized mantle
    - dipole: two opposed mantles
    - ring: annular mantle
    - mixed: three-center asymmetric mantle
    """

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2.0
    Xn = (xx - c) / size
    Yn = (yy - c) / size
    r = np.sqrt(Xn * Xn + Yn * Yn) + 1e-12
    theta = np.arctan2(Yn, Xn)

    if family == "compact":
        X = np.exp(-(r / 0.18) ** 2) * (1.0 + 0.15 * np.cos(3.0 * theta))
        phi = theta + 0.35 * r
    elif family == "dipole":
        X = np.exp(-(((Xn + 0.16) ** 2 + Yn**2) / 0.012)) - 0.82 * np.exp(
            -(((Xn - 0.16) ** 2 + Yn**2) / 0.016)
        )
        phi = 0.70 * np.sin(2.0 * np.pi * Xn) + theta
    elif family == "ring":
        X = np.exp(-((r - 0.22) / 0.045) ** 2) + 0.25 * np.exp(-(r / 0.12) ** 2)
        phi = theta + 0.25 * np.sin(4.0 * theta)
    else:
        X = (
            np.exp(-(((Xn + 0.12) ** 2 + (Yn + 0.10) ** 2) / 0.012))
            - 0.75 * np.exp(-(((Xn - 0.10) ** 2 + (Yn - 0.11) ** 2) / 0.014))
            + 0.35 * np.exp(-(r / 0.18) ** 2)
        )
        phi = theta + 0.25 * r + 0.08 * np.sin(3.0 * theta)

    X = X + 0.0015 * rng.normal(size=X.shape)
    phi = phi + 0.0015 * rng.normal(size=phi.shape)
    K = 0.02 + 0.008 * np.abs(rng.normal(size=X.shape))
    return FieldState(normalize_l2(X), phi, K)


def derived_fields(
    state: FieldState,
    alpha_k: float = 0.50,
    metric_weights: tuple[float, float, float, float, float] = (0.20, 0.20, 0.18, 0.17, 0.25),
    horizon_quantile: float = 0.94,
) -> DerivedFields:
    """Compute derived fields from an OMM-SBT state."""

    X, phi, K = state.X, state.phi, state.K
    amplitude = np.abs(X)
    gx_phi, gy_phi = gradient(phi)

    Jx = amplitude * gx_phi
    Jy = amplitude * gy_phi
    flux = np.sqrt(Jx * Jx + Jy * Jy)
    vorticity = curl_2d(Jx, Jy)
    curvature = np.abs(laplacian(X))
    density = X * X + alpha_k * K

    density_n = density / (density.mean() + 1e-12)
    curvature_n = curvature / (curvature.mean() + 1e-12)
    vorticity_n = np.abs(vorticity) / (np.abs(vorticity).mean() + 1e-12)
    flux_n = flux / (flux.mean() + 1e-12)
    kinetic_n = K / (K.mean() + 1e-12)

    a, b, c, d, e = metric_weights
    metric = 1.0 + a * density_n + b * curvature_n + c * vorticity_n + d * flux_n + e * kinetic_n

    horizon = metric >= np.quantile(metric, horizon_quantile)
    boundary = horizon ^ (
        np.roll(horizon, 1, axis=0)
        & np.roll(horizon, -1, axis=0)
        & np.roll(horizon, 1, axis=1)
        & np.roll(horizon, -1, axis=1)
    )

    radiation = (
        0.34 * boundary.astype(float)
        + 0.24 * np.tanh(flux / (flux.mean() + 1e-12))
        + 0.24 * np.tanh(np.abs(vorticity) / (np.abs(vorticity).mean() + 1e-12))
        + 0.18 * np.tanh(curvature / (curvature.mean() + 1e-12))
    )

    return DerivedFields(
        density=density,
        Jx=Jx,
        Jy=Jy,
        flux=flux,
        vorticity=vorticity,
        curvature=curvature,
        metric=metric,
        horizon=horizon,
        boundary=boundary,
        radiation=radiation,
    )
