"""Numerical operators used by the OMM-SBT reference implementation."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def roll_x(field: Array, shift: int) -> Array:
    """Periodic roll along the x axis."""
    return np.roll(field, shift, axis=1)


def roll_y(field: Array, shift: int) -> Array:
    """Periodic roll along the y axis."""
    return np.roll(field, shift, axis=0)


def laplacian(field: Array) -> Array:
    """Second-order periodic finite-difference Laplacian."""
    return (
        roll_x(field, 1)
        + roll_x(field, -1)
        + roll_y(field, 1)
        + roll_y(field, -1)
        - 4.0 * field
    )


def gradient(field: Array) -> tuple[Array, Array]:
    """Central finite-difference gradient as (gx, gy)."""
    gx = 0.5 * (roll_x(field, -1) - roll_x(field, 1))
    gy = 0.5 * (roll_y(field, -1) - roll_y(field, 1))
    return gx, gy


def divergence(jx: Array, jy: Array) -> Array:
    """Central finite-difference divergence of a 2D vector field."""
    return gradient(jx)[0] + gradient(jy)[1]


def curl_2d(jx: Array, jy: Array) -> Array:
    """Scalar 2D curl: d_y/dx - d_x/dy."""
    return gradient(jy)[0] - gradient(jx)[1]


def grad_magnitude(field: Array) -> Array:
    """Gradient magnitude."""
    gx, gy = gradient(field)
    return np.sqrt(gx * gx + gy * gy)


def normalize_l2(field: Array, eps: float = 1e-12) -> Array:
    """L2-normalize a field."""
    return field / (np.linalg.norm(field) + eps)


def safe_corr(a: Array, b: Array) -> float:
    """Safe Pearson correlation between arrays."""
    av = np.asarray(a).ravel()
    bv = np.asarray(b).ravel()
    n = min(len(av), len(bv))
    if n < 3:
        return 0.0
    av = av[:n]
    bv = bv[:n]
    if np.std(av) < 1e-12 or np.std(bv) < 1e-12:
        return 0.0
    c = np.corrcoef(av, bv)[0, 1]
    return float(c) if np.isfinite(c) else 0.0
