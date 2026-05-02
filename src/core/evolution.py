import numpy as np
from .operators import laplacian, gradient, curl

def update_X(X, D, gamma, lam):
    return X + D * laplacian(X) - gamma * X - lam * X**3

def update_phi(phi, alpha, beta, W):
    return phi + alpha * laplacian(phi) + beta * W

def compute_flux(X, phi):
    gx, gy = gradient(phi)
    Jx = np.abs(X) * gx
    Jy = np.abs(X) * gy
    return Jx, Jy

def update_K(K, kappa, delta, feedback):
    return K + kappa * laplacian(K) - delta * K + feedback