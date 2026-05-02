import numpy as np
from core.evolution import *
from core.fields import *

def run_simulation(size=64, steps=100):
    X = np.random.randn(size, size)
    phi = np.random.randn(size, size)
    K = np.zeros((size, size))

    for _ in range(steps):
        Jx, Jy = compute_flux(X, phi)
        W = curl(Jx, Jy)

        X = update_X(X, D=0.1, gamma=0.01, lam=0.02)
        phi = update_phi(phi, alpha=0.05, beta=0.02, W=W)
        K = update_K(K, kappa=0.1, delta=0.01, feedback=np.abs(W))

    return X, phi, K

if __name__ == "__main__":
    X, phi, K = run_simulation()
    print("Simulation complete")