import numpy as np

def effective_density(X, K, alpha=0.1):
    return X**2 + alpha * K

def effective_metric(X, Jx, Jy, W, K):
    return 1 + X**2 + np.abs(W) + np.abs(Jx) + np.abs(Jy) + K