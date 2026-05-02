import numpy as np

def laplacian(field):
    return (
        np.roll(field, 1, axis=0) +
        np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) +
        np.roll(field, -1, axis=1) -
        4 * field
    )

def gradient(field):
    gx = np.roll(field, -1, axis=1) - field
    gy = np.roll(field, -1, axis=0) - field
    return gx, gy

def curl(Jx, Jy):
    return np.roll(Jy, -1, axis=1) - Jy - (np.roll(Jx, -1, axis=0) - Jx)