import numpy as np


def f(x):
    x = np.asarray(x, dtype=float)
    x1, x2 = x
    return (1.0 - x1) ** 2 + 100.0 * (x2 - x1 ** 2) ** 2


def grad(x):
    x = np.asarray(x, dtype=float)
    x1, x2 = x
    return np.array([
        -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1 ** 2),
        200.0 * (x2 - x1 ** 2),
    ])


def hess(x):
    x = np.asarray(x, dtype=float)
    x1, x2 = x
    return np.array([
        [2.0 - 400.0 * x2 + 1200.0 * x1 ** 2, -400.0 * x1],
        [-400.0 * x1, 200.0],
    ])
