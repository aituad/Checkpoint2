import numpy as np


def check_grad(f, grad, x, eps=1e-6):
    """Central-difference gradient check.

    Returns the maximum absolute componentwise discrepancy.
    """
    x = np.asarray(x, dtype=float).copy()
    analytic = np.asarray(grad(x), dtype=float)
    numeric = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy(); xm = x.copy()
        xp[i] += eps; xm[i] -= eps
        numeric[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return float(np.max(np.abs(analytic - numeric))), analytic, numeric
