import numpy as np


def _stop_threshold(grad0_norm, tol, relative):
    return tol * grad0_norm if relative else tol


def gradient_descent(f, grad, x0, alpha, tol=1e-6, relative=False, max_iter=100_000):
    """Fixed-step gradient descent.

    Returns (x, hist, k), where hist contains x^(0),...,x^(k) and k is the
    number of updates. The stopping test is performed before each update.
    """
    x = np.asarray(x0, dtype=float).copy()
    hist = [x.copy()]
    g0_norm = float(np.linalg.norm(grad(x)))
    threshold = _stop_threshold(g0_norm, tol, relative)

    for k in range(max_iter + 1):
        g = np.asarray(grad(x), dtype=float)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(g)) or np.linalg.norm(x) > 1e12:
            return x, np.asarray(hist), k
        if np.linalg.norm(g) < threshold:
            return x, np.asarray(hist), k
        if k == max_iter:
            return x, np.asarray(hist), k
        x = x - alpha * g
        hist.append(x.copy())

    return x, np.asarray(hist), max_iter


def gradient_descent_backtracking(
    f, grad, x0, tol=1e-6, relative=False, max_iter=100_000,
    alpha0=1.0, armijo_c=1e-4, return_steps=False
):
    """Gradient descent with Armijo backtracking.

    Uses d=-grad(x), starts alpha=1, halves alpha until the Armijo condition
    in the assignment is satisfied. Returns (x, hist, k), or additionally
    accepted step sizes and halving counts when return_steps=True.
    """
    x = np.asarray(x0, dtype=float).copy()
    hist = [x.copy()]
    g0_norm = float(np.linalg.norm(grad(x)))
    threshold = _stop_threshold(g0_norm, tol, relative)
    accepted = []
    halvings = []

    for k in range(max_iter + 1):
        g = np.asarray(grad(x), dtype=float)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(g)) or np.linalg.norm(x) > 1e12:
            break
        if np.linalg.norm(g) < threshold:
            break
        if k == max_iter:
            break

        d = -g
        alpha = float(alpha0)
        h = 0
        fx = float(f(x))
        gd = float(np.dot(g, d))
        while float(f(x + alpha * d)) > fx + armijo_c * alpha * gd:
            alpha *= 0.5
            h += 1
            if alpha == 0.0:
                break
        if alpha == 0.0:
            break

        x = x + alpha * d
        hist.append(x.copy())
        accepted.append(alpha)
        halvings.append(h)

    out = (x, np.asarray(hist), len(hist) - 1)
    if return_steps:
        return out + (np.asarray(accepted), np.asarray(halvings, dtype=int))
    return out
