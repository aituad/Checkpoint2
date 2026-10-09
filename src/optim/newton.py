"""Pure and safeguarded Newton; NumPy and standard library only."""
import numpy as np


def newton(f, grad, hess, x0, *, damped=False, relative=False,
           tol=1e-6, max_iter=100):
    x = np.asarray(x0, dtype=float).copy()
    hist = [x.copy()]
    initial = np.linalg.norm(grad(x))
    threshold = tol * initial if relative else tol
    for k in range(max_iter + 1):
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e12:
            return x, np.asarray(hist), k
        g = np.asarray(grad(x))
        if np.linalg.norm(g) < threshold or (relative and initial == 0):
            return x, np.asarray(hist), k
        if k == max_iter:
            return x, np.asarray(hist), k
        try:
            p = np.linalg.solve(hess(x), -g)
        except np.linalg.LinAlgError:
            # A singular Hessian is a failed run; do not change pure Newton.
            return x, np.asarray(hist), k
        alpha = 1.0
        if damped:
            if np.dot(g, p) >= 0:
                p = -g
            fx, slope = f(x), np.dot(g, p)
            for _ in range(1075):
                trial = f(x + alpha * p)
                if np.isfinite(trial) and trial <= fx + 1e-4 * alpha * slope:
                    break
                alpha *= 0.5
                if alpha == 0 or np.array_equal(x + alpha * p, x):
                    return x, np.asarray(hist), k
            else:
                return x, np.asarray(hist), k
        x = x + alpha * p
        hist.append(x.copy())


def pure_newton(f, grad, hess, x0, **kwargs):
    return newton(f, grad, hess, x0, damped=False, **kwargs)


def damped_newton(f, grad, hess, x0, **kwargs):
    return newton(f, grad, hess, x0, damped=True, **kwargs)
