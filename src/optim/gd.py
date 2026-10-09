import numpy as np


def gradient_descent(f, grad, x0, alpha, max_iter=100_000, tol_abs=None, tol_rel=None):
    """Fixed-step gradient descent.

    Returns (x, hist, k), where hist contains x^(0),...,x^(k) and k is the
    number of updates. The stopping test is performed before each update.
    """
    x = np.asarray(x0, dtype=float).copy()
    hist = [x.copy()]
    g0_norm = float(np.linalg.norm(grad(x)))

    for k in range(max_iter + 1):
        g = np.asarray(grad(x), dtype=float)
        g_norm = np.linalg.norm(g)
        
        # Проверка на NaN, Inf или слишком большие значения (Blow-up guard)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(g)) or np.linalg.norm(x) > 1e12:
            return x, np.asarray(hist), k
            
        # Проверка абсолютной остановки (как для Розенброка)
        if tol_abs is not None and g_norm < tol_abs:
            return x, np.asarray(hist), k
            
        # Проверка относительной остановки (как для квадратичных функций)
        if tol_rel is not None and g_norm < tol_rel * g0_norm:
            return x, np.asarray(hist), k
            
        if k == max_iter:
            return x, np.asarray(hist), k
            
        # Шаг алгоритма
        x = x - alpha * g
        hist.append(x.copy())

    return x, np.asarray(hist), max_iter


def gradient_descent_backtracking(
    f, grad, x0, max_iter=100_000, tol_abs=None, tol_rel=None,
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
    accepted = []
    halvings = []

    for k in range(max_iter + 1):
        g = np.asarray(grad(x), dtype=float)
        g_norm = np.linalg.norm(g)
        
        # Проверка на NaN, Inf или взрыв
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(g)) or np.linalg.norm(x) > 1e12:
            break
            
        # Проверки остановки
        if tol_abs is not None and g_norm < tol_abs:
            break
        if tol_rel is not None and g_norm < tol_rel * g0_norm:
            break
            
        if k == max_iter:
            break

        d = -g
        alpha = float(alpha0)
        h = 0
        fx = float(f(x))
        gd = float(np.dot(g, d))
        
        # Условие Армихо
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