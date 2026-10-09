import numpy as np

from src.optim.gd import gradient_descent, gradient_descent_backtracking


def f(x):
    return 2.0 * x[0] ** 2 + 7.0 * x[1] ** 2


def grad(x):
    return np.array([4.0 * x[0], 14.0 * x[1]])


def test_h1_fixed_gd_trace():
    x0 = np.array([4.0, 1.0])

    x, hist, k = gradient_descent(
        f=f,
        grad=grad,
        x0=x0,
        alpha=0.05,
        tol=0.0,
        max_iter=2,
    )

    assert k == 2
    assert len(hist) == 3

    assert np.allclose(hist[0], [4.0, 1.0], atol=1e-3)
    assert np.allclose(hist[1], [3.2, 0.3], atol=1e-3)
    assert np.allclose(hist[2], [2.56, 0.09], atol=1e-3)

    assert abs(f(hist[0]) - 39.0) < 1e-3
    assert abs(f(hist[1]) - 21.11) < 1e-3
    assert abs(f(hist[2]) - 13.1639) < 1e-3


def test_h1_backtracking_trace():
    x0 = np.array([4.0, 1.0])

    x, hist, k, accepted, halvings = gradient_descent_backtracking(
        f=f,
        grad=grad,
        x0=x0,
        tol=0.0,
        max_iter=1,
        alpha0=1.0,
        armijo_c=1e-4,
        return_steps=True,
    )

    assert k == 1
    assert len(hist) == 2

    assert len(accepted) == 1
    assert len(halvings) == 1

    assert abs(accepted[0] - 0.125) < 1e-3
    assert halvings[0] == 3

    assert np.allclose(hist[1], [2.0, -0.75], atol=1e-3)
    assert abs(f(hist[1]) - 11.9375) < 1e-3
