"""Tests for the GD owner (M1): R1 reference counts, Q1/Q2 consistency, check_grad, H1 hand trace."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.optim.gd import gradient_descent, gradient_descent_backtracking
from src.problems.check_grad import check_grad
from src.problems.quadratic import f_q1, grad_q1, X0_Q1, f_q2, grad_q2, X0_Q2
from src.problems.rosenbrock import f as rf, grad as rg


# ---------------------------------------------------------------- R1 unit tests
def test_r1_fixed_gd_reference():
    x, hist, k = gradient_descent(rf, rg, [-1.2, 1.0], 1e-3, tol_abs=1e-6)
    assert k == 32076
    assert len(hist) == k + 1
    assert np.linalg.norm(rg(x)) < 1e-6


def test_r1_backtracking_reference():
    x, hist, k = gradient_descent_backtracking(rf, rg, [-1.2, 1.0], tol_abs=1e-6)
    assert k == 13756
    assert len(hist) == k + 1
    assert np.linalg.norm(rg(x)) < 1e-6


# ---------------------------------------------------------------- task-file check values
def test_start_check_values():
    assert np.isclose(rf(np.array([-1.2, 1.0])), 24.2)
    assert np.isclose(np.linalg.norm(rg(np.array([-1.2, 1.0]))), 232.86, atol=1e-2)
    assert np.isclose(rf(np.array([-1.51, 2.17])), 7.5123, atol=5e-5)
    assert np.isclose(np.linalg.norm(rg(np.array([-1.51, 2.17]))), 74.8335, atol=5e-5)
    for f, g, x0 in ((f_q1, grad_q1, X0_Q1), (f_q2, grad_q2, X0_Q2)):
        assert np.isclose(f(x0), 64.9)
        assert np.isclose(np.linalg.norm(g(x0)), 180.619, atol=5e-4)


# ---------------------------------------------------------------- Q1 vs Q2 consistency
def test_q1_q2_same_counts_gd_and_bt():
    alpha = 1.0 / 130.0  # theoretical step 1/(1+c), c = 129
    _, _, k1 = gradient_descent(f_q1, grad_q1, X0_Q1, alpha, tol_rel=1e-6)
    _, _, k2 = gradient_descent(f_q2, grad_q2, X0_Q2, alpha, tol_rel=1e-6)
    assert abs(k1 - k2) <= 1
    assert k1 == 892
    _, _, b1 = gradient_descent_backtracking(f_q1, grad_q1, X0_Q1, tol_rel=1e-6)
    _, _, b2 = gradient_descent_backtracking(f_q2, grad_q2, X0_Q2, tol_rel=1e-6)
    assert abs(b1 - b2) <= 1


# ---------------------------------------------------------------- interface / safeguards
def test_cap_and_blowup_guard():
    # iteration cap: exactly max_iter updates, len(hist) = k + 1
    _, hist, k = gradient_descent(rf, rg, [-1.2, 1.0], 1e-3, tol_abs=1e-12, max_iter=50)
    assert k == 50 and len(hist) == 51
    # alpha above 2/lambda_max on Q1 (t = 1.01) must hit the blow-up guard, not converge
    alpha = 1.01 * 2.0 / 258.0
    x, _, k = gradient_descent(f_q1, grad_q1, X0_Q1, alpha, tol_rel=1e-6)
    assert np.linalg.norm(x) > 1e12 and k < 100_000


def test_stop_checked_before_update():
    # start already satisfies the absolute rule -> zero updates
    x, hist, k = gradient_descent(rf, rg, [1.0, 1.0], 1e-3, tol_abs=1e-6)
    assert k == 0 and len(hist) == 1
    x, hist, k = gradient_descent_backtracking(rf, rg, [1.0, 1.0], tol_abs=1e-6)
    assert k == 0 and len(hist) == 1


# ---------------------------------------------------------------- H1 hand trace (to 1e-3)
def _h1():
    f = lambda x: 2 * x[0] ** 2 + 7 * x[1] ** 2
    g = lambda x: np.array([4 * x[0], 14 * x[1]])
    return f, g


def test_h1_fixed_gd_table():
    f, g = _h1()
    x, hist, k = gradient_descent(f, g, [4.0, 1.0], 0.05, tol_abs=0.0, max_iter=2)
    assert k == 2 and len(hist) == 3
    for xk, xe, ge, fe in (
        (hist[0], [4.0, 1.0], [16.0, 14.0], 39.0),
        (hist[1], [3.2, 0.3], [12.8, 4.2], 21.11),
        (hist[2], [2.56, 0.09], [10.24, 1.26], 13.1639),
    ):
        assert np.allclose(xk, xe, atol=1e-3)
        assert np.allclose(g(xk), ge, atol=1e-3)
        assert abs(f(xk) - fe) < 1e-3


def test_h1_backtracking_step():
    f, g = _h1()
    x, hist, k, acc, halv = gradient_descent_backtracking(
        f, g, [4.0, 1.0], tol_abs=0.0, max_iter=1, return_steps=True
    )
    assert k == 1
    assert abs(acc[0] - 0.125) < 1e-3 and halv[0] == 3
    assert np.allclose(hist[1], [2.0, -0.75], atol=1e-3)
    assert abs(f(hist[1]) - 11.9375) < 1e-3
    # rejected trials: alpha = 1, 0.5, 0.25 (Armijo violated), alpha = 0.125 accepted
    x0 = np.array([4.0, 1.0]); d = -g(x0)
    for a, ok in ((1.0, False), (0.5, False), (0.25, False), (0.125, True)):
        assert (f(x0 + a * d) <= f(x0) + 1e-4 * a * (g(x0) @ d)) == ok


# ---------------------------------------------------------------- check_grad (central differences, < 1e-6)
def test_rosenbrock_gradient_check():
    for p in ([-1.2, 1.0], [-1.51, 2.17], [0.3, -0.4]):
        err, _, _ = check_grad(rf, rg, np.array(p))
        assert err < 1e-6


def test_q1_q2_gradient_check():
    assert check_grad(f_q1, grad_q1, X0_Q1)[0] < 1e-6
    assert check_grad(f_q2, grad_q2, X0_Q2)[0] < 1e-6
