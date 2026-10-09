import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.optim.gd import gradient_descent, gradient_descent_backtracking
from src.problems.rosenbrock import f as rf, grad as rg
from src.problems.quadratic import q1, q1_grad


def test_rosenbrock_reference_fixed_gd():
    x, hist, k = gradient_descent(rf, rg, [-1.2, 1.0], 1e-3, relative=False)
    assert k == 32076
    assert np.linalg.norm(rg(x)) < 1e-6


def test_rosenbrock_reference_backtracking():
    x, hist, k = gradient_descent_backtracking(rf, rg, [-1.2, 1.0])
    assert k == 13756
    assert np.linalg.norm(rg(x)) < 1e-6


def test_q1_start_check():
    x0 = np.array([1.3, 0.7])
    assert np.isclose(q1(x0), 64.9)
    assert np.isclose(np.linalg.norm(q1_grad(x0)), 180.619, atol=5e-4)


def test_h1_two_fixed_gd_steps():
    f = lambda x: 2*x[0]**2 + 7*x[1]**2
    g = lambda x: np.array([4*x[0], 14*x[1]])
    x, hist, k = gradient_descent(f, g, [4.0, 1.0], 0.05, tol=0.0, max_iter=2)
    expected = np.array([2.56, 0.09])
    assert k == 2
    assert np.allclose(x, expected, atol=1e-12)


def test_h1_backtracking_first_step():
    f = lambda x: 2*x[0]**2 + 7*x[1]**2
    g = lambda x: np.array([4*x[0], 14*x[1]])
    x, hist, k, steps, halves = gradient_descent_backtracking(
        f, g, [4.0, 1.0], tol=0.0, max_iter=1, return_steps=True
    )
    assert k == 1
    assert np.isclose(steps[0], 0.125)
    assert halves[0] == 3
    assert np.allclose(x, [2.0, -0.75])
