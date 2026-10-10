
import numpy as np
import pytest

from src.optim.momentum import optimize


def f(x):
    return x[0]**2 + 3 * x[1]**2


def grad_f(x):
    return np.array([2 * x[0], 6 * x[1]])


def test_momentum_first_iteration():
    x, hist, iterations = optimize(
        f, grad_f, [5.0, 2.0],
        alpha=0.1, beta=0.6, max_iter=1
    )

    # После одного шага Momentum
    assert np.allclose(x, [4.0, 0.8])
    assert np.allclose(hist[0], [5.0, 2.0])
    assert np.allclose(hist[-1], [4.0, 0.8])
    assert len(hist) == 2
    assert iterations == 1


def test_momentum_second_iteration():
    x, hist, iterations = optimize(
        f, grad_f, [5.0, 2.0],
        alpha=0.1, beta=0.6, max_iter=2
    )

    assert np.allclose(x, [2.6, -0.4])
    assert np.isclose(f(x), 7.24)
    assert len(hist) == 3
    assert iterations == 2


def test_absolute_tolerance():
    x, hist, iterations = optimize(
        f, grad_f, [0.0, 0.0],
        tol_abs=1e-6
    )

    # В начальной точке градиент равен нулю
    assert np.allclose(x, [0.0, 0.0])
    assert len(hist) == 1
    assert iterations == 0


def test_relative_tolerance():
    x, hist, iterations = optimize(
        f, grad_f, [0.0, 0.0],
        tol_rel=1e-6
    )

    # Нулевой начальный градиент означает немедленную остановку
    assert np.allclose(x, [0.0, 0.0])
    assert iterations == 0


def test_max_iterations():
    x, hist, iterations = optimize(
        f, grad_f, [5.0, 2.0],
        alpha=0.1, beta=0.6, max_iter=5
    )

    assert iterations == 5
    assert len(hist) == 6
    assert np.allclose(x, hist[-1])


def test_default_parameters():
    x, hist, iterations = optimize(
        f, grad_f, [0.0, 0.0]
    )

    assert np.allclose(x, [0.0, 0.0])
    assert iterations == 0
    assert len(hist) == 1


def test_invalid_large_position_stops():
    def grad_large(x):
        return np.array([1.0])

    x, hist, iterations = optimize(
        f=lambda x: x[0]**2,
        grad=grad_large,
        x0=[1e13],
        max_iter=10
    )

    assert iterations == 0
    assert np.allclose(x, [1e13])


def test_history_contains_initial_point():
    x0 = np.array([5.0, 2.0])

    x, hist, iterations = optimize(
        f, grad_f, x0,
        alpha=0.1, beta=0.6, max_iter=3
    )

    assert np.allclose(hist[0], x0)
    assert np.allclose(hist[-1], x)
    assert len(hist) == iterations + 1