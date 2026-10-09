"""Q1/Q2: f(x)=x1²+c*x2² and f(R.T@x)."""
import numpy as np


def quadratic(c=129, theta_deg=0):
    theta = np.deg2rad(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    H = R @ np.diag([2., 2.*c]) @ R.T
    def f(x):
        return float(np.asarray(x) @ H @ np.asarray(x) / 2)
    def grad(x):
        return H @ np.asarray(x)
    def hess(x):
        return H.copy()
    return f, grad, hess, R @ np.array([1.3, .7])
