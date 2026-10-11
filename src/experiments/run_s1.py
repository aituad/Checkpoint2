"""S1 experiments (GD owner): step size and stability.  Run from the repo root:

    python -m src.experiments.run_s1

Writes results/s1_stability_q1.csv, results/s1_backtracking.csv and
results/s1_alpha0_sensitivity.csv.  The project-block rows need
src/problems/project.py (names below); without it they are skipped.
"""
import csv
from pathlib import Path

import numpy as np

from src.optim.gd import gradient_descent, gradient_descent_backtracking
from src.problems.quadratic import f_q1, grad_q1, hess_q1, X0_Q1
from src.problems.rosenbrock import f_rosenbrock, grad_rosenbrock, hess_rosenbrock

RESULTS = Path(__file__).resolve().parents[2] / "results"
SOLUTIONS = {"R1": np.array([1.0, 1.0])}


def save(name, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def gnorms(grad, hist):
    return np.array([np.linalg.norm(grad(p)) for p in hist])


def stability_q1():
    lam = np.linalg.eigvalsh(hess_q1(X0_Q1))
    lmax = lam.max()
    rows = []
    for t in (0.5, 0.9, 0.99, 1.01, 1.1):
        a = t * 2.0 / lmax
        factors = np.abs(1.0 - a * lam)          # per-mode |1 - alpha*lambda_i|
        x, hist, k = gradient_descent(f_q1, grad_q1, X0_Q1, a, tol_rel=1e-6)
        g = gnorms(grad_q1, hist)
        converged = bool(g[-1] < 1e-6 * g[0])
        n = min(50, len(g) - 1)
        rows.append({
            "t": t, "alpha": a,
            "rho_pred": factors.max(),
            "factor_lambda_min": factors[0], "factor_lambda_max": factors[1],
            "measured_last50": (g[-1] / g[-1 - n]) ** (1.0 / n),
            "converged": converged, "iterations": k if converged else "—",
            "stopped_by": "tolerance" if converged else "blow-up guard",
        })
    save("s1_stability_q1.csv", rows)
    return rows


def project_problem():
    try:  # adapt the names if project.py exports different ones
        from src.problems.project import f_project, grad_project, hess_project, X0_PROJECT
    except ImportError:
        print("project.py not found: project-block rows skipped")
        return None
    return f_project, grad_project, hess_project, X0_PROJECT


def problems():
    out = [
        ("R1", f_rosenbrock, grad_rosenbrock, hess_rosenbrock, np.array([-1.2, 1.0]), dict(tol_abs=1e-6), SOLUTIONS["R1"]),
        ("Q1", f_q1, grad_q1, hess_q1, X0_Q1, dict(tol_rel=1e-6), np.zeros(2)),
    ]
    p = project_problem()
    if p is not None:
        f, g, h, x0 = p
        out.append(("Project", f, g, h, x0, dict(tol_rel=1e-6), None))
    return out


def backtracking_stats():
    rows, sens = [], []
    for name, f, g, h, x0, tol, xstar in problems():
        x, hist, k, acc, halv = gradient_descent_backtracking(f, g, x0, return_steps=True, **tol)
        xs = xstar if xstar is not None else x       # project: minimizer found by the run itself
        lmax = np.linalg.eigvalsh(h(xs)).max()
        rows.append({"problem": name, "iterations": k, "mean_alpha": acc.mean(),
                     "mean_halvings": halv.mean(), "last_alpha": acc[-1],
                     "inv_lambda_max_at_solution": 1.0 / lmax})
        for a0 in (1.0, 1.0 / 64, 16.0, 64.0):
            _, _, k0, a0s, h0s = gradient_descent_backtracking(f, g, x0, alpha0=a0, return_steps=True, **tol)
            sens.append({"problem": name, "alpha0": a0, "iterations": k0,
                         "mean_alpha": a0s.mean(), "mean_halvings": h0s.mean()})
    save("s1_backtracking.csv", rows)
    save("s1_alpha0_sensitivity.csv", sens)
    return rows, sens


if __name__ == "__main__":
    for r in stability_q1():
        print(r)
    rows, sens = backtracking_stats()
    for r in rows + sens:
        print(r)
