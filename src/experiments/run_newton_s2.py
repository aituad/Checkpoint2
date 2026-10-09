"""S2 experiments using the team's GD, Momentum, Adam and Newton implementations."""
from src.optim.momentum import optimize as optimize_momentum
import csv
from pathlib import Path

import numpy as np

from src.optim.adam import optimize as adam
from src.optim.gd import gradient_descent
from src.optim.newton import pure_newton, damped_newton
from src.problems.quadratic import quadratic


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TOL = 1e-6
MAX_ITER = 100000


def save_csv(name, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / name).open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def converged(grad, x, x0, relative):
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e12:
        return False
    g = np.asarray(grad(x), dtype=float)
    threshold = TOL * np.linalg.norm(grad(x0)) if relative else TOL
    return bool(
        np.all(np.isfinite(g))
        and np.linalg.norm(g) < threshold
    )


def tune_adam(f, grad, x0, c):
    """Choose the fewest updates among genuinely converged runs."""
    candidates = []
    runs = []

    def evaluate(alpha):
        with np.errstate(over="ignore", invalid="ignore"):
            x, _, k = adam(
                f, grad, x0,
                alpha=float(alpha),
                tol_rel=TOL,
                max_iter=MAX_ITER,
            )
            success = converged(grad, x, x0, relative=True)

        runs.append({
            "c": c,
            "method": "Adam",
            "alpha": float(alpha),
            "iterations": int(k),
            "converged": success,
        })
        if success:
            candidates.append((int(k), float(alpha)))

    for alpha in np.logspace(-5, 0, 11):
        evaluate(alpha)

    edge = 1.0
    while candidates and min(candidates)[1] == edge and edge < 1e4:
        edge = min(edge * np.sqrt(10.0), 1e4)
        evaluate(edge)

    return min(candidates) if candidates else None, runs


def rosenbrock():
    def f(x):
        u, v = np.asarray(x, dtype=float)
        return float((1 - u) ** 2 + 100 * (v - u**2) ** 2)

    def grad(x):
        u, v = np.asarray(x, dtype=float)
        return np.array([
            2 * (u - 1) - 400 * u * (v - u**2),
            200 * (v - u**2),
        ])

    def hess(x):
        u, v = np.asarray(x, dtype=float)
        return np.array([
            [2 - 400 * v + 1200 * u**2, -400 * u],
            [-400 * u, 200],
        ])

    return f, grad, hess


def project():
    """Team T16 cloud CPU problem, with the last logit fixed to zero."""
    capacities = np.array(
        [59, 46, 57, 35, 49, 43, 32, 52, 47], dtype=float
    )
    demand = 201.0
    a = demand**2 / (len(capacities) * capacities**2)

    def shares(z):
        logits = np.r_[np.asarray(z, dtype=float), 0.0]
        e = np.exp(logits - np.max(logits))
        return e / np.sum(e)

    def f(z):
        w = shares(z)
        return float(np.dot(a, w**2))

    def grad(z):
        w = shares(z)
        value = np.dot(a, w**2)
        return 2 * w[:-1] * (a[:-1] * w[:-1] - value)

    def hess(z):
        w = shares(z)
        value = np.dot(a, w**2)
        B = np.diag(w) - np.outer(w, w)
        H = (
            2 * np.diag(a * w - value) @ B
            - 2 * np.outer(w, B.T @ (2 * a * w))
            + 2 * np.diag(w * a) @ B
        )
        H = H[:-1, :-1]
        return (H + H.T) / 2

    optimal_shares = capacities**2 / np.sum(capacities**2)
    optimal_value = demand**2 / (
        len(capacities) * np.sum(capacities**2)
    )
    return (
        f, grad, hess, np.zeros(len(capacities) - 1),
        shares, optimal_shares, optimal_value,
    )


def tune_momentum(f, grad, x0, c):
    candidates, runs = [], []

    def evaluate(alpha):
        for beta in [0.5, 0.8, 0.9, 0.95, 0.99]:
            with np.errstate(over="ignore", invalid="ignore"):
                x, _, k = optimize_momentum(
                    f, grad, x0, alpha=float(alpha), beta=beta,
                    tol_rel=TOL, max_iter=MAX_ITER,
                )
                success = converged(grad, x, x0, True)
            runs.append({
                "c": c, "method": "Momentum", "alpha": float(alpha),
                "beta": beta, "iterations": int(k), "converged": success,
            })
            if success:
                candidates.append((int(k), float(alpha), beta))

    for alpha in np.logspace(-5, 0, 11):
        evaluate(alpha)
    edge = 1.0
    while candidates and min(candidates)[1] == edge and edge < 1e4:
        edge = min(edge * np.sqrt(10.0), 1e4)
        evaluate(edge)
    return min(candidates) if candidates else None, runs


def run_conditioning():
    rows, adam_grid, momentum_grid = [], [], []
    for c in [10, 100, 1000, 129]:
        f, grad, hess, x0 = quadratic(c)
        alpha = 1.0 / (1.0 + c)
        x, _, k = gradient_descent(
            f, grad, x0, alpha=alpha,
            tol=TOL, relative=True, max_iter=MAX_ITER,
        )
        rows.append({
            "c": c, "method": "GD alpha*", "iterations": int(k),
            "alpha": alpha, "beta": "",
            "status": "converged" if converged(grad, x, x0, True) else "failed",
        })
        best, trials = tune_adam(f, grad, x0, c)
        adam_grid.extend(trials)
        rows.append({
            "c": c, "method": "Adam", "iterations": best[0] if best else "",
            "alpha": best[1] if best else "", "beta": "",
            "status": "converged" if best else "failed",
        })
        print(f"Conditioning c={c}: tuning Momentum...", flush=True)
        best, trials = tune_momentum(f, grad, x0, c)
        momentum_grid.extend(trials)
        rows.append({
            "c": c, "method": "Momentum", "iterations": best[0] if best else "",
            "alpha": best[1] if best else "", "beta": best[2] if best else "",
            "status": "converged" if best else "failed",
        })
        if best:
            print(f"Momentum c={c}: k={best[0]}, alpha={best[1]:.10g}, beta={best[2]}", flush=True)
        else:
            print(f"Momentum c={c}: no converged run in the tested grid", flush=True)
        for name, solver in [
            ("Newton pure", pure_newton), ("Newton damped", damped_newton),
        ]:
            x, _, k = solver(
                f, grad, hess, x0, tol=TOL, relative=True, max_iter=100,
            )
            rows.append({
                "c": c, "method": name, "iterations": int(k),
                "alpha": "", "beta": "",
                "status": "converged" if converged(grad, x, x0, True) else "failed",
            })
        print(f"Conditioning c={c}: completed", flush=True)
    save_csv("s2_conditioning_team.csv", rows)
    save_csv("s2_adam_grid_team.csv", adam_grid)
    save_csv("s2_momentum_grid_team.csv", momentum_grid)
    q1_m = next(r for r in rows if r["c"] == 129 and r["method"] == "Momentum")
    q1_n = next(r for r in rows if r["c"] == 129 and r["method"] == "Newton pure")
    if q1_m["status"] == q1_n["status"] == "converged":
        cutoff = np.sqrt(10 * q1_m["iterations"] / q1_n["iterations"])
        print(f"Q1 cost model: Momentum cheaper for n > {cutoff:.6f}; integer n >= {int(np.floor(cutoff)) + 1}.", flush=True)


def run_safeguards():
    rf, rg, rh = rosenbrock()
    pf, pg, ph, z0, shares, w_star, j_star = project()

    # Check project derivatives before running Newton.
    eps = 1e-5
    basis = np.eye(len(z0))
    numeric_grad = np.array([
        (pf(z0 + eps * e) - pf(z0 - eps * e)) / (2 * eps)
        for e in basis
    ])
    numeric_hess = np.column_stack([
        (pg(z0 + eps * e) - pg(z0 - eps * e)) / (2 * eps)
        for e in basis
    ])
    np.testing.assert_allclose(
        pg(z0), numeric_grad, rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        ph(z0), numeric_hess, rtol=1e-5, atol=1e-7
    )

    rows = []
    histories = []
    problems = [
        ("R2", rf, rg, rh, np.array([-1.51, 2.17]), False),
        ("Project", pf, pg, ph, z0, True),
    ]

    for problem, f, grad, hess, x0, relative in problems:
        for method, solver in [
            ("pure", pure_newton),
            ("damped", damped_newton),
        ]:
            x, hist, k = solver(
                f, grad, hess, x0,
                tol=TOL, relative=relative, max_iter=100,
            )
            values = np.array([f(point) for point in hist])
            eigenvalues = np.linalg.eigvalsh(hess(x))

            rows.append({
                "problem": problem,
                "method": method,
                "iterations": int(k),
                "converged": converged(grad, x, x0, relative),
                "monotone": bool(np.all(np.diff(values) <= 1e-12)),
                "f": float(f(x)),
                "gradient_norm": float(np.linalg.norm(grad(x))),
                "min_eigenvalue": float(eigenvalues.min()),
                "max_eigenvalue": float(eigenvalues.max()),
                "shares_error_inf": (
                    float(np.max(np.abs(shares(x) - w_star)))
                    if problem == "Project" else ""
                ),
                "objective_error": (
                    float(abs(f(x) - j_star))
                    if problem == "Project" else ""
                ),
            })

            for i, point in enumerate(hist):
                histories.append({
                    "problem": problem,
                    "method": method,
                    "k": i,
                    "f": float(f(point)),
                    "gradient_norm": float(
                        np.linalg.norm(grad(point))
                    ),
                })

    save_csv("s2_safeguards_team.csv", rows)
    save_csv("s2_histories_team.csv", histories)


def main():
    run_conditioning()
    run_safeguards()
    print("S2 results saved in results/. Check status in s2_conditioning_team.csv.")


if __name__ == "__main__":
    main()