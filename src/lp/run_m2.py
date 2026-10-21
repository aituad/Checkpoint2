"""M2: solve the LP relaxation of the T16 instance and write the numbers of Table 1 (Task 2).

    python3 -m src.lp.run_m2

Writes results/m2_lp.csv and results/m2_fractional.csv. The model is M1's
(src/model/cloud.py); src/lp/reference_b.py is M2's independent build, used only
in tests/test_m2.py to check that both give identical matrices.
"""
import csv
from pathlib import Path

import numpy as np

from checkpoint3_generator_PUBLIC import cp34_instance
from src.lp.adapter import to_linprog
from src.model.cloud import build_matrices
from src.lp.solve import solve_lp, fractional_report

SEED, STORY, VARIANT = 16, "B", 3
RESULTS = Path(__file__).resolve().parents[2] / "results"
TOL = 1e-6


def run():
    d = cp34_instance(SEED, STORY, VARIANT)
    mm = build_matrices(d)
    c, A_ub, b_ub, A_eq, b_eq, bounds, _, _ = to_linprog(mm)
    info = {"n_vars": len(mm.c), "n_rows": len(mm.b),
            "n_fixed": int(np.sum(mm.upper == 0)), "M": int(d["M"]), "S": int(d["S"])}
    res = solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
    x = np.where(np.abs(res.x) < 1e-12, 0.0, res.x)      # remove -0.0 / 1e-17 noise
    y = x[: info["M"] * info["S"]].reshape(info["M"], info["S"])
    t = x[info["M"] * info["S"]:]
    frac = fractional_report(y, TOL)
    return d, info, res, y, t, frac


def main():
    d, info, res, y, t, frac = run()
    RESULTS.mkdir(exist_ok=True)
    rows = [
        ("instance", f"cp34_instance({SEED}, '{STORY}', {VARIANT})"),
        ("n_variables", info["n_vars"]),
        ("n_constraint_rows", info["n_rows"]),
        ("n_fixed_to_zero", info["n_fixed"]),
        ("z_LP", f"{res.fun:.6f}"),
        ("solver", res.solver),
        ("seconds", f"{res.seconds:.4f}"),
        ("n_fractional_y", frac["n_fractional_vars"]),
        ("n_split_vms", frac["n_split_vms"]),
        ("max_servers_per_vm", frac["max_servers_per_vm"]),
    ]
    with open(RESULTS / "m2_lp.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["quantity", "value"]); w.writerows(rows)
    with open(RESULTS / "m2_fractional.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vm", "cpu_req", "needs_gpu", "n_servers"] +
                   [f"y_server{l}" for l in range(info["S"])])
        for k in frac["split_vms"]:
            w.writerow([k, int(d["cpu_req"][k]), int(d["needs_gpu"][k]),
                        frac["servers_per_split_vm"][k]] + [f"{v:.6f}" for v in y[k]])
    for name, val in rows:
        print(f"{name:20s} {val}")
    print("t_l =", np.round(t, 6))
    print("split VMs:", frac["split_vms"])


if __name__ == "__main__":
    main()
