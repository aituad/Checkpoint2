"""
M4 Sensitivity Analysis
Cloud Resource Allocation (Storyline B3)

Perturbs the RHS of one B1 assignment constraint and compares
finite-difference slopes with the dual price.
"""

import os
import numpy as np
import pandas as pd

from src.lp.solve import solve_lp
from src.model.cloud import build_matrices


def run_perturbations(assign_index=9):
    """
    assign_index:
        index of the equality constraint corresponding
        to VM k*=9 assignment constraint.
    """

    mm = build_matrices()

    senses = np.asarray(mm.senses)

    eq_mask = senses == "="
    ub_mask = senses == "<="

    A_eq = mm.A[eq_mask]
    b_eq = mm.b[eq_mask]

    A_ub = mm.A[ub_mask]
    b_ub = mm.b[ub_mask]

    bounds = list(zip(mm.lower, mm.upper))

    rhs_values = [0.99, 1.00, 1.01]

    results = []

    for rhs in rhs_values:

        b_eq_new = b_eq.copy()
        b_eq_new[assign_index] = rhs

        lp = solve_lp(
            c=mm.c,
            A_eq=A_eq,
            b_eq=b_eq_new,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds
        )

        dual_price = (
            None
            if lp.dual_eq is None
            else float(lp.dual_eq[assign_index])
        )

        results.append(
            {
                "rhs": rhs,
                "objective": float(lp.fun),
                "dual_price": dual_price,
            }
        )

    df = pd.DataFrame(results)

    # -----------------------------
    # Compute slopes
    # -----------------------------

    z_minus = float(
        df.loc[df["rhs"] == 0.99, "objective"].iloc[0]
    )

    z0 = float(
        df.loc[df["rhs"] == 1.00, "objective"].iloc[0]
    )

    z_plus = float(
        df.loc[df["rhs"] == 1.01, "objective"].iloc[0]
    )

    pi = float(
        df.loc[df["rhs"] == 1.00, "dual_price"].iloc[0]
    )

    left_slope = (z0 - z_minus) / 0.01
    right_slope = (z_plus - z0) / 0.01

    condition_holds = (
        left_slope <= pi <= right_slope
        or
        right_slope <= pi <= left_slope
    )

    table3 = pd.DataFrame(
        [{
            "left_slope": left_slope,
            "dual_price": pi,
            "right_slope": right_slope,
            "condition_holds": condition_holds,
        }]
    )

    # -----------------------------
    # Save results
    # -----------------------------

    os.makedirs("results", exist_ok=True)

    perturbation_file = "results/perturbation_runs.csv"
    table3_file = "results/m4_table3.csv"

    df.to_csv(perturbation_file, index=False)
    table3.to_csv(table3_file, index=False)

    # -----------------------------
    # Console output
    # -----------------------------

    print("\n=== M4 Perturbation Results ===")
    print(df)

    print("\n=== Table 3 ===")
    print(table3)

    print("\nDetailed Values")
    print(f"z(0.99)      = {z_minus:.6f}")
    print(f"z(1.00)      = {z0:.6f}")
    print(f"z(1.01)      = {z_plus:.6f}")
    print(f"Dual Price π = {pi:.6f}")
    print(f"Left Slope   = {left_slope:.6f}")
    print(f"Right Slope  = {right_slope:.6f}")

    print(f"\nSaved: {perturbation_file}")
    print(f"Saved: {table3_file}")

    return df, table3


if __name__ == "__main__":
    run_perturbations()
