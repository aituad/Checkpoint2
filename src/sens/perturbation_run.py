import os
import numpy as np
import pandas as pd

from src.lp.solve import solve_lp
from src.model.cloud import build_matrices


def run_perturbations(assign_index=9):

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

        results.append({
            "rhs": rhs,
            "objective": lp.fun,
            "dual_price": dual_price
        })

    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)

    output_file = "results/perturbation_runs.csv"

    df.to_csv(output_file, index=False)

    print("\nPerturbation Results")
    print(df)

    print(f"\nSaved: {output_file}")

    return df


if __name__ == "__main__":
    run_perturbations()
