import numpy as np

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

    results = {}

    for rhs in [0.99, 1.00, 1.01]:

        b_eq_new = b_eq.copy()
        b_eq_new[assign_index] = rhs

        lp = solve_lp(
            c=mm.c,
            A_eq=A_eq,
            b_eq=b_eq_new,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
        )

        dual_price = (
            None
            if lp.dual_eq is None
            else float(lp.dual_eq[assign_index])
        )

        results[rhs] = {
            "objective": float(lp.fun),
            "dual_price": dual_price,
        }

    z_minus = results[0.99]["objective"]
    z0 = results[1.00]["objective"]
    z_plus = results[1.01]["objective"]

    pi = results[1.00]["dual_price"]

    left_slope = (z0 - z_minus) / 0.01
    right_slope = (z_plus - z0) / 0.01

    print("\n=== M4 Sensitivity Analysis ===")
    print(f"z(0.99) = {z_minus}")
    print(f"z(1.00) = {z0}")
    print(f"z(1.01) = {z_plus}")

    print(f"\nDual price (π) = {pi}")

    print(f"\nLeft slope  = {left_slope}")
    print(f"Right slope = {right_slope}")

    if (
        left_slope <= pi <= right_slope
        or
        right_slope <= pi <= left_slope
    ):
        print("\nSensitivity condition satisfied.")
    else:
        print("\nSensitivity condition NOT satisfied.")


if __name__ == "__main__":
    run_perturbations()
