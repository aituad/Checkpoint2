  # src/sens/perturbation.py

import pandas as pd


def run_perturbations(build_lp_model):
    """
    Solve LP for three RHS values:
    0.99, 1.00, 1.01
    """

    rhs_values = [0.99, 1.00, 1.01]
    results = []

    for rhs in rhs_values:

        model, y, target_con = build_lp_model(rhs)

        model.optimize()

        if model.Status != 2:
            raise RuntimeError(
                f"Optimization failed for RHS={rhs}"
            )

        results.append({
            "rhs": rhs,
            "objective": model.ObjVal,
            "dual_price": target_con.Pi
        })

    df = pd.DataFrame(results)

    return df
