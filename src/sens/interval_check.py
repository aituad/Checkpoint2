# src/sens/interval_check.py

import os
import pandas as pd


def interval_check(
    input_file="results/perturbation_runs.csv",
    output_file="results/interval_check.csv"
):
    """
    Compare the dual price with finite-difference slopes.

    left_slope  = (z(1.00) - z(0.99)) / 0.01
    right_slope = (z(1.01) - z(1.00)) / 0.01
    """

    df = pd.read_csv(input_file)

    z99 = df.loc[df["rhs"] == 0.99, "objective"].iloc[0]
    z100 = df.loc[df["rhs"] == 1.00, "objective"].iloc[0]
    z101 = df.loc[df["rhs"] == 1.01, "objective"].iloc[0]

    dual_price = df.loc[
        df["rhs"] == 1.00,
        "dual_price"
    ].iloc[0]

    left_slope = (z100 - z99) / 0.01
    right_slope = (z101 - z100) / 0.01

    result = pd.DataFrame([{
        "left_slope": left_slope,
        "dual_price": dual_price,
        "right_slope": right_slope,
        "left_minus_dual": left_slope - dual_price,
        "right_minus_dual": right_slope - dual_price
    }])

    os.makedirs("results", exist_ok=True)

    result.to_csv(
        output_file,
        index=False
    )

    print("\nInterval Check Results")
    print(result)

    print(f"\nSaved: {output_file}")

    return result


if __name__ == "__main__":
    interval_check()
