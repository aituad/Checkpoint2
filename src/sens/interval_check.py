 # src/sens/interval_check.py

def interval_check(df, tolerance=1e-4):

    z099 = float(
        df.loc[df["rhs"] == 0.99, "objective"].iloc[0]
    )

    z100 = float(
        df.loc[df["rhs"] == 1.00, "objective"].iloc[0]
    )

    z101 = float(
        df.loc[df["rhs"] == 1.01, "objective"].iloc[0]
    )

    pi = float(
        df.loc[df["rhs"] == 1.00, "dual_price"].iloc[0]
    )

    left_slope = (z100 - z099) / 0.01

    right_slope = (z101 - z100) / 0.01

    left_error = abs(left_slope - pi)

    right_error = abs(right_slope - pi)

    basis_valid = (
        left_error < tolerance
        and right_error < tolerance
    )

    return {
        "dual_price": pi,
        "left_slope": left_slope,
        "right_slope": right_slope,
        "left_error": left_error,
        "right_error": right_error,
        "basis_valid": basis_valid
    }
