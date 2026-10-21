"""Adapter: M1's MatrixModel (src/model/cloud.py) -> arguments of src.lp.solve.solve_lp."""
import numpy as np


def to_linprog(mm):
    """Split M1's rows by sense: '<=' rows go to (A_ub, b_ub), '=' rows to (A_eq, b_eq).

    Row order inside each block follows mm.rows. Returns
    (c, A_ub, b_ub, A_eq, b_eq, bounds, ub_idx, eq_idx), where ub_idx / eq_idx are
    the positions of those rows in mm.rows (needed later to map the duals back).
    """
    senses = np.array(mm.senses)
    unknown = set(senses) - {"<=", "="}
    if unknown:
        raise ValueError(f"unsupported row senses: {unknown}")
    ub_idx = np.flatnonzero(senses == "<=")
    eq_idx = np.flatnonzero(senses == "=")
    bounds = [(float(lo), None if np.isinf(hi) else float(hi))
              for lo, hi in zip(mm.lower, mm.upper)]
    return (mm.c, mm.A[ub_idx], mm.b[ub_idx], mm.A[eq_idx], mm.b[eq_idx],
            bounds, ub_idx, eq_idx)
