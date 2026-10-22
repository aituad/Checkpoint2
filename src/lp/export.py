"""M2 -> M3/M4: full solution vectors of the LP in the row/column order of M1's model.

pi[i]  dual price of row i of mm (pi_i = d z_LP / d b_i, scipy/HiGHS marginals, no sign change)
r[j]   reduced cost r_j = c_j - sum_i pi_i A_ij, computed from the definition of Task 3
x      primal solution (all 378 values, y_kl first, then t_l), same order as mm.columns
"""
import numpy as np


def solution_vectors(mm, res, ub_idx, eq_idx):
    """Return (pi, r) in the row order of mm (M1). ub_idx / eq_idx come from adapter.to_linprog."""
    pi = np.zeros(len(mm.b))
    pi[eq_idx] = res.dual_eq
    pi[ub_idx] = res.dual_ub
    r = mm.c - mm.A.T @ pi
    return pi, r


def strong_duality_gap(mm, x, pi, r):
    """c^T x - ( b^T pi + sum_j min(0, r_j) * upper_j ) over variables with a finite upper bound.
    Zero (up to rounding) at an LP optimum; fixed variables (upper = 0) contribute nothing."""
    finite = np.isfinite(mm.upper)
    return float(mm.c @ x - (mm.b @ pi + np.sum(np.minimum(0.0, r[finite]) * mm.upper[finite])))


def save_solution(path, mm, res, pi, r):
    """Write the arrays that M3's checker reads (results/m2_solution.npz)."""
    np.savez(path, x=res.x, pi=pi, r=r, z_lp=res.fun,
             dual_lower=res.dual_lower, dual_upper=res.dual_upper,
             rows=np.array(mm.rows), columns=np.array(mm.columns))
