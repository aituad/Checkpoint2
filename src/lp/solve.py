"""M2: solving an LP in the matrix form of Checkpoint 3 and reading the solution.

The solver takes the model in the form

    min c^T x   s.t.   A_ub x <= b_ub,   A_eq x = b_eq,   lb <= x <= ub

(the same form M1 builds in src/model/) and returns the primal point, the optimal
value, the wall-clock time and the raw HiGHS marginals. The marginals are returned
exactly as scipy gives them: M3 (src/kkt/) and M4 (src/sens/) decide about sign
conventions, this module does not interpret them.
"""
from dataclasses import dataclass, field
import time

import numpy as np
from scipy.optimize import linprog

SOLVER_NAME = "scipy.optimize.linprog(method='highs')"


@dataclass
class LPResult:
    status: int                  # 0 = optimal (scipy convention)
    message: str
    x: np.ndarray                # primal solution
    fun: float                   # optimal value z_LP
    seconds: float               # wall-clock time of the solver call only
    solver: str = SOLVER_NAME
    dual_ub: np.ndarray = field(default=None)    # marginals of the <= rows
    dual_eq: np.ndarray = field(default=None)    # marginals of the = rows
    dual_lower: np.ndarray = field(default=None)  # marginals of lower bounds
    dual_upper: np.ndarray = field(default=None)  # marginals of upper bounds

    @property
    def optimal(self):
        return self.status == 0


def solve_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """Solve the LP with HiGHS. Raises RuntimeError if it is not solved to optimality."""
    t0 = time.perf_counter()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    seconds = time.perf_counter() - t0
    if res.status != 0:
        raise RuntimeError(f"LP not solved to optimality: status={res.status}, {res.message}")
    return LPResult(
        status=res.status,
        message=res.message,
        x=np.asarray(res.x, dtype=float),
        fun=float(res.fun),
        seconds=seconds,
        dual_ub=None if A_ub is None else np.asarray(res.ineqlin.marginals),
        dual_eq=None if A_eq is None else np.asarray(res.eqlin.marginals),
        dual_lower=np.asarray(res.lower.marginals),
        dual_upper=np.asarray(res.upper.marginals),
    )


def fractional_report(y, tol=1e-6):
    """Describe the fractional part of the assignment matrix y (shape M x S, VM x server).

    A variable y_kl is fractional if tol < y_kl < 1 - tol. A VM is split if it has
    positive load on at least two servers (equivalently, all its y_kl are < 1 - tol,
    because row k sums to 1).
    """
    y = np.asarray(y, dtype=float)
    frac = (y > tol) & (y < 1 - tol)
    split_vms = np.where((y > tol).sum(axis=1) >= 2)[0]
    return {
        "n_fractional_vars": int(frac.sum()),
        "n_split_vms": int(len(split_vms)),
        "split_vms": [int(k) for k in split_vms],
        "servers_per_split_vm": {int(k): int((y[k] > tol).sum()) for k in split_vms},
        "max_servers_per_vm": int((y > tol).sum(axis=1).max()),
    }
