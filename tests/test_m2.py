"""Tests for the LP owner (M2): triangle problem (hand task H2) and the T16 reference instance.

The triangle problem is data in this test, not part of the reference model:
    min 3 y1 + 5 y2 + 6 y3   s.t.  y1+y2 >= 1,  y2+y3 >= 1,  y1+y3 >= 1,  0 <= y <= 1.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from checkpoint3_generator_PUBLIC import cp34_instance
from src.lp.adapter import to_linprog
from src.lp.reference_b import build_b
from src.lp.solve import solve_lp, fractional_report
from src.model.cloud import build_matrices, build_pulp_model

# ---------------------------------------------------------------- triangle problem (H2)
C = np.array([3.0, 5.0, 6.0])
A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=float)   # rows (T1), (T2), (T3)
B = np.ones(3)


def triangle_cost(y):
    return float(C @ np.asarray(y, dtype=float))


def triangle_feasible(y, tol=1e-9):
    return bool(np.all(A @ np.asarray(y, dtype=float) >= B - tol))


def test_h2a_half_point_feasible_and_cost():
    y = np.array([0.5, 0.5, 0.5])
    assert triangle_feasible(y)
    assert triangle_cost(y) == pytest.approx(7.0, abs=1e-6)


def test_h2a_half_point_is_the_lp_optimum():
    res = solve_lp(C, A_ub=-A, b_ub=-B, bounds=[(0, 1)] * 3)
    assert res.fun == pytest.approx(7.0, abs=1e-6)
    assert np.allclose(res.x, [0.5, 0.5, 0.5], atol=1e-6)


def test_h2a_dual_certificate_of_optimality():
    # pi = (1, 4, 2) is dual feasible (A^T pi <= c) and b^T pi = 7 = cost of (1/2,1/2,1/2)
    pi = np.array([1.0, 4.0, 2.0])
    assert np.all(A.T @ pi <= C + 1e-9)
    assert float(B @ pi) == pytest.approx(7.0, abs=1e-6)


def test_h2b_roundings():
    up, down = np.ones(3), np.zeros(3)
    assert triangle_feasible(up) and triangle_cost(up) == pytest.approx(14.0)
    assert not triangle_feasible(down)
    # mixed roundings (each 1/2 rounded independently): feasible iff at least two ones
    costs = sorted(triangle_cost(y) for y in itertools.product([0, 1], repeat=3)
                   if triangle_feasible(y))
    assert costs == [8.0, 9.0, 11.0, 14.0]


def test_h2c_integer_optimum_gap_and_non_integral_lp_optimum():
    feas = [y for y in itertools.product([0, 1], repeat=3) if triangle_feasible(y)]
    z_star = min(triangle_cost(y) for y in feas)
    z_lp = solve_lp(C, A_ub=-A, b_ub=-B, bounds=[(0, 1)] * 3).fun
    assert z_star == pytest.approx(8.0)
    assert (z_star - z_lp) / z_star == pytest.approx(0.125, abs=1e-6)
    # the LP optimum (1/2,1/2,1/2) is unique and not integral, so no integral point is LP-optimal
    assert all(abs(triangle_cost(y) - z_lp) > 0.5 for y in feas)


# ---------------------------------------------------------------- reference instance T16 (B3)
@pytest.fixture(scope="module")
def t16():
    d = cp34_instance(16, "B", 3)
    mm = build_matrices(d)                               # M1's model
    c, A_ub, b_ub, A_eq, b_eq, bounds, _, _ = to_linprog(mm)
    info = {"n_vars": len(mm.c), "n_rows": len(mm.b), "n_fixed": int(np.sum(mm.upper == 0)),
            "M": int(d["M"]), "S": int(d["S"])}
    res = solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
    return d, (c, A_ub, b_ub, A_eq, b_eq, bounds), info, res


def test_t16_check_values(t16):
    d, _, info, _ = t16
    assert (d["S"], d["M"]) == (9, 41)
    assert (d["cpu_cap"].sum(), d["mem_cap"].sum()) == (420, 1339)
    assert (d["cpu_req"].sum(), d["mem_req"].sum()) == (201, 706)
    assert info["n_fixed"] == 35
    assert info["n_vars"] == 378
    assert info["n_rows"] == 95


def test_t16_lp_is_feasible_and_optimal(t16):
    _, (c, A_ub, b_ub, A_eq, b_eq, bounds), info, res = t16
    assert res.optimal
    assert np.max(A_ub @ res.x - b_ub) <= 1e-6
    assert np.max(np.abs(A_eq @ res.x - b_eq)) <= 1e-6
    lo = np.array([b[0] for b in bounds]); hi = np.array([np.inf if b[1] is None else b[1] for b in bounds])
    assert np.all(res.x >= lo - 1e-6) and np.all(res.x <= hi + 1e-6)
    assert res.fun == pytest.approx(float(c @ res.x), abs=1e-9)


def test_t16_z_lp_value(t16):
    # value pinned from the HiGHS run and cross-checked with CBC in the next test
    assert t16[3].fun == pytest.approx(7.328502, abs=1e-6)


def test_t16_z_lp_agrees_with_cbc(t16):
    pulp = pytest.importorskip("pulp")
    _, (c, A_ub, b_ub, A_eq, b_eq, bounds), _, res = t16
    n = len(c)
    prob = pulp.LpProblem("t16_lp", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{j}", lowBound=bounds[j][0], upBound=bounds[j][1]) for j in range(n)]
    prob += pulp.lpSum(c[j] * x[j] for j in range(n) if c[j] != 0)
    for i in range(A_ub.shape[0]):
        prob += pulp.lpSum(A_ub[i, j] * x[j] for j in np.nonzero(A_ub[i])[0]) <= b_ub[i]
    for i in range(A_eq.shape[0]):
        prob += pulp.lpSum(A_eq[i, j] * x[j] for j in np.nonzero(A_eq[i])[0]) == b_eq[i]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[prob.status] == "Optimal"
    assert pulp.value(prob.objective) == pytest.approx(res.fun, abs=1e-6)


def test_t16_fractional_report_consistent(t16):
    d, _, info, res = t16
    y = res.x[: info["M"] * info["S"]].reshape(info["M"], info["S"])
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-6)            # (B1)
    rep = fractional_report(y)
    assert rep["n_split_vms"] == len(rep["split_vms"]) <= info["M"]
    # a split VM has >= 2 fractional entries, an unsplit VM has none
    assert rep["n_fractional_vars"] == sum(rep["servers_per_split_vm"].values())
    # no GPU VM has weight on a non-GPU server (fixed variables stay at 0)
    forb = (d["needs_gpu"][:, None] == 1) & (d["is_gpu_server"][None, :] == 0)
    assert np.max(y[forb]) <= 1e-9


def test_fractional_report_small_example():
    y = np.array([[1.0, 0.0, 0.0], [0.25, 0.75, 0.0], [0.0, 0.2, 0.8], [0.0, 0.0, 1.0]])
    rep = fractional_report(y)
    assert rep["n_fractional_vars"] == 4
    assert rep["split_vms"] == [1, 2]
    assert rep["max_servers_per_vm"] == 2


def test_m1_model_equals_independent_reference_build(t16):
    """M1's matrices (src/model) and M2's independent build (src/lp/reference_b) coincide."""
    d, (c, A_ub, b_ub, A_eq, b_eq, bounds), _, res = t16
    c2, A_ub2, b_ub2, A_eq2, b_eq2, bounds2, _ = build_b(d)
    assert np.array_equal(c, c2)
    assert np.array_equal(A_ub, A_ub2) and np.array_equal(b_ub, b_ub2)
    assert np.array_equal(A_eq, A_eq2) and np.array_equal(b_eq, b_eq2)
    assert bounds == bounds2
    res2 = solve_lp(c2, A_ub2, b_ub2, A_eq2, b_eq2, bounds2)
    assert res.fun == pytest.approx(res2.fun, abs=1e-9)


def test_m1_pulp_adapter_gives_same_z_lp(t16):
    """M1's own PuLP model (build_pulp_model), solved with CBC, gives z_LP of HiGHS."""
    pulp = pytest.importorskip("pulp")
    model, _, _ = build_pulp_model(matrix=build_matrices(t16[0]))
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[model.status] == "Optimal"
    assert pulp.value(model.objective) == pytest.approx(t16[3].fun, abs=1e-6)


def test_adapter_row_split(t16):
    mm = build_matrices(t16[0])
    _, A_ub, _, A_eq, _, _, ub_idx, eq_idx = to_linprog(mm)
    assert A_eq.shape == (41, 378) and A_ub.shape == (54, 378)
    assert [mm.families[i] for i in eq_idx] == ["B1_assignment"] * 41
    assert sorted(set(ub_idx) | set(eq_idx)) == list(range(95))
def test_t16_optimal_cpu_loads_are_unique_and_not_integral(t16):
    """Every LP-optimal point has the same CPU load on every server, and 8 of the 9 loads
    are not integers. An integer solution has integer loads, so none is LP-optimal: z* > z_LP."""
    from scipy.optimize import linprog
    d, (c, A_ub, b_ub, A_eq, b_eq, bounds), info, res = t16
    M, S = info["M"], info["S"]
    A2 = np.vstack([A_ub, c])
    b2 = np.r_[b_ub, res.fun + 1e-9]                 # restrict to the optimal face
    cpu = d["cpu_req"].astype(float)
    loads = []
    for l in range(S):
        g = np.zeros(len(c))
        g[l:M * S:S] = cpu
        lo = linprog(g, A_ub=A2, b_ub=b2, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs").fun
        hi = -linprog(-g, A_ub=A2, b_ub=b2, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs").fun
        assert hi - lo < 1e-5                        # min load == max load on the optimal face
        loads.append(lo)
    assert np.allclose(loads, [35.4, 15.1, 34.2, 8.75, 29.4, 10.75, 8.0, 31.2, 28.2], atol=1e-4)
    assert sum(abs(x - round(x)) > 1e-3 for x in loads) == 8
