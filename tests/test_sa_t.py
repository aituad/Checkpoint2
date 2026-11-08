import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import checkpoint4_generator_PUBLIC as gen
from src.meta.sa import solve


def test_same_seed_same_result():
    """The same seed should always produce the same result."""
    d = gen.cp34_instance(16, 'B', 3)

    r1 = solve(d, 3, 0, 1000)
    r2 = solve(d, 3, 0, 1000)

    assert r1["solution"] == r2["solution"]
    assert abs(r1["cost"] - r2["cost"]) < 1e-9


def test_budget_used_exactly():
    """The algorithm must use exactly the given evaluation budget."""
    d = gen.cp34_instance(16, 'B', 3)

    r = solve(d, 3, 0, 1000)

    assert r["evals"] == 1000


def test_feasible_solution():
    """The returned solution should be feasible."""
    d = gen.cp34_instance(16, 'B', 3)

    r = solve(d, 3, 0, 1000)

    assert r["feasible"] is True


def test_solution_length():
    """There must be one server assignment for every VM."""
    d = gen.cp34_instance(16, 'B', 3)

    r = solve(d, 3, 0, 1000)

    assert len(r["solution"]) == d["M"]


def test_trace_monotonic():
    """Best-so-far cost in the trace should never increase."""
    d = gen.cp34_instance(16, 'B', 3)

    r = solve(d, 3, 0, 1000)

    trace = r["trace"]

    for i in range(1, len(trace)):
        assert trace[i][1] <= trace[i - 1][1] + 1e-9
