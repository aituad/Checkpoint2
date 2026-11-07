"""Role test of SA-I (SA, transfer), storyline B.

Run from the repository root:  python3 -m pytest tests/test_sa_i.py -v

Guards (the four required checks are 1-4):
 1. the incremental cost equals the checker cost on 1000 random moves (up to 1e-9)
 2. the same seed gives the same run
 3. exactly the budget is spent
 4. the best-so-far trace never increases
 5. interface: cost and feasible flag are the checker's, trace spacing, start argument
 6. other instances / variants / small budgets (hidden instances), infeasible instance is reported as infeasible
 7. only random.Random(seed): the global random state is neither used nor changed
 8. the improving-move counter of Table 2 against a hand example
"""
import copy
import random

import numpy as np
import pytest

import checkpoint4_generator_PUBLIC as gen
from src.meta.interface import check
from src.meta.sa import State, solve, make_start
from src.meta.sa_checks import improving_moves

D16 = gen.cp34_instance(16, 'B', 3)


def _random_assign(d, rng):
    """Random assignment that respects the GPU / zone conditions (capacities may be violated)."""
    st = State(d, [0] * d['M'])
    return [rng.choice(st.allowed[k]) for k in range(d['M'])]


# ------------------------------------------------------------------ 1. incremental cost
@pytest.mark.parametrize('seed,variant', [(16, 3), (5, 3), (7, 1), (9, 2)])
def test_incremental_cost_equals_checker_on_1000_random_moves(seed, variant):
    d = gen.cp34_instance(seed, 'B', variant)
    rng = random.Random(1)
    state = State(d, _random_assign(d, rng))
    movable = [k for k in range(d['M']) if len(state.allowed[k]) >= 2]
    worst = 0.0
    for _ in range(1000):
        k = rng.choice(movable)
        t = rng.choice([s for s in state.allowed[k] if s != state.assign[k]])
        before = state.cost
        delta = state.delta(k, t)                       # the evaluation of the candidate
        fits = state.fits(k, t)
        state.apply(k, t)
        cost_checker, problems = gen.evaluate_B(d, state.assign)
        assert abs(state.cost - cost_checker) <= 1e-9
        assert abs((before + delta) - cost_checker) <= 1e-9          # delta alone is correct, too
        worst = max(worst, abs(state.cost - cost_checker))
        if state.violation <= 1e-12:                     # no overload: the checker must agree
            assert not any('CPU' in p or 'memory' in p for p in problems)
    assert worst <= 1e-9


def test_fits_matches_checker_for_a_feasible_state():
    d = D16
    rng = random.Random(2)
    state = State(d, make_start(d, rng))
    assert gen.evaluate_B(d, state.assign)[1] == []
    for _ in range(500):
        k = rng.randrange(d['M'])
        opts = [s for s in state.allowed[k] if s != state.assign[k]]
        if not opts:
            continue
        t = rng.choice(opts)
        cand = list(state.assign); cand[k] = t
        assert state.fits(k, t) == (gen.evaluate_B(d, cand)[1] == [])


# ------------------------------------------------------------------ 2. same seed, same run
def test_same_seed_same_run_and_other_seed_differs():
    a = solve(D16, 3, 3, 20000)
    b = solve(D16, 3, 3, 20000)
    assert a['solution'] == b['solution'] and a['cost'] == b['cost'] and a['trace'] == b['trace']
    assert a['stats'] == b['stats']
    c = solve(D16, 3, 4, 20000)
    assert c['trace'] != a['trace'] or c['solution'] != a['solution']


# ------------------------------------------------------------------ 3. exactly the budget
@pytest.mark.parametrize('budget', [1, 2, 3, 10, 57, 999, 1000, 1001, 4321, 20000])
def test_budget_spent_exactly(budget):
    r = solve(D16, 3, 0, budget)
    assert r['evals'] == budget
    r = solve(D16, 3, 0, budget, params={'T0': 1.0, 'Tend': 0.01})
    assert r['evals'] == budget


def test_budget_100000_final_setting():
    r = solve(D16, 3, 0, 100000)
    assert r['evals'] == 100000 and r['feasible']


# ------------------------------------------------------------------ 4. trace never increases
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_trace_never_increases_and_is_dense_enough(seed):
    budget = 25000
    r = solve(D16, 3, seed, budget)
    tr = r['trace']
    evs = [e for e, _ in tr]
    vals = [v for _, v in tr]
    assert all(vals[i + 1] <= vals[i] for i in range(len(vals) - 1))
    assert evs == sorted(evs) and evs[-1] == budget
    assert all(evs[i + 1] - evs[i] <= 1000 for i in range(len(evs) - 1))
    assert vals[-1] == pytest.approx(r['cost'], abs=1e-9)          # the trace ends at the returned solution


# ------------------------------------------------------------------ 5. interface
def test_cost_and_feasible_flag_are_the_checkers():
    r = solve(D16, 3, 5, 10000)
    cost, problems = check(D16, 3, r['solution'])
    assert r['cost'] == pytest.approx(cost, abs=1e-12)
    assert r['feasible'] == (problems == [])
    assert len(r['solution']) == D16['M']
    assert all(isinstance(x, int) for x in r['solution'])
    assert set(r) >= {'solution', 'cost', 'evals', 'feasible', 'trace'}


def test_params_are_used_and_start_is_accepted():
    cold = solve(D16, 3, 1, 5000, params={'T0': 1e-4, 'Tend': 1e-5})
    hot = solve(D16, 3, 1, 5000, params={'T0': 5.0, 'Tend': 4.0})
    assert cold['stats']['T0'] == 1e-4 and hot['stats']['T0'] == 5.0
    assert cold['stats']['walk'] == 0 and cold['stats']['K'] == 4999    # no calibration when both are given
    st = make_start(D16, random.Random(0))
    r = solve(D16, 3, 1, 3000, start=st)                                # start is accepted and used
    assert r['evals'] == 3000 and r['feasible']
    assert r['trace'][0][1] == pytest.approx(gen.evaluate_B(D16, st)[0], abs=1e-9)


def test_budget_one_returns_the_start():
    st = make_start(D16, random.Random(0))
    r = solve(D16, 3, 0, 1, start=st)
    assert r['solution'] == st and r['evals'] == 1


def test_overloading_candidates_are_rejected_but_counted():
    """Constraint handling: feasible-only. Candidates that overload the receiving server are
    rejected (counted as evaluations, stats['infeasible_proposals'] > 0) and every state of the
    run stays feasible, so every trace value is a cost the checker accepts."""
    r = solve(D16, 3, 0, 20000)
    st = r['stats']
    assert st['mode'] == 'feasible-only'
    assert st['infeasible_proposals'] > 0
    assert st['accepted'] <= st['K'] - st['infeasible_proposals']
    assert r['feasible']


# ------------------------------------------------------------------ 6. other instances, tight cases
@pytest.mark.parametrize('seed,variant', [(3, 3), (11, 3), (21, 1), (31, 2), (41, 3)])
def test_hidden_instances_small_budget(seed, variant):
    d = gen.cp34_instance(seed, 'B', variant)
    r = solve(d, variant, 0, 3000)
    cost, problems = gen.evaluate_B(d, r['solution'])
    assert r['evals'] == 3000
    assert r['cost'] == pytest.approx(cost, abs=1e-9)
    assert r['feasible'] == (problems == [])
    if r['feasible']:
        assert r['cost'] <= gen.evaluate_B(d, make_start(d, random.Random(0)))[0] + 1e-9   # never worse than its start


def test_infeasible_instance_is_reported_not_hidden():
    d = copy.deepcopy(D16)
    d['cpu_cap'] = np.maximum(1, d['cpu_cap'] // 4)               # 105 CPU for 201 requested: no solution exists
    r = solve(d, 3, 0, 4000)
    assert r['evals'] == 4000
    assert r['feasible'] is False
    assert check(d, 3, r['solution'])[1] != []


def test_gpu_condition_is_kept():
    r = solve(D16, 3, 2, 20000)
    for k in range(D16['M']):
        if D16['needs_gpu'][k] == 1:
            assert D16['is_gpu_server'][r['solution'][k]] == 1


# ------------------------------------------------------------------ 7. randomness
def test_global_random_state_untouched_and_not_used():
    random.seed(123); np.random.seed(123)
    py_state, np_state = random.getstate(), np.random.get_state()
    a = solve(D16, 3, 7, 8000)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    random.seed(999); np.random.seed(999)
    b = solve(D16, 3, 7, 8000)
    assert a['solution'] == b['solution'] and a['trace'] == b['trace']


# ------------------------------------------------------------------ 8. Table 2 helper on a hand example
def test_improving_moves_hand_example():
    """3 servers with CPU capacity 10 each (memory never binds); 4 VMs with CPU 4, 4, 2, 2.
    latency(u) = max of the tangents of u/(1-u) at 0, 0.4, 0.7, 0.9.
    Start (0,0,1,1): loads (8,4,0), cost lat(0.8)+lat(0.4)+lat(0) = 3.444+0.667+0 = 4.111.
      Transfers (8 candidates): VM0 or VM1 to server 1 gives loads (4,8,0): same cost, not strictly better;
      VM0 or VM1 to server 2 gives (4,4,4): cost 2.000, better; VM2 or VM3 to server 0 gives (10,2,0): worse;
      VM2 or VM3 to server 2 gives (8,2,2): 3.844, better.  -> 4 improving transfers.
      Swaps (4 pairs on different servers: VM0/VM1 with VM2/VM3): each gives loads (6,6,0): 2*1.222 = 2.444, better.
      -> 4 improving swaps.
    Start (0,1,2,2): loads (4,4,4) is the balanced optimum: no transfer and no swap improves (swapping the two
    4-CPU VMs would keep the loads, delta = 0, which is not strictly smaller)."""
    d = {'S': 3, 'M': 4, 'story': 'B', 'variant': 1,
         'cpu_cap': np.array([10, 10, 10]), 'mem_cap': np.array([100, 100, 100]),
         'cpu_req': np.array([4, 4, 2, 2]), 'mem_req': np.array([1, 1, 1, 1])}
    n_tr, n_sw, best_tr, best_sw = improving_moves(d, [0, 0, 1, 1])
    assert (n_tr, n_sw) == (4, 4)
    assert best_tr == pytest.approx(2.0 - 4.1111111, abs=1e-5)       # the best transfer: loads (4,4,4)
    assert best_sw == pytest.approx(2.4444444 - 4.1111111, abs=1e-5)   # swaps: loads (6,6,0)
    assert improving_moves(d, [0, 1, 2, 2])[:2] == (0, 0)


def test_improving_moves_rejects_infeasible_start():
    with pytest.raises(ValueError):
        improving_moves(D16, [0] * D16['M'])
