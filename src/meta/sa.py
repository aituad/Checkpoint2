"""Simulated annealing, configuration "SA, transfer" (role SA-I), storyline B.

State: assign[k] = server of virtual machine k.
Move:  transfer -- one VM goes to another server that it is allowed to use
       (variant 3: a VM with needs_gpu = 1 only to a GPU server; variant 2: only to its zone).
       No swap and no other move is used.
Cost:  sum over servers of latency_B(min(cpu_l / cpu_cap_l, 1)), updated incrementally:
       a transfer changes the loads of two servers only, so the cost of a candidate is
       delta = lat(new_s) + lat(new_t) - lat(old_s) - lat(old_t).
Constraint handling: a candidate that would overload the receiving server (CPU or memory) is
       REJECTED (it is still counted as one evaluation). The search therefore stays inside the
       feasible region if its start is feasible. Only if the start is infeasible (own start
       fails on a very tight instance, or an infeasible `start` is passed in), the run switches
       to a penalty f = cost + RHO * violation until the first feasible state is reached.
Schedule: geometric, T_k = T0 * alpha^k, alpha = (Tend / T0)^(1/K), K = number of annealing
       proposals. T0 and Tend come from params={"T0":..., "Tend":...} (tuner) or, if not
       given, from the rule of the lecture (random walk, see calibrate()).
Budget: one evaluation = one candidate (start, every random-walk proposal, every annealing
       proposal; rejected or not). evals == budget exactly.
Randomness: only random.Random(seed).
"""
import math
import random

import numpy as np

import checkpoint4_generator_PUBLIC as gen
from src.meta.interface import check

RHO = 20.0            # penalty weight, used only when the state is infeasible (relative overload)
WALK = 3000           # proposals of the calibration random walk (lecture rule)
_EPS = 1e-9


# ---------------------------------------------------------------- problem data
def _lines():
    """Tangent lines (a_m, b_m) of g(u) = u/(1-u): latency(u) = max_m (a_m u + b_m)."""
    return [(1.0 / (1.0 - um) ** 2, -um ** 2 / (1.0 - um) ** 2) for um in gen.TANGENT_POINTS]


_LINES = _lines()


def _lat(u):
    if u > 1.0:
        u = 1.0
    best = _LINES[0][0] * u + _LINES[0][1]
    for a, b in _LINES[1:]:
        v = a * u + b
        if v > best:
            best = v
    return best


def allowed_servers(d):
    """allowed[k] = list of servers VM k may use (read from d; nothing is hard-coded)."""
    S, M = d['S'], d['M']
    allowed = []
    for k in range(M):
        servers = []
        for l in range(S):
            if d['variant'] == 2 and d['vm_required_zone'][k] != -1 and d['zone'][l] != d['vm_required_zone'][k]:
                continue
            if d['variant'] == 3 and d['needs_gpu'][k] == 1 and d['is_gpu_server'][l] == 0:
                continue
            servers.append(l)
        allowed.append(servers)
    return allowed


# ---------------------------------------------------------------- state with incremental cost
class State:
    """Assignment with loads, per-server latency and total cost kept up to date."""

    def __init__(self, d, assign):
        self.S, self.M = int(d['S']), int(d['M'])
        self.cpu_req = [int(x) for x in d['cpu_req']]
        self.mem_req = [int(x) for x in d['mem_req']]
        self.cpu_cap = [int(x) for x in d['cpu_cap']]
        self.mem_cap = [int(x) for x in d['mem_cap']]
        self.allowed = allowed_servers(d)
        self.assign = [int(a) for a in assign]
        self.cpu = [0] * self.S
        self.mem = [0] * self.S
        for k, s in enumerate(self.assign):
            self.cpu[s] += self.cpu_req[k]
            self.mem[s] += self.mem_req[k]
        self.lat = [_lat(self.cpu[l] / self.cpu_cap[l]) for l in range(self.S)]
        self.cost = float(sum(self.lat))
        self.viol_s = [self._over(l, self.cpu[l], self.mem[l]) for l in range(self.S)]

    def _over(self, l, cpu, mem):
        """Relative overload of server l (0 if both capacities hold)."""
        return (max(0, cpu - self.cpu_cap[l]) / self.cpu_cap[l]
                + max(0, mem - self.mem_cap[l]) / self.mem_cap[l])

    @property
    def violation(self):
        return sum(self.viol_s)

    def fits(self, k, t):
        """True if server t can take VM k without exceeding CPU or memory (reads loads only)."""
        return (self.cpu[t] + self.cpu_req[k] <= self.cpu_cap[t]
                and self.mem[t] + self.mem_req[k] <= self.mem_cap[t])

    def delta(self, k, t):
        """Cost change of moving VM k to server t (the one evaluation of the candidate)."""
        s = self.assign[k]
        c = self.cpu_req[k]
        return (_lat((self.cpu[s] - c) / self.cpu_cap[s]) + _lat((self.cpu[t] + c) / self.cpu_cap[t])
                - self.lat[s] - self.lat[t])

    def delta_violation(self, k, t):
        s = self.assign[k]
        c, m = self.cpu_req[k], self.mem_req[k]
        return (self._over(s, self.cpu[s] - c, self.mem[s] - m) + self._over(t, self.cpu[t] + c, self.mem[t] + m)
                - self.viol_s[s] - self.viol_s[t])

    def apply(self, k, t):
        s = self.assign[k]
        c, m = self.cpu_req[k], self.mem_req[k]
        self.cpu[s] -= c; self.mem[s] -= m
        self.cpu[t] += c; self.mem[t] += m
        new_s, new_t = _lat(self.cpu[s] / self.cpu_cap[s]), _lat(self.cpu[t] / self.cpu_cap[t])
        self.cost += new_s + new_t - self.lat[s] - self.lat[t]
        self.lat[s], self.lat[t] = new_s, new_t
        self.viol_s[s] = self._over(s, self.cpu[s], self.mem[s])
        self.viol_s[t] = self._over(t, self.cpu[t], self.mem[t])
        self.assign[k] = t


# ---------------------------------------------------------------- start solution
def greedy_start(d, rng=None, noise=0.0):
    """Feasible start if the greedy rule finds one: VMs in order (most restricted first, then
    larger CPU first), each to the allowed server with the smallest CPU utilisation after the
    placement among those where CPU and memory fit. `noise` > 0 perturbs the order (retries)."""
    allowed = allowed_servers(d)
    M, S = d['M'], d['S']
    cpu_req, mem_req = d['cpu_req'], d['mem_req']
    cpu_cap, mem_cap = d['cpu_cap'], d['mem_cap']
    key = [(len(allowed[k]), -float(cpu_req[k]) * (1.0 + noise * (rng.random() if rng else 0.0)),
            -float(mem_req[k])) for k in range(M)]
    order = sorted(range(M), key=lambda k: key[k])
    cpu = [0] * S; mem = [0] * S
    assign = [-1] * M
    ok = True
    for k in order:
        best, best_u = None, None
        for l in allowed[k]:
            if cpu[l] + cpu_req[k] <= cpu_cap[l] and mem[l] + mem_req[k] <= mem_cap[l]:
                u = (cpu[l] + cpu_req[k]) / cpu_cap[l]
                if best is None or u < best_u:
                    best, best_u = l, u
        if best is None:                         # nothing fits: put it on the least loaded allowed server
            ok = False
            best = min(allowed[k], key=lambda l: (cpu[l] + cpu_req[k]) / cpu_cap[l]) if allowed[k] else 0
        assign[k] = best
        cpu[best] += cpu_req[k]; mem[best] += mem_req[k]
    return assign, ok


def make_start(d, rng):
    assign, ok = greedy_start(d)
    tries = 0
    while not ok and tries < 50:
        assign, ok = greedy_start(d, rng, noise=1.0)
        tries += 1
    return assign


# ---------------------------------------------------------------- the annealer
def _random_move(state, rng, movable):
    """Uniform VM among those with at least two allowed servers, then uniform other allowed server."""
    k = movable[rng.randrange(len(movable))]
    opts = state.allowed[k]
    cur = state.assign[k]
    t = opts[rng.randrange(len(opts))]
    while t == cur:
        t = opts[rng.randrange(len(opts))]
    return k, t


def calibrate(state, rng, movable, n, rho, use_penalty):
    """Random walk of n proposals that accepts every proposal that fits (every proposal in penalty
    mode). Returns the list of positive changes f of the objective (cost, or cost + rho * violation)."""
    deltas = []
    for _ in range(n):
        k, t = _random_move(state, rng, movable)
        if use_penalty:
            f = state.delta(k, t) + rho * state.delta_violation(k, t)
            state.apply(k, t)
        elif state.fits(k, t):
            f = state.delta(k, t)
            state.apply(k, t)
        else:
            continue
        if f > 0:
            deltas.append(f)
    return deltas


def solve(d, variant, seed, budget, params=None, start=None):
    """Simulated annealing with transfer moves; see the module docstring and the README.
    Returns the dictionary of the interface (solution, cost, evals, feasible, trace) plus the
    diagnostics 'stats' used by the report (temperatures, acceptance of worsening moves)."""
    if d.get('story') != 'B':
        raise ValueError('sa.py implements storyline B only')
    params = dict(params) if params else {}
    rho = float(params.get('rho', RHO))
    rng = random.Random(seed)
    M = d['M']

    assign0 = [int(a) for a in start] if start is not None else make_start(d, rng)
    state = State(d, assign0)
    movable = [k for k in range(M) if len(state.allowed[k]) >= 2]
    use_penalty = state.violation > _EPS

    evals = 1                                           # the start counts
    best_cost = state.cost if not use_penalty else float('inf')
    best_assign = list(state.assign)                    # best feasible, else least violating
    best_viol = state.violation
    trace = [(1, best_cost)]
    next_mark = 1000

    def note(evals_now):
        nonlocal next_mark
        while evals_now >= next_mark:
            trace.append((next_mark, best_cost)); next_mark += 1000

    def update_best(cur_state):
        nonlocal best_cost, best_assign, best_viol
        v = cur_state.violation
        if v <= _EPS:
            if cur_state.cost < best_cost:
                best_cost, best_assign, best_viol = cur_state.cost, list(cur_state.assign), 0.0
        elif best_cost == float('inf') and v < best_viol:
            best_assign, best_viol = list(cur_state.assign), v

    stats = {'mode': 'penalty' if use_penalty else 'feasible-only', 'T0': None, 'Tend': None, 'alpha': None,
             'K': 0, 'walk': 0, 'delta_bar': None, 'delta_min': None,
             'first5': (0, 0), 'last5': (0, 0), 'infeasible_proposals': 0, 'accepted': 0,
             'curve': [], 'best_at_eval': 1, 'T_at_best': None, 'current_cost_end': state.cost}

    if not movable or budget <= 1:                      # nothing can move: spend the budget on null candidates
        evals = max(1, budget)
        note(evals)
        return _finish(d, variant, best_assign, best_cost, evals, trace, stats)

    # -- temperatures: from params, otherwise from the random-walk rule
    T0, Tend = params.get('T0'), params.get('Tend')
    walk = 0
    if T0 is None or Tend is None:
        walk = int(params.get('walk', min(WALK, max(20, (budget - 1) // 5))))
        walk = min(walk, budget - 1)
        wstate = State(d, state.assign)
        # the walk runs on a copy; its n candidates are evaluations, its states are not used
        deltas = calibrate(wstate, rng, movable, walk, rho, use_penalty)
        evals += walk
        note(evals)
        if len(deltas) >= 5:
            deltas.sort()
            dbar = float(np.median(deltas))
            dmin = float(np.percentile(deltas, 5))
        else:
            dbar, dmin = (max(deltas) if deltas else 1.0), (min(deltas) if deltas else 0.1)
        stats['delta_bar'], stats['delta_min'] = dbar, dmin
        if T0 is None:
            T0 = 4.5 * dbar
        if Tend is None:
            Tend = 0.22 * dmin
    T0, Tend = float(T0), float(Tend)
    if not (T0 > 0 and Tend > 0):
        raise ValueError('T0 and Tend must be positive')
    stats['walk'] = walk
    K = budget - evals                                  # annealing proposals
    alpha = (Tend / T0) ** (1.0 / K) if K > 0 else 1.0
    stats.update(T0=T0, Tend=Tend, alpha=alpha, K=K)

    # -- annealing
    first_n = max(1, math.ceil(0.05 * K))
    last_from = K - first_n                             # proposals with index >= last_from are the last 5%
    nblocks = 50
    block = max(1, K // nblocks)
    blk_w = blk_acc = 0
    f5w = f5a = l5w = l5a = 0
    accepted = infeas = 0
    T = T0
    exp = math.exp
    for i in range(K):
        k, t = _random_move(state, rng, movable)
        evals += 1
        if use_penalty:
            f = state.delta(k, t) + rho * state.delta_violation(k, t)
        elif state.fits(k, t):
            f = state.delta(k, t)
        else:
            infeas += 1
            note(evals); T *= alpha
            if (i + 1) % block == 0 and len(stats['curve']) < nblocks:
                stats['curve'].append((T / alpha, blk_w, blk_acc)); blk_w = blk_acc = 0
            continue
        acc = f <= 0 or rng.random() < exp(-f / T)
        if f > 0:
            blk_w += 1
            if i < first_n:
                f5w += 1; f5a += acc
            if i >= last_from:
                l5w += 1; l5a += acc
        if acc:
            state.apply(k, t)
            accepted += 1
            before = best_cost
            update_best(state)
            if best_cost < before:
                stats['best_at_eval'], stats['T_at_best'] = evals, T
            if f > 0:
                blk_acc += 1
        note(evals)
        T *= alpha
        if (i + 1) % block == 0 and len(stats['curve']) < nblocks:
            stats['curve'].append((T / alpha, blk_w, blk_acc)); blk_w = blk_acc = 0
    stats.update(first5=(f5a, f5w), last5=(l5a, l5w), infeasible_proposals=infeas, accepted=accepted,
                 current_cost_end=state.cost, end_assign=list(state.assign))
    if trace[-1][0] != budget:
        trace.append((budget, best_cost))
    return _finish(d, variant, best_assign, best_cost, evals, trace, stats)


def _finish(d, variant, assign, best_cost, evals, trace, stats):
    cost, problems = check(d, variant, assign)
    if trace[-1][0] != evals:
        trace.append((evals, best_cost))
    return {'solution': [int(a) for a in assign], 'cost': cost, 'evals': evals,
            'feasible': len(problems) == 0, 'trace': trace, 'stats': stats}
