"""Checks of the SA-I report: improving moves from a final solution (Table 2, question 4).

A move is improving if the new solution passes the checker and its cost is strictly smaller.
Both types are counted, whatever the method uses:
  transfer: VM k goes to another server;
  swap:     VMs i and j on different servers exchange their servers.
Every candidate is evaluated with the checker of the generator (evaluate_B), not with the
incremental formula of sa.py, so the count is independent of the code under test.
"""
import checkpoint4_generator_PUBLIC as gen

IMPROVE_TOL = 1e-12      # "strictly smaller": the cost must fall by more than rounding noise


def improving_move_list(d, assign):
    """All improving moves from the feasible solution assign:
    ([(k, from, to, delta)] for transfers, [(i, j, server_i, server_j, delta)] for swaps)."""
    assign = [int(a) for a in assign]
    cost0, problems = gen.evaluate_B(d, assign)
    if problems:
        raise ValueError('the solution must be feasible: %s' % problems[:2])
    M, S = d['M'], d['S']
    transfers, swaps = [], []
    for k in range(M):
        for t in range(S):
            if t == assign[k]:
                continue
            cand = list(assign); cand[k] = t
            c, pr = gen.evaluate_B(d, cand)
            if not pr and c < cost0 - IMPROVE_TOL:
                transfers.append((k, assign[k], t, c - cost0))
    for i in range(M):
        for j in range(i + 1, M):
            if assign[i] == assign[j]:
                continue
            cand = list(assign); cand[i], cand[j] = assign[j], assign[i]
            c, pr = gen.evaluate_B(d, cand)
            if not pr and c < cost0 - IMPROVE_TOL:
                swaps.append((i, j, assign[i], assign[j], c - cost0))
    return transfers, swaps


def improving_moves(d, assign):
    """Return (n_transfer, n_swap, best_transfer_delta, best_swap_delta) from the solution assign."""
    transfers, swaps = improving_move_list(d, assign)
    best_tr = min((m[-1] for m in transfers), default=0.0)
    best_sw = min((m[-1] for m in swaps), default=0.0)
    return len(transfers), len(swaps), best_tr, best_sw
