"""Independent build of the Checkpoint 3 reference model, storyline B (B1, B2, B3).

Written by M2 only to cross-check the LP value and the model sizes against the
check values of the task file. The model proper (with the row/column bookkeeping
used by the team) is M1's src/model/; both must give the same sizes and z_LP.

Variable order: y_kl at index k*S + l (k = VM, l = server), then t_l at M*S + l.
Row order: (B1) k = 0..M-1, (B2) l, (B3) l, (B4) for l, then m = 1..4.
"""
import numpy as np

from checkpoint3_generator_PUBLIC import TANGENT_POINTS


def forbidden_pairs(d):
    """Boolean M x S mask of pairs (k, l) with y_kl fixed to 0 (variants B2, B3)."""
    M, S = d["M"], d["S"]
    forb = np.zeros((M, S), dtype=bool)
    if d["variant"] == 2:
        req = d["vm_required_zone"][:, None]
        forb = (req != -1) & (d["zone"][None, :] != req)
    elif d["variant"] == 3:
        forb = (d["needs_gpu"][:, None] == 1) & (d["is_gpu_server"][None, :] == 0)
    return forb


def build_b(d):
    """Return (c, A_ub, b_ub, A_eq, b_eq, bounds, info) of the reference model."""
    S, M = d["S"], d["M"]
    cpu, mem = d["cpu_req"].astype(float), d["mem_req"].astype(float)
    C, R = d["cpu_cap"].astype(float), d["mem_cap"].astype(float)
    u = np.array(TANGENT_POINTS)
    a = 1.0 / (1.0 - u) ** 2
    b = -(u ** 2) / (1.0 - u) ** 2
    ny, nv = M * S, M * S + S

    c = np.zeros(nv)
    c[ny:] = 1.0                                   # minimise sum_l t_l

    A_eq = np.zeros((M, nv))                       # (B1)
    for k in range(M):
        A_eq[k, k * S:(k + 1) * S] = 1.0
    b_eq = np.ones(M)

    rows, rhs = [], []
    for l in range(S):                             # (B2) CPU capacity
        r = np.zeros(nv); r[l:ny:S] = cpu; rows.append(r); rhs.append(C[l])
    for l in range(S):                             # (B3) memory capacity
        r = np.zeros(nv); r[l:ny:S] = mem; rows.append(r); rhs.append(R[l])
    for l in range(S):                             # (B4) latency tangents
        for m in range(len(u)):
            r = np.zeros(nv)
            r[l:ny:S] = a[m] * cpu / C[l]
            r[ny + l] = -1.0
            rows.append(r); rhs.append(-b[m])
    A_ub, b_ub = np.array(rows), np.array(rhs)

    forb = forbidden_pairs(d)
    bounds = [(0.0, 0.0 if forb[k, l] else 1.0) for k in range(M) for l in range(S)]
    bounds += [(0.0, None)] * S
    info = {"n_vars": nv, "n_rows": A_eq.shape[0] + A_ub.shape[0],
            "n_fixed": int(forb.sum()), "M": M, "S": S}
    return c, A_ub, b_ub, A_eq, b_eq, bounds, info
