"""Team T16 cloud CPU relaxation in eight unconstrained softmax logits.

Official CPU data come from T16_B3_packet.docx, cloud_variant(16, 3).
Total demand is computed from all 41 VM CPU requests.
The mean squared utilization objective, softmax parametrization, zero start
and relative stopping rule follow T16_Checkpoint2_task.docx.

Fixing the ninth logit to zero removes softmax's translation ambiguity.
F(z) = sum_j (D*w_j(z)/C_j)**2 / N_SERVERS.
This relaxation ignores indivisible VMs, memory/GPU constraints and CPU caps.
"""

import numpy as np


CAPACITIES = np.array([59., 46., 57., 35., 49., 43., 32., 52., 47.])
CPU_REQUESTS = np.array([
    1, 2, 7, 6, 5, 2, 2, 6, 3, 8, 4, 8, 3, 6, 2, 2, 8, 4, 7, 3, 1,
    8, 8, 3, 8, 5, 4, 4, 6, 2, 8, 4, 2, 7, 8, 4, 8, 6, 5, 5, 6,
], dtype=float)
DEMAND = float(CPU_REQUESTS.sum())
N_SERVERS = CAPACITIES.size
X0_PROJECT = np.zeros(N_SERVERS - 1)
_A = DEMAND**2 / (N_SERVERS * CAPACITIES**2)

# Closed-form optimum in shares, represented by finite anchored logits.
W_STAR = CAPACITIES**2 / np.sum(CAPACITIES**2)
Z_STAR = np.log(W_STAR[:-1] / W_STAR[-1])
F_STAR = float(DEMAND**2 / (N_SERVERS * np.sum(CAPACITIES**2)))

# Capacity-proportional split has equal utilization on every server.
W_CAPACITY = CAPACITIES / CAPACITIES.sum()
Z_CAPACITY = np.log(W_CAPACITY[:-1] / W_CAPACITY[-1])
F_CAPACITY = float((DEMAND / CAPACITIES.sum())**2)


def shares(z):
    """Return nine positive CPU shares summing to one, using stable softmax."""
    z = np.asarray(z, dtype=float)
    if z.shape != (N_SERVERS - 1,):
        raise ValueError(f"Expected {N_SERVERS - 1} free logits, got {z.shape}.")
    if not np.all(np.isfinite(z)):
        raise ValueError("Logits must be finite.")
    logits = np.r_[z, 0.0]
    weights = np.exp(logits - np.max(logits))
    return weights / weights.sum()


def f_project(z):
    """Mean squared relative CPU utilization."""
    w = shares(z)
    return float(np.dot(_A, w**2))


def grad_project(z):
    """Gradient in the eight free logits: 2*w_i*(a_i*w_i - F)."""
    w = shares(z)
    value = np.dot(_A, w**2)
    return 2.0 * w[:-1] * (_A[:-1] * w[:-1] - value)


def hess_project(z):
    """Analytical Hessian in the eight free logits.

    B = diag(w) - w*w.T is the full softmax Jacobian.
    Differentiating 2*w*(a*w-F) gives
    H = 2*diag(2*a*w-F)*B - 4*outer(w, B.T@(a*w)).
    Restrict rows and columns to the free logits.
    """
    w = shares(z)
    value = np.dot(_A, w**2)
    B = np.diag(w) - np.outer(w, w)
    H = (2.0 * np.diag(2.0 * _A * w - value) @ B
         - 4.0 * np.outer(w, B.T @ (_A * w)))
    H = H[:-1, :-1]
    return (H + H.T) / 2.0


def cpu_loads(z):
    """CPU demand allocated to each server in the continuous relaxation."""
    return DEMAND * shares(z)


def utilizations(z):
    """Allocated CPU load divided by server capacity."""
    return cpu_loads(z) / CAPACITIES


def solution_errors(z):
    """Return the share infinity error and objective error required in Part C."""
    return (float(np.max(np.abs(shares(z) - W_STAR))),
            float(abs(f_project(z) - F_STAR)))
