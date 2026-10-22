import numpy as np

def check_optimality_conditions(c, A, b, row_signs, x, pi, r, upper_bounds):
    """
    Checks 5 KKT conditions for LP.
    Returns 5 values (maximum violations) for Table 2.
    """
    tol = 1e-6
    N = len(x)
    M = len(b)

    # (a) Primal feasibility
    Ax = A @ x
    prim_viol = 0.0
    for i in range(M):
        if row_signs[i] == '<=':
            viol = max(0, Ax[i] - b[i])
        elif row_signs[i] == '>=':
            viol = max(0, b[i] - Ax[i])
        else:
            viol = abs(Ax[i] - b[i])
        prim_viol = max(prim_viol, viol)
    
    bound_viol = max(np.max(np.maximum(0, -x)), 
                     np.max(np.maximum(0, x - upper_bounds)))
    prim_viol = max(prim_viol, bound_viol)

    # (b) Dual feasibility (signs)
    # By M2 convention and specification: for '<=' rows, price pi <= 0
    dual_viol = 0.0
    for i in range(M):
        if row_signs[i] == '<=':
            dual_viol = max(dual_viol, max(0, pi[i]))   # Violation if pi > 0
        elif row_signs[i] == '>=':
            dual_viol = max(dual_viol, max(0, -pi[i]))  # Violation if pi < 0

    # (c) Complementary slackness
    comp_slack_viol = np.max(np.abs(pi * (b - Ax)))

    # (d) Reduced costs
    red_cost_viol = 0.0
    for j in range(N):
        if upper_bounds[j] == 0.0:
            # Ignore fixed variables (M2 advice)
            continue
            
        if x[j] <= tol:
            red_cost_viol = max(red_cost_viol, max(0, -r[j])) # Should be r_j >= 0
        elif x[j] >= upper_bounds[j] - tol:
            red_cost_viol = max(red_cost_viol, max(0, r[j]))  # Should be r_j <= 0
        else:
            red_cost_viol = max(red_cost_viol, abs(r[j]))     # Should be r_j == 0

    # (e) Strong duality
    # M2 formula: c^T x = b^T pi + sum(min(0, r_j) * upper_j) for finite bounds
    finite_mask = ~np.isinf(upper_bounds)
    bound_corrections = np.sum(np.minimum(0, r[finite_mask]) * upper_bounds[finite_mask])
    
    dual_obj = np.dot(b, pi) + bound_corrections
    primal_obj = np.dot(c, x)
    strong_duality_viol = abs(primal_obj - dual_obj)

    return (prim_viol, dual_viol, comp_slack_viol, red_cost_viol, strong_duality_viol)