import numpy as np

def check_optimality_conditions(c, A, b, row_signs, x, pi, r, upper_bounds):
    """
        Checks the 5 KKT conditions for an LP. 
        row_signs: an array of strings ('<=', '>=', '==') for each constraint. 
        upper_bounds: an array of upper bounds for the variables (1.0 for binary variables, np.inf for continuous variables). 
        Returns 5 numbers (the maximum violations) for Table 2.
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
    dual_viol = 0.0
    for i in range(M):
        if row_signs[i] == '<=':
            dual_viol = max(dual_viol, max(0, pi[i]))  # pi <= 0
        elif row_signs[i] == '>=':
            dual_viol = max(dual_viol, max(0, -pi[i])) # pi >= 0
        # For '==', pi can be anything (free); 0 violations.

    # (c) Complementary slackness
    comp_slack_viol = np.max(np.abs(pi * (b - Ax)))

    # (d) Reduced costs
    red_cost_viol = 0.0
    for j in range(N):
        if x[j] <= tol:
            red_cost_viol = max(red_cost_viol, max(0, -r[j]))
        elif x[j] >= upper_bounds[j] - tol:
            red_cost_viol = max(red_cost_viol, max(0, r[j]))
        else: # 0 < x < UB (including t_l > 0)
            red_cost_viol = max(red_cost_viol, abs(r[j]))

    # (e) Strong duality
    # The sum min(0, r_j) is taken only for variables with an upper bound of 1
    bound_corrections = np.sum(np.minimum(0, r[upper_bounds == 1.0]))
    dual_obj = np.dot(b, pi) + bound_corrections
    primal_obj = np.dot(c, x)
    strong_duality_viol = abs(primal_obj - dual_obj)

    return (prim_viol, dual_viol, comp_slack_viol, red_cost_viol, strong_duality_viol)
