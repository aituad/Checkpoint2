import numpy as np
from .ga_problem import random_individual, evaluate

def solve(d, variant, seed, budget, params=None, start=None):
    rng = np.random.default_rng(seed)
    evals = 0
    best_cost = float('inf')
    best_fit = float('inf')
    best_sol = None
    is_feas = False
    
    while evals < budget:
        ind = random_individual(d, rng)
        res = evaluate(d, ind, rho=10.0) 
        evals += 1
        
        if res['feasible'] and res['cost'] < best_cost:
            best_cost = res['cost']
            best_sol = ind.copy()
            is_feas = True
        elif not is_feas and res['fitness'] < best_fit:
            best_fit = res['fitness']
            best_cost = res['cost']
            best_sol = ind.copy()
            
    return {
        "solution": best_sol.tolist() if best_sol is not None else None, 
        "cost": float(best_cost), 
        "evals": evals, 
        "feasible": is_feas,
        "trace": []
    }