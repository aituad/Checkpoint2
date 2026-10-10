import os
import numpy as np
import pandas as pd

# =============================================================================
# 1. ALGORITHM IMPORTS
# =============================================================================
from src.optim.gd import gradient_descent, gradient_descent_backtracking
from src.optim.newton import pure_newton, damped_newton
from src.optim.momentum import optimize as optimize_momentum
from src.optim.adam import optimize as optimize_adam

# =============================================================================
# 2. PROBLEM IMPORTS
# =============================================================================
from src.problems.rosenbrock import f_rosenbrock, grad_rosenbrock, hess_rosenbrock
from src.problems.quadratic import f_q1, grad_q1, hess_q1, f_q2, grad_q2, hess_q2
from src.problems.project import f_project as f_proj, grad_project as grad_proj, hess_project as hess_proj

# =============================================================================
# 3. TEAM DATA (TEAM T16)
# =============================================================================
SEED = 16
C_VAL = 129
THETA_DEG = 71

X0_R1 = np.array([-1.2, 1.0])
X0_R2 = np.array([-1.51, 2.17])
X0_Q1 = np.array([1.3, 0.7])

def get_rotation_matrix(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

X0_Q2 = get_rotation_matrix(THETA_DEG) @ X0_Q1
X0_PROJ = np.zeros(8) # z ∈ R^8, start z⁰ = 0 

# =============================================================================
# 4. TUNING LOGIC
# =============================================================================
def tune_optimizer(solver_name, solver_func, prob_name, f, grad, x0, tol_abs, tol_rel, requires_hess, hess=None):
    # Начальная сетка: 10^-5, 10^-4.5, ..., 10^0
    alphas_to_test = [float(10**x) for x in np.arange(-5.0, 0.5, 0.5)]
    
    if solver_name == 'GD' and prob_name in ['Q1', 'Q2']:
        alphas_to_test.append(float(1.0 / (1.0 + C_VAL)))

    betas_to_test = [0.5, 0.8, 0.9, 0.95, 0.99] if solver_name == 'Momentum' else [None]

    best_k = float('inf')
    best_alpha = None
    best_beta = None
    
    history = {}

    while True:
        for a in alphas_to_test:
            if a in history: continue
            
            for b in betas_to_test:
                kwargs = {'alpha': a}
                if b is not None:
                    kwargs['beta'] = b
                
                try:
                    if requires_hess:
                        x, hist, k = solver_func(f, grad, hess, x0, tol_abs=tol_abs, tol_rel=tol_rel, **kwargs)
                    else:
                        x, hist, k = solver_func(f, grad, x0, tol_abs=tol_abs, tol_rel=tol_rel, **kwargs)
                    
                    history[a] = True
                    
                    if k < 100000 and np.linalg.norm(x) < 1e12 and not np.isnan(f(x)):
                        if k < best_k:
                            best_k = k
                            best_alpha = a
                            best_beta = b
                except Exception as e:
                    history[a] = True
                    pass

        standard_alphas = [a for a in history.keys() if a != float(1.0 / (1.0 + C_VAL))]
        max_tested_alpha = max(standard_alphas) if standard_alphas else None
        
        if best_alpha is not None and max_tested_alpha is not None and best_alpha == max_tested_alpha and best_alpha < 9999:
            new_alpha = best_alpha * (10**0.5)
            alphas_to_test = [new_alpha]
            print(f"[{solver_name}] Optimum on boundary! Extending alpha grid to {new_alpha:.4e}")
        else:
            break

    if best_alpha is None:
        return "—", "—"
    
    params_str = f"a={best_alpha:.2e}" if best_beta is None else f"a={best_alpha:.2e}, b={best_beta}"
    return best_k, params_str

# =============================================================================
# 5. MAIN EXPERIMENT RUNNER
# =============================================================================
def run_experiments():
    os.makedirs('results', exist_ok=True)
    
    problems_part_a = [
        ('R1', f_rosenbrock, grad_rosenbrock, hess_rosenbrock, X0_R1, 1e-6, None),  # Absolute rule
        ('R2', f_rosenbrock, grad_rosenbrock, hess_rosenbrock, X0_R2, 1e-6, None),  # Absolute rule
        ('Q1', f_q1, grad_q1, hess_q1, X0_Q1, None, 1e-6),                          # Relative rule
        ('Q2', f_q2, grad_q2, hess_q2, X0_Q2, None, 1e-6)                           # Relative rule
    ]
    
    problems_part_c = [
        ('Project', f_proj, grad_proj, hess_proj, X0_PROJ, None, 1e-6)              # Relative rule
    ]

    # needs_tuning (bool), requires_hess (bool), is_newton (bool)
    solvers = [
        ('GD', gradient_descent, True, False, False),
        ('GD-BT', gradient_descent_backtracking, False, False, False),
        ('Newton (pure)', pure_newton, False, True, True),
        ('Newton (damped)', damped_newton, False, True, True),
        ('Momentum', optimize_momentum, True, False, False),
        ('Adam', optimize_adam, True, False, False)
    ]

    def process_problems(problem_list, table_name):
        results = []
        for prob_name, f, grad, hess, x0, tol_abs, tol_rel in problem_list:
            print(f"\n--- Solving problem {prob_name} ---")
            for solver_name, solver_func, needs_tuning, requires_hess, is_newton in solvers:
                
                kwargs = {}

                if is_newton:
                    kwargs['tol'] = tol_rel if tol_rel is not None else tol_abs
                    kwargs['relative'] = tol_rel is not None

                try:
                    if needs_tuning:
                        k_opt, params_opt = tune_optimizer(
                            solver_name, solver_func, prob_name, f, grad, x0, tol_abs, tol_rel, requires_hess, hess)
                    else:
                        if requires_hess:
                            x, hist, k = solver_func(f, grad, hess, x0, **kwargs)
                        else:
                            x, hist, k = solver_func(f, grad, x0, tol_abs=tol_abs, tol_rel=tol_rel)
                        
                        k_opt = k if k < 100000 else "—"
                        params_opt = "—"
                        
                    results.append({
                        'Problem': prob_name, 
                        'Solver': solver_name, 
                        'Iterations': k_opt, 
                        'Parameters': params_opt
                    })
                    print(f"[{prob_name}] {solver_name:15s}: Iterations = {k_opt}, Parameters = {params_opt}")
                except Exception as e:
                    print(f"Error running {solver_name} on {prob_name}: {e}")
                    results.append({'Problem': prob_name, 'Solver': solver_name, 'Iterations': "—", 'Parameters': "Error"})
                    
        df = pd.DataFrame(results)
        df.to_csv(f'results/{table_name}.csv', index=False)
        print(f"\nTable saved to results/{table_name}.csv")

    process_problems(problems_part_a, 'table1')
    process_problems(problems_part_c, 'table2')

if __name__ == "__main__":
    run_experiments()