import numpy as np
import pandas as pd

# 1. ALGORITHM IMPORTS
from src.optim.adam import optimize as optimize_adam
from src.optim.gd import gradient_descent as optimize_gd

# Import Newton (names adjusted according to newton.py)
from src.optim.newton import pure_newton as optimize_newton_pure
from src.optim.newton import damped_newton as optimize_newton_damped
from src.optim.momentum import optimize as optimize_momentum

# 2. PROBLEM IMPORTS (Hessian imports added: hess_...)
from src.problems.rosenbrock import f_rosenbrock, grad_rosenbrock, hess_rosenbrock, X0_R1, X0_R2
from src.problems.quadratic import f_q1, grad_q1, hess_q1, X0_Q1
from src.problems.quadratic import f_q2, grad_q2, hess_q2, X0_Q2

def generate_alpha_grid(base_start=-5, base_end=0, step=0.5):
    return [10**x for x in np.arange(base_start, base_end + step, step)]

def tune_optimizer(opt_func, f, grad, x0, alphas, tol_abs=None, tol_rel=None, **kwargs):
    best_k = float('inf')
    best_alpha = None
    
    current_alphas = alphas.copy()
    
    while True:
        for alpha in current_alphas:
            x, hist, k = opt_func(f, grad, x0, alpha=alpha, tol_abs=tol_abs, tol_rel=tol_rel, **kwargs)
            if k < 100000 and np.linalg.norm(x) < 1e12: # Valid run
                if k < best_k:
                    best_k = k
                    best_alpha = alpha
                    
        if best_alpha is not None and best_alpha == current_alphas[-1] and best_alpha < 1e4:
            new_exp = np.log10(best_alpha) + 0.5
            current_alphas = [10**new_exp]
        else:
            break
            
    return best_alpha, best_k

def run_part_a():
    results = []
    base_alphas = generate_alpha_grid()
    
    # Problem list now includes Hessian: (Name, f, grad, hess, x0, tol_abs, tol_rel)
    problems = [
        ('R1', f_rosenbrock, grad_rosenbrock, hess_rosenbrock, X0_R1, 1e-6, None),
        ('R2', f_rosenbrock, grad_rosenbrock, hess_rosenbrock, X0_R2, 1e-6, None),
        ('Q1', f_q1, grad_q1, hess_q1, X0_Q1, None, 1e-6),
        ('Q2', f_q2, grad_q2, hess_q2, X0_Q2, None, 1e-6)
    ]

    # Solver list: (Name, function, needs_alpha_tuning, requires_hessian)
    solvers = [
        ('Adam', optimize_adam, True, False),
        ('GD', optimize_gd, True, False),
        ('Newton (pure)', optimize_newton_pure, False, True),
        ('Newton (damped)', optimize_newton_damped, False, True),
        ('Momentum', optimize_momentum, True, False)
    ]

    print("Starting the experiments...")
    
    for prob_name, f, grad, hess, x0, tol_abs, tol_rel in problems:
        for solver_name, solver_func, needs_tuning, requires_hess in solvers:
            try:
                if requires_hess:
                    # Adapt parameters for the newton.py API
                    tol = tol_rel if tol_rel is not None else (tol_abs if tol_abs is not None else 1e-6)
                    relative = tol_rel is not None
                    
                    # Newton accepts: f, grad, hess, x0, tol, relative
                    _, _, k_opt = solver_func(f, grad, hess, x0, tol=tol, relative=relative)
                    alpha_opt = "-"
                else:
                    if needs_tuning:
                        alpha_opt, k_opt = tune_optimizer(solver_func, f, grad, x0, base_alphas, 
                                                          tol_abs=tol_abs, tol_rel=tol_rel)
                    else:
                        _, _, k_opt = solver_func(f, grad, x0, tol_abs=tol_abs, tol_rel=tol_rel)
                        alpha_opt = "-"
                
                results.append({
                    'Problem': prob_name, 
                    'Solver': solver_name, 
                    'Iterations': k_opt, 
                    'Alpha': alpha_opt
                })
                print(f"[{prob_name}] {solver_name}: Iterations = {k_opt}, Alpha = {alpha_opt}")
            except Exception as e:
                print(f"Error running {solver_name} on {prob_name}: {e}")

    # Save Table 1
    df = pd.DataFrame(results)
    df.to_csv('results/table1.csv', index=False)
    print("\nTable 1 successfully saved to results/table1.csv")

if __name__ == "__main__":
    run_part_a()