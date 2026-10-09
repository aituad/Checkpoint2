import numpy as np
import pandas as pd
from src.optim.adam import optimize as optimize_adam
from src.problems.rosenbrock import f_rosenbrock, grad_rosenbrock, X0_R1, X0_R2
from src.problems.quadratic import f_q1, grad_q1, X0_Q1
from src.problems.quadratic import f_q2, grad_q2, X0_Q2
from src.optim.adam import optimize
# import other optimizers and problems from your team's code...

def generate_alpha_grid(base_start=-5, base_end=0, step=0.5):
    return [10**x for x in np.arange(base_start, base_end + step, step)]

def tune_optimizer(opt_func, f, grad, x0, alphas, tol_abs=None, tol_rel=None, **kwargs):
    best_k = float('inf')
    best_alpha = None
    best_result = None
    
    current_alphas = alphas.copy()
    
    while True:
        for alpha in current_alphas:
            x, hist, k = opt_func(f, grad, x0, alpha=alpha, tol_abs=tol_abs, tol_rel=tol_rel, **kwargs)
            if k < 100000 and np.linalg.norm(x) < 1e12: # Valid run
                if k < best_k:
                    best_k = k
                    best_alpha = alpha
                    best_result = (x, hist, k)
                    
        # Check boundary condition (upper edge expansion)
        if best_alpha is not None and best_alpha == current_alphas[-1] and best_alpha < 1e4:
            new_exp = np.log10(best_alpha) + 0.5
            current_alphas = [10**new_exp]
        else:
            break
            
    return best_alpha, best_k, best_result

def run_part_a():
    results = []
    base_alphas = generate_alpha_grid()
    
    # --- Example: Tune Adam on Rosenbrock (R1) ---
    # f_r1, grad_r1, x0_r1 should be imported from src.problems.rosenbrock
    # alpha_opt, k_opt, _ = tune_optimizer(optimize_adam, f_r1, grad_r1, x0_r1, base_alphas, tol_abs=1e-6)
    # results.append({'Problem': 'R1', 'Solver': 'Adam', 'Iterations': k_opt, 'Alpha': alpha_opt})
    

    # Пример запуска Adam для Q1 с относительной остановкой (как требует инструкция)
    x_final, hist, k = optimize(
        f=f_q1, 
        grad=grad_q1, 
        x0=X0_Q1, 
        alpha=0.1, 
        tol_rel=1e-6  # Для Q1 и Q2 правило остановки - относительное (relative)
    )
    print(f"Adam converged on Q1 in {k} iterations.")
    # Save to CSV
    # df = pd.DataFrame(results)
    # df.to_csv('results/table1.csv', index=False)
    # print(df)

if __name__ == "__main__":
    run_part_a()