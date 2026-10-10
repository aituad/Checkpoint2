import numpy as np

def optimize(f, grad, x0, alpha=0.01, beta=0.6, tol_abs=None, tol_rel=None, max_iter=100000):
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros_like(x)
    hist = [x.copy()]
    
    initial_grad_norm = np.linalg.norm(grad(x))
    
    if tol_rel is not None:
        threshold = tol_rel * initial_grad_norm
        relative = True
    elif tol_abs is not None:
        threshold = tol_abs
        relative = False
    else:
        threshold = 1e-6
        relative = False
        
    for k in range(max_iter + 1):
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e12:
            return x, np.asarray(hist), k
            
        g = grad(x)
        
        if np.linalg.norm(g) < threshold or (relative and initial_grad_norm == 0):
            return x, np.asarray(hist), k
            
        if k == max_iter:
            return x, np.asarray(hist), k
            
        v = beta * v - alpha * g
        x = x + v
        hist.append(x.copy())
        
    return x, np.asarray(hist), max_iter