import numpy as np

def optimize(f, grad, x0, alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-8, 
             max_iter=100000, tol_abs=None, tol_rel=None):
    """
    Adam optimizer based on Checkpoint 2 conventions.
    """
    x = np.array(x0, dtype=float)
    hist = []
    
    m = np.zeros_like(x)
    s = np.zeros_like(x)
    
    g0_norm = np.linalg.norm(grad(x))
    k = 0
    
    while k < max_iter:
        g = grad(x)
        g_norm = np.linalg.norm(g)
        
        # 1. Checking stopping rules before the update
        if tol_abs is not None and g_norm < tol_abs:
            hist.append(x.copy())
            break
        if tol_rel is not None and g_norm < (tol_rel * g0_norm):
            hist.append(x.copy())
            break
            
        hist.append(x.copy())
        
        # 2. Updating logic
        k += 1
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * (g**2)
        
        m_hat = m / (1 - beta1**k)
        s_hat = s / (1 - beta2**k)
        
        x = x - alpha * (m_hat / (np.sqrt(s_hat) + eps))
        
        # 3. Blow-up guard
        if np.linalg.norm(x) > 1e12:
            break
            
    return x, np.array(hist), k