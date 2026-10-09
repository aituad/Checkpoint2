import numpy as np
from src.optim.adam import optimize

def test_adam_hand_trace():
    # 1. f(x, y) = 2x^2 + 6y^2
    def f(x): return 2 * x[0]**2 + 6 * x[1]**2
    def grad(x): return np.array([4 * x[0], 12 * x[1]])
    
    x0 = np.array([3.0, 2.0])
    
    x_final, hist, k = optimize(f, grad, x0, alpha=0.1, max_iter=2)
    
    print("  [ Manual calculation results (H4) ]")
    print(f"Step 0 (x0): {hist[0]}")
    print(f"Step 1 (x1): {hist[1]}")
    print(f"Step 2 (x2 / x_final): {x_final}")
    print(f"Total steps taken: {k}\n")
    
    # Checking
    np.testing.assert_allclose(hist[1], np.array([2.9, 1.9]), atol=1e-3)
    np.testing.assert_allclose(x_final, np.array([2.8001, 1.8002]), atol=1e-3)
    
    # 2. Test (Part A)
    def rosenbrock_grad(x):
        return np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 
                         200*(x[1]-x[0]**2)])
    x0_rosen = np.array([-1.2, 1.0])
    x_rosen, _, k_rosen = optimize(None, rosenbrock_grad, x0_rosen, alpha=0.1, tol_abs=1e-6)
    
    print("  [ Results for the Rosenbrock function (R1) ]")
    print(f"A minimum point has been found: {x_rosen}")
    print(f"Number of iterations k_rosen: {k_rosen}")
    print(f"Expected: approximately 1706 iterations\n")
    
    assert abs(k_rosen - 1706) < 10, f"Expected ~1706, got {k_rosen}"

if __name__ == "__main__":
    test_adam_hand_trace()
    print("Success!")