import numpy as np

def test_h3_verbose():
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    
    print("=== Checkpoint 3: Task H3 (Triangle Problem) ===")
    c = np.array([3.0, 5.0, 6.0])
    A = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0]
    ])
    b = np.array([1.0, 1.0, 1.0])
    x = np.array([0.5, 0.5, 0.5])
    
    print(f"Objective function c: {c}")
    print(f"Decision point x:   {x}")
    print("-" * 50)
    
    # --- H3(a) ---
    print("H3(a) KKT check for a vector pi = (1, 4, 2)")
    pi_a = np.array([1.0, 4.0, 2.0])
    
    Ax = A @ x
    slack = Ax - b
    prim_viol_a = np.max(np.maximum(0, -slack))
    print(f"(a) Primal feasibility: Ax = {Ax}, slack = {slack}")
    print(f"    Max. violation: {prim_viol_a:.3f}")
    
    dual_viol_a = np.max(np.maximum(0, -pi_a))
    print(f"(b) Dual feasibility: pi = {pi_a}")
    print(f"    Max. violation: {dual_viol_a:.3f}")
    
    comp = pi_a * slack
    comp_viol_a = np.max(np.abs(comp))
    print(f"(c) Comp. slackness: pi * slack = {comp}")
    print(f"    Max. violation: {comp_viol_a:.3f}")
    
    r_a = c - A.T @ pi_a
    red_viol_a = np.max(np.abs(r_a)) 
    print(f"(d) Reduced costs: r = c - A^T*pi = {r_a}")
    print(f"    Max. violation (since 0 < x < 1): {red_viol_a:.3f}")
    
    cx = c @ x
    bpi_a = b @ pi_a
    duality_gap_a = np.abs(cx - bpi_a)
    print(f"(e) Strong duality: c^T*x = {cx:.3f}, b^T*pi = {bpi_a:.3f}")
    print(f"    Difference (Gap): {duality_gap_a:.3f}")
    
    print(f"-> Result H3(a) (five numbers): ({prim_viol_a:.3f}, {dual_viol_a:.3f}, {comp_viol_a:.3f}, {red_viol_a:.3f}, {duality_gap_a:.3f})\n")
    
    # Assert-check
    assert prim_viol_a < 1e-6
    assert dual_viol_a < 1e-6
    assert comp_viol_a < 1e-6
    assert red_viol_a < 1e-6
    assert duality_gap_a < 1e-6

    # --- H3(b) ---
    print("H3(b) KKT check for a vector pi' = (2, 3, 1)")
    pi_b = np.array([2.0, 3.0, 1.0])
    
    r_b = c - A.T @ pi_b
    red_viol_b = np.max(np.abs(r_b))
    print(f"(d) Reduced costs: r' = {r_b}")
    print(f"    Expected violation (r_3 != 0): {red_viol_b:.3f}")
    
    bpi_b = b @ pi_b
    duality_gap_b = np.abs(cx - bpi_b)
    print(f"(e) Strong duality: c^T*x = {cx:.3f}, b^T*pi' = {bpi_b:.3f}")
    print(f"   Expected Gap: {duality_gap_b:.3f}")
    print("-" * 50)
    

    assert np.isclose(red_viol_b, 2.0, atol=1e-6)
    assert np.isclose(duality_gap_b, 1.0, atol=1e-6)
    
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_h3_verbose()