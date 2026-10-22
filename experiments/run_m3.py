import csv
import numpy as np
from pathlib import Path
from src.kkt.checker import check_optimality_conditions

def main():
    # File paths
    out = Path(__file__).resolve().parents[1] / 'results'
    
    # 1. Load M1 and M2 data
    print("Loading matrices from M1 and solution from M2...")
    m1 = np.load(out / 'm1_matrices.npz')
    m2 = np.load(out / 'm2_solution.npz')
    
    # 2. Call the optimality checker
    violations = check_optimality_conditions(
        c=m1['c'], 
        A=m1['A'], 
        b=m1['b'], 
        row_signs=m1['senses'], 
        x=m2['x'], 
        pi=m2['pi'], 
        r=m2['r'], 
        upper_bounds=m1['upper']
    )
    
    # 3. Save Table 2
    table2_path = out / 'table2.csv'
    with table2_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'description', 'max_violation'])
        w.writerow(['a', 'primal feasibility', f"{violations[0]:.2e}"])
        w.writerow(['b', 'dual feasibility', f"{violations[1]:.2e}"])
        w.writerow(['c', 'complementary slackness', f"{violations[2]:.2e}"])
        w.writerow(['d', 'reduced costs', f"{violations[3]:.2e}"])
        w.writerow(['e', 'strong duality', f"{violations[4]:.2e}"])
        
    print(f"Table 2 successfully saved to {table2_path}")
    print("Violations:", violations)

if __name__ == '__main__':
    main()