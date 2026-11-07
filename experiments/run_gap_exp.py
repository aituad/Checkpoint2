import csv
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from checkpoint4_generator_PUBLIC import cp34_instance
from src.meta.ga import solve

if __name__ == "__main__":
    TEAM_SEED = 16
    VARIANT = 3
    STORY = 'B'
    BUDGET = 100000
    
    d = cp34_instance(TEAM_SEED, STORY, VARIANT)
    results = []
    
    # Launch of the Dilnaz experiment (GA-P)
    for rho in [1.0, 10.0, 100.0]:
        params = {"N": 30, "pc": 0.9, "tournament_size": 2, "rho": rho}
        for seed in range(100, 105):
            print(f"Running GA (rho={rho}) on tuning seed {seed}...")
            start_time = time.time()
            res_ga = solve(d, VARIANT, seed, BUDGET, params=params)
            duration = time.time() - start_time
            
            results.append({
                "team": "T16", "role": "GA-P", "configuration": f"GA, penalty", 
                "seed": seed, "budget": BUDGET, "cost": res_ga["cost"], 
                "evals": res_ga["evals"], "feasible": res_ga["feasible"], 
                "seconds": round(duration, 2), "params": str(params),
                "offspring_evaluated": res_ga["offspring_evaluated"],
                "offspring_feasible": res_ga["offspring_feasible"]
            })

    os.makedirs('../results', exist_ok=True)
    with open('../results/bekturova_exp.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "team", "role", "configuration", "seed", "budget", 
            "cost", "evals", "feasible", "seconds", "params",
            "offspring_evaluated", "offspring_feasible"
        ])
        writer.writeheader()
        writer.writerows(results)
    print("GA-P experiment saved to results/bekturova_exp.csv")