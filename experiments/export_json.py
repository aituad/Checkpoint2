import json, sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from checkpoint4_generator_PUBLIC import cp34_instance
from src.meta.ga import solve

d = cp34_instance(16, 'B', 3)
params = {"N": 30, "pc": 0.9, "tournament_size": 2, "rho": 10.0}
data = {}

print("Collecting traces and assign arrays for Dilnaz...")
for seed in range(10):
    res = solve(d, 3, seed, 100000, params=params)
    data[seed] = {
        "solution": res["solution"],
        "trace": res["trace"],
        "diversity": res.get("diversity", [])
    }

results_dir = os.path.join(root_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'T16_GA_traces_assignments.json')

with open(out_path, 'w') as f:
    json.dump(data, f)
    
print(f"Done: {out_path}")