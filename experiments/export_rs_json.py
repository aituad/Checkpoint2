import json, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from checkpoint4_generator_PUBLIC import cp34_instance
from src.meta.random_search import solve

d = cp34_instance(16, 'B', 3)
rs_data = {}

print("Выгрузка массивов Random Search...")
for seed in range(10):
    res = solve(d, 3, seed, 100000)
    rs_data[seed] = {
        "solution": res["solution"],
        "cost": res["cost"],
        "feasible": res["feasible"]
    }

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'T16_RS_assignments.json')
with open(out_path, 'w') as f:
    json.dump(rs_data, f, indent=2)
print(f"Готово: {out_path}")