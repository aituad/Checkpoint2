import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

import checkpoint4_generator_PUBLIC as gen
from src.meta.sa import solve

d = gen.cp34_instance(16, 'B', 3)

costs = []

print("Rule-based results:")

for seed in range(100, 105):
    res = solve(d, 3, seed, 100000, params=None)

    cost = res["cost"]
    costs.append(cost)

    print(f"{seed}: {cost}")

average = sum(costs) / len(costs)

print("\nAverage cost:", average)
