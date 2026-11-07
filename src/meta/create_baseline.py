import sys
import os
import csv
import time

# Путь к корню проекта
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

import checkpoint4_generator_PUBLIC as gen
from src.meta.random_search import solve

# Экземпляр задачи для команды T16, Story B, Variant 3
d = gen.cp34_instance(16, 'B', 3)

# Файл результатов
results_file = os.path.join(ROOT, "results", "random_baseline.csv")

with open(results_file, "w", newline="") as f:
    w = csv.writer(f)

    w.writerow([
        "team",
        "role",
        "configuration",
        "seed",
        "budget",
        "cost",
        "evals",
        "feasible",
        "seconds",
        "params"
    ])

    for seed in range(10):
        print(f"Running seed {seed}...")

        start_time = time.time()

        res = solve(
            d=d,
            variant=3,
            seed=seed,
            budget=100000
        )

        seconds = round(time.time() - start_time, 2)

        w.writerow([
            "T16",
            "SA-T",
            "random",
            seed,
            100000,
            res["cost"],
            res["evals"],
            res["feasible"],
            seconds,
            ""
        ])

        print(
            f"Seed {seed}: "
            f"cost={res['cost']:.6f}, "
            f"time={seconds}s"
        )

print(f"\nГотово. Файл сохранён:")
print(results_file)
