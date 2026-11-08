"""GA-A experiment: effect of the population size N on the final feasible cost.
Same instance, budget 100000, seeds 0-9, all other parameters as in the final runs (pc=0.9, tournament 2, rho=10).
N=30 is NOT rerun: it is the 'GA, penalty' block of results/Asem_runs.csv.
Run from the repository root:  python experiments/run_popsize.py
Output: results/Asem_exp.csv (columns of Section 3 of the instructions).
"""
import csv, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint4_generator_PUBLIC as gen
from src.meta.interface import check
from src.meta.ga import solve

COLS = ['team', 'role', 'configuration', 'seed', 'budget', 'cost', 'evals', 'feasible', 'seconds', 'params']
d = gen.cp34_instance(16, 'B', 3)
os.makedirs('results', exist_ok=True)
with open('results/Asem_exp.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(COLS)
    for N in (10, 100):
        params = {"N": N, "pc": 0.9, "tournament_size": 2, "rho": 10.0}
        for seed in range(10):
            t0 = time.time()
            r = solve(d, 3, seed, 100000, params=params)
            sec = time.time() - t0
            cost, problems = check(d, 3, r['solution'])
            assert r['evals'] == 100000
            w.writerow(['T16', 'GA-A', f'GA, penalty, N={N}', seed, 100000, f'{cost:.9f}', r['evals'],
                        len(problems) == 0, f'{sec:.2f}', json.dumps(params)])
            fh.flush()
            print(N, seed, f'{cost:.6f}', len(problems) == 0, f'{sec:.1f}s', flush=True)
