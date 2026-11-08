"""GA-A report extras. Run from the repository root:  python experiments/report_extras.py
Reads : results/Asem_runs.csv, results/Asem_exp.csv, results/T16_GA_traces_assignments.json
Writes: results/Asem_worst_seed_moves.csv, results/Asem_report_numbers.json, results/Asem_F2_diversity_N.png
"""
import csv, json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint4_generator_PUBLIC as gen

ZREF, ZLP = 7.345083, 7.328502
d = gen.cp34_instance(16, 'B', 3)
S, M = d['S'], d['M']
runs = pd.read_csv('results/Asem_runs.csv')
ga = runs[runs.configuration == 'GA, penalty'].sort_values('seed')
rs = runs[runs.configuration == 'random'].sort_values('seed')
exp = pd.read_csv('results/Asem_exp.csv')
J = json.load(open('results/T16_GA_traces_assignments.json'))
out = {}

# ---- worst seed: every transfer and every swap, steepest descent with both
worst = int(ga.seed[ga.cost.idxmax()])
sol = list(J[str(worst)]['solution'])
base, _ = gen.evaluate_B(d, sol)

def moves(sol):
    base, _ = gen.evaluate_B(d, sol)
    tr, sw = [], []
    for k in range(M):
        for l in range(S):
            if l != sol[k]:
                t = list(sol); t[k] = l; c, p = gen.evaluate_B(d, t)
                if not p and c < base - 1e-12: tr.append((c - base, t))
    for a in range(M):
        for b in range(a + 1, M):
            if sol[a] != sol[b]:
                t = list(sol); t[a], t[b] = t[b], t[a]; c, p = gen.evaluate_B(d, t)
                if not p and c < base - 1e-12: sw.append((c - base, t))
    return tr, sw

tr, sw = moves(sol)
cur, steps = list(sol), 0
while True:
    t_, s_ = moves(cur)
    allm = t_ + s_
    if not allm: break
    cur = min(allm, key=lambda x: x[0])[1]; steps += 1
desc_cost = gen.evaluate_B(d, cur)[0]
cpu = np.bincount(sol, weights=d['cpu_req'], minlength=S); mem = np.bincount(sol, weights=d['mem_req'], minlength=S)
out['worst'] = dict(seed=worst, cost=base, improving_transfers=len(tr), improving_swaps=len(sw),
                    best_transfer_delta=min(x[0] for x in tr), best_swap_delta=min(x[0] for x in sw),
                    descent_steps=steps, descent_cost=desc_cost,
                    max_mem_util=float((mem / d['mem_cap']).max()), mem_util=(mem / d['mem_cap']).round(3).tolist(),
                    cpu_util=(cpu / d['cpu_cap']).round(3).tolist())
with open('results/Asem_worst_seed_moves.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['seed', 'cost', 'improving_transfers', 'improving_swaps', 'best_transfer_delta',
                                     'best_swap_delta', 'descent_steps', 'descent_cost'])
    o = out['worst']
    w.writerow([worst, f"{base:.6f}", o['improving_transfers'], o['improving_swaps'], f"{o['best_transfer_delta']:.6f}",
                f"{o['best_swap_delta']:.6f}", steps, f"{desc_cost:.6f}"])

# ---- convergence numbers from the traces
T = np.array([[p[1] for p in J[str(s)]['trace']] for s in range(10)])
ev = [p[0] for p in J['0']['trace']]
mean = T.mean(0)
out['trace'] = {str(e): float(mean[ev.index(e)]) for e in (1000, 10000, 20000, 50000, 80000, 100000)}

# ---- diversity
dv = {s: np.array(J[str(s)]['diversity']) for s in range(10)}
out['div'] = dict(gens=len(dv[0]), last_values=sorted(set(int(v[-1]) for v in dv.values())),
                  last_full=sorted(set(int(v[-2]) for v in dv.values())),
                  gens_below_30=[int((v[:-1] < 30).sum()) for v in dv.values()],
                  min_before_last=int(min(v[:-1].min() for v in dv.values())))

# ---- population size
tab = {}
for name, vals in [('N=30', ga.cost.values)] + [(c, g.sort_values('seed').cost.values) for c, g in exp.groupby('configuration')]:
    tab[name] = dict(mean=float(vals.mean()), std=float(vals.std(ddof=1)), best=float(vals.min()), worst=float(vals.max()),
                     gap_ref_mean=float(100 * (vals.mean() - ZREF) / ZREF))
out['popsize'] = tab
c10 = exp[exp.configuration == 'GA, penalty, N=10'].sort_values('seed').cost.values
c100 = exp[exp.configuration == 'GA, penalty, N=100'].sort_values('seed').cost.values
out['paired'] = dict(N10_better_than_N30=int((c10 < ga.cost.values).sum()), N100_worse_than_N30=int((c100 > ga.cost.values).sum()),
                     GA_better_than_RS=int((ga.cost.values < rs.cost.values).sum()))
out['ref_vs_lp_gap_pct'] = 100 * (ZREF - ZLP) / ZREF
json.dump(out, open('results/Asem_report_numbers.json', 'w'), indent=1)

# ---- figure: diversity per generation (left) and effect of N (right)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={'width_ratios': [1.25, 1]})
for s in range(10):
    ax[0].plot(dv[s][:-1], lw=0.8, alpha=0.6)
ax[0].scatter([len(dv[0]) - 1] * 10, [dv[s][-1] for s in range(10)], c='red', s=14, zorder=5, label='last generation (cut by budget): 8')
ax[0].set_ylim(0, 33); ax[0].set_xlabel('Generation'); ax[0].set_ylabel('Distinct individuals')
ax[0].set_title('Diversity per generation, 10 seeds (N=30)'); ax[0].legend(loc='lower left', fontsize=8)
labels = ['N=10', 'N=30', 'N=100']; data = [c10, ga.cost.values, c100]
ax[1].boxplot(data, showmeans=True); ax[1].set_xticks([1,2,3]); ax[1].set_xticklabels(labels)
for i, v in enumerate(data):
    ax[1].scatter(np.full(len(v), i + 1) + np.linspace(-0.12, 0.12, len(v)), v, s=9, c='k', alpha=0.6)
ax[1].axhline(ZREF, color='r', ls='--', lw=1, label='z_ref'); ax[1].set_ylabel('Final feasible cost')
ax[1].set_title('Effect of N (seeds 0-9, budget 100000)'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('results/Asem_F2_diversity_N.png', dpi=200)
print(json.dumps(out, indent=1))
