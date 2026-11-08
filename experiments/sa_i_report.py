"""SA-I: Table 1, Table 2, Figure F1 and the acceptance curve from results/Mutali_runs.csv.

    python3 experiments/sa_i_report.py                                  # SA only
    python3 experiments/sa_i_report.py --baseline results/<tuner>_tuning.csv   # + random search of the partner

What it does
  1. reads the final runs (seed, params, cost) from the CSV and re-runs solve() with the same seed
     and params (a run takes about 0.3 s); it stops if a re-run cost differs from the CSV by more than 1e-6,
     so Table 1, Table 2 and the figures are tied to the file;
  2. Table 1: best, median, mean, worst, sample standard deviation (n-1) of the cost, feasible runs,
     mean time per run (from the CSV), gap_ref and gap_LP of best/median/mean/worst feasible run;
  3. Table 2 (worst seed = feasible run with the largest cost): final cost, improving transfers and
     swaps (src/meta/sa_checks.py, checker-based), share of accepted worsening proposals in the first
     and in the last 5 % of the annealing proposals;
  4. F1: best-so-far against evaluations (mean of the runs, min-max band, lines z_LP, z_ref);
     F2: share of accepted worsening proposals against the temperature;
  5. writes results/Mutali_table1.csv, results/Mutali_table2.csv, results/Mutali_numbers.json,
     figures/F1_Mutali.png, figures/F2_Mutali_acceptance.png.

z_ref and z_LP are used only here (reporting), never inside src/meta/sa.py.
"""
import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import checkpoint4_generator_PUBLIC as gen
from src.meta.sa import solve
from src.meta.sa import State
from src.meta.sa_checks import improving_moves, improving_move_list

Z_REF, Z_LP = 7.345083, 7.328502          # task file of team T16 (6 decimals)


def gap_ref(z):
    if abs(z - Z_REF) < 1e-6:                       # z_ref is given with 6 decimals: a difference below 1e-6 is rounding
        return 0.0
    return 100.0 * (z - Z_REF) / Z_REF


def gap_lp(z):
    return 100.0 * (z - Z_LP) / z


def summary(costs):
    s = sorted(costs)
    return {'best': s[0], 'median': statistics.median(s), 'mean': statistics.fmean(s), 'worst': s[-1],
            'std': statistics.stdev(s) if len(s) > 1 else float('nan')}


def read_runs(path, role=None, configuration=None):
    rows = []
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            if role and r['role'] != role:
                continue
            if configuration and r['configuration'] != configuration:
                continue
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='results/Mutali_runs.csv')
    ap.add_argument('--params-filter', default='', help="use only rows whose params column equals this text ('' = rule-based)")
    ap.add_argument('--baseline', default=None, help='CSV of the random-search baseline (configuration "random")')
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--figdir', default='figures')
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)

    d = gen.cp34_instance(16, 'B', 3)
    rows = [r for r in read_runs(a.runs, role='SA-I') if (r['params'] or '') == a.params_filter]
    rows.sort(key=lambda r: int(r['seed']))
    assert rows, 'no rows with that params text'
    budget = int(rows[0]['budget'])
    runs = []
    for r in rows:
        params = json.loads(r['params']) if r['params'] else None
        res = solve(d, 3, int(r['seed']), budget, params=params)
        assert abs(res['cost'] - float(r['cost'])) < 1e-6, ('re-run differs from the CSV', r['seed'], res['cost'], r['cost'])
        runs.append((int(r['seed']), res, float(r['seconds']), r['feasible'] == 'True'))

    feas = [(s, res) for s, res, sec, f in runs if f]
    costs = [res['cost'] for s, res in feas]
    sm = summary(costs)
    mean_time = statistics.fmean(sec for s, res, sec, f in runs)
    n_feas = len(feas)

    # ---- Table 1
    t1 = [('best', sm['best']), ('median', sm['median']), ('mean', sm['mean']), ('worst', sm['worst'])]
    with open(os.path.join(a.outdir, 'Mutali_table1.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['statistic', 'cost', 'gap_ref_percent', 'gap_LP_percent'])
        for name, v in t1:
            w.writerow([name, f'{v:.6f}', f'{gap_ref(v):.4f}', f'{gap_lp(v):.4f}'])
        w.writerow(['std (n-1)', f'{sm["std"]:.6f}', '', ''])
        w.writerow(['feasible runs', f'{n_feas}/{len(runs)}', '', ''])
        w.writerow(['mean time per run (s)', f'{mean_time:.2f}', '', ''])

    # ---- Table 2 (worst feasible seed)
    ws, wres = max(feas, key=lambda x: x[1]['cost'])
    n_tr, n_sw, best_tr, best_sw = improving_moves(d, wres['solution'])
    st = wres['stats']
    tr_list, sw_list = improving_move_list(d, wres['solution'])
    sstate = State(d, wres['solution'])
    end_tr, end_sw, _, _ = improving_moves(d, st['end_assign'])      # the state in which the run stopped (not the returned best)
    f5 = st['first5'][0] / st['first5'][1]
    l5 = st['last5'][0] / st['last5'][1]
    with open(os.path.join(a.outdir, 'Mutali_table2.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['quantity', 'value'])
        w.writerows([('worst_seed', ws), ('final_cost', f'{wres["cost"]:.6f}'),
                     ('improving_transfers', n_tr), ('improving_swaps', n_sw),
                     ('best_transfer_delta', f'{best_tr:.6f}'), ('best_swap_delta', f'{best_sw:.6f}'),
                     ('accepted_worsening_first5pct', f'{st["first5"][0]}/{st["first5"][1]} = {f5:.4f}'),
                     ('accepted_worsening_last5pct', f'{st["last5"][0]}/{st["last5"][1]} = {l5:.4f}'),
                     ('T0', f'{st["T0"]:.5f}'), ('Tend', f'{st["Tend"]:.6f}'),
                     ('annealing_proposals_K', st['K']), ('calibration_walk', st['walk']),
                     ('best_found_at_eval', st['best_at_eval']), ('T_when_best_found', f'{st["T_at_best"]:.5f}'),
                     ('cost_of_current_state_at_end', f'{st["current_cost_end"]:.6f}'),
                     ('end_state_improving_transfers', end_tr), ('end_state_improving_swaps', end_sw)])

    # ---- convergence numbers (question 3)
    grid = np.arange(1000, budget + 1, 1000)
    traces = []
    for s, res in feas:
        tr = res['trace']
        ev = np.array([e for e, _ in tr]); val = np.array([v for _, v in tr])
        traces.append(np.interp(grid, ev, val))     # best-so-far is a step function; trace points are at multiples of 1000
    traces = np.array(traces)
    mean_tr, lo, hi = traces.mean(axis=0), traces.min(axis=0), traces.max(axis=0)
    final_mean = mean_tr[-1]
    flat_at = next(int(g) for g, v in zip(grid, mean_tr) if v <= final_mean + 1e-3)
    last_improve = {}
    for s, res in feas:
        tr = res['trace']
        last = tr[0][0]
        for (e0, v0), (e1, v1) in zip(tr, tr[1:]):
            if v1 < v0 - 1e-12:
                last = e1
        last_improve[s] = last
    at90 = float(np.interp(0.9 * budget, grid, mean_tr))

    # ---- acceptance of worsening proposals at fixed temperatures (question 5): mean over the runs
    acc_at = {}
    for T_target in (1.0, 0.1, 0.01):
        vals = []
        for s_, res in feas:
            cv = [(T, acc / w) for T, w, acc in res['stats']['curve'] if w > 0]
            x = np.array([-np.log10(T) for T, _ in cv]); y = np.array([v for _, v in cv])   # cooling: x increases
            if x.min() <= -np.log10(T_target) <= x.max():
                vals.append(float(np.interp(-np.log10(T_target), x, y)))
        acc_at[str(T_target)] = float(np.mean(vals)) if vals else None

    # ---- the other parameter sets in the same file (SA-I runs both: rule-based and the tuned set of SA-T), seeds 0-9
    alt = {}
    for r in read_runs(a.runs, role='SA-I'):
        if (r['params'] or '') != a.params_filter and r['feasible'] == 'True':
            alt.setdefault(r['params'], []).append(float(r['cost']))
    alt_summary = {}
    for k_, v_ in alt.items():
        sm_ = summary(v_)
        alt_summary[k_] = {'n': len(v_), 'mean': sm_['mean'], 'worst': sm_['worst'], 'best': sm_['best'],
                           'gap_ref_mean': gap_ref(sm_['mean']), 'gap_ref_worst': gap_ref(sm_['worst'])}

    # ---- numbers for the explanation of the worst seed
    from src.meta.sa import allowed_servers
    allowed = allowed_servers(d)
    p_improve = sum(1.0 / (d['M'] * (len(allowed[k]) - 1)) for k, f_, t_, dl in tr_list)   # chance that one proposal is an improving transfer
    cv = [(T, w, acc) for T, w, acc in st['curve'] if w > 0]
    Tb = st['T_at_best']
    blk = max(1, st['K'] // 50)
    near = min(cv, key=lambda x: abs(np.log10(x[0]) - np.log10(Tb)))
    p_leave = near[2] / blk                               # accepted worsening proposals per proposal near T_at_best
    bs_, bres_ = min(feas, key=lambda x: x[1]['cost'])
    best_loads = State(d, bres_['solution']).cpu
    first_improve = {}
    for s_, res in feas:
        tr0 = res['trace'][0][1]
        first_improve[s_] = next((e for e, v in res['trace'] if v < tr0 - 1e-12), None)

    # ---- random search of the partner (question 6)
    base = None
    if a.baseline and os.path.exists(a.baseline):
        brows = [r for r in read_runs(a.baseline) if r['configuration'].strip().lower() == 'random' and r['feasible'] == 'True']
        if brows:
            bc = [float(r['cost']) for r in brows]
            bs = summary(bc)
            base = {k: v for k, v in bs.items()}
            base.update({'n': len(bc), 'gap_ref_mean': gap_ref(bs['mean']), 'gap_ref_best': gap_ref(bs['best']),
                         'gap_ref_worst': gap_ref(bs['worst']), 'gap_LP_mean': gap_lp(bs['mean']),
                         'gap_LP_best': gap_lp(bs['best']), 'gap_LP_worst': gap_lp(bs['worst']),
                         'file': a.baseline})

    numbers = {'z_ref': Z_REF, 'z_LP': Z_LP, 'budget': budget, 'n_runs': len(runs), 'n_feasible': n_feas,
               'costs_by_seed': {s: res['cost'] for s, res in feas}, 'summary': sm,
               'gaps': {n: {'ref': gap_ref(v), 'LP': gap_lp(v)} for n, v in t1}, 'mean_time': mean_time,
               'worst_seed': ws, 'improving_transfers': n_tr, 'improving_swaps': n_sw,
               'best_transfer_delta': best_tr, 'best_swap_delta': best_sw,
               'first5': st['first5'], 'last5': st['last5'], 'first5_share': f5, 'last5_share': l5,
               'end_state_improving_transfers': end_tr, 'end_state_improving_swaps': end_sw,
               'best_found_at_eval_worst': st['best_at_eval'], 'T_at_best_worst': st['T_at_best'],
               'current_cost_end_worst': st['current_cost_end'],
               'transfers_list_worst': tr_list, 'swaps_list_worst': sw_list,
               'worst_loads_cpu': sstate.cpu, 'worst_cpu_cap': sstate.cpu_cap,
               'T0_worst': st['T0'], 'Tend_worst': st['Tend'],
               'T0_all': {s: res['stats']['T0'] for s, res in feas}, 'Tend_all': {s: res['stats']['Tend'] for s, res in feas},
               'flat_at_evals_mean_curve': flat_at, 'last_improvement_by_seed': last_improve,
               'mean_best_so_far_at_90pct': at90, 'mean_best_so_far_final': float(final_mean),
               'infeasible_proposals_share': {s: res['stats']['infeasible_proposals'] / res['stats']['K'] for s, res in feas},
               'start_cost_mean': float(np.mean([res['trace'][0][1] for s, res in feas])),
               'other_parameter_sets': alt_summary, 'acceptance_at_T': acc_at, 'p_improving_transfer_per_proposal': p_improve,
               'p_leave_per_proposal_near_best': p_leave, 'best_seed': bs_, 'best_seed_loads_cpu': best_loads,
               'first_improvement_by_seed': first_improve,
               'n_transfer_candidates': sum(len(a_) - 1 for a_ in allowed), 'baseline': base}
    with open(os.path.join(a.outdir, 'Mutali_numbers.json'), 'w') as fh:
        json.dump(numbers, fh, indent=1, default=float)

    # ---- figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 2.9), dpi=200)
    ax.fill_between(grid, lo, hi, color='#9ecae1', alpha=0.6, label='min-max of %d runs' % n_feas)
    ax.plot(grid, mean_tr, color='#08519c', lw=1.4, label='mean best-so-far')
    ax.axhline(Z_REF, color='#d94801', ls='--', lw=1, label=r'$z_{ref}$ = %.6f' % Z_REF)
    ax.axhline(Z_LP, color='#238b45', ls=':', lw=1.2, label=r'$z_{LP}$ = %.6f' % Z_LP)
    ax.set_xlabel('evaluations'); ax.set_ylabel('best feasible cost so far')
    ax.set_ylim(Z_LP - 0.01, Z_LP + 0.42)          # the curves start at the cost of the start (about 7.96), above the window
    ax.set_xlim(0, budget); ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(a.figdir, 'F1_Mutali.png')); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 2.9), dpi=200)
    for s, res in feas:
        cv = [(T, 0 if w == 0 else acc / w) for T, w, acc in res['stats']['curve'] if w > 0]
        if s == ws:
            ax.plot([T for T, _ in cv], [v for _, v in cv], color='#d94801', lw=1.6, label='worst seed %d' % ws, zorder=5)
        else:
            ax.plot([T for T, _ in cv], [v for _, v in cv], color='#6baed6', lw=0.7, alpha=0.8,
                    label='other seeds' if s == min(x for x, _ in feas if x != ws) else None)
    ax.set_xscale('log'); ax.invert_xaxis()
    ax.set_xlabel('temperature T (cooling goes to the right)'); ax.set_ylabel('accepted / worsening proposals')
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.figdir, 'F2_Mutali_acceptance.png')); plt.close(fig)

    # ---- print
    print('feasible %d/%d, mean time %.2f s' % (n_feas, len(runs), mean_time))
    for n, v in t1:
        print('%-7s %.6f  gap_ref %.4f %%  gap_LP %.4f %%' % (n, v, gap_ref(v), gap_lp(v)))
    print('std (n-1) %.6f' % sm['std'])
    print('Table 2: worst seed %d cost %.6f, improving transfers %d, swaps %d (best deltas %.6f / %.6f)'
          % (ws, wres['cost'], n_tr, n_sw, best_tr, best_sw))
    print('accepted worsening: first 5%% %d/%d = %.4f, last 5%% %d/%d = %.4f' % (st['first5'][0], st['first5'][1], f5, st['last5'][0], st['last5'][1], l5))
    print('mean curve within 1e-3 of its final value from %d evals; last improvement by seed %s' % (flat_at, last_improve))
    print('baseline:', base if base else 'PENDING (no --baseline file with configuration random)')


if __name__ == '__main__':
    main()
