"""GA-P penalty experiment; uses Miras's GA-A without changing it.
Run as: python -m experiments.run_bekturova_exp
Default: tuning seeds 100..104, 100000 evaluations, rho=1,10,100.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from checkpoint4_generator_PUBLIC import cp34_instance, evaluate_B
from src.meta.ga import solve

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['team','role','configuration','seed','budget','cost','evals','feasible','seconds','params','offspring_evaluated','offspring_feasible','feasible_offspring_share']


def save(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--budget',type=int,default=100000)
    ap.add_argument('--smoke',action='store_true',help='Use 300 evaluations and seed 100; separate outputs, not report results')
    args=ap.parse_args()
    budget=300 if args.smoke else args.budget
    if budget < 30: ap.error('budget must be at least 30')
    seeds=[100] if args.smoke else list(range(100,105))
    suffix='_smoke' if args.smoke else ''
    out=ROOT/'results'/('bekturova_exp'+suffix+'.csv')
    d=cp34_instance(16,'B',3)
    rows=[]
    for rho in [1.,10.,100.]:
        params=dict(N=30,pc=.9,tournament_size=2,rho=rho)
        for seed in seeds:
            print(f'rho={rho:g}, seed={seed}, budget={budget}: running...',flush=True)
            started=time.perf_counter()
            result=solve(d,3,seed,budget,params=params)
            seconds=time.perf_counter()-started
            checked_cost,problems=evaluate_B(d,result['solution'])
            if int(result['evals']) != budget: raise AssertionError('GA spent an incorrect budget')
            if bool(result['feasible']) != (not problems): raise AssertionError('GA feasible flag disagrees with checker')
            if not np.isclose(checked_cost,result['cost'],rtol=0,atol=1e-9): raise AssertionError('GA cost disagrees with checker')
            count=int(result['offspring_evaluated']);feas=int(result['offspring_feasible'])
            if count != budget-params['N'] or not 0<=feas<=count: raise AssertionError('Invalid offspring counters')
            row=dict(team='T16',role='GA-P',configuration='GA, penalty',seed=seed,budget=budget,
                     cost=checked_cost,evals=budget,feasible=not problems,seconds=seconds,
                     params=json.dumps(params,sort_keys=True),offspring_evaluated=count,offspring_feasible=feas,
                     feasible_offspring_share=feas/count if count else '')
            rows.append(row)
            save(out,rows,FIELDS)
            print(f"  cost={checked_cost:.9f}, feasible={not problems}, offspring share={feas/count:.4f}; saved {out.name}",flush=True)
    summary=[]
    for rho in [1.,10.,100.]:
        group=[r for r in rows if json.loads(r['params'])['rho']==rho]
        costs=[r['cost'] for r in group if r['feasible']]
        summary.append(dict(rho=rho,runs=len(group),feasible_runs=len(costs),infeasible_run_share=1-len(costs)/len(group),
                            feasible_offspring_share=sum(r['offspring_feasible'] for r in group)/sum(r['offspring_evaluated'] for r in group),
                            mean_feasible_cost=float(np.mean(costs)) if costs else '',
                            best_feasible_cost=min(costs) if costs else '',worst_feasible_cost=max(costs) if costs else ''))
    save(ROOT/'results'/('bekturova_penalty_summary'+suffix+'.csv'),summary,list(summary[0]))
    print('Experiment complete. Smoke results must not be used in the report.' if args.smoke else 'All 15 tuning runs complete. Review summary before final seeds 0–9.',flush=True)


if __name__=='__main__':main()
