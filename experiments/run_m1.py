"""Export M1 matrices, bookkeeping and H1 enumeration; does not solve the LP."""
import csv
from pathlib import Path
import numpy as np
from src.model.cloud import build_matrices, build_pulp_model
from src.model.triangle import enumerate_binary

def write(path, fields, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(fields); w.writerows(rows)

def main():
    out=Path(__file__).resolve().parents[1]/'results';out.mkdir(exist_ok=True)
    m=build_matrices(); model,y,t=build_pulp_model(matrix=m)
    np.savez(out/'m1_matrices.npz',c=m.c,A=m.A,b=m.b,lower=m.lower,upper=m.upper,
             senses=np.array(m.senses),rows=np.array(m.rows),columns=np.array(m.columns),families=np.array(m.families))
    write(out/'m1_rows.csv',['index','name','family','sense','rhs'],
          [(i,m.rows[i],m.families[i],m.senses[i],m.b[i]) for i in range(len(m.b))])
    write(out/'m1_columns.csv',['index','name','lower','upper','objective'],
          [(j,m.columns[j],m.lower[j],m.upper[j],m.c[j]) for j in range(len(m.c))])
    summary=[('variables',len(model.variables())),('rows',len(model.constraints)),
             ('forbidden_pairs',sum(v.upBound==0 for v in y.values())),
             ('B1_assignment',41),('B2_CPU',9),('B3_memory',9),('B4_latency',36)]
    write(out/'m1_model_sizes.csv',['item','value'],summary)
    write(out/'h1_binary_points.csv',['y1','y2','y3','feasible','cost'],
          [(*p,ok,cost) for p,ok,cost in enumerate_binary()])
    model.writeLP(str(out/'T16_B3_relaxation.lp'))
    print(dict(summary));print('M1 artifacts saved in results/. LP solution is M2 work.')

if __name__=='__main__': main()
