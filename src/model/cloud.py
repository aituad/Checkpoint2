"""Reference B3 LP matrices. Bounds are not constraint rows."""
from dataclasses import dataclass
import numpy as np
from checkpoint3_generator_PUBLIC import cp34_instance, TANGENT_POINTS

@dataclass
class MatrixModel:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    senses: tuple
    lower: np.ndarray
    upper: np.ndarray
    columns: tuple
    rows: tuple
    families: tuple
    data: dict

    def y_index(self, k, l):
        return k * int(self.data['S']) + l

    def t_index(self, l):
        return int(self.data['M']) * int(self.data['S']) + l


def build_matrices(data=None):
    d = cp34_instance(16, 'B', 3) if data is None else data
    if d.get('story') != 'B' or d.get('variant') != 3:
        raise ValueError('This builder implements reference variant B3 only.')
    S, M = int(d['S']), int(d['M'])
    n = M*S + S
    c = np.r_[np.zeros(M*S), np.ones(S)]
    lower, upper = np.zeros(n), np.r_[np.ones(M*S), np.full(S, np.inf)]
    columns = tuple([f'y_{k}_{l}' for k in range(M) for l in range(S)] + [f't_{l}' for l in range(S)])
    A, b, senses, rows, families = [], [], [], [], []
    def add(row, rhs, sense, name, family):
        A.append(row); b.append(float(rhs)); senses.append(sense)
        rows.append(name); families.append(family)
    for k in range(M):
        row = np.zeros(n); row[k*S:(k+1)*S] = 1
        add(row, 1, '=', f'assign_{k}', 'B1_assignment')
        for l in range(S):
            if d['needs_gpu'][k] and not d['is_gpu_server'][l]:
                upper[k*S+l] = 0
    for l in range(S):
        row = np.zeros(n); row[l:M*S:S] = d['cpu_req']
        add(row, d['cpu_cap'][l], '<=', f'cpu_{l}', 'B2_CPU')
    for l in range(S):
        row = np.zeros(n); row[l:M*S:S] = d['mem_req']
        add(row, d['mem_cap'][l], '<=', f'memory_{l}', 'B3_memory')
    for l in range(S):
        for m, u in enumerate(TANGENT_POINTS):
            a, intercept = 1/(1-u)**2, -u*u/(1-u)**2
            row = np.zeros(n)
            row[l:M*S:S] = a*np.asarray(d['cpu_req'])/d['cpu_cap'][l]
            row[M*S+l] = -1
            # a * utilization - t <= -intercept (equivalent to B4).
            add(row, -intercept, '<=', f'latency_{l}_{m}', 'B4_latency')
    return MatrixModel(c, np.asarray(A), np.asarray(b), tuple(senses), lower, upper,
                       columns, tuple(rows), tuple(families), d)


def build_pulp_model(data=None, matrix=None):
    import pulp
    mm = build_matrices(data) if matrix is None else matrix
    model = pulp.LpProblem('T16_B3_LP', pulp.LpMinimize)
    variables = [pulp.LpVariable(name, lowBound=float(mm.lower[j]),
                 upBound=None if np.isinf(mm.upper[j]) else float(mm.upper[j]),
                 cat=pulp.LpContinuous) for j, name in enumerate(mm.columns)]
    model += pulp.lpSum(float(mm.c[j])*variables[j] for j in range(len(variables)))
    for i, name in enumerate(mm.rows):
        expr = pulp.lpSum(float(mm.A[i,j])*variables[j] for j in np.flatnonzero(mm.A[i]))
        model += (expr == float(mm.b[i])) if mm.senses[i] == '=' else (expr <= float(mm.b[i])), name
    S, M = int(mm.data['S']), int(mm.data['M'])
    y = {(k,l): variables[mm.y_index(k,l)] for k in range(M) for l in range(S)}
    t = {l: variables[mm.t_index(l)] for l in range(S)}
    return model, y, t
