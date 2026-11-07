"""B3 problem operators for GA penalty. NumPy Generator supplied by GA-A.
No population loop, reference values or global random state.
"""
import numpy as np


def _valid(d, individual):
    a = np.asarray(individual)
    if a.shape != (int(d['M']),) or not np.issubdtype(a.dtype, np.integer):
        raise ValueError('individual must be a one-dimensional integer assignment of length M')
    if np.any(a < 0) or np.any(a >= int(d['S'])):
        raise ValueError('server index outside [0,S)')
    return a


def allowed_servers(d, k):
    if d.get('story') != 'B' or int(d.get('variant',0)) != 3:
        raise ValueError('These operators implement B3 only')
    servers = np.arange(int(d['S']))
    if d['needs_gpu'][k]:
        servers = servers[np.asarray(d['is_gpu_server']) == 1]
    if not len(servers):
        raise ValueError('GPU VM has no compatible server; no compatible individual exists')
    return servers


def random_individual(d, rng):
    """Uniform compatible server per VM; capacity violations are permitted."""
    return np.array([rng.choice(allowed_servers(d,k)) for k in range(int(d['M']))],dtype=int)


def evaluate(d, individual, rho=10.0):
    """One candidate cost evaluation. Return fitness, cost, feasible, violation.
    V = sum normalized CPU excess + sum normalized memory excess
        + number of GPU-incompatible VMs. Cost matches official checker,
    including its utilization clipping for infeasible CPU-overloaded states.
    """
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError('rho must be finite and positive')
    if d.get('story') != 'B' or int(d.get('variant',0)) != 3:
        raise ValueError('These operators implement B3 only')
    a = _valid(d,individual)
    S=int(d['S'])
    C=np.asarray(d['cpu_cap'],dtype=float); R=np.asarray(d['mem_cap'],dtype=float)
    if np.any(C <= 0) or np.any(R <= 0): raise ValueError('capacities must be positive')
    cpu=np.bincount(a,weights=d['cpu_req'],minlength=S)
    mem=np.bincount(a,weights=d['mem_req'],minlength=S)
    gpu_bad=(np.asarray(d['needs_gpu']) == 1) & (np.asarray(d['is_gpu_server'])[a] == 0)
    violation=float(np.maximum(cpu-C,0).dot(1/C)+np.maximum(mem-R,0).dot(1/R)+gpu_bad.sum())
    u=np.minimum(cpu/C,1.0)
    points=np.array([0.,.4,.7,.9])
    slopes=1/(1-points)**2; intercepts=-points**2/(1-points)**2
    cost=float(np.max(u[:,None]*slopes+intercepts,axis=1).sum())
    feasible=bool(np.all(cpu <= C+1e-9) and np.all(mem <= R+1e-9) and not gpu_bad.any())
    return dict(fitness=cost+float(rho)*violation,cost=cost,feasible=feasible,violation=violation)


def crossover(d, parent1, parent2, rng):
    """Uniform crossover, returning ONE new child without changing parents.
    GA-A owns the crossover probability (default 0.9); call this when selected.
    """
    a,b=_valid(d,parent1),_valid(d,parent2)
    return np.where(rng.random(int(d['M'])) < .5,a,b).astype(int)


def mutate(d, individual, rng, n_changes=None):
    """Return a copy; change 1 or 2 distinct movable VM assignments.
    If fewer VMs have alternative compatible servers, change those available.
    Never repairs CPU/memory overloads and never evaluates the objective.
    """
    a=_valid(d,individual).copy()
    count=int(rng.integers(1,3)) if n_changes is None else n_changes
    if count not in (1,2): raise ValueError('n_changes must be 1 or 2')
    alternatives=[allowed_servers(d,k)[allowed_servers(d,k) != a[k]] for k in range(int(d['M']))]
    movable=[k for k,choices in enumerate(alternatives) if len(choices)]
    if movable:
        for k in rng.choice(movable,size=min(count,len(movable)),replace=False):
            a[k]=rng.choice(alternatives[k])
    return a
