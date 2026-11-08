import numpy as np
from .ga_problem import random_individual, evaluate, crossover, mutate

def tournament_select(fitnesses, k, rng):
    """Return the index of the best (lowest fitness) among k distinct, randomly chosen entrants."""
    idxs = rng.choice(len(fitnesses), k, replace=False)
    best_idx = idxs[0]
    for idx in idxs[1:]:
        if fitnesses[idx] < fitnesses[best_idx]:
            best_idx = idx
    return best_idx


def solve(d, variant, seed, budget, params=None, start=None):
    # Using the NumPy generator, as required by the GA-P interface.
    rng = np.random.default_rng(seed)
    
    if params is None:
        params = {}
    pop_size = params.get("N", 30)
    pc = params.get("pc", 0.9)
    k_tourn = params.get("tournament_size", 2)
    rho = params.get("rho", 10.0)
    
    evals = 0
    trace = []
    best_feasible_cost = float('inf')
    best_feasible_sol = None
    diversity_history = []
    
    # Metrics for the GA-P report
    offspring_evaluated = 0
    offspring_feasible = 0
    
    population = []
    fitnesses = []
    costs = []
    feasibilities = []
    
    # 1. Initialization
    for _ in range(pop_size):
        if evals >= budget: break
        ind = random_individual(d, rng) if start is None else np.array(start)
        res = evaluate(d, ind, rho=rho)
        population.append(ind)
        fitnesses.append(res['fitness'])
        costs.append(res['cost'])
        feasibilities.append(res['feasible'])
        evals += 1
        
        if res['feasible'] and res['cost'] < best_feasible_cost:
            best_feasible_cost = res['cost']
            best_feasible_sol = ind.copy()
            
    def get_diversity(pop):
        return len(set(tuple(x.tolist()) for x in pop))
        
    best_fitness_history = []   # lowest fitness in the population after each generation (used by the elitism test)
    if evals > 0:
        diversity_history.append(get_diversity(population))
        best_fitness_history.append(min(fitnesses))
    
    # Tournament selection: see tournament_select (module level, so it can be unit-tested)
    def tournament():
        return tournament_select(fitnesses, k_tourn, rng)

    # 2. Main Loop
    while evals < budget:
        next_pop, next_fit, next_cost, next_feas = [], [], [], []
        
        # Elitism: the best individual moves across for free
        best_idx = np.argmin(fitnesses)
        next_pop.append(population[best_idx].copy())
        next_fit.append(fitnesses[best_idx])
        next_cost.append(costs[best_idx])
        next_feas.append(feasibilities[best_idx])
        
        while len(next_pop) < pop_size and evals < budget:
            p1_idx = tournament()
            p2_idx = tournament()
            
            # Crossover with probability pc; otherwise, copying one of the parents.
            if rng.random() < pc:
                child = crossover(d, population[p1_idx], population[p2_idx], rng)
            else:
                child = population[p1_idx].copy()
                
            child = mutate(d, child, rng)
            res = evaluate(d, child, rho=rho)
            
            evals += 1
            offspring_evaluated += 1
            
            if res['feasible']:
                offspring_feasible += 1
                if res['cost'] < best_feasible_cost:
                    best_feasible_cost = res['cost']
                    best_feasible_sol = child.copy()
                    
            if evals % 1000 == 0 and best_feasible_cost != float('inf'):
                trace.append((evals, best_feasible_cost))
                
            next_pop.append(child)
            next_fit.append(res['fitness'])
            next_cost.append(res['cost'])
            next_feas.append(res['feasible'])
            
        population, fitnesses = next_pop, next_fit
        costs, feasibilities = next_cost, next_feas
        diversity_history.append(get_diversity(population))
        best_fitness_history.append(min(fitnesses))

    # Formulating Answer
    if best_feasible_sol is None:
        best_idx = np.argmin(fitnesses)
        final_sol = population[best_idx].tolist()
        final_cost = costs[best_idx]
        final_feas = False
    else:
        final_sol = best_feasible_sol.tolist()
        final_cost = best_feasible_cost
        final_feas = True
        
    return {
        "solution": final_sol,
        "cost": float(final_cost),
        "evals": evals,
        "feasible": final_feas,
        "trace": trace,
        "diversity": diversity_history,
        "best_fitness_history": best_fitness_history,
        "offspring_evaluated": offspring_evaluated,
        "offspring_feasible": offspring_feasible
    }