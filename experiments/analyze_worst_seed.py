import sys
import os
import csv

# Add the parent directory to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from checkpoint4_generator_PUBLIC import cp34_instance, evaluate_B
from src.meta.ga import solve

def analyze():
    TEAM_SEED = 16
    VARIANT = 3
    STORY = 'B'
    BUDGET = 100000

    # Worst seed from Table 1
    WORST_SEED = 4  
    RHO = 10.0

    d = cp34_instance(TEAM_SEED, STORY, VARIANT)
    params = {"N": 30, "pc": 0.9, "tournament_size": 2, "rho": RHO}

    print(f"Running GA for the worst seed ({WORST_SEED})...")
    res = solve(d, VARIANT, WORST_SEED, BUDGET, params=params)

    assign = res["solution"]
    base_cost = res["cost"]
    
    # 1. Diversity
    div_history = res["diversity"]
    div_first = div_history[0]
    div_last = div_history[-1]

    improving_transfers = 0
    improving_swaps = 0

    print("Analyzing Transfers...")
    # 2. Check all possible Transfer moves
    for vm in range(d['M']):
        original_server = assign[vm]
        for target_server in range(d['S']):
            if target_server != original_server:
                new_assign = list(assign)
                new_assign[vm] = target_server
                cost, problems = evaluate_B(d, new_assign)
                if len(problems) == 0 and cost < base_cost - 1e-9:
                    improving_transfers += 1

    print("Analyzing Swaps...")
    # 3. Check all possible Swap moves
    for vm1 in range(d['M']):
        for vm2 in range(vm1 + 1, d['M']):
            if assign[vm1] != assign[vm2]:
                new_assign = list(assign)
                new_assign[vm1], new_assign[vm2] = new_assign[vm2], new_assign[vm1]
                cost, problems = evaluate_B(d, new_assign)
                if len(problems) == 0 and cost < base_cost - 1e-9:
                    improving_swaps += 1

    if improving_transfers == 0 and improving_swaps == 0:
        conclusion = "Local optimum with respect to BOTH neighborhoods."
    elif improving_transfers == 0 and improving_swaps > 0:
        conclusion = "Local optimum for Transfer, but Swap improves it."
    else:
        conclusion = "Stopped BEFORE reaching local optimum (budget limit)."
        
    print("-" * 40)
    print(f"Baseline cost (Seed {WORST_SEED}): {base_cost:.6f}")
    print(f"Unique individuals (Gen 1):  {div_first}")
    print(f"Unique individuals (Final):  {div_last}")
    print(f"Improving Transfers:         {improving_transfers}")
    print(f"Improving Swaps:             {improving_swaps}")
    print(f"Conclusion:                  {conclusion}")
    print("-" * 40)

    # 4. Keeping in .csv
    os.makedirs('../results', exist_ok=True)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'Table2_worst_seed_analysis.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Worst_Seed", "Base_Cost", "Div_Gen_1", "Div_Gen_Last", "Improving_Transfers", "Improving_Swaps", "Conclusion"])
        writer.writerow([WORST_SEED, f"{base_cost:.6f}", div_first, div_last, improving_transfers, improving_swaps, conclusion])
        
    print(f" >The data for Table 2 are securely stored in {csv_path}")

if __name__ == "__main__":
    analyze()