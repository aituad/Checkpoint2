import json
import os
import numpy as np
import matplotlib.pyplot as plt


def generate_plots():
    # Configuration
    BUDGET = 100000
    Z_LP = 7.328502
    Z_REF = 7.345083

    # 1. Define reliable paths independent of the terminal's working directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')

    # Ensure that the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, 'T16_GA_traces_assignments.json')
    f1_path = os.path.join(results_dir, 'Figure_F1.png')
    div_path = os.path.join(results_dir, 'Diversity_plot.png')

    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found! Run export_json.py first.")
        return

    # Load experimental data
    with open(json_path, 'r') as f:
        data = json.load(f)

    ga_traces = []
    ga_diversities = []

    for i in range(10):
        seed_data = data[str(i)]
        ga_traces.append(seed_data["trace"])

        if "diversity" in seed_data:
            ga_diversities.append(seed_data["diversity"])

    plt.style.use('seaborn-v0_8-whitegrid')

    # ==========================================
    # 1. FIGURE F1: CONVERGENCE (TRACE)
    # ==========================================
    print("Generating Figure F1 (Convergence)...")

    x_evals = np.arange(1000, BUDGET + 1, 1000)
    y_costs = np.zeros((10, len(x_evals)))

    for i, trace in enumerate(ga_traces):
        trace_dict = {e: c for e, c in trace}
        last_val = float('nan')

        for j, x in enumerate(x_evals):
            if x in trace_dict:
                last_val = trace_dict[x]

            y_costs[i, j] = last_val

    # Replace missing values with the largest observed value
    max_val_global = np.nanmax(y_costs)
    y_costs = np.nan_to_num(y_costs, nan=max_val_global)

    mean_line = np.mean(y_costs, axis=0)
    min_line = np.min(y_costs, axis=0)
    max_line = np.max(y_costs, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_evals,
        mean_line,
        label='Mean Best-so-Far (GA)',
        color='blue',
        linewidth=2
    )
    plt.fill_between(
        x_evals,
        min_line,
        max_line,
        color='blue',
        alpha=0.2,
        label='Min-Max Band'
    )
    plt.axhline(
        Z_REF,
        color='red',
        linestyle='--',
        label=f'z_ref ({Z_REF})'
    )
    plt.axhline(
        Z_LP,
        color='green',
        linestyle='-.',
        label=f'z_LP ({Z_LP})'
    )

    plt.title("Figure F1: Convergence (Best-so-Far vs. Evaluations)")
    plt.xlabel("Evaluations")
    plt.ylabel("Cost")
    plt.legend()

    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 2. POPULATION DIVERSITY PLOT
    # ==========================================
    if ga_diversities:
        print("Generating Population Diversity Plot...")

        plt.figure(figsize=(10, 6))

        max_len = max(len(div) for div in ga_diversities)
        padded_divs = []

        for div in ga_diversities:
            # Extend shorter arrays using their last recorded value
            padded = div + [div[-1]] * (max_len - len(div))
            padded_divs.append(padded)

            # Plot an individual line for each seed
            plt.plot(
                range(max_len),
                padded,
                color='purple',
                alpha=0.15
            )

        mean_div = np.mean(padded_divs, axis=0)

        plt.plot(
            range(max_len),
            mean_div,
            color='purple',
            linewidth=2,
            label='Mean Diversity'
        )

        plt.title("Population Diversity vs. Generation (Role: GA-A)")
        plt.xlabel("Generation")
        plt.ylabel("Unique Individuals in Population")
        plt.legend()

        plt.savefig(div_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots successfully saved in the directory: {results_dir}")


if __name__ == "__main__":
    generate_plots()