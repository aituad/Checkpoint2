"""M2 entry point: LP relaxation of the T16 instance (Task 2).

    python3 -m experiments.run_m2

Writes results/m2_lp.csv, results/m2_fractional.csv and results/m2_solution.npz.
"""
from src.lp.run_m2 import main

if __name__ == "__main__":
    main()
