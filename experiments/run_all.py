import sys
from pathlib import Path

from experiments import run_m1
from experiments import run_m2
from experiments import run_m3
from experiments import run_m4

def main():
    print("="*50)
    print("Starting Checkpoint 3 Full Regeneration")
    print("Team T16 (Storyline B, Variant 3, Seed 16)")
    print("="*50)

    try:
        print("\n[1/4] Running M1 (Model Builder) ...")
        run_m1.main()
        
        print("\n[2/4] Running M2 (LP Solver) ...")
        run_m2.main()
        
        print("\n[3/4] Running M3 (KKT Checker) ...")
        run_m3.main()
        
        print("\n[4/4] Running M4 (Sensitivity Analysis) ...")
        run_m4.run_perturbations()
        
        print("\n" + "="*50)
        print("SUCCESS: All Checkpoint 3 experiments completed.")
        print("Data for Tables 1, 2, and 3 are ready in the results/ directory.")
        print("="*50)

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()