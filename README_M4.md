M4 Sensitivity Analysis
Purpose
This script performs the M4 sensitivity analysis for the Cloud Resource Allocation model.

The right-hand side (RHS) of the selected B1 assignment constraint is perturbed by ±0.01, and the resulting LP objective values are recorded.

Method
The script solves the LP relaxation three times:

RHS = 0.99
RHS = 1.00
RHS = 1.01
For each run it records:

objective value
dual price of the selected assignment constraint
The results are saved to:

results/perturbation_runs.csv
Output
Example output:

rhs	objective	dual_price
0.99	...	...
1.00	...	...
1.01	...	...
These values are used in Task M4 to compute:

Left Slope
Dual Price (π)
Right Slope
and to verify the LP sensitivity relationship.

Running the script
From the project root directory:

python perturbation_run.py
or

python sens/perturbation_run.py 
depending on the project structure.

Files Used
The script depends on:

src/model/cloud.py
src/lp/solve.py

libraries: 
numpy
pandas
