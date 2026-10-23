# Checkpoint 2 — Team T16

## Unconstrained Optimization and Cloud Load Balancing

- **Section:** 4
- **Storyline:** B — Cloud
- **Variant:** 3
- **Seed:** 16
- **Submission tag:** `checkpoint2`

This project compares six optimization configurations on Rosenbrock, two quadratic problems, and a continuous cloud load-balancing model based on our Checkpoint 1 data.

## Team Roles

| Role | Member | Code responsibilities | Analysis | Hand trace |
|---|---|---|---|---|
| M1 — GD | Adil Mutali | `src/optim/gd.py`; `src/problems/rosenbrock.py`; gradient checking | S1 — Step size and stability | H1 |
| M2 — Newton | Dilnaz Bekturova | `src/optim/newton.py`; `src/problems/quadratic.py` | S2 — Conditioning, cost and safeguards | H2 |
| M3 — Momentum | Alina Suleimenova | `src/optim/momentum.py`; `src/problems/project.py` | S3 — Momentum in the valley | H3 |
| M4 — Adam | Miras Asem | `src/optim/adam.py`; experiment runner and tables | S4 — Adam and the rotated axes | H4 |
| Group | All four members | Shared experiments, figures and report assembly | S5 and synthesis | H5 |



## Environment

The project uses Python 3, NumPy and Matplotlib. Pytest is included for test execution.

Dependencies in `requirements.txt`:

- `numpy>=1.24`
- `matplotlib>=3.7`
- `pytest>=7`

The solver implementations use NumPy and the Python standard library. Matplotlib generates the figures.

## Environment

The project uses Python 3, NumPy for numerical computations, and Matplotlib for plotting. Solver implementations use NumPy and the Python standard library.

Dependencies are installed directly using pip.

## Installation and Execution

Run the following commands from the repository root.

Create and activate a virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required libraries:

```bash
python3 -m pip install numpy matplotlib
```

Run the experiment script located in `src/experiments/run_all.py`:

```bash
python3 -m src.experiments.run_all
```

Generated tables, numerical results and figures are saved in `results/`. Open the saved image files to view the graphs.

To inspect the environment used for the experiments:

```bash
python3 --version
python3 -m pip show numpy matplotlib
```
## Repository Structure

```text
src/
  optim/
    gd.py
    newton.py
    momentum.py
    adam.py
  problems/
    rosenbrock.py
    quadratic.py
    project.py
    check_grad.py
  experiments/
    run_all.py
    plots.py

experiments/
  run_all.py

tests/
results/
hand/
report/
requirements.txt
README.md
```

- `src/optim/`: optimization methods.
- `src/problems/`: objectives and analytical derivatives.
- `src/experiments/`: experiment orchestration and plotting.
- `tests/`: verification of implementations and numerical results.
- `results/`: CSV data behind the report tables and generated figures.
- `hand/`: signed hand traces.
- `report/`: report source and submission materials.

## Methods and Experimental Protocol

The six configurations are:

1. Gradient Descent with a fixed step.
2. Gradient Descent with Armijo backtracking.
3. Pure Newton.
4. Damped Newton with Armijo backtracking.
5. Momentum.
6. Adam.

R1 and R2 use the absolute stopping rule:

```text
||gradient f(x_k)|| < 1e-6
```

Q1, Q2 and the project use the relative stopping rule:

```text
||gradient f(x_k)|| < 1e-6 * ||gradient f(x_0)||
```

Iteration counts represent completed updates. Gradient methods have a limit of 100,000 updates; Newton methods have a limit of 100.

The initial step-size grid is:

```text
alpha = 10^(-5 + 0.5*j), j = 0, ..., 10
```

Momentum also tests:

```text
beta ∈ {0.5, 0.8, 0.9, 0.95, 0.99}
```

GD additionally tests `alpha = 1/130` on Q1 and Q2. Upper grid boundaries are extended by half-decades when required, up to `1e4`. Successful runs are ranked by update count; ties use the smaller alpha, then the smaller beta.

## Benchmark Results — Table 1

| Problem | GD | GD-BT | Pure Newton | Damped Newton | Momentum | Adam |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 32076 | 13756 | 6 | 21 | 857 | 1034 |
| R2 | 32794 | 17291 | 7 | 22 | 1117 | 1219 |
| Q1 | 892 | 39 | 1 | 1 | 120 | 186 |
| Q2 | 892 | 39 | 1 | 1 | 120 | 213 |

Selected parameters and experiment data are recorded in the generated CSV files and the report.

## Cloud Project — Table 2

The model uses 41 VM CPU requests with total demand `D = 201` and nine server capacities:

```text
C = (59, 46, 57, 35, 49, 43, 32, 52, 47)
```

The objective is:

```text
J(w) = (201² / 9) * sum_j(w_j² / C_j²)
```

Shares are parametrized by `softmax(z_1, ..., z_8, 0)`, starting from eight zero logits.

The closed-form optimum is:

```text
w*_j = C_j² / 20258
J* = 0.2215914700
```

| Method | Updates | Reaches the known optimum numerically? |
|---|---:|---|
| GD | 175 | Yes |
| GD-BT | 1805 | Yes |
| Pure Newton | 7 | Yes |
| Damped Newton | 6 | Yes |
| Momentum — prescribed selection | 2 | No |
| Adam | 247 | Yes |

The two-update Momentum run satisfies the gradient stopping rule because softmax saturates, but its objective is approximately `1.381655894`. A small logit gradient alone does not establish optimality.

A supplementary Momentum run reaches the known optimum in 42 updates with `alpha = 10^1.5` and `beta = 0.5`. This accuracy-verified comparison is reported separately and does not replace the prescribed Table 2 selection.

## Figures

- **F1:** Rosenbrock R1 trajectories for all six configurations.
- **F2:** Gradient-norm convergence on Q1 and Q2.
- **F3:** Gradient-norm convergence on the cloud project.

These figures are generated by `src/experiments/plots.py` through the common runner and saved in `results/`.

## Declarations

### Adil Mutali — M1

AI assistance was used for explanations, implementation and documentation drafting, debugging guidance, and numerical verification of the M1 contribution.

### Dilnaz Bekturova — M2

AI assistance was used for implementation and documentation drafting, debugging, numerical verification, and preparation of the S2 analysis.

### Alina Suleimenova - M3

AI tools were used to support explanations, code implementation, documentation preparation, debugging, and numerical validation of the M3 contribution.

### Miras Asem - M4

AI tools were utilized to support explanations, implementation, documentation preparation, debugging, and numerical verification of the M4 contribution.

### Shared Work

AI assistance was used to draft `src/problems/project.py`, assist with the common experiment runner and verification, and prepare the LaTeX report and this README.

The cloud data and problem conventions come from the course task files and Team T16's Checkpoint 1 materials. The solver implementations do not use an external optimization solver library.


# Checkpoint 3 — Team T16

## Constrained Optimization and Cloud Resource Allocation

- **Section:** 4
- **Storyline:** B — Cloud Resource Allocation
- **Variant:** 3 — GPU compatibility
- **Seed:** 16
- **Submission tag:** `checkpoint3`
- **Deadline:** October 23, 2026, 23:59
- **Repository:** https://github.com/aituad/Checkpoint2

This checkpoint studies the prescribed B3 reference model for placing virtual machines on servers subject to CPU, memory and GPU-compatibility restrictions.

The team solves the LP relaxation, examines fractional assignments, checks LP optimality conditions and studies sensitivity to a designated right-hand-side perturbation. The LP optimum provides a lower bound for the integer reference model used in Checkpoint 4.

Checkpoint 3 uses the existing repository from Checkpoint 2. Previous checkpoint files and history are retained.

## Team Roles

| Role | Member | Code responsibilities | Analysis | Hand task |
|---|---|---|---|---|
| M1 — Model | Dilnaz Bekturova | `src/model/`: data, reference model, matrices and row/column bookkeeping | S1 — The model and what it ignores | H1 |
| M2 — LP | Adil Mutali | `src/lp/`: solving the LP and reading its solution | S2 — Value and fractionality of the LP | H2 |
| M3 — KKT | Miras Asem | `src/kkt/`: independent checker of five optimality conditions | S3 — Is the solution optimal, and how do we know? | H3 |
| M4 — Sensitivity | [Confirm member] | `src/sens/`: RHS perturbations and slope interval check | S4 — What the dual price says about the storyline | H4 |
| Group | All four members | Task 5, common runner, tables, report and contribution statement | S5 and Synthesis | H5 |

The remaining team members are Alina Suleimenova and Miras Asem. Their M3/M4 assignments must be confirmed against the team's agreed roles.

## Environment

The M1 implementation was tested with:

- Python 3.11.9
- NumPy 2.4.4
- PuLP 3.3.2

PuLP must remain below version 4, as required by the assignment.

### Installation

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy "pulp<4"
```

Check the installed version:

```bash
python -c "import pulp; print(pulp.__version__)"
```

The output must show a version below 4.

## Data Generation

The public generator is supplied by the course:

```python
from checkpoint3_generator_PUBLIC import cp34_instance

data = cp34_instance(16, "B", 3)
```

The generator is used unchanged. The same instance call must be retained for Checkpoint 4.

### Verified T16 Check Values

| Check | Value |
|---|---:|
| Servers | 9 |
| Virtual machines | 41 |
| Total CPU capacity | 420 |
| Total memory capacity | 1339 |
| Total CPU request | 201 |
| Total memory request | 706 |
| Forbidden VM–server pairs | 35 |
| Variables | 378 |
| Constraint rows | 95 |

The designated sensitivity constraint is the assignment row for VM index `9`, whose CPU request is `8`. Indices start at zero.

## Reference Model — B3

### Variables

- `y[k,l]`: assignment of VM `k` to server `l`.
- `t[l]`: nonnegative latency approximation for server `l`.

The integer reference model uses binary assignments. Its LP relaxation uses:

```text
0 <= y[k,l] <= 1
t[l] >= 0
```

If VM `k` requires a GPU and server `l` is not GPU-enabled, `y[k,l]` is retained with upper bound zero.

### Objective

Minimize total server latency:

```text
minimize sum_l t[l]
```

### Constraint Families

| Family | Meaning | Number of rows |
|---|---|---:|
| B1 | Each VM is assigned completely in total | 41 |
| B2 | CPU capacity of each server | 9 |
| B3 | Memory capacity of each server | 9 |
| B4 | Four latency tangent inequalities per server | 36 |
| Total | Variable bounds excluded | 95 |

For utilization `u`, the latency function is:

```text
g(u) = u / (1-u)
```

The prescribed tangent points are:

```text
0.0, 0.4, 0.7, 0.9
```

For a tangent point `v`:

```text
a_v = 1 / (1-v)^2
b_v = -v^2 / (1-v)^2
```

The latency constraint is:

```text
t[l] >= a_v * CPU_load[l] / CPU_capacity[l] + b_v
```

The matrix implementation stores its equivalent `<=` form:

```text
a_v * CPU_load[l] / CPU_capacity[l] - t[l] <= -b_v
```

No zone restrictions are added to variant B3.

## M1 — Model, Matrices and Bookkeeping

**Owner: Dilnaz Bekturova**

### Files

| File | Purpose |
|---|---|
| `src/model/cloud.py` | Reference B3 matrices and PuLP LP-model builder |
| `src/model/triangle.py` | Independent enumeration for the H1 triangle problem |
| `tests/test_model.py` | Model, generator and H1 verification |
| `experiments/run_m1.py` | Export of matrices, indexing, model sizes and H1 results |
| `docs/M1_report.tex` | Task 1 formulation, S1 analysis and H1 worked solution |
| `README_M1.md` | Detailed M1 integration instructions |

### Matrix Convention

The variable vector contains all assignments first, followed by latency variables:

```text
x = (y[0,0], ..., y[40,8], t[0], ..., t[8])
```

Zero-based column indices:

```text
column(y[k,l]) = 9*k + l
column(t[l])   = 369 + l
```

The objective vector has:

```text
369 zero coefficients for assignments
9 unit coefficients for latency variables
```

The matrix has shape:

```text
A.shape = (95, 378)
```

Zero-based row indices:

```text
B1 assignment: k
B2 CPU:        41 + l
B3 memory:     50 + l
B4 latency:    59 + 4*l + m
```

Here `m = 0,1,2,3` indexes the ordered tangent points.

Rows B1 are equalities. Rows B2–B4 are stored as `<=` inequalities.

Assignment upper bounds are one, except for 35 forbidden pairs with upper bound zero. Latency variables have no upper bound. Bounds are not counted as constraint rows.

### Interface for Other Roles

```python
from src.model.cloud import build_matrices, build_pulp_model

matrix = build_matrices()
model, y, t = build_pulp_model(matrix=matrix)
```

The matrix object provides:

```text
c, A, b, senses
lower, upper
columns, rows, families
data
```

M2 can solve the returned PuLP model and read `y[k,l]` and `t[l]`.

M3 should align primal values and row prices using the stored names and indices, rather than assuming the solver's internal ordering matches the matrix ordering.

Since B4 is multiplied by `-1`, its matrix-row dual price has the opposite sign to the original `>=` row price. Variables fixed at zero require appropriate treatment of both bounds in the KKT checker.

M4 uses:

```text
Row name:  assign_9
Row index: 9
Base RHS:  1
```

The perturbed values are `0.99`, `1.00` and `1.01`. This changes the assignment RHS, not the CPU request or server capacity.

### Run M1 Tests

```bash
python -m unittest discover -s tests -p "test_model.py" -v
```

Six tests passed in the verified M1 run:

1. Assignment and capacity coefficients.
2. Generator check values.
3. H1 enumeration and integer optimum.
4. H1 matrices and dual-sign conversion.
5. Latency inequalities against the original tangent form.
6. Actual PuLP model sizes and GPU bounds.

### Generate M1 Outputs

```bash
python -m experiments.run_m1
```

The verified output reports:

```text
variables:      378
rows:           95
forbidden_pairs: 35
B1_assignment:   41
B2_CPU:           9
B3_memory:        9
B4_latency:      36
```

### Generated Files

| File | Contents |
|---|---|
| `results/m1_matrices.npz` | Matrices, bounds and indexing metadata |
| `results/m1_rows.csv` | Row indices, names, families, senses and RHS values |
| `results/m1_columns.csv` | Column indices, names, bounds and objective coefficients |
| `results/m1_model_sizes.csv` | Actual model sizes and family counts |
| `results/h1_binary_points.csv` | All eight binary points for H1 |
| `results/T16_B3_relaxation.lp` | Inspectable LP-model export |

The M1 runner builds and exports the model. It does not compute the main LP optimum, fractionality or sensitivity results.

## H1 — Model Owner Hand Task

The T16 triangle problem is:

```text
minimize 3*y1 + 5*y2 + 6*y3

y1 + y2 >= 1
y2 + y3 >= 1
y1 + y3 >= 1
```

Its LP relaxation has `0 <= y[j] <= 1`, three variables and three constraint rows.

The hand task contains:

- The vectors `c`, `b` and matrix `A`.
- Conversion from `>=` to `<=` rows.
- The corresponding reversal of dual-price signs.
- Enumeration of all eight binary points.
- Identification of the integer optimum.

| Binary point | Feasible? | Cost |
|---|---|---:|
| `(0,0,0)` | No | 0 |
| `(0,0,1)` | No | 6 |
| `(0,1,0)` | No | 5 |
| `(0,1,1)` | Yes | 11 |
| `(1,0,0)` | No | 3 |
| `(1,0,1)` | Yes | 9 |
| `(1,1,0)` | Yes | 8 |
| `(1,1,1)` | Yes | 14 |

The integer optimum is:

```text
z* = 8
y* = (1,1,0)
```

`tests/test_model.py` reproduces these results and checks the matrix/sign conversion.

The final signed hand page is to be stored as:

```text
hand/CP3_H1_Model.pdf
```

The typeset source does not replace the requirement to complete the hand work and provide a real signature.

## Tasks 2–5 and Required Tables

The following results must be supplied by their owners and integrated into the complete team report.

### Table 1 — Model and LP Solution

Include:

- Model sizes and check values.
- LP objective `zLP`, reported to six decimals.
- Solver and solution time.
- Number of fractional assignment variables.
- Number of VMs split across multiple servers.

M1 has verified the model sizes and check values. LP results must come from the actual M2 solver run.

### Table 2 — Optimality Conditions

Report the largest violation or residual for:

1. Primal feasibility.
2. Dual-price signs.
3. Row complementary slackness.
4. Reduced-cost conditions.
5. Strong duality.

The required tolerance is `1e-6`. The checker must accept another solver's solution and must account for variable bounds.

### Table 3 — Sensitivity

For assignment row `assign_9`, solve at RHS values:

```text
0.99, 1.00, 1.01
```

Report:

```text
left slope = (z(1.00) - z(0.99)) / 0.01
dual price = solver price of assign_9 at RHS 1.00
right slope = (z(1.01) - z(1.00)) / 0.01
```

Check, to tolerance `1e-6`:

```text
left slope <= dual price <= right slope
```

### Task 5 — Group Work

Include:

- A short proof that `zLP <= z*`.
- Interpretation and limitations of `(z-zLP)/z`.
- Active-row counts by constraint family.
- A prediction about the tightness of the LP bound.
- The connection to Checkpoint 4.

## Common Reproduction Command

The required final command is:

```bash
python -m experiments.run_all
```

The common runner must regenerate Tables 1–3 and include the M1 exports.

The M1 contribution currently provides:

```bash
python -m experiments.run_m1
```

The owner of the common runner must connect this to the other roles' computations. The separate M1 command alone does not reproduce all team results.

## Repository Structure — Checkpoint 3 Additions

```text
checkpoint3_generator_PUBLIC.py

src/
  model/
    __init__.py
    cloud.py
    triangle.py
  lp/
    ... M2 implementation
  kkt/
    ... M3 implementation
  sens/
    ... M4 implementation

experiments/
  __init__.py
  run_m1.py
  run_all.py                 # complete team integration

tests/
  test_model.py
  test_m2.py                # M2 contribution
  ... M3 and M4 tests

results/
  m1_matrices.npz
  m1_rows.csv
  m1_columns.csv
  m1_model_sizes.csv
  h1_binary_points.csv
  T16_B3_relaxation.lp
  ... LP, KKT and sensitivity results

docs/
  M1_report.tex
  ... other report material

hand/
  CP3_H1_Model.pdf           # signed final hand page
  ... other signed hand tasks

README.md
README_M1.md
```

This structure lists both M1 files and the required team integration targets. A listed target is not a claim that another member has already completed or uploaded it.

Checkpoint 2 files remain in the repository.

## Submission

Submit one PDF named:

```text
T16_Checkpoint3.pdf
```

The PDF must contain:

- Cover with team ID, repository link, tag, final commit hash, roles and declarations.
- Tables 1–3.
- Signed S1–S5 sections.
- Group Synthesis.
- Signed H1–H5 pages in the appendix.
- A contribution statement on the last page, signed by all members.

Limits:

- Main text: at most six pages.
- Each individual analysis section: at most half a page.
- Synthesis: at most half a page.
- Each hand task: one page.

Contribution percentages must sum to 100 and agree with the Git history.

Use the existing private repository and retain the instructor's access. Tag the final submission commit `checkpoint3` after the complete team work has been checked.

## Declarations

### Dilnaz Bekturova — M1

ChatGPT/Codex assistance was used to draft:

- `src/model/cloud.py`
- `src/model/triangle.py`
- `tests/test_model.py`
- `experiments/run_m1.py`
- Task 1, S1 and H1 report material
- M1 documentation and this Checkpoint 3 README draft

Assistance included translating the prescribed B3 formulation into matrices, implementation drafting, debugging, test preparation and numerical verification.

The M1 model and tests were run successfully, confirming 378 variables, 95 constraint rows and 35 forbidden assignment pairs. The member is responsible for reviewing and understanding the submitted material and completing and signing the required hand work.

### Adil Mutali — M2

AI assistance (Claude) was used to assist with the code in `src/lp/`, testing and validation of `tests/test_m2.py`, generation and verification of the results CSV and NPZ files, and drafting and reviewing the S2 text and H2 solution. I reviewed and ran the code and checked the numerical results. The H2 hand solution was handwritten and signed by me.

### M3 — KKT

Miras Asem — M3
AI assistance (Gemini) was used to draft:

- `src/kkt/checker.py`

- `tests/test_m3.py`

- `experiments/run_m3.py`

- `Task 3, S3 and H3 report material`
- Assistance included mathematical verification of the KKT conditions, writing the validation logic for all five LP optimality conditions, matrix integration, and debugging. I reviewed and ran the code, confirming all KKT violations are below the 1e-6 tolerance; the H3 hand solution is written and signed by me.

### Alina Suleimenova - M4 
This project was completed with the assistance of Artificial Intelligence (AI) tools. It was used for reviewing linear programming, sensitivity analysis concepts and drafting the code in src/sens/perturbation_run.py, interval_check.py and run_m4.py   

### Course Materials

The reference model, triangle-task data and conventions come from `Checkpoint_3_Instructions.docx` and `T16_B3_Checkpoint3_packet.docx`, supplied through the course LMS.

`checkpoint3_generator_PUBLIC.py` is the course-supplied generator and is used unchanged.

Unlike Checkpoint 2's from-scratch optimization routines, this assignment permits external LP solvers. The M1 adapter uses PuLP to construct the reference LP.
