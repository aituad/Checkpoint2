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

The M3 and M4 assignments follow the report and must be checked against the roles recorded in the original README.

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

### Shared Work

AI assistance was used to draft `src/problems/project.py`, assist with the common experiment runner and verification, and prepare the LaTeX report and this README.

The cloud data and problem conventions come from the course task files and Team T16's Checkpoint 1 materials. The solver implementations do not use an external optimization solver library.

### Declarations Still Requiring Completion

Alina Suleimenova and Miras Asem must provide their own declarations of AI assistance and external sources. Each member must review the description of their contribution and take responsibility for understanding and defending the submitted work.

## Submission Record

The report currently contains a placeholder for the final commit hash. Replace it with the actual submitted commit:

```bash
git rev-parse HEAD
```

Before submission, confirm the original role assignments, complete all individual declarations, and include the required signed hand traces.
