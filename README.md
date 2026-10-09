# Checkpoint 2 — Team T16

## Individual Role: M1 — Gradient Descent (GD)

**Author:** Adil Mutali.

### Code

- `src/optim/gd.py`
  - `gradient_descent(f, grad, x0, alpha, max_iter=100_000, tol_abs=None, tol_rel=None)` — fixed step.
  - `gradient_descent_backtracking(f, grad, x0, max_iter=100_000, tol_abs=None, tol_rel=None, alpha0=1.0, armijo_c=1e-4, return_steps=False)` —
    d = −∇f, α starts at 1 and is halved while `f(x+αd) > f(x) + 1e-4·α·⟨∇f,d⟩`.
    With `return_steps=True` it also returns the accepted steps and the numbers of halvings.
- `src/problems/rosenbrock.py` — `f_rosenbrock`, `grad_rosenbrock`, `hess_rosenbrock` (aliases `f`, `grad`, `hess`) and the starts R1 = (−1.2, 1), R2 = (−1.51, 2.17).
- `src/problems/check_grad.py` — central-difference gradient check, returns `(max_error, analytic, numeric)`.
- `tests/test_gd.py`, `tests/test_grad.py`; `src/experiments/run_s1.py`; `docs/S1_GD.md`; `hand/H1_GD.pdf`.

### Verification (reproduced by `python -m pytest tests/test_gd.py tests/test_grad.py`)

| Check | Reference | Result |
|---|---|---|
| R1, GD α = 1e-3, absolute rule | 32 076 | 32 076 |
| R1, GD-BT | 13 756 | 13 756 |
| R1 start: f, ‖∇f‖ | 24.2, 232.86 | 24.2, 232.868 |
| R2 start: f, ‖∇f‖ | 7.5123, 74.8335 | 7.5123, 74.8335 |
| Q1 / Q2 start: f, ‖∇f‖ | 64.9, 180.619 | 64.9, 180.6187 (both) |
| Q1 vs Q2, GD at α* = 1/130 | equal up to ±1 | 892 / 892 |
| Q1 vs Q2, GD-BT | equal up to ±1 | 39 / 39 |
| `check_grad` on R1 (−1.2, 1) | below 1e-6 | 2.24e-08 |
| `check_grad` on R2 start, Q1, Q2 | below 1e-6 | 3.4e-09, 5.1e-09, 5.6e-09 |

H1 (f = 2x² + 7y², x⁰ = (4, 1)) is reproduced to 1e-3 by `tests/test_gd.py`:
two fixed steps with α = 0.05 give (3.2, 0.3) and (2.56, 0.09) with f = 21.11 and 13.1639;
backtracking rejects α = 1, 0.5, 0.25 and accepts α = 0.125, giving x¹ = (2, −0.75), f = 11.9375.

Additional GD numbers (Q1/Q2, c = 129, relative rule): GD at α* = 1/(1+c) needs 892 updates,
the best grid value α = 10^−2.5 needs 1510, GD-BT needs 39 (R2: GD α = 1e-3 needs 32 794, GD-BT 17 291).

### Analysis

`docs/S1_GD.md` (S1, signed): the t·2/λ_max sweep on Q1 against ρ(α), backtracking
statistics on R1, Q1 and the project block, and the role of the starting guess α₀ = 1.

### Declaration (M1)

- **External code:** none in `src/optim/gd.py`. No `scipy.optimize`, `sklearn`, `autograd`, JAX or PyTorch is used.
- **AI assistance:** an AI assistant (Claude, Anthropic) worked as my assistant on the M1 part. It explained the methods, helped debug, helped draft parts of `src/optim/gd.py`, `src/problems/rosenbrock.py`, `src/problems/check_grad.py` and the documentation, helped write the tests (`tests/test_gd.py`, `tests/test_grad.py`) and the S1 experiment script (`src/experiments/run_s1.py`), and checked my work against the assignment and the task file: it re-ran the reference counts and the gradient check and pointed out what did not match (broken test imports, a missing GD-BT column and the α* candidate in Table 1, mistakes in the S1 text). I reviewed and re-ran everything and I am responsible for understanding and defending the submitted work.
- **Hand trace H1** is my own work on paper; AI was not used to produce it.




## Individual Role: M2 — Newton Method

**Author:** Dilnaz Bekturova, Team T16.

### Responsibilities

- Implement pure Newton and damped Newton with Armijo backtracking.
- Implement Q1 and rotated Q2 objectives, analytical gradients and Hessians.
- Prepare and verify the H2 hand trace.
- Perform S2 experiments and analyze the results: conditioning,
  iteration counts, computational cost, monotonicity and Hessian eigenvalues.
- Compare Newton results with GD and Adam; complete the Momentum
  comparison after verification of the team's implementation.

### Implementation

- `src/optim/newton.py`: pure and damped Newton methods.
- `src/problems/quadratic.py`: Q1 and Q2 objectives, gradients and Hessians.
- `tests/test_newton.py`: Newton verification tests.
- `src/experiments/run_newton_s2.py`: reproducible S2 experiments.
- `docs/S2_draft.md`: S2 analysis draft.
- `hand/H2_Newton.pdf`: signed H2 hand trace.

Newton directions are computed using `numpy.linalg.solve`.
The optimizer implementation uses NumPy and the Python standard library.

### Running the Tests and Experiments

Run from the repository root:

```bash
python -m unittest discover -s tests -p "test_newton.py" -v
python -m src.experiments.run_newton_s2
```

All four Newton tests passed, covering:

- H2 hand-trace reproduction.
- Quadratic rotation and analytical derivatives.
- R1 reference iteration counts.
- Stationary initial points and the iteration cap.

On R1, starting at (−1.2, 1.0), pure Newton required 6 updates
and damped Newton required 21 updates.
The stopping criterion was the gradient norm below 1e-6.

### S2 Results and Analysis

For f_c(x) = x₁² + c x₂², all runs start at (1.3, 0.7).
The stopping criterion is:

||∇f(x_k)|| < 1e-6 ||∇f(x_0)||.

GD uses α* = 1 / (1 + c). Adam uses the prescribed tuning grid.

| c | GD α* updates | Tuned Adam updates | Pure Newton updates | Damped Newton updates |
|---|---|---|---|---|
| 10 | 69 | 205 | 1 | 1 |
| 100 | 691 | 186 | 1 | 1 |
| 1000 | 6908 | 185 | 1 | 1 |
| 129 | 892 | 186 | 1 | 1 |

GD slows as conditioning increases. Tuned Adam shows no comparable
slowdown in these experiments; this does not establish general
independence from conditioning.

Both Newton variants solve these positive definite quadratics
in one update because the quadratic Taylor model is exact.
Iteration counts alone do not account for the cost of solving
the Newton linear system.

| Problem | Newton variant | Updates | Stopping criterion reached | Objective monotone |
|---|---|---|---|---|
| R2 | Pure | 7 | Yes | No |
| R2 | Damped | 22 | Yes | Yes |
| Project | Pure | 7 | Yes | No |
| Project | Damped | 6 | Yes | Yes |

The final Hessians have positive eigenvalues.
For the project, both Newton variants reach an objective value
of approximately 0.22159147, agreeing numerically with the known
global optimum.

A small gradient indicates approximate stationarity. Positive
Hessian eigenvalues support the local minimum assessment; the
project comparison with the known optimum provides additional
numerical verification.

GD, Adam and Newton results were reproduced using the team's
implementations. Momentum results and the associated cost
comparison remain provisional pending team verification.

Generated files:

- `results/s2_conditioning_team.csv`
- `results/s2_adam_grid_team.csv`
- `results/s2_safeguards_team.csv`
- `results/s2_histories_team.csv`

### H2 — Newton Hand Trace

For f(x, y) = x² + exp(y) − 4y, starting at (3, 0):

| k | Point (x, y) | f(x, y) | Gradient dot Newton direction | Absolute y error |
|---|---|---|---|---|
| 0 | (3, 0) | 10.0000 | −27.0000 | 1.3863 |
| 1 | (0, 3) | 8.0855 | −12.8821 | 1.6137 |
| 2 | (0, 2.1991) | 0.2207 | −2.7917 | 0.8129 |

The minimizer is (0, ln 4) ≈ (0, 1.3863),
with f* = 4 − 4 ln 4 ≈ −1.5452.

The objective decreases over the two computed updates, although
the absolute y error initially increases.

At k = 0, Armijo accepts the full step because:

8.0855 ≤ 10 + 1e-4 × (−27) = 9.9973.

The error ratio at k = 1 is approximately 0.3122.
The theoretical asymptotic ratio is 0.5; one measured ratio
alone does not establish quadratic convergence.

The direction at k = 2 is evaluated for the table;
a third update is not performed.

Numerical values are in `results/h2_values.csv`.
The signed hand trace is in `hand/H2_Newton.pdf`.

### M2 AI Assistance Declaration

AI assistance was used for implementation drafting, debugging
and numerical verification.

Dilnaz Bekturova performed the test runs, reviewed the numerical
outputs and analyzed the experimental results. She is responsible
for reviewing and understanding her submitted work.
