# Checkpoint 2 — Team T16

## Individual Role: M1 — Gradient Descent (GD)
Author: Adil Mutali

### Responsibilities

- Implement Gradient Descent with fixed step size.
- Implement Gradient Descent with Armijo backtracking.
- Implement the Rosenbrock objective and analytical gradient.
- Implement gradient checking using central finite differences.
- Perform the required convergence and gradient-check experiments.
- Prepare the M1 hand trace and analysis.

### Implementation

The Gradient Descent implementation is located in:

- `src/optim/gd.py`
- `src/problems/rosenbrock.py`
- `src/problems/check_grad.py`

Tests are located in:

- `tests/test_gd.py`
- `tests/test_grad.py`

The implementation uses NumPy and the Python standard library only.

### Verification

For the Rosenbrock starting point `(-1.2, 1.0)`:

- Fixed-step GD converges in 32076 iterations.
- GD with Armijo backtracking converges in 13756 iterations.
- The stopping criterion is `||grad f(x)|| < 1e-6`.

Gradient checking at `(-1.2, 1.0)` gives:

- Maximum gradient error: approximately `2.24e-08`
- Required tolerance: below `1e-6`

Therefore, the analytical gradient passes the required gradient check.

## Declaration

AI assistance was used during development of the M1 implementation and documentation. The AI assistant was used for explanation, debugging guidance, numerical verification, and drafting parts of the implementation/documentation. The resulting code was reviewed and tested by the student, and the student is responsible for understanding and defending the submitted work.

No external optimization library was used for the Gradient Descent implementation.



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
