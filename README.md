# Checkpoint 2 — Team T16

## Individual Role: M1 — Gradient Descent (GD)

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
