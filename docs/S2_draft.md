# S2 Conditioning, cost and safeguards

For f_c(x) = x₁² + c·x₂², the Hessian is diag(2, 2c),
so κ = c. All runs start at (1.3, 0.7) and stop before
updating when ||∇f(x_k)|| < 10⁻⁶ ||∇f(x⁰)||.
GD uses α* = 1/(1+c); Adam and Momentum use the prescribed
parameter grids.

| c | GD α* | Momentum† | Adam | Newton pure | Newton damped |
|---|---:|---:|---:|---:|---:|
| 10 | 69 | 41 | 205 | 1 | 1 |
| 100 | 691 | 105 | 186 | 1 | 1 |
| 1000 | 6908 | 297 | 185 | 1 | 1 |
| 129 | 892 | 120 | 186 | 1 | 1 |

GD and Adam results were reproduced using the team's implementations
with src/experiments/run_newton_s2.py. Newton results were also
reproduced successfully. †Momentum results remain auxiliary reference
values pending verification against the team's implementation.

Increasing κ from 10 to 1000 increased GD from 69 to 6908
updates, while reference Momentum increased from 41 to 297.
Tuned Adam required 205, 186 and 185 updates, showing no
comparable slowdown in these experiments. This result concerns
the tuned runs and the relative stopping rule; it does not
establish general independence from conditioning.

Both Newton variants required one update for every c because
their quadratic Taylor model is exact. Rotation preserves the
Newton update, so Q1 and Q2 also require one update. However,
iteration counts do not include the cost of solving the
Newton system.

For Q1, k_N = 1 and the reference Momentum count is k_M = 120.
The stipulated total costs are n³·k_N for Newton and 10n·k_M
for Momentum. Momentum is cheaper when
n > √(10k_M/k_N) = √1200 ≈ 34.6410, hence integer n ≥ 35.
This threshold awaits verification of the team's Momentum
count. It assumes these counts persist as dimension grows
and omits derivative costs and implementation constants.

| Problem | Newton | Updates | f monotone | Final gradient norm | Hessian eigenvalue range |
|---|---|---:|---|---:|---|
| R2 | pure | 7 | No | 6.221×10⁻⁷ | [0.399361, 1001.60] |
| R2 | damped | 22 | Yes | 9.956×10⁻⁸ | [0.399361, 1001.60] |
| Project | pure | 7 | No | 1.019×10⁻⁹ | [0.00525790, 0.0739971] |
| Project | damped | 6 | Yes | 1.346×10⁻¹² | [0.00525790, 0.0739971] |

Both variants reached the stopping rule. Damping maintained
monotone decrease on both problems: it required more updates
on R2, but one fewer update on the project.

All returned Hessians were positive definite. At an exactly
stationary point, this establishes a strict local minimum.
A small gradient alone certifies only approximate stationarity.
R2 returned close to the known global minimizer (1,1).
Both project objective values agreed numerically with the
known global optimum J* = 0.2215914700; share and objective
errors are recorded in results/s2_safeguards.csv. The objective
is strictly convex in shares w, but the softmax parametrization
need not be globally convex in z.

| c | GD α* | Momentum α† | Momentum β† | Adam α |
|---|---:|---:|---:|---:|
| 10 | 0.09090909 | 0.1 | 0.5 | 0.1 |
| 100 | 0.00990099 | 0.01 | 0.8 | 0.1 |
| 1000 | 0.000999001 | 0.001 | 0.9 | 0.0316227766 |
| 129 | 0.00769231 | 0.01 | 0.8 | 0.1 |

Adam uses β₁ = 0.9, β₂ = 0.999 and ε = 10⁻⁸.
None of the selected tuned α values is on the upper edge
of the initial grid. †Momentum settings await verification.

Author: Dilnaz Bekturova
