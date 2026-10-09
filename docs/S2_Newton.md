# S2 — Conditioning, cost and safeguards

## Conditioning experiments

For f_c(x) = x₁² + c·x₂², the Hessian is diag(2, 2c), so its condition number is κ = c. All conditioning runs start at (1.3, 0.7). The stopping criterion is ||∇f(x_k)|| < 10⁻⁶ ||∇f(x₀)|| and is checked before each update. Counts below denote completed updates.

GD uses the optimal constant step α* = 1/(1+c). Adam and Momentum were tuned over α ∈ {10^(-5 + 0.5j): j = 0, ..., 10}. Momentum additionally uses β ∈ {0.5, 0.8, 0.9, 0.95, 0.99}. Among converged runs, the setting with the fewest updates was selected.

| c | GD α* | Momentum | Adam | Newton pure | Newton damped |
|---|---:|---:|---:|---:|---:|
| 10 | 69 | 41 | 205 | 1 | 1 |
| 100 | 691 | 105 | 186 | 1 | 1 |
| 1000 | 6908 | 297 | 185 | 1 | 1 |
| 129 | 892 | 120 | 186 | 1 | 1 |

These results were reproduced using the team's GD, Momentum, Adam and Newton implementations through `src/experiments/run_newton_s2.py`. All selected runs reached the relative stopping criterion.

| c | GD α* | Momentum α | Momentum β | Adam α |
|---|---:|---:|---:|---:|
| 10 | 0.0909090909 | 0.1 | 0.5 | 0.1 |
| 100 | 0.0099009901 | 0.01 | 0.8 | 0.1 |
| 1000 | 0.0009990010 | 0.001 | 0.9 | 0.0316227766 |
| 129 | 0.0076923077 | 0.01 | 0.8 | 0.1 |

Adam uses β₁ = 0.9, β₂ = 0.999 and ε = 10⁻⁸. None of the selected tuned α values lies on the upper edge of the initial grid, so no upper-grid extension was needed.

Increasing κ from 10 to 1000 increased GD from 69 to 6908 updates, while tuned Momentum increased from 41 to 297. Momentum therefore reduced the update count substantially compared with GD, but still slowed as conditioning increased.

Tuned Adam required 205, 186 and 185 updates for these three values of c, showing no comparable slowdown in these experiments. This observation concerns the selected settings and the relative stopping criterion; it does not establish general independence from conditioning.

Both Newton variants required one update for every c because their quadratic Taylor model is exact. For the corresponding rotated quadratic Q2, rotating the Hessian and starting point preserves the Newton update, so Q1 and Q2 each require one update. Update counts alone do not include the cost of forming the Hessian or solving the Newton system.

## Computational cost

For Q1, the measured counts are k_N = 1 for Newton and k_M = 120 for tuned Momentum.

Under the stipulated cost model, total costs are n³k_N for Newton and 10nk_M for Momentum. Momentum is cheaper when:

n > √(10k_M/k_N) = √1200 ≈ 34.6410.

Thus, Momentum is cheaper for integer n ≥ 35 under this model. This comparison assumes that the measured iteration counts remain applicable as dimension grows and omits derivative costs and implementation constants.

## Newton safeguards

R2 starts at (-1.51, 2.17) and uses the absolute stopping criterion ||∇f(x_k)|| < 10⁻⁶. The project starts at z₀ = 0 and uses the relative criterion ||∇F(z_k)|| < 10⁻⁶ ||∇F(z₀)||.

The project uses nine servers with capacities C = (59, 46, 57, 35, 49, 43, 32, 52, 47) and total CPU demand D = 201. The last softmax logit is fixed to zero. Project gradients and Hessians were checked against central finite differences before the Newton runs.

| Problem | Newton | Updates | f monotone | Final gradient norm | Final Hessian eigenvalue range |
|---|---|---:|---|---:|---|
| R2 | pure | 7 | No | 6.221 × 10⁻⁷ | [0.399361, 1001.600] |
| R2 | damped | 22 | Yes | 9.956 × 10⁻⁸ | [0.399361, 1001.601] |
| Project | pure | 7 | No | 1.019 × 10⁻⁹ | [0.0052579, 0.0739971] |
| Project | damped | 6 | Yes | 1.346 × 10⁻¹² | [0.0052579, 0.0739971] |

All four runs reached their stopping criteria. Monotonicity was assessed from the recorded objective values, allowing an absolute numerical tolerance of 10⁻¹².

Pure Newton was non-monotone on both problems, whereas damped Newton maintained monotone objective decrease. On R2, damping increased the update count from 7 to 22; on the project, it reduced the count from 7 to 6. These observations illustrate the trade-off between full Newton steps and step-size control rather than a universal ranking of the methods.

All final Hessians were positive definite. At an exactly stationary point, a positive definite Hessian establishes a strict local minimum. A small gradient alone indicates approximate stationarity; finite stopping tolerances make the minimum assessment numerical.

For R2, the final objective values were approximately 9.675 × 10⁻¹⁴ for pure Newton and 1.860 × 10⁻¹⁷ for damped Newton, consistent with the known global minimum f(1,1) = 0.

For the project, the objective is strictly convex in the shares w, with the known minimizer w*_j = C_j² / ΣC_i² and global minimum J* = D²/(9ΣC_i²) ≈ 0.2215914700. Both Newton variants agreed numerically with this value.

| Project Newton | Maximum absolute share error | Absolute objective error |
|---|---:|---:|
| pure | 1.256 × 10⁻⁹ | 2.776 × 10⁻¹⁷ |
| damped | 2.868 × 10⁻¹² | 0.0 at floating-point precision |

The reported zero objective error is a floating-point result, not a claim of exact arithmetic equality. Strict convexity in shares does not imply global convexity after softmax parametrization in z.

## Reproducibility

Run from the repository root:

python -m src.experiments.run_newton_s2

Numerical results and tuning trials are stored in:

- `results/s2_conditioning_team.csv`
- `results/s2_adam_grid_team.csv`
- `results/s2_momentum_grid_team.csv`
- `results/s2_safeguards_team.csv`
- `results/s2_histories_team.csv`

The Momentum results reported here cover the conditioning experiments. They do not verify Momentum results for R1, R2, rotated Q2 or the project.

Author: Dilnaz Bekturova
