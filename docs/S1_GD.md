# S1 — Step Size and Stability

## Q1 fixed-step GD

For Q1, the condition number is c = 129. The Hessian is

H = diag(2, 258),

so

λmax = 258.

The stability boundary is

α < 2 / λmax.

Therefore,

2 / λmax = 2 / 258 = 1/129 ≈ 0.00775194.

We tested

α = t · 2/λmax

for

t ∈ {0.5, 0.9, 0.99, 1.01, 1.1}.

| t | α | Predicted ρ(α) | Result | Iterations |
|---:|---:|---:|---|---:|
| 0.50 | 0.00387597 | 0.992248 | converged | 1231 |
| 0.90 | 0.00697674 | 0.986047 | converged | 682 |
| 0.99 | 0.00767442 | 0.984651 | converged | 688 |
| 1.01 | 0.00782946 | 1.020000 | failed | — |
| 1.10 | 0.00852713 | 1.200000 | failed | — |

The experiments show the expected stability boundary. For t < 1,
the predicted spectral factor is below one and GD converges. At
t > 1, the factor exceeds one and the method fails to converge.

At t = 0.99, the measured final gradient-norm ratio was approximately
0.980568, which is close to the predicted asymptotic factor 0.984651.

The iteration count is not minimized by taking α arbitrarily close to
the stability boundary. In this experiment, t = 0.90 required 682
iterations, while t = 0.99 required 688 iterations.

## GD-BT step sizes

For the required GD-BT experiments, the mean accepted step sizes and
mean numbers of halvings were:

| Problem | Mean accepted α | Mean halvings | 1/λmax(H) at solution |
|---|---:|---:|---:|
| R1 | 0.002096 | 8.945 | ≈ 0.000998 |
| Q1 | 0.020533 | 6.923 | ≈ 0.003876 |
| Project | 1.000000 | 0.000 | ≈ 13.514 |

For R1, the backtracking method accepts a step substantially larger
than the local reciprocal curvature estimate at the solution on average,
but the final accepted step is approximately 0.001953, close to
1/λmax ≈ 0.000998.

For Q1, the final accepted step is approximately 0.003906, which is
very close to 1/λmax ≈ 0.003876.

For the project block, GD-BT accepted α = 1.0 without halving on
average. Here 1/λmax(H) ≈ 13.514, so the initial α0 = 1 is smaller
than the local inverse-curvature scale. Because backtracking only
decreases α, it cannot increase the step when α0 is already too small.
Thus α0 = 1 is harmless here, but it prevents GD-BT from exploiting
the larger local step suggested by 1/λmax(H).

## Conclusion

The Q1 sweep confirms that the theoretical stability threshold predicts
the observed transition between convergence and divergence. Backtracking
automatically reduces the step size when the initial α = 1 is too large,
as observed on R1 and Q1, while on the project block α = 1 is accepted
without halving because it is already below the local inverse-curvature
scale. Since GD-BT only decreases α, an initial step that is too small
cannot be increased automatically.

Name: Adil