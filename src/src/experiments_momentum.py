
import numpy as np
import pandas as pd


# Objective function
def f(x):
    return x[0]**2 + 3*x[1]**2


# Gradient
def grad_f(x):
    return np.array([
        2*x[0],
        6*x[1]
    ])


# Parameters
alpha = 0.1
beta = 0.6
x0 = np.array([5.0, 2.0])


# --------------------------------
# Momentum experiment
# --------------------------------

x = x0.copy()
v = np.zeros_like(x)

momentum_rows = []

momentum_rows.append({
    "Method": "Momentum",
    "Iteration": 0,
    "x1": x[0],
    "x2": x[1],
    "v1": v[0],
    "v2": v[1],
    "f(x)": f(x)
})

for k in range(1, 3):
    g = grad_f(x)

    v = beta * v - alpha * g
    x = x + v

    momentum_rows.append({
        "Method": "Momentum",
        "Iteration": k,
        "x1": x[0],
        "x2": x[1],
        "v1": v[0],
        "v2": v[1],
        "f(x)": f(x)
    })

momentum_df = pd.DataFrame(momentum_rows)

print("MOMENTUM RESULTS")
print(momentum_df.to_string(index=False))


# --------------------------------
# Gradient Descent experiment
# --------------------------------

x_gd = x0.copy()
gd_rows = []

gd_rows.append({
    "Method": "Gradient Descent",
    "Iteration": 0,
    "x1": x_gd[0],
    "x2": x_gd[1],
    "f(x)": f(x_gd)
})

for k in range(1, 3):
    x_gd = x_gd - alpha * grad_f(x_gd)

    gd_rows.append({
        "Method": "Gradient Descent",
        "Iteration": k,
        "x1": x_gd[0],
        "x2": x_gd[1],
        "f(x)": f(x_gd)
    })

gd_df = pd.DataFrame(gd_rows)

print("\nGRADIENT DESCENT RESULTS")
print(gd_df.to_string(index=False))


# --------------------------------
# Comparison
# --------------------------------

comparison_df = pd.concat(
    [momentum_df, gd_df],
    ignore_index=True
)

comparison_df.to_csv(
    "momentum_experiment_results.csv",
    index=False
)

print("\nCOMPARISON")
print(
    comparison_df[
        ["Method", "Iteration", "x1", "x2", "f(x)"]
    ].to_string(index=False)
)

print("\nResults saved to momentum_experiment_results.csv")