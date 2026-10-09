import numpy as np
import pandas as pd

# Функция
def f(x):
    return x[0]**2 + 3*x[1]**2

# Градиент
def grad_f(x):
    return np.array([
        2*x[0],
        6*x[1]
    ])

# Параметры
alpha = 0.1
beta = 0.6

# Начальная точка
x = np.array([5.0, 2.0])
v = np.array([0.0, 0.0])

# Таблица для Momentum
rows = []
rows.append([0, v.copy(), x.copy(), f(x)])

# Шаг 1
g = grad_f(x)
v = beta*v - alpha*g
x = x + v
rows.append([1, v.copy(), x.copy(), f(x)])

# Шаг 2
g = grad_f(x)
v = beta*v - alpha*g
x = x + v
rows.append([2, v.copy(), x.copy(), f(x)])

momentum_table = pd.DataFrame(
    rows,
    columns=["k", "v^(k)", "x^(k)", "f(x^(k))"]
)

print("Momentum")
print(momentum_table)

# ----------------------------
# Plain Gradient Descent
# ----------------------------

x_gd = np.array([5.0, 2.0])

for _ in range(2):
    x_gd = x_gd - alpha * grad_f(x_gd)

print("\nGradient Descent")
print("x^(2) =", x_gd)
print("f(x^(2)) =", f(x_gd))