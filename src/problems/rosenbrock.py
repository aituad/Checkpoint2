import numpy as np

# Стартовые точки
X0_R1 = np.array([-1.2, 1.0])
# Специфичный старт для команды T16 (из файла задания)
X0_R2 = np.array([-1.51, 2.17])

def f_rosenbrock(x):
    """
    Функция Розенброка: f(x) = (1 - x1)^2 + 100 * (x2 - x1^2)^2
    """
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    """
    Градиент функции Розенброка.
    """
    df_dx1 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2)
    df_dx2 = 200.0 * (x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2])

def hess_rosenbrock(x):
    """
    Гессиан функции Розенброка (нужен для метода Ньютона, роли M2).
    """
    h11 = 2.0 - 400.0 * (x[1] - x[0]**2) + 800.0 * x[0]**2
    h12 = -400.0 * x[0]
    h21 = -400.0 * x[0]
    h22 = 200.0
    return np.array([[h11, h12],
                     [h21, h22]])

def check_grad(f, grad_f, x, h=1e-5):
    """
    Проверка градиента методом центральных разностей.
    """
    n = len(x)
    approx_grad = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        approx_grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    exact_grad = grad_f(x)
    error = np.linalg.norm(approx_grad - exact_grad)
    return error