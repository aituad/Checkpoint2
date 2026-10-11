import numpy as np

# Starting points
X0_R1 = np.array([-1.2, 1.0])
# R2 start for team T16 (from the task file)
X0_R2 = np.array([-1.51, 2.17])

def f_rosenbrock(x):
    """
    Rosenbrock function: f(x) = (1 - x1)^2 + 100 * (x2 - x1^2)^2
    """
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    """
    Gradient of the Rosenbrock function.
    """
    df_dx1 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2)
    df_dx2 = 200.0 * (x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2])

def hess_rosenbrock(x):
    """
    Hessian of the Rosenbrock function (used by Newton, role M2).
    """
    h11 = 2.0 - 400.0 * (x[1] - x[0]**2) + 800.0 * x[0]**2
    h12 = -400.0 * x[0]
    h21 = -400.0 * x[0]
    h22 = 200.0
    return np.array([[h11, h12],
                     [h21, h22]])


# Short aliases used by tests and the experiment scripts.
f = f_rosenbrock
grad = grad_rosenbrock
hess = hess_rosenbrock
# check_grad lives only in src/problems/check_grad.py (single implementation).
