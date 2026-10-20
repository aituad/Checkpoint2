"""Independent enumeration for the M1 hand task H1."""
from itertools import product
import numpy as np

def triangle_data():
    return np.array([3.,5.,6.]), np.array([[1.,1.,0.],[0.,1.,1.],[1.,0.,1.]]), np.ones(3)

def enumerate_binary():
    c, A, b = triangle_data()
    return [(point, bool(np.all(A @ np.array(point) >= b)), float(c @ point))
            for point in product([0,1], repeat=3)]

def integer_optimum():
    return min((cost, point) for point, feasible, cost in enumerate_binary() if feasible)
