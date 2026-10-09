import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.problems.check_grad import check_grad
from src.problems.rosenbrock import f, grad
from src.problems.quadratic import q1, q1_grad

def test_rosenbrock_gradient_check():
    err, _, _ = check_grad(f, grad, np.array([-1.2, 1.0]))
    assert err < 1e-6

def test_q1_gradient_check():
    err, _, _ = check_grad(lambda x:q1(x,129), lambda x:q1_grad(x,129), np.array([1.3,.7]))
    assert err < 1e-6
