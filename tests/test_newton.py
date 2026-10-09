import unittest
import numpy as np
from src.optim.newton import pure_newton, damped_newton
from src.problems.quadratic import quadratic


def rosenbrock():
    def f(x):
        a,b=x
        return (1-a)**2+100*(b-a*a)**2
    def g(x):
        a,b=x
        return np.array([2*(a-1)-400*a*(b-a*a),200*(b-a*a)])
    def h(x):
        a,b=x
        return np.array([[2-400*b+1200*a*a,-400*a],[-400*a,200.]])
    return f,g,h


class NewtonTests(unittest.TestCase):
    def test_r1_reference_counts(self):
        f,g,h=rosenbrock()
        for solver,count in [(pure_newton,6),(damped_newton,21)]:
            x,hist,k=solver(f,g,h,[-1.2,1])
            self.assertEqual(k,count)
            self.assertEqual(len(hist),k+1)
            self.assertLess(np.linalg.norm(g(x)),1e-6)
            np.testing.assert_allclose(x,[1,1],atol=1e-5)

    def test_quadratic_rotation_and_derivatives(self):
        for theta in [0,71]:
            f,g,h,x0=quadratic(theta_deg=theta)
            eps=1e-5
            numeric=np.array([(f(x0+eps*e)-f(x0-eps*e))/(2*eps)
                              for e in np.eye(2)])
            self.assertLess(np.linalg.norm(g(x0)-numeric),1e-6)
            for solver in [pure_newton,damped_newton]:
                x,hist,k=solver(f,g,h,x0,relative=True)
                self.assertEqual(k,1)
                np.testing.assert_allclose(x,[0,0],atol=1e-12)
            np.testing.assert_allclose(np.linalg.eigvalsh(h(x0)),[2,258])

    def test_h2_hand_trace(self):
        f=lambda x:x[0]**2+np.exp(x[1])-4*x[1]
        g=lambda x:np.array([2*x[0],np.exp(x[1])-4])
        h=lambda x:np.diag([2.,np.exp(x[1])])
        x,hist,k=pure_newton(f,g,h,[3,0],max_iter=2)
        np.testing.assert_allclose(hist,[[3,0],[0,3],[0,2.1991482735]],atol=1e-3)
        self.assertTrue(all(f(hist[i+1]) < f(hist[i]) for i in range(2)))
        _,dhist,_=damped_newton(f,g,h,[3,0],max_iter=1)
        np.testing.assert_allclose(dhist[1],[0,3],atol=1e-12)

    def test_stationary_start_and_cap(self):
        f,g,h,x0=quadratic()
        self.assertEqual(pure_newton(f,g,h,[0,0],relative=True)[2],0)
        self.assertEqual(pure_newton(f,g,h,x0,max_iter=0)[2],0)


if __name__ == '__main__':
    unittest.main()
