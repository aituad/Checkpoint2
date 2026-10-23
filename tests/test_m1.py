import unittest
import numpy as np
from src.model.cloud import build_matrices, build_pulp_model
from src.model.triangle import triangle_data, enumerate_binary, integer_optimum

class ModelTests(unittest.TestCase):
    def test_team_sizes_and_bounds(self):
        m = build_matrices()
        self.assertEqual(m.A.shape, (95,378))
        self.assertEqual(sum(m.upper == 0), 35)
        self.assertTrue(np.all(np.isinf(m.upper[-9:])))
        self.assertEqual(m.rows.index('assign_9'), 9)
        self.assertEqual(m.b[9], 1)
        model, y, t = build_pulp_model(matrix=m)
        self.assertEqual(len(model.variables()),378)
        self.assertEqual(len(model.constraints),95)
        self.assertEqual(sum(v.upBound == 0 for v in y.values()),35)

    def test_generator_check_values(self):
        d=build_matrices().data
        actual=[int(d['S']),int(d['M']),int(np.sum(d['cpu_cap'])),int(np.sum(d['mem_cap'])),int(np.sum(d['cpu_req'])),int(np.sum(d['mem_req'])),int(np.argmax(d['cpu_req'])),int(d['cpu_req'][9])]
        self.assertEqual(actual,[9,41,420,1339,201,706,9,8])

    def test_assignment_and_capacity_coefficients(self):
        m = build_matrices(); d=m.data
        for k in range(41):
            np.testing.assert_array_equal(np.flatnonzero(m.A[k]), np.arange(k*9,(k+1)*9))
        for l in range(9):
            np.testing.assert_allclose(m.A[41+l,l:369:9],d['cpu_req'])
            np.testing.assert_allclose(m.A[50+l,l:369:9],d['mem_req'])

    def test_latency_matches_tangent_in_original_form(self):
        m=build_matrices(); x=np.linspace(0,1,378)
        for l in range(9):
            utilization=np.dot(m.data['cpu_req'],x[l:369:9])/m.data['cpu_cap'][l]
            for j,u in enumerate([0,.4,.7,.9]):
                i=59+4*l+j
                original=x[369+l] - (utilization/(1-u)**2-u*u/(1-u)**2)
                self.assertAlmostEqual(m.b[i]-m.A[i]@x, original, places=10)

    def test_h1_matrix_and_sign_conversion(self):
        c,A,b=triangle_data()
        np.testing.assert_allclose(c,[3,5,6],atol=1e-6)
        np.testing.assert_allclose(A,[[1,1,0],[0,1,1],[1,0,1]],atol=1e-6)
        self.assertEqual(A.shape,(3,3))
        # Multiplying each row by -1 preserves primal feasibility.
        for p, feasible, cost in enumerate_binary():
            self.assertEqual(feasible,bool(np.all((-A)@p <= -b)))
        pi=np.array([1.,4.,2.])
        np.testing.assert_allclose(A.T@pi,c,atol=1e-6)
        np.testing.assert_allclose((-A).T@(-pi),c,atol=1e-6)

    def test_h1_all_points_and_optimum(self):
        entries=enumerate_binary()
        np.testing.assert_allclose([r[2] for r in entries],[0,6,5,11,3,9,8,14],atol=1e-6)
        self.assertEqual([r[1] for r in entries],[False,False,False,True,False,True,True,True])
        self.assertEqual(integer_optimum(),(8.,(1,1,0)))

if __name__ == '__main__': unittest.main()
