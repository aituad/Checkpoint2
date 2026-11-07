import unittest
import numpy as np
from checkpoint4_generator_PUBLIC import cp34_instance,evaluate_B
from src.meta.ga_problem import random_individual,evaluate,crossover,mutate

class GAPTests(unittest.TestCase):
    def setUp(self):
        self.d=cp34_instance(16,'B',3)

    def test_evaluate_1000_random_individuals(self):
        rng=np.random.default_rng(123)
        feasible_cases=0;infeasible_cases=0
        for i in range(1000):
            d=cp34_instance([16,23,40,7][i%4],'B',3)
            a=rng.integers(0,d['S'],size=d['M'])
            cost,problems=evaluate_B(d,a)
            result=evaluate(d,a,rho=10)
            self.assertAlmostEqual(cost,result['cost'],delta=1e-9)
            self.assertEqual(not problems,result['feasible'])
            self.assertAlmostEqual(result['fitness'],result['cost']+10*result['violation'],delta=1e-9)
            feasible_cases+=not problems;infeasible_cases+=bool(problems)
        self.assertGreater(infeasible_cases,0)

    def test_1000_pairs_operators(self):
        rng=np.random.default_rng(456);d=self.d
        for i in range(1000):
            p=random_individual(d,rng);q=random_individual(d,rng)
            oldp,oldq=p.copy(),q.copy()
            child=crossover(d,p,q,rng)
            self.assertTrue(np.all((child==p)|(child==q)))
            mutated=mutate(d,child,rng,n_changes=1+i%2)
            self.assertEqual(np.count_nonzero(mutated!=child),1+i%2)
            for a in [p,q,child,mutated]:
                self.assertEqual(a.shape,(d['M'],))
                self.assertTrue(np.issubdtype(a.dtype,np.integer))
                self.assertTrue(np.all((a>=0)&(a<d['S'])))
                self.assertFalse(np.any((d['needs_gpu']==1)&(d['is_gpu_server'][a]==0)))
            np.testing.assert_array_equal(oldp,p);np.testing.assert_array_equal(oldq,q)

    def test_hand_penalty(self):
        d=dict(story='B',variant=3,S=2,M=2,cpu_cap=np.array([10,10]),mem_cap=np.array([10,10]),cpu_req=np.array([6,6]),mem_req=np.array([6,6]),needs_gpu=np.array([0,0]),is_gpu_server=np.array([1,1]))
        result=evaluate(d,np.array([0,0]),rho=10)
        self.assertAlmostEqual(result['cost'],19,delta=1e-9)
        self.assertAlmostEqual(result['violation'],.4,delta=1e-9)
        self.assertAlmostEqual(result['fitness'],23,delta=1e-9)
        self.assertFalse(result['feasible'])
        result=evaluate(d,np.array([0,1]),rho=10)
        self.assertTrue(result['feasible'])
        self.assertEqual(result['violation'],0)
        self.assertAlmostEqual(result['cost'],2*(.6-.16)/.36,delta=1e-9)
        self.assertEqual(result['feasible'],not evaluate_B(d,[0,1])[1])
        d['needs_gpu'][0]=1; d['is_gpu_server'][0]=0
        self.assertFalse(evaluate(d,np.array([0,1]))['feasible'])
        self.assertAlmostEqual(evaluate(d,np.array([0,1]))['violation'],1,delta=1e-9)

    def test_reproducible_operators_and_single_server(self):
        def steps():
            rng=np.random.default_rng(7);p=random_individual(self.d,rng)
            q=random_individual(self.d,rng)
            return mutate(self.d,crossover(self.d,p,q,rng),rng)
        np.testing.assert_array_equal(steps(),steps())
        d=dict(story='B',variant=3,S=1,M=1,needs_gpu=np.array([1]),is_gpu_server=np.array([1]))
        np.testing.assert_array_equal(mutate(d,np.array([0]),np.random.default_rng(0)),[0])
        d['is_gpu_server'][0]=0
        with self.assertRaises(ValueError): random_individual(d,np.random.default_rng(0))

    def test_bad_input(self):
        for a in [np.zeros(41,dtype=float),np.zeros(40,dtype=int),np.full(41,9)]:
            with self.assertRaises(ValueError):evaluate(self.d,a)
        with self.assertRaises(ValueError):evaluate(self.d,np.zeros(41,dtype=int),rho=0)

if __name__=='__main__':unittest.main()
