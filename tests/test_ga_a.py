"""GA-A role test (Checkpoint 4): same seed -> same run, exactly the budget is spent,
elitism keeps the best individual, the tournament never returns the worst entrant.
Run from the repository root:  python -m unittest tests.test_ga_a -v
"""
import unittest
import numpy as np
import checkpoint4_generator_PUBLIC as gen
from src.meta.ga import solve, tournament_select


class GAATests(unittest.TestCase):
    def setUp(self):
        self.d = gen.cp34_instance(16, 'B', 3)

    def test_same_seed_same_run(self):
        a = solve(self.d, 3, 7, 3000)
        b = solve(self.d, 3, 7, 3000)
        self.assertEqual(a['solution'], b['solution'])
        self.assertEqual(a['cost'], b['cost'])
        self.assertEqual(a['trace'], b['trace'])
        self.assertEqual(a['diversity'], b['diversity'])
        c = solve(self.d, 3, 8, 3000)
        self.assertNotEqual(a['solution'], c['solution'])   # a different seed gives a different run

    def test_exact_budget(self):
        # budgets that are and are not multiples of the generation size (29 offspring + 1 elite)
        for budget in (30, 31, 59, 1000, 1001, 2999):
            r = solve(self.d, 3, 1, budget)
            self.assertEqual(r['evals'], budget, f'budget {budget}')

    def test_reported_cost_agrees_with_checker(self):
        r = solve(self.d, 3, 2, 5000)
        cost, problems = gen.evaluate_B(self.d, r['solution'])
        self.assertAlmostEqual(cost, r['cost'], delta=1e-9)
        self.assertEqual(r['feasible'], len(problems) == 0)

    def test_elitism_keeps_best(self):
        # the lowest fitness in the population never gets worse from one generation to the next
        for seed in (0, 1, 2):
            h = solve(self.d, 3, seed, 5000)['best_fitness_history']
            self.assertGreater(len(h), 10)
            for x, y in zip(h, h[1:]):
                self.assertLessEqual(y, x + 1e-12)

    def test_best_so_far_trace_never_increases(self):
        tr = solve(self.d, 3, 3, 8000)['trace']
        self.assertTrue(len(tr) >= 1)
        for (_, x), (_, y) in zip(tr, tr[1:]):
            self.assertLessEqual(y, x + 1e-12)

    def test_tournament_never_returns_worst(self):
        # re-draw the entrants with the same generator state and compare with the winner
        for s in range(2000):
            fit = np.random.default_rng(1000 + s).random(30)
            r1 = np.random.default_rng(s); r2 = np.random.default_rng(s)
            k = 2 + s % 4
            entrants = r1.choice(30, k, replace=False)
            winner = tournament_select(fit, k, r2)
            self.assertIn(winner, entrants)
            self.assertEqual(fit[winner], fit[entrants].min())
            self.assertLess(fit[winner], fit[entrants].max())   # strictly better than the worst entrant


if __name__ == '__main__':
    unittest.main()
