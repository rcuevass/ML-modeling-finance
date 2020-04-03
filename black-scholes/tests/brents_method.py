import unittest
from utils.solvers import solve_for_volatility


class TestBrentRootFinder(unittest.TestCase):

    def test_root_accuracy(self):
        # provide values for arguments for Brent's solver
        s_test = 3.7442436500446084e-06
        sox_test = 0.70314451
        r_test = 0.04635324
        t_test = 0.91229427

        root_computed = solve_for_volatility(s_test, sox_test, r_test, t_test)
        root_test = 0.0919281
        delta_computed = abs(root_computed-root_test)
        delta_test = 1.e-06

        self.assertLessEqual(delta_computed, delta_test, "Should be less than 1e-06")


if __name__ == '__main__':
    unittest.main()
