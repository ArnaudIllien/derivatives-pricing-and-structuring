"""
test_lsm.py - Tests of the Longstaff-Schwartz Monte Carlo pricer.

Checks reproducibility with a seed, the comparison with the binomial tree
(LSM out-of-sample is biased low), the lower bounds, the 1/sqrt(N) decay of
the standard error, variance reduction, convergence in the number of exercise
dates, discrete dividends and the edge cases.
"""
import unittest

from pricer.baw import deterministic_american_price
from pricer.binomial import bermudan_binomial_price, binomial_price
from pricer.contract import Option
from pricer.european import european_price
from pricer.lsm import LSMParameters, lsm_price
from pricer.market import Dividend, Market

PUT = Option("put", 40, 1)
MARKET = Market(36, 0.06, 0.2)
FAST = LSMParameters(n_paths=40_000, n_dates=50)


class TestLSM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = binomial_price(PUT, MARKET)
        cls.result = lsm_price(PUT, MARKET, FAST)

    def test_reproducible_with_seed(self):
        self.assertEqual(lsm_price(PUT, MARKET, FAST), self.result)
        self.assertNotEqual(lsm_price(PUT, MARKET, LSMParameters(n_paths=40_000, seed=1)).price, self.result.price)

    def test_against_reference(self):
        # out of sample, LSM is biased low (sub-optimal policy, discrete dates):
        # it must not exceed the reference beyond the noise, and the bias must stay small
        price, error = self.result.price, self.result.std_error
        self.assertLess(price, self.reference + 3*error)
        self.assertGreater(price, 0.985*self.reference)

    def test_bounded_by_exact_bermudan(self):
        # same exercise grid: only the regression bias is left, downwards out of sample
        bermudan = bermudan_binomial_price(PUT, MARKET, FAST.n_dates)
        self.assertLess(self.result.price, bermudan + 3*self.result.std_error)

    def test_lower_bounds(self):
        self.assertGreaterEqual(self.result.price, PUT.payoff(MARKET.spot))
        self.assertGreater(self.result.price, european_price(PUT, MARKET) + 3*self.result.std_error)

    def test_call_without_dividend_close_to_european(self):
        call, market = Option("call", 40, 1), Market(36, 0.06, 0.2)
        result = lsm_price(call, market, FAST)
        self.assertLess(abs(result.price - european_price(call, market)), 3*result.std_error)

    def test_discrete_dividends_against_tree(self):
        market = Market(100, 0.05, 0.25, dividends=(Dividend(0.25, 3), Dividend(0.75, 3)))
        for option_type in ("call", "put"):
            option = Option(option_type, 100, 1)
            with self.subTest(option_type=option_type):
                reference = binomial_price(option, market, 1000)
                result = lsm_price(option, market, FAST)
                self.assertLess(result.price, reference + 3*result.std_error)
                self.assertGreater(result.price, 0.97*reference)

    def test_std_error_as_root_n(self):
        small = lsm_price(PUT, MARKET, LSMParameters(n_paths=10_000)).std_error
        large = lsm_price(PUT, MARKET, LSMParameters(n_paths=40_000)).std_error
        self.assertAlmostEqual(small/large, 2, delta=0.3)

    def test_antithetic_reduces_variance(self):
        plain = lsm_price(PUT, MARKET, LSMParameters(n_paths=40_000, antithetic=False))
        self.assertLess(self.result.std_error, plain.std_error)

    def test_more_dates_raise_the_price(self):
        # Bermudan with 5 dates <= option with 50 dates <= American
        bermudan = lsm_price(PUT, MARKET, LSMParameters(n_paths=40_000, n_dates=5))
        self.assertLess(bermudan.price, self.result.price)

    def test_in_sample(self):
        result = lsm_price(PUT, MARKET, LSMParameters(n_paths=20_000, out_of_sample=False))
        self.assertEqual(result.price, result.in_sample_price)

    def test_laguerre_basis(self):
        result = lsm_price(PUT, MARKET, LSMParameters(n_paths=40_000, basis="laguerre", degree=3))
        self.assertLess(abs(result.price - self.reference), 0.01*self.reference)

    def test_edge_cases(self):
        self.assertEqual(lsm_price(Option("put", 40, 0), MARKET).price, 4)
        deep_in = lsm_price(PUT, Market(5, 0.06, 0.2), FAST)
        self.assertEqual(deep_in.price, 35)
        self.assertTrue(deep_in.exercise_now)
        self.assertEqual(lsm_price(PUT, Market(400, 0.06, 0.2), FAST).price, 0)

    def test_zero_volatility(self):
        option, market = Option("put", 100, 2), Market(95, 0.05, 0.0, 0.03)
        result = lsm_price(option, market, LSMParameters(n_paths=1_000, n_dates=200))
        exact = deterministic_american_price(option, market)
        self.assertEqual(result.std_error, 0)
        self.assertLessEqual(result.price, exact + 1e-12)
        self.assertAlmostEqual(result.price, exact, delta=1e-3)

    def test_validations(self):
        with self.assertRaises(ValueError):
            lsm_price(PUT, MARKET, LSMParameters(n_paths=1_001))
        with self.assertRaises(ValueError):
            lsm_price(PUT, MARKET, LSMParameters(basis="hermite"))


if __name__ == "__main__":
    unittest.main()
