"""
test_bermudan.py - Tests of the Bermudan pricers: LSM on given exercise dates and the CRR reference tree.

Checks the tree on its own (limit cases, ordering European <= Bermudan <=
American, more dates worth more), then LSM against the tree (out of sample,
LSM is biased low: price < tree + 3 s.e. and a bounded bias), the consistency
with the American LSM on the same grid, the exercise at t = 0, discrete
dividends (cum-dividend exercise on a dividend date) and the input checks.
"""
import unittest

from pricer.binomial import bermudan_binomial_price, bermudan_tree_price, binomial_price
from pricer.contract import Option
from pricer.european import european_price
from pricer.lsm import LSMParameters, bermudan_lsm_price, lsm_price
from pricer.market import Dividend, Market

PUT = Option("put", 40, 1)
MARKET = Market(36, 0.06, 0.2)
QUARTERLY = (0.25, 0.5, 0.75)
FAST = LSMParameters(n_paths=40_000)


class TestBermudanTree(unittest.TestCase):
    def test_single_date_is_european(self):
        european = binomial_price(PUT, MARKET, american=False)
        self.assertAlmostEqual(bermudan_tree_price(PUT, MARKET, ()), european, places=12)
        self.assertAlmostEqual(bermudan_tree_price(PUT, MARKET, (1,)), european, places=12)

    def test_regular_dates_match_the_regular_tree(self):
        dates = [m/10 for m in range(11)]
        self.assertAlmostEqual(bermudan_tree_price(PUT, MARKET, dates), bermudan_binomial_price(PUT, MARKET, 10),
                               delta=2e-3)

    def test_ordering(self):
        half_yearly = bermudan_tree_price(PUT, MARKET, (0.5,))
        quarterly = bermudan_tree_price(PUT, MARKET, QUARTERLY)
        self.assertLess(european_price(PUT, MARKET), half_yearly)
        self.assertLess(half_yearly, quarterly)
        self.assertLess(quarterly, binomial_price(PUT, MARKET))

    def test_validations(self):
        with self.assertRaises(ValueError):
            bermudan_tree_price(PUT, MARKET, (0.5, 1.2))
        with self.assertRaises(ValueError):
            bermudan_tree_price(PUT, Market(36, 0.06, 0.0), QUARTERLY)


class TestBermudanLSM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = bermudan_tree_price(PUT, MARKET, QUARTERLY)
        cls.result = bermudan_lsm_price(PUT, MARKET, QUARTERLY, FAST)

    def test_against_tree(self):
        price, error = self.result.price, self.result.std_error
        self.assertLess(price, self.reference + 3*error)
        self.assertGreater(price, 0.99*self.reference)

    def test_reproducible_and_order_free(self):
        self.assertEqual(bermudan_lsm_price(PUT, MARKET, (0.75, 0.25, 0.5, 0.5), FAST), self.result)

    def test_between_european_and_american(self):
        self.assertGreater(self.result.price, european_price(PUT, MARKET) + 3*self.result.std_error)
        self.assertLess(self.result.price, binomial_price(PUT, MARKET))

    def test_single_date_is_european(self):
        # no regression at all: plain Monte Carlo of the European payoff
        result = bermudan_lsm_price(PUT, MARKET, (), FAST)
        european = european_price(PUT, MARKET)
        self.assertLess(abs(result.price - european), 3*result.std_error)
        self.assertLess(abs(result.in_sample_price - european), 3*result.std_error)
        self.assertEqual(bermudan_lsm_price(PUT, MARKET, (1,), FAST), result)

    def test_regular_grid_is_the_american_lsm(self):
        # same grid, same seeds: the Bermudan entry point reproduces lsm_price
        dates = [m/50 for m in range(51)]
        self.assertAlmostEqual(bermudan_lsm_price(PUT, MARKET, dates, FAST).price, lsm_price(PUT, MARKET, FAST).price,
                               places=10)

    def test_in_sample(self):
        result = bermudan_lsm_price(PUT, MARKET, QUARTERLY, LSMParameters(n_paths=20_000, out_of_sample=False))
        self.assertEqual(result.price, result.in_sample_price)

    def test_exercise_at_zero_only_if_listed(self):
        market = Market(5, 0.06, 0.2)   # deep in the money: exercising now is optimal
        now = bermudan_lsm_price(PUT, market, (0,) + QUARTERLY, FAST)
        self.assertEqual(now.price, 35)
        self.assertTrue(now.exercise_now)
        later = bermudan_lsm_price(PUT, market, QUARTERLY, FAST)
        self.assertFalse(later.exercise_now)
        self.assertLess(later.price, 35)
        self.assertAlmostEqual(later.price, bermudan_tree_price(PUT, market, QUARTERLY), delta=3*later.std_error + 1e-3)

    def test_discrete_dividends_against_tree(self):
        # exercise dates on and around the dividend date, cum-dividend exercise in both models
        market = Market(100, 0.05, 0.25, dividends=(Dividend(0.5, 5),))
        dates = (0.25, 0.5, 0.75)
        for option_type in ("call", "put"):
            option = Option(option_type, 100, 1)
            with self.subTest(option_type=option_type):
                reference = bermudan_tree_price(option, market, dates, 1000)
                result = bermudan_lsm_price(option, market, dates, FAST)
                self.assertLess(result.price, reference + 3*result.std_error)
                self.assertGreater(result.price, 0.98*reference)

    def test_call_exercised_before_dividend(self):
        # exercising just before a large dividend is worth more than waiting
        option, market = Option("call", 100, 1), Market(110, 0.05, 0.25, dividends=(Dividend(0.5, 8),))
        european = bermudan_lsm_price(option, market, (), FAST)
        on_dividend = bermudan_lsm_price(option, market, (0.5,), FAST)
        self.assertGreater(on_dividend.price, european.price + 3*(on_dividend.std_error + european.std_error))

    def test_validations(self):
        with self.assertRaises(ValueError):
            bermudan_lsm_price(PUT, MARKET, (-0.1, 0.5))
        with self.assertRaises(ValueError):
            bermudan_lsm_price(PUT, MARKET, (0.5, 1.5))
        with self.assertRaises(ValueError):
            bermudan_lsm_price(PUT, MARKET, QUARTERLY, LSMParameters(n_paths=1_001))
        self.assertEqual(bermudan_lsm_price(Option("put", 40, 0), MARKET, ()).price, 4)


if __name__ == "__main__":
    unittest.main()
