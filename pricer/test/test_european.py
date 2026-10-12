"""
test_european.py - Tests of the contract, the market data and the Black-Scholes-Merton pricer.

Checks the prices of the old notebook, put-call parity, the zero-maturity and
zero-volatility limits, the Bos-Vandermark discrete-dividend approximation
against the exact tree, and the input validations.
"""
import math
import unittest
from datetime import date

from pricer.binomial import binomial_price
from pricer.contract import Option, year_fraction
from pricer.european import call_option, european_price, put_option
from pricer.market import Dividend, Market


class TestEuropean(unittest.TestCase):
    def test_matches_old_notebook(self):
        args = (date(2025, 9, 1), date(2025, 9, 30), 200, 190, 0.04, 0.36)
        self.assertAlmostEqual(call_option(*args), 14.28722958551674, places=12)
        self.assertAlmostEqual(put_option(*args), 3.684352473017775, places=12)

    def test_put_call_parity_with_yield(self):
        market = Market(100, 0.05, 0.25, 0.03)
        T, K = 1.5, 110
        c = european_price(Option("call", K, T), market)
        p = european_price(Option("put", K, T), market)
        self.assertAlmostEqual(c - p, 100*math.exp(-0.03*T) - K*math.exp(-0.05*T), places=12)

    def test_zero_maturity(self):
        market = Market(105, 0.05, 0.2)
        self.assertEqual(european_price(Option("call", 100, 0), market), 5)
        self.assertEqual(european_price(Option("put", 100, 0), market), 0)

    def test_zero_volatility_and_limit(self):
        market = Market(100, 0.05, 0.0, 0.01)
        expected = 100*math.exp(-0.01) - 95*math.exp(-0.05)
        self.assertAlmostEqual(european_price(Option("call", 95, 1), market), expected, places=12)
        almost_zero = Market(100, 0.05, 1e-8, 0.01)
        self.assertAlmostEqual(european_price(Option("call", 95, 1), almost_zero), expected, places=8)

    def test_discrete_dividends_close_to_exact_tree(self):
        for dividends in [(Dividend(0.25, 3), Dividend(0.75, 3)), (Dividend(0.9, 4),)]:
            for option_type in ("call", "put"):
                option, market = Option(option_type, 100, 1), Market(100, 0.05, 0.25, dividends=dividends)
                with self.subTest(dividends=dividends, option_type=option_type):
                    exact = binomial_price(option, market, 1000, american=False)
                    self.assertLess(abs(european_price(option, market)/exact - 1), 0.005)

    def test_dividend_after_maturity_ignored(self):
        option = Option("call", 100, 1)
        with_late = Market(100, 0.05, 0.2, dividends=(Dividend(2, 5),))
        self.assertEqual(european_price(option, with_late), european_price(option, Market(100, 0.05, 0.2)))

    def test_validations(self):
        with self.assertRaises(ValueError):
            year_fraction(date(2025, 9, 30), date(2025, 9, 1))
        with self.assertRaises(ValueError):
            Option("digital", 100, 1)
        with self.assertRaises(ValueError):
            Market(100, 0.05, -0.2)
        with self.assertRaises(ValueError):
            Dividend(0, 1)
        with self.assertRaises(ValueError):
            european_price(Option("call", 100, 1), Market(10, 0.05, 0.2, dividends=(Dividend(0.5, 50),)))
        self.assertEqual(Option("Call", 100, 1).option_type, "call")


if __name__ == "__main__":
    unittest.main()
