"""
test_baw.py - Tests of the Barone-Adesi-Whaley pricer.

Checks the lower bounds (intrinsic value, European price), the cases without
early exercise, the smooth-pasting conditions at the boundary, the accuracy
against the binomial tree, the edge cases (zero maturity, zero or extreme
volatility, deep in or out of the money, zero or negative rates) and the
discrete-dividend approximation.
"""
import math
import unittest
from dataclasses import replace

import numpy as np

from pricer.baw import baw_boundary, baw_price, baw_result, deterministic_american_price
from pricer.binomial import binomial_price
from pricer.contract import Option
from pricer.european import european_price
from pricer.market import Dividend, Market

# (option, market) pairs where early exercise is possible
SHORT_CASES = [
    (Option("call", 100, 0.1), Market(90, 0.10, 0.15, 0.10)),
    (Option("call", 100, 0.1), Market(100, 0.10, 0.15, 0.10)),
    (Option("call", 100, 0.5), Market(100, 0.05, 0.30, 0.08)),
    (Option("put", 100, 0.25), Market(95, 0.08, 0.20, 0.12)),
    (Option("put", 100, 0.5), Market(100, 0.05, 0.35)),
]
LONG_CASES = [
    (Option("put", 40, 1), Market(36, 0.06, 0.2)),
    (Option("put", 40, 2), Market(36, 0.06, 0.4)),
    (Option("put", 40, 2), Market(44, 0.06, 0.2)),
    (Option("put", 100, 3), Market(100, 0.08, 0.2, 0.04)),
    (Option("call", 100, 3), Market(100, 0.03, 0.3, 0.07)),
]
EDGE_CASES = [
    (Option("put", 100, 1e-6), Market(100, 0.05, 0.2)),
    (Option("put", 100, 1), Market(1, 0.05, 0.2)),           # deep in the money
    (Option("call", 100, 1), Market(1, 0.05, 0.2, 0.03)),    # deep out of the money
    (Option("call", 100, 1), Market(1e4, 0.05, 0.2, 0.03)),  # deep in the money
    (Option("put", 100, 1), Market(1e4, 0.05, 0.2)),         # deep out of the money
    (Option("put", 100, 1), Market(100, 0.05, 3.0)),          # extreme volatility
    (Option("put", 100, 1), Market(100, 0.05, 1e-4)),         # almost zero volatility
    (Option("call", 100, 1), Market(100, 0.0, 0.2, 0.05)),    # r = 0 (limit of M / K(T))
    (Option("call", 100, 1), Market(100, -0.01, 0.2, 0.02)),  # negative rate
    (Option("put", 100, 30), Market(100, 0.05, 0.2)),         # very long maturity
]
DIVIDEND_CASES = [
    (Option(option_type, 100, 1), Market(S, 0.05, vol, q, dividends))
    for option_type in ("call", "put") for S in (90, 100, 115) for vol in (0.25, 0.4)
    for dividends, q in [((Dividend(0.3, 2),), 0.02), ((Dividend(0.5, 1),), 0.0), ((Dividend(0.1, 2),), 0.0)]
]


class TestBAW(unittest.TestCase):
    def test_lower_bounds(self):
        for option, market in SHORT_CASES + LONG_CASES + EDGE_CASES + DIVIDEND_CASES:
            with self.subTest(option=option, market=market):
                price = baw_price(option, market)
                self.assertTrue(math.isfinite(price))
                self.assertGreaterEqual(price, option.payoff(market.spot) - 1e-12)
                self.assertGreaterEqual(price, european_price(option, market) - 1e-12)

    def test_call_without_dividend_equals_european(self):
        for rate in (0.05, 0.0, -0.01):
            option, market = Option("call", 100, 2), Market(95, rate, 0.3)
            result = baw_result(option, market)
            self.assertEqual(result.price, european_price(option, market))
            self.assertIsNone(result.boundary)

    def test_put_with_zero_rate_equals_european(self):
        option, market = Option("put", 100, 1), Market(95, 0.0, 0.3, 0.02)
        self.assertEqual(baw_price(option, market), european_price(option, market))

    def test_smooth_pasting(self):
        for option, market in SHORT_CASES + LONG_CASES:
            with self.subTest(option=option, market=market):
                boundary = baw_boundary(option, market)
                # continuation side: below S* for the call, above S** for the put
                side = -option.sign
                h = 1e-4*boundary

                def value(spot):
                    return baw_price(option, replace(market, spot=spot))

                edge = value(boundary*(1 + side*1e-12))
                self.assertAlmostEqual(edge, option.payoff(boundary), delta=1e-8*option.strike)
                # second-order one-sided derivative, taken in the continuation region
                slope = side*(-3*edge + 4*value(boundary + side*h) - value(boundary + 2*side*h))/(2*h)
                self.assertAlmostEqual(slope, option.sign, delta=1e-5)

    def test_immediate_exercise_in_region(self):
        option, market = Option("put", 40, 1), Market(36, 0.06, 0.2)
        boundary = baw_boundary(option, market)
        exercise_market = replace(market, spot=0.9*boundary)
        self.assertEqual(baw_price(option, exercise_market), option.payoff(0.9*boundary))

    def test_close_to_reference_short_maturities(self):
        for option, market in SHORT_CASES:
            with self.subTest(option=option, market=market):
                reference = binomial_price(option, market)
                self.assertLess(abs(baw_price(option, market) - reference), 0.005*reference + 1e-3)

    def test_close_to_reference_long_maturities(self):
        # BAW is an approximation: the gap grows with T, 3 % is tolerated
        for option, market in LONG_CASES:
            with self.subTest(option=option, market=market):
                reference = binomial_price(option, market)
                self.assertLess(abs(baw_price(option, market) - reference), 0.03*reference)

    def test_discrete_dividends_small(self):
        # small or early dividends: gap below 1.5 % (measured; large late dividends are worse, see baw.py)
        for option, market in DIVIDEND_CASES:
            with self.subTest(option=option, market=market):
                reference = binomial_price(option, market, 1000)
                self.assertLess(abs(baw_price(option, market)/reference - 1), 0.015)

    def test_zero_maturity(self):
        option = Option("put", 100, 0)
        self.assertEqual(baw_price(option, Market(90, 0.05, 0.2)), 10)

    def test_zero_volatility_is_exact(self):
        cases = [(Option("put", 100, 2), Market(95, 0.05, 0.0, 0.03)),
                 (Option("call", 100, 2), Market(105, 0.03, 0.0, 0.05)),
                 (Option("put", 100, 2), Market(120, 0.05, 0.0, 0.01)),
                 (Option("call", 100, 1), Market(105, 0.05, 0.0, dividends=(Dividend(0.5, 10),)))]
        for option, market in cases:
            with self.subTest(option=option, market=market):
                # brute force on a fine time grid, with the spot dropping at each dividend
                t = np.linspace(0, option.maturity, 200_001)
                spot = market.spot*np.exp(market.carry*t)
                for d in market.dividends_before(option.maturity):
                    after = t > d.time
                    spot[after] -= d.amount*np.exp(market.carry*(t[after] - d.time))
                gains = option.sign*(spot - option.strike)*np.exp(-market.rate*t)
                expected = max(gains.max(), 0.0)
                self.assertAlmostEqual(deterministic_american_price(option, market), expected, places=6)
                self.assertEqual(baw_price(option, market), deterministic_american_price(option, market))

    def test_increasing_in_maturity(self):
        market = Market(100, 0.05, 0.25)
        prices = [baw_price(Option("put", 100, T), market) for T in (0.1, 0.5, 1, 2, 5)]
        self.assertEqual(prices, sorted(prices))


if __name__ == "__main__":
    unittest.main()
