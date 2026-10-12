"""
test_greeks.py - Tests of the finite-difference Greeks.

Checks the European Greeks against the closed-form formulas, the signs and
the exercise-region values of the American put Greeks with BAW, and Theta
with a discrete dividend.
"""
import math
import unittest

from scipy.stats import norm

from pricer.baw import baw_price
from pricer.contract import Option
from pricer.european import d1, european_price
from pricer.greeks import compute_greeks
from pricer.market import Dividend, Market


class TestGreeks(unittest.TestCase):
    def test_european_against_closed_forms(self):
        S, K, T, r, q, vol = 100, 95, 0.75, 0.04, 0.02, 0.3
        greeks = compute_greeks(european_price, Option("call", K, T), Market(S, r, vol, q))
        D1 = d1(S, K, T, r, q, vol)
        D2 = D1 - vol*math.sqrt(T)
        density = norm.pdf(D1)
        self.assertAlmostEqual(greeks.delta, math.exp(-q*T)*norm.cdf(D1), places=5)
        self.assertAlmostEqual(greeks.gamma, math.exp(-q*T)*density/(S*vol*math.sqrt(T)), places=5)
        self.assertAlmostEqual(greeks.vega, S*math.exp(-q*T)*density*math.sqrt(T), places=4)
        theta = (-S*math.exp(-q*T)*density*vol/(2*math.sqrt(T))
                 - r*K*math.exp(-r*T)*norm.cdf(D2) + q*S*math.exp(-q*T)*norm.cdf(D1))
        self.assertAlmostEqual(greeks.theta, theta, delta=0.01*abs(theta))

    def test_american_put_baw(self):
        option = Option("put", 40, 1)
        greeks = compute_greeks(baw_price, option, Market(38, 0.06, 0.2))
        self.assertTrue(-1 <= greeks.delta <= 0)
        self.assertGreater(greeks.gamma, 0)
        self.assertGreater(greeks.vega, 0)
        # in the exercise region V = K - S: delta = -1 and gamma = 0
        exercise = compute_greeks(baw_price, option, Market(25, 0.06, 0.2))
        self.assertAlmostEqual(exercise.delta, -1, places=8)
        self.assertAlmostEqual(exercise.gamma, 0, places=6)

    def test_theta_moves_dividend_dates(self):
        # the dividend date gets closer as time passes; the Greeks stay finite
        market = Market(100, 0.05, 0.2, dividends=(Dividend(0.5, 2),))
        greeks = compute_greeks(european_price, Option("call", 100, 1), market)
        self.assertTrue(all(math.isfinite(g) for g in (greeks.delta, greeks.gamma, greeks.vega, greeks.theta)))
        self.assertLess(greeks.theta, 0)
        self.assertEqual(market.shifted(0.6).dividends, ())


if __name__ == "__main__":
    unittest.main()
