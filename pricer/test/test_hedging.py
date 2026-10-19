"""
test_hedging.py - Tests of the delta-hedging backtest.

Checks the vectorised BSM delta against finite differences, the hedge
accounting on hand-checkable cases, the theory on simulated paths (zero mean
P&L when the hedge volatility is the true one, error falling with the
rebalancing frequency, sign and size of the P&L when the volatility is wrong,
explained by the gamma P&L), transaction costs, the historical backtest on
the fake Yahoo client and the input checks.
"""
import math
import unittest
from datetime import date

import numpy as np

from pricer.contract import Option
from pricer.european import bsm_price, bsm_price_and_delta
from pricer.hedging import HedgeParameters, delta_hedge, historical_hedge, simulated_hedge
from pricer.market import Dividend, Market
from pricer.test.fake_yahoo import AS_OF, FakeYahoo

OPTION, MARKET = Option("call", 100, 1), Market(100, 0.03, 0.2, 0.01)


class TestDelta(unittest.TestCase):
    def test_vectorised_price_and_delta(self):
        spots = np.array([80.0, 100.0, 125.0])
        for is_call in (True, False):
            prices, deltas = bsm_price_and_delta(is_call, spots, 100, 0.5, 0.03, 0.01, 0.2)
            for S, price, delta in zip(spots, prices, deltas):
                h = 1e-4*S
                bumped = (bsm_price(is_call, S + h, 100, 0.5, 0.03, 0.01, 0.2)
                          - bsm_price(is_call, S - h, 100, 0.5, 0.03, 0.01, 0.2))/(2*h)
                with self.subTest(is_call=is_call, spot=S):
                    self.assertAlmostEqual(price, bsm_price(is_call, S, 100, 0.5, 0.03, 0.01, 0.2), places=10)
                    self.assertAlmostEqual(delta, bumped, places=6)


class TestAccounting(unittest.TestCase):
    def test_pnl_starts_at_zero_and_ends_with_the_payoff(self):
        times = np.array([0, 0.5, 1.0])
        result = delta_hedge(OPTION, times, [[100, 110, 120]], 0.03, 0.0, HedgeParameters(0.2))
        self.assertEqual(result.pnl[0, 0], 0)
        self.assertEqual(result.option_values[0, -1], 20)
        self.assertEqual(result.deltas[0, -1], 0)
        self.assertAlmostEqual(result.initial_price, bsm_price(True, 100, 100, 1, 0.03, 0, 0.2), places=10)

    def test_deterministic_path_without_rebalancing(self):
        # one hedge at t0, never rebalanced: P&L = premium e^{rT} + delta0 (S_T - S0 e^{rT}) - payoff
        times, path = np.linspace(0, 1, 5), np.array([[100, 102, 98, 105, 110]])
        result = delta_hedge(OPTION, times, path, 0.03, 0.0, HedgeParameters(0.2, rebalance_every=100))
        delta0 = result.deltas[0, 0]
        expected = result.initial_price*math.exp(0.03) + delta0*(110 - 100*math.exp(0.03)) - 10
        self.assertAlmostEqual(result.final_pnl[0], expected, places=10)
        np.testing.assert_allclose(result.deltas[0, :-1], delta0)


class TestSimulated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily = simulated_hedge(OPTION, MARKET, HedgeParameters(0.2), n_paths=2_000)
        cls.weekly = simulated_hedge(OPTION, MARKET, HedgeParameters(0.2, rebalance_every=5), n_paths=2_000)

    def test_fair_volatility_gives_zero_mean(self):
        final = self.daily.final_pnl
        self.assertLess(abs(final.mean()), 3*final.std(ddof=1)/math.sqrt(len(final)))
        self.assertLess(final.std(), 0.1*self.daily.initial_price)

    def test_error_falls_with_rebalancing_frequency(self):
        # hedging error std ~ 1/sqrt(number of rebalancings): weekly / daily ~ sqrt(5)
        ratio = self.weekly.final_pnl.std()/self.daily.final_pnl.std()
        self.assertAlmostEqual(ratio, math.sqrt(5), delta=0.4)

    def test_wrong_volatility_and_gamma_pnl(self):
        # selling at 25 % when the true vol is 20 % earns about the premium gap
        rich = simulated_hedge(OPTION, MARKET, HedgeParameters(0.25), n_paths=2_000)
        cheap = simulated_hedge(OPTION, MARKET, HedgeParameters(0.15), n_paths=2_000)
        gap = bsm_price(True, 100, 100, 1, 0.03, 0.01, 0.25) - bsm_price(True, 100, 100, 1, 0.03, 0.01, 0.2)
        self.assertGreater(rich.final_pnl.mean(), 0.8*gap*math.exp(0.03))
        self.assertLess(cheap.final_pnl.mean(), 0)
        for result in (rich, cheap, self.daily):
            # the gamma P&L explains the realised P&L path by path
            residual = result.final_pnl - result.gamma_pnl
            self.assertLess(residual.std(), 0.5*result.final_pnl.std() + 0.3)
        self.assertAlmostEqual(self.daily.realized_volatility.mean(), 0.2, delta=0.005)

    def test_transaction_costs(self):
        # same paths (same seed): the P&L drops by the costs, carried to T at the rate r
        costly = simulated_hedge(OPTION, MARKET, HedgeParameters(0.2, transaction_cost=5e-4), n_paths=2_000)
        loss = self.daily.final_pnl - costly.final_pnl
        self.assertTrue(np.all(costly.costs > 0))
        self.assertTrue(np.all(loss >= costly.costs - 1e-12))
        self.assertTrue(np.all(loss <= costly.costs*math.exp(0.03) + 1e-12))

    def test_summary(self):
        summary = self.daily.summary()
        self.assertEqual(summary["premium"], self.daily.initial_price)
        self.assertIn("std P&L", summary)


class TestHistoricalAndChecks(unittest.TestCase):
    def test_historical_hedge_on_fake_prices(self):
        client = FakeYahoo()
        start, end = date(2025, 10, 13), AS_OF
        result = historical_hedge("FAKE", start, end, "put", 0.95, HedgeParameters(0.25), client)
        prices = client.history("FAKE", start=start, end=end)
        self.assertAlmostEqual(result.spots[0, 0], prices.iloc[0])
        self.assertAlmostEqual(result.times[-1], (end - start).days/365)
        self.assertEqual(result.spots.shape, (1, len(prices)))
        self.assertTrue(abs(result.final_pnl[0]) < result.initial_price)

    def test_validations(self):
        with self.assertRaises(ValueError):
            delta_hedge(OPTION, [0, 0.5], [[100, 101, 102]], 0.03, 0, HedgeParameters(0.2))
        with self.assertRaises(ValueError):
            delta_hedge(OPTION, [0, 0.5], [[100, 101]], 0.03, 0, HedgeParameters(0.2))
        with self.assertRaises(ValueError):
            delta_hedge(OPTION, [0, 1], [[100, 101]], 0.03, 0, HedgeParameters(0.0))
        with self.assertRaises(ValueError):
            simulated_hedge(OPTION, Market(100, 0.03, 0.2, dividends=(Dividend(0.5, 1),)), HedgeParameters(0.2))
        with self.assertRaises(ValueError):
            historical_hedge("NOPE", date(2025, 1, 2), date(2025, 6, 30), "call", 1, HedgeParameters(0.2), FakeYahoo())


if __name__ == "__main__":
    unittest.main()
