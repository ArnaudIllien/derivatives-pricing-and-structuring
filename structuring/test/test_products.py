"""
test_products.py - Payoffs and contract conditions of every structure, on hand-built paths.

Each test feeds explicit paths of the performance to `cash_flows` and checks
the flows against the term sheet, computed by hand: protection and
participation, caps, conversion below the strike, maturity and daily barriers,
conditional coupons with and without memory, early redemption and its first
date, option legs, and the state of a contract already alive.
"""
import math
import unittest

import numpy as np

from pricer.market import Market
from structuring.products import (Autocallable, CapitalProtectedNote, CashFlows, DirectExposure, Leg, OptionPackage,
                                  ReverseConvertible, call_spread, collar, put_spread)
from structuring.simulation import time_grid

MARKET = Market(1.0, 0.03, 0.2, 0.02)


def flows_on(product, levels_at_observations, daily_min=None):
    """Flows on one path through the given levels at the observation dates (and a daily minimum if given)."""
    times = time_grid(product, [product.maturity/2 + 1e-3])   # an interior date that can hold the dip
    obs = np.array(product.observation_times())
    log_levels = np.interp(times, np.concatenate([[0.0], obs]), np.log(np.concatenate([[1.0], levels_at_observations])))
    path = np.exp(log_levels)[None, :]
    if daily_min is not None:
        path[0, len(times)//2] = daily_min
    return product.cash_flows(path, times, MARKET)


class TestTerminalStructures(unittest.TestCase):
    def test_direct_exposure(self):
        flows = flows_on(DirectExposure(2), [1.3])
        self.assertAlmostEqual(flows.total[0], 1.3*math.exp(0.02*2))
        self.assertFalse(flows.issuer_risk)

    def test_capital_protected_note(self):
        note = CapitalProtectedNote(3, protection=1.0, participation=0.8)
        self.assertAlmostEqual(flows_on(note, [1.25]).total[0], 1 + 0.8*0.25)
        self.assertAlmostEqual(flows_on(note, [0.4]).total[0], 1.0)
        capped = CapitalProtectedNote(3, protection=0.9, participation=1.5, cap=1.2)
        self.assertAlmostEqual(flows_on(capped, [1.5]).total[0], 0.9 + 1.5*0.2)
        self.assertAlmostEqual(flows_on(capped, [0.5]).total[0], 0.9)
        self.assertEqual(capped.contractual_protection(), 0.9)

    def test_bearish_note(self):
        note = CapitalProtectedNote(1, participation=1.0, bearish=True, cap=0.7)
        self.assertAlmostEqual(flows_on(note, [0.8]).total[0], 1.2)
        self.assertAlmostEqual(flows_on(note, [0.5]).total[0], 1.3)   # floored at 70 %
        self.assertAlmostEqual(flows_on(note, [1.4]).total[0], 1.0)

    def test_note_validations(self):
        with self.assertRaises(ValueError):
            CapitalProtectedNote(1, cap=0.9)
        with self.assertRaises(ValueError):
            CapitalProtectedNote(0)


class TestReverseConvertible(unittest.TestCase):
    def test_plain(self):
        rc = ReverseConvertible(1, 0.08, coupon_frequency=2, barrier=None)
        flows = flows_on(rc, [1.0, 0.8])
        np.testing.assert_allclose(flows.amounts[0], [0.04, 0.04 + 0.8])
        self.assertAlmostEqual(flows_on(rc, [1.1, 1.3]).total[0], 1.08)

    def test_maturity_barrier(self):
        rc = ReverseConvertible(1, 0.06, barrier=0.7)
        self.assertAlmostEqual(flows_on(rc, [0.75]).total[0], 1.06)            # below strike, above barrier
        self.assertAlmostEqual(flows_on(rc, [0.65]).total[0], 0.06 + 0.65)     # converted
        self.assertAlmostEqual(flows_on(rc, [0.9], daily_min=0.5).total[0], 1.06)  # European: the dip does not count

    def test_daily_barrier(self):
        rc = ReverseConvertible(1, 0.06, barrier=0.7, barrier_monitoring="daily")
        self.assertTrue(rc.daily_monitoring)
        self.assertAlmostEqual(flows_on(rc, [0.9], daily_min=0.6).total[0], 0.06 + 0.9)    # breached, ends below strike
        self.assertAlmostEqual(flows_on(rc, [1.05], daily_min=0.6).total[0], 1.06)         # breached, ends above strike
        self.assertAlmostEqual(flows_on(rc, [0.9], daily_min=0.75).total[0], 1.06)         # never breached
        breached = ReverseConvertible(1, 0.06, barrier=0.7, barrier_monitoring="daily", barrier_breached=True)
        self.assertAlmostEqual(flows_on(breached, [0.9]).total[0], 0.06 + 0.9)

    def test_strike_and_validations(self):
        rc = ReverseConvertible(1, 0.05, barrier=None, strike=0.9)
        self.assertAlmostEqual(flows_on(rc, [0.81]).total[0], 0.05 + 0.81/0.9)
        with self.assertRaises(ValueError):
            ReverseConvertible(1, 0.05, barrier=1.2)
        with self.assertRaises(ValueError):
            ReverseConvertible(1, 0.05, barrier_monitoring="weekly")


class TestAutocallable(unittest.TestCase):
    AC = Autocallable(4, 0.08, observation_frequency=1, autocall_trigger=1.0, coupon_barrier=0.7, memory=True,
                      capital_barrier=0.6)

    def test_called_on_first_date(self):
        flows = flows_on(self.AC, [1.05, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(flows.amounts[0], [1.08, 0, 0, 0])
        self.assertEqual(flows.end_time[0], 1.0)

    def test_memory_coupons_and_later_call(self):
        flows = flows_on(self.AC, [0.9, 0.65, 0.68, 1.02])
        # year 1: coupon (>= 70 %), no call; years 2-3: missed; year 4 (maturity): 3 coupons + 100 %
        np.testing.assert_allclose(flows.amounts[0], [0.08, 0, 0, 0.08*3 + 1.0])

    def test_without_memory(self):
        ac = Autocallable(4, 0.08, memory=False)
        np.testing.assert_allclose(flows_on(ac, [0.9, 0.65, 0.68, 0.8]).amounts[0], [0.08, 0, 0, 0.08 + 1.0])

    def test_capital_loss_at_maturity(self):
        flows = flows_on(self.AC, [0.9, 0.8, 0.7, 0.55])
        np.testing.assert_allclose(flows.amounts[0], [0.08, 0.08, 0.08, 0.55])
        self.assertAlmostEqual(flows_on(self.AC, [0.9, 0.8, 0.7, 0.62]).amounts[0, -1], 1.0)  # above 60 %, no coupon

    def test_daily_capital_barrier(self):
        daily = Autocallable(4, 0.08, barrier_monitoring="daily")
        self.assertAlmostEqual(flows_on(daily, [0.9, 0.8, 0.75, 0.9], daily_min=0.5).amounts[0, -1], 0.08 + 0.9)
        self.assertAlmostEqual(flows_on(daily, [0.9, 0.8, 0.75, 0.9]).amounts[0, -1], 0.08 + 1.0)

    def test_first_call_and_state(self):
        late = Autocallable(4, 0.08, first_call=2)
        np.testing.assert_allclose(flows_on(late, [1.1, 1.1, 1, 1]).amounts[0], [0.08, 1.08, 0, 0])
        remembered = Autocallable(2, 0.08, missed_coupons=2)
        np.testing.assert_allclose(flows_on(remembered, [0.9, 0.95]).amounts[0], [0.08*3, 0.08 + 1.0])

    def test_quarterly_schedule(self):
        ac = Autocallable(1, 0.08, observation_frequency=4)
        np.testing.assert_allclose(ac.observation_times(), [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(flows_on(ac, [0.9, 1.01, 1, 1]).amounts[0], [0.02, 1.02, 0, 0])


class TestPackages(unittest.TestCase):
    def test_call_spread(self):
        package = call_spread(1, 1.0, 1.2, quantity=2.0)
        premium = package.premium(MARKET)
        cash = (1 - premium)*math.exp(0.03)
        self.assertAlmostEqual(flows_on(package, [1.1]).total[0], cash + 2*0.1)
        self.assertAlmostEqual(flows_on(package, [1.5]).total[0], cash + 2*0.2)
        self.assertAlmostEqual(flows_on(package, [0.7]).total[0], cash)

    def test_put_spread_and_collar(self):
        spread = put_spread(1, 1.0, 0.8)
        cash = (1 - spread.premium(MARKET))*math.exp(0.03)
        self.assertAlmostEqual(flows_on(spread, [0.5]).total[0], cash + 0.2)
        band = collar(1, 0.9, 1.1)
        cash = (1 - band.premium(MARKET))*math.exp(0.03)
        held = math.exp(0.02)
        self.assertAlmostEqual(flows_on(band, [0.5]).total[0], cash + held*0.5 + 0.4)
        self.assertAlmostEqual(flows_on(band, [1.5]).total[0], cash + held*1.5 - 0.4)

    def test_fees_and_validation(self):
        package = OptionPackage(1, (Leg("call", 1.0, 1.0),), fees=0.01)
        no_fees = OptionPackage(1, (Leg("call", 1.0, 1.0),))
        diff = flows_on(no_fees, [1.0]).total[0] - flows_on(package, [1.0]).total[0]
        self.assertAlmostEqual(diff, 0.01*math.exp(0.03))
        with self.assertRaises(ValueError):
            OptionPackage(1, (Leg("digital"),))
        with self.assertRaises(ValueError):
            OptionPackage(1, ())


class TestCashFlows(unittest.TestCase):
    def test_end_time(self):
        flows = CashFlows(np.array([1.0, 2.0, 3.0]), np.array([[1.1, 0, 0], [0.1, 0.1, 1.1], [0, 0, 0]]), True)
        np.testing.assert_allclose(flows.end_time, [1, 3, 3])
        np.testing.assert_allclose(flows.total, [1.1, 1.3, 0])

    def test_names_are_keyword_only(self):
        self.assertEqual(ReverseConvertible(1, 0.05).coupon_rate, 0.05)
        self.assertEqual(Autocallable(5, 0.06, name="X").name, "X")


if __name__ == "__main__":
    unittest.main()
