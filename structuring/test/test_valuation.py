"""
test_valuation.py - Valuation, issuer credit, decomposition and solving of the terms.

Checks the closed forms against the Monte Carlo (within 3 standard errors),
limit cases with exact answers (barrier at the strike = plain reverse
convertible, unreachable trigger and barriers = zero-coupon bond, trigger at 0
= redemption on the first date), monotonicity in the barrier and its
monitoring, the issuer credit formula, the cost decomposition, and that a
solved term hits its target.
"""
import math
import unittest

import numpy as np

from structuring.market import UNDERLYINGS, hypothetical_market
from structuring.products import Autocallable, CapitalProtectedNote, DirectExposure, ReverseConvertible, call_spread, collar
from structuring.simulation import SimulationSettings
from structuring.valuation import analytic_value, solve_term, survival, value

MD = hypothetical_market(UNDERLYINGS["Euro Stoxx 50"], rate=0.03, carry=0.02, volatility=0.2)
FREE = MD.with_overrides(credit_spread=0.0)
SETTINGS = SimulationSettings(n_paths=40_000)


class TestClosedForms(unittest.TestCase):
    def assert_close_to_mc(self, product, market_data=MD):
        v = value(product, market_data, SETTINGS)
        mc = v.monte_carlo_value
        # the Monte Carlo of the same flows must agree with the closed form
        from structuring.simulation import simulate
        from structuring.valuation import present_values
        from pricer.monte_carlo import independent_draws, mean_and_error
        market = market_data.pricer_market()
        times, paths = simulate(product, market, SETTINGS)
        pv = present_values(product.cash_flows(paths, times, market), market.rate, market_data.credit_spread.value,
                            market_data.recovery.value) if product.issuer_risk else \
            present_values(product.cash_flows(paths, times, market), market.rate)
        mean, error = mean_and_error(independent_draws(pv, True))
        self.assertEqual(v.method, "closed form (BSM)")
        self.assertLess(abs(v.value - mean), 3*error + 1e-12)
        self.assertAlmostEqual(mc, mean)

    def test_direct_exposure_is_worth_par(self):
        self.assertEqual(value(DirectExposure(5), MD).value, 1.0)
        self.assert_close_to_mc(DirectExposure(5))

    def test_protected_note(self):
        note = CapitalProtectedNote(5, protection=1.0, participation=0.9, cap=1.6)
        self.assert_close_to_mc(note)
        self.assert_close_to_mc(CapitalProtectedNote(3, protection=0.95, participation=1.2, bearish=True, cap=0.6))
        v = value(note, FREE)
        self.assertAlmostEqual(v.bond_component, math.exp(-0.15))
        self.assertAlmostEqual(v.derivative_component, v.closed_form_option)

    def test_plain_reverse_convertible(self):
        self.assert_close_to_mc(ReverseConvertible(2, 0.07, coupon_frequency=2, barrier=None))

    def test_packages(self):
        for package in (call_spread(1, 1.0, 1.2, 2.0, fees=0.01), collar(2, 0.9, 1.1, fees=0.005)):
            with self.subTest(package=package.name):
                self.assertAlmostEqual(value(package, MD).value, 1 - package.fees)
                self.assert_close_to_mc(package)


class TestPathDependent(unittest.TestCase):
    def test_barrier_at_strike_is_plain(self):
        plain = analytic_value(ReverseConvertible(1, 0.06, barrier=None), MD.pricer_market(), 0.01, 0.4)
        barrier = value(ReverseConvertible(1, 0.06, barrier=1.0), MD, SETTINGS)
        self.assertLess(abs(barrier.value - plain), 3*barrier.std_error + 1e-4)

    def test_barrier_ordering(self):
        maturity = value(ReverseConvertible(1, 0.06, barrier=0.7), MD, SETTINGS).value
        daily = value(ReverseConvertible(1, 0.06, barrier=0.7, barrier_monitoring="daily"), MD, SETTINGS).value
        lower = value(ReverseConvertible(1, 0.06, barrier=0.5), MD, SETTINGS).value
        plain = analytic_value(ReverseConvertible(1, 0.06, barrier=None), MD.pricer_market(), 0.01, 0.4)
        self.assertLess(plain, daily)       # a barrier protects the investor...
        self.assertLess(daily, maturity)    # ...less if it is watched every day
        self.assertLess(maturity, lower)    # ...more if it is lower

    def test_autocall_limit_cases(self):
        r = 0.03
        never = Autocallable(3, 0.1, autocall_trigger=1e9, coupon_barrier=1e9, capital_barrier=1e-9)
        self.assertAlmostEqual(value(never, FREE, SETTINGS).value, math.exp(-3*r), places=10)
        first = Autocallable(3, 0.1, autocall_trigger=1e-9, coupon_barrier=1e-9)
        self.assertAlmostEqual(value(first, FREE, SETTINGS).value, 1.1*math.exp(-r), places=10)

    def test_autocall_value_moves(self):
        base = value(Autocallable(5, 0.06), MD, SETTINGS)
        richer = value(Autocallable(5, 0.08), MD, SETTINGS)
        riskier = value(Autocallable(5, 0.06, capital_barrier=0.8), MD, SETTINGS)
        self.assertGreater(richer.value, base.value)
        self.assertLess(riskier.value, base.value)
        self.assertGreater(base.std_error, 0)
        self.assertTrue(base.method.startswith("Monte Carlo"))


class TestCreditAndCosts(unittest.TestCase):
    def test_survival(self):
        np.testing.assert_allclose(survival([0, 5], 0.0, 0.4), [1, 1])
        s = math.exp(-0.01/0.6*5)
        self.assertAlmostEqual(survival([5], 0.01, 0.4)[0], s + 0.4*(1 - s))
        self.assertAlmostEqual(survival([5], 0.01, 0.0)[0], math.exp(-0.05))

    def test_credit_lowers_notes_only(self):
        note = CapitalProtectedNote(5)
        risky, free = value(note, MD), value(note, FREE)
        self.assertLess(risky.value, free.value)
        self.assertAlmostEqual(risky.risk_free_value, free.value)
        self.assertAlmostEqual(risky.credit_cost, free.value - risky.value)
        self.assertEqual(value(DirectExposure(5), MD).credit_cost, 0)

    def test_costs_split(self):
        v = value(CapitalProtectedNote(5, participation=0.5), MD, documented_fees=0.03)
        self.assertAlmostEqual(v.embedded_costs, 1 - v.value)
        self.assertAlmostEqual(v.estimated_margin, v.embedded_costs - 0.03)


class TestSolving(unittest.TestCase):
    def test_solved_terms_hit_the_target(self):
        for product in (CapitalProtectedNote(5), ReverseConvertible(1, 0.0, barrier=0.6), Autocallable(5, 0.0)):
            with self.subTest(product=product.name):
                solved, term = solve_term(product, MD, 0.97, SETTINGS)
                self.assertGreater(term, 0)
                self.assertAlmostEqual(value(solved, MD, SETTINGS).value, 0.97, places=10)

    def test_impossible_protection(self):
        # zero rates and a riskless issuer: a 100 % protection costs 100 % before any option or fee
        # (with a credit spread it is cheaper: the investor is paid for the issuer risk)
        zero = hypothetical_market(UNDERLYINGS["Euro Stoxx 50"], rate=0.0, carry=0.0).with_overrides(credit_spread=0.0)
        with self.assertRaises(ValueError):
            solve_term(CapitalProtectedNote(5), zero, 0.97)
        with self.assertRaises(ValueError):
            solve_term(DirectExposure(5), MD, 0.97)


if __name__ == "__main__":
    unittest.main()
