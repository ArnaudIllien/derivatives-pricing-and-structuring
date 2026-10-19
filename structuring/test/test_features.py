"""
test_features.py - Detailed scenarios, stress tests, structuring lab, simple-solution benchmarks and the
professional view (components, full sensitivities, convergence).

Offline, hypothetical markets. Checks that the described scenario is validated
and drives the simulations (mixture shares, medians at each horizon) and the
candidates (spread at the case horizon, structures per probabilities), that
the stress cases break the view as stated, that the components add up to the
model value, that the sensitivities have the expected signs, that the lab
reproduces its fair term, and that the benchmark verdict detects a dominated
structure.
"""
import math
import unittest

import numpy as np

from structuring.benchmarks import alternatives, compare_with_simple, mixed_allocation
from structuring.components import MODEL_LIMITS, decompose
from structuring.investor import Conviction, InvestorProfile, ScenarioCase, ScenarioView
from structuring.lab import DEFAULT_TERMS, FAMILIES, LAB_SETTINGS, build, lab_report
from structuring.market import UNDERLYINGS, hypothetical_market
from structuring.products import Autocallable, CapitalProtectedNote, DirectExposure, ReverseConvertible
from structuring.risk import convergence, full_greeks, max_contractual_loss
from structuring.scenarios import view_distribution
from structuring.simulation import regimes_for_move, simulate_regimes
from structuring.strategies import generate_candidates
from structuring.stress import stress_test
from structuring.valuation import solve_term, value

MD = hypothetical_market(UNDERLYINGS["Euro Stoxx 50"], rate=0.03, carry=0.02, volatility=0.2)
MARKET = MD.pricer_market()
MIXED = ScenarioView("Euro Stoxx 50", (ScenarioCase("Rise", 0.5, 0.08, 0.5, -0.03),
                                       ScenarioCase("Stagnation", 0.3, 0.0, 5.0, 0.0),
                                       ScenarioCase("Decline", 0.2, -0.15, 1.0, 0.05)))


class TestScenarioView(unittest.TestCase):
    def test_validation_and_probabilities(self):
        with self.assertRaises(ValueError):
            ScenarioView("X", (ScenarioCase("a", 0.5), ScenarioCase("b", 0.3)))
        with self.assertRaises(ValueError):
            ScenarioCase("a", 1.2)
        self.assertAlmostEqual(MIXED.probability("up"), 0.5)
        self.assertAlmostEqual(MIXED.probability("flat"), 0.3)
        self.assertAlmostEqual(MIXED.expected_move, 0.5*0.08 - 0.2*0.15)
        self.assertEqual(MIXED.main_direction, "up")
        self.assertEqual((MIXED.upside_target, MIXED.downside_target), (0.08, -0.15))

    def test_profile_uses_the_detailed_scenario(self):
        profile = InvestorProfile(scenarios=(MIXED,))
        self.assertIs(profile.view_on("Euro Stoxx 50"), MIXED)
        simple = profile.view_on("Gold")                       # no scenario on gold: neutral conviction
        self.assertEqual(simple.cases[0].move, 0.0)
        self.assertEqual(InvestorProfile.from_dict(profile.to_dict()), profile)
        from_conviction = ScenarioView.from_conviction(Conviction("CAC 40", "down", 0.2, 2, "rising"))
        self.assertEqual((from_conviction.cases[0].move, from_conviction.cases[0].vol_change), (-0.2, 0.05))

    def test_regimes_reach_the_stated_medians(self):
        product = DirectExposure(3)
        regimes = regimes_for_move(0.08, 0.5, 0.2, 3.0, later_move=-0.1)
        times, paths = simulate_regimes(product, regimes, 40_000, seed=1)
        half = np.argmin(np.abs(times - 0.5))
        self.assertAlmostEqual(float(np.median(paths[:, half])), 1.08, delta=0.01)
        self.assertAlmostEqual(float(np.median(paths[:, -1])), 1.08*0.9, delta=0.015)

    def test_mixture_shares(self):
        # nearly deterministic cases (volatility floored at 1 %): each case keeps its share of the paths
        view = ScenarioView("X", (ScenarioCase("up", 0.25, 0.20, 1.0, -0.19), ScenarioCase("down", 0.75, -0.20, 1.0, -0.19)))
        d = view_distribution(DirectExposure(1), MARKET, view, n_paths=4_000)
        income = math.exp(0.02)
        self.assertAlmostEqual(float(np.mean(d.totals > income)), 0.25, delta=0.01)
        self.assertAlmostEqual(float(np.median(d.totals[d.totals > income]))/income, 1.2, delta=0.01)


class TestStrategiesFromScenario(unittest.TestCase):
    def test_spread_follows_the_case_horizon(self):
        profile = InvestorProfile(scenarios=(MIXED,))
        candidates, _ = generate_candidates(profile, MIXED, MD)
        spread = next(c for c in candidates if "call spread" in c.product.name)
        self.assertEqual(spread.product.maturity, 0.5)
        self.assertIn("scenario horizon", spread.rationale)
        autocall = next(c for c in candidates if c.product.family == "Autocallable")
        self.assertIn("30% to stagnation", autocall.rationale)

    def test_bearish_scenario(self):
        bearish = ScenarioView("Euro Stoxx 50", (ScenarioCase("Fall", 0.7, -0.2, 1.0, 0.05), ScenarioCase("Flat", 0.3, 0.0, 1.0)))
        families = {c.product.family for c in generate_candidates(InvestorProfile(scenarios=(bearish,)), bearish, MD)[0]}
        self.assertNotIn("Autocallable", families)
        names = [c.product.name for c in generate_candidates(InvestorProfile(scenarios=(bearish,)), bearish, MD)[0]]
        self.assertTrue(any("put spread 100%/80%" in n for n in names))


class TestStress(unittest.TestCase):
    def test_cases_and_directions(self):
        results = {r.case.label: r for r in stress_test(DirectExposure(3), MARKET, MIXED, n_paths=4_000)}
        self.assertEqual(len(results), 7)
        self.assertGreater(results["Right, but slower"].median, results["Wrong direction"].median)
        self.assertGreater(results["Crash"].probability_of_loss, 0.7)
        self.assertLess(results["Right, then reversal"].median, results["Right, but slower"].median)

    def test_volatility_explosion_hurts_a_barrier_product(self):
        rc = ReverseConvertible(3, 0.06, barrier=0.6)
        results = {r.case.label: r for r in stress_test(rc, MARKET, MIXED, n_paths=4_000)}
        base = next(r for r in stress_test(rc, MARKET, ScenarioView("X", (ScenarioCase("Rise", 1.0, 0.08, 0.5, -0.03),)),
                                          n_paths=4_000) if r.case.label == "Your scenario")
        self.assertGreater(results["Right, volatility explodes"].probability_of_loss, base.probability_of_loss)


class TestProfessionalView(unittest.TestCase):
    def test_components_add_up(self):
        products = (DirectExposure(5), CapitalProtectedNote(5, participation=0.9, cap=1.5),
                    ReverseConvertible(2, 0.05, barrier=0.6), ReverseConvertible(1, 0.05, barrier=None),
                    Autocallable(5, 0.06), build("Option package", DEFAULT_TERMS["Option package"]))
        for product in products:
            with self.subTest(product=product.name):
                parts = [c for c in decompose(product, MD) if not c.name.startswith("  ") and c.name != "Fees"]
                self.assertAlmostEqual(sum(c.value for c in parts), value(product, MD).value, places=10)
                self.assertIn(product.family, MODEL_LIMITS)

    def test_full_greeks(self):
        note = full_greeks(CapitalProtectedNote(5), MD)
        self.assertGreater(note.delta, 0)
        self.assertGreater(note.vega, 0)
        self.assertFalse(math.isnan(note.theta_month))
        rc = full_greeks(ReverseConvertible(2, 0.05, barrier=0.6), MD)
        self.assertLess(rc.vega, 0)
        self.assertGreater(rc.delta, 0)
        monthly = full_greeks(Autocallable(2, 0.06, observation_frequency=12), MD)
        self.assertTrue(math.isnan(monthly.theta_month))     # an observation date passes within the month

    def test_convergence(self):
        rows = convergence(Autocallable(3, 0.06), MD, (2_000, 8_000, 32_000))
        errors = [e for _, _, e in rows]
        self.assertAlmostEqual(errors[0]/errors[2], 4, delta=0.8)


class TestLabAndBenchmarks(unittest.TestCase):
    def test_lab_builds_every_family(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                product = build(family, DEFAULT_TERMS[family])
                report = lab_report(product, MD, MIXED, 0.95)
                if report.fair_term:
                    name, fair = report.fair_term
                    solved = build(family, {**DEFAULT_TERMS[family], name: fair})
                    self.assertAlmostEqual(value(solved, MD, LAB_SETTINGS).value, 0.95, places=8)
                self.assertEqual(len(report.grid), 41)
        with self.assertRaises(ValueError):
            build("Swap", {"maturity": 1})

    def test_other_currency_alternative_is_not_in_the_verdict(self):
        # a USD world ETF with a much higher rate would "dominate" a EUR note: it must not decide the verdict
        world = hypothetical_market(UNDERLYINGS["MSCI ACWI (world ETF)"], rate=0.08, volatility=0.15)
        note, _ = solve_term(CapitalProtectedNote(5), MD, 0.95)
        report = compare_with_simple(note, MD, MIXED, world=world)
        self.assertIn("USD", set(report.table["currency"]))
        self.assertNotIn("ACWI", report.verdict)

    def test_mixed_allocation_has_the_same_max_loss(self):
        mix = mixed_allocation(0.1, MD, 5, "mix", "")
        self.assertAlmostEqual(max_contractual_loss(mix.product, MARKET).contractual, 0.1, places=6)
        self.assertIsNone(mixed_allocation(1.0, MD, 5, "mix", ""))

    def test_diy_note_and_dominated_structure(self):
        # a costly protected note (10 % embedded costs) against the same protection built with listed calls
        costly, _ = solve_term(CapitalProtectedNote(5), MD, 0.90)
        labels = [a.label for a in alternatives(costly, MD)]
        self.assertIn("DIY: deposit + listed options", labels)
        diy = next(a for a in alternatives(costly, MD) if a.label.startswith("DIY"))
        self.assertFalse(diy.product.issuer_risk)
        self.assertGreater(diy.product.legs[0].quantity, costly.participation)
        report = compare_with_simple(costly, MD, MIXED)
        self.assertTrue(report.verdict.startswith("A simpler solution does at least as well"))
        self.assertEqual(report.table.iloc[0]["solution"], "The structure")


if __name__ == "__main__":
    unittest.main()
