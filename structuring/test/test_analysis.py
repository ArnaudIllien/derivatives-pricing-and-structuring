"""
test_analysis.py - Scenarios, risk measures, strategy generation, suitability, comparison, market data and client.

Offline: every market is hypothetical. Checks the deterministic path scenarios
(the autocallable follows the path), the drift of the investor scenario, the
consistency of the risk-neutral distribution with the value, the contractual
maximum losses, the signs of the sensitivities, the early-exit states, the
candidate families per view, the suitability checks and verdicts, the
provenance tags, the profile validation and the text report of the client.
"""
import math
import unittest

import numpy as np

from structuring.analysis import analyse
from structuring.client import demo_profile, main, report, run_analysis
from structuring.comparison import comparison_table
from structuring.investor import Conviction, InvestorProfile, Objectives, RiskConstraints
from structuring.market import UNDERLYINGS, Parameter, Provenance, _curve_rate, hypothetical_market
from structuring.products import Autocallable, CapitalProtectedNote, DirectExposure, ReverseConvertible, call_spread
from structuring.risk import early_exit_values, max_contractual_loss, sensitivities, state_after
from structuring.scenarios import conviction_distribution, maturity_grid, path_scenarios, return_distribution
from structuring.simulation import SimulationSettings, drift_from_view, simulate
from structuring.strategies import CostAssumptions, generate_candidates, product_maturity
from structuring.suitability import FAIL, PASS, WARNING, check_product, verdict
from structuring.valuation import value

MD = hypothetical_market(UNDERLYINGS["Euro Stoxx 50"], rate=0.03, carry=0.02, volatility=0.2)
MARKET = MD.pricer_market()


class TestScenarios(unittest.TestCase):
    def test_maturity_grid_of_a_protected_note(self):
        grid = maturity_grid(CapitalProtectedNote(5, participation=0.5), MARKET, [0.0, 0.5, 1.0, 1.5])
        self.assertEqual([round(o.total_received, 10) for o in grid], [1.0, 1.0, 1.0, 1.25])

    def test_autocall_follows_the_path(self):
        ac = Autocallable(5, 0.07)
        outcomes = {o.label: o for o in path_scenarios(ac, MARKET, 0.10)}
        self.assertAlmostEqual(outcomes["Stagnation"].total_received, 1.07)   # called at the first date
        self.assertEqual(outcomes["Stagnation"].end_time, 1.0)
        self.assertAlmostEqual(outcomes["Crash then partial recovery"].total_received, 1 + 5*0.07)   # memory
        # with memory, the coupons missed during a dip are paid back at maturity; without memory they are lost
        for memory, lost in ((True, False), (False, True)):
            product = Autocallable(5, 0.07, memory=memory)
            dip = maturity_grid(product, MARKET, [0.7], trough=0.5)[0]
            straight = maturity_grid(product, MARKET, [0.7])[0]
            self.assertEqual(dip.total_received < straight.total_received - 1e-9, lost)

    def test_drift_from_view(self):
        mu = drift_from_view(0.10, 3, 0.2)
        settings = SimulationSettings(n_paths=40_000, seed=3)
        times, paths = simulate(DirectExposure(3), MARKET, settings, drift=mu)
        self.assertAlmostEqual(float(np.median(paths[:, -1])), 1.10, delta=0.01)

    def test_risk_neutral_distribution_matches_the_value(self):
        note = CapitalProtectedNote(4, participation=0.8)
        d = return_distribution(note, MARKET, None)
        free = value(note, MD.with_overrides(credit_spread=0.0)).value
        self.assertAlmostEqual(d.expected_total*math.exp(-0.03*4), free, delta=0.01)
        self.assertEqual(d.probability_of_loss, 0.0)

    def test_conviction_distribution(self):
        d = conviction_distribution(DirectExposure(3), MARKET, 0.10, 3, "rising")
        self.assertAlmostEqual(d.volatility, 0.25)
        self.assertIn("rising", d.measure)
        summary = d.summary()
        self.assertLess(summary["adverse (10th pct)"], summary["central (median)"])
        self.assertLess(summary["central (median)"], summary["favourable (90th pct)"])
        self.assertGreater(summary["probability of loss"], 0)


class TestRisk(unittest.TestCase):
    def test_max_losses(self):
        self.assertAlmostEqual(max_contractual_loss(DirectExposure(5), MARKET).contractual, 1.0)
        protected = max_contractual_loss(CapitalProtectedNote(5), MARKET, recovery=0.4)
        self.assertEqual(protected.contractual, 0.0)
        self.assertAlmostEqual(protected.with_issuer_default, 0.6)
        rc = max_contractual_loss(ReverseConvertible(2, 0.05, barrier=0.6), MARKET)
        self.assertAlmostEqual(rc.contractual, 1 - 2*0.05)
        partial = max_contractual_loss(CapitalProtectedNote(5, protection=0.9), MARKET)
        self.assertAlmostEqual(partial.contractual, 0.1)
        self.assertAlmostEqual(max_contractual_loss(DirectExposure(5), MARKET).with_issuer_default, 1.0)

    def test_sensitivity_signs(self):
        settings = SimulationSettings(n_paths=20_000)
        protected = sensitivities(CapitalProtectedNote(5), MD, settings)
        self.assertGreater(protected.vega, 0)     # long an option
        self.assertLess(protected.credit, 0)
        rc = sensitivities(ReverseConvertible(1, 0.06, barrier=0.7), MD, settings)
        self.assertLess(rc.vega, 0)               # short a put
        self.assertEqual(sensitivities(DirectExposure(3), MD, settings).credit, 0)

    def test_states_after_one_year(self):
        ac = Autocallable(5, 0.06)
        self.assertIsNone(state_after(ac, 1.0, 1.0).product)                 # flat market: called at par
        low = state_after(ac, 1.0, 0.65)
        self.assertEqual(low.product.missed_coupons, 1)
        self.assertEqual(low.received, 0)
        self.assertEqual(low.product.maturity, 4)
        rc = state_after(ReverseConvertible(3, 0.05, barrier=0.7, barrier_monitoring="daily"), 1.0, 0.6)
        self.assertTrue(rc.product.barrier_breached)
        self.assertAlmostEqual(rc.received, 0.05)

    def test_early_exit(self):
        exits = {e.label: e for e in early_exit_values(CapitalProtectedNote(5), MD)}
        self.assertLess(exits["spot -20 %"].model_value, exits["unchanged"].model_value)
        self.assertLess(exits["rates +100 bp"].model_value, exits["unchanged"].model_value)
        self.assertAlmostEqual(exits["unchanged"].indicative_bid, exits["unchanged"].model_value - 0.01)
        package = {e.label: e for e in early_exit_values(call_spread(2, 1.0, 1.2), MD)}
        self.assertGreater(package["spot +20 %"].model_value, package["unchanged"].model_value)


class TestStrategiesAndSuitability(unittest.TestCase):
    def test_families_per_view(self):
        profile = demo_profile()
        for direction, expected, absent in (("up", {"Autocallable", "Option strategy", "Capital protection"}, set()),
                                            ("down", {"Option strategy", "Capital protection"}, {"Autocallable"}),
                                            ("flat", {"Reverse convertible", "Autocallable"}, set())):
            conviction = Conviction("Euro Stoxx 50", direction, 0.10, 3)
            candidates, _ = generate_candidates(profile, conviction, MD, settings=SimulationSettings(n_paths=4_000))
            families = {c.product.family for c in candidates}
            with self.subTest(direction=direction):
                self.assertIn("Direct exposure", families)
                self.assertTrue(expected <= families)
                self.assertFalse(absent & families)
        names = [c.product.name for c in generate_candidates(profile, Conviction("Euro Stoxx 50", "down", 0.1, 3), MD,
                                                             settings=SimulationSettings(n_paths=4_000))[0]]
        self.assertTrue(any("bearish" in n for n in names) and any("put spread" in n for n in names))

    def test_package_respects_the_loss_budget(self):
        profile = demo_profile()
        candidates, _ = generate_candidates(profile, profile.convictions[0], MD, CostAssumptions(package_fees=0.005),
                                            SimulationSettings(n_paths=4_000))
        spread = next(c.product for c in candidates if "call spread" in c.product.name)
        self.assertAlmostEqual(spread.premium(MARKET) + spread.fees, 0.10)
        loss = max_contractual_loss(spread, MARKET).contractual
        self.assertLessEqual(loss, 0.10)

    def test_maturity(self):
        profile = InvestorProfile(Objectives(horizon=7.5), constraints=RiskConstraints(max_lockup_years=6))
        self.assertEqual(product_maturity(profile), 6.0)

    def test_checks(self):
        cautious = InvestorProfile(Objectives(main_objective="capital protection", needs_regular_income=True),
                                   constraints=RiskConstraints(accepts_issuer_credit_risk=False, knowledge="basic",
                                                               accepts_capital_at_risk=False))
        rc = ReverseConvertible(5, 0.05, barrier=0.6)
        checks = check_product(rc, cautious, max_loss=0.75, max_loss_with_default=0.75, currency="USD",
                               scenario_outcomes={"Crash then partial recovery": 0.8}, conviction_summary={},
                               worst_exit_loss=0.3, total_cost=0.05)
        status = {c.criterion: c.status for c in checks}
        self.assertEqual(status["Capital protection"], FAIL)
        self.assertEqual(status["Issuer credit risk"], FAIL)
        self.assertEqual(status["Knowledge"], FAIL)
        self.assertEqual(status["Currency"], WARNING)
        self.assertEqual(status["Regular income"], PASS)
        self.assertEqual(status["Feared scenario: crash"], FAIL)
        self.assertEqual(verdict(checks), "Not consistent with the stated constraints")

    def test_protection_without_issuer(self):
        # a package whose premium is covered by the interest preserves the capital without an issuer
        cautious = InvestorProfile(Objectives(main_objective="capital protection"),
                                   constraints=RiskConstraints(accepts_issuer_credit_risk=False))
        checks = check_product(call_spread(5, 1.0, 1.2), cautious, max_loss=0.0, max_loss_with_default=0.0,
                               currency="EUR", scenario_outcomes={}, conviction_summary={}, worst_exit_loss=None,
                               total_cost=0.0)
        status = {c.criterion: c.status for c in checks}
        self.assertEqual(status["Capital protection"], PASS)
        self.assertNotIn("Issuer credit risk", status)

    def test_full_analysis_and_table(self):
        profile = demo_profile()
        analyses, skipped = run_analysis(profile, MD)
        table = comparison_table(analyses)
        self.assertEqual(len(table), len(analyses))
        self.assertEqual(table["structure"].iloc[0], "Direct exposure")    # generation order, not ranked
        protected = next(a for a in analyses if a.product.name.startswith("100 %"))
        self.assertEqual(protected.max_loss.contractual, 0.0)
        self.assertNotEqual(protected.verdict, "Not consistent with the stated constraints")
        direct = analyses[0]
        self.assertEqual(direct.verdict, "Not consistent with the stated constraints")   # 100 % loss > 10 % accepted
        self.assertFalse(skipped)


class TestMarketProfileClient(unittest.TestCase):
    def test_provenance(self):
        self.assertTrue(all(p.provenance == Provenance.HYPOTHETICAL for p in MD.parameters().values()))
        observed = MD.__class__(**{**MD.__dict__, "volatility": Parameter(0.15, Provenance.ESTIMATED, "hist")})
        overridden = observed.with_overrides(volatility=0.3)
        self.assertEqual(overridden.volatility.provenance, Provenance.HYPOTHETICAL)
        self.assertEqual(overridden.volatility.value, 0.3)
        gold = hypothetical_market(UNDERLYINGS["Gold"], rate=0.04)
        self.assertEqual(gold.carry.value, 0.04)                    # q = r for a future
        self.assertEqual(gold.with_overrides(rate=0.05).carry.value, 0.05)

    def test_curve_conversion(self):
        import pandas as pd
        curve = pd.DataFrame({1.0: [3.0], 5.0: [4.0], 10.0: [5.0]}, index=pd.to_datetime(["2026-10-09"]))
        rate, as_of = _curve_rate(curve, 7.5, "annual")
        self.assertAlmostEqual(rate, math.log(1.045))
        self.assertEqual(str(as_of), "2026-10-09")
        self.assertAlmostEqual(_curve_rate(curve, 1.0, "semi-annual")[0], 2*math.log(1.015))

    def test_profile(self):
        profile = demo_profile()
        self.assertEqual(InvestorProfile.from_dict(profile.to_dict()), profile)
        self.assertEqual(profile.max_loss_amount(), 10_000)
        self.assertEqual(Conviction(direction="down", expected_move=0.2).signed_move, -0.2)
        self.assertEqual(profile.conviction_on("Gold").direction, "flat")
        for bad in (lambda: Objectives(capital=-1), lambda: Conviction(direction="sideways"),
                    lambda: RiskConstraints(max_loss_fraction=1.5), lambda: RiskConstraints(feared_scenarios=("ufo",)),
                    lambda: InvestorProfile(convictions=())):
            with self.assertRaises(ValueError):
                bad()

    def test_client_report(self):
        text = report(demo_profile(), MD)
        self.assertIn("[hypothetical]", text)
        self.assertIn("100 % capital-protected note", text)
        self.assertIn("fit with profile", text)


if __name__ == "__main__":
    unittest.main()
