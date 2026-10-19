"""
test_client.py - Tests of the interactive client, driven by scripted answers.

Checks the number and time-list parsing, the input validation loops, the model
menu per product, that `price_request` returns the same prices as the pricers
called directly (European, American, Bermudan, Asian), and full sessions from
the questions to the printed report.
"""
import unittest
from datetime import date

from pricer.asian import AsianParameters, asian_price
from pricer.baw import baw_price
from pricer.binomial import bermudan_tree_price, binomial_price
from pricer.client import (AMERICAN, ASIAN, BAW, BERMUDAN, BINOMIAL, BSM, EUROPEAN, LSM, MONTE_CARLO, PricingRequest,
                           ask_number, ask_times, available_models, collect_request, main, parse_number, parse_times,
                           price_request)
from pricer.contract import AsianOption, Option
from pricer.european import european_price
from pricer.lsm import LSMParameters, bermudan_lsm_price, lsm_price
from pricer.market import Dividend, Market
from pricer.market_data import MarketSnapshot


def scripted(answers):
    """Replaces input(): returns the answers in order."""
    iterator = iter(answers)
    return lambda prompt="": next(iterator)


class TestInput(unittest.TestCase):
    def test_parse_number(self):
        self.assertEqual(parse_number("0.05"), 0.05)
        self.assertAlmostEqual(parse_number("5%"), 0.05)
        self.assertEqual(parse_number(" 1,5 "), 1.5)

    def test_parse_times(self):
        self.assertEqual(parse_times("0.25 0.5  0.75"), (0.25, 0.5, 0.75))
        self.assertEqual(parse_times("0,25; 0,5;1"), (0.25, 0.5, 1.0))

    def test_ask_times_retries_until_valid(self):
        shown = []
        times = ask_times("Dates", (0.5, 1.0), 1, ask=scripted(["a b", "0.5 2", "0.75 0.25"]), show=shown.append)
        self.assertEqual(times, (0.25, 0.75))
        self.assertEqual(len(shown), 2)
        self.assertEqual(ask_times("Dates", (0.5, 1.0), 1, ask=scripted([""]), show=lambda _: None), (0.5, 1.0))

    def test_ask_number_retries_until_valid(self):
        shown = []
        value = ask_number("Spot", 100.0, minimum=0, strict=True, ask=scripted(["abc", "-3", "0", "120"]),
                           show=shown.append)
        self.assertEqual(value, 120)
        self.assertEqual(len(shown), 3)

    def test_ask_number_default(self):
        self.assertEqual(ask_number("Strike", 100.0, ask=scripted([""]), show=lambda _: None), 100.0)

    def test_models_per_product(self):
        self.assertEqual(available_models(EUROPEAN), [BSM, BINOMIAL])
        self.assertEqual(available_models(AMERICAN), [BAW, LSM, BINOMIAL])
        self.assertEqual(available_models(BERMUDAN), [LSM, BINOMIAL])
        self.assertEqual(available_models(ASIAN), [MONTE_CARLO])


class TestPricing(unittest.TestCase):
    def test_same_prices_as_pricers(self):
        option, market = Option("put", 100, 1), Market(95, 0.05, 0.25, 0.01)
        dates = (0.25, 0.5, 0.75)
        cases = [(BSM, EUROPEAN, european_price(option, market)),
                 (BINOMIAL, AMERICAN, binomial_price(option, market, 1000)),
                 (BAW, AMERICAN, baw_price(option, market)),
                 (LSM, AMERICAN, lsm_price(option, market, LSMParameters(n_paths=50_000)).price),
                 (LSM, BERMUDAN, bermudan_lsm_price(option, market, dates, LSMParameters(n_paths=50_000)).price),
                 (BINOMIAL, BERMUDAN, bermudan_tree_price(option, market, dates, 1000))]
        for model, style, expected in cases:
            with self.subTest(model=model, style=style):
                report = price_request(PricingRequest(option, style, market, model, exercise_times=dates))
                self.assertEqual(report.price, expected)
                self.assertTrue(-1 <= report.greeks.delta <= 0)

    def test_asian(self):
        option, market = AsianOption.regular("call", 100, 1, 12), Market(100, 0.05, 0.2)
        parameters = AsianParameters(n_paths=20_000)
        report = price_request(PricingRequest(option, ASIAN, market, MONTE_CARLO, asian=parameters))
        self.assertEqual(report.price, asian_price(option, market, parameters).price)
        self.assertTrue(0 < report.greeks.delta < 1)
        labels = [label for label, _ in report.details]
        self.assertIn("Geometric Asian (closed form)", labels)
        self.assertIn("European option (closed form)", labels)

    def test_asian_with_discrete_dividends_has_no_control_variate(self):
        option = AsianOption.regular("put", 100, 1, 4)
        market = Market(100, 0.05, 0.2, dividends=(Dividend(0.5, 2),))
        report = price_request(PricingRequest(option, ASIAN, market, MONTE_CARLO, asian=AsianParameters(n_paths=10_000)))
        self.assertTrue(any("No control variate" in note for note in report.notes))

    def test_bermudan_details_and_notes(self):
        report = price_request(PricingRequest(Option("put", 100, 1), BERMUDAN, Market(100, 0.05, 0.25), BINOMIAL,
                                              tree_steps=400, exercise_times=(0.5,)))
        labels = [label for label, _ in report.details]
        self.assertIn("Early-exercise premium", labels)
        self.assertIn("Exercise dates (years): 0.5, 1.", report.notes)

    def test_american_details_and_notes(self):
        market = Market(100, 0.05, 0.25, dividends=(Dividend(0.5, 2),))
        report = price_request(PricingRequest(Option("put", 100, 1), AMERICAN, market, BAW))
        labels = [label for label, _ in report.details]
        self.assertIn("Early-exercise premium", labels)
        self.assertTrue(any("Bos-Vandermark" in note for note in report.notes))

    def test_rejects_invalid_model(self):
        with self.assertRaises(ValueError):
            price_request(PricingRequest(Option("put", 100, 1), EUROPEAN, Market(100, 0.05, 0.2), BAW))
        with self.assertRaises(ValueError):
            price_request(PricingRequest(Option("put", 100, 1), BERMUDAN, Market(100, 0.05, 0.2), BAW))


class TestSession(unittest.TestCase):
    def test_collect_discrete_dividends(self):
        # put, American, discrete dividends, binomial tree, then the fields
        answers = ["2", "2", "1", "3", "3", "95", "100", "1", "5%", "25%", "2", "0.25", "1.5", "", "", "500"]
        request = collect_request(ask=scripted(answers), show=lambda _: None)
        self.assertEqual(request.option, Option("put", 100, 1))
        self.assertEqual(request.style, AMERICAN)
        self.assertEqual(request.model, BINOMIAL)
        self.assertEqual(request.tree_steps, 500)
        self.assertAlmostEqual(request.market.rate, 0.05)
        self.assertEqual(request.market.dividends, (Dividend(0.25, 1.5), Dividend(0.6667, 1.9)))

    def test_collect_bermudan_custom_dates(self):
        # put, Bermudan, no dividend, LSM, defaults, custom list of dates, 20,000 paths, seed 3
        answers = ["2", "3", "1", "1", "1", "", "", "", "", "", "2", "0.5; 0.25", "20000", "3"]
        request = collect_request(ask=scripted(answers), show=lambda _: None)
        self.assertEqual(request.style, BERMUDAN)
        self.assertEqual(request.model, LSM)
        self.assertEqual(request.exercise_times, (0.25, 0.5))
        self.assertEqual(request.lsm, LSMParameters(n_paths=20_000, seed=3))

    def test_collect_asian_regular_grid(self):
        # call, Asian (one model: no model menu), no dividend, defaults, regular grid of 4 dates, MC settings
        answers = ["1", "4", "1", "1", "", "", "", "", "", "1", "4", "10000", "", "2"]
        request = collect_request(ask=scripted(answers), show=lambda _: None)
        self.assertEqual(request.option, AsianOption("call", 100, 1, (0.25, 0.5, 0.75, 1)))
        self.assertEqual(request.model, MONTE_CARLO)
        self.assertEqual(request.asian, AsianParameters(n_paths=10_000, control_variate=False))

    def test_collect_with_yahoo_data(self):
        # call, European, Yahoo data: a bad ticker first, then SPY; BSM; every field at its Yahoo default
        def fetch(ticker):
            if ticker != "SPY":
                raise ValueError("unknown ticker")
            return MarketSnapshot("SPY", "SPDR S&P 500", "USD", date(2026, 10, 12), 778.1, 0.0413,
                                  "US 13-week T-bill (^IRX)", 0.0072, 0.13, 252)
        answers = ["1", "1", "2", "xyz", "spy", "1", "", "", "0.5", "", "", ""]
        shown = []
        request = collect_request(ask=scripted(answers), show=shown.append, fetch_snapshot=fetch)
        self.assertEqual(request.option, Option("call", 778, 0.5))
        self.assertEqual(request.market, Market(778.1, 0.0413, 0.13, 0.0072))
        self.assertTrue(any("Could not load XYZ" in line for line in shown))

    def test_full_session(self):
        # European call, continuous yield, closed form, defaults everywhere, then quit
        answers = ["1", "1", "1", "2", "1", "", "", "", "", "", "", "2"]
        shown = []
        main(ask=scripted(answers), show=shown.append)
        output = "\n".join(map(str, shown))
        expected = european_price(Option("call", 100, 1), Market(100, 0.05, 0.2, 0.02))
        self.assertIn(f"{expected:.4f}", output)
        self.assertIn("Delta", output)
        self.assertIn("Goodbye.", output)

    def test_full_asian_session(self):
        answers = ["1", "4", "1", "1", "", "", "", "", "", "", "", "10000", "", "", "2"]
        shown = []
        main(ask=scripted(answers), show=shown.append)
        output = "\n".join(map(str, shown))
        expected = asian_price(AsianOption.regular("call", 100, 1, 12), Market(100, 0.05, 0.2),
                               AsianParameters(n_paths=10_000)).price
        self.assertIn("Asian call (12 observations)", output)
        self.assertIn(f"{expected:.4f}", output)


if __name__ == "__main__":
    unittest.main()
