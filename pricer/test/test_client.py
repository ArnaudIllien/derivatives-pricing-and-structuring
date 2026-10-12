"""
test_client.py - Tests of the interactive client, driven by scripted answers.

Checks the number parsing, the input validation loops, the model menu per
exercise style, that `price_request` returns the same prices as the pricers
called directly, and a full session from the questions to the printed report.
"""
import unittest

from pricer.baw import baw_price
from pricer.binomial import binomial_price
from pricer.client import (BAW, BINOMIAL, BSM, LSM, PricingRequest, ask_number, available_models, collect_request,
                           main, parse_number, price_request)
from pricer.contract import Option
from pricer.european import european_price
from pricer.lsm import LSMParameters, lsm_price
from pricer.market import Dividend, Market


def scripted(answers):
    """Replaces input(): returns the answers in order."""
    iterator = iter(answers)
    return lambda prompt="": next(iterator)


class TestInput(unittest.TestCase):
    def test_parse_number(self):
        self.assertEqual(parse_number("0.05"), 0.05)
        self.assertAlmostEqual(parse_number("5%"), 0.05)
        self.assertEqual(parse_number(" 1,5 "), 1.5)

    def test_ask_number_retries_until_valid(self):
        shown = []
        value = ask_number("Spot", 100.0, minimum=0, strict=True, ask=scripted(["abc", "-3", "0", "120"]),
                           show=shown.append)
        self.assertEqual(value, 120)
        self.assertEqual(len(shown), 3)

    def test_ask_number_default(self):
        self.assertEqual(ask_number("Strike", 100.0, ask=scripted([""]), show=lambda _: None), 100.0)

    def test_models_per_style(self):
        self.assertEqual(available_models(american=False), [BSM, BINOMIAL])
        self.assertEqual(available_models(american=True), [BAW, LSM, BINOMIAL])


class TestPricing(unittest.TestCase):
    def test_same_prices_as_pricers(self):
        option, market = Option("put", 100, 1), Market(95, 0.05, 0.25, 0.01)
        cases = [(BSM, False, european_price(option, market)),
                 (BINOMIAL, True, binomial_price(option, market, 1000)),
                 (BAW, True, baw_price(option, market)),
                 (LSM, True, lsm_price(option, market, LSMParameters(n_paths=50_000)).price)]
        for model, american, expected in cases:
            with self.subTest(model=model):
                report = price_request(PricingRequest(option, american, market, model))
                self.assertEqual(report.price, expected)
                self.assertTrue(-1 <= report.greeks.delta <= 0)

    def test_american_details_and_notes(self):
        market = Market(100, 0.05, 0.25, dividends=(Dividend(0.5, 2),))
        report = price_request(PricingRequest(Option("put", 100, 1), True, market, BAW))
        labels = [label for label, _ in report.details]
        self.assertIn("Early-exercise premium", labels)
        self.assertTrue(any("Bos-Vandermark" in note for note in report.notes))

    def test_rejects_invalid_model(self):
        with self.assertRaises(ValueError):
            price_request(PricingRequest(Option("put", 100, 1), False, Market(100, 0.05, 0.2), BAW))


class TestSession(unittest.TestCase):
    def test_collect_discrete_dividends(self):
        # put, American, discrete dividends, binomial tree, then the fields
        answers = ["2", "2", "3", "3", "95", "100", "1", "5%", "25%", "2", "0.25", "1.5", "", "", "500"]
        request = collect_request(ask=scripted(answers), show=lambda _: None)
        self.assertEqual(request.option, Option("put", 100, 1))
        self.assertTrue(request.american)
        self.assertEqual(request.model, BINOMIAL)
        self.assertEqual(request.tree_steps, 500)
        self.assertAlmostEqual(request.market.rate, 0.05)
        self.assertEqual(request.market.dividends, (Dividend(0.25, 1.5), Dividend(0.6667, 1.9)))

    def test_full_session(self):
        # European call, continuous yield, closed form, defaults everywhere, then quit
        answers = ["1", "1", "2", "1", "", "", "", "", "", "", "2"]
        shown = []
        main(ask=scripted(answers), show=shown.append)
        output = "\n".join(map(str, shown))
        expected = european_price(Option("call", 100, 1), Market(100, 0.05, 0.2, 0.02))
        self.assertIn(f"{expected:.4f}", output)
        self.assertIn("Delta", output)
        self.assertIn("Goodbye.", output)


if __name__ == "__main__":
    unittest.main()
