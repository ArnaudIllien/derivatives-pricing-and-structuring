"""
test_binomial.py - Tests of the binomial tree and of the simulation model.

Checks that the tree converges to Black-Scholes, that the American and
Bermudan prices are ordered and converge, and that discrete dividends are
consistent between the tree and a plain Monte Carlo on the simulated paths.
"""
import unittest

import numpy as np

from pricer.binomial import bermudan_binomial_price, binomial_price
from pricer.contract import Option
from pricer.european import european_price
from pricer.market import Dividend, Market
from pricer.model import BlackScholesModel, nearest_columns

DIVIDENDS = (Dividend(0.25, 3), Dividend(0.75, 3))


class TestBinomialTree(unittest.TestCase):
    def test_converges_to_black_scholes(self):
        option, market = Option("put", 40, 1), Market(36, 0.06, 0.2, 0.02)
        exact = european_price(option, market)
        errors = [abs(binomial_price(option, market, n, american=False) - exact) for n in (100, 400, 1600)]
        self.assertLess(errors[-1], 1e-3)
        self.assertLess(errors[2], errors[0])

    def test_american_converges(self):
        option, market = Option("put", 40, 1), Market(36, 0.06, 0.2)
        coarse, fine, finest = (binomial_price(option, market, n) for n in (500, 2000, 8000))
        self.assertLess(abs(finest - fine), abs(fine - coarse) + 1e-4)
        self.assertLess(abs(finest - fine), 1e-3)
        self.assertGreater(fine, european_price(option, market))

    def test_bermudan_between_european_and_american(self):
        option, market = Option("put", 40, 1), Market(36, 0.06, 0.2)
        bermudans = [bermudan_binomial_price(option, market, n) for n in (5, 25, 100)]
        self.assertEqual(bermudans, sorted(bermudans))
        self.assertGreater(bermudans[0], european_price(option, market))
        self.assertLess(bermudans[-1], binomial_price(option, market))

    def test_dividend_tree_matches_monte_carlo(self):
        # same drop model on both sides: the tree must sit within the Monte Carlo noise
        market = Market(100, 0.05, 0.25, dividends=DIVIDENDS)
        model = BlackScholesModel(market)
        times = model.simulation_times(1, 4)
        paths = model.simulate(times, 400_000, np.random.default_rng(7), antithetic=True)
        for option_type in ("call", "put"):
            option = Option(option_type, 100, 1)
            flows = option.payoff(paths[:, -1])*np.exp(-0.05)
            flows = (flows[:200_000] + flows[200_000:])/2
            mean, error = flows.mean(), flows.std(ddof=1)/np.sqrt(len(flows))
            with self.subTest(option_type=option_type):
                tree = binomial_price(option, market, 2000, american=False)
                self.assertLess(abs(tree - mean), 4*error + 0.01)

    def test_dividends_lower_calls_and_raise_puts(self):
        plain = Market(100, 0.05, 0.25)
        with_dividends = Market(100, 0.05, 0.25, dividends=DIVIDENDS)
        call, put = Option("call", 100, 1), Option("put", 100, 1)
        self.assertLess(binomial_price(call, with_dividends), binomial_price(call, plain))
        self.assertGreater(binomial_price(put, with_dividends), binomial_price(put, plain))

    def test_american_call_exercised_before_dividend(self):
        # without dividends an American call is worth its European price; a large dividend changes that
        option = Option("call", 100, 1)
        plain = Market(110, 0.05, 0.25)
        self.assertAlmostEqual(binomial_price(option, plain), binomial_price(option, plain, american=False), places=3)
        market = Market(110, 0.05, 0.25, dividends=(Dividend(0.5, 8),))
        self.assertGreater(binomial_price(option, market), binomial_price(option, market, american=False) + 0.1)


class TestModel(unittest.TestCase):
    def test_grid_contains_dividend_dates(self):
        model = BlackScholesModel(Market(100, 0.05, 0.2, dividends=(Dividend(0.3, 1), Dividend(0.5, 1))))
        times = model.simulation_times(1, 4)
        np.testing.assert_allclose(times, [0, 0.25, 0.3, 0.5, 0.75, 1])
        np.testing.assert_allclose(model.dividend_payouts(times), [0, 0, 1, 1, 0, 0])

    def test_time_grid_with_given_dates(self):
        model = BlackScholesModel(Market(100, 0.05, 0.2, dividends=(Dividend(0.5, 1),)))
        times = model.time_grid(1, [0.75, 0.25, 0.5, 0.25])
        np.testing.assert_allclose(times, [0, 0.25, 0.5, 0.75, 1])
        np.testing.assert_array_equal(nearest_columns(times, [0.25, 1, 0.25]), [1, 4, 1])
        with self.assertRaises(ValueError):
            model.time_grid(1, [1.5])

    def test_forward_with_dividends(self):
        # E[S_T] = (S e^{bt} - D) e^{b(T - t)} for one dividend D paid at t (no floor reached)
        market = Market(100, 0.05, 0.2, 0.01, dividends=(Dividend(0.5, 4),))
        model = BlackScholesModel(market)
        times = model.simulation_times(1, 10)
        final = model.simulate(times, 200_000, np.random.default_rng(3))[:, -1]
        expected = (100*np.exp(0.04*0.5) - 4)*np.exp(0.04*0.5)
        self.assertLess(abs(final.mean() - expected), 4*final.std()/np.sqrt(len(final)))

    def test_antithetic_pairs(self):
        model = BlackScholesModel(Market(100, 0.0, 0.2))
        paths = model.simulate(np.linspace(0, 1, 3), 4, np.random.default_rng(0), antithetic=True)
        # with r = q = 0, log returns of the pairs are symmetric around -sigma² t / 2
        drift = -0.02*np.array([0.5, 1.0])
        np.testing.assert_allclose(np.log(paths[:2, 1:]/100) - drift, -(np.log(paths[2:, 1:]/100) - drift))


if __name__ == "__main__":
    unittest.main()
