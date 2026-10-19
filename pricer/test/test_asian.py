"""
test_asian.py - Tests of the arithmetic Asian Monte Carlo pricer and of the AsianOption contract.

Checks the geometric closed form against a plain simulation, the limit cases
with exact answers (one observation = European option, zero volatility, zero
maturity), the price bounds (geometric <= arithmetic <= European for a call),
put-call parity on the average, the control variate (unbiased and much less
noisy), the 1/sqrt(N) decay of the standard error, convergence in N, discrete
dividends and the input checks.
"""
import math
import unittest

import numpy as np

from pricer.asian import AsianParameters, asian_convergence, asian_price, geometric_asian_price
from pricer.contract import AsianOption, Option
from pricer.european import european_price
from pricer.greeks import compute_greeks
from pricer.market import Dividend, Market
from pricer.model import BlackScholesModel

MARKET = Market(100, 0.05, 0.2, 0.01)
CALL = AsianOption.regular("call", 100, 1, 12)
PUT = AsianOption.regular("put", 100, 1, 12)
FAST = AsianParameters(n_paths=40_000)


class TestContract(unittest.TestCase):
    def test_regular_dates(self):
        option = AsianOption.regular("CALL", 100, 2, 4)
        self.assertEqual(option.option_type, "call")
        self.assertEqual(option.observation_times, (0.5, 1.0, 1.5, 2.0))
        self.assertEqual(option.vanilla, Option("call", 100, 2))

    def test_validations(self):
        for times in ((), (-0.1, 0.5), (0.5, 1.5)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                AsianOption("call", 100, 1, times)
        with self.assertRaises(ValueError):
            AsianOption("straddle", 100, 1, (1,))
        with self.assertRaises(ValueError):
            AsianOption.regular("call", 100, 1, 0)

    def test_shifted_fixes_passed_dates_at_zero(self):
        option = AsianOption("call", 100, 1, (0.1, 0.5, 1)).shifted(0.2)
        self.assertEqual(option.maturity, 0.8)
        np.testing.assert_allclose(option.observation_times, (0, 0.3, 0.8))


class TestGeometricClosedForm(unittest.TestCase):
    def test_against_simulation(self):
        # independent check: lognormal G simulated directly on the observation dates
        model = BlackScholesModel(MARKET)
        times = model.time_grid(1, CALL.observation_times)
        paths = model.simulate(times, 400_000, np.random.default_rng(11))[:, 1:]
        geometric = np.exp(np.log(paths).mean(axis=1))
        for option in (CALL, PUT):
            flows = math.exp(-MARKET.rate)*option.payoff(geometric)
            mean, error = flows.mean(), flows.std(ddof=1)/math.sqrt(len(flows))
            with self.subTest(option_type=option.option_type):
                self.assertLess(abs(geometric_asian_price(option, MARKET) - mean), 4*error)

    def test_one_observation_is_black_scholes(self):
        option = AsianOption("put", 95, 1, (1,))
        self.assertAlmostEqual(geometric_asian_price(option, MARKET), european_price(option.vanilla, MARKET), places=10)

    def test_rejects_discrete_dividends(self):
        with self.assertRaises(ValueError):
            geometric_asian_price(CALL, Market(100, 0.05, 0.2, dividends=(Dividend(0.5, 2),)))


class TestAsianMonteCarlo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.call = asian_price(CALL, MARKET, FAST)
        cls.put = asian_price(PUT, MARKET, FAST)

    def test_reproducible_with_seed(self):
        self.assertEqual(asian_price(CALL, MARKET, FAST), self.call)
        other = asian_price(CALL, MARKET, AsianParameters(n_paths=40_000, seed=1))
        self.assertNotEqual(other.price, self.call.price)

    def test_one_observation_is_european(self):
        # A = S_T: with the control variate, arithmetic = geometric on every path, the price is exact
        option = AsianOption("call", 100, 1, (1,))
        exact = european_price(option.vanilla, MARKET)
        result = asian_price(option, MARKET, FAST)
        self.assertAlmostEqual(result.price, exact, places=8)
        self.assertLess(abs(result.plain_price - exact), 3*result.plain_std_error)

    def test_bounds(self):
        # AM-GM: G <= A path by path, so geometric call <= arithmetic call; averaging lowers the volatility
        self.assertGreater(self.call.price, geometric_asian_price(CALL, MARKET))
        self.assertLess(self.call.price, european_price(CALL.vanilla, MARKET))
        self.assertLess(self.put.price, geometric_asian_price(PUT, MARKET))

    def test_put_call_parity(self):
        # C - P = e^{-rT} (E[A] - K), with E[S_t] = S e^{(r - q) t}
        expected_average = np.mean([MARKET.spot*math.exp(MARKET.carry*t) for t in CALL.observation_times])
        parity = math.exp(-MARKET.rate)*(expected_average - CALL.strike)
        self.assertLess(abs(self.call.price - self.put.price - parity), 3*(self.call.std_error + self.put.std_error))

    def test_control_variate_unbiased_and_efficient(self):
        call = self.call
        self.assertTrue(call.control_variate_used)
        self.assertLess(abs(call.price - call.plain_price), 3*call.plain_std_error)
        self.assertGreater(call.variance_reduction, 50)
        plain = asian_price(CALL, MARKET, AsianParameters(n_paths=40_000, control_variate=False))
        self.assertFalse(plain.control_variate_used)
        self.assertEqual(plain.price, call.plain_price)

    def test_antithetic_reduces_variance(self):
        plain = asian_price(CALL, MARKET, AsianParameters(n_paths=40_000, antithetic=False, control_variate=False))
        self.assertLess(self.call.plain_std_error, plain.std_error)

    def test_std_error_as_root_n(self):
        small = asian_price(CALL, MARKET, AsianParameters(n_paths=10_000, control_variate=False)).std_error
        large = asian_price(CALL, MARKET, AsianParameters(n_paths=40_000, control_variate=False)).std_error
        self.assertAlmostEqual(small/large, 2, delta=0.2)

    def test_convergence(self):
        # every estimate sits within the noise of the most precise one, and the error falls as 1/sqrt(N)
        results = asian_convergence(CALL, MARKET, (5_000, 20_000, 80_000), AsianParameters(control_variate=False))
        reference = asian_price(CALL, MARKET, AsianParameters(n_paths=200_000, seed=5))
        for n, result in results:
            with self.subTest(n=n):
                self.assertLess(abs(result.price - reference.price), 3*result.std_error + 3*reference.std_error)
        errors = [result.std_error for _, result in results]
        self.assertAlmostEqual(errors[0]/errors[2], 4, delta=0.5)

    def test_batches_do_not_change_the_estimator(self):
        batched = asian_price(CALL, MARKET, AsianParameters(n_paths=40_000, batch_size=10_000))
        self.assertLess(abs(batched.price - self.call.price), 3*(batched.std_error + self.call.std_error))

    def test_zero_volatility(self):
        market = Market(100, 0.05, 0.0)
        exact = math.exp(-0.05)*max(np.mean([100*math.exp(0.05*t) for t in CALL.observation_times]) - 100, 0)
        result = asian_price(CALL, market, AsianParameters(n_paths=1_000))
        self.assertAlmostEqual(result.price, exact, places=10)
        self.assertEqual(result.std_error, 0)

    def test_zero_maturity_and_past_fixings(self):
        self.assertEqual(asian_price(AsianOption("put", 100, 0, (0,)), MARKET).price, 0)
        self.assertEqual(asian_price(AsianOption("call", 90, 0, (0,)), MARKET).price, 10)

    def test_discrete_dividends(self):
        market = Market(100, 0.05, 0.2, dividends=(Dividend(0.5, 3),))
        call, put = asian_price(CALL, market, FAST), asian_price(PUT, market, FAST)
        self.assertFalse(call.control_variate_used)
        no_dividend = Market(100, 0.05, 0.2)
        self.assertLess(call.price, asian_price(CALL, no_dividend, FAST).price)
        self.assertGreater(put.price, asian_price(PUT, no_dividend, FAST).price)
        # parity with E[S_t] = S e^{rt} - D e^{r(t - 0.5)} after the dividend
        forwards = [100*math.exp(0.05*t) - (3*math.exp(0.05*(t - 0.5)) if t > 0.5 else 0)
                    for t in CALL.observation_times]
        parity = math.exp(-0.05)*(np.mean(forwards) - 100)
        self.assertLess(abs(call.price - put.price - parity), 3*(call.std_error + put.std_error))

    def test_greeks(self):
        greeks = compute_greeks(lambda o, m: asian_price(o, m, FAST).price, CALL, MARKET,
                                spot_bump=1e-2, vol_bump=1e-2, time_bump=7/365)
        self.assertTrue(0 < greeks.delta < 1)
        self.assertGreater(greeks.vega, 0)
        self.assertLess(greeks.theta, 0)

    def test_parameter_validations(self):
        with self.assertRaises(ValueError):
            asian_price(CALL, MARKET, AsianParameters(n_paths=1_001))
        with self.assertRaises(ValueError):
            asian_price(CALL, MARKET, AsianParameters(n_paths=2))


if __name__ == "__main__":
    unittest.main()
