"""
test_market_data.py - Tests of the Yahoo Finance connection and of the implied volatility tools, offline.

A fake Yahoo client (fake_yahoo.py) replaces the network. Checks the rate,
yield and volatility conventions of the snapshot, the option-quote parsing,
the implied volatility inversion (European and American, round trips and
no-arbitrage errors), and that the smile and the surface recover the smile
used to generate the fake quotes.
"""
import math
import unittest
from dataclasses import replace
from datetime import date

from pricer.baw import baw_price
from pricer.contract import Option
from pricer.european import european_price
from pricer.implied_vol import (atm_volatility, implied_volatility, parity_forward, pick_expiries, term_structure,
                                volatility_smile, volatility_surface)
from pricer.market import Market
from pricer.market_data import (fetch_snapshot, historical_volatility, option_expiries, option_quotes,
                                risk_free_rate, tbill_rate)
from pricer.test.fake_yahoo import AS_OF, DIVIDEND_YIELD, SPOT, TBILL_PERCENT, TRUE_VOL, FakeYahoo, smile_volatility


class TestSnapshot(unittest.TestCase):
    def test_tbill_rate(self):
        # discount yield 4 %: price 1 - 0.04 * 91/360, continuous rate a bit above 4 %
        self.assertAlmostEqual(tbill_rate(4.0), -math.log(1 - 0.04*91/360)*365/91, places=14)
        self.assertGreater(tbill_rate(4.0), 0.04)

    def test_historical_volatility(self):
        closes = [100*math.exp(0.01*(-1)**k) for k in range(253)]   # returns of ±0.02 after the first one
        self.assertAlmostEqual(historical_volatility(closes, 251), 0.02*math.sqrt(252), delta=1e-3)
        with self.assertRaises(ValueError):
            historical_volatility([100, 101])

    def test_fetch_snapshot(self):
        snapshot = fetch_snapshot("FAKE", FakeYahoo())
        self.assertEqual(snapshot.as_of, AS_OF)
        self.assertEqual(snapshot.name, "Fake Corp")
        self.assertAlmostEqual(snapshot.spot, SPOT)
        self.assertAlmostEqual(snapshot.rate, tbill_rate(TBILL_PERCENT))
        self.assertAlmostEqual(snapshot.dividend_yield, DIVIDEND_YIELD)
        self.assertAlmostEqual(snapshot.historical_volatility, TRUE_VOL, delta=0.03)
        self.assertEqual(snapshot.market(0.3), Market(snapshot.spot, snapshot.rate, 0.3, snapshot.dividend_yield))

    def test_rates_by_currency(self):
        client = FakeYahoo()
        self.assertEqual(risk_free_rate("EUR", client, estr=lambda: 0.024)[0], math.log(1.024))
        rate, source = risk_free_rate("JPY", client)
        self.assertAlmostEqual(rate, tbill_rate(TBILL_PERCENT))
        self.assertIn("no free JPY rate", source)
        self.assertEqual(fetch_snapshot("FAKE", client, rate=0.01).rate_source, "user input")

    def test_unknown_ticker(self):
        with self.assertRaises(ValueError):
            fetch_snapshot("NOPE", FakeYahoo())

    def test_option_quotes(self):
        client = FakeYahoo()
        expiries = option_expiries("FAKE", client)
        quotes = option_quotes("FAKE", expiries[1], client)
        self.assertEqual(len(quotes), 26)
        self.assertEqual(quotes[0].option_type, "call")
        self.assertEqual(quotes[0].open_interest, 0.0)   # NaN in the table
        self.assertAlmostEqual(quotes[0].mid, quotes[0].last)
        self.assertAlmostEqual(quotes[0].relative_spread, 0.02)


class TestImpliedVolatility(unittest.TestCase):
    MARKET = Market(100, 0.04, 0.2, 0.01)

    def test_round_trip_european(self):
        for option_type in ("call", "put"):
            for strike in (70, 100, 140):
                for sigma in (0.05, 0.3, 1.2):
                    option = Option(option_type, strike, 0.75)
                    price = european_price(option, Market(100, 0.04, sigma, 0.01))
                    implied = implied_volatility(price, option, self.MARKET)
                    vega = (european_price(option, Market(100, 0.04, sigma + 1e-4, 0.01)) - price)/1e-4
                    with self.subTest(option_type=option_type, strike=strike, sigma=sigma):
                        # the price is always matched; sigma only where the price depends on it (vega > 0)
                        self.assertAlmostEqual(european_price(option, Market(100, 0.04, implied, 0.01)), price,
                                               places=7)
                        if vega > 1e-2:
                            self.assertAlmostEqual(implied, sigma, places=6)

    def test_round_trip_american(self):
        option = Option("put", 110, 1)
        price = baw_price(option, Market(100, 0.04, 0.35, 0.01))
        self.assertAlmostEqual(implied_volatility(price, option, self.MARKET, baw_price), 0.35, places=6)

    def test_no_arbitrage_errors(self):
        option = Option("call", 90, 1)
        with self.assertRaises(ValueError):
            implied_volatility(5.0, option, self.MARKET)    # below the discounted intrinsic value
        with self.assertRaises(ValueError):
            implied_volatility(150.0, option, self.MARKET)  # above the spot
        with self.assertRaises(ValueError):
            implied_volatility(10.0, Option("call", 90, 0), self.MARKET)


class TestSmile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = FakeYahoo()
        cls.snapshot = fetch_snapshot("FAKE", cls.client)

    def test_smile_recovers_the_generating_smile(self):
        expiry = option_expiries("FAKE", self.client)[1]
        points = volatility_smile(option_quotes("FAKE", expiry, self.client), self.snapshot)
        self.assertEqual(len(points), 13)   # one out-of-the-money option per strike
        for p in points:
            forward = SPOT*math.exp((self.snapshot.rate - DIVIDEND_YIELD)*p.maturity)
            with self.subTest(strike=p.strike):
                self.assertAlmostEqual(p.implied_volatility, smile_volatility(p.strike, forward), places=6)
                self.assertEqual(p.option_type, "put" if p.strike < forward else "call")
        self.assertAlmostEqual(atm_volatility(points), 0.2, delta=2e-3)

    def test_forward_from_parity(self):
        # the fake chain is priced with BSM at r = tbill_rate(4 %) and q = 1 %: parity recovers F and D exactly
        expiry = option_expiries("FAKE", self.client)[2]
        quotes = option_quotes("FAKE", expiry, self.client)
        maturity = quotes[0].maturity(AS_OF)
        forward, discount = parity_forward(quotes, SPOT)
        rate = tbill_rate(TBILL_PERCENT)
        self.assertAlmostEqual(discount, math.exp(-rate*maturity), places=10)
        self.assertAlmostEqual(forward, SPOT*math.exp((rate - DIVIDEND_YIELD)*maturity), places=8)
        # a snapshot with a wrong carry still gives the right smile once the forward comes from parity
        wrong = replace(self.snapshot, dividend_yield=0.05)
        points = volatility_smile(quotes, wrong, forward_from_parity=True)
        for p in points:
            with self.subTest(strike=p.strike):
                self.assertAlmostEqual(p.implied_volatility, smile_volatility(p.strike, forward), places=6)
        with self.assertRaises(ValueError):
            parity_forward(quotes[:2], SPOT)

    def test_filters(self):
        expiry = option_expiries("FAKE", self.client)[1]
        quotes = option_quotes("FAKE", expiry, self.client)
        self.assertEqual(len(volatility_smile(quotes, self.snapshot, otm_only=False)), 26)
        self.assertEqual(volatility_smile(quotes, self.snapshot, max_spread=0.01), [])

    def test_pick_expiries(self):
        expiries = [AS_OF.replace(day=13), date(2026, 11, 13), date(2026, 12, 18), date(2027, 6, 18)]
        self.assertEqual(pick_expiries(expiries, AS_OF, (30, 60, 400)),
                         [date(2026, 11, 13), date(2026, 12, 18), date(2027, 6, 18)])
        self.assertEqual(pick_expiries(expiries[:1], AS_OF), [])

    def test_surface_and_term_structure(self):
        points = volatility_surface("FAKE", self.client, self.snapshot, target_days=(30, 90))
        rows = term_structure(points)
        self.assertEqual([row[0] for row in rows], list(self.client.expiry_list[1:]))   # 3-day expiry skipped
        for _, _, atm in rows:
            self.assertAlmostEqual(atm, 0.2, delta=3e-3)


if __name__ == "__main__":
    unittest.main()
