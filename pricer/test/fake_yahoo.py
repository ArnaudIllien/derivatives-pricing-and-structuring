"""
fake_yahoo.py - Offline stand-in for market_data.YahooClient, used by the tests.

It serves a synthetic price history (GBM with a known volatility), a constant
T-bill quote, an info dict, and option chains priced with BSM on a known smile
sigma(K) = 0.2 - 0.1 ln(K / F), so the tests can check that the implied
volatilities are recovered exactly.
"""
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from pricer.european import bsm_price

AS_OF = date(2026, 10, 12)
SPOT, TBILL_PERCENT, DIVIDEND_YIELD, TRUE_VOL = 100.0, 4.0, 0.01, 0.25


def smile_volatility(strike: float, forward: float) -> float:
    return 0.2 - 0.1*math.log(strike/forward)


class FakeYahoo:
    def __init__(self, n_days: int = 600, seed: int = 1):
        rng = np.random.default_rng(seed)
        returns = TRUE_VOL/math.sqrt(252)*rng.standard_normal(n_days - 1)
        prices = SPOT*np.exp(np.concatenate([[0.0], np.cumsum(returns)]) - np.sum(returns))   # ends at SPOT
        self.dates = [AS_OF - timedelta(days=n_days - 1 - i) for i in range(n_days)]
        self.prices = pd.Series(prices, index=self.dates)
        self.expiry_list = (AS_OF + timedelta(days=3), AS_OF + timedelta(days=31), AS_OF + timedelta(days=94))
        self.calls = 0

    def history(self, ticker, period="2y", start=None, end=None):
        self.calls += 1
        if ticker == "^IRX":
            return pd.Series(TBILL_PERCENT, index=self.dates)
        if ticker != "FAKE":
            return pd.Series(dtype=float)
        if start is None:
            return self.prices
        return self.prices[[start <= d <= end for d in self.dates]]

    def info(self, ticker):
        return {"currency": "USD", "shortName": " Fake Corp ", "trailingAnnualDividendYield": math.exp(DIVIDEND_YIELD) - 1}

    def expiries(self, ticker):
        return tuple(e.isoformat() for e in self.expiry_list)

    def option_chain(self, ticker, expiry):
        from pricer.market_data import tbill_rate
        maturity = (date.fromisoformat(expiry) - AS_OF).days/365
        rate = tbill_rate(TBILL_PERCENT)
        forward = SPOT*math.exp((rate - DIVIDEND_YIELD)*maturity)
        tables = []
        for is_call in (True, False):
            rows = []
            for strike in range(70, 135, 5):
                price = bsm_price(is_call, SPOT, strike, maturity, rate, DIVIDEND_YIELD, smile_volatility(strike, forward))
                rows.append({"strike": float(strike), "bid": price*0.99, "ask": price*1.01, "lastPrice": price,
                             "volume": 10, "openInterest": float("nan"), "impliedVolatility": 0.2})
            tables.append(pd.DataFrame(rows))
        return tables[0], tables[1]
