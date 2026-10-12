"""
european.py - Black-Scholes-Merton pricer for European options.

Extracted from the old notebook z_old/Pricing_option_method_BlackScholes.ipynb
and extended to a continuous dividend yield q. With q = 0 the prices are the
same as in the old notebook; `call_option` and `put_option` keep its
date-based interface.

Discrete dividends use the escrowed-dividend approximation of Bos and
Vandermark (2002). Each dividend D paid at t is split in two:
- a "near" part, (T - t) / T of its present value, removed from the spot;
- a "far" part, t / T of its value carried forward to T, added to the strike.
Plain escrowing removes the whole PV from the spot and underprices calls by up
to several %. Against the exact drop model (binomial tree), the gap measured
with this split stays below 0.3 %. For the exact model, use the binomial tree.
"""
import math
from dataclasses import replace
from datetime import date

from scipy.stats import norm

from .contract import Option, year_fraction
from .market import Market


def d1(spot, strike, maturity, rate, dividend_yield, vol):
    return (math.log(spot/strike) + (rate - dividend_yield + vol**2/2)*maturity)/(vol*math.sqrt(maturity))


def bsm_price(is_call: bool, spot, strike, maturity, rate, dividend_yield, vol) -> float:
    """Black-Scholes-Merton price of a European call or put."""
    if maturity == 0:
        return max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)

    discounted_spot = spot*math.exp(-dividend_yield*maturity)
    discounted_strike = strike*math.exp(-rate*maturity)
    if vol == 0:
        # deterministic underlying: S_T = S e^{(r-q)T}, known discounted payoff
        gap = discounted_spot - discounted_strike
        return max(gap, 0.0) if is_call else max(-gap, 0.0)

    D1 = d1(spot, strike, maturity, rate, dividend_yield, vol)
    D2 = D1 - vol*math.sqrt(maturity)
    if is_call:
        return float(discounted_spot*norm.cdf(D1) - discounted_strike*norm.cdf(D2))
    return float(discounted_strike*norm.cdf(-D2) - discounted_spot*norm.cdf(-D1))


def escrowed_dividends(option: Option, market: Market) -> tuple[Option, Market]:
    """Bos-Vandermark adjustment: contract and market without discrete dividends."""
    T, r = option.maturity, market.rate
    dividends = market.dividends_before(T)
    if not dividends:
        return option, market
    spot = market.spot - sum((T - d.time)/T*d.amount*math.exp(-r*d.time) for d in dividends)
    strike = option.strike + sum(d.time/T*d.amount*math.exp(r*(T - d.time)) for d in dividends)
    if spot <= 0:
        raise ValueError("The dividends are too large compared with the spot.")
    return replace(option, strike=strike), replace(market, spot=spot, dividends=())


def european_price(option: Option, market: Market) -> float:
    o, m = escrowed_dividends(option, market)
    return bsm_price(o.is_call, m.spot, o.strike, o.maturity, m.rate, m.dividend_yield, m.volatility)


#- Interface of the old notebook -

def call_option(valuation_date: date, expiry_date: date, spot, strike, rate, vol, dividend_yield=0.0):
    maturity = year_fraction(valuation_date, expiry_date)
    return bsm_price(True, spot, strike, maturity, rate, dividend_yield, vol)


def put_option(valuation_date: date, expiry_date: date, spot, strike, rate, vol, dividend_yield=0.0):
    maturity = year_fraction(valuation_date, expiry_date)
    return bsm_price(False, spot, strike, maturity, rate, dividend_yield, vol)
