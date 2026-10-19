"""
implied_vol.py - Implied volatility, volatility smile and surface from market option prices.

USAGE
    from pricer.implied_vol import implied_volatility, volatility_smile, volatility_surface
    sigma = implied_volatility(price, option, market)                    # European (BSM)
    sigma = implied_volatility(price, option, market, price_fn=baw_price)  # American (BAW)
    points = volatility_surface("SPY")                                   # live, from Yahoo

Method
------
The implied volatility is the sigma that makes the model price equal to the
market price. Every pricer of the package is increasing in sigma, so Brent's
method on [low, high] finds it once the price is bracketed. A price below the
model value at sigma = low (the no-arbitrage floor, about the intrinsic value)
or above the value at sigma = high has no implied volatility: ValueError.

Smile and surface conventions
-----------------------------
- Market price = mid of bid and ask (last trade if one side is missing).
- Only out-of-the-money options are kept by default (puts below the forward,
  calls above): they are the liquid ones, and in-the-money prices add the
  intrinsic value, which drowns the time value.
- Quotes without a bid, or with a bid-ask spread wider than `max_spread` of
  the mid, are dropped.
- Moneyness is ln(K / F), with the forward F = S e^{(r - q) T}.
- `american=True` inverts the Barone-Adesi-Whaley price (single stocks and
  ETFs such as SPY have American options); otherwise BSM is used (index
  options such as SPX are European).
- The Yahoo implied volatility is kept next to ours for comparison. Yahoo does
  not document its method, so small gaps are expected.
- Forward: by default F = S e^{(r - q) T} with the snapshot's rate and dividend
  yield. With `forward_from_parity=True`, each expiry uses instead the forward and
  discount factor implied by put-call parity on its own quotes,
      C(K) - P(K) = D (F - K)   (European options),
  fitted by least squares on the strikes near the money: slope -D, intercept D F.
  This removes the jump between put and call volatilities at the money that a
  wrong rate or dividend creates on long expiries.
"""
import math
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
from scipy.optimize import brentq

from .baw import baw_price
from .contract import CALL, Option
from .european import european_price
from .market import Market
from .market_data import MarketSnapshot, OptionQuote, YahooClient, fetch_snapshot, option_expiries, option_quotes


def implied_volatility(price: float, option: Option, market: Market, price_fn=european_price,
                       low: float = 1e-4, high: float = 5.0, tolerance: float = 1e-8) -> float:
    """Sigma such that price_fn(option, market with sigma) = price."""
    if option.maturity <= 0:
        raise ValueError("No implied volatility at zero maturity.")

    def gap(sigma):
        return price_fn(option, replace(market, volatility=sigma)) - price

    floor, cap = gap(low), gap(high)
    if floor > 0:
        raise ValueError(f"Price {price:.4f} is below the model value at sigma={low} (no-arbitrage floor).")
    if cap < 0:
        raise ValueError(f"Price {price:.4f} is above the model value at sigma={high}.")
    return float(brentq(gap, low, high, xtol=tolerance))


@dataclass(frozen=True)
class SmilePoint:
    option_type: str
    strike: float
    expiry: date
    maturity: float
    log_moneyness: float          # ln(K / F)
    price: float                  # market mid
    implied_volatility: float
    yahoo_implied_volatility: float


def parity_forward(quotes: list[OptionQuote], spot: float, band: float = 0.1,
                   max_spread: float = 0.5) -> tuple[float, float]:
    """
    (forward, discount factor) of one expiry from put-call parity, C - P = D (F - K), by least
    squares on the strikes within `band` of the spot that have a two-sided call and put quote.
    """
    calls = {q.strike: q.mid for q in quotes if q.option_type == CALL and q.bid > 0 and q.relative_spread <= max_spread}
    puts = {q.strike: q.mid for q in quotes if q.option_type != CALL and q.bid > 0 and q.relative_spread <= max_spread}
    strikes = np.array(sorted(k for k in calls.keys() & puts.keys() if abs(k/spot - 1) <= band))
    if len(strikes) < 3:
        raise ValueError("Not enough strikes with both a call and a put quote near the money.")
    gaps = np.array([calls[k] - puts[k] for k in strikes])
    slope, intercept = np.polyfit(strikes, gaps, 1)
    discount = -slope
    if not 0 < discount <= 1.5:
        raise ValueError(f"Put-call parity gives an implausible discount factor {discount:.3f}.")
    return intercept/discount, discount


def volatility_smile(quotes: list[OptionQuote], snapshot: MarketSnapshot, american: bool = False,
                     otm_only: bool = True, max_spread: float = 0.5,
                     forward_from_parity: bool = False) -> list[SmilePoint]:
    """Implied volatilities of one expiry; quotes that cannot be inverted are skipped."""
    market = snapshot.market()
    if forward_from_parity and quotes:
        # rate and carry implied by this expiry's own quotes (European options)
        maturity = quotes[0].maturity(snapshot.as_of)
        forward, discount = parity_forward(quotes, snapshot.spot, max_spread=max_spread)
        rate = -math.log(discount)/maturity
        market = Market(snapshot.spot, rate, market.volatility, rate - math.log(forward/snapshot.spot)/maturity)
    price_fn = baw_price if american else european_price
    points = []
    for quote in quotes:
        maturity = quote.maturity(snapshot.as_of)
        if maturity <= 0 or quote.bid <= 0 or quote.relative_spread > max_spread:
            continue
        forward = snapshot.spot*math.exp((market.rate - market.dividend_yield)*maturity)
        is_call = quote.option_type == CALL
        if otm_only and (quote.strike < forward if is_call else quote.strike > forward):
            continue
        option = Option(quote.option_type, quote.strike, maturity)
        try:
            sigma = implied_volatility(quote.mid, option, market, price_fn)
        except ValueError:
            continue
        points.append(SmilePoint(quote.option_type, quote.strike, quote.expiry, maturity,
                                 math.log(quote.strike/forward), quote.mid, sigma, quote.yahoo_implied_volatility))
    return sorted(points, key=lambda p: p.strike)


def atm_volatility(points: list[SmilePoint]) -> float:
    """Implied volatility at the forward (log-moneyness 0), interpolated linearly on one smile."""
    if not points:
        raise ValueError("Empty smile.")
    x = np.array([p.log_moneyness for p in points])
    y = np.array([p.implied_volatility for p in points])
    order = np.argsort(x)
    return float(np.interp(0.0, x[order], y[order]))


def pick_expiries(expiries: list[date], as_of: date, target_days=(30, 60, 91, 182, 365, 730),
                  min_days: int = 7) -> list[date]:
    """The listed expiry closest to each target maturity, at least `min_days` away, without duplicates."""
    candidates = [e for e in expiries if (e - as_of).days >= min_days]
    chosen = {min(candidates, key=lambda e: abs((e - as_of).days - days)) for days in target_days} if candidates else set()
    return sorted(chosen)


def volatility_surface(ticker: str, client: YahooClient | None = None, snapshot: MarketSnapshot | None = None,
                       target_days=(30, 60, 91, 182, 365, 730), min_days: int = 7, american: bool = False,
                       max_spread: float = 0.5, forward_from_parity: bool = False) -> list[SmilePoint]:
    """
    Smiles of the expiries closest to the target maturities (in days). Expiries
    closer than `min_days` are skipped: their time values are tiny, so their
    implied volatilities are noisy.
    """
    client = client or YahooClient()
    snapshot = snapshot or fetch_snapshot(ticker, client)
    points = []
    for expiry in pick_expiries(option_expiries(ticker, client), snapshot.as_of, target_days, min_days):
        points += volatility_smile(option_quotes(ticker, expiry, client), snapshot, american, max_spread=max_spread,
                                   forward_from_parity=forward_from_parity)
    return points


def term_structure(points: list[SmilePoint]) -> list[tuple[date, float, float]]:
    """(expiry, maturity, ATM implied volatility) for each expiry of a surface."""
    rows = []
    for expiry in sorted({p.expiry for p in points}):
        smile = [p for p in points if p.expiry == expiry]
        rows.append((expiry, smile[0].maturity, atm_volatility(smile)))
    return rows
