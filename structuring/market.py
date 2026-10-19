"""
market.py - Underlyings and market data, with the provenance and date of every parameter.

Every number used in a valuation is a `Parameter` tagged with one of:
- OBSERVED       read from a market source on a given date (closing price, government yield);
- ESTIMATED      computed from observed data with a stated method (historical
                 volatility, implied volatility inverted from listed option quotes,
                 ETF distribution yield used as a dividend proxy);
- HYPOTHETICAL   chosen by the user or a documented default (issuer credit
                 spread, fees, any parameter typed by hand, every fallback
                 when a source is unavailable).
Nothing hypothetical is ever presented as a market price.

Model by asset class (all reduce to a GBM with a carry yield q, priced by pricer/):
- equity index / single stock: q = dividend yield (Black-Scholes-Merton);
- fx (price of 1 unit of the base currency in the quote currency): q = interest
  rate of the base currency (Garman-Kohlhagen);
- commodity: the front-month future is the underlying, q = r (Black's model). The
  futures curve is assumed flat (a hypothesis: there is no free futures curve).
The product is denominated in the quote currency of the underlying, and the
performance S_t / S_0 is simulated (spot normalised to 1).

Rates: the risk-free rate is the government yield at the product maturity
(US Treasury par curve for USD, ECB AAA spot curve for EUR), used as a flat
continuously compounded rate. Other currencies have no free curve: hypothetical.
"""
import math
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum

import numpy as np

from pricer.market import Market


class Provenance(Enum):
    OBSERVED = "observed"
    ESTIMATED = "model estimate"
    HYPOTHETICAL = "hypothetical"


@dataclass(frozen=True)
class Parameter:
    value: float
    provenance: Provenance
    source: str
    as_of: date | None = None

    def with_value(self, value: float, source: str = "user input") -> "Parameter":
        """The same parameter overridden by the user: always hypothetical."""
        return Parameter(value, Provenance.HYPOTHETICAL, source, None)


@dataclass(frozen=True)
class Underlying:
    name: str
    ticker: str                    # Yahoo Finance symbol
    asset_class: str               # "equity index", "single stock", "fx", "commodity"
    currency: str                  # currency of the price (quote currency for fx)
    base_currency: str | None = None    # fx only
    dividend_proxy: str | None = None   # ETF whose distribution yield proxies the index dividends
    options_ticker: str | None = None   # Yahoo symbol with listed options, for implied volatility
    risk_factors: tuple[str, ...] = ()


EQUITY_FACTORS = ("Equity volatility and its skew: barrier and autocall values depend on the downside volatility.",
                  "Dividends: a higher dividend yield lowers the forward, cheapens calls, raises put values.",
                  "Interest rates: they set the forward and the cost of capital protection (zero-coupon bond).")
UNDERLYINGS = {u.name: u for u in (
    Underlying("Euro Stoxx 50", "^STOXX50E", "equity index", "EUR", dividend_proxy="EXW1.DE",
               risk_factors=EQUITY_FACTORS + ("Price index: dividends are not reinvested in the index level.",)),
    Underlying("CAC 40", "^FCHI", "equity index", "EUR", dividend_proxy="CAC.PA",
               risk_factors=EQUITY_FACTORS + ("Concentrated in luxury, energy and industrials.",)),
    Underlying("S&P 500", "^GSPC", "equity index", "USD", dividend_proxy="SPY", options_ticker="^SPX",
               risk_factors=EQUITY_FACTORS + ("For a EUR investor: EUR/USD exchange-rate risk (not hedged).",)),
    Underlying("LVMH", "MC.PA", "single stock", "EUR",
               risk_factors=EQUITY_FACTORS + ("Single-stock risk: earnings, sector and event risk, higher volatility "
                                              "than an index.",)),
    Underlying("TotalEnergies", "TTE.PA", "single stock", "EUR",
               risk_factors=EQUITY_FACTORS + ("Oil and gas price sensitivity; high dividend.",)),
    Underlying("Apple", "AAPL", "single stock", "USD", options_ticker="AAPL",
               risk_factors=EQUITY_FACTORS + ("Single-stock risk; for a EUR investor, EUR/USD risk.",)),
    Underlying("MSCI ACWI (world ETF)", "ACWI", "equity index", "USD", dividend_proxy="ACWI",
               risk_factors=EQUITY_FACTORS + ("Diversified across about 50 countries; about 60 % US equities.",
                                              "USD-listed ETF: EUR/USD risk for a EUR investor.")),
    Underlying("EUR/USD", "EURUSD=X", "fx", "USD", base_currency="EUR",
               risk_factors=("FX volatility.", "Interest-rate differential USD - EUR: it sets the forward "
                             "(Garman-Kohlhagen).", "Central-bank policy and macro surprises.")),
    Underlying("Gold", "GC=F", "commodity", "USD",
               risk_factors=("Commodity volatility.", "Futures curve (contango/backwardation) and roll: assumed "
                             "flat here.", "USD risk for a EUR investor.")),
    Underlying("Brent crude", "BZ=F", "commodity", "USD",
               risk_factors=("High volatility, supply shocks and geopolitics.", "Futures curve and roll: assumed "
                             "flat here.", "USD risk for a EUR investor.")),
)}


@dataclass(frozen=True)
class MarketData:
    """Everything a valuation needs, for one underlying on one date."""
    underlying: Underlying
    as_of: date
    spot: Parameter               # absolute level, for display; simulations use the performance (spot = 1)
    rate: Parameter               # continuous risk-free rate of the product currency
    carry: Parameter              # q: dividend yield, foreign rate, or r for a future
    volatility: Parameter
    credit_spread: Parameter      # issuer credit spread over the risk-free rate (per year)
    recovery: Parameter           # issuer recovery rate on default
    notes: tuple[str, ...] = ()

    def parameters(self) -> dict[str, Parameter]:
        return {"spot": self.spot, "risk-free rate": self.rate, "carry yield q": self.carry,
                "volatility": self.volatility, "issuer credit spread": self.credit_spread,
                "issuer recovery rate": self.recovery}

    def pricer_market(self, vol_shift: float = 0.0, rate_shift: float = 0.0) -> Market:
        """Market of the normalised performance (spot 1) for the pricer/ package."""
        carry = self.carry.value + (rate_shift if self.underlying.asset_class == "commodity" else 0.0)
        return Market(1.0, self.rate.value + rate_shift, max(self.volatility.value + vol_shift, 0.0), carry)

    def with_overrides(self, **values: float) -> "MarketData":
        """Replaces parameters by user values (they become hypothetical). Keys: rate, carry, volatility,
        credit_spread, recovery."""
        changes = {name: getattr(self, name).with_value(value) for name, value in values.items()}
        if "rate" in values and self.underlying.asset_class == "commodity" and "carry" not in values:
            changes["carry"] = Parameter(values["rate"], Provenance.HYPOTHETICAL, "q = r (future)", None)
        return replace(self, **changes)


# Documented defaults, used when a source is unavailable or for parameters with no free source
DEFAULT_CREDIT_SPREAD = 0.010   # 100 bp: indicative order of magnitude for a large bank issuer, NOT observed
DEFAULT_RECOVERY = 0.40         # market convention for senior unsecured debt
FALLBACK = {"rate": 0.03, "carry": 0.02, "volatility": 0.20}


def hypothetical_market(underlying: Underlying, as_of: date | None = None, rate: float = FALLBACK["rate"],
                        carry: float | None = None, volatility: float = FALLBACK["volatility"]) -> MarketData:
    """A fully hypothetical market (no network, demos, tests). Every parameter is tagged HYPOTHETICAL."""
    as_of = as_of or date.today()
    if carry is None:
        carry = rate if underlying.asset_class == "commodity" else FALLBACK["carry"]
    h = lambda v, s: Parameter(v, Provenance.HYPOTHETICAL, s, None)
    return MarketData(underlying, as_of, h(1.0, "normalised level"), h(rate, "default"), h(carry, "default"),
                      h(volatility, "default"), h(DEFAULT_CREDIT_SPREAD, "indicative default, not observed"),
                      h(DEFAULT_RECOVERY, "market convention"),
                      ("Hypothetical market: no observed data.",))


def _curve_rate(curve, maturity: float, compounding: str) -> tuple[float, date]:
    """Interpolates the latest curve at the maturity; converts to a continuous rate."""
    latest = curve.dropna(how="all").iloc[-1].dropna()
    tenors = np.array(latest.index, dtype=float)
    value = float(np.interp(maturity, tenors, latest.values))/100
    if compounding == "annual":
        rate = math.log(1 + value)
    else:  # semi-annual par yield, read as a semi-annual zero rate (flat-curve approximation)
        rate = 2*math.log(1 + value/2)
    return rate, curve.dropna(how="all").index[-1].date()


def risk_free_rate(currency: str, maturity: float) -> Parameter:
    """Government yield at the maturity (observed), or a hypothetical default for other currencies."""
    from market_watch.sources import ecb, treasury
    from datetime import timedelta
    today = date.today()
    if currency == "USD":
        rate, as_of = _curve_rate(treasury.us_yield_curve([today.year - (today.month == 1), today.year]),
                                  maturity, "semi-annual")
        return Parameter(rate, Provenance.OBSERVED, f"US Treasury par curve at {maturity:g}y (flat)", as_of)
    if currency == "EUR":
        rate, as_of = _curve_rate(ecb.euro_yield_curve(today - timedelta(days=10)), maturity, "annual")
        return Parameter(rate, Provenance.OBSERVED, f"ECB AAA euro area spot curve at {maturity:g}y (flat)", as_of)
    return Parameter(FALLBACK["rate"], Provenance.HYPOTHETICAL, f"no free {currency} curve: default", None)


def load_market(underlying: Underlying, maturity: float, volatility_source: str = "implied",
                credit_spread: float = DEFAULT_CREDIT_SPREAD, recovery: float = DEFAULT_RECOVERY) -> MarketData:
    """
    Live market data (network). `volatility_source`: "implied" (ATM implied volatility of the
    listed expiry closest to the maturity, when the underlying has listed options on Yahoo;
    otherwise historical) or "historical" (one year of daily returns).
    """
    from pricer.implied_vol import atm_volatility, pick_expiries, volatility_smile
    from pricer.market_data import fetch_snapshot, option_expiries, option_quotes

    notes = []
    snapshot = fetch_snapshot(underlying.ticker, rate=0.0)
    spot = Parameter(snapshot.spot, Provenance.OBSERVED, f"Yahoo Finance close ({underlying.ticker})", snapshot.as_of)
    try:
        rate = risk_free_rate(underlying.currency, maturity)
    except Exception as error:   # the curve source is down: say so, use the default
        rate = Parameter(FALLBACK["rate"], Provenance.HYPOTHETICAL, f"curve unavailable ({error}): default", None)

    if underlying.asset_class == "commodity":
        carry = Parameter(rate.value, rate.provenance, "q = r: the underlying is a future (Black)", rate.as_of)
        notes.append("Flat futures curve assumed: the future of the product maturity is taken at today's "
                     "front-month price.")
    elif underlying.asset_class == "fx":
        try:
            foreign = risk_free_rate(underlying.base_currency, maturity)
            carry = replace(foreign, source=f"{underlying.base_currency} rate (Garman-Kohlhagen): {foreign.source}")
        except Exception as error:
            carry = Parameter(FALLBACK["rate"], Provenance.HYPOTHETICAL, f"foreign curve unavailable ({error})", None)
    elif underlying.dividend_proxy:
        from pricer.market_data import YahooClient
        proxy_yield = YahooClient().info(underlying.dividend_proxy).get("yield")
        if proxy_yield:
            carry = Parameter(math.log(1 + float(proxy_yield)), Provenance.ESTIMATED,
                              f"distribution yield of ETF {underlying.dividend_proxy} (dividend proxy)", snapshot.as_of)
        else:
            carry = Parameter(FALLBACK["carry"], Provenance.HYPOTHETICAL, "dividend proxy unavailable: default", None)
    else:
        carry = Parameter(snapshot.dividend_yield, Provenance.ESTIMATED, "trailing 12-month dividend yield (Yahoo)",
                          snapshot.as_of)

    volatility = Parameter(snapshot.historical_volatility, Provenance.ESTIMATED,
                           "historical volatility, 1 year of daily returns", snapshot.as_of)
    if volatility_source == "implied" and underlying.options_ticker:
        try:
            options_snapshot = fetch_snapshot(underlying.options_ticker, rate=rate.value)
            options_snapshot = replace(options_snapshot, dividend_yield=carry.value)
            expiries = pick_expiries(option_expiries(underlying.options_ticker), options_snapshot.as_of,
                                     (round(maturity*365),))
            smile = volatility_smile(option_quotes(underlying.options_ticker, expiries[0]), options_snapshot)
            volatility = Parameter(atm_volatility(smile), Provenance.ESTIMATED,
                                   f"ATM implied volatility, {underlying.options_ticker} options expiring {expiries[0]}",
                                   options_snapshot.as_of)
            if (expiries[0] - options_snapshot.as_of).days < 0.75*maturity*365:
                notes.append(f"The longest listed expiry ({expiries[0]}) is shorter than the product maturity.")
        except Exception as error:
            notes.append(f"Implied volatility unavailable ({error}): historical volatility used.")
    elif volatility_source == "implied":
        notes.append("No listed options for this underlying on the free source: historical volatility used.")
    notes.append("Flat volatility (no skew): barrier and autocallable values ignore the steeper downside "
                 "volatility, which usually overstates their fair value for the investor.")

    return MarketData(underlying, snapshot.as_of, spot, rate, carry, volatility,
                      Parameter(credit_spread, Provenance.HYPOTHETICAL, "issuer spread: user input or indicative default"),
                      Parameter(recovery, Provenance.HYPOTHETICAL, "market convention"), tuple(notes))
