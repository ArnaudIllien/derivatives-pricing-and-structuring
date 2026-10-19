"""
market_data.py - Live market data from Yahoo Finance (and the ECB for euro rates) for the pricers.

USAGE
    from pricer.market_data import fetch_snapshot, option_quotes, option_expiries
    snapshot = fetch_snapshot("SPY")          # spot, rate, dividend yield, historical volatility
    market = snapshot.market()                # a pricer Market, volatility = historical
    quotes = option_quotes("SPY", option_expiries("SPY")[3])

    Needs network access and `pip install yfinance`. Only this module imports
    yfinance, and only when it is called: the pricers work without it.

What is fetched, and the conventions used
-----------------------------------------
- Spot: last daily close.
- Rate (continuously compounded), by the currency of the underlying:
  - USD: 13-week T-bill (^IRX), quoted as a discount yield d in %:
    price = 1 - d * 91/360, r = -ln(price) / (91/365);
  - EUR: euro short-term rate (€STR) from the ECB Data Portal, r = ln(1 + estr);
  - other currencies: no free source, the USD rate is used and `rate_source` says so.
  Pass `rate=` to override.
- Dividend yield: Yahoo trailing annual yield y (a fraction), q = ln(1 + y); 0 for indices.
- Historical volatility: std of daily log returns over `volatility_window`
  trading days, annualised with sqrt(252).
- Option quotes: bid, ask, last, volume, open interest and Yahoo's own implied
  volatility, per strike, for one expiry.

`YahooClient` is the only object that talks to the network. Tests replace it
with a fake that returns recorded data.
"""
import json
import math
import urllib.request
from dataclasses import dataclass
from datetime import date

import numpy as np

from .contract import CALL, PUT, year_fraction
from .market import Market

ECB_ESTR_URL = "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?lastNObservations=1&format=jsondata"
US_TBILL = "^IRX"


class YahooClient:
    """Thin wrapper over yfinance. Every method returns plain Python or pandas objects."""

    @staticmethod
    def _yf():
        try:
            import yfinance
        except ImportError as error:
            raise ImportError("Live market data needs yfinance: python -m pip install yfinance") from error
        return yfinance

    def history(self, ticker: str, period: str = "2y", start=None, end=None):
        """Daily closes as a pandas Series indexed by date (empty if Yahoo has nothing)."""
        frame = self._yf().Ticker(ticker).history(period=None if start else period, start=start, end=end,
                                                  auto_adjust=False)
        closes = frame["Close"].dropna() if "Close" in frame else frame
        closes.index = [d.date() for d in closes.index]
        return closes

    def info(self, ticker: str) -> dict:
        try:
            return dict(self._yf().Ticker(ticker).info or {})
        except Exception:   # Yahoo returns 404 for some indices: the price history is enough
            return {}

    def expiries(self, ticker: str) -> tuple[str, ...]:
        return tuple(self._yf().Ticker(ticker).options)

    def option_chain(self, ticker: str, expiry: str):
        """(calls, puts) as pandas DataFrames with Yahoo's columns."""
        chain = self._yf().Ticker(ticker).option_chain(expiry)
        return chain.calls, chain.puts


@dataclass(frozen=True)
class MarketSnapshot:
    ticker: str
    name: str
    currency: str
    as_of: date
    spot: float
    rate: float
    rate_source: str
    dividend_yield: float
    historical_volatility: float
    volatility_window: int

    def market(self, volatility: float | None = None) -> Market:
        """Pricer market data; the volatility defaults to the historical one."""
        sigma = self.historical_volatility if volatility is None else volatility
        return Market(self.spot, self.rate, sigma, self.dividend_yield)


def tbill_rate(discount_yield_percent: float) -> float:
    """Continuously compounded rate of a 13-week T-bill quoted as a discount yield in %."""
    price = 1 - discount_yield_percent/100*91/360
    return -math.log(price)/(91/365)


def historical_volatility(closes, window: int = 252) -> float:
    """Annualised std of the last `window` daily log returns."""
    values = np.asarray(closes, dtype=float)[-(window + 1):]
    if len(values) < 3:
        raise ValueError("Not enough prices to estimate a volatility.")
    returns = np.diff(np.log(values))
    return float(returns.std(ddof=1)*math.sqrt(252))


def fetch_estr() -> float:
    """Latest €STR, as a fraction (ECB Data Portal, no key needed)."""
    request = urllib.request.Request(ECB_ESTR_URL, headers={"User-Agent": "pricer/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)
    observations = data["dataSets"][0]["series"]["0:0:0"]["observations"]
    return observations[max(observations, key=int)][0]/100


def risk_free_rate(currency: str, client: YahooClient, estr=fetch_estr) -> tuple[float, str]:
    """(continuous rate, description of its source) for the currency of the underlying."""
    if currency == "EUR":
        return math.log(1 + estr()), "ECB euro short-term rate (€STR)"
    closes = client.history(US_TBILL, period="1mo")
    if len(closes) == 0:
        raise ValueError("No US T-bill rate available from Yahoo.")
    source = "US 13-week T-bill (^IRX)"
    if currency != "USD":
        source += f" - no free {currency} rate, USD used instead"
    return tbill_rate(float(closes.iloc[-1])), source


def fetch_snapshot(ticker: str, client: YahooClient | None = None, volatility_window: int = 252,
                   rate: float | None = None, estr=fetch_estr) -> MarketSnapshot:
    client = client or YahooClient()
    closes = client.history(ticker, period="2y")
    if len(closes) == 0:
        raise ValueError(f"Yahoo Finance has no price history for {ticker!r}.")
    info = client.info(ticker)
    currency = info.get("currency") or "USD"
    if rate is None:
        rate, rate_source = risk_free_rate(currency, client, estr)
    else:
        rate_source = "user input"
    trailing = info.get("trailingAnnualDividendYield") or 0.0
    return MarketSnapshot(
        ticker=ticker, name=(info.get("shortName") or info.get("longName") or ticker).strip(), currency=currency,
        as_of=closes.index[-1], spot=float(closes.iloc[-1]), rate=rate, rate_source=rate_source,
        dividend_yield=math.log(1 + float(trailing)),
        historical_volatility=historical_volatility(closes.values, volatility_window),
        volatility_window=volatility_window)


@dataclass(frozen=True)
class OptionQuote:
    option_type: str
    strike: float
    expiry: date
    bid: float
    ask: float
    last: float
    volume: float
    open_interest: float
    yahoo_implied_volatility: float

    @property
    def mid(self) -> float:
        """Mid price if there is a two-sided quote, else the last trade."""
        if self.bid > 0 and self.ask >= self.bid:
            return (self.bid + self.ask)/2
        return self.last

    @property
    def relative_spread(self) -> float:
        return (self.ask - self.bid)/self.mid if self.bid > 0 and self.mid > 0 else math.inf

    def maturity(self, as_of: date) -> float:
        return year_fraction(as_of, self.expiry)


def option_expiries(ticker: str, client: YahooClient | None = None) -> list[date]:
    client = client or YahooClient()
    return [date.fromisoformat(e) for e in client.expiries(ticker)]


def _number(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(value) else value


def option_quotes(ticker: str, expiry: date, client: YahooClient | None = None) -> list[OptionQuote]:
    """Calls and puts of one expiry, sorted by type then strike."""
    client = client or YahooClient()
    calls, puts = client.option_chain(ticker, expiry.isoformat())
    quotes = []
    for option_type, table in ((CALL, calls), (PUT, puts)):
        for row in table.to_dict("records"):
            quotes.append(OptionQuote(option_type, _number(row["strike"]), expiry, _number(row.get("bid")),
                                      _number(row.get("ask")), _number(row.get("lastPrice")),
                                      _number(row.get("volume")), _number(row.get("openInterest")),
                                      _number(row.get("impliedVolatility"))))
    return sorted(quotes, key=lambda q: (q.option_type, q.strike))
