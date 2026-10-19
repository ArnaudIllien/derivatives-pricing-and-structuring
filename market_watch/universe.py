"""
universe.py - The instruments shown by the dashboard, all available on Yahoo Finance.

Each instrument has a Yahoo ticker, a display name, a category and a region.
Not available for free on Yahoo (checked): VSTOXX (the dashboard shows the
realised volatility of the Euro Stoxx 50 instead), the Nigerian and Moroccan
indices. Edit the lists to change what is shown.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    category: str     # "Index", "FX", "Commodity", "Volatility", "Rate"
    region: str       # region for indices, group for the other categories


def _many(category, region, pairs):
    return tuple(Instrument(ticker, name, category, region) for ticker, name in pairs)


INDICES = (
    _many("Index", "Europe", [("^STOXX50E", "Euro Stoxx 50"), ("^FCHI", "CAC 40"), ("^GDAXI", "DAX"),
                              ("^FTSE", "FTSE 100"), ("^IBEX", "IBEX 35"), ("FTSEMIB.MI", "FTSE MIB"),
                              ("^AEX", "AEX"), ("^SSMI", "SMI")])
    + _many("Index", "Americas", [("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^IXIC", "Nasdaq Composite"),
                                  ("^RUT", "Russell 2000"), ("^GSPTSE", "S&P/TSX"), ("^BVSP", "Bovespa"),
                                  ("^MXX", "IPC Mexico")])
    + _many("Index", "Asia-Pacific", [("^N225", "Nikkei 225"), ("^HSI", "Hang Seng"), ("000001.SS", "Shanghai Composite"),
                                      ("^BSESN", "BSE Sensex"), ("^NSEI", "Nifty 50"), ("^KS11", "KOSPI"),
                                      ("^TWII", "Taiwan Weighted"), ("^AXJO", "S&P/ASX 200"), ("^STI", "Straits Times")])
    + _many("Index", "Africa", [("^J203.JO", "JSE All Share"), ("^JN0U.JO", "JSE Top 40"), ("^CASE30", "EGX 30")])
)

FX = (
    _many("FX", "Majors", [("EURUSD=X", "EUR/USD"), ("GBPUSD=X", "GBP/USD"), ("USDJPY=X", "USD/JPY"),
                           ("USDCHF=X", "USD/CHF"), ("AUDUSD=X", "AUD/USD"), ("USDCAD=X", "USD/CAD"),
                           ("EURGBP=X", "EUR/GBP"), ("DX-Y.NYB", "Dollar index")])
    + _many("FX", "Emerging", [("USDCNY=X", "USD/CNY"), ("USDINR=X", "USD/INR"), ("USDBRL=X", "USD/BRL"),
                               ("USDMXN=X", "USD/MXN"), ("USDZAR=X", "USD/ZAR"), ("USDEGP=X", "USD/EGP"),
                               ("USDNGN=X", "USD/NGN"), ("USDMAD=X", "USD/MAD")])
    + _many("FX", "Crypto", [("BTC-USD", "Bitcoin"), ("ETH-USD", "Ether")])
)

COMMODITIES = (
    _many("Commodity", "Energy", [("CL=F", "WTI crude"), ("BZ=F", "Brent crude"), ("NG=F", "US natural gas"),
                                  ("TTF=F", "Dutch TTF gas"), ("HO=F", "Heating oil"), ("RB=F", "Gasoline")])
    + _many("Commodity", "Metals", [("GC=F", "Gold"), ("SI=F", "Silver"), ("PL=F", "Platinum"), ("HG=F", "Copper")])
    + _many("Commodity", "Agriculture", [("ZW=F", "Wheat"), ("ZC=F", "Corn"), ("ZS=F", "Soybeans"),
                                         ("KC=F", "Coffee"), ("SB=F", "Sugar"), ("CC=F", "Cocoa"), ("CT=F", "Cotton")])
)

VOLATILITY = _many("Volatility", "Implied volatility", [
    ("^VIX", "VIX (S&P 500)"), ("^VXN", "VXN (Nasdaq 100)"), ("^VVIX", "VVIX (vol of VIX)"),
    ("^MOVE", "MOVE (US Treasuries)"), ("^OVX", "OVX (crude oil)"), ("^GVZ", "GVZ (gold)"), ("^SKEW", "CBOE SKEW")])

US_RATES = _many("Rate", "US Treasury (Yahoo)", [("^IRX", "13-week T-bill"), ("^FVX", "5-year"),
                                                 ("^TNX", "10-year"), ("^TYX", "30-year")])

ALL = INDICES + FX + COMMODITIES + VOLATILITY + US_RATES
BY_TICKER = {instrument.ticker: instrument for instrument in ALL}


def tickers(instruments) -> list[str]:
    return [instrument.ticker for instrument in instruments]


def name_of(ticker: str) -> str:
    instrument = BY_TICKER.get(ticker)
    return instrument.name if instrument else ticker
