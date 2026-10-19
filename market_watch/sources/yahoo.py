"""
yahoo.py - Daily closes of many tickers at once from Yahoo Finance (yfinance).

`download_closes(tickers, period)` returns a DataFrame indexed by date with one
column per ticker that Yahoo knows; tickers without data are left out (the
caller can compare the columns with what it asked for). Prices are adjusted
for splits and dividends (auto_adjust=True).
"""
import pandas as pd

from ..net import SourceError


def download_closes(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    import yfinance as yf
    try:
        data = yf.download(list(tickers), period=period, progress=False, auto_adjust=True, threads=True)
    except Exception as error:
        raise SourceError(f"Yahoo Finance: {error}") from error
    if data is None or data.empty:
        raise SourceError("Yahoo Finance returned no data.")
    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].set_axis(tickers, axis=1)
    closes = closes.dropna(axis=1, how="all")
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes.sort_index()
