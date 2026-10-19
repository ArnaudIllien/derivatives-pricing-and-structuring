"""
fred.py - Economic series of the St. Louis Fed (FRED), through its public CSV download (no key).

`fred_series(series_id, start)` returns a float Series indexed by date
(missing observations, written '.', are dropped). `MACRO_SERIES` lists the
indicators shown by the dashboard; any FRED id works (search on fred.stlouisfed.org).
"""
import io
from datetime import date

import pandas as pd

from ..net import SourceError, get_text

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# id: (name, unit, how to read it)
MACRO_SERIES = {
    "DFF": ("Fed funds effective rate", "%", "level"),
    "SOFR": ("SOFR", "%", "level"),
    "T10Y2Y": ("US 10y - 2y spread", "%", "level"),
    "T10YIE": ("US 10y breakeven inflation", "%", "level"),
    "BAMLH0A0HYM2": ("US high-yield spread", "%", "level"),
    "CPIAUCSL": ("US CPI", "index", "yoy"),
    "PCEPILFE": ("US core PCE", "index", "yoy"),
    "UNRATE": ("US unemployment rate", "%", "level"),
    "PAYEMS": ("US nonfarm payrolls", "thousands", "change"),
    "GDPC1": ("US real GDP", "bn chained $", "yoy"),
    "INDPRO": ("US industrial production", "index", "yoy"),
    "UMCSENT": ("Michigan consumer sentiment", "index", "level"),
}


def parse_csv(text: str) -> pd.Series:
    frame = pd.read_csv(io.StringIO(text))
    if len(frame.columns) != 2:
        raise SourceError("FRED: unexpected answer")
    date_column, value_column = frame.columns
    values = pd.to_numeric(frame[value_column], errors="coerce")
    series = pd.Series(values.values, index=pd.to_datetime(frame[date_column]), name=value_column).dropna()
    series.index.name = "date"
    return series


def fred_series(series_id: str, start: date) -> pd.Series:
    return parse_csv(get_text(URL, {"id": series_id, "cosd": start.isoformat()}))


def transform(series: pd.Series, how: str) -> pd.Series:
    """'level' as is, 'yoy' year-on-year % change, 'change' change since the previous observation."""
    if how == "level":
        return series
    if how == "change":
        return series.diff().dropna()
    # observations per year, from the typical spacing: about 365 (daily), 52, 12 or 4
    spacing = series.index.to_series().diff().dt.days.median()
    periods = max(1, round(365/spacing)) if spacing and spacing > 3 else 252
    return (100*(series/series.shift(periods) - 1)).dropna()
