"""
analytics.py - Market statistics for the dashboard, on DataFrames of daily closes (pure pandas, no network).

- `performance_table`: last price and % change over 1 day, 1 week, 1 month,
  year to date and 1 year (rates and volatility indices: change in points).
- `big_moves`: today's moves measured against their own history, as a z-score
  z = today's log return / std of the daily log returns over the past year.
  |z| >= 2 is a move that happens about one day in twenty for a normal law.
- `correlation_matrix`: correlations of daily log returns over a window.
  Markets close at different times (Asia before Europe before the US), so
  same-day correlations across time zones are understated.
- `realized_volatility`: annualised rolling volatility of the log returns.
- `curve_snapshots`: a yield curve today, about 1 month ago and 1 year ago.

Conventions: "1 week ago" is the last close on or before the same date 7
calendar days earlier, and likewise for the other horizons, so holidays do not
break the table. Year to date compares with the last close of the previous year.
"""
import numpy as np
import pandas as pd

HORIZONS = {"1D": None, "1W": pd.DateOffset(weeks=1), "1M": pd.DateOffset(months=1),
            "YTD": "ytd", "1Y": pd.DateOffset(years=1)}
POINT_CATEGORIES = ("Rate", "Volatility")


def _value_on_or_before(series: pd.Series, when) -> float:
    past = series.loc[:when]
    return float(past.iloc[-1]) if len(past) else np.nan


def _reference(series: pd.Series, horizon) -> float:
    last_date = series.index[-1]
    if horizon is None:
        return float(series.iloc[-2]) if len(series) > 1 else np.nan
    if horizon == "ytd":
        return _value_on_or_before(series, pd.Timestamp(year=last_date.year - 1, month=12, day=31))
    return _value_on_or_before(series, last_date - horizon)


def performance_table(closes: pd.DataFrame, instruments) -> pd.DataFrame:
    """One row per instrument found in `closes`: name, group, last close, date, then one column per horizon."""
    rows = []
    for instrument in instruments:
        if instrument.ticker not in closes:
            continue
        series = closes[instrument.ticker].dropna()
        if len(series) < 2:
            continue
        last = float(series.iloc[-1])
        in_points = instrument.category in POINT_CATEGORIES
        row = {"ticker": instrument.ticker, "name": instrument.name, "group": instrument.region,
               "last": last, "date": series.index[-1].date()}
        for label, horizon in HORIZONS.items():
            reference = _reference(series, horizon)
            row[label] = last - reference if in_points else 100*(last/reference - 1)
        row["unit"] = "points" if in_points else "%"
        rows.append(row)
    return pd.DataFrame(rows)


def log_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns of each column, on its own trading days (no filling across holidays)."""
    return pd.DataFrame({column: np.log(closes[column].dropna()).diff() for column in closes.columns}).dropna(how="all")


def big_moves(closes: pd.DataFrame, instruments, window: int = 252, threshold: float = 2.0,
              max_age_days: int = 1) -> pd.DataFrame:
    """
    Instruments whose last daily move is at least `threshold` standard deviations, largest
    first. A market whose last close is more than `max_age_days` before the latest date in
    `closes` (closed today: holiday, weekend) is left out, so old moves do not show up as news.
    """
    rows = []
    latest = closes.index.max()
    for instrument in instruments:
        if instrument.ticker not in closes or instrument.category in POINT_CATEGORIES:
            continue
        returns = np.log(closes[instrument.ticker].dropna()).diff().dropna()
        if len(returns) < 30 or (latest - returns.index[-1]).days > max_age_days:
            continue
        today, past = float(returns.iloc[-1]), returns.iloc[-window - 1:-1]
        std = float(past.std(ddof=1))
        if std == 0:
            continue
        z = today/std
        if abs(z) >= threshold:
            rows.append({"ticker": instrument.ticker, "name": instrument.name, "category": instrument.category,
                         "group": instrument.region, "date": returns.index[-1].date(),
                         "move %": 100*(np.exp(today) - 1), "z-score": z, "typical daily move %": 100*std})
    frame = pd.DataFrame(rows)
    return frame.reindex(frame["z-score"].abs().sort_values(ascending=False).index) if len(frame) else frame


def correlation_matrix(closes: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    returns = log_returns(closes).iloc[-window:]
    return returns.corr(min_periods=max(20, window//3))


def realized_volatility(closes: pd.Series, window: int = 21) -> pd.Series:
    """Annualised volatility of the daily log returns over a rolling window, in %."""
    returns = np.log(closes.dropna()).diff()
    return 100*returns.rolling(window).std()*np.sqrt(252)


def curve_snapshots(curve: pd.DataFrame) -> pd.DataFrame:
    """Columns: today, ~1 month ago, ~1 year ago (last curve on or before each date); rows: tenors."""
    curve = curve.dropna(how="all")
    if curve.empty:
        return pd.DataFrame()
    last = curve.index[-1]
    snapshots = {}
    for label, offset in (("today", None), ("1 month ago", pd.DateOffset(months=1)),
                          ("1 year ago", pd.DateOffset(years=1))):
        past = curve.loc[:last - offset] if offset is not None else curve
        if len(past):
            snapshots[f"{label} ({past.index[-1].date()})"] = past.iloc[-1]
    return pd.DataFrame(snapshots)
