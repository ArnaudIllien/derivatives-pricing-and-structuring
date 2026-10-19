"""
treasury.py - US Treasury par yield curve from the US Treasury website (no key).

`us_yield_curve(years)` returns the daily curve, in %, one column per tenor in
years (1 month to 30 years), for the given calendar years. The Treasury
publishes one CSV per year, so a comparison with one year ago needs two years.
"""
import io

import pandas as pd

from ..net import SourceError, get_text

URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
       "{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv")
TENORS = {"1 Mo": 1/12, "1.5 Month": 1.5/12, "2 Mo": 2/12, "3 Mo": 0.25, "4 Mo": 4/12, "6 Mo": 0.5, "1 Yr": 1,
          "2 Yr": 2, "3 Yr": 3, "5 Yr": 5, "7 Yr": 7, "10 Yr": 10, "20 Yr": 20, "30 Yr": 30}


def parse_curve(text: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text))
    if "Date" not in frame.columns:
        raise SourceError("US Treasury: unexpected answer (no Date column)")
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
    columns = [c for c in frame.columns if c in TENORS]
    curve = frame.set_index("Date")[columns].rename(columns=TENORS)
    curve.index.name, curve.columns.name = "date", "tenor (years)"
    return curve.sort_index()


def us_yield_curve(years) -> pd.DataFrame:
    return pd.concat([parse_curve(get_text(URL.format(year=year))) for year in sorted(set(years))]).sort_index()
