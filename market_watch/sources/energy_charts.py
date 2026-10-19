"""
energy_charts.py - European day-ahead electricity prices from Energy-Charts (Fraunhofer ISE, no key).

`day_ahead_prices(zone, start, end)` returns the price in EUR/MWh, indexed by
delivery time (Europe/Paris), at the market resolution (15 min since the
2025 switch of the European day-ahead market, 1 h before). Data licence CC BY 4.0.

The API throttles bursts: net.py spaces the calls, and the dashboard caches them.
"""
from datetime import date

import pandas as pd

from ..net import SourceError, get_json

URL = "https://api.energy-charts.info/price"
ZONES = {"France": "FR", "Germany-Luxembourg": "DE-LU", "Spain": "ES", "Italy North": "IT-North",
         "Netherlands": "NL", "Belgium": "BE", "Austria": "AT", "Poland": "PL", "Switzerland": "CH",
         "Denmark West": "DK1", "Norway South-West": "NO2", "Sweden Stockholm": "SE3"}


def parse_prices(data: dict) -> pd.Series:
    if "unix_seconds" not in data or "price" not in data:
        raise SourceError("Energy-Charts: unexpected answer")
    index = pd.to_datetime(data["unix_seconds"], unit="s", utc=True).tz_convert("Europe/Paris")
    return pd.Series(data["price"], index=index, name=data.get("unit", "EUR / MWh"), dtype=float).dropna()


def day_ahead_prices(zone: str, start: date, end: date) -> pd.Series:
    return parse_prices(get_json(URL, {"bzn": zone, "start": start.isoformat(), "end": end.isoformat()}))


def daily_average(prices: pd.Series) -> pd.Series:
    """Baseload price: average over each delivery day."""
    return prices.groupby(prices.index.date).mean()
