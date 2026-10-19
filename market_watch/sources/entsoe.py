"""
entsoe.py - Day-ahead electricity prices of every European bidding zone from the
ENTSO-E Transparency Platform (free key: ENTSOE_API_KEY).

The API answers an XML "Publication_MarketDocument" (document type A44): one or
more TimeSeries, each with Periods (start, resolution PT15M or PT60M) holding
Points (position, price.amount). A missing position repeats the previous price
(the A03 curve type). Prices are in EUR/MWh, times in UTC, converted to
Europe/Paris.
"""
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import pandas as pd

from ..config import api_key
from ..net import SourceError, get_bytes

URL = "https://web-api.tp.entsoe.eu/api"
# EIC codes of the bidding zones
ZONES = {"France": "10YFR-RTE------C", "Germany-Luxembourg": "10Y1001A1001A82H", "Spain": "10YES-REE------0",
         "Italy North": "10Y1001A1001A73I", "Netherlands": "10YNL----------L", "Belgium": "10YBE----------2",
         "Austria": "10YAT-APG------L", "Poland": "10YPL-AREA-----S", "Switzerland": "10YCH-SWISSGRIDZ",
         "Portugal": "10YPT-REN------W", "Denmark West": "10YDK-1--------W", "Norway South-West": "10YNO-2--------T",
         "Sweden Stockholm": "10Y1001A1001A46L", "Finland": "10YFI-1--------U", "Greece": "10YGR-HTSO-----Y",
         "Hungary": "10YHU-MAVIR----U", "Romania": "10YRO-TEL------P", "Czechia": "10YCZ-CEPS-----N"}
RESOLUTIONS = {"PT15M": timedelta(minutes=15), "PT30M": timedelta(minutes=30), "PT60M": timedelta(hours=1)}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(node, name):
    return next((c for c in node if _local(c.tag) == name), None)


def parse_prices(content: bytes) -> pd.Series:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise SourceError(f"ENTSO-E: not XML ({error})") from error
    if _local(root.tag) == "Acknowledgement_MarketDocument":
        reason = next((n.text for n in root.iter() if _local(n.tag) == "text"), "no data")
        raise SourceError(f"ENTSO-E: {reason}")
    prices = {}
    for period in (n for n in root.iter() if _local(n.tag) == "Period"):
        interval = _child(period, "timeInterval")
        start = pd.Timestamp(_child(interval, "start").text)
        end = pd.Timestamp(_child(interval, "end").text)
        step = RESOLUTIONS[_child(period, "resolution").text]
        points = {int(_child(p, "position").text): float(_child(p, "price.amount").text)
                  for p in period if _local(p.tag) == "Point"}
        n_steps = int((end - start)/step)
        last = None
        for position in range(1, n_steps + 1):
            last = points.get(position, last)
            if last is not None:
                prices[start + (position - 1)*step] = last
    if not prices:
        raise SourceError("ENTSO-E: no price in the answer")
    series = pd.Series(prices, name="EUR / MWh").sort_index()
    series.index = pd.DatetimeIndex(series.index).tz_convert("Europe/Paris")
    return series


def day_ahead_prices(zone_code: str, start: date, end: date) -> pd.Series:
    key = api_key("ENTSOE_API_KEY")
    if not key:
        raise SourceError("ENTSO-E: no ENTSOE_API_KEY (see the 'API keys' page)")
    params = {"securityToken": key, "documentType": "A44", "in_Domain": zone_code, "out_Domain": zone_code,
              "periodStart": start.strftime("%Y%m%d0000"), "periodEnd": end.strftime("%Y%m%d0000")}
    return parse_prices(get_bytes(URL, params))
