"""
ecb.py - Euro rates from the ECB Data Portal (no key): yield curve and €STR.

- `euro_yield_curve(start)`: spot rates of the euro area AAA government curve
  (Svensson model fitted daily by the ECB), in %, one column per tenor in years.
- `estr(start)`: euro short-term rate, in %.

API: https://data-api.ecb.europa.eu/service/data/<flow>/<key>?startPeriod=...&format=csvdata
Several series are fetched in one request by joining their codes with '+'.
"""
import io
from datetime import date

import pandas as pd

from ..net import SourceError, get_text

BASE = "https://data-api.ecb.europa.eu/service/data"
CURVE_TENORS = {"3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "15Y": 15,
                "20Y": 20, "30Y": 30}


def parse_csv(text: str) -> pd.DataFrame:
    """ECB 'csvdata' answer -> KEY, TIME_PERIOD (datetime), OBS_VALUE (float)."""
    frame = pd.read_csv(io.StringIO(text))
    missing = {"KEY", "TIME_PERIOD", "OBS_VALUE"} - set(frame.columns)
    if missing:
        raise SourceError(f"ECB: unexpected answer, columns {sorted(missing)} missing")
    frame = frame[["KEY", "TIME_PERIOD", "OBS_VALUE"]].copy()
    frame["TIME_PERIOD"] = pd.to_datetime(frame["TIME_PERIOD"])
    return frame


def parse_yield_curve(text: str) -> pd.DataFrame:
    frame = parse_csv(text)
    frame["tenor"] = frame["KEY"].str.rsplit("SR_", n=1).str[-1].map(CURVE_TENORS)
    curve = frame.pivot_table(index="TIME_PERIOD", columns="tenor", values="OBS_VALUE")
    curve.index.name, curve.columns.name = "date", "tenor (years)"
    return curve.sort_index()


def euro_yield_curve(start: date) -> pd.DataFrame:
    codes = "+".join(f"SR_{tenor}" for tenor in CURVE_TENORS)
    text = get_text(f"{BASE}/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{codes}",
                    {"startPeriod": start.isoformat(), "format": "csvdata"})
    return parse_yield_curve(text)


def parse_estr(text: str) -> pd.Series:
    frame = parse_csv(text)
    return frame.set_index("TIME_PERIOD")["OBS_VALUE"].rename("€STR").sort_index()


def estr(start: date) -> pd.Series:
    return parse_estr(get_text(f"{BASE}/EST/B.EU000A2X2A25.WT", {"startPeriod": start.isoformat(),
                                                                "format": "csvdata"}))
