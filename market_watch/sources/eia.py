"""
eia.py - US electricity retail prices by state from the EIA API v2 (free key: EIA_API_KEY).

`retail_prices(states, start)` returns the monthly average retail price, in
cents per kWh, all sectors, one column per state ("US" is the national average).
"""
import pandas as pd

from ..config import api_key
from ..net import SourceError, get_json

URL = "https://api.eia.gov/v2/electricity/retail-sales/data/"


def parse_retail(data: dict) -> pd.DataFrame:
    rows = data.get("response", {}).get("data")
    if rows is None:
        raise SourceError(f"EIA: unexpected answer {str(data)[:80]}")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["period"] = pd.to_datetime(frame["period"])
    return frame.pivot_table(index="period", columns="stateid", values="price").sort_index()


def retail_prices(states=("US", "CA", "TX", "NY", "FL"), start: str = "2023-01") -> pd.DataFrame:
    key = api_key("EIA_API_KEY")
    if not key:
        raise SourceError("EIA: no EIA_API_KEY (see the 'API keys' page)")
    params = [("api_key", key), ("frequency", "monthly"), ("data[0]", "price"), ("facets[sectorid][]", "ALL"),
              ("start", start), ("sort[0][column]", "period"), ("sort[0][direction]", "desc"), ("length", 5000)]
    params += [("facets[stateid][]", state) for state in states]
    return parse_retail(get_json(URL, params))
