"""
frankfurter.py - ECB euro reference exchange rates through the Frankfurter API (no key).

The ECB fixes about 30 currencies against the euro once a day, around 16:00 CET.
These official fixings complement the Yahoo intraday quotes.
"""
import pandas as pd

from ..net import SourceError, get_json

URL = "https://api.frankfurter.dev/v1/latest"


def parse_latest(data: dict) -> pd.DataFrame:
    if "rates" not in data:
        raise SourceError("Frankfurter: unexpected answer")
    frame = pd.DataFrame({"currency": list(data["rates"]), f"per 1 {data.get('base', 'EUR')}": list(data["rates"].values())})
    frame.attrs["date"] = data.get("date")
    return frame.sort_values("currency").reset_index(drop=True)


def reference_rates(base: str = "EUR") -> pd.DataFrame:
    return parse_latest(get_json(URL, {"base": base}))
