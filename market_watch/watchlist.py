"""
watchlist.py - Personal watchlist and threshold alerts, saved as JSON.

An alert watches one ticker with one rule:
- "above" / "below": the last close crosses a level;
- "daily move": the absolute daily % change is at least the level.
`evaluate` checks every alert against the latest closes. The file is
market_watch/watchlist.json by default (ignored by git: it is personal).
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_FILE = Path(__file__).with_name("watchlist.json")
RULES = ("above", "below", "daily move")


@dataclass(frozen=True)
class Alert:
    ticker: str
    rule: str
    level: float

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"Unknown rule {self.rule!r} (choose among {RULES}).")


@dataclass
class Watchlist:
    tickers: list[str] = field(default_factory=lambda: ["^GSPC", "^STOXX50E", "EURUSD=X", "GC=F", "CL=F", "SPY"])
    alerts: list[Alert] = field(default_factory=list)


def load(path: Path = DEFAULT_FILE) -> Watchlist:
    if not path.exists():
        return Watchlist()
    data = json.loads(path.read_text(encoding="utf-8"))
    return Watchlist(list(data.get("tickers", [])), [Alert(**a) for a in data.get("alerts", [])])


def save(watchlist: Watchlist, path: Path = DEFAULT_FILE):
    data = {"tickers": watchlist.tickers, "alerts": [asdict(a) for a in watchlist.alerts]}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class AlertStatus:
    alert: Alert
    triggered: bool
    value: float          # last close, or last daily move in %
    message: str


def evaluate(alerts: list[Alert], closes: pd.DataFrame) -> list[AlertStatus]:
    statuses = []
    for alert in alerts:
        series = closes[alert.ticker].dropna() if alert.ticker in closes else pd.Series(dtype=float)
        if len(series) < 2:
            statuses.append(AlertStatus(alert, False, float("nan"), "no data"))
            continue
        last, previous = float(series.iloc[-1]), float(series.iloc[-2])
        if alert.rule == "daily move":
            value = 100*(last/previous - 1)
            triggered = abs(value) >= alert.level
            message = f"moved {value:+.2f}% (alert at ±{alert.level:g}%)"
        else:
            value = last
            triggered = last > alert.level if alert.rule == "above" else last < alert.level
            message = f"at {last:,.4g} ({alert.rule} {alert.level:,.4g})"
        statuses.append(AlertStatus(alert, triggered, value, message))
    return statuses
