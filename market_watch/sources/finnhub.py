"""
finnhub.py - Market and company news from Finnhub (free key: FINNHUB_API_KEY).

- `market_news(category)`: "general", "forex", "crypto" or "merger".
- `company_news(symbol, start, end)`: news of one US company (used for the big
  moves of single stocks in the watchlist).
Free plan: 60 requests per minute. The economic calendar is not in the free plan.
"""
from datetime import date, datetime, timezone

from ..config import api_key
from ..net import SourceError, get_json
from .news import NewsItem

BASE = "https://finnhub.io/api/v1"


def _key() -> str:
    key = api_key("FINNHUB_API_KEY")
    if not key:
        raise SourceError("Finnhub: no FINNHUB_API_KEY (see the 'API keys' page)")
    return key


def parse_news(data) -> list[NewsItem]:
    if not isinstance(data, list):
        raise SourceError(f"Finnhub: unexpected answer {str(data)[:80]}")
    return [NewsItem(item.get("headline", "").strip(), item.get("url", ""),
                     datetime.fromtimestamp(item["datetime"], timezone.utc) if item.get("datetime") else None,
                     item.get("source", "Finnhub")) for item in data if item.get("headline")]


def market_news(category: str = "general") -> list[NewsItem]:
    return parse_news(get_json(f"{BASE}/news", {"category": category, "token": _key()}))


def company_news(symbol: str, start: date, end: date) -> list[NewsItem]:
    return parse_news(get_json(f"{BASE}/company-news", {"symbol": symbol, "from": start.isoformat(),
                                                        "to": end.isoformat(), "token": _key()}))
