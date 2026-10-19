"""
alphavantage.py - News with sentiment scores from Alpha Vantage (free key: ALPHAVANTAGE_API_KEY).

`news_sentiment(topics)` returns the latest articles of the topics with
Alpha Vantage's overall sentiment label (Bearish ... Bullish). Topics include
economy_macro, economy_monetary, economy_fiscal, financial_markets, finance,
energy_transportation, technology. Free plan: 25 requests per day, so the
dashboard caches the answer for hours.
"""
from datetime import datetime, timezone

from ..config import api_key
from ..net import SourceError, get_json
from .news import NewsItem

URL = "https://www.alphavantage.co/query"


def parse_feed(data: dict) -> list[tuple[NewsItem, str, float]]:
    """(item, sentiment label, sentiment score) for each article."""
    if "feed" not in data:
        message = data.get("Information") or data.get("Note") or data.get("Error Message") or str(data)[:80]
        raise SourceError(f"Alpha Vantage: {message}")
    rows = []
    for article in data["feed"]:
        try:
            published = datetime.strptime(article["time_published"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            published = None
        item = NewsItem(article.get("title", "").strip(), article.get("url", ""), published,
                        article.get("source", "Alpha Vantage"))
        rows.append((item, article.get("overall_sentiment_label", ""),
                     float(article.get("overall_sentiment_score", 0.0))))
    return rows


def news_sentiment(topics: str = "economy_macro,financial_markets", limit: int = 50):
    key = api_key("ALPHAVANTAGE_API_KEY")
    if not key:
        raise SourceError("Alpha Vantage: no ALPHAVANTAGE_API_KEY (see the 'API keys' page)")
    return parse_feed(get_json(URL, {"function": "NEWS_SENTIMENT", "topics": topics, "limit": limit,
                                     "sort": "LATEST", "apikey": key}))
