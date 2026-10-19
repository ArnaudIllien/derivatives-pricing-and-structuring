"""
gdelt.py - World events from the GDELT DOC 2.0 API (no key, best effort).

GDELT monitors news worldwide in 65 languages. Its free API allows one request
every 5 seconds per IP (net.py waits accordingly) and often answers with an
empty result or HTTP 429 at busy times: the dashboard treats it as an extra
source, never as the only one.
"""
from datetime import datetime, timezone

from ..net import SourceError, get_json
from .news import NewsItem

URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def parse_articles(data: dict) -> list[NewsItem]:
    if not isinstance(data, dict):
        raise SourceError("GDELT: unexpected answer")
    items = []
    for article in data.get("articles", []):
        try:
            published = datetime.strptime(article.get("seendate", ""), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            published = None
        items.append(NewsItem(article.get("title", "").strip(), article.get("url", ""), published,
                              article.get("domain", "GDELT")))
    return [item for item in items if item.title]


def articles(query: str, timespan: str = "24h", max_records: int = 25) -> list[NewsItem]:
    data = get_json(URL, {"query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
                          "maxrecords": max_records, "timespan": timespan, "sort": "hybridrel"}, retries=1)
    return parse_articles(data)
