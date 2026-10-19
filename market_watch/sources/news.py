"""
news.py - Headlines from RSS feeds and from Google News search (no key).

- `FEEDS`: official and media feeds by theme (central banks, markets, economy,
  world politics). `feed_items(url)` reads one.
- `google_news(query, days)`: Google News search results as RSS, used for the
  event themes and to explain the big moves of the day ("why did X move?").

Every item is a `NewsItem` with a timezone-aware UTC publication time.
Headlines only: follow the links for the articles.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

from ..net import SourceError, get_bytes

GOOGLE_NEWS = "https://news.google.com/rss/search?q={query}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
FEEDS = {
    "Central banks": {"Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
                      "ECB": "https://www.ecb.europa.eu/rss/press.html",
                      "Bank of England": "https://www.bankofengland.co.uk/rss/news"},
    "Markets and economy": {"CNBC Economy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
                            "CNBC Finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
                            "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml"},
    "World and politics": {"BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
                           "BBC Africa": "https://feeds.bbci.co.uk/news/world/africa/rss.xml"},
}
# Google News queries of the event themes
THEMES = {
    "Macroeconomics": '(inflation OR GDP OR unemployment OR "central bank" OR "interest rates" OR recession OR PMI)',
    "Politics and geopolitics": "(election OR government OR sanctions OR tariffs OR geopolitics OR war OR summit)",
    "Financial markets": '(stocks OR bonds OR "stock market" OR selloff OR rally OR "Wall Street" OR yields)',
    "Africa": '(JSE OR "Egyptian Exchange" OR "Nigerian Exchange" OR "Casablanca Stock Exchange" OR rand OR naira)',
}


@dataclass(frozen=True)
class NewsItem:
    title: str
    link: str
    published: datetime | None
    source: str


def _parse_date(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        moment = parsedate_to_datetime(text.strip())
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def parse_rss(content: bytes, default_source: str = "") -> list[NewsItem]:
    """RSS 2.0 items, newest first. Google News puts the publisher in <source>."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise SourceError(f"{default_source or 'RSS'}: not a valid feed ({error})") from error
    items = []
    for item in root.iter("item"):
        title = " ".join((item.findtext("title") or "").split())
        if not title:
            continue
        source = (item.findtext("source") or default_source).strip()
        items.append(NewsItem(title, (item.findtext("link") or "").strip(), _parse_date(item.findtext("pubDate")),
                              source))
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(items, key=lambda i: i.published or epoch, reverse=True)


def feed_items(url: str, source: str = "") -> list[NewsItem]:
    return parse_rss(get_bytes(url), source)


def google_news(query: str, days: int = 1, limit: int = 20) -> list[NewsItem]:
    url = GOOGLE_NEWS.format(query=quote_plus(query), days=days)
    return parse_rss(get_bytes(url), "Google News")[:limit]
