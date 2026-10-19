"""
sources - One module per data provider.

Each module separates the network call (`fetch_*`, through market_watch.net)
from the parsing (`parse_*`, pure functions on the raw text or JSON), so the
parsers are tested offline on recorded answers.

Keyless:  yahoo, ecb, treasury, fred, frankfurter, energy_charts, news (RSS
          feeds and Google News), gdelt (best effort: heavily rate-limited).
With key: finnhub, alphavantage, entsoe, eia (see market_watch/config.py).
"""
