"""
market_watch - Interactive daily market dashboard built on free data sources.

HOW TO RUN
    From the repository root:
        python -m streamlit run market_watch/app.py
    The dashboard opens in the browser. Data is cached (5 min to 6 h depending
    on the source), so the first load of each page is the slowest.

    Tests (offline, recorded responses):
        python -m unittest discover -s market_watch/test -t .

WHAT IT SHOWS
    Overview, rates (US and euro curves today, 1 month and 1 year ago), FX,
    equity indices (Europe, Americas, Asia-Pacific, Africa), commodities,
    electricity day-ahead prices, volatility indices and correlations, macro
    indicators, events and news (macro, politics, markets, big moves with
    headlines), a watchlist with threshold alerts, an option pricer fed with
    today's data (pricer/ package) and a delta-hedging backtest.

LAYOUT
    config.py      API keys (environment variables or market_watch/.env)
    net.py         the only module that talks to the network (polite, retries)
    universe.py    the instruments shown, by category and region
    sources/       one module per data provider: fetch (network) + parse (pure)
    analytics.py   performance, big moves, correlations, curves (pure pandas)
    watchlist.py   watchlist and alerts, saved in market_watch/watchlist.json
    app.py         the Streamlit pages
"""
