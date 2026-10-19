"""
app.py - Streamlit dashboard: daily market watch, option pricer on live data, delta-hedging backtest.

HOW TO RUN
    From the repository root:
        python -m streamlit run market_watch/app.py
    Pick a page in the sidebar. Every source is cached (Yahoo 15 min, rates and
    macro 6 h, news 30 min, electricity 1 h); "Refresh data" clears the cache.

The pages only lay out data: downloads are in sources/, computations in
analytics.py and in the pricer/ package (pricing, implied volatility, hedging).
"""
import sys
from datetime import date, timedelta
from pathlib import Path

# `streamlit run` puts market_watch/ on the import path, not the repository root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from market_watch import analytics, universe
from market_watch import watchlist as wl
from market_watch.config import KEYS, api_key
from market_watch.net import SourceError
from market_watch.sources import (alphavantage, ecb, eia, energy_charts, entsoe, finnhub, frankfurter, fred, gdelt,
                                  news, treasury, yahoo)
from pricer.client import (AMERICAN, ASIAN, BERMUDAN, BINOMIAL, EUROPEAN, LSM, MODEL_NAMES, MONTE_CARLO, STYLE_NAMES,
                           PricingRequest, available_models, price_request)
from pricer.asian import AsianParameters
from pricer.contract import AsianOption, Option
from pricer.hedging import HedgeParameters, historical_hedge, simulated_hedge
from pricer.implied_vol import atm_volatility, pick_expiries, term_structure, volatility_smile, volatility_surface
from pricer.lsm import LSMParameters
from pricer.market import Market
from pricer.market_data import fetch_snapshot, option_expiries, option_quotes

st.set_page_config(page_title="Market Watch", page_icon="📈", layout="wide")
TODAY = date.today()


#- Cached data access -

@st.cache_data(ttl=900, show_spinner="Downloading prices from Yahoo Finance...")
def closes_for(tickers: tuple[str, ...], period: str = "2y") -> pd.DataFrame:
    return yahoo.download_closes(list(tickers), period)


@st.cache_data(ttl=6*3600, show_spinner="Downloading the US Treasury curve...")
def us_curve() -> pd.DataFrame:
    return treasury.us_yield_curve([TODAY.year - 1, TODAY.year])


@st.cache_data(ttl=6*3600, show_spinner="Downloading the euro curve from the ECB...")
def euro_curve() -> pd.DataFrame:
    return ecb.euro_yield_curve(TODAY - timedelta(days=400))


@st.cache_data(ttl=6*3600, show_spinner=False)
def estr_history() -> pd.Series:
    return ecb.estr(TODAY - timedelta(days=730))


@st.cache_data(ttl=6*3600, show_spinner=False)
def fred_history(series_id: str, years: int = 6) -> pd.Series:
    return fred.fred_series(series_id, TODAY - timedelta(days=365*years))


@st.cache_data(ttl=6*3600, show_spinner=False)
def ecb_fixings() -> pd.DataFrame:
    return frankfurter.reference_rates()


@st.cache_data(ttl=3600, show_spinner=False)
def power_prices(zone: str, days: int) -> pd.Series:
    return energy_charts.day_ahead_prices(zone, TODAY - timedelta(days=days), TODAY + timedelta(days=1))


@st.cache_data(ttl=3600, show_spinner=False)
def entsoe_prices(zone_code: str, days: int) -> pd.Series:
    return entsoe.day_ahead_prices(zone_code, TODAY - timedelta(days=days), TODAY + timedelta(days=1))


@st.cache_data(ttl=6*3600, show_spinner=False)
def eia_prices() -> pd.DataFrame:
    return eia.retail_prices()


@st.cache_data(ttl=1800, show_spinner=False)
def headlines(query: str, days: int = 1, limit: int = 20):
    return news.google_news(query, days, limit)


@st.cache_data(ttl=1800, show_spinner=False)
def feed(url: str, source: str):
    return news.feed_items(url, source)


@st.cache_data(ttl=1800, show_spinner="Asking GDELT (one request every 5 s)...")
def gdelt_articles(query: str):
    return gdelt.articles(query)


@st.cache_data(ttl=1800, show_spinner=False)
def finnhub_news(category: str):
    return finnhub.market_news(category)


@st.cache_data(ttl=1800, show_spinner=False)
def company_news(symbol: str):
    return finnhub.company_news(symbol, TODAY - timedelta(days=7), TODAY)


@st.cache_data(ttl=4*3600, show_spinner=False)
def alpha_news(topics: str):
    return alphavantage.news_sentiment(topics)


@st.cache_data(ttl=900, show_spinner="Loading today's market data...")
def snapshot_for(ticker: str):
    return fetch_snapshot(ticker)


@st.cache_data(ttl=900, show_spinner="Inverting the option chains (BAW takes about 30 s)...")
def surface_for(ticker: str, american: bool):
    snapshot = snapshot_for(ticker)
    return snapshot, volatility_surface(ticker, snapshot=snapshot, american=american)


@st.cache_data(ttl=900, show_spinner=False)
def atm_implied_vol(ticker: str, maturity: float):
    """ATM implied volatility of the listed expiry closest to the maturity."""
    snapshot = snapshot_for(ticker)
    expiries = pick_expiries(option_expiries(ticker), snapshot.as_of, (max(7, round(maturity*365)),))
    if not expiries:
        raise SourceError(f"No listed option on {ticker}.")
    smile = volatility_smile(option_quotes(ticker, expiries[0]), snapshot)
    return atm_volatility(smile), expiries[0]


#- Small layout helpers -

def guarded(function, *args, message: str = "Source unavailable", **kwargs):
    """Calls a data function; shows a warning instead of crashing when the source fails."""
    try:
        return function(*args, **kwargs)
    except (SourceError, ValueError, KeyError) as error:
        st.warning(f"{message}: {error}")
    except Exception as error:   # unexpected format change of a free API
        st.warning(f"{message}: {type(error).__name__}: {error}")
    return None


def open_pricer(ticker: str):
    st.session_state["page"] = "Option pricer"
    st.session_state["pricer_ticker"] = ticker


def pricer_button(ticker: str, key: str):
    st.button(f"Price an option on {universe.name_of(ticker)}", key=key, on_click=open_pricer, args=(ticker,),
              help="Opens the option pricer with today's market data of this underlying")


def heatmap(table: pd.DataFrame, columns=("1D", "1W", "1M", "YTD", "1Y")):
    """Performance table colored green (up) to red (down), each column on a symmetric scale."""
    if table is None or table.empty:
        st.info("No data.")
        return
    shown = table[["name", "group", "last", *columns, "unit"]].set_index("name")
    styler = shown.style.format({"last": "{:,.4g}", **{c: "{:+.2f}" for c in columns}}, na_rep="-")
    for column in columns:
        bound = float(np.nanmax(np.abs(shown[column].to_numpy(dtype=float)))) if shown[column].notna().any() else 1
        styler = styler.background_gradient(cmap="RdYlGn", subset=[column], vmin=-bound or -1, vmax=bound or 1)
    st.dataframe(styler, height=min(38*(len(shown) + 1), 900))


def show_news(items, limit: int = 10):
    if not items:
        st.caption("No headline.")
        return
    for item in items[:limit]:
        when = item.published.strftime("%d %b %H:%M UTC") if item.published else ""
        st.markdown(f"- [{item.title}]({item.link})  \n  <small>{item.source} · {when}</small>", unsafe_allow_html=True)


def rebased(closes: pd.DataFrame) -> pd.DataFrame:
    """Each series rebased to 100 at its first valid close, for comparison on one chart."""
    return 100*closes/closes.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)


def category_page(title: str, instruments, intro: str, pricer: bool = True):
    st.header(title)
    st.caption(intro)
    closes = guarded(closes_for, tuple(universe.tickers(instruments)), message="Yahoo Finance")
    if closes is None:
        return
    table = analytics.performance_table(closes, instruments)
    for group in dict.fromkeys(i.region for i in instruments):
        st.subheader(group)
        heatmap(table[table["group"] == group])
    st.subheader("Chart")
    names = {i.name: i.ticker for i in instruments if i.ticker in closes}
    chosen = st.multiselect("Compare (rebased to 100)", list(names), default=list(names)[:3], key=f"chart_{title}")
    period = st.select_slider("Period", ["1M", "3M", "6M", "1Y", "2Y"], value="1Y", key=f"period_{title}")
    start = closes.index[-1] - pd.DateOffset(months={"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "2Y": 24}[period])
    if chosen:
        data = closes.loc[start:, [names[n] for n in chosen]].rename(columns={v: k for k, v in names.items()})
        st.line_chart(rebased(data.ffill()))
        if pricer:
            pricer_button(names[chosen[0]], key=f"price_{title}")


#- Pages -

def page_overview():
    st.header("Market overview")
    st.caption(f"Latest closes from Yahoo Finance ({TODAY:%A %d %B %Y}). Changes in %, rates and volatility "
               "indices in points.")
    key_tickers = ["^GSPC", "^STOXX50E", "^N225", "^J203.JO", "EURUSD=X", "BZ=F", "GC=F", "^TNX", "^VIX"]
    everything = universe.INDICES + universe.FX + universe.COMMODITIES + universe.VOLATILITY + universe.US_RATES
    closes = guarded(closes_for, tuple(universe.tickers(everything)), message="Yahoo Finance")
    if closes is None:
        return
    table = analytics.performance_table(closes, everything).set_index("ticker")
    for column, ticker in zip(st.columns(len(key_tickers)), key_tickers):
        if ticker in table.index:
            row = table.loc[ticker]
            delta = f"{row['1D']:+.2f}{'' if row['unit'] == 'points' else '%'}"
            column.metric(row["name"], f"{row['last']:,.4g}", delta)

    st.subheader("Big moves today")
    threshold = st.slider("Minimum size of a move, in standard deviations of its daily moves over a year",
                          1.0, 4.0, 2.0, 0.5)
    moves = analytics.big_moves(closes, everything, threshold=threshold)
    if moves.empty:
        st.success(f"No move of {threshold:g} standard deviations or more on the latest session.")
    else:
        st.dataframe(moves.drop(columns=["ticker"]).style.format(
            {"move %": "{:+.2f}", "z-score": "{:+.1f}", "typical daily move %": "{:.2f}"}),
            hide_index=True)
        st.caption("Headlines of the last two days for the largest moves (Google News):")
        for _, move in moves.head(5).iterrows():
            with st.expander(f"{move['name']}: {move['move %']:+.2f}% ({move['z-score']:+.1f} sd)"):
                query = f'"{move["name"]}" ' + ("price" if move["category"] != "Index" else "stocks")
                show_news(guarded(headlines, query, 2, 5, message="Google News") or [], 5)

    st.subheader("Performance heatmap")
    for tab, (label, instruments) in zip(st.tabs(["Indices", "FX", "Commodities", "Volatility", "US rates"]),
                                         [("Indices", universe.INDICES), ("FX", universe.FX),
                                          ("Commodities", universe.COMMODITIES), ("Volatility", universe.VOLATILITY),
                                          ("US rates", universe.US_RATES)]):
        with tab:
            heatmap(analytics.performance_table(closes, instruments))


def curve_section(title: str, curve: pd.DataFrame | None, source: str):
    st.subheader(title)
    if curve is None or curve.empty:
        return
    snapshots = analytics.curve_snapshots(curve)
    left, right = st.columns([3, 2])
    with left:
        st.line_chart(snapshots, x_label="tenor (years)", y_label="yield (%)")
    with right:
        changes = snapshots.copy()
        today = changes.columns[0]
        for column in changes.columns[1:]:
            changes[f"change vs {column.split(' (')[0]} (bp)"] = 100*(changes[today] - changes[column])
        changes.index = [f"{t:g}y" if t >= 1 else f"{round(t*12)}m" for t in changes.index]
        st.dataframe(changes.style.format("{:.2f}", na_rep="-"))
    st.caption(source)


def page_rates():
    st.header("Rates")
    curve_section("US Treasury par curve", guarded(us_curve, message="US Treasury"),
                  "Source: US Treasury daily par yield curve.")
    curve_section("Euro area AAA government curve", guarded(euro_curve, message="ECB"),
                  "Source: ECB Data Portal, spot rates of the AAA euro area curve (Svensson fit).")
    st.subheader("Short-term rates")
    series = {}
    estr_series = guarded(estr_history, message="ECB €STR")
    if estr_series is not None:
        series["€STR"] = estr_series
    for series_id in ("DFF", "SOFR"):
        values = guarded(fred_history, series_id, 2, message=f"FRED {series_id}")
        if values is not None:
            series[fred.MACRO_SERIES[series_id][0]] = values
    if series:
        st.line_chart(pd.DataFrame(series).ffill())
    st.subheader("US yields on Yahoo Finance")
    closes = guarded(closes_for, tuple(universe.tickers(universe.US_RATES)), message="Yahoo Finance")
    if closes is not None:
        heatmap(analytics.performance_table(closes, universe.US_RATES))


def page_fx():
    category_page("Foreign exchange", universe.FX, "Spot rates from Yahoo Finance (price of one unit of the first "
                  "currency in the second).", pricer=False)
    st.subheader("ECB reference rates (official daily fixing)")
    fixings = guarded(ecb_fixings, message="Frankfurter")
    if fixings is not None:
        st.caption(f"Fixing of {fixings.attrs.get('date')}, per 1 EUR.")
        st.dataframe(fixings, hide_index=True, height=400)


def page_electricity():
    st.header("Electricity: day-ahead prices")
    st.caption("Energy-Charts (Fraunhofer ISE, CC BY 4.0), EUR/MWh. The day-ahead auction fixes tomorrow's prices "
               "around 12:45 CET. Baseload = average over the delivery day.")
    zones = st.multiselect("Bidding zones", list(energy_charts.ZONES),
                           default=["France", "Germany-Luxembourg", "Spain", "Italy North"])
    days = st.slider("Days of history", 3, 60, 14)
    daily, latest = {}, {}
    for name in zones:
        prices = guarded(power_prices, energy_charts.ZONES[name], days, message=f"Energy-Charts {name}")
        if prices is not None and len(prices):
            daily[name] = energy_charts.daily_average(prices)
            last_day = prices.index[-1].date()
            latest[name] = prices[prices.index.date == last_day].set_axis(
                prices[prices.index.date == last_day].index.strftime("%H:%M"))
    if daily:
        frame = pd.DataFrame(daily)
        st.subheader("Baseload price per day")
        st.line_chart(frame, y_label="EUR/MWh")
        st.dataframe(frame.tail(7).style.format("{:.1f}"))
        st.subheader(f"Intraday profile of the latest delivery day")
        st.line_chart(pd.DataFrame(latest), y_label="EUR/MWh")

    with st.expander("More European zones (ENTSO-E, needs ENTSOE_API_KEY)"):
        if not api_key("ENTSOE_API_KEY"):
            st.info("Add an ENTSO-E key to see every European bidding zone (see the 'API keys' page).")
        else:
            chosen = st.multiselect("ENTSO-E zones", list(entsoe.ZONES), default=["Portugal", "Finland", "Greece"])
            series = {name: guarded(entsoe_prices, entsoe.ZONES[name], days, message=f"ENTSO-E {name}")
                      for name in chosen}
            series = {k: energy_charts.daily_average(v) for k, v in series.items() if v is not None}
            if series:
                st.line_chart(pd.DataFrame(series), y_label="EUR/MWh")
    with st.expander("US retail electricity prices (EIA, needs EIA_API_KEY)"):
        if not api_key("EIA_API_KEY"):
            st.info("Add an EIA key to see US electricity prices by state (see the 'API keys' page).")
        else:
            prices = guarded(eia_prices, message="EIA")
            if prices is not None and not prices.empty:
                st.line_chart(prices, y_label="cents/kWh")


def page_volatility():
    st.header("Volatility and correlations")
    instruments = universe.VOLATILITY
    closes = guarded(closes_for, tuple(universe.tickers(instruments) + ["^STOXX50E", "^GSPC"]),
                     message="Yahoo Finance")
    if closes is None:
        return
    heatmap(analytics.performance_table(closes, instruments))
    st.subheader("Implied against realised volatility")
    st.caption("VSTOXX is not available for free: the Euro Stoxx 50 line is its 21-day realised volatility.")
    frame = pd.DataFrame({"VIX (implied, S&P 500)": closes["^VIX"],
                          "S&P 500 realised 21d": analytics.realized_volatility(closes["^GSPC"]),
                          "Euro Stoxx 50 realised 21d": analytics.realized_volatility(closes["^STOXX50E"])})
    st.line_chart(frame.loc[closes.index[-1] - pd.DateOffset(years=1):].ffill(), y_label="%")

    st.subheader("Correlation matrix of daily returns")
    choices = {i.name: i.ticker for i in universe.ALL if i.category != "Rate"}
    default = ["S&P 500", "Euro Stoxx 50", "Nikkei 225", "JSE All Share", "EUR/USD", "USD/JPY", "Brent crude",
               "Gold", "Copper", "Bitcoin"]
    chosen = st.multiselect("Assets", list(choices), default=default)
    window = st.select_slider("Window (trading days)", [20, 60, 90, 120, 250], value=90)
    if len(chosen) >= 2:
        data = guarded(closes_for, tuple(choices[n] for n in chosen), message="Yahoo Finance")
        if data is not None:
            matrix = analytics.correlation_matrix(data, window).rename(index=universe.name_of,
                                                                        columns=universe.name_of)
            st.dataframe(matrix.style.format("{:+.2f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1))
            st.caption("Markets close at different times (Asia, then Europe, then the US): same-day correlations "
                       "across time zones are understated.")


def page_macro():
    st.header("Macro indicators (FRED)")
    st.caption("St. Louis Fed FRED data, latest release of each series. CPI, PCE, GDP and industrial production "
               "as year-on-year % change.")
    rows, histories = [], {}
    for series_id, (name, unit, how) in fred.MACRO_SERIES.items():
        values = guarded(fred_history, series_id, message=f"FRED {series_id}")
        if values is None or values.empty:
            continue
        shown = fred.transform(values, how)
        histories[name] = shown
        rows.append({"indicator": name, "latest": shown.iloc[-1], "previous": shown.iloc[-2] if len(shown) > 1 else np.nan,
                     "date": shown.index[-1].date(), "unit": "% y/y" if how == "yoy" else
                     ("change, " + unit if how == "change" else unit), "FRED id": series_id})
    if rows:
        st.dataframe(pd.DataFrame(rows).style.format({"latest": "{:,.2f}", "previous": "{:,.2f}"}),
                     hide_index=True)
        chosen = st.selectbox("Chart", list(histories))
        st.line_chart(histories[chosen])


def page_events():
    st.header("Events and news")
    st.caption("Google News search (last 24 h), official feeds of central banks and media RSS feeds. Optional: "
               "GDELT (world events), Finnhub and Alpha Vantage (with keys).")
    themes = list(news.THEMES)
    tabs = st.tabs(themes + ["Central banks", "Media feeds", "More sources"])
    for tab, theme in zip(tabs, themes):
        with tab:
            show_news(guarded(headlines, news.THEMES[theme], 1, 25, message="Google News") or [], 25)
    with tabs[len(themes)]:
        for source, url in news.FEEDS["Central banks"].items():
            st.subheader(source)
            show_news(guarded(feed, url, source, message=source) or [], 6)
    with tabs[len(themes) + 1]:
        for group in ("Markets and economy", "World and politics"):
            for source, url in news.FEEDS[group].items():
                st.subheader(source)
                show_news(guarded(feed, url, source, message=source) or [], 6)
    with tabs[len(themes) + 2]:
        st.subheader("GDELT world events")
        query = st.text_input("GDELT query", '"central bank" OR inflation OR election')
        if st.button("Search GDELT"):
            show_news(guarded(gdelt_articles, query, message="GDELT (often rate-limited: try again later)") or [])
        st.subheader("Finnhub market news")
        if api_key("FINNHUB_API_KEY"):
            category = st.selectbox("Category", ["general", "forex", "crypto", "merger"])
            show_news(guarded(finnhub_news, category, message="Finnhub") or [], 15)
        else:
            st.info("Add a FINNHUB_API_KEY to see Finnhub news (see the 'API keys' page).")
        st.subheader("Alpha Vantage news with sentiment")
        if api_key("ALPHAVANTAGE_API_KEY"):
            rows = guarded(alpha_news, "economy_macro,economy_monetary,financial_markets", message="Alpha Vantage")
            for item, label, score in (rows or [])[:15]:
                st.markdown(f"- **{label}** ({score:+.2f}) [{item.title}]({item.link}) · {item.source}")
        else:
            st.info("Add an ALPHAVANTAGE_API_KEY to see news with sentiment scores (see the 'API keys' page).")


def page_watchlist():
    st.header("Watchlist and alerts")
    watch = wl.load()
    text = st.text_input("Tickers (Yahoo symbols, separated by commas)", ", ".join(watch.tickers))
    tickers = [t.strip().upper() for t in text.split(",") if t.strip()]
    if tickers != watch.tickers:
        watch.tickers = tickers
        wl.save(watch)
    if not tickers:
        st.info("Add tickers to the watchlist.")
        return
    alert_tickers = {a.ticker for a in watch.alerts}
    closes = guarded(closes_for, tuple(sorted(set(tickers) | alert_tickers)), message="Yahoo Finance")
    if closes is None:
        return
    instruments = [universe.BY_TICKER.get(t, universe.Instrument(t, t, "Stock", "Watchlist")) for t in tickers]
    heatmap(analytics.performance_table(closes, instruments))
    missing = [t for t in tickers if t not in closes]
    if missing:
        st.warning(f"No Yahoo data for: {', '.join(missing)}")

    st.subheader("Alerts")
    for status in wl.evaluate(watch.alerts, closes):
        label = f"{universe.name_of(status.alert.ticker)} ({status.alert.ticker}) {status.message}"
        (st.error if status.triggered else st.success)(("TRIGGERED - " if status.triggered else "ok - ") + label)
    with st.form("new_alert", clear_on_submit=True):
        columns = st.columns(3)
        ticker = columns[0].selectbox("Ticker", tickers)
        rule = columns[1].selectbox("Rule", wl.RULES)
        level = columns[2].number_input("Level (price, or % for a daily move)", value=0.0, format="%.4f")
        if st.form_submit_button("Add alert"):
            watch.alerts.append(wl.Alert(ticker, rule, level))
            wl.save(watch)
            st.rerun()
    if watch.alerts:
        remove = st.multiselect("Remove alerts", range(len(watch.alerts)),
                                format_func=lambda i: f"{watch.alerts[i].ticker} {watch.alerts[i].rule} "
                                                      f"{watch.alerts[i].level:g}")
        if remove and st.button("Remove the selected alerts"):
            watch.alerts = [a for i, a in enumerate(watch.alerts) if i not in remove]
            wl.save(watch)
            st.rerun()
    st.subheader("News and option pricer")
    chosen = st.selectbox("Underlying", [t for t in tickers if t in closes])
    if chosen:
        # plain symbols (AAPL, SPY) are single stocks or ETFs: Finnhub company news; indices, FX, futures: Google News
        is_stock = not any(mark in chosen for mark in "^=-.")
        if is_stock and api_key("FINNHUB_API_KEY"):
            st.caption("Company news of the last 7 days (Finnhub)")
            show_news(guarded(company_news, chosen, message="Finnhub") or [], 10)
        else:
            st.caption("Headlines of the last 2 days (Google News)")
            show_news(guarded(headlines, f'"{universe.name_of(chosen)}"', 2, 10, message="Google News") or [], 10)
        pricer_button(chosen, key="price_watchlist")


def page_pricer():
    st.header("Option pricer on today's market data")
    ticker = st.text_input("Yahoo ticker of the underlying", st.session_state.get("pricer_ticker", "SPY")).strip().upper()
    snapshot = guarded(snapshot_for, ticker, message=f"Market data for {ticker}")
    if snapshot is None:
        return
    columns = st.columns(5)
    columns[0].metric("Spot", f"{snapshot.spot:,.4f}", help=f"{snapshot.name}, close of {snapshot.as_of}")
    columns[1].metric("Rate", f"{snapshot.rate:.3%}", help=snapshot.rate_source)
    columns[2].metric("Dividend yield", f"{snapshot.dividend_yield:.2%}")
    columns[3].metric("Historical vol (1y)", f"{snapshot.historical_volatility:.2%}")
    columns[4].metric("Currency", snapshot.currency)
    st.caption(f"Rate source: {snapshot.rate_source}.")

    left, right = st.columns(2)
    with left:
        option_type = st.radio("Option", ["call", "put"], horizontal=True)
        style = st.selectbox("Product", [EUROPEAN, AMERICAN, BERMUDAN, ASIAN], format_func=STYLE_NAMES.get)
        model = st.selectbox("Model", available_models(style), format_func=MODEL_NAMES.get)
        strike = st.number_input("Strike", value=float(round(snapshot.spot)), min_value=0.01, format="%.4f")
        maturity = st.number_input("Maturity (years)", value=0.25, min_value=0.0, step=0.05, format="%.4f")
    with right:
        vol_source = st.radio("Volatility", ["historical", "implied ATM (listed options)", "custom"])
        volatility = snapshot.historical_volatility
        if vol_source.startswith("implied"):
            result = guarded(atm_implied_vol, ticker, maturity, message="Implied volatility")
            if result:
                volatility, expiry = result
                st.caption(f"ATM implied volatility of the {expiry} expiry: {volatility:.2%}")
        elif vol_source == "custom":
            volatility = st.number_input("Volatility", value=round(snapshot.historical_volatility, 4),
                                         min_value=0.0, format="%.4f")
        exercise_times, n_obs, paths = (), 12, 50_000
        if style == BERMUDAN:
            text = st.text_input("Exercise times in years (the maturity is always added)",
                                 " ".join(f"{maturity*m/4:g}" for m in range(1, 4)))
            exercise_times = tuple(sorted(float(t) for t in text.replace(";", " ").split() if t))
        if style == ASIAN:
            n_obs = st.number_input("Observations (regular grid)", value=12, min_value=1, max_value=500)
        if model in (LSM, MONTE_CARLO):
            paths = int(st.number_input("Monte Carlo paths", value=50_000, min_value=1_000, step=10_000))
        steps = int(st.number_input("Tree steps", value=1000, min_value=50, step=100)) if model == BINOMIAL else 1000

    if st.button("Price", type="primary"):
        market = snapshot.market(volatility)
        option = (AsianOption.regular(option_type, strike, maturity, int(n_obs)) if style == ASIAN
                  else Option(option_type, strike, maturity))
        request = PricingRequest(option, style, market, model, tree_steps=steps,
                                 lsm=LSMParameters(n_paths=paths + paths % 2),
                                 asian=AsianParameters(n_paths=paths + paths % 2), exercise_times=exercise_times)
        with st.spinner("Pricing..."):
            report = guarded(price_request, request, message="Pricing")
        if report:
            columns = st.columns(5)
            columns[0].metric("Price", f"{report.price:,.4f}")
            columns[1].metric("Delta", f"{report.greeks.delta:.4f}")
            columns[2].metric("Gamma", f"{report.greeks.gamma:.5f}")
            columns[3].metric("Vega (1 vol pt)", f"{report.greeks.vega/100:.4f}")
            columns[4].metric("Theta (1 day)", f"{report.greeks.theta/365:.4f}")
            if report.details:
                st.dataframe(pd.DataFrame(report.details, columns=["detail", "value"]).style.format({"value": "{:,.4f}"}),
                             hide_index=True)
            for note in report.notes:
                st.caption(f"• {note}")

    st.subheader("Volatility smile and surface")
    american = st.toggle("Invert with Barone-Adesi-Whaley (American options: single stocks, ETFs)", value=False)
    if st.button("Build the surface from the listed options"):
        result = guarded(surface_for, ticker, american, message="Option chains")
        if result:
            _, points = result
            if not points:
                st.info("No usable quote (no listed options, or no two-sided quotes).")
            else:
                frame = pd.DataFrame([{"expiry": str(p.expiry), "strike": p.strike, "log-moneyness ln(K/F)":
                                       p.log_moneyness, "implied vol (%)": 100*p.implied_volatility,
                                       "Yahoo implied vol (%)": 100*p.yahoo_implied_volatility, "type": p.option_type,
                                       "mid": p.price} for p in points])
                frame = frame[frame["log-moneyness ln(K/F)"].abs() < 0.5]
                st.scatter_chart(frame, x="log-moneyness ln(K/F)", y="implied vol (%)", color="expiry")
                rows = term_structure(points)
                st.dataframe(pd.DataFrame(rows, columns=["expiry", "maturity (y)", "ATM implied vol"]).style.format(
                    {"maturity (y)": "{:.3f}", "ATM implied vol": "{:.2%}"}), hide_index=True)
                with st.expander("All quotes"):
                    st.dataframe(frame, hide_index=True)
                st.caption("Out-of-the-money options only, mid prices, spreads under 50 % of the mid. "
                           "Our inversion next to Yahoo's own implied volatility.")


def page_hedging():
    st.header("Delta-hedging backtest")
    st.caption("Sell a European option at its BSM price with the hedge volatility, then hold the BSM delta in the "
               "underlying, rebalanced on real closing prices. The P&L measures how well the hedge works and "
               "whether the option was sold above or below the volatility the market then delivered.")
    columns = st.columns(4)
    ticker = columns[0].text_input("Underlying (Yahoo)", st.session_state.get("pricer_ticker", "SPY")).strip().upper()
    start = columns[1].date_input("Start", TODAY - timedelta(days=365))
    end = columns[2].date_input("Expiry", TODAY - timedelta(days=1))
    option_type = columns[3].radio("Option", ["call", "put"], horizontal=True)
    columns = st.columns(4)
    moneyness = columns[0].number_input("Strike / spot at start", value=1.0, min_value=0.1, step=0.05)
    hedge_vol = columns[1].number_input("Hedge volatility", value=0.18, min_value=0.01, step=0.01, format="%.3f")
    every = columns[2].selectbox("Rebalancing", [1, 5, 21], format_func={1: "daily", 5: "weekly", 21: "monthly"}.get)
    cost_bp = columns[3].number_input("Transaction cost (bp of traded notional)", value=0.0, min_value=0.0, step=1.0)
    if ticker in ("SPY", "^GSPC") and st.checkbox("Use the VIX on the start date as hedge volatility "
                                                  "(the option market's 30-day implied volatility of the S&P 500)"):
        vix = guarded(closes_for, ("^VIX",), "10y", message="VIX")
        if vix is not None:
            past = vix["^VIX"].dropna().loc[:pd.Timestamp(start)]
            if len(past):
                hedge_vol = float(past.iloc[-1])/100
                st.caption(f"VIX on {past.index[-1].date()}: {hedge_vol:.2%}")
    if not st.button("Run the backtest", type="primary"):
        return
    parameters = HedgeParameters(hedge_vol, every, cost_bp/1e4)
    with st.spinner("Downloading prices and hedging..."):
        result = guarded(historical_hedge, ticker, start, end, option_type, moneyness, parameters,
                         message="Backtest")
    if result is None:
        return
    summary = result.summary()
    columns = st.columns(5)
    columns[0].metric("Premium received", f"{summary['premium']:,.2f}")
    columns[1].metric("Final hedge P&L", f"{summary['mean P&L']:,.2f}")
    columns[2].metric("Realised vol", f"{summary['mean realised vol']:.2%}", f"{summary['mean realised vol'] - hedge_vol:+.2%} vs hedge")
    columns[3].metric("Gamma P&L (theory)", f"{summary['mean gamma P&L']:,.2f}")
    columns[4].metric("Transaction costs", f"{summary['mean costs']:,.2f}")
    dates = pd.to_datetime(list(result.dates))
    st.subheader("Hedge P&L")
    st.line_chart(pd.DataFrame({"hedge P&L": result.pnl[0]}, index=dates))
    left, right = st.columns(2)
    left.subheader("Underlying")
    left.line_chart(pd.DataFrame({"spot": result.spots[0], "strike": result.spots[0, 0]*moneyness}, index=dates))
    right.subheader("Delta held")
    right.line_chart(pd.DataFrame({"delta": result.deltas[0]}, index=dates))
    st.caption("Theory: P&L ≈ sum of ½ Γ S² (σ_hedge² dt − realised return²). Selling above the realised volatility "
               "earns money, below loses money; the gap between the P&L and its gamma estimate is the "
               "discrete-rebalancing error.")

    with st.expander("Same option on 2,000 simulated GBM paths (true volatility = realised volatility above)"):
        option = Option(option_type, result.spots[0, 0]*moneyness, float(result.times[-1]))
        market = Market(float(result.spots[0, 0]), result.rate, float(result.realized_volatility[0]))
        simulated = simulated_hedge(option, market, parameters, n_paths=2_000, n_steps=len(result.times) - 1)
        counts, edges = np.histogram(simulated.final_pnl, bins=40)
        st.bar_chart(pd.DataFrame({"paths": counts}, index=np.round((edges[:-1] + edges[1:])/2, 2)),
                     x_label="final hedge P&L", y_label="number of paths")
        st.caption(f"Mean P&L {simulated.final_pnl.mean():+.3f}, std {simulated.final_pnl.std():.3f}: the spread "
                   "of outcomes that luck alone produces with this rebalancing frequency.")


def page_keys():
    st.header("API keys")
    st.markdown("Every page works **without any key**. A free key adds a source. Put your keys in "
                "`market_watch/.env` (one `NAME=value` per line, this file is ignored by git) or in environment "
                "variables, then click *Refresh data* in the sidebar.")
    st.dataframe(pd.DataFrame([{"variable": k.variable, "status": "found" if api_key(k.variable) else "missing",
                                "provider": k.provider, "adds": k.adds, "sign up": k.sign_up} for k in KEYS]),
                 hide_index=True,
                 column_config={"sign up": st.column_config.LinkColumn("sign up")})
    st.code("FINNHUB_API_KEY=your_key\nALPHAVANTAGE_API_KEY=your_key\nENTSOE_API_KEY=your_key\nEIA_API_KEY=your_key",
            language="ini")
    st.subheader("Keyless sources used")
    st.markdown("Yahoo Finance (yfinance) · ECB Data Portal · US Treasury · FRED (CSV) · Frankfurter (ECB fixings) · "
                "Energy-Charts · Google News RSS · Fed, ECB, Bank of England, CNBC and BBC RSS feeds · GDELT")


PAGES = {"Overview": page_overview, "Rates": page_rates, "FX": page_fx,
         "Equity indices": lambda: category_page("Equity indices", universe.INDICES,
                                                 "Europe, Americas, Asia-Pacific and Africa (Johannesburg, Cairo)."),
         "Commodities": lambda: category_page("Commodities", universe.COMMODITIES,
                                              "Front-month futures from Yahoo Finance."),
         "Electricity": page_electricity, "Volatility & correlations": page_volatility, "Macro": page_macro,
         "Events & news": page_events, "Watchlist & alerts": page_watchlist, "Option pricer": page_pricer,
         "Delta hedging": page_hedging, "API keys": page_keys}

with st.sidebar:
    st.title("📈 Market Watch")
    page = st.radio("Page", list(PAGES), key="page")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Free data sources; Yahoo Finance quotes may be delayed. Not investment advice.")

PAGES[page]()
