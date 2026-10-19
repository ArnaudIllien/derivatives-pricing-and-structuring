"""
test_analytics.py - Tests of the dashboard statistics, the watchlist alerts, the key configuration and
the polite HTTP layer (all offline).
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from market_watch import analytics, net
from market_watch import watchlist as wl
from market_watch.config import api_key, read_env_file
from market_watch.universe import ALL, BY_TICKER, Instrument

STOCK = Instrument("AAA", "Stock A", "Index", "Test")
RATE = Instrument("^TNX", "US 10y", "Rate", "Test")


def closes_frame():
    dates = pd.bdate_range("2024-12-02", "2026-10-12")
    rng = np.random.default_rng(0)
    stock = 100*np.exp(np.cumsum(0.01*rng.standard_normal(len(dates))))
    stock[-1] = stock[-2]*1.05                      # a +5 % day: 5 standard deviations
    rate = 4 + np.linspace(0, 1, len(dates))
    return pd.DataFrame({"AAA": stock, "^TNX": rate}, index=dates)


class TestAnalytics(unittest.TestCase):
    def setUp(self):
        self.closes = closes_frame()

    def test_performance_table(self):
        table = analytics.performance_table(self.closes, [STOCK, RATE]).set_index("ticker")
        stock = self.closes["AAA"]
        self.assertAlmostEqual(table.loc["AAA", "1D"], 5.0, places=8)
        week_ago = stock.loc[:"2026-10-05"].iloc[-1]
        self.assertAlmostEqual(table.loc["AAA", "1W"], 100*(stock.iloc[-1]/week_ago - 1))
        year_end = stock.loc[:"2025-12-31"].iloc[-1]
        self.assertAlmostEqual(table.loc["AAA", "YTD"], 100*(stock.iloc[-1]/year_end - 1))
        self.assertEqual(table.loc["^TNX", "unit"], "points")
        self.assertAlmostEqual(table.loc["^TNX", "1D"], self.closes["^TNX"].diff().iloc[-1])

    def test_big_moves(self):
        moves = analytics.big_moves(self.closes, [STOCK, RATE], threshold=3)
        self.assertEqual(list(moves["ticker"]), ["AAA"])     # rates are not in z-scores
        self.assertGreater(moves.iloc[0]["z-score"], 4)
        self.assertTrue(analytics.big_moves(self.closes, [STOCK], threshold=10).empty)

    def test_big_moves_skip_closed_markets(self):
        closes = self.closes.copy()
        closes.loc[closes.index[-1], "AAA"] = np.nan          # holiday: the last move is from the day before
        closes["BBB"] = closes["^TNX"]
        moves = analytics.big_moves(closes, [STOCK], threshold=0.0)
        self.assertTrue(moves.empty)

    def test_correlations_and_volatility(self):
        closes = self.closes.copy()
        closes["BBB"] = closes["AAA"]**2
        matrix = analytics.correlation_matrix(closes[["AAA", "BBB"]], 90)
        self.assertAlmostEqual(matrix.loc["AAA", "BBB"], 1.0)
        vol = analytics.realized_volatility(closes["AAA"].iloc[:-1], 250).iloc[-1]
        self.assertAlmostEqual(vol, 100*0.01*np.sqrt(252), delta=2)

    def test_curve_snapshots(self):
        dates = pd.bdate_range("2025-09-01", "2026-10-09")
        curve = pd.DataFrame({1: np.arange(len(dates), dtype=float), 10: np.arange(len(dates)) + 1.0}, index=dates)
        snapshots = analytics.curve_snapshots(curve)
        self.assertEqual(len(snapshots.columns), 3)
        self.assertIn("2026-09-09", snapshots.columns[1])
        self.assertIn("2025-10-09", snapshots.columns[2])

    def test_universe(self):
        self.assertEqual(len(BY_TICKER), len(ALL))   # no duplicate ticker
        self.assertTrue({"Europe", "Americas", "Asia-Pacific", "Africa"} <= {i.region for i in ALL})


class TestWatchlist(unittest.TestCase):
    def test_save_load_and_alerts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"watch.json"
            self.assertEqual(wl.load(path).tickers[0], "^GSPC")   # default list
            watch = wl.Watchlist(["AAA"], [wl.Alert("AAA", "above", 1), wl.Alert("AAA", "below", 1),
                                           wl.Alert("AAA", "daily move", 4), wl.Alert("ZZZ", "above", 1)])
            wl.save(watch, path)
            loaded = wl.load(path)
        self.assertEqual(loaded.alerts, watch.alerts)
        statuses = wl.evaluate(loaded.alerts, closes_frame())
        self.assertEqual([s.triggered for s in statuses], [True, False, True, False])
        self.assertEqual(statuses[3].message, "no data")
        with self.assertRaises(ValueError):
            wl.Alert("AAA", "sideways", 1)


class TestConfigAndNet(unittest.TestCase):
    def test_env_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/".env"
            path.write_text("# my keys\nFINNHUB_API_KEY = 'abc'\n\nBROKEN LINE\nEIA_API_KEY=\n", encoding="utf-8")
            self.assertEqual(read_env_file(path), {"FINNHUB_API_KEY": "abc", "EIA_API_KEY": ""})
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(api_key("FINNHUB_API_KEY", path), "abc")
                self.assertIsNone(api_key("EIA_API_KEY", path))
            with mock.patch.dict(os.environ, {"FINNHUB_API_KEY": "env"}):
                self.assertEqual(api_key("FINNHUB_API_KEY", path), "env")

    def test_retries_on_rate_limit(self):
        responses = [mock.Mock(status_code=429), mock.Mock(status_code=200, text="ok")]
        with mock.patch.object(net.requests, "get", side_effect=responses) as fake, \
                mock.patch.object(net.time, "sleep"):
            self.assertEqual(net.get_text("https://example.org/x", backoff=0), "ok")
        self.assertEqual(fake.call_count, 2)

    def test_errors_become_source_errors(self):
        with mock.patch.object(net.requests, "get", return_value=mock.Mock(status_code=404)):
            with self.assertRaisesRegex(net.SourceError, "HTTP 404"):
                net.get_text("https://example.org/missing")
        with mock.patch.object(net.requests, "get", side_effect=net.requests.ConnectionError()), \
                mock.patch.object(net.time, "sleep"):
            with self.assertRaisesRegex(net.SourceError, "ConnectionError"):
                net.get_text("https://example.org/down", retries=1)


if __name__ == "__main__":
    unittest.main()
