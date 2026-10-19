"""
test_sources.py - Parsers of every data source, on short recorded answers (no network).

The samples keep the exact structure of the real answers (checked live when
the sources were written). The keyed sources (Finnhub, Alpha Vantage,
ENTSO-E, EIA) are also checked to refuse politely when their key is missing.
"""
import os
import unittest
from datetime import date, datetime, timezone
from unittest import mock

from market_watch.net import SourceError
from market_watch.sources import alphavantage, ecb, eia, energy_charts, entsoe, finnhub, frankfurter, fred, gdelt, news, treasury

ECB_CURVE = """KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE,OBS_STATUS
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,2026-10-08,3.50,A
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,2026-10-09,3.51,A
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3M,B,U2,2026-10-08,2.53,A
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3M,B,U2,2026-10-09,2.531,A
"""
ECB_ESTR = """KEY,FREQ,BENCHMARK_ITEM,TIME_PERIOD,OBS_VALUE
EST.B.EU000A2X2A25.WT,B,EU000A2X2A25,2026-10-08,2.439
EST.B.EU000A2X2A25.WT,B,EU000A2X2A25,2026-10-09,2.437
"""
TREASURY = """Date,"1 Mo","1.5 Month","2 Mo","3 Mo","6 Mo","1 Yr","2 Yr","10 Yr","30 Yr"
10/09/2026,4.13,4.13,4.13,4.25,4.32,4.47,4.80,5.24,5.60
10/08/2026,4.14,4.14,4.13,4.26,4.33,4.48,4.81,5.25,5.61
"""
FRED = "observation_date,DGS10\n2026-10-07,5.20\n2026-10-08,.\n2026-10-09,5.24\n"
RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>  Older   headline </title><link>https://a</link><pubDate>Thu, 08 Oct 2026 13:30:00 +0200</pubDate></item>
<item><title>Newer headline</title><link>https://b</link><pubDate>Fri, 09 Oct 2026 14:00:00 GMT</pubDate>
<source url="https://pub">Publisher</source></item>
<item><title></title><link>https://empty</link></item>
</channel></rss>"""
ENTSOE = b"""<?xml version="1.0" encoding="utf-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
<TimeSeries><Period><timeInterval><start>2026-10-11T22:00Z</start><end>2026-10-12T02:00Z</end></timeInterval>
<resolution>PT60M</resolution>
<Point><position>1</position><price.amount>50.5</price.amount></Point>
<Point><position>2</position><price.amount>40</price.amount></Point>
<Point><position>4</position><price.amount>60</price.amount></Point>
</Period></TimeSeries></Publication_MarketDocument>"""
ENTSOE_ERROR = b"""<?xml version="1.0"?><Acknowledgement_MarketDocument xmlns="urn:x"><Reason><code>999</code>
<text>No matching data found</text></Reason></Acknowledgement_MarketDocument>"""


class TestRates(unittest.TestCase):
    def test_ecb_curve(self):
        curve = ecb.parse_yield_curve(ECB_CURVE)
        self.assertEqual(list(curve.columns), [0.25, 10])
        self.assertAlmostEqual(curve.loc["2026-10-09", 10], 3.51)

    def test_ecb_estr_and_errors(self):
        self.assertAlmostEqual(ecb.parse_estr(ECB_ESTR).iloc[-1], 2.437)
        with self.assertRaises(SourceError):
            ecb.parse_csv("a,b\n1,2\n")

    def test_treasury(self):
        curve = treasury.parse_curve(TREASURY)
        self.assertEqual(curve.index[0], datetime(2026, 10, 8))   # sorted
        self.assertAlmostEqual(curve.loc["2026-10-09", 10], 5.24)
        self.assertAlmostEqual(curve.columns[0], 1/12)

    def test_fred(self):
        series = fred.parse_csv(FRED)
        self.assertEqual(len(series), 2)   # '.' dropped
        self.assertAlmostEqual(series.iloc[-1], 5.24)

    def test_fred_year_on_year(self):
        import pandas as pd
        monthly = pd.Series(range(100, 125), index=pd.date_range("2024-01-01", periods=25, freq="MS"), dtype=float)
        self.assertAlmostEqual(fred.transform(monthly, "yoy").iloc[-1], 100*(124/112 - 1))
        quarterly = pd.Series([100.0, 101, 102, 103, 110], index=pd.date_range("2024-01-01", periods=5, freq="QS"))
        self.assertAlmostEqual(fred.transform(quarterly, "yoy").iloc[-1], 10.0)
        self.assertEqual(list(fred.transform(quarterly, "change")), [1, 1, 1, 7])


class TestMarketsAndPower(unittest.TestCase):
    def test_frankfurter(self):
        frame = frankfurter.parse_latest({"base": "EUR", "date": "2026-10-12", "rates": {"USD": 1.1181, "JPY": 177.3}})
        self.assertEqual(list(frame["currency"]), ["JPY", "USD"])
        self.assertEqual(frame.attrs["date"], "2026-10-12")
        with self.assertRaises(SourceError):
            frankfurter.parse_latest({"message": "not found"})

    def test_energy_charts(self):
        prices = energy_charts.parse_prices({"unix_seconds": [1791669600, 1791673200], "price": [100.0, None],
                                             "unit": "EUR / MWh"})
        self.assertEqual(len(prices), 1)
        self.assertEqual(str(prices.index.tz), "Europe/Paris")
        self.assertEqual(prices.index[0].hour, 0)   # 22:00 UTC is midnight in Paris in summer time
        self.assertEqual(list(energy_charts.daily_average(prices)), [100.0])

    def test_entsoe_fills_missing_positions(self):
        prices = entsoe.parse_prices(ENTSOE)
        self.assertEqual(list(prices), [50.5, 40, 40, 60])   # position 3 repeats position 2
        self.assertEqual(prices.index[0].isoformat(), "2026-10-12T00:00:00+02:00")
        with self.assertRaisesRegex(SourceError, "No matching data"):
            entsoe.parse_prices(ENTSOE_ERROR)

    def test_eia(self):
        frame = eia.parse_retail({"response": {"data": [
            {"period": "2026-07", "stateid": "US", "price": "13.2"}, {"period": "2026-07", "stateid": "CA", "price": "30.1"},
            {"period": "2026-06", "stateid": "US", "price": "12.9"}]}})
        self.assertAlmostEqual(frame.loc["2026-07-01", "CA"], 30.1)
        self.assertEqual(len(frame), 2)


class TestNews(unittest.TestCase):
    def test_rss(self):
        items = news.parse_rss(RSS, "Feed")
        self.assertEqual([i.title for i in items], ["Newer headline", "Older headline"])
        self.assertEqual(items[0].source, "Publisher")
        self.assertEqual(items[1].source, "Feed")
        self.assertEqual(items[1].published, datetime(2026, 10, 8, 11, 30, tzinfo=timezone.utc))
        with self.assertRaises(SourceError):
            news.parse_rss(b"<html><body>", "Broken")

    def test_google_news_url(self):
        with mock.patch.object(news, "get_bytes", return_value=RSS) as fake:
            items = news.google_news('"S&P 500" stocks', days=2, limit=1)
        self.assertEqual(len(items), 1)
        self.assertIn("%22S%26P+500%22+stocks+when:2d", fake.call_args[0][0])

    def test_gdelt(self):
        items = gdelt.parse_articles({"articles": [{"title": "Rates rise", "url": "u", "seendate": "20261012T101500Z",
                                                    "domain": "example.com"}, {"title": ""}]})
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].published, datetime(2026, 10, 12, 10, 15, tzinfo=timezone.utc))
        self.assertEqual(gdelt.parse_articles({}), [])

    def test_finnhub(self):
        items = finnhub.parse_news([{"headline": "Fed holds", "url": "u", "datetime": 1791669600, "source": "Reuters"}])
        self.assertEqual(items[0].source, "Reuters")
        with self.assertRaises(SourceError):
            finnhub.parse_news({"error": "Invalid API key."})

    def test_alphavantage(self):
        rows = alphavantage.parse_feed({"feed": [{"title": "Oil jumps", "url": "u", "time_published": "20261012T093000",
                                                  "source": "X", "overall_sentiment_label": "Bullish",
                                                  "overall_sentiment_score": "0.41"}]})
        self.assertEqual(rows[0][1:], ("Bullish", 0.41))
        with self.assertRaisesRegex(SourceError, "25 requests"):
            alphavantage.parse_feed({"Information": "Our standard API rate limit is 25 requests per day."})


class TestMissingKeys(unittest.TestCase):
    def test_keyed_sources_refuse_without_key(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("market_watch.config.read_env_file",
                                                                     return_value={}):
            for call in (lambda: finnhub.market_news(), lambda: alphavantage.news_sentiment(),
                         lambda: entsoe.day_ahead_prices("10YFR-RTE------C", date(2026, 10, 1), date(2026, 10, 2)),
                         lambda: eia.retail_prices()):
                with self.assertRaisesRegex(SourceError, "API keys"):
                    call()


if __name__ == "__main__":
    unittest.main()
