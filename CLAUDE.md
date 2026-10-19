# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this folder is

Quant finance experiments. There is no `requirements.txt`.

- `pricer/` is the option-pricing package (all pricers live here). Details are in the next section. **Everything in it is in English**: names, comments and UI text.
- `market_watch/` is the Streamlit market dashboard built on free APIs, also **all in English**. See its section below.
- `structuring/` is the Structured Products Intelligence Platform, also **all in English**. It turns an investor profile into several structured solutions, then values and compares them. See its section below.
- `.gitignore` excludes `__pycache__/`, `market_watch/.env` (API keys: never commit it) and `market_watch/watchlist.json`.
- `quant_trading_backtester.ipynb` backtests a long/short momentum strategy on data downloaded from Yahoo Finance via `yfinance`. Its comments, variable names and UI text are in French.

## Pricer package (`pricer/`)

- Modules sit directly in `pricer/`, tests in `pricer/test/` (unittest), and archived code in `pricer/z_old/`. `z_old/Pricing_option_method_BlackScholes.ipynb` is the original French single-cell European notebook, kept as is; its formula now lives in `european.py`.
- Every Python file starts with a header docstring (purpose, how to run, what it outputs). Keep that convention for new files.
- Shared building blocks:
  - `contract.py` holds `Option` and `AsianOption` (fixed strike, arithmetic average over `observation_times`, reusing `Option` for checks and payoff). A contract has no exercise style; the pricer decides that, and Bermudan exercise dates are a pricer argument. Both contracts have `shifted(dt)` for Theta.
  - `market.py` holds `Market` (constant `r`, `q`, `sigma`) and `Dividend`, for discrete cash dividends.
  - `model.py` simulates the GBM with the exact scheme. `time_grid(T, dates)` merges any dates with the dividend dates, and `nearest_columns` maps dates to grid columns.
  - `monte_carlo.py` holds the shared Monte Carlo statistics: antithetic pairing, mean and standard error, control variate. LSM and Asian both use it; don't re-implement them.
  - `european.py` is the Black-Scholes-Merton pricer. It keeps the old `call_option`/`put_option` date interface.
- Pricers:
  - `baw.py` is the Barone-Adesi–Whaley approximation, with the boundary found by Brent.
  - `lsm.py` is Longstaff-Schwartz; it uses `regression.py`. One core `_lsm` runs on a grid plus a mask of exercisable columns: `lsm_price` (American, regular grid plus dividend dates) and `bermudan_lsm_price` (given dates plus T; t = 0 only if listed; dividend dates simulated but not exercisable).
  - `asian.py` is the arithmetic Asian Monte Carlo, with the geometric Asian closed form (through `bsm_price`) as control variate. The control is switched off with discrete dividends. `asian_convergence` gives the price for increasing N.
  - `binomial.py` holds the CRR trees: European, American, Bermudan on the LSM grid, and Bermudan on any dates (`bermudan_tree_price`, dates moved to the nearest step). The exercise style is a per-step rule. They are both a client model and the independent reference for tests. Don't reuse BAW or LSM logic in them.
- Discrete-dividend convention: the spot drops by D right after the dividend date, and exercise or observation on that date is cum-dividend.
  - The tree (Vellekoop-Nieuwenhuis interpolation), LSM and Asian (dividend dates added to the simulation grid) are exact.
  - `european.py` and BAW use the Bos-Vandermark escrowed approximation. BAW calls also use Black's pseudo-American rule.
- `greeks.py` computes Greeks by finite differences for any `price_fn(option, market)`. Theta uses `option.shifted` and `Market.shifted`, so dividend and observation dates move closer. The client shifts the Bermudan exercise dates itself.
- Market data and analysis:
  - `market_data.py` is the only module that imports `yfinance`, and only when called, so the pricers still need only numpy and scipy. `YahooClient` is the network boundary; tests replace it with `test/fake_yahoo.py`. `fetch_snapshot` gives spot, rate (USD: ^IRX discount yield converted to continuous; EUR: €STR from the ECB; other currencies fall back to USD and say so), dividend yield (Yahoo `trailingAnnualDividendYield`, a fraction; `dividendYield` is in %, don't use it) and historical volatility.
  - `implied_vol.py` inverts any pricer with Brent. Smiles keep out-of-the-money mids with a bounded spread. `pick_expiries` picks expiries near target maturities, because SPY lists daily expiries.
  - `hedging.py` is the delta-hedging backtest: short a European option at the hedge volatility, BSM deltas from `bsm_price_and_delta` (vectorised, in `european.py`), on real or simulated paths. `gamma_pnl` explains the result.
- `client.py` is the interactive terminal client (`python -m pricer.client`). `price_request` holds the logic, separate from the input/output functions, so tests can drive it with scripted answers. Its "Market data?" question can load a Yahoo snapshot; tests pass a fake `fetch_snapshot`.
- Run the tests from the repo root: `python -m unittest discover -s pricer/test -t .` (about 25 s, offline). Run the convergence tables with `python -m pricer.diagnostics` (about 1 min) and the hedging demo with `python -m pricer.hedging` (network).
- Validation conventions:
  - BAW is an approximation, so its tests use tolerances against the tree:
    - 0.5 % for short maturities;
    - 3 % for long ones;
    - 1.5 % for small discrete dividends. A large late dividend can reach 4–7 %, and that is documented, not tested.
  - Out-of-sample LSM is biased low, so its tests check `price < reference + 3 s.e.` and a bounded bias. Never test LSM equality within the standard error alone. Bermudan LSM is checked against `bermudan_tree_price` the same way.
  - Asian has exact references: one observation at T gives the BSM price, zero volatility gives a closed form, and the geometric closed form is checked against an independent simulation. Plain Monte Carlo estimates are tested within 3 s.e.

## Market dashboard (`market_watch/`)

- Run it with `python -m streamlit run market_watch/app.py`. Run its tests with `python -m unittest discover -s market_watch/test -t .` (offline, under 1 s).
- Layout:
  - `net.py` is the only module that touches the network. It sets a User-Agent, retries on 429/5xx and spaces calls per host (GDELT 5.5 s, Energy-Charts 2 s). Every failure becomes `SourceError`. Don't name a module `http.py` here: `streamlit run` puts `market_watch/` on `sys.path`, and that would shadow the standard library module that `requests` needs.
  - `sources/` has one module per provider, with `fetch` (network) separate from `parse` (pure). Parsers are tested on recorded answers.
  - `analytics.py` is pure pandas. `watchlist.py` holds the JSON watchlist and alerts. `universe.py` lists all 73 Yahoo tickers (checked live).
  - `app.py` is the Streamlit pages only. It caches every source with `st.cache_data`, wraps calls in `guarded` so a dead source shows a warning, and reuses `pricer.client.price_request`, `implied_vol` and `hedging`.
- Free sources:
  - Keyless: Yahoo, ECB Data Portal, US Treasury CSV, FRED `fredgraph.csv`, Frankfurter, Energy-Charts, Google News RSS, and the Fed, ECB, BoE, CNBC and BBC feeds.
  - GDELT is keyless too, but often returns 429 or empty answers. Keep it on demand only.
  - With keys (read from env vars or `market_watch/.env`, see `config.py`): Finnhub, Alpha Vantage, ENTSO-E, EIA. These were written against the documented formats and tested on samples, not live.
  - Not available for free on Yahoo: VSTOXX (realised Euro Stoxx vol is shown instead), and the Nigeria and Morocco indices.
- Big moves: z-score of today's log return against one year of daily returns. Markets closed on the latest date (holidays) are skipped.
- Test the app headlessly with `streamlit.testing.v1.AppTest` from the repo root. In AppTest, set form values and click submit in the same run.

## Structuring platform (`structuring/`)

- `client.py` is the single entry point. `python -m structuring.client` launches the Streamlit app. `--report [--offline]` prints the comparison for a demo profile.
- Run the tests with `python -m unittest discover -s structuring/test -t .` (offline, about 3 s). The methods are documented in `structuring/METHODOLOGY.md`, which the app shows.
- Module names must not shadow the standard library (`streamlit run` puts `structuring/` on `sys.path`). That is why the profile module is `investor.py`, not `profile.py`.
- The engine is separated from the interface:
  - `investor.py` (profile), `market.py` (data with provenance), `products.py` (payoffs) and `simulation.py`;
  - `valuation.py`, `scenarios.py` and `risk.py`;
  - `strategies.py` (candidates) and `suitability.py` (checks);
  - `analysis.py` (runs everything for one candidate) and `comparison.py` (the table);
  - `stress.py` (conviction stress cases), `benchmarks.py` (simple alternatives and the value-added verdict), `lab.py` (editable terms) and `components.py` (option blocks for the professional view, model limits);
  - `disclosures.py` (texts) and `app.py` (UI only: 8 pages plus a sidebar "Professional structurer mode" toggle).
- Views: `InvestorProfile.view_on(underlying)` always returns a `ScenarioView`, a set of weighted `ScenarioCase`s whose probabilities sum to 1. It falls back to the simple `Conviction`. The engine (strategies, distributions, stress) reads only ScenarioViews. Cases use piecewise drift from `simulation.simulate_regimes`: the median move is reached at the case horizon, then the median is flat.
- Components must add up exactly to `value(...)`. Use closed forms wherever `value` does (terminal structures), and isolate path-dependent blocks by linearity on the same paths (same default settings and seed).
- Core conventions:
  - Products map simulated **performance** paths (P_0 = 1) to `CashFlows` (fraction of notional per payment date). One representation serves valuation, scenarios and risk.
  - `name` is keyword-only on every product: positional arguments after `maturity` are the contract terms.
  - Every market parameter is a `Parameter` tagged OBSERVED / ESTIMATED / HYPOTHETICAL with its date. User overrides become HYPOTHETICAL. Issuer spread (default 100 bp), recovery (40 %), fees (default 1 %/yr) and issue price (100 %) are always hypothetical.
  - Never present a model value as an executable price.
- Models (all reuse `pricer`):
  - Equity, FX (Garman-Kohlhagen, q = foreign rate) and commodities (Black, q = r, flat futures curve) all reduce to a GBM with carry q.
  - Rates are the government yield at maturity (US Treasury, ECB AAA), read through `market_watch.sources`.
  - Volatility is the ATM implied vol when Yahoo lists options (^SPX, US stocks), else historical. Index dividends come from an ETF proxy's `yield` field.
- Valuation:
  - Terminal structures use the BSM closed form (Monte Carlo shown as a check). Barrier and autocallable structures use Monte Carlo.
  - Issuer credit is a constant intensity, spread / (1 − R), on notes only.
  - Coupons and participations are solved exactly, since the value is linear in the term.
  - The risk-neutral measure is for prices. The investor-scenario measure (median move = conviction) is only for distributions and P(loss), always labelled.
- Tests:
  - Payoffs are tested on hand-built paths, with an interior grid date so a dip can be represented.
  - The issuer credit spread makes a 100 % protection cheaper, so "impossible protection" tests need a zero spread.
  - Memory coupons are paid back after a dip, so path dependence is tested on a no-memory autocallable.

## Environment

- Notebook kernels were saved with **Python 3.14** (`py -3.14`, the `py` default), but that install is currently broken ("fichier introuvable"). Bare `python` on PATH is **Python 3.12**. It has numpy, scipy, matplotlib, pandas, requests, yfinance and streamlit (installed with pip), but no pytest. Use it for `pricer/` and `market_watch/`.
- Libraries used: `pricer/` needs only numpy and scipy (yfinance only for `market_data`, imported lazily). `market_watch/` needs pandas, requests, yfinance, streamlit and matplotlib (for the heatmap colours). The backtester uses pandas, numpy, matplotlib, yfinance and streamlit. Network access is needed for live data, not for the tests.
- In PowerShell 5.1, piping text into `python -m pricer.client` corrupts the answers (encoding marker). Use bash `printf ... |` or the scripted-input tests instead.
- Run a notebook headlessly: `py -3.14 -m jupyter nbconvert --to notebook --execute --inplace "<notebook>.ipynb"`.

## Backtester structure

- Cells 0–6 are exploratory: they download EUR/USD and define `strategy`, `sharpe_ratio` and `optimize_window`. Then they split 70/30 into train and test, pick the best momentum window (1–50) by train Sharpe ratio, and report the out-of-sample Sharpe ratio.
- **Cell 7 is a self-contained Streamlit app.** It re-defines the same three functions and adds a ticker picker, date inputs, window and fee sliders, train/test equity curves, and a button that runs the optimisation on the train set only. It only works when copied into a `.py` file and launched with `streamlit run <file>.py`; it doesn't work when the cell runs inside Jupyter.
- Because the three functions are duplicated, a change to the strategy logic must be made in both cell 1/5/6 **and** cell 7.
- Strategy conventions:
  - Returns are logarithmic.
  - The position is `+1` or `-1` depending on the sign of the summed returns over the window.
  - The position is shifted one period before it is applied, to avoid look-ahead.
  - Each change of position costs `fee`.
  - The Sharpe ratio is annualised with `sqrt(252)` and computed on `strat_net`.
- In the app, the strategy runs on the full series and is split afterwards, so the test period doesn't start with an empty rolling window. Keep that ordering.

## Working conventions

- `* - Copie.ipynb` files are manual backups. Edit the base notebook, and don't delete copies without asking.
