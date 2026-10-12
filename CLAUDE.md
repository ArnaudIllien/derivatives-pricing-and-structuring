# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this folder is

Quant finance experiments. There is no `requirements.txt`.

- `pricer/` is the option-pricing package (all pricers live here). Details are in the next section. **Everything in it is in English**: names, comments and UI text.
- `quant_trading_backtester.ipynb` backtests a long/short momentum strategy on data downloaded from Yahoo Finance via `yfinance`. Its comments, variable names and UI text are in French.

## Pricer package (`pricer/`)

- Modules sit directly in `pricer/`, tests in `pricer/test/` (unittest), and archived code in `pricer/z_old/`. `z_old/Pricing_option_method_BlackScholes.ipynb` is the original French single-cell European notebook, kept as is; its formula now lives in `european.py`.
- Every Python file starts with a header docstring (purpose, how to run, what it outputs). Keep that convention for new files.
- Shared building blocks:
  - `contract.py` holds `Option` and its payoff. The contract has no exercise style; the pricer decides that.
  - `market.py` holds `Market` (constant `r`, `q`, `sigma`) and `Dividend`, for discrete cash dividends.
  - `model.py` simulates the GBM with the exact scheme.
  - `european.py` is the Black-Scholes-Merton pricer. It keeps the old `call_option`/`put_option` date interface.
- Pricers:
  - `baw.py` is the Barone-Adesi–Whaley approximation, with the boundary found by Brent.
  - `lsm.py` is Longstaff-Schwartz; it uses `regression.py`.
  - `binomial.py` holds the CRR trees: European, American, and Bermudan on the LSM grid. They are both a client model and the independent reference for tests. Don't reuse BAW or LSM logic in them.
- Discrete-dividend convention: the spot drops by D right after the dividend date, and exercise on that date is cum-dividend.
  - The tree (Vellekoop-Nieuwenhuis interpolation) and LSM (dividend dates added to the simulation and exercise grid) are exact.
  - `european.py` and BAW use the Bos-Vandermark escrowed approximation. BAW calls also use Black's pseudo-American rule.
- `greeks.py` computes Greeks by finite differences for any `price_fn(option, market)`. Theta shifts the dividend dates with `Market.shifted`.
- `client.py` is the interactive terminal client (`python -m pricer.client`). `price_request` holds the logic, separate from the input/output functions, so tests can drive it with scripted answers.
- Run the tests from the repo root: `python -m unittest discover -s pricer/test -t .` (about 20 s). Run the convergence tables with `python -m pricer.diagnostics` (about 1 min).
- Validation conventions:
  - BAW is an approximation, so its tests use tolerances against the tree:
    - 0.5 % for short maturities;
    - 3 % for long ones;
    - 1.5 % for small discrete dividends. A large late dividend can reach 4–7 %, and that is documented, not tested.
  - Out-of-sample LSM is biased low, so its tests check `price < reference + 3 s.e.` and a bounded bias. Never test LSM equality within the standard error alone.

## Environment

- Notebook kernels were saved with **Python 3.14** (`py -3.14`, the `py` default), but that install is currently broken ("fichier introuvable"). Bare `python` on PATH is **Python 3.12** (numpy, scipy, matplotlib; no pytest). Use it for `pricer/`.
- Libraries used: `pricer/` needs only numpy and scipy (the client uses the standard library for input). The backtester uses pandas, numpy, matplotlib, yfinance and streamlit. `yfinance` needs network access.
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
