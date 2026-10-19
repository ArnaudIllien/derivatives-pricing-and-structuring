# Quantitative Finance Toolkit: Option Pricing, Market Data and Structured Products

A personal project in Python covering the main building blocks of an equity derivatives desk:

- **pricing** of options with closed forms, trees and Monte Carlo methods;
- **live market data** from free APIs, including implied volatility surfaces;
- **risk management**, with a delta-hedging backtest;
- a **structuring platform** that turns an investor's views and constraints into several structured products, then values and compares them transparently.

Every model is validated against an independent method, every module is unit-tested (212 tests, all offline), and every number the interfaces show states where it comes from: observed, model estimate or hypothetical.

> **Disclaimer.** This is an educational project. Model values are estimates, not executable prices, and nothing here is investment advice.

---

## Contents

| Folder | What it is | Entry point |
|---|---|---|
| [`pricer/`](pricer) | Option pricing library: European, American, Bermudan and Asian options, Greeks, implied volatility, delta hedging | `python -m pricer.client` |
| [`market_watch/`](market_watch) | Daily market dashboard built on free data sources (rates, FX, equity indices on four continents, commodities, electricity, macro, news) | `python -m streamlit run market_watch/app.py` |
| [`structuring/`](structuring) | Structured Products Intelligence Platform: investor questionnaire → candidate structures → valuation, scenarios, stress tests, comparison | `python -m structuring.client` |
| `quant_trading_backtester.ipynb` | Long/short momentum strategy backtest with train/test split (comments in French) | Jupyter |

---

## 1. Objectives

1. **Implement the standard pricing methods from scratch and prove they are right.** Each method is checked against another one: the closed forms against the trees, the trees against Monte Carlo, the approximations against exact references.
2. **Connect the models to real data** without hiding what is observed and what is assumed.
3. **Reproduce the reasoning of a structurer.** Start from a client need, build several solutions, set their terms so that model value = issue price − costs, then show costs, risks and scenarios. Never rank solutions by headline return alone.

---

## 2. The pricing library (`pricer/`)

### Models and mathematics

**Black-Scholes-Merton (European options).** The underlying follows a geometric Brownian motion under the risk-neutral measure:

$$dS_t = (r-q)\,S_t\,dt + \sigma S_t\,dW_t, \qquad C = S e^{-qT}N(d_1) - K e^{-rT}N(d_2),$$

$$d_{1,2} = \frac{\ln(S/K) + (r - q \pm \sigma^2/2)T}{\sigma\sqrt{T}}.$$

Discrete cash dividends use the Bos-Vandermark escrowed-dividend approximation, which splits each dividend between the spot and the strike.

**Cox-Ross-Rubinstein binomial tree (European, American, Bermudan).**
- The tree uses $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$ and $p = (e^{(r-q)\Delta t} - d)/(u-d)$, with backward induction $V = \max(\text{payoff},\; e^{-r\Delta t}[pV_u + (1-p)V_d])$ on exercise dates.
- Discrete dividends are exact, using the Vellekoop-Nieuwenhuis method: the tree stays recombining, and the value before the drop is interpolated as $V_{cum}(S) = V_{ex}(S - D)$.
- Averaging the trees with $n$ and $n+1$ steps removes the odd/even oscillation.
- The tree is the **independent reference** used to validate the other American pricers.

**Barone-Adesi-Whaley (American, analytic approximation).** The early-exercise premium solves a quadratic ODE. The critical price $S^*$ is found with Brent's method:

$$C_{Am}(S) = C_{Eu}(S) + A_2 \left(\frac{S}{S^*}\right)^{q_2}, \quad S < S^*.$$

**Longstaff-Schwartz least-squares Monte Carlo (American and Bermudan).**
- Paths use the exact GBM scheme.
- Going backwards, the continuation value is regressed on a basis of $S/K$ (polynomial or weighted Laguerre), using the in-the-money paths only.
- The exercise policy is learnt on one set of paths and evaluated on an **independent** set, which gives an estimator biased low in expectation.
- One core algorithm serves both cases: it runs on a time grid with a mask of exercisable dates. The American option is the regular grid; a Bermudan option is any list of dates.

**Arithmetic Asian options (Monte Carlo with a control variate).** The geometric average of a GBM is lognormal:

$$\ln G \sim \mathcal N\!\left(\ln S_0 + (r-q-\tfrac{\sigma^2}{2})\,\bar t,\; \tfrac{\sigma^2}{n^2}\sum_{i,j}\min(t_i,t_j)\right).$$

So the geometric Asian option has a closed form, which serves as a **control variate** for the arithmetic one: $\hat Y_{cv} = \bar Y - \beta(\bar X - \mathbb E[X])$, with $\beta = \mathrm{Cov}(Y,X)/\mathrm{Var}(X)$. Antithetic variates and batching complete the estimator.

**Greeks.** Delta, Gamma, Vega and Theta come from central finite differences that work with any pricer. Theta moves the dividend and observation dates. Monte Carlo pricers use common random numbers, so the noise does not dominate the differences.

**Implied volatility.** Brent's method inverts any pricer: BSM for European options, BAW for American ones. Smiles and surfaces are built from live Yahoo Finance option chains (out-of-the-money mid quotes, spread filter), with moneyness $\ln(K/F)$ and an ATM term structure.

**Delta-hedging backtest.** It sells an option at volatility $\sigma_h$, then holds the BSM delta, rebalanced on real or simulated prices. The P&L is explained by the gamma P&L:

$$\text{P\&L} \approx \sum_k \tfrac12\,\Gamma_k S_k^2\left(\sigma_h^2\,\Delta t - r_k^2\right),$$

where $r_k$ is the realised return over step $k$. The hedge earns money when the option was sold above the realised volatility, and loses money below.

### Validation results (reproducible: `python -m pricer.diagnostics`)

American puts from Longstaff and Schwartz (2001), table 1 (K = 40, r = 6 %), against a 2,000-step tree:

| S | σ | T | Tree | BAW | BAW error | LSM (100k paths) |
|---|---|---|---|---|---|---|
| 36 | 0.2 | 1 | 4.4867 | 4.4596 | −0.60 % | 4.4609 |
| 40 | 0.4 | 2 | 6.9236 | 6.9753 | +0.75 % | 6.8872 |
| 44 | 0.2 | 2 | 1.6934 | 1.7331 | +2.34 % | 1.6719 |

- **BAW** stays within 0.25 % of the tree for at-the-money puts up to one year. On the 12 cases of the table it stays within 2.4 %, with the largest errors on out-of-the-money options. The error grows with the maturity: about 1.5 % at 5 years and 2 % at 10 years. This is the known weakness of the quadratic approximation.
- **LSM** stays within about 1.3 % of the tree and **always below it**. This matches the theory: an out-of-sample policy is sub-optimal, and 50 exercise dates price a Bermudan rather than an American option. The exact Bermudan tree on the same grid isolates the two effects.
- **Asian control variate.** It divides the variance by about 560 for a 1-year at-the-money call with 12 fixings: a standard error of 0.0008 instead of 0.019.
- **Delta hedging on simulated paths** matches the theory. The mean P&L is about 0 when $\sigma_h = \sigma_{real}$, and its standard deviation grows like $\sqrt{\text{rebalancing interval}}$. A ±5-point volatility error produces ±1.99, exactly the gamma P&L estimate.
- **Delta hedging on real data.** An at-the-money SPY call was sold in October 2025 at the VIX level of 21.7 % and hedged daily. The market realised 12.7 %, and the hedge earned 17.17 for a gamma P&L estimate of 17.25.

---

## 3. The market dashboard (`market_watch/`)

A Streamlit dashboard of daily market information, built **only on free sources**:

- **Market data.** Yahoo Finance gives 73 instruments: equity indices from Europe, the Americas, Asia-Pacific and Africa (JSE, EGX 30), FX including emerging and African currencies, commodities, and volatility indices.
- **Rates and macro.** The US Treasury and ECB yield curves are shown today, one month ago and one year ago. FRED provides macro indicators, and the ECB provides €STR and its reference FX rates.
- **Electricity.** Energy-Charts gives European day-ahead prices. ENTSO-E and EIA add more zones with a free API key.
- **News.** Google News RSS, central-bank feeds (Fed, ECB, BoE) and GDELT, with Finnhub and Alpha Vantage as optional sources.

**Analytics:**
- a performance heatmap (1D / 1W / 1M / YTD / 1Y);
- **unusual moves** detected as a z-score of the day's return against one year of daily returns, with the related headlines;
- correlation matrices and realised against implied volatility;
- a watchlist with threshold alerts.

The dashboard also embeds an **option pricer fed with today's data**, a volatility surface and the delta-hedging backtest.

**Engineering:**
- A single network module handles user agents, retries on HTTP 429/5xx and per-host rate limits.
- Every source separates fetching from parsing, so parsers are tested offline on recorded answers.
- A failing source shows a warning instead of crashing the page.

---

## 4. The structuring platform (`structuring/`)

**Goal:** help an investor compare several structured solutions built from their **objectives, market scenario and constraints**, with transparent valuations and risks.

### Workflow

1. **Questionnaire.**
   - *Objectives*: capital, target return, horizon, income need, currency.
   - *A described market scenario.* The investor lists weighted cases instead of only "bullish/bearish", for example 50 % "+8 % within six months, volatility −3 pts", 30 % stagnation, 20 % "−15 % within a year".
   - *Risk constraints*: maximum loss, lock-up, liquidity, issuer credit risk, knowledge of derivatives, feared scenarios.
2. **Market data with provenance.**
   - Spot and government yields are *observed*.
   - Implied or historical volatility and dividends (from an ETF proxy) are *model estimates*.
   - Issuer spread, recovery, fees and issue price are *hypothetical*.
   - Each parameter carries its source and date.
3. **Candidate structures.** The platform proposes:
   - direct exposure, as the reference;
   - 100 % and 90 % capital-protected notes;
   - a Phoenix autocallable with memory coupons;
   - a barrier reverse convertible;
   - call or put spreads;
   - a collar.

   Coupons and participations are **solved** so that the issuer-adjusted model value equals the issue price minus costs, exactly as a structurer prices a term sheet.
4. **Valuation sheet.**
   - The risk-free value, the issuer credit adjustment, and the bond and derivative parts.
   - The embedded costs, split into documented fees and the estimated margin.
   - The option components, valued one by one: zero-coupon bond, calls, digital coupon strip, down-and-in put. They add up exactly to the model value.
5. **Scenarios and stress tests.** Deterministic paths (crash then recovery, V-shape…) show the path dependency of barriers and autocalls. Distributions are given under the investor's scenario and under the risk-neutral measure, always labelled. The **stress test of the convictions** covers:
   - right, but the volatility explodes;
   - right, but slower;
   - right, then a reversal;
   - stagnation for the whole life;
   - wrong direction;
   - crash.
6. **Suitability checks.** Every declared preference is compared with the **contractual** property of each structure. For example, the declared maximum loss is checked against the worst loss the contract allows, and issuer default is reported separately. The verdict is never a ranking by return.
7. **Comparison with simple solutions.** Each structure is compared with direct exposure, a do-it-yourself option combination without issuer, and a deposit + underlying allocation with the same maximum loss. The platform says whether the complexity **adds value**, **is dominated**, or is a **trade-off**.
8. **Structuring lab.** Every term (barriers, coupons, maturity, protection, option legs) can be edited, with an immediate update of the theoretical cost and the return profile.
9. **Professional mode.** It shows the valuation assumptions, full Greeks (delta, gamma, vega, rho, credit, theta), Monte Carlo convergence and the model limits of each product.

### Mathematics

- **Dynamics.** A single GBM covers three asset classes through the carry yield $q$:
  - equities: $q$ = dividend yield;
  - FX (Garman-Kohlhagen): $q$ = foreign interest rate;
  - commodities (Black's model on futures): $q = r$.
- **Payoffs as cash flows.** Each structure maps simulated performance paths $P_t = S_t/S_0$ to cash flows, so one representation serves valuation, scenarios and risk. Barriers can be observed at maturity or daily.
- **Issuer credit.** Default follows a constant intensity $\lambda = s/(1-R)$, independent of the market. A promised flow at $t$ is worth $e^{-rt}\left(S(t) + R\,(1-S(t))\right)$ with $S(t) = e^{-\lambda t}$.
- **Solving a term.** The value is linear in the coupon or participation on fixed paths, so two valuations give the exact solution: $c^* = (V_{target} - V(0))/(V(1) - V(0))$.
- **Closed forms where they exist.** Protected notes, plain reverse convertibles and option packages use BSM; Monte Carlo is shown only as a check. Path-dependent structures use Monte Carlo with antithetic variates and a reported standard error.
- **Scenario mixture.** Each case is simulated on a share of paths equal to its subjective probability, with a piecewise drift chosen so that the **median** move is reached at the case horizon: $\mu = \ln(1+m)/h + \sigma^2/2$.

The full methodology, with assumptions and limits, is in [`structuring/METHODOLOGY.md`](structuring/METHODOLOGY.md).

---

## 5. Software engineering

- **Separation of concerns.** Calculation engines are pure Python modules; the user interfaces (terminal client, Streamlit apps) only lay out their results.
- **Reuse instead of duplication.**
  - The structuring platform reuses the pricer (BSM, exact GBM simulation, Monte Carlo statistics, Yahoo data, implied volatility) and the dashboard's yield-curve sources.
  - A single LSM core serves American and Bermudan options.
- **Tests.** 212 unit tests run offline in about 40 seconds.

  | Suite | Tests |
  |---|---|
  | `pricer` | 120 |
  | `market_watch` | 25 |
  | `structuring` | 67 |

  They check payoffs on hand-built paths, closed forms against Monte Carlo (within 3 standard errors), limit cases with exact answers, convergence rates, and the parsers on recorded API answers. The Streamlit apps were also driven headlessly with `streamlit.testing`.
- **Data honesty.** No fabricated market data: every fallback is labelled hypothetical, and model values are never presented as prices.

---

## 6. Limitations

- **Constant volatility.** There is no smile dynamics: flat volatility ignores the skew. As a result:
  - barrier reverse convertibles and autocallables are likely **overvalued** for the investor;
  - their solved coupons are higher than a real offer.
- **Constant rates and a flat curve at the maturity yield.** Products on interest rates are not covered.
- **Simplified issuer credit.** The default intensity is constant and independent of the market, and the issuer spread is a user input, not observed.
- **Single-underlying structures only.** No worst-of baskets, since correlations would be hard to justify without reliable data.
- **Free data sources.** Yahoo Finance is unofficial and can change or be delayed. There are no listed options on European indices in free data, so their volatility is historical. Index dividends come from ETF proxies.
- **LSM.** The estimate is biased low, as theory predicts, and the standard error does not measure that bias.
- **BAW.** Its error grows with maturity: about 2 % at 10 years.
- **Delta-hedging backtest.** It uses BSM deltas, a continuous dividend yield and proportional transaction costs only.
- **Suitability.** The checks are indicative. They do not replace a MiFID II suitability assessment, PRIIPs documentation or product governance.

## 7. Possible improvements

- **Local or stochastic volatility** (Dupire, Heston), calibrated to the implied volatility surface the project already builds. This matters most for barriers and autocallables.
- **A term structure of rates and a Hull-White model**, to cover rate products and stochastic discounting.
- **Multi-asset structures** (worst-of autocallables) with a correlation model and quanto adjustments.
- **An upper bound for American and Bermudan options** with the Andersen-Broadie dual method, giving a price interval around LSM.
- **Pathwise and likelihood-ratio Greeks** for the Monte Carlo pricers, instead of bump-and-revalue.
- **Variance reduction for path-dependent structures**: Brownian bridge for continuous barriers, quasi-Monte Carlo with Sobol sequences.
- **A database of real term sheets and KIDs**, to compare model values with actual issue prices and estimate real embedded costs.
- **Packaging**: a `requirements.txt` or `pyproject.toml`, continuous integration running the tests on every push, and a deployed demo.

---

## How to run

Requires Python 3.12 with `numpy`, `scipy`, `pandas`, `matplotlib`, `requests`, `yfinance` and `streamlit`:

```bash
pip install numpy scipy pandas matplotlib requests yfinance streamlit

python -m pricer.client                        # interactive option pricer (terminal)
python -m pricer.diagnostics                   # validation and convergence tables
python -m streamlit run market_watch/app.py    # market dashboard
python -m structuring.client                   # structured products platform
python -m structuring.client --report --offline   # text report, no network needed

# tests (offline)
python -m unittest discover -s pricer/test -t .
python -m unittest discover -s market_watch/test -t .
python -m unittest discover -s structuring/test -t .
```

Optional API keys for the dashboard (Finnhub, Alpha Vantage, ENTSO-E, EIA) go in `market_watch/.env`, which git ignores. Every page also works without them.

---

*Author: Arnaud Illien. Built as a personal project to practise derivatives pricing, risk management and product structuring.*
