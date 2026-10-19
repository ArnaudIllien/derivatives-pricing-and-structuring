# Methodology

This page documents how every number of the platform is produced, what it assumes and where it stops being reliable.

## 1. Data and provenance

Every parameter carries its provenance and date:

| Tag | Meaning | Examples |
|---|---|---|
| observed | read from a source on the date shown | closing level (Yahoo Finance), government yield (US Treasury, ECB) |
| model estimate | computed from observed data with a stated method | historical volatility, ATM implied volatility from listed option mids, ETF distribution yield as a dividend proxy |
| hypothetical | chosen by the user or a documented default | issuer credit spread (default 100 bp), recovery (40 %), fees (default 1 %/year), issue price (100 %), any fallback |

Conventions:

- **Risk-free rate.** The government yield at the product maturity is used as a flat, continuously compounded rate. For USD it comes from the US Treasury par curve, for EUR from the ECB AAA spot curve. Other currencies have no free curve, so the rate is hypothetical.
- **Dividends of an index.** They come from the distribution yield of an ETF on that index: EXW1.DE for the Euro Stoxx 50, CAC.PA for the CAC 40, SPY for the S&P 500. For single stocks, the trailing 12-month dividend yield is used.
- **Volatility.** The ATM implied volatility is used when the free source lists options on the underlying: ^SPX for the S&P 500, plus US stocks. Otherwise, one year of historical volatility.
- **FX.** The model is Garman-Kohlhagen: the carry is the interest rate of the base currency.
- **Commodities.** The model is Black's: the underlying is the front-month future, with a carry equal to r. The futures curve is assumed flat, which is a hypothesis.

## 2. Model

- **Dynamics.** Geometric Brownian motion under the risk-neutral measure: dS/S = (r − q) dt + σ dW. Volatility is constant and has no skew, and there are no jumps. Paths are simulated with the exact scheme of `pricer.model`. Payoffs read the performance P_t = S_t / S_0.
- **Barrier monitoring.** A barrier observed "at maturity" uses the final fixing only. A "daily" barrier is checked on every business day (252 per year).
- **Issuer credit.** Default follows a constant intensity λ = spread / (1 − R), independent of the underlying. A promised flow at t is worth its risk-free value × (S(t) + R (1 − S(t))), with S(t) = e^{−λt}. Option packages and direct holdings carry no issuer risk.

## 3. Valuation and costs

- **Fair value.** The fair value is the expected discounted cash flow under the risk-neutral measure.
  - Terminal structures have a closed form (BSM, from `pricer.european`): the direct exposure, protected notes, reverse convertibles without barrier and option packages. The Monte Carlo value is shown as a check.
  - Path-dependent structures (barrier reverse convertibles, autocallables) use Monte Carlo with antithetic variates. Their standard error is shown.
- **Decomposition.** The bond part is the present value of the guaranteed flows (protection, fixed coupons, cash), with issuer credit. The derivative part is the fair value minus the bond part.
- **Embedded costs.** They equal the issue price minus the issuer-adjusted fair value, and split into two lines:
  - the fees the user entered (hypothetical unless taken from a real KID);
  - an estimated issuer margin, which is the residual.
  Nothing is invented: an undocumented cost stays in the residual.
- **Solving the terms.**
  - Coupons and participations are set so that the fair value equals the issue price minus the targeted costs. This is linear in the term, so it is solved exactly with two valuations on the same paths.
  - Option packages are sized so that premium plus fees equal the investor's loss budget: the declared maximum loss, or the interest the cash earns until maturity when the investor accepts no loss.

**None of these values is an executable price.** A real issue price depends on the issuer's funding, hedging costs, skew, and the distributor's fees.

## 4. Scenarios and probabilities

- **Maturity grid.** The result is computed on deterministic paths: a straight log-linear path to each final level, and a path that first falls to 50 % at mid-life. Barriers and autocalls therefore follow the path.
- **Named path scenarios:**
  - conviction realised;
  - stagnation;
  - moderate decline (−15 %);
  - crash (−45 % within 3 months) followed by a partial recovery (to −25 %);
  - V-shape (−35 % at mid-life, back to +5 %);
  - strong rise (+40 %).
- **Distributions** are produced under two explicitly separated measures:
  - *Investor scenario.* The drift is such that the median move over the conviction horizon equals the stated view: μ = ln(1 + move)/h + σ²/2. The volatility is shifted by the volatility view: +5 points if rising, −3 points if falling. These shifts are hypotheses.
  - *Risk-neutral.* Drift r − q. It is market-implied and is not a forecast.
- **Outputs.** The favourable outcome is the 90th percentile, the central outcome the median, and the adverse outcome the 10th percentile. The platform also gives the probability of loss and the average of the worst 5 %. All of these hold only under the stated hypotheses.

## 4b. Described scenario, stress test, lab and simple solutions

**Described scenario.** The investor lists cases, each with a subjective probability, for example "+8 % within six months, volatility −3 points".
- *Simulation.* Each case is simulated on a share of the paths equal to its probability. The drift is piecewise: the median move is reached at the case horizon, then the median stays flat. The volatility is the market volatility plus the change stated for the case.
- *Candidates.* The probabilities drive them:
  - rise → participation, and a call spread up to the largest expected rise, at the horizon of that case;
  - stagnation → autocallable and reverse convertible;
  - fall more likely than rise → bearish notes and a put spread.
- *Volatility view.* A rising view favours bought options, a falling view sold options.

**Stress test of the convictions.** Each stress keeps or breaks one part of the main case (move m within horizon h):
- right, but volatility +15 points;
- right, but slower (m reached only at maturity);
- right, then reversal to −15 %;
- stagnation for the whole life;
- wrong direction (−m);
- crash: −35 % in 3 months, volatility +15 points for a year.

Each stress is a simulation with piecewise drift and volatility, not a single path.

**Structuring lab.** Every term is editable. The page shows:
- the model value (10,000 paths, standard error shown);
- the embedded cost against the 100 % issue price;
- the fair coupon or participation for the targeted costs;
- the maximum loss, the maturity profile, the result under the investor's scenario, vega and rho.

**Simple solutions.** Each structure is set against three simpler alternatives:
- the direct exposure;
- a do-it-yourself combination without issuer: a deposit plus listed options for a protected note, or a buy-write for income structures;
- a deposit plus the underlying (or a world equity ETF) with the same maximum loss. Its weight is w = 1 − (1 − L) e^{−rT}.

The verdict compares them only with alternatives whose maximum loss is no larger:
- *dominated*: an alternative does at least as well in expectation and in the 10th percentile;
- *value added*: the structure has the higher expectation;
- *trade-off*: the structure expects less but protects the adverse case better.

**Professional view:**
- the valuation assumptions with their provenance;
- the option components, valued one by one. Vanilla blocks use BSM. Path-dependent blocks (coupon strip, down-and-in put) are isolated by linearity on the same paths, so they add up exactly to the model value;
- delta and gamma, from bumps of the spot against the fixed initial level, with common random numbers;
- vega, rho, credit and a one-month theta;
- the Monte Carlo convergence;
- the model limits specific to each family.

## 5. Risk measures

- **Maximum contractual loss.** It is the worst total received over final levels from 0 % to 300 %, on straight paths and on paths that dip to 0 %, with the issuer solvent. It is reported separately from the loss on an issuer default, which is up to 1 − R for any note.
- **Sensitivities.** They are the change of the issuer-adjusted value for +1 volatility point, +25 bp of rates and +25 bp of issuer spread. Common random numbers are used so the Monte Carlo noise does not distort the differences.
- **Early exit.** It is the model value of the remaining flows after one year, in several moves of spot, volatility and rates. The hypothesis is that the spot moved to its new level just after inception, so past observations saw that level. The indicative bid subtracts a hypothetical 1 % bid-offer.

## 6. Suitability checks

Each declared preference is compared with a contractual or modelled property:

- capital protection;
- maximum loss at maturity;
- total loss;
- issuer credit risk;
- lock-up;
- liquidity;
- temporary loss;
- income;
- knowledge versus complexity;
- currency;
- the feared scenarios on their paths;
- the target return under the investor's scenario.

The verdict is "not consistent" if any check fails, and "consistent with reservations" if any check warns. It is never a ranking by return, and it is **not** a MiFID II suitability assessment.

## 7. Known limits and extensions

- **Flat volatility.** A down-and-in put is worth more under a real skew. Fair values of reverse convertibles and autocallables are therefore likely overstated, and their solved coupons are likely too high compared with a real offer. The planned extension is a local or stochastic volatility model calibrated to the surface (`pricer.implied_vol`).
- **Constant rates.** Products on interest rates are not covered. The planned extension is a Hull-White model.
- **Single underlying only.** Worst-of baskets need correlations, which are hard to justify without reliable data.
- **Exchange-rate risk.** A product in another currency is reported but not hedged; quanto structures are not modelled.
- **Simplified issuer credit.** The default intensity is constant and independent of the market. No funding spread is applied beyond the credit spread.
- **Settlement.** No settlement lag, no day-count subtleties, and a cash-settled equivalent of physical delivery.
