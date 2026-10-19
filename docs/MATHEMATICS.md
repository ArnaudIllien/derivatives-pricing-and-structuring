# Mathematics of the models

The detailed methods behind the [README](../README.md). The structuring platform has its own methodology, with its assumptions and limits, in [`structuring/METHODOLOGY.md`](../structuring/METHODOLOGY.md).

## Option pricing (`pricer/`)

### Black-Scholes-Merton (European options)

Under the risk-neutral measure the underlying follows a geometric Brownian motion:

$$dS_t = (r-q)\,S_t\,dt + \sigma S_t\,dW_t, \qquad C = S e^{-qT}N(d_1) - K e^{-rT}N(d_2),$$

$$d_{1,2} = \frac{\ln(S/K) + (r - q \pm \sigma^2/2)T}{\sigma\sqrt{T}}.$$

Discrete cash dividends use the Bos-Vandermark escrowed-dividend approximation, which splits each dividend between the spot (its "near" part) and the strike (its "far" part).

### Cox-Ross-Rubinstein binomial tree (European, American, Bermudan)

- **The tree.** $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$ and $p = (e^{(r-q)\Delta t} - d)/(u-d)$. Backward induction gives $V = \max(\text{payoff},\; e^{-r\Delta t}[pV_u + (1-p)V_d])$ on the exercise dates.
- **Discrete dividends are exact** (Vellekoop-Nieuwenhuis). The tree stays recombining, and the value before the drop is interpolated as $V_{cum}(S) = V_{ex}(S - D)$.
- **Smoothing.** Averaging the trees with $n$ and $n+1$ steps removes the odd/even oscillation.
- **Exercise style.** The exercise style is a rule per step: always (American), never (European), or on given steps (Bermudan).

### Barone-Adesi-Whaley (American, analytic approximation)

The early-exercise premium solves a quadratic ODE. The critical price $S^*$ is found with Brent's method:

$$C_{Am}(S) = C_{Eu}(S) + A_2 \left(\frac{S}{S^*}\right)^{q_2}, \quad S < S^*.$$

Calls with discrete dividends also use Black's pseudo-American rule.

### Longstaff-Schwartz least-squares Monte Carlo (American and Bermudan)

1. Simulate paths with the exact GBM scheme.
2. Going backwards from maturity, on the in-the-money paths, regress the discounted future cash flow on a basis of $x = S/K$ (polynomial, or weighted Laguerre $e^{-x/2}L_k(x)$). The fitted value $\hat C(x) = B(x)\hat\beta$ is the continuation value.
3. Exercise where the immediate payoff exceeds $\hat C$.

The policy is learnt on one set of paths and evaluated on an **independent** one. Any policy is sub-optimal, so the estimate is a lower bound in expectation. One core algorithm serves both options: it runs on a time grid with a mask of exercisable dates, the regular grid for an American option and any list of dates for a Bermudan one.

### Arithmetic Asian options (Monte Carlo with a control variate)

The geometric average of a GBM over fixing dates $t_1,\dots,t_n$ is lognormal:

$$\ln G \sim \mathcal N\!\left(\ln S_0 + (r-q-\tfrac{\sigma^2}{2})\,\bar t,\; \tfrac{\sigma^2}{n^2}\sum_{i,j}\min(t_i,t_j)\right).$$

The geometric Asian option therefore has a closed form (a Black formula on $G$). It serves as a control variate for the arithmetic option: $\hat Y_{cv} = \bar Y - \beta(\bar X - \mathbb E[X])$ with $\beta = \mathrm{Cov}(Y,X)/\mathrm{Var}(X)$. Antithetic pairs and batching complete the estimator.

### Greeks

The Greeks come from central finite differences on any pricer: Delta and Gamma by spot bumps, Vega by a volatility bump, Theta by moving the contract and the dividend dates forward. Monte Carlo pricers use common random numbers, so the noise does not dominate the differences.

### Implied volatility

- **Inversion.** Brent's method on $\sigma \mapsto \text{price}(\sigma) - \text{market}$ inverts any pricer: BSM for European options, BAW for American ones.
- **Quotes used.** Smiles use out-of-the-money mid quotes with a bid-ask filter. Moneyness is $\ln(K/F)$.
- **The forward.** It can come from **put-call parity** on each expiry's own quotes, $C(K) - P(K) = D\,(F - K)$, fitted by least squares near the money (slope $-D$, intercept $DF$). This avoids a spurious jump between put and call volatilities when the rate or dividend assumption is wrong.

### Delta-hedging P&L

Sell an option at volatility $\sigma_h$ and hold the BSM delta, rebalanced at discrete dates. To second order, the final P&L is

$$\text{P\&L} \approx \sum_k \tfrac12\,\Gamma_k S_k^2\left(\sigma_h^2\,\Delta t - r_k^2\right) e^{r(T-t_k)},$$

where $r_k$ is the realised log return over step $k$. Its mean is driven by $\sigma_h^2 - \sigma_{real}^2$. Its spread, the hedging error, shrinks like $\sqrt{\Delta t}$.

## Structured products (`structuring/`)

- **Dynamics.** A single GBM covers three asset classes through the carry yield $q$:
  - equities: $q$ = dividend yield;
  - FX (Garman-Kohlhagen): $q$ = foreign interest rate;
  - commodities (Black's model on futures): $q = r$.
- **Payoffs as cash flows.** Each structure maps simulated performance paths $P_t = S_t/S_0$ to cash flows. Barriers are observed at maturity or daily, on 252 business days a year.
- **Issuer credit.** Default follows a constant intensity $\lambda = s/(1-R)$, independent of the market. A promised flow at $t$ is worth $e^{-rt}\left(S(t) + R\,(1-S(t))\right)$, with $S(t) = e^{-\lambda t}$.
- **Solving a term.** On fixed paths the value is linear in the coupon or the participation, so $c^* = (V_{target} - V(0))/(V(1) - V(0))$ is exact.
- **Components.** Path-dependent blocks (the conditional coupon strip, the down-and-in put) are isolated by linearity on the same paths, so they add up exactly to the model value.
- **Investor scenario.** A mixture of cases, each simulated on a share of paths equal to its subjective probability. The drift is piecewise, chosen so that the **median** move $m$ is reached at the case horizon $h$: $\mu = \ln(1+m)/h + \sigma^2/2$, then a flat median.
- **Stress tests.** The same piecewise simulation with broken views: volatility +15 points, a slower move, a reversal, stagnation, the wrong direction, a crash.
