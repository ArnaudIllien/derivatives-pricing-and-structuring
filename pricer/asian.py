"""
asian.py - Monte Carlo pricer for fixed-strike arithmetic Asian options.

USAGE
    from pricer import AsianOption, AsianParameters, Market, asian_price
    option = AsianOption.regular("call", 100, 1, 12)        # 12 monthly observations
    result = asian_price(option, Market(100, 0.05, 0.2))
    result.price, result.std_error, result.confidence_interval_95

Algorithm
---------
1. Simulate N GBM paths under the risk-neutral measure with the exact scheme
   (model.py), on a grid made of the observation dates and the dividend dates.
   There is no discretisation error: the payoff only needs the spot at the
   observation dates.
2. On each path, A = arithmetic average of the spot at the observation dates,
   and the discounted payoff is e^{-rT} max(sign * (A - K), 0).
3. Price = average of the discounted payoffs; standard error = std / sqrt(N),
   computed on independent draws (antithetic pairs averaged).

Control variate
---------------
The geometric average G = (prod S(t_i))^(1/n) of a GBM is lognormal, so the
geometric Asian option has a closed form (`geometric_asian_price`):
    ln G ~ N(mu, v),  mu = ln S0 + (r - q - sigma²/2) mean(t_i),
                      v  = sigma² / n² sum_i sum_j min(t_i, t_j).
Its payoff is very close to the arithmetic one on each path, so subtracting
beta (geometric payoff - closed form) removes most of the noise: the standard
error drops by one to two orders of magnitude. The price is unchanged in
expectation.

Assumptions and limits
----------------------
- GBM with constant r, q, sigma (market.py); payment at T.
- Discrete dividends are simulated exactly (drop of D after the dividend date,
  cum-dividend observation on that date), but G is then no longer lognormal:
  the control variate is switched off and the result says so.
- The standard error measures the Monte Carlo noise only; there is no other
  error source here (no discretisation, no regression).
"""
import math
from dataclasses import dataclass, replace

import numpy as np

from .contract import AsianOption
from .european import bsm_price
from .market import Market
from .model import BlackScholesModel, nearest_columns
from .monte_carlo import control_variate, independent_draws, mean_and_error


@dataclass(frozen=True)
class AsianParameters:
    n_paths: int = 100_000
    seed: int | None = 0            # None: not reproducible
    antithetic: bool = True
    control_variate: bool = True    # geometric Asian control; ignored with discrete dividends
    batch_size: int = 50_000        # paths simulated at once, to bound the memory with many observations


@dataclass(frozen=True)
class AsianResult:
    price: float
    std_error: float
    plain_price: float              # plain Monte Carlo estimate, without the control variate
    plain_std_error: float
    control_variate_used: bool

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        return self.price - 1.96*self.std_error, self.price + 1.96*self.std_error

    @property
    def variance_reduction(self) -> float:
        """Variance of the plain estimator divided by the variance of the one returned."""
        if self.std_error == 0:
            return math.inf if self.plain_std_error > 0 else 1.0
        return (self.plain_std_error/self.std_error)**2


def geometric_asian_price(option: AsianOption, market: Market) -> float:
    """Exact price of the option on the geometric average (GBM without discrete dividends)."""
    if market.dividends_before(option.maturity):
        raise ValueError("The geometric Asian closed form does not hold with discrete dividends.")
    if option.maturity == 0:
        return option.payoff(market.spot)
    t = np.array(option.observation_times)
    sigma, T = market.volatility, option.maturity
    mu = math.log(market.spot) + (market.carry - sigma**2/2)*t.mean()
    variance = sigma**2*np.minimum.outer(t, t).sum()/len(t)**2
    forward = math.exp(mu + variance/2)                     # E[G]
    # Black's formula on G: a BSM price with spot e^{-rT} E[G], q = 0 and total variance v
    return bsm_price(option.is_call, forward*math.exp(-market.rate*T), option.strike, T, market.rate, 0.0,
                     math.sqrt(variance/T))


def _discounted_payoffs(option: AsianOption, market: Market, p: AsianParameters, with_geometric: bool):
    """Independent draws of the discounted arithmetic payoff (and of the geometric one if asked)."""
    model = BlackScholesModel(market)
    times = model.time_grid(option.maturity, option.observation_times)
    columns = nearest_columns(times, option.observation_times)
    discount = math.exp(-market.rate*option.maturity)
    rng = np.random.default_rng(p.seed)
    batch = max(2, p.batch_size - p.batch_size % 2)

    arithmetic, geometric = [], []
    for start in range(0, p.n_paths, batch):
        observed = model.simulate(times, min(batch, p.n_paths - start), rng, p.antithetic)[:, columns]
        arithmetic.append(independent_draws(discount*option.payoff(observed.mean(axis=1)), p.antithetic))
        if with_geometric:
            average = np.exp(np.log(observed).mean(axis=1))
            geometric.append(independent_draws(discount*option.payoff(average), p.antithetic))
    return np.concatenate(arithmetic), np.concatenate(geometric) if with_geometric else None


def asian_price(option: AsianOption, market: Market, parameters: AsianParameters = AsianParameters()) -> AsianResult:
    p = parameters
    if option.maturity == 0:
        value = option.payoff(market.spot)
        return AsianResult(value, 0.0, value, 0.0, False)
    if p.n_paths < 4:
        raise ValueError("At least 4 paths are needed.")
    if p.antithetic and p.n_paths % 2:
        raise ValueError("The number of paths must be even in antithetic mode.")

    use_control = p.control_variate and not market.dividends_before(option.maturity)
    payoffs, controls = _discounted_payoffs(option, market, p, use_control)
    plain_price, plain_error = mean_and_error(payoffs)
    if not use_control:
        return AsianResult(plain_price, plain_error, plain_price, plain_error, False)
    price, error, _ = control_variate(payoffs, controls, geometric_asian_price(option, market))
    return AsianResult(price, error, plain_price, plain_error, True)


def asian_convergence(option: AsianOption, market: Market, path_counts=(10_000, 40_000, 160_000, 640_000),
                      parameters: AsianParameters = AsianParameters()) -> list[tuple[int, AsianResult]]:
    """Prices for increasing numbers of paths (independent seeds): the error should fall as 1/sqrt(N)."""
    seeds = np.random.SeedSequence(parameters.seed).spawn(len(path_counts))
    return [(n, asian_price(option, market, replace(parameters, n_paths=n, seed=int(seed.generate_state(1)[0]))))
            for n, seed in zip(path_counts, seeds)]
