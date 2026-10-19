"""
hedging.py - Delta-hedging backtest of a European option, on simulated or historical paths.

HOW TO RUN
    python -m pricer.hedging          (needs network for the historical part)
    It prints (1) a simulated check of the theory and (2) a one-year hedge of an
    at-the-money SPY call on real Yahoo prices.

    In code:
        from pricer.hedging import HedgeParameters, simulated_hedge, historical_hedge
        result = historical_hedge("SPY", date(2025, 1, 2), date(2025, 12, 31), "call", 1.0,
                                  HedgeParameters(hedge_volatility=0.18))
        result.final_pnl, result.summary()

The strategy
------------
At t0 the trader SELLS the option at its BSM price with the hedge volatility
sigma_h and buys delta shares. At each rebalancing date the stock position is
reset to the BSM delta (at sigma_h, remaining maturity). The cash account earns
the rate r, the shares earn the dividend yield q. At maturity everything is
liquidated and the option payoff is paid.

    hedge P&L(t) = cash(t) + delta(t) S(t) - V(t),   V = BSM value at sigma_h,

so the P&L starts at 0 and ends at cash(T) + delta S(T) - payoff(S(T)).

What the P&L means
------------------
With continuous rebalancing and no costs, the P&L is the gamma P&L
    sum_k 1/2 Gamma_k S_k² (sigma_h² dt - (dS_k / S_k)²),  discounted,
so:
- if the realised volatility equals sigma_h, the P&L is 0 up to the
  discretisation noise, whose std decreases as 1/sqrt(number of rebalancings);
- selling at sigma_h above the realised volatility earns money, below loses money.
`gamma_pnl` returns this approximation path by path, to explain the result.

Assumptions: European payoff, constant r and q over the backtest, BSM deltas,
proportional transaction costs on the traded stock notional, no discrete
dividends, no financing spread.
"""
import math
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
from scipy.stats import norm

from .contract import Option
from .european import bsm_price_and_delta
from .market import Market
from .model import BlackScholesModel


@dataclass(frozen=True)
class HedgeParameters:
    hedge_volatility: float          # sigma_h: sale price and deltas
    rebalance_every: int = 1         # rebalance at one date in k of the path (1 = every date)
    transaction_cost: float = 0.0    # proportional cost on traded stock notional (0.0005 = 5 bp)


@dataclass(frozen=True)
class HedgeResult:
    times: np.ndarray                # (n_times,) in years from the start
    spots: np.ndarray                # (n_paths, n_times)
    deltas: np.ndarray               # (n_paths, n_times) stock held after rebalancing (0 at maturity)
    option_values: np.ndarray        # (n_paths, n_times) BSM value at sigma_h, payoff at maturity
    pnl: np.ndarray                  # (n_paths, n_times) hedge P&L, 0 at t0
    costs: np.ndarray                # (n_paths,) total transaction costs paid
    initial_price: float             # premium received at t0
    gamma_pnl: np.ndarray            # (n_paths,) gamma P&L approximation, discounted to T
    realized_volatility: np.ndarray  # (n_paths,) annualised from the path's log returns
    dates: tuple = ()                # calendar dates of the path (historical backtest only)
    rate: float = 0.0

    @property
    def final_pnl(self) -> np.ndarray:
        return self.pnl[:, -1]

    def summary(self) -> dict[str, float]:
        final = self.final_pnl
        return {"premium": self.initial_price, "mean P&L": float(final.mean()),
                "std P&L": float(final.std(ddof=1)) if len(final) > 1 else 0.0,
                "mean P&L / premium": float(final.mean()/self.initial_price) if self.initial_price else math.nan,
                "mean costs": float(self.costs.mean()), "mean gamma P&L": float(self.gamma_pnl.mean()),
                "mean realised vol": float(self.realized_volatility.mean())}


def delta_hedge(option: Option, times, spots, rate: float, dividend_yield: float,
                parameters: HedgeParameters) -> HedgeResult:
    """Short option, delta-hedged along given paths. times[0] = 0, times[-1] = option.maturity."""
    times = np.asarray(times, dtype=float)
    spots = np.atleast_2d(np.asarray(spots, dtype=float))
    if len(times) < 2 or spots.shape[1] != len(times):
        raise ValueError("The paths need one price per date, and at least two dates.")
    if abs(times[0]) > 1e-12 or abs(times[-1] - option.maturity) > 1e-9:
        raise ValueError("The dates must run from 0 to the option maturity.")
    if parameters.hedge_volatility <= 0 or parameters.rebalance_every < 1:
        raise ValueError("The hedge volatility must be positive and rebalance_every at least 1.")

    sigma, r, q, K = parameters.hedge_volatility, rate, dividend_yield, option.strike
    n_paths, n_times = spots.shape
    last = n_times - 1
    remaining = option.maturity - times
    values = np.empty_like(spots)
    model_deltas = np.empty_like(spots)
    values[:, :last], model_deltas[:, :last] = bsm_price_and_delta(option.is_call, spots[:, :last], K,
                                                                   remaining[None, :last], r, q, sigma)
    values[:, last] = option.payoff(spots[:, last])
    model_deltas[:, last] = 0.0

    held = np.zeros_like(spots)
    pnl = np.zeros_like(spots)
    costs = np.zeros(n_paths)
    premium = values[:, 0]
    held[:, 0] = model_deltas[:, 0]
    cost = parameters.transaction_cost*np.abs(held[:, 0])*spots[:, 0]
    cash = premium - held[:, 0]*spots[:, 0] - cost
    costs += cost
    for k in range(1, n_times):
        dt = times[k] - times[k - 1]
        cash = cash*math.exp(r*dt) + held[:, k - 1]*spots[:, k - 1]*(math.exp(q*dt) - 1)
        target = model_deltas[:, k] if (k % parameters.rebalance_every == 0 or k == last) else held[:, k - 1]
        cost = parameters.transaction_cost*np.abs(target - held[:, k - 1])*spots[:, k]
        cash = cash - (target - held[:, k - 1])*spots[:, k] - cost
        costs += cost
        held[:, k] = target
        pnl[:, k] = cash + held[:, k]*spots[:, k] - values[:, k]

    log_returns = np.diff(np.log(spots), axis=1)
    dts = np.diff(times)
    realized = np.sqrt((log_returns**2).sum(axis=1)/times[-1])
    return HedgeResult(times, spots, held, values, pnl, costs, float(premium[0]),
                       _gamma_pnl(spots, times, log_returns, dts, K, r, q, sigma, option.maturity), realized, rate=r)


def _gamma_pnl(spots, times, log_returns, dts, strike, r, q, sigma, maturity):
    """sum_k 1/2 Gamma_k S_k² (sigma² dt_k - r_k²), each term carried to T at the rate r."""
    S, tau = spots[:, :-1], maturity - times[:-1]
    d1 = (np.log(S/strike) + (r - q + sigma**2/2)*tau)/(sigma*np.sqrt(tau))
    gamma = np.exp(-q*tau)*norm.pdf(d1)/(S*sigma*np.sqrt(tau))
    terms = 0.5*gamma*S**2*(sigma**2*dts - log_returns**2)
    return (terms*np.exp(r*tau)).sum(axis=1)


def simulated_hedge(option: Option, market: Market, parameters: HedgeParameters, n_paths: int = 2_000,
                    n_steps: int = 252, seed: int | None = 0) -> HedgeResult:
    """Hedge along GBM paths whose true volatility is market.volatility (continuous yield only)."""
    if market.dividends:
        raise ValueError("The hedging backtest supports a continuous dividend yield only.")
    model = BlackScholesModel(market)
    times = model.simulation_times(option.maturity, n_steps)
    paths = model.simulate(times, n_paths, np.random.default_rng(seed))
    return delta_hedge(option, times, paths, market.rate, market.dividend_yield, parameters)


def historical_hedge(ticker: str, start: date, end: date, option_type: str, moneyness: float,
                     parameters: HedgeParameters, client=None, rate: float | None = None,
                     dividend_yield: float = 0.0) -> HedgeResult:
    """
    Hedge an option written on `ticker` at the first close on or after `start`, with strike
    moneyness * S0, expiring at the last close on or before `end` (dates in actual/365).
    The rate defaults to the 13-week T-bill on the start date.
    """
    from .market_data import US_TBILL, YahooClient, tbill_rate
    client = client or YahooClient()
    closes = client.history(ticker, start=start, end=end)
    closes = closes[[start <= d <= end for d in closes.index]]
    if len(closes) < 3:
        raise ValueError(f"Not enough prices for {ticker!r} between {start} and {end}.")
    dates = list(closes.index)
    if rate is None:
        bills = client.history(US_TBILL, start=start, end=end)
        if len(bills) == 0:
            raise ValueError("No T-bill rate on the start date: pass rate=.")
        rate = tbill_rate(float(bills.iloc[0]))
    times = np.array([(d - dates[0]).days/365 for d in dates])
    option = Option(option_type, moneyness*float(closes.iloc[0]), float(times[-1]))
    result = delta_hedge(option, times, closes.values, rate, dividend_yield, parameters)
    return replace(result, dates=tuple(dates))


def _demo():
    option, market = Option("call", 100, 1), Market(100, 0.03, 0.2)
    print("1. Simulated GBM paths, true volatility 20 %, ATM call T=1 (2,000 paths)")
    print(f"{'hedge vol':>9} {'rebalance':>10} {'cost':>6} {'mean P&L':>9} {'std P&L':>8} {'gamma P&L':>10}")
    for sigma_h, every, cost in ((0.2, 1, 0), (0.2, 5, 0), (0.2, 21, 0), (0.25, 1, 0), (0.15, 1, 0), (0.2, 1, 5e-4)):
        r = simulated_hedge(option, market, HedgeParameters(sigma_h, every, cost)).summary()
        label = {1: "daily", 5: "weekly", 21: "monthly"}[every]
        print(f"{sigma_h:>9.0%} {label:>10} {cost*1e4:>4.0f}bp {r['mean P&L']:>9.3f} {r['std P&L']:>8.3f}"
              f" {r['mean gamma P&L']:>10.3f}")

    print("\n2. Real prices: short 1-year ATM SPY call from 2025-01-02, hedged daily")
    try:
        for sigma_h in (0.15, 0.20):
            result = historical_hedge("SPY", date(2025, 1, 2), date(2025, 12, 31), "call", 1.0,
                                      HedgeParameters(sigma_h))
            s = result.summary()
            print(f"   sigma_h={sigma_h:.0%}: premium {s['premium']:.2f}, final P&L {s['mean P&L']:.2f},"
                  f" gamma P&L {s['mean gamma P&L']:.2f}, realised vol {s['mean realised vol']:.1%}")
    except Exception as error:   # no network, Yahoo change...
        print(f"   unavailable: {error}")


if __name__ == "__main__":
    _demo()
