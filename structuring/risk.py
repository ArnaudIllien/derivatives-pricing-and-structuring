"""
risk.py - Maximum loss, sensitivities and early-exit (resale) analysis.

- Maximum theoretical loss (`max_contractual_loss`): the worst total received
  over final levels from 0 % to 300 %, on straight paths and on paths that dip to
  0 %, issuer solvent. Contract terms only: no probability involved. If the issuer
  defaults, a note can lose up to 100 % minus the recovery, whatever its terms.
- Sensitivities (`sensitivities`): change of the issuer-adjusted model value for
  +1 volatility point, +25 bp of rates, +25 bp of issuer spread, on the same
  random numbers (common random numbers keep the noise out of the differences).
- Early exit (`early_exit_values`): indicative model value of the remaining
  flows after `elapsed` years in several market moves. HYPOTHESIS: the spot
  jumped to its new level right after inception and stayed there until the exit
  date, so every observation date already passed saw that level (coupons paid or
  missed, autocall, daily barrier breached below the barrier). A real resale
  price is set by the issuer's secondary market: it adds a bid-offer spread (a
  hypothetical input) and the issuer may refuse to quote in stressed markets.
"""
import math
from dataclasses import dataclass, replace

import numpy as np

from pricer.market import Market

from .market import MarketData
from .products import Autocallable, OptionPackage, Product, ReverseConvertible
from .scenarios import maturity_grid
from .simulation import SimulationSettings, simulate
from .valuation import present_values, value


@dataclass(frozen=True)
class MaxLoss:
    contractual: float           # worst loss, fraction of capital, issuer solvent
    worst_level: float           # final level where it happens
    with_issuer_default: float   # worst loss if the issuer defaults (= contractual without issuer risk)


def max_contractual_loss(product: Product, market: Market, recovery: float = 0.4) -> MaxLoss:
    levels = np.linspace(0.0, 3.0, 121)
    outcomes = maturity_grid(product, market, levels) + maturity_grid(product, market, levels, trough=1e-12)
    worst = min(outcomes, key=lambda o: o.total_received)
    loss = max(0.0, 1.0 - worst.total_received)
    default_loss = 1.0 - recovery if product.issuer_risk else 0.0
    return MaxLoss(loss, worst.final_level, max(loss, default_loss))


@dataclass(frozen=True)
class Sensitivities:
    base: float
    vega: float          # per +1 volatility point
    rho: float           # per +25 bp of the risk-free rate
    credit: float        # per +25 bp of issuer spread
    std_error: float


def sensitivities(product: Product, market_data: MarketData,
                  settings: SimulationSettings = SimulationSettings()) -> Sensitivities:
    base = value(product, market_data, settings)
    vol_up = value(product, market_data, settings, vol_shift=0.01)
    rate_up = value(product, market_data, settings, rate_shift=0.0025)
    spread_up = value(product, market_data.with_overrides(credit_spread=market_data.credit_spread.value + 0.0025),
                      settings)
    return Sensitivities(base.value, vol_up.value - base.value, rate_up.value - base.value,
                         spread_up.value - base.value, base.std_error)


@dataclass(frozen=True)
class FullGreeks:
    """Per unit of notional. Delta and gamma per 1 % move of the spot; theta over one month, spot unchanged."""
    delta: float
    gamma: float
    vega: float
    rho: float
    credit: float
    theta_month: float


def _value_on_paths(product, market_data, market, times, paths) -> float:
    flows = product.cash_flows(paths, times, market)
    if product.issuer_risk:
        return float(present_values(flows, market.rate, market_data.credit_spread.value,
                                    market_data.recovery.value).mean())
    return float(present_values(flows, market.rate).mean())


def full_greeks(product: Product, market_data: MarketData,
                settings: SimulationSettings = SimulationSettings()) -> FullGreeks:
    """Bump-and-revalue on common random numbers. A spot bump scales the paths (fixing unchanged)."""
    market = market_data.pricer_market()
    times, paths = simulate(product, market, settings)
    v0 = _value_on_paths(product, market_data, market, times, paths)
    up = _value_on_paths(product, market_data, market, times, paths*1.01)
    down = _value_on_paths(product, market_data, market, times, paths*0.99)
    basic = sensitivities(product, market_data, settings)
    elapsed = min(1/12, 0.5*product.maturity)
    first_date = product.observation_times()[0]
    theta = float("nan")
    if elapsed < first_date - 1e-9 or isinstance(product, OptionPackage):
        if isinstance(product, OptionPackage):
            later_value = next(e.model_value for e in early_exit_values(product, market_data, elapsed)
                               if e.label == "unchanged")
        else:   # no observation date passes within the month: nothing is paid or triggered
            state = state_after(product, elapsed, 1.0)
            t2, p2 = simulate(state.product, market, settings)
            later_value = _value_on_paths(state.product, market_data, market, t2, p2)
        theta = later_value - v0
    return FullGreeks((up - down)/2, up - 2*v0 + down, basic.vega, basic.rho, basic.credit, theta)


def convergence(product: Product, market_data: MarketData, path_counts=(2_500, 5_000, 10_000, 20_000, 40_000)):
    """(n_paths, value, standard error) for increasing numbers of paths (independent seeds)."""
    rows = []
    for i, n in enumerate(path_counts):
        v = value(product, market_data, SimulationSettings(n_paths=n, seed=100 + i))
        rows.append((n, v.monte_carlo_value, v.monte_carlo_error))
    return rows


@dataclass(frozen=True)
class AliveState:
    product: Product | None      # remaining contract (None: already redeemed early)
    received: float              # flows received before the exit date
    note: str = ""


def state_after(product: Product, elapsed: float, level: float) -> AliveState:
    """The contract after `elapsed` years when the level was `level` on every date since inception."""
    if not 0 < elapsed < product.maturity:
        raise ValueError("The elapsed time must lie strictly between 0 and the maturity.")
    remaining = product.maturity - elapsed
    passed = [t for t in product.observation_times() if t <= elapsed + 1e-9]
    if isinstance(product, ReverseConvertible):
        received = len(passed)*product.coupon_rate/product.coupon_frequency
        breached = product.barrier is not None and product.barrier_monitoring == "daily" and level < product.barrier
        return AliveState(replace(product, maturity=remaining, barrier_breached=breached), received)
    if isinstance(product, Autocallable):
        coupon = product.coupon_rate/product.observation_frequency
        paid = level >= product.coupon_barrier
        callable_dates = [i + 1 for i in range(len(passed)) if i + 1 >= product.first_call]
        if callable_dates and level >= product.autocall_trigger:
            first = callable_dates[0]
            return AliveState(None, 1.0 + (first*coupon if paid else 0.0),
                              f"redeemed early at par on observation {first}")
        missed = 0 if paid else (len(passed) + product.missed_coupons if product.memory else 0)
        breached = product.barrier_monitoring == "daily" and level < product.capital_barrier
        alive = replace(product, maturity=remaining, first_call=max(1, product.first_call - len(passed)),
                        missed_coupons=missed, barrier_breached=breached)
        return AliveState(alive, len(passed)*coupon if paid else 0.0)
    return AliveState(replace(product, maturity=remaining), 0.0)


@dataclass(frozen=True)
class ExitValue:
    label: str
    spot_move: float
    vol_shift: float
    rate_shift: float
    model_value: float           # model value of the remaining flows (nan if redeemed early)
    received_so_far: float
    indicative_bid: float        # model value minus the hypothetical bid-offer spread
    note: str = ""

    @property
    def total_if_sold(self) -> float:
        return self.received_so_far + (self.indicative_bid if not math.isnan(self.indicative_bid) else 0.0)


EXIT_SCENARIOS = (("unchanged", 0.0, 0.0, 0.0), ("spot -20 %", -0.2, 0.0, 0.0), ("spot -40 %, vol +5 pts", -0.4, 0.05, 0.0),
                  ("spot +20 %", 0.2, 0.0, 0.0), ("volatility +10 pts", 0.0, 0.10, 0.0), ("rates +100 bp", 0.0, 0.0, 0.01))


def early_exit_values(product: Product, market_data: MarketData, elapsed: float = 1.0, bid_offer: float = 0.01,
                      settings: SimulationSettings = SimulationSettings(n_paths=10_000)) -> list[ExitValue]:
    elapsed = min(elapsed, 0.8*product.maturity)
    spread, recovery = market_data.credit_spread.value, market_data.recovery.value
    rows = []
    for label, move, vol_shift, rate_shift in EXIT_SCENARIOS:
        level = 1.0 + move
        market = market_data.pricer_market(vol_shift, rate_shift)
        if isinstance(product, OptionPackage):
            legs_market = Market(level, market.rate, market.volatility, market.dividend_yield)
            legs = sum(leg.price(legs_market, product.maturity - elapsed) for leg in product.legs)
            cash = (1.0 - product.premium(market_data.pricer_market()) - product.fees)*math.exp(
                market_data.rate.value*elapsed)
            rows.append(ExitValue(label, move, vol_shift, rate_shift, legs + cash, 0.0, legs + cash - bid_offer))
            continue
        state = state_after(product, elapsed, level)
        if state.product is None:
            rows.append(ExitValue(label, move, vol_shift, rate_shift, math.nan, state.received, math.nan, state.note))
            continue
        times, paths = simulate(state.product, market, settings)
        flows = state.product.cash_flows(paths*level, times, market)
        pv = present_values(flows, market.rate, spread, recovery) if state.product.issuer_risk else \
            present_values(flows, market.rate)
        model = float(pv.mean())
        cost = bid_offer if product.issuer_risk else 0.0
        rows.append(ExitValue(label, move, vol_shift, rate_shift, model, state.received, model - cost, state.note))
    return rows
