"""
valuation.py - Fair value, issuer credit, embedded costs, decomposition and solving of the terms.

Definitions (all in % of the notional, at inception)
----------------------------------------------------
- Risk-free fair value   V0 = sum_j E_Q[CF_j] e^{-r t_j}   (Monte Carlo, risk-neutral).
- Issuer-adjusted value  V  = sum_j E_Q[CF_j] e^{-r t_j} (S_j + R (1 - S_j)),
                         S_j = e^{-lambda t_j}, lambda = spread / (1 - R): the survival
                         probability of the issuer, independent of the underlying. Only
                         for notes (issuer_risk); a package or direct holding has V = V0.
                         The spread and the recovery R are HYPOTHETICAL inputs.
- Issue price            100 % by convention (HYPOTHETICAL: no real offer is observed).
- Embedded costs         issue price - V: everything the investor pays above the model
                         value (distribution and structuring fees, issuer margin, hedging
                         costs). Split into the fees the user documented (from a term sheet
                         or KID) and the remaining estimated margin. Nothing is invented: an
                         undocumented fee stays inside the "estimated margin" line.
- Decomposition          bond part = PV of the guaranteed flows (protection, fixed
                         coupons) with issuer credit; derivative part = V - bond part.
                         For terminal structures, the option is also priced in closed form
                         (pricer.european) as a check of the Monte Carlo.

Solving a term
--------------
A structurer fixes the coupon or the participation so that the model value equals
the issue price minus the targeted costs. Every structure here is LINEAR in its
coupon (paid on fixed paths) or participation, so two valuations on the same
paths give the exact solution: term = (target - V(0)) / (V(1) - V(0)).
"""
import math
from dataclasses import dataclass

import numpy as np

from pricer.european import bsm_price
from pricer.market import Market
from pricer.monte_carlo import independent_draws, mean_and_error

from .market import MarketData
from .products import (Autocallable, CapitalProtectedNote, CashFlows, DirectExposure, OptionPackage, Product,
                       ReverseConvertible, with_value)
from .simulation import SimulationSettings, simulate

ISSUE_PRICE = 1.0


def survival(times: np.ndarray, spread: float, recovery: float) -> np.ndarray:
    """Expected fraction of a promised flow received: S + R (1 - S), S = exp(-spread / (1 - R) t)."""
    hazard = spread/(1 - recovery) if recovery < 1 else 0.0
    s = np.exp(-hazard*np.asarray(times))
    return s + recovery*(1 - s)


def present_values(flows: CashFlows, rate: float, spread: float = 0.0, recovery: float = 0.0) -> np.ndarray:
    """Discounted value of the flows on each path; issuer credit applied only to issuer flows."""
    factors = np.exp(-rate*flows.times)
    if flows.issuer_risk:
        factors = factors*survival(flows.times, spread, recovery)
    return flows.amounts @ factors


@dataclass(frozen=True)
class Valuation:
    product: Product
    risk_free_value: float          # V0, model, no issuer credit
    value: float                    # V, model, with issuer credit (= V0 without issuer risk)
    std_error: float                # Monte Carlo standard error (0 when the value is in closed form)
    method: str                     # "closed form (BSM)" or "Monte Carlo"
    monte_carlo_value: float        # always computed: the check of the closed form
    monte_carlo_error: float        # its standard error
    issue_price: float              # hypothetical convention: 100 %
    documented_fees: float          # fees the user entered (hypothetical unless taken from a real KID)
    bond_component: float           # PV of the guaranteed flows, with issuer credit
    derivative_component: float     # V - bond component
    credit_cost: float              # V0 - V
    closed_form_option: float | None    # option part priced in closed form, when the payoff allows it

    @property
    def embedded_costs(self) -> float:
        return self.issue_price - self.value

    @property
    def estimated_margin(self) -> float:
        """Embedded costs not explained by the documented fees (model estimate)."""
        return self.embedded_costs - self.documented_fees


def _guaranteed_flows(product: Product, market: Market) -> tuple[np.ndarray, np.ndarray]:
    """(times, amounts) of the flows paid whatever the underlying does, issuer solvent."""
    if isinstance(product, CapitalProtectedNote):
        return np.array([product.maturity]), np.array([product.protection])
    if isinstance(product, ReverseConvertible):
        times = np.array(product.observation_times())
        return times, np.full(len(times), product.coupon_rate/product.coupon_frequency)
    if isinstance(product, OptionPackage):
        cash = 1.0 - product.premium(market) - product.fees
        return np.array([product.maturity]), np.array([cash*math.exp(market.rate*product.maturity)])
    return np.array([product.maturity]), np.array([0.0])


def _closed_form_option(product: Product, market: Market) -> float | None:
    """Option part of the terminal structures by BSM, per unit of notional."""
    T, r, q, sigma = product.maturity, market.rate, market.dividend_yield, market.volatility
    if isinstance(product, CapitalProtectedNote):
        call = lambda k: bsm_price(True, 1.0, k, T, r, q, sigma)
        put = lambda k: bsm_price(False, 1.0, k, T, r, q, sigma)
        if product.bearish:
            floor = product.cap or 0.0
            value = put(product.strike) - (put(floor) if floor > 0 else 0.0)
        else:
            value = call(product.strike) - (call(product.cap) if product.cap is not None else 0.0)
        return product.participation*value
    if isinstance(product, ReverseConvertible) and product.barrier is None:
        return -bsm_price(False, 1.0, product.strike, T, r, q, sigma)/product.strike
    if isinstance(product, OptionPackage):
        return product.premium(market)
    return None


def analytic_value(product: Product, market: Market, spread: float = 0.0, recovery: float = 0.0) -> float | None:
    """Exact value (BSM) of the terminal structures, issuer credit included for notes; None if path-dependent."""
    T, r = product.maturity, market.rate
    option = _closed_form_option(product, market)
    if isinstance(product, DirectExposure):
        return 1.0
    if isinstance(product, OptionPackage):
        return 1.0 - product.fees
    if option is None:
        return None
    credit = (lambda t: survival(np.array([t]), spread, recovery)[0]) if product.issuer_risk else (lambda t: 1.0)
    if isinstance(product, CapitalProtectedNote):
        return (product.protection*math.exp(-r*T) + option)*credit(T)
    if isinstance(product, ReverseConvertible):
        coupons = sum(product.coupon_rate/product.coupon_frequency*math.exp(-r*t)*credit(t)
                      for t in product.observation_times())
        return coupons + (math.exp(-r*T) + option)*credit(T)
    return None


def value(product: Product, market_data: MarketData, settings: SimulationSettings = SimulationSettings(),
          documented_fees: float = 0.0, vol_shift: float = 0.0, rate_shift: float = 0.0) -> Valuation:
    market = market_data.pricer_market(vol_shift, rate_shift)
    times, paths = simulate(product, market, settings)
    flows = product.cash_flows(paths, times, market)
    rate = market.rate
    spread, recovery = market_data.credit_spread.value, market_data.recovery.value
    risky = present_values(flows, rate, spread, recovery) if product.issuer_risk else present_values(flows, rate)
    mc, error = mean_and_error(independent_draws(risky, settings.antithetic))
    mc_error = error
    mc0 = mean_and_error(independent_draws(present_values(flows, rate), settings.antithetic))[0]
    exact = analytic_value(product, market, spread, recovery)
    if exact is not None:
        v, v0, error, method = exact, analytic_value(product, market), 0.0, "closed form (BSM)"
    else:
        v, v0, method = mc, mc0, f"Monte Carlo ({settings.n_paths:,} paths)"
    g_times, g_amounts = _guaranteed_flows(product, market)
    factors = np.exp(-rate*g_times)*(survival(g_times, spread, recovery) if product.issuer_risk else 1.0)
    bond = float(g_amounts @ factors)
    return Valuation(product, v0, v, error, method, mc, mc_error, ISSUE_PRICE, documented_fees, bond, v - bond, v0 - v,
                     _closed_form_option(product, market))


SOLVABLE = {CapitalProtectedNote: "participation", ReverseConvertible: "coupon_rate", Autocallable: "coupon_rate"}


def solve_term(product: Product, market_data: MarketData, target_value: float,
               settings: SimulationSettings = SimulationSettings()) -> tuple[Product, float]:
    """
    The product with its coupon (or participation) set so that its issuer-adjusted model value
    equals `target_value` (issue price minus targeted costs). Returns (product, solved term).
    Raises ValueError if no non-negative term reaches the target (the structure cannot be
    built at that cost with these market parameters).
    """
    field_name = SOLVABLE.get(type(product))
    if field_name is None:
        raise ValueError(f"No term to solve for {type(product).__name__}.")
    market = market_data.pricer_market()
    times, paths = simulate(product, market, settings)    # same paths for both valuations
    spread, recovery = market_data.credit_spread.value, market_data.recovery.value

    def model_value(term):
        candidate = with_value(product, field_name, term)
        exact = analytic_value(candidate, market, spread, recovery)
        if exact is not None:
            return exact
        flows = candidate.cash_flows(paths, times, market)
        return float(present_values(flows, market.rate, spread, recovery).mean())

    base, unit = model_value(0.0), model_value(1.0)
    if unit - base <= 1e-12:
        raise ValueError("The term has no value in this structure.")
    term = (target_value - base)/(unit - base)
    if term < 0:
        raise ValueError(f"Even a zero {field_name.replace('_', ' ')} is worth more than the target: "
                         "the structure cannot be built at this cost (protection too expensive at these rates).")
    return with_value(product, field_name, term), term
