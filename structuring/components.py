"""
components.py - Decomposition of each structure into its building blocks, valued one by one.

What a structurer shows to explain a price: the bond, the options bought, the
options sold, the coupon strips. Values are per unit of notional, at inception,
issuer credit included for the notes (the issuer owes every block).

- Closed forms (BSM, pricer.european) for the vanilla blocks.
- Path-dependent blocks are isolated by LINEARITY on the same Monte Carlo paths:
  coupon strip = V(coupon c) - V(coupon 0); capital-loss block (the sold
  down-and-in put) = V(coupon 0) - V(coupon 0, no capital loss possible). Their
  sum is the structure's value exactly (same paths, no extra noise).
"""
import math
from dataclasses import dataclass, replace

import numpy as np

from pricer.european import bsm_price

from .market import MarketData
from .products import (Autocallable, CapitalProtectedNote, DirectExposure, OptionPackage, Product, ReverseConvertible)
from .simulation import SimulationSettings, simulate
from .valuation import present_values, survival


@dataclass(frozen=True)
class Component:
    name: str
    position: str        # "long", "short" or "-"
    value: float         # per unit of notional
    method: str


def _mc_value(product, market_data, market, times, paths) -> float:
    flows = product.cash_flows(paths, times, market)
    return float(present_values(flows, market.rate, market_data.credit_spread.value,
                                market_data.recovery.value).mean())


def decompose(product: Product, market_data: MarketData,
              settings: SimulationSettings = SimulationSettings()) -> list[Component]:
    market = market_data.pricer_market()
    T, r, q, sigma = product.maturity, market.rate, market.dividend_yield, market.volatility
    spread, recovery = market_data.credit_spread.value, market_data.recovery.value
    credit_T = survival(np.array([T]), spread, recovery)[0] if product.issuer_risk else 1.0
    call = lambda k: bsm_price(True, 1.0, k, T, r, q, sigma)
    put = lambda k: bsm_price(False, 1.0, k, T, r, q, sigma)

    if isinstance(product, DirectExposure):
        return [Component("Underlying, income reinvested", "long", 1.0, "spot")]

    if isinstance(product, OptionPackage):
        rows = [Component(f"{leg.kind} {leg.strike:.0%}" if leg.kind != "underlying" else "underlying",
                          "long" if leg.quantity > 0 else "short", leg.price(market, T),
                          "spot" if leg.kind == "underlying" else "BSM") for leg in product.legs]
        cash = 1.0 - product.premium(market) - product.fees
        return rows + [Component("Cash deposit", "long", cash, "risk-free"),
                       Component("Fees", "-", product.fees, "hypothetical input")]

    if isinstance(product, CapitalProtectedNote):
        rows = [Component(f"Zero-coupon bond paying {product.protection:.0%}", "long",
                          product.protection*math.exp(-r*T)*credit_T, "discounting + issuer credit")]
        p, k = product.participation, product.strike
        if product.bearish:
            rows.append(Component(f"{p:.2f} x put {k:.0%}", "long", p*put(k)*credit_T, "BSM"))
            if product.cap:
                rows.append(Component(f"{p:.2f} x put {product.cap:.0%}", "short", -p*put(product.cap)*credit_T, "BSM"))
        else:
            rows.append(Component(f"{p:.2f} x call {k:.0%}", "long", p*call(k)*credit_T, "BSM"))
            if product.cap:
                rows.append(Component(f"{p:.2f} x call {product.cap:.0%}", "short", -p*call(product.cap)*credit_T,
                                      "BSM"))
        return rows

    if isinstance(product, ReverseConvertible) and product.barrier is None:   # terminal: closed forms
        dates = np.array(product.observation_times())
        factors = np.exp(-r*dates)*(survival(dates, spread, recovery) if product.issuer_risk else 1.0)
        coupons = float(np.sum(product.coupon_rate/product.coupon_frequency*factors))
        return [Component("Zero-coupon bond paying 100 %", "long", math.exp(-r*T)*credit_T, "discounting + issuer credit"),
                Component(f"Fixed coupons {product.coupon_rate:.2%}/yr", "long", coupons, "discounting + issuer credit"),
                Component(f"Put {product.strike:.0%} (1/strike units)", "short",
                          -put(product.strike)/product.strike*credit_T, "BSM")]

    times, paths = simulate(product, market, settings)
    coupon_field = "coupon_rate"
    with_coupon = _mc_value(product, market_data, market, times, paths)
    no_coupon = _mc_value(replace(product, **{coupon_field: 0.0}), market_data, market, times, paths)
    if isinstance(product, ReverseConvertible):
        bond = math.exp(-r*T)*credit_T
        vanilla = -put(product.strike)/product.strike*credit_T
        return [Component("Zero-coupon bond paying 100 %", "long", bond, "discounting + issuer credit"),
                Component(f"Fixed coupons {product.coupon_rate:.2%}/yr", "long", with_coupon - no_coupon,
                          "discounting + issuer credit"),
                Component(f"Down-and-in put {product.strike:.0%}, barrier {product.barrier:.0%} "
                          f"({product.barrier_monitoring})", "short", no_coupon - bond, "Monte Carlo (same paths)"),
                Component("  for reference: vanilla put at the strike", "short", vanilla, "BSM")]

    if isinstance(product, Autocallable):
        safe = _mc_value(replace(product, coupon_rate=0.0, capital_barrier=1e-9, barrier_monitoring="maturity"),
                         market_data, market, times, paths)
        return [Component("Par redemption (early or at maturity): autocallable zero-coupon", "long", safe,
                          "Monte Carlo (same paths)"),
                Component(f"Conditional coupons {product.coupon_rate:.2%}/yr (digital strip"
                          + (", memory" if product.memory else "") + ")", "long", with_coupon - no_coupon,
                          "Monte Carlo (same paths)"),
                Component(f"Down-and-in put at {product.capital_barrier:.0%} if not called", "short", no_coupon - safe,
                          "Monte Carlo (same paths)")]
    return []


MODEL_LIMITS = {
    "Direct exposure": ["Income reinvested at the carry yield; no transaction costs or taxes."],
    "Capital protection": ["Protection costs depend on the long rate: a flat curve at the maturity yield is used.",
                           "The issuer spread is hypothetical and moves the bond part directly.",
                           "Flat volatility: a real skew changes the value of the capped call (call spread)."],
    "Reverse convertible": ["The sold put is valued at the ATM volatility; with skew, the downside volatility is "
                            "higher, so the real put is worth more and the fair coupon is lower.",
                            "Daily barrier observed on 252 business days a year; continuous monitoring would hit more.",
                            "Physical delivery replaced by its cash equivalent."],
    "Autocallable": ["Highly sensitive to the forward (rates, dividends) and to the skew between the trigger and the "
                     "capital barrier; flat volatility ignores it.",
                     "Uncertain duration: the early-redemption probability depends on the model drift (risk-neutral).",
                     "No stochastic volatility or jumps: the gap risk around the barrier is understated."],
    "Option strategy": ["Options valued with BSM at a single volatility, no bid-ask beyond the fee input.",
                        "Listed options rarely exceed 2-3 years: long maturities need OTC options (counterparty risk)."],
}
