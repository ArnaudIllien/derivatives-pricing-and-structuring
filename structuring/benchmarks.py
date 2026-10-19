"""
benchmarks.py - Does the complexity of a structure add value? Comparison with simple solutions.

For each structure, three simpler alternatives with the same maturity:
1. Direct exposure to the underlying.
2. A do-it-yourself option combination, without an issuer:
   - protected note  -> cash deposit + bought calls (puts if bearish) with the
     same protection: the deposit pays the protected amount at maturity and the
     rest of the capital, minus option fees, buys as many options as it can;
   - reverse convertible / autocallable (income from a sold option) -> buy-write:
     the underlying with a call sold at 110 %, the listed-market income strategy;
   - option packages are already simple: no further combination.
3. A diversified allocation with the SAME contractual maximum loss L: a share
   w = 1 - (1 - L) e^{-rT} in the underlying (or in a world equity ETF when its
   market data is given) and the rest in a risk-free deposit; at worst the
   deposit alone returns 1 - L.

All are evaluated under the investor's scenario (the world ETF under a flat
median: no view on it). An alternative in another currency is shown for
information but left out of the verdict (other rate, exchange-rate risk: not
comparable as such). Among the same-currency alternatives whose maximum loss is
no larger:
- one that does at least as well in expectation AND in the adverse case (10th
  percentile) dominates the structure: its complexity adds no value;
- otherwise the structure either expects more (value added), or expects less
  but protects the adverse case better: a trade-off, stated as such.
"""
import math
from dataclasses import dataclass

import pandas as pd

from .investor import ScenarioCase, ScenarioView
from .market import MarketData
from .products import (Autocallable, CapitalProtectedNote, DirectExposure, Leg, OptionPackage, Product,
                       ReverseConvertible)
from .risk import max_contractual_loss
from .scenarios import view_distribution
from .valuation import value


@dataclass(frozen=True)
class Alternative:
    label: str
    product: Product
    market_data: MarketData
    note: str


def diy_options(product: Product, market_data: MarketData, fees: float) -> Alternative | None:
    market = market_data.pricer_market()
    T = product.maturity
    if isinstance(product, CapitalProtectedNote):
        legs = [Leg("put" if product.bearish else "call", product.strike, 1.0)]
        if product.cap is not None:
            legs.append(Leg(legs[0].kind, product.cap, -1.0))
        unit = sum(leg.price(market, T) for leg in legs)
        budget = 1.0 - product.protection*math.exp(-market.rate*T) - fees
        if budget <= 0 or unit <= 0:
            return None
        quantity = budget/unit
        legs = tuple(Leg(leg.kind, leg.strike, leg.quantity*quantity) for leg in legs)
        return Alternative("DIY: deposit + listed options", OptionPackage(T, legs, fees, name="Deposit + options"),
                           market_data, f"same {product.protection:.0%} protection without issuer; participation "
                                        f"{quantity:.0%} vs {product.participation:.0%} in the note")
    if isinstance(product, ReverseConvertible | Autocallable):
        return Alternative("DIY: buy-write (underlying + call sold at 110 %)",
                           OptionPackage(T, (Leg("underlying"), Leg("call", 1.1, -1.0)), fees, name="Buy-write 110 %"),
                           market_data, "income from a sold option, listed, no issuer; full downside")
    return None


def mixed_allocation(max_loss: float, market_data: MarketData, maturity: float, label: str,
                     note: str) -> Alternative | None:
    rate = market_data.rate.value
    if max_loss >= 0.999:
        return None
    weight = 1 - (1 - max_loss)*math.exp(-rate*maturity)
    return Alternative(label, OptionPackage(maturity, (Leg("underlying", 1.0, weight),), 0.0,
                                            name=f"{weight:.0%} {market_data.underlying.name} + deposit"),
                       market_data, note)


def alternatives(product: Product, market_data: MarketData, fees: float = 0.005,
                 world: MarketData | None = None) -> list[Alternative]:
    rows = [Alternative("Direct exposure", DirectExposure(product.maturity), market_data, "the underlying itself")]
    diy = diy_options(product, market_data, fees)
    if diy:
        rows.append(diy)
    loss = max_contractual_loss(product, market_data.pricer_market()).contractual
    mix = mixed_allocation(loss, market_data, product.maturity, "Deposit + underlying, same max loss",
                           f"maximum loss {loss:.0%} like the structure")
    if mix:
        rows.append(mix)
    if world is not None:
        mix = mixed_allocation(loss, world, product.maturity, f"Deposit + {world.underlying.name}, same max loss",
                               "diversified equity: other underlying, currency and risks; no view on it")
        if mix:
            rows.append(mix)
    return rows


@dataclass(frozen=True)
class BenchmarkReport:
    table: pd.DataFrame
    verdict: str


def _row(label, product, market_data, view, note):
    d = view_distribution(product, market_data.pricer_market(), view, n_paths=10_000)
    v = value(product, market_data)
    return {"solution": label, "product": product.name, "currency": market_data.underlying.currency,
            "expected": d.expected_total - 1,
            "median": d.quantile(0.5) - 1, "adverse (10th pct)": d.quantile(0.1) - 1,
            "P(loss)": d.probability_of_loss,
            "max loss": max_contractual_loss(product, market_data.pricer_market()).contractual,
            "embedded costs": v.embedded_costs, "issuer risk": "yes" if product.issuer_risk else "no", "note": note}


def compare_with_simple(product: Product, market_data: MarketData, view: ScenarioView, fees: float = 0.005,
                        world: MarketData | None = None) -> BenchmarkReport:
    rows = [_row("The structure", product, market_data, view, "")]
    flat = ScenarioView(world.underlying.name if world else "", (ScenarioCase("flat", 1.0, 0.0, product.maturity),))
    for alt in alternatives(product, market_data, fees, world):
        alt_view = flat if alt.market_data is world else view
        rows.append(_row(alt.label, alt.product, alt.market_data, alt_view, alt.note))
    table = pd.DataFrame(rows)
    base = table.iloc[0]
    others = table.iloc[1:]
    # an alternative in another currency earns another rate and adds FX risk: shown, but not used in the verdict
    rivals = others[(others["max loss"] <= base["max loss"] + 1e-6) & (others["currency"] == base["currency"])]
    better = rivals[(rivals["expected"] >= base["expected"] - 1e-9) & (rivals["adverse (10th pct)"] >= base["adverse (10th pct)"] - 1e-9)]
    if len(better):
        best = better.sort_values("expected").iloc[-1]
        verdict = (f"A simpler solution does at least as well under your scenario: '{best['solution']}' (expected "
                   f"{best['expected']:+.1%}, adverse {best['adverse (10th pct)']:+.1%}) with a maximum loss of "
                   f"{best['max loss']:.0%} or less. The complexity does not add value here.")
    elif len(rivals):
        best = rivals.sort_values("expected").iloc[-1]
        if base["expected"] >= best["expected"]:
            verdict = (f"The structure adds value under your scenario: expected {base['expected']:+.1%} vs "
                       f"{best['expected']:+.1%} for the best simpler solution with no larger maximum loss "
                       f"('{best['solution']}'), at the price of its costs, issuer and liquidity risks.")
        else:
            verdict = (f"Trade-off, not a free gain: '{best['solution']}' expects more ({best['expected']:+.1%} vs "
                       f"{base['expected']:+.1%}) but has a worse adverse case ({best['adverse (10th pct)']:+.1%} vs "
                       f"{base['adverse (10th pct)']:+.1%}). The structure buys a better adverse case with expected "
                       "return, costs and issuer risk; worth it only if that adverse case matters most to you.")
    else:
        verdict = ("No simpler alternative has a maximum loss as small: the structure's protection is its value added; "
                   "check that its costs are worth it.")
    return BenchmarkReport(table, verdict)
