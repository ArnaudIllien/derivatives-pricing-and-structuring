"""
lab.py - Structuring laboratory: build a structure from editable terms and see their effect at once.

`build(family, terms)` turns a dictionary of terms (as edited in the interface)
into a product; `lab_report(product, market_data, view, target)` gives what a
structurer looks at after each change: model value with issuer credit, embedded
costs versus the issue price, the fair coupon or participation for the target,
the contractual maximum loss, the payoff grid, the result under the investor's
scenario and the main sensitivities. Fewer paths than the full analysis keep
it interactive (the standard error is reported).
"""
from dataclasses import dataclass

from .investor import ScenarioView
from .market import MarketData
from .products import (Autocallable, CapitalProtectedNote, DirectExposure, Leg, OptionPackage, Product,
                       ReverseConvertible)
from .risk import max_contractual_loss, sensitivities
from .scenarios import Outcome, maturity_grid, view_distribution
from .simulation import SimulationSettings
from .valuation import SOLVABLE, Valuation, solve_term, value

LAB_SETTINGS = SimulationSettings(n_paths=10_000)
FAMILIES = ("Capital-protected note", "Reverse convertible", "Autocallable", "Option package", "Direct exposure")

DEFAULT_TERMS = {
    "Capital-protected note": {"maturity": 5.0, "protection": 1.0, "participation": 0.8, "strike": 1.0, "cap": None,
                               "bearish": False},
    "Reverse convertible": {"maturity": 2.0, "coupon_rate": 0.06, "coupon_frequency": 1, "barrier": 0.6,
                            "barrier_monitoring": "maturity", "strike": 1.0},
    "Autocallable": {"maturity": 5.0, "coupon_rate": 0.06, "observation_frequency": 1, "autocall_trigger": 1.0,
                     "coupon_barrier": 0.7, "memory": True, "capital_barrier": 0.6, "barrier_monitoring": "maturity",
                     "first_call": 1},
    "Option package": {"maturity": 1.0, "legs": [("call", 1.0, 1.0), ("call", 1.2, -1.0)], "fees": 0.005},
    "Direct exposure": {"maturity": 5.0},
}


def build(family: str, terms: dict) -> Product:
    t = dict(terms)
    if family == "Capital-protected note":
        return CapitalProtectedNote(name="Lab: protected note", **t)
    if family == "Reverse convertible":
        return ReverseConvertible(name="Lab: reverse convertible", **t)
    if family == "Autocallable":
        return Autocallable(name="Lab: autocallable", **t)
    if family == "Option package":
        legs = tuple(Leg(kind, float(strike), float(quantity)) for kind, strike, quantity in t.pop("legs"))
        return OptionPackage(t["maturity"], legs, t.get("fees", 0.0), name="Lab: option package")
    if family == "Direct exposure":
        return DirectExposure(t["maturity"], name="Lab: direct exposure")
    raise ValueError(f"Unknown family {family!r}.")


@dataclass(frozen=True)
class LabReport:
    valuation: Valuation
    fair_term: tuple[str, float] | None      # (term name, value giving the target), None if not solvable
    fair_term_error: str
    max_loss: float
    grid: list[Outcome]
    median: float
    adverse: float
    probability_of_loss: float
    vega: float
    rho: float


def lab_report(product: Product, market_data: MarketData, view: ScenarioView, target: float) -> LabReport:
    v = value(product, market_data, LAB_SETTINGS)
    fair, error = None, ""
    field_name = SOLVABLE.get(type(product))
    if field_name:
        try:
            fair = (field_name, solve_term(product, market_data, target, LAB_SETTINGS)[1])
        except ValueError as exc:
            error = str(exc)
    market = market_data.pricer_market()
    d = view_distribution(product, market, view, n_paths=8_000)
    s = sensitivities(product, market_data, LAB_SETTINGS)
    return LabReport(v, fair, error, max_contractual_loss(product, market).contractual,
                     maturity_grid(product, market, [i/20 for i in range(0, 41)]), d.quantile(0.5) - 1,
                     d.quantile(0.1) - 1, d.probability_of_loss, s.vega, s.rho)
