"""
strategies.py - Candidate structures built from the investor's described scenario and constraints.

The generator always proposes SEVERAL architectures, including the simplest
ones, and never filters them on expected return: suitability.py then explains
which fit the profile. Each candidate states why the scenario makes it relevant.

How the scenario is translated (ScenarioView: weighted cases)
--------------------------------------------------------------
- Probability of a rise > 0: protected notes with upside participation and a
  call spread up to the largest expected rise, with the maturity of that case
  when it is shorter than the investment horizon ("+8 % within six months"
  calls for a six-month spread, not a five-year note).
- Probability of a fall larger than of a rise: bearish protected notes and a
  put spread down to the largest expected fall.
- Probability of stagnation (|move| <= 3 %) or of a moderate rise: autocallable
  and barrier reverse convertible, which pay for a market that does not fall
  much; their rationale quotes that probability.
- Expected volatility change: falling volatility favours selling options
  (reverse convertible, autocallable), rising volatility favours buying them
  (protected notes, spreads); the rationale says which applies.

How the terms are set (the structurer's logic)
----------------------------------------------
- Notes: the coupon or participation is SOLVED so that the issuer-adjusted model
  value equals the issue price (100 %) minus the targeted costs, which are
  HYPOTHETICAL (default 1 % per year) unless taken from a real term sheet.
- Option packages: sized so that premium + fees equal the loss budget (the
  declared maximum loss, or the interest earned by the cash if no loss is
  accepted: the capital is then preserved at maturity without issuer).
- Direct exposure is always included as the reference.
"""
import math
from dataclasses import dataclass

from .investor import Conviction, InvestorProfile, ScenarioView
from .market import MarketData
from .products import (Autocallable, CapitalProtectedNote, DirectExposure, Product, ReverseConvertible, call_spread,
                       collar, put_spread)
from .simulation import SimulationSettings
from .valuation import ISSUE_PRICE, solve_term


@dataclass(frozen=True)
class CostAssumptions:
    note_fees_per_year: float = 0.01     # embedded in the issue price of notes; hypothetical default
    package_fees: float = 0.005          # commissions and bid-ask on options, upfront; hypothetical default
    documented: bool = False             # True when the user took them from a real term sheet / KID

    def note_target(self, maturity: float) -> float:
        return ISSUE_PRICE - self.note_fees_per_year*maturity

    def note_fees(self, maturity: float) -> float:
        return self.note_fees_per_year*maturity


@dataclass(frozen=True)
class Candidate:
    product: Product
    rationale: str
    solved: str = ""              # which term was solved, and its value
    documented_fees: float = 0.0  # fees counted in the valuation sheet


def product_maturity(profile: InvestorProfile) -> float:
    years = min(profile.objectives.horizon, profile.constraints.max_lockup_years)
    return float(max(1, math.floor(years + 1e-9)))


def _budget(profile: InvestorProfile, market_data: MarketData, maturity: float) -> float:
    if profile.constraints.max_loss_fraction > 0:
        return min(profile.constraints.max_loss_fraction, 0.5)
    return 1 - math.exp(-market_data.rate.value*maturity)


def _spread_maturity(view: ScenarioView, sign: int, horizon: float) -> float:
    """Horizon of the main case in that direction, in quarters, if shorter than the investment horizon."""
    cases = [c for c in view.cases if sign*c.move > 0.03]
    if not cases:
        return horizon
    case = max(cases, key=lambda c: c.probability)
    return min(horizon, max(0.25, round(case.horizon*4)/4))


def generate_candidates(profile: InvestorProfile, view: ScenarioView | Conviction, market_data: MarketData,
                        costs: CostAssumptions = CostAssumptions(),
                        settings: SimulationSettings = SimulationSettings()) -> tuple[list[Candidate], list[str]]:
    """(candidates, structures that could not be built and why)."""
    if isinstance(view, Conviction):
        view = ScenarioView.from_conviction(view)
    T = product_maturity(profile)
    market = market_data.pricer_market()
    p_up, p_down, p_flat = view.probability("up"), view.probability("down"), view.probability("flat")
    vol = view.expected_vol_change
    vol_note = ("your falling-volatility view favours selling options" if vol < -0.005 else
                "your rising-volatility view favours buying options" if vol > 0.005 else "no volatility view")
    candidates, skipped = [Candidate(DirectExposure(T), "Reference: the underlying itself, no structure, no "
                                                       "embedded costs, full upside and downside.")], []
    target, fees = costs.note_target(T), costs.note_fees(T)

    def solved(product, rationale, label):
        try:
            built, term = solve_term(product, market_data, target, settings)
            candidates.append(Candidate(built, rationale, f"{label} solved: {term:.2%}", fees))
        except ValueError as error:
            skipped.append(f"{product.name}: {error}")

    bearish = p_down > p_up
    side = "downside" if bearish else "upside"
    solved(CapitalProtectedNote(T, protection=1.0, bearish=bearish,
                                name="100 % capital-protected note" + (" (bearish)" if bearish else "")),
           f"Capital protected at maturity (issuer risk) with {side} participation for your "
           f"{(p_down if bearish else p_up):.0%} probability of a {'fall' if bearish else 'rise'}; {vol_note}.",
           "participation")
    solved(CapitalProtectedNote(T, protection=0.9, bearish=bearish,
                                name="90 % partially protected note" + (" (bearish)" if bearish else "")),
           "Gives up 10 % of protection to buy more participation.", "participation")
    if p_flat + p_up >= 0.5 and not bearish:
        solved(Autocallable(T, 0.0, name="Phoenix autocallable"),
               f"Pays conditional coupons when the market does not fall much: your scenario gives "
               f"{p_flat:.0%} to stagnation and {p_up:.0%} to a rise; early redemption at or above the initial "
               f"level; capital at risk below 60 % at maturity; {vol_note}.", "coupon")
        solved(ReverseConvertible(T, 0.0, barrier=0.6, name="Barrier reverse convertible 60 %"),
               f"Fixed coupons whatever the market does above 60 %: suited to the {p_flat:.0%} stagnation probability; "
               "the investor sells a down-and-in put and bears the losses below 60 % at maturity.", "coupon")

    budget = _budget(profile, market_data, T)
    for sign, probability, build in ((1, p_up, call_spread), (-1, p_down, put_spread)):
        if probability < 0.25 or (sign > 0 and bearish) or (sign < 0 and not bearish):
            continue
        target_move = view.upside_target if sign > 0 else view.downside_target
        width = max(abs(target_move), 0.05)
        maturity = _spread_maturity(view, sign, T)
        strikes = (1.0, 1.0 + width) if sign > 0 else (1.0, max(1.0 - width, 0.05))
        unit = build(maturity, *strikes)
        unit_premium = unit.premium(market)
        quantity = (budget - costs.package_fees)/unit_premium if unit_premium > 0 else 0.0
        if quantity <= 0:
            skipped.append(f"{unit.name}: the budget does not cover the fees.")
            continue
        package = build(maturity, *strikes, quantity=quantity, fees=costs.package_fees)
        horizon_note = (f" Its maturity ({maturity:g}y) follows your scenario horizon, not the {T:g}-year investment "
                        "horizon: the rest of the period is not invested." if maturity < T else "")
        candidates.append(Candidate(package, f"Option spread up to your largest expected {'rise' if sign > 0 else 'fall'} "
                                             f"({target_move:+.0%}), sized so that premium and fees equal {budget:.1%} of "
                                             f"the capital, the most the investor can lose.{horizon_note}",
                                    f"size: {quantity:.2f} spreads per unit", costs.package_fees))
    if not bearish:
        candidates.append(Candidate(collar(T, 0.9, 1.1, fees=costs.package_fees),
                                    "Holds the underlying with a 90 % put financed by selling a 110 % call: limited "
                                    "loss and limited gain, no issuer.", "", costs.package_fees))
    return candidates, skipped
