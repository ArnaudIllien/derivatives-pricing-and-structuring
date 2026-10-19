"""
analysis.py - Runs every calculation for one candidate structure and gathers the results.

`analyse(candidate, profile, market_data)` returns a `ProductAnalysis`: valuation
sheet, maximum loss, sensitivities, maturity grid, named path scenarios,
distributions (investor scenario and risk-neutral), early-exit values and the
suitability checks. The interface and the comparator only read these objects.
"""
from dataclasses import dataclass

import numpy as np

from .investor import InvestorProfile
from .market import MarketData
from .risk import ExitValue, MaxLoss, Sensitivities, early_exit_values, max_contractual_loss, sensitivities
from .scenarios import Distribution, Outcome, maturity_grid, path_scenarios, return_distribution, view_distribution
from .simulation import SimulationSettings
from .strategies import Candidate
from .suitability import Check, check_product, verdict
from .valuation import Valuation, value


@dataclass(frozen=True)
class ProductAnalysis:
    candidate: Candidate
    market_data: MarketData
    valuation: Valuation
    max_loss: MaxLoss
    sensitivities: Sensitivities
    grid: list[Outcome]                  # straight paths to each final level
    grid_with_trough: list[Outcome]      # paths that first fall to 50 %
    scenarios: list[Outcome]
    investor_distribution: Distribution
    risk_neutral_distribution: Distribution
    exits: list[ExitValue]
    checks: list[Check]

    @property
    def product(self):
        return self.candidate.product

    @property
    def verdict(self) -> str:
        return verdict(self.checks)

    @property
    def best_case(self) -> tuple[float, bool]:
        """
        (highest total received on the maturity grid up to 200 %, True if the upside is capped).
        Capped means less than 20 % participation at the top of the grid: a collar still gains the
        dividends of the held underlying above its cap, which is not a real upside.
        """
        totals = [o.total_received for o in self.grid]
        levels = [o.final_level for o in self.grid]
        slope = (totals[-1] - totals[-5])/(levels[-1] - levels[-5])
        return max(totals), slope < 0.2

    @property
    def worst_exit_loss(self) -> float | None:
        values = [e.total_if_sold for e in self.exits if not np.isnan(e.indicative_bid)]
        return max(0.0, 1 - min(values)) if values else None


def analyse(candidate: Candidate, profile: InvestorProfile, market_data: MarketData,
            settings: SimulationSettings = SimulationSettings()) -> ProductAnalysis:
    product = candidate.product
    market = market_data.pricer_market()
    view = profile.view_on(market_data.underlying.name)
    valuation = value(product, market_data, settings, documented_fees=candidate.documented_fees)
    max_loss = max_contractual_loss(product, market, market_data.recovery.value)
    scenarios = path_scenarios(product, market, view.expected_move)
    investor = view_distribution(product, market, view)
    neutral = return_distribution(product, market, None, measure="Risk-neutral measure (drift r - q): market-implied, "
                                                                  "not a forecast")
    exits = early_exit_values(product, market_data)
    analysis = ProductAnalysis(candidate, market_data, valuation, max_loss, sensitivities(product, market_data, settings),
                               maturity_grid(product, market), maturity_grid(product, market, trough=0.5), scenarios,
                               investor, neutral, exits, [])
    checks = check_product(product, profile, max_loss=max_loss.contractual,
                           max_loss_with_default=max_loss.with_issuer_default,
                           currency=market_data.underlying.currency,
                           scenario_outcomes={o.label: o.total_received for o in scenarios},
                           conviction_summary=investor.summary(), worst_exit_loss=analysis.worst_exit_loss,
                           total_cost=valuation.embedded_costs)
    return ProductAnalysis(**{**analysis.__dict__, "checks": checks})
