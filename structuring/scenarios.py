"""
scenarios.py - What the investor receives in given market situations.

1. Maturity grid (`maturity_grid`): the result for final levels from 0 % to
   200 % of the initial level, on a path that goes straight (log-linearly) to
   that level, and, for path-dependent structures, on a path that first dips to
   a given trough. Barriers and autocalls therefore follow the PATH, not only
   the final level.
2. Named path scenarios (`path_scenarios`): steady rise to the conviction,
   stagnation, moderate decline, crash with partial recovery, V-shape, and a
   strong rise. Deterministic paths, so each result is exact for that path.
3. Distributions: Monte Carlo under a stated measure. `view_distribution` is
   the investor's described scenario, a mixture of its cases (each on a share
   of the paths equal to its subjective probability, median move reached at
   the case horizon, flat afterwards, volatility shifted by the case);
   `return_distribution` is risk-neutral or any stated drift. Gives the favourable (90th
   percentile), central (median) and adverse (10th percentile) outcomes and the
   probability of loss UNDER THOSE HYPOTHESES ONLY.

Results are per investor: gain or loss in currency and in % of the capital,
total of the flows received (coupons not reinvested), annualised return over
the actual holding period (an autocall can end early), issuer solvent.
"""
from dataclasses import dataclass

import numpy as np

from pricer.market import Market

from .products import Product
from .simulation import (SimulationSettings, drift_from_view, path_from_levels, regimes_for_move, simulate,
                         simulate_regimes)

VOL_VIEW_SHIFT = {"rising": 0.05, "stable": 0.0, "falling": -0.03}   # absolute volatility change, hypothesis


@dataclass(frozen=True)
class Outcome:
    label: str
    final_level: float
    total_received: float        # sum of flows, fraction of capital
    end_time: float              # years until the last flow
    description: str = ""

    @property
    def gain(self) -> float:
        return self.total_received - 1.0

    @property
    def annualised(self) -> float:
        if self.total_received <= 0:
            return -1.0
        return self.total_received**(1/self.end_time) - 1


def outcome_on_path(product: Product, market: Market, times, path, label: str, description: str = "") -> Outcome:
    flows = product.cash_flows(path, times, market)
    return Outcome(label, float(path[0, -1]), float(flows.total[0]), float(flows.end_time[0]), description)


def maturity_grid(product: Product, market: Market, levels=None, trough: float | None = None) -> list[Outcome]:
    """Outcomes for final levels on straight paths, or on paths that first fall to `trough` at mid-life."""
    levels = np.linspace(0.0, 2.0, 81) if levels is None else np.asarray(levels)
    outcomes = []
    for level in levels:
        if trough is None:
            times, path = path_from_levels(product, [product.maturity], [max(level, 1e-12)])
        else:
            times, path = path_from_levels(product, [product.maturity/2, product.maturity],
                                           [min(trough, level), max(level, 1e-12)])
        outcomes.append(outcome_on_path(product, market, times, path, f"{level:.0%}"))
    return outcomes


def path_scenarios(product: Product, market: Market, conviction_move: float) -> list[Outcome]:
    T = product.maturity
    view = max(conviction_move, -0.95)
    definitions = [
        ("Conviction realised", [T], [1 + view], f"steady move to {view:+.0%} at maturity"),
        ("Stagnation", [T], [1.0], "flat market"),
        ("Moderate decline", [T], [0.85], "steady fall to -15 %"),
        ("Crash then partial recovery", [min(0.25, T/4), T], [0.55, 0.75], "-45 % within 3 months, -25 % at maturity"),
        ("V-shape", [T/2, T], [0.65, 1.05], "-35 % at mid-life, back to +5 %"),
        ("Strong rise", [T], [1.40], "steady rise to +40 %"),
    ]
    outcomes = []
    for label, anchor_times, anchor_levels, description in definitions:
        times, path = path_from_levels(product, anchor_times, anchor_levels)
        outcomes.append(outcome_on_path(product, market, times, path, label, description))
    return outcomes


@dataclass(frozen=True)
class Distribution:
    measure: str                 # description of the hypotheses
    drift: float
    volatility: float
    totals: np.ndarray           # total received per path
    end_times: np.ndarray

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.totals, q))

    @property
    def probability_of_loss(self) -> float:
        return float((self.totals < 1.0 - 1e-12).mean())

    @property
    def expected_total(self) -> float:
        return float(self.totals.mean())

    def expected_shortfall(self, level: float = 0.05) -> float:
        """Average total received in the worst `level` fraction of the scenarios."""
        cutoff = np.quantile(self.totals, level)
        return float(self.totals[self.totals <= cutoff].mean())

    def summary(self) -> dict[str, float]:
        return {"favourable (90th pct)": self.quantile(0.9) - 1, "central (median)": self.quantile(0.5) - 1,
                "adverse (10th pct)": self.quantile(0.1) - 1, "expected": self.expected_total - 1,
                "probability of loss": self.probability_of_loss, "average of worst 5 %": self.expected_shortfall() - 1}


def return_distribution(product: Product, market: Market, drift: float | None, volatility: float | None = None,
                        measure: str = "", settings: SimulationSettings = SimulationSettings(n_paths=20_000, seed=7)
                        ) -> Distribution:
    """drift=None: risk-neutral (r - q). The volatility of the scenario may differ from the market one."""
    scenario_market = market if volatility is None else Market(1.0, market.rate, volatility, market.dividend_yield)
    times, paths = simulate(product, scenario_market, settings, drift)
    flows = product.cash_flows(paths, times, market)    # terms priced with the MARKET (e.g. package premium)
    used_drift = scenario_market.rate - scenario_market.dividend_yield if drift is None else drift
    return Distribution(measure, used_drift, scenario_market.volatility, flows.total, flows.end_time)


def view_distribution(product: Product, market: Market, view, n_paths: int = 20_000, seed: int = 11) -> Distribution:
    """
    The investor's described scenario: a MIXTURE of its cases, each simulated on a share of the
    paths equal to its subjective probability (median move reached at the case horizon, flat
    median afterwards, volatility = market volatility + the case's change).
    """
    totals, end_times = [], []
    for i, case in enumerate(view.cases):
        n = int(round(case.probability*n_paths))
        if n == 0:
            continue
        sigma = max(market.volatility + case.vol_change, 0.01)
        times, paths = simulate_regimes(product, regimes_for_move(case.move, case.horizon, sigma, product.maturity),
                                        n, seed + i)
        flows = product.cash_flows(paths, times, market)
        totals.append(flows.total)
        end_times.append(flows.end_time)
    measure = f"Your scenario (subjective probabilities): {view.describe()}. Flat median after each horizon."
    return Distribution(measure, float("nan"), market.volatility, np.concatenate(totals), np.concatenate(end_times))


def conviction_distribution(product: Product, market: Market, expected_move: float, horizon: float,
                            volatility_view: str) -> Distribution:
    """The investor's scenario: median move over the conviction horizon, volatility shifted by the view."""
    sigma = max(market.volatility + VOL_VIEW_SHIFT[volatility_view], 0.01)
    mu = drift_from_view(expected_move, horizon, sigma)
    measure = (f"Investor scenario: median move {expected_move:+.0%} over {horizon:g} years "
               f"(drift {mu:+.2%}/year), volatility {sigma:.1%} ({volatility_view} view)")
    return return_distribution(product, market, mu, sigma, measure)
