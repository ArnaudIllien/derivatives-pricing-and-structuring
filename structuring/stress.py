"""
stress.py - Stress test of the investor's convictions: what if they are right, and what if they are wrong.

The base case is the most probable case of the investor's scenario (move m
within horizon h). Each stress keeps or breaks one part of that view:

- Your scenario                the full mixture of cases, with their subjective probabilities;
- Right, volatility explodes   move m within h, but volatility +15 points for the whole life;
- Right, but slower            move m reached only at maturity;
- Right, then reversal         move m within h, then down to -15 % at maturity;
- Stagnation for the whole life   median flat, volatility unchanged;
- Wrong direction              move -m within h;
- Crash                        -35 % within 3 months with volatility +15 points for a year,
                               then flat median at the market volatility.

Each stress is simulated (real-world GBM with piecewise drift and volatility,
simulation.simulate_regimes), not a single path: results are distributions
(median, 10th percentile, probability of loss), issuer solvent. They are
what-ifs under stated hypotheses, not forecasts.
"""
from dataclasses import dataclass

import numpy as np

from pricer.market import Market

from .investor import ScenarioView
from .products import Product
from .scenarios import Distribution, view_distribution
from .simulation import Regime, regimes_for_move, simulate_regimes

VOL_SHOCK = 0.15


@dataclass(frozen=True)
class StressCase:
    label: str
    description: str


@dataclass(frozen=True)
class StressResult:
    case: StressCase
    distribution: Distribution

    @property
    def median(self) -> float:
        return self.distribution.quantile(0.5) - 1

    @property
    def adverse(self) -> float:
        return self.distribution.quantile(0.1) - 1

    @property
    def probability_of_loss(self) -> float:
        return self.distribution.probability_of_loss


def base_case(view: ScenarioView):
    return max(view.cases, key=lambda c: c.probability)


def stress_regimes(view: ScenarioView, volatility: float, maturity: float) -> list[tuple[StressCase, list[Regime]]]:
    base = base_case(view)
    m, h = base.move, min(base.horizon, maturity)
    sigma = max(volatility + base.vol_change, 0.01)
    reversal = (0.85/(1 + m)) - 1 if m > -1 else 0.0
    crash_end = min(0.25, maturity)
    calm_after = min(1.0, maturity)
    crash = [Regime(0.0, crash_end, np.log(0.65)/crash_end + (volatility + VOL_SHOCK)**2/2, volatility + VOL_SHOCK)]
    if calm_after > crash_end:
        crash.append(Regime(crash_end, calm_after, (volatility + VOL_SHOCK)**2/2, volatility + VOL_SHOCK))
    if maturity > calm_after:
        crash.append(Regime(calm_after, maturity, volatility**2/2, volatility))
    return [
        (StressCase("Right, volatility explodes", f"{m:+.0%} within {h:g}y, volatility +15 pts all along"),
         regimes_for_move(m, h, sigma + VOL_SHOCK, maturity)),
        (StressCase("Right, but slower", f"{m:+.0%} reached only at maturity ({maturity:g}y)"),
         regimes_for_move(m, maturity, sigma, maturity)),
        (StressCase("Right, then reversal", f"{m:+.0%} within {h:g}y, then down to -15 % at maturity"),
         regimes_for_move(m, h, sigma, maturity, later_move=reversal)),
        (StressCase("Stagnation for the whole life", "flat median, market volatility"),
         regimes_for_move(0.0, maturity, volatility, maturity)),
        (StressCase("Wrong direction", f"{-m:+.0%} within {h:g}y"),
         regimes_for_move(max(-m, -0.95), h, sigma, maturity)),
        (StressCase("Crash", "-35 % within 3 months, volatility +15 pts for a year, then flat"), crash),
    ]


def stress_test(product: Product, market: Market, view: ScenarioView, n_paths: int = 8_000,
                seed: int = 21) -> list[StressResult]:
    results = [StressResult(StressCase("Your scenario", view.describe()),
                            view_distribution(product, market, view, n_paths, seed))]
    for i, (case, regimes) in enumerate(stress_regimes(view, market.volatility, product.maturity)):
        times, paths = simulate_regimes(product, regimes, n_paths, seed + 1 + i)
        flows = product.cash_flows(paths, times, market)
        results.append(StressResult(case, Distribution(case.description, float("nan"), market.volatility,
                                                       flows.total, flows.end_time)))
    return results
