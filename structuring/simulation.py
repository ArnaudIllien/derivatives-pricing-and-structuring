"""
simulation.py - Paths of the underlying's performance, for valuation and for scenarios.

Two measures, never mixed:
- RISK-NEUTRAL (valuation): drift r - q, the market volatility. Prices are
  expectations under this measure. Its probabilities are not forecasts.
- SCENARIO (what-if analysis): a drift and a volatility stated explicitly, for
  example the investor's conviction (+10 % over 3 years, volatility +5 points).
  Used only for return distributions and loss probabilities, always shown with
  their hypotheses, never for prices.

Paths come from pricer.model.BlackScholesModel (exact GBM scheme), on a grid
holding every observation date of the product, plus every business day when a
barrier is monitored daily. Spot normalised to 1: the paths are performances.
"""
from dataclasses import dataclass

import numpy as np

from pricer.market import Market
from pricer.model import BlackScholesModel

from .products import DAYS_PER_YEAR, Product


@dataclass(frozen=True)
class SimulationSettings:
    n_paths: int = 20_000
    seed: int | None = 0
    antithetic: bool = True


def time_grid(product: Product, extra_times=()) -> np.ndarray:
    """0, the observation dates, the extra dates, and the business days if a barrier is monitored daily."""
    dates = list(product.observation_times()) + list(extra_times)
    if product.daily_monitoring:
        n_days = max(1, round(product.maturity*DAYS_PER_YEAR))
        dates += list(np.linspace(0, product.maturity, n_days + 1)[1:])
    return BlackScholesModel(Market(1.0, 0.0, 0.0)).time_grid(product.maturity, dates)


def simulate(product: Product, market: Market, settings: SimulationSettings = SimulationSettings(),
             drift: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    (times, paths) of the performance. drift=None: risk-neutral (r - q). Otherwise the
    real-world drift mu (continuous, per year) of a scenario: the paths then follow
    dP/P = mu dt + sigma dW, implemented by setting the carry to r - mu in the GBM.
    """
    times = time_grid(product)
    simulated = market if drift is None else Market(1.0, market.rate, market.volatility, market.rate - drift)
    paths = BlackScholesModel(Market(1.0, simulated.rate, simulated.volatility, simulated.dividend_yield)).simulate(
        times, settings.n_paths, np.random.default_rng(settings.seed), settings.antithetic)
    return times, paths


def path_from_levels(product: Product, anchor_times, anchor_levels) -> tuple[np.ndarray, np.ndarray]:
    """
    One deterministic path through given (time, level) points, log-linear in between, on the
    product's grid: used for the named path scenarios (crash then recovery, steady rise...).
    """
    times = time_grid(product, anchor_times)
    anchors_t = np.concatenate([[0.0], np.asarray(anchor_times, dtype=float)])
    anchors_x = np.concatenate([[0.0], np.log(np.maximum(np.asarray(anchor_levels, dtype=float), 1e-12))])
    return times, np.exp(np.interp(times, anchors_t, anchors_x))[None, :]


@dataclass(frozen=True)
class Regime:
    """From `start` to `end` (years), the log-performance has drift mu - sigma²/2 and volatility sigma."""
    start: float
    end: float
    drift: float
    volatility: float


def regimes_for_move(move: float, horizon: float, volatility: float, maturity: float,
                     later_volatility: float | None = None, later_move: float = 0.0) -> list[Regime]:
    """
    Median performance reaches 1 + move at `horizon`, then moves by a further `later_move`
    (0: flat median) until `maturity`; the volatility may change after the horizon.
    """
    later = volatility if later_volatility is None else later_volatility
    horizon = min(horizon, maturity)
    regimes = [Regime(0.0, horizon, drift_from_view(move, horizon, volatility), volatility)]
    if horizon < maturity:
        rest = maturity - horizon
        regimes.append(Regime(horizon, maturity, np.log(1 + later_move)/rest + later**2/2, later))
    return regimes


def simulate_regimes(product: Product, regimes: list[Regime], n_paths: int, seed: int | None = 0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Real-world paths with piecewise-constant drift and volatility, exact on each interval of the
    grid (the regime boundaries are added to it). Used by the scenario mixtures and the stress tests.
    """
    times = time_grid(product, [r.end for r in regimes if r.end < product.maturity])
    starts, dts = times[:-1], np.diff(times)
    mu, sigma = np.empty(len(dts)), np.empty(len(dts))
    for k, t in enumerate(starts):
        regime = next((r for r in regimes if r.start - 1e-12 <= t < r.end - 1e-12), regimes[-1])
        mu[k], sigma[k] = regime.drift, max(regime.volatility, 0.0)
    z = np.random.default_rng(seed).standard_normal((n_paths, len(dts)))
    increments = (mu - sigma**2/2)*dts + sigma*np.sqrt(dts)*z
    paths = np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1))
    return times, paths


def drift_from_view(expected_move: float, horizon: float, volatility: float) -> float:
    """
    Drift mu such that the MEDIAN performance after `horizon` years is 1 + expected_move:
    median of P_h = exp((mu - sigma²/2) h). Stated as a median so that a +10 % view means
    "half of the scenarios end above +10 %".
    """
    return np.log(1 + expected_move)/horizon + volatility**2/2
