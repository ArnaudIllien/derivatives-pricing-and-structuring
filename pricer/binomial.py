"""
binomial.py - Cox-Ross-Rubinstein binomial tree (European, American, Bermudan).

Used both as a pricing model in the client and as the independent reference
that validates BAW and LSM. It shares only the contract and the market data
with them, not their pricing logic.

Discrete dividends follow the exact drop model of market.py, with the method of
Vellekoop and Nieuwenhuis (2006). The tree stays recombining, and at a dividend
date the value before the drop is read on the tree by linear interpolation:
V_cum(S) = V_ex(S - D). Each dividend is moved to the nearest tree step, and
exercise at that step uses the cum-dividend spot.

The CRR price oscillates between even and odd numbers of steps; binomial_price
and bermudan_tree_price return the average of the trees with n and n + 1
steps, which smooths this out. The remaining error is of order 1/n.

The exercise style is a rule "can the holder exercise at step i?": always
(American), never (European), or at given steps (Bermudan).
"""
import math
from collections.abc import Callable

import numpy as np

from .contract import Option
from .market import Market


def _interpolate(nodes: np.ndarray, values: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Linear interpolation of values(nodes) at points, linear extrapolation outside. Nodes are decreasing."""
    x, y = nodes[::-1], values[::-1]
    if len(x) == 1:
        return np.full_like(points, y[0])
    i = np.clip(np.searchsorted(x, points), 1, len(x) - 1)
    return y[i - 1] + (y[i] - y[i - 1])*(points - x[i - 1])/(x[i] - x[i - 1])


def _crr_tree(option: Option, market: Market, n_steps: int, can_exercise: Callable[[int], bool]) -> float:
    dt = option.maturity/n_steps
    u = math.exp(market.volatility*math.sqrt(dt))
    d = 1/u
    p = (math.exp(market.carry*dt) - d)/(u - d)
    if not 0 < p < 1:
        raise ValueError("Risk-neutral probability outside (0, 1): increase the number of steps.")
    discount = math.exp(-market.rate*dt)

    dividend_steps = {}
    for dividend in market.dividends_before(option.maturity):
        step = min(max(round(dividend.time/dt), 1), n_steps - 1)
        dividend_steps[step] = dividend_steps.get(step, 0.0) + dividend.amount

    # node j at step i: S u^(i - 2j)
    values = option.payoff(market.spot*u**np.arange(n_steps, -n_steps - 1, -2.0))
    for i in range(n_steps - 1, -1, -1):
        values = discount*(p*values[:-1] + (1 - p)*values[1:])
        nodes = market.spot*u**np.arange(i, -i - 1, -2.0)
        if i in dividend_steps:
            ex_dividend_spot = np.maximum(nodes - dividend_steps[i], 0.0)
            values = np.maximum(_interpolate(nodes, values, ex_dividend_spot), 0.0)
        if can_exercise(i):
            values = np.maximum(values, option.payoff(nodes))
    return float(values[0])


def _check_tree(market: Market, n_steps: int):
    if market.volatility == 0:
        raise ValueError("The CRR tree is not defined for a zero volatility.")
    if n_steps < 2:
        raise ValueError("The tree needs at least 2 steps.")


def binomial_price(option: Option, market: Market, n_steps: int = 2000, american: bool = True) -> float:
    if option.maturity == 0:
        return option.payoff(market.spot)
    _check_tree(market, n_steps)
    rule = (lambda i: True) if american else (lambda i: False)
    return (_crr_tree(option, market, n_steps, rule) + _crr_tree(option, market, n_steps + 1, rule))/2


def bermudan_binomial_price(option: Option, market: Market, n_dates: int, steps_per_date: int = 40) -> float:
    """
    Option exercisable at t = 0 and at the dates m T / n_dates, m = 1..n_dates: the LSM grid
    when there are no discrete dividends.

    Comparing LSM with this price isolates the regression bias; comparing this
    price with the American price isolates the error from the discrete exercise dates.
    """
    if option.maturity == 0:
        return option.payoff(market.spot)
    return _crr_tree(option, market, n_dates*steps_per_date, lambda i: i % steps_per_date == 0)


def bermudan_tree_price(option: Option, market: Market, exercise_times, n_steps: int = 2000) -> float:
    """
    Option exercisable at `exercise_times` (years, in [0, T]) and at the maturity, the
    convention of lsm.bermudan_lsm_price. Each date is moved to the nearest tree step,
    a shift of at most T / (2 n_steps): an error of order 1/n, like the tree itself.
    """
    times = [float(t) for t in exercise_times]
    if any(t < 0 or t > option.maturity for t in times):
        raise ValueError(f"Every exercise date must lie between 0 and the maturity T={option.maturity}.")
    if option.maturity == 0:
        return option.payoff(market.spot)
    _check_tree(market, n_steps)

    def price(n):
        steps = {round(t/option.maturity*n) for t in times}
        return _crr_tree(option, market, n, steps.__contains__)
    return (price(n_steps) + price(n_steps + 1))/2
