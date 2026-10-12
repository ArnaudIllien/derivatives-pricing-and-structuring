"""
greeks.py - Greeks by central finite differences, shared by every pricer.

`price_fn(option, market) -> float` can be any pricer: european_price,
baw_price, a binomial tree, or an LSM pricer with a fixed seed, for example
lambda o, m: lsm_price(o, m, LSMParameters(seed=0)).price.
With LSM, the fixed seed gives common random numbers to the bumped prices, but
the Greeks remain noisy, Gamma above all: use wider bumps (1 % of the spot,
for example).

Units: Vega is for a change of sigma of 1 (100 volatility points), and Theta
is the change in value per year as time passes (-dV/dT, with the spot held
fixed and the dividend dates moving closer).
"""
from collections.abc import Callable
from dataclasses import dataclass, replace

from .contract import Option
from .market import Market

PriceFunction = Callable[[Option, Market], float]


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float    # per unit of sigma
    theta: float   # per year


def compute_greeks(price_fn: PriceFunction, option: Option, market: Market,
                   spot_bump: float = 1e-3, vol_bump: float = 1e-3, time_bump: float = 1/365) -> Greeks:
    """spot_bump is relative to the spot; vol_bump and time_bump are absolute."""
    v0 = price_fn(option, market)
    h = market.spot*spot_bump
    v_up = price_fn(option, replace(market, spot=market.spot + h))
    v_down = price_fn(option, replace(market, spot=market.spot - h))
    delta = (v_up - v_down)/(2*h)
    gamma = (v_up - 2*v0 + v_down)/h**2

    vol = market.volatility
    if vol > vol_bump:
        vega = (price_fn(option, replace(market, volatility=vol + vol_bump))
                - price_fn(option, replace(market, volatility=vol - vol_bump)))/(2*vol_bump)
    else:
        vega = (price_fn(option, replace(market, volatility=vol + vol_bump)) - v0)/vol_bump

    dt = min(time_bump, option.maturity)
    if dt == 0:
        theta = 0.0
    else:
        later = price_fn(replace(option, maturity=option.maturity - dt), market.shifted(dt))
        theta = (later - v0)/dt
    return Greeks(delta, gamma, vega, theta)
