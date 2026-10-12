"""
baw.py - Barone-Adesi and Whaley (1987) analytic approximation for American options.

Principle
---------
The American price is split as V = v_eur + e, where e is the early-exercise
premium. The premium satisfies the same Black-Scholes PDE as V. BAW writes
e = K(T) f(S, K(T)), with K(T) = 1 - e^{-rT}, and drops the term in the
derivative of f with respect to K(T). What is left is an ODE in S, whose
bounded solution is A (S / S*)^lambda, where:

    M = 2r / sigma², N = 2(r - q) / sigma²
    lambda = [-(N - 1) + sign * sqrt((N - 1)² + 4M / K(T))] / 2
    (q2 > 1 for the call, q1 < 0 for the put)

The boundary S* and the coefficient A come from the smooth-pasting conditions:
- value matching:  v_eur(S*) + A = sign (S* - K);
- slope matching:  e^{-qT} sign N(sign d1(S*)) + A lambda / S* = sign.

This gives the scalar equation in S*, solved with Brent's method:

    sign (S* - K) = v_eur(S*) + sign [1 - e^{-qT} N(sign d1(S*))] S* / lambda

and A = sign (S* / lambda) [1 - e^{-qT} N(sign d1(S*))].

Regions
-------
- Call: exercise if S >= S*, otherwise V = c(S) + A2 (S / S*)^q2.
- Put:  exercise if S <= S**, otherwise V = p(S) + A1 (S / S**)^q1.

Assumptions and limits
----------------------
- This is an APPROXIMATION, not an exact solution. Against a fine binomial tree
  (see diagnostics.py), the gap is about 0.1 to 0.3 % for maturities of a few
  months, 0.2 to 2.5 % for 1 to 3 years, and about 2 % at 10 years for an
  at-the-money put. It is larger in relative terms for out-of-the-money
  options, and its sign depends on the case.
- r, q and sigma are constant.
- Call with q <= 0, or put with r <= 0: early exercise is never optimal, so the
  American price equals the European price. This result is exact.
- sigma = 0: the formula is undefined (division by sigma²). The exact price of
  the deterministic problem, max over t of e^{-rt} g(S_t), is returned instead.
- Discrete dividends (extra approximation): BAW is applied to the contract and
  market adjusted by Bos-Vandermark (see european.py). For a call, Black's
  pseudo-American rule also considers exercising just before each ex-dividend
  date, and the price is the highest of these values. No exercise boundary is
  reported then. Measured against the exact tree, the gap stays below 1.5 %
  for small dividends or a continuous yield on top. It reaches 4 % (calls,
  underpriced) to 7 % (puts, overpriced) when a large dividend falls late in
  the life of the option. Prefer the binomial tree or LSM in that case.
"""
import math
from dataclasses import dataclass, replace

from scipy.optimize import brentq
from scipy.stats import norm

from .contract import Option
from .european import d1, escrowed_dividends, european_price
from .market import Market

# beyond these bounds, the exercise region is treated as empty
_MAX_SEARCH_FACTOR = 1e12


@dataclass(frozen=True)
class BAWResult:
    price: float
    european_price: float
    boundary: float | None   # S* (call) or S** (put); None without early exercise

    @property
    def early_exercise_premium(self) -> float:
        return self.price - self.european_price


def early_exercise_possible(option: Option, market: Market) -> bool:
    """With continuous dividends only. Call: only if q > 0. Put: only if r > 0."""
    return market.dividend_yield > 0 if option.is_call else market.rate > 0


def deterministic_american_price(option: Option, market: Market) -> float:
    """
    Exact American price when sigma = 0, discrete dividends included.

    Between two dividend dates, exercising at t pays, in present value,
    f(t) = sign (S0' e^{-qt} - K e^{-rt}), where S0' is the spot of the segment
    brought back to time 0. f has at most one critical point,
    t* = ln(rK / (q S0')) / (r - q), so on each segment the maximum is reached
    at an end of the segment or at t*. At a dividend date, exercise uses the
    cum-dividend spot.
    """
    K, T = option.strike, option.maturity
    r, q = market.rate, market.dividend_yield

    def best_on_segment(segment_spot, start, end):
        spot_at_zero = segment_spot*math.exp(-(r - q)*start)

        def gain(t):
            return option.sign*(spot_at_zero*math.exp(-q*t) - K*math.exp(-r*t))

        times = [start, end]
        if q != 0 and r != q and spot_at_zero > 0 and r*K/(q*spot_at_zero) > 0:
            critical = math.log(r*K/(q*spot_at_zero))/(r - q)
            if start < critical < end:
                times.append(critical)
        return max(gain(t) for t in times)

    best, spot, start = 0.0, market.spot, 0.0
    for dividend in market.dividends_before(T):
        best = max(best, best_on_segment(spot, start, dividend.time))
        spot = max(spot*math.exp((r - q)*(dividend.time - start)) - dividend.amount, 0.0)
        start = dividend.time
    return max(best, best_on_segment(spot, start, T))


def _exponent(option: Option, market: Market) -> float:
    """lambda: q2 for the call, q1 for the put."""
    r, T, s2 = market.rate, option.maturity, market.volatility**2
    # r / K(T) = r / (1 - e^{-rT}), extended by continuity at r = 0 (limit 1/T)
    r_over_k = 1/T if r == 0 else r/(-math.expm1(-r*T))
    n = 2*market.carry/s2
    root = math.sqrt((n - 1)**2 + 8*r_over_k/s2)
    return (-(n - 1) + option.sign*root)/2


def _factor(option: Option, market: Market, spot: float) -> float:
    """1 - e^{-qT} N(sign d1(S)): factor shared by A and by the boundary equation."""
    T = option.maturity
    D1 = d1(spot, option.strike, T, market.rate, market.dividend_yield, market.volatility)
    return 1 - math.exp(-market.dividend_yield*T)*norm.cdf(option.sign*D1)


def _boundary_equation(spot: float, option: Option, market: Market, exponent: float) -> float:
    """Zero at S*; negative in the continuation region near the strike."""
    european = european_price(option, replace(market, spot=spot))
    return option.sign*(spot - option.strike) - european - option.sign*_factor(option, market, spot)*spot/exponent


def baw_boundary(option: Option, market: Market) -> float:
    """
    Critical spot S* (call) or S** (put), solved with Brent on a bracketing interval.

    The equation is negative at S = K. It becomes positive when S goes to +inf
    for the call (since q2 > 1), and when S goes to 0 for the put (since r > 0).
    The interval is widened geometrically until the sign changes.
    Returns inf (call) or 0 (put) if the exercise region is numerically empty.
    """
    if option.maturity == 0 or market.volatility == 0:
        raise ValueError("The BAW boundary is only defined for T > 0 and sigma > 0.")
    if market.dividends_before(option.maturity):
        raise ValueError("The BAW boundary is not defined with discrete dividends.")
    if not early_exercise_possible(option, market):
        raise ValueError("Early exercise is never optimal: there is no boundary.")

    K = option.strike
    exponent = _exponent(option, market)

    def f(spot):
        return _boundary_equation(spot, option, market, exponent)

    if f(K) >= 0:
        return K
    if option.is_call:
        low, high = K, 2*K
        while f(high) <= 0:
            low, high = high, 2*high
            if high > K*_MAX_SEARCH_FACTOR:
                return math.inf
    else:
        low, high = K/2, K
        while f(low) <= 0:
            low, high = low/2, low
            if low < K/_MAX_SEARCH_FACTOR:
                return 0.0
    return brentq(f, low, high, xtol=1e-12*K, maxiter=200)


def _discrete_dividend_result(option: Option, market: Market, european: float) -> BAWResult:
    """Bos-Vandermark adjustment, plus Black's pseudo-American rule for calls."""
    price = baw_result(*escrowed_dividends(option, market)).price
    if option.is_call:
        for dividend in market.dividends_before(option.maturity):
            # exercise just before this ex-dividend date: an option maturing then
            before = replace(market, dividends=tuple(d for d in market.dividends if d.time < dividend.time))
            shorter = replace(option, maturity=dividend.time)
            price = max(price, baw_result(*escrowed_dividends(shorter, before)).price)
    return BAWResult(max(price, option.payoff(market.spot)), european, None)


def baw_result(option: Option, market: Market) -> BAWResult:
    S = market.spot
    intrinsic = option.payoff(S)
    european = european_price(option, market)

    if option.maturity == 0:
        return BAWResult(intrinsic, intrinsic, None)
    if market.volatility == 0:
        return BAWResult(deterministic_american_price(option, market), european, None)
    if market.dividends_before(option.maturity):
        return _discrete_dividend_result(option, market, european)
    if not early_exercise_possible(option, market):
        return BAWResult(european, european, None)

    boundary = baw_boundary(option, market)
    if boundary in (0.0, math.inf):
        return BAWResult(european, european, None)

    in_exercise_region = S >= boundary if option.is_call else S <= boundary
    if in_exercise_region:
        return BAWResult(intrinsic, european, boundary)

    exponent = _exponent(option, market)
    A = option.sign*boundary/exponent*_factor(option, market, boundary)
    return BAWResult(european + A*(S/boundary)**exponent, european, boundary)


def baw_price(option: Option, market: Market) -> float:
    return baw_result(option, market).price
