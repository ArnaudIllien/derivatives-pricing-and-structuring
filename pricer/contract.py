"""
contract.py - Option contracts and payoffs.

- `Option`: the vanilla call or put priced by every model of the package, with
  a strike and a maturity in years;
- `AsianOption`: a fixed-strike call or put on the arithmetic average of the
  spot over discrete observation dates, paid at maturity;
- `year_fraction`: converts two calendar dates into a maturity in years.

The contracts carry no exercise style: the pricer decides whether early
exercise is allowed (European formula, BAW, LSM, binomial tree) and at which
dates (Bermudan pricers take their exercise dates as an argument).

Both contracts have `shifted(dt)`, the same contract seen dt years later,
which greeks.py uses for Theta.
"""
from dataclasses import dataclass, replace
from datetime import date

import numpy as np

CALL = "call"
PUT = "put"


def year_fraction(valuation_date: date, expiry_date: date) -> float:
    """Maturity in years between two dates (actual/365)."""
    if expiry_date < valuation_date:
        raise ValueError("The expiry date must be on or after the valuation date.")
    return (expiry_date - valuation_date).days/365


@dataclass(frozen=True)
class Option:
    """Call or put with strike K and maturity T (in years)."""
    option_type: str
    strike: float
    maturity: float

    def __post_init__(self):
        option_type = self.option_type.lower()
        if option_type not in (CALL, PUT):
            raise ValueError(f"option_type must be '{CALL}' or '{PUT}', not {self.option_type!r}.")
        if self.strike <= 0:
            raise ValueError("The strike must be positive.")
        if self.maturity < 0:
            raise ValueError("The maturity must be zero or positive.")
        object.__setattr__(self, "option_type", option_type)

    @property
    def is_call(self) -> bool:
        return self.option_type == CALL

    @property
    def sign(self) -> float:
        """+1 for a call, -1 for a put: payoff = max(sign * (S - K), 0)."""
        return 1.0 if self.is_call else -1.0

    def payoff(self, spot):
        """Payoff max(S - K, 0) or max(K - S, 0), for a scalar or a numpy array."""
        value = np.maximum(self.sign*(np.asarray(spot, dtype=float) - self.strike), 0.0)
        return float(value) if value.ndim == 0 else value

    def shifted(self, dt: float) -> "Option":
        """Same contract seen dt years later."""
        return replace(self, maturity=self.maturity - dt)


@dataclass(frozen=True)
class AsianOption:
    """
    Fixed-strike arithmetic Asian option: pays max(sign * (A - K), 0) at T, where
    A = (1/n) sum S(t_i) over the observation dates t_1 <= ... <= t_n in [0, T].

    An observation on a discrete dividend date uses the cum-dividend spot. A date
    listed twice counts twice in the average.
    """
    option_type: str
    strike: float
    maturity: float
    observation_times: tuple[float, ...]

    def __post_init__(self):
        vanilla = Option(self.option_type, self.strike, self.maturity)   # same checks as a vanilla
        times = tuple(sorted(float(t) for t in self.observation_times))
        if not times:
            raise ValueError("An Asian option needs at least one observation date.")
        if times[0] < 0 or times[-1] > self.maturity:
            raise ValueError(f"Every observation date must lie between 0 and the maturity T={self.maturity}.")
        object.__setattr__(self, "option_type", vanilla.option_type)
        object.__setattr__(self, "observation_times", times)

    @classmethod
    def regular(cls, option_type: str, strike: float, maturity: float, n_observations: int) -> "AsianOption":
        """Observations at m T / n for m = 1..n (S0 excluded, S_T included)."""
        if n_observations < 1:
            raise ValueError("An Asian option needs at least one observation date.")
        times = tuple(maturity*m/n_observations for m in range(1, n_observations + 1))
        return cls(option_type, strike, maturity, times)

    @property
    def vanilla(self) -> Option:
        """European option with the same type, strike and maturity."""
        return Option(self.option_type, self.strike, self.maturity)

    @property
    def is_call(self) -> bool:
        return self.vanilla.is_call

    def payoff(self, average):
        """Payoff on the average A, for a scalar or a numpy array of averages."""
        return self.vanilla.payoff(average)

    def shifted(self, dt: float) -> "AsianOption":
        """
        Same contract seen dt years later. A date that passes is fixed at the current
        spot (its time becomes 0): with the spot held fixed, as in Theta, this is the
        natural convention, and the error is of order dt.
        """
        return replace(self, maturity=self.maturity - dt,
                       observation_times=tuple(max(t - dt, 0.0) for t in self.observation_times))
