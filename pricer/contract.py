"""
contract.py - Option contract and payoff.

Defines the vanilla option priced by every model of the package:
- `Option`: a call or a put with a strike and a maturity in years;
- `year_fraction`: converts two calendar dates into a maturity in years.

The contract carries no exercise style: the pricer decides whether early
exercise is allowed (European formula, BAW, LSM, binomial tree).
"""
from dataclasses import dataclass
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
