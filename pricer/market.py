"""
market.py - Market data of the Black-Scholes model.

- `Dividend`: one discrete cash dividend (payment time in years, amount);
- `Market`: spot, risk-free rate, volatility, continuous dividend yield and
  discrete dividends, all constant over the life of the option.

Conventions
-----------
- Rates and yields are continuously compounded.
- A discrete dividend paid at time t lowers the spot by its amount right after
  t: S(t+) = S(t) - D. Exercise at t happens before the drop (cum-dividend).
- Only dividends paid strictly between 0 and the maturity affect an option.

Future extensions (rate curves, local volatility) will replace these constants
without touching the contract.
"""
import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Dividend:
    time: float     # payment time in years from the valuation date
    amount: float   # cash amount

    def __post_init__(self):
        if self.time <= 0:
            raise ValueError("A dividend must be paid after the valuation date (time > 0).")
        if self.amount < 0:
            raise ValueError("A dividend amount must be zero or positive.")


@dataclass(frozen=True)
class Market:
    spot: float                      # S, underlying price
    rate: float                      # r, risk-free rate
    volatility: float                # sigma
    dividend_yield: float = 0.0      # q, continuous dividend yield
    dividends: tuple[Dividend, ...] = ()

    def __post_init__(self):
        if self.spot <= 0:
            raise ValueError("The spot must be positive.")
        if self.volatility < 0:
            raise ValueError("The volatility must be zero or positive.")
        object.__setattr__(self, "dividends", tuple(sorted(self.dividends, key=lambda d: d.time)))

    @property
    def carry(self) -> float:
        """Cost of carry b = r - q: risk-neutral drift of the underlying between dividends."""
        return self.rate - self.dividend_yield

    def dividends_before(self, maturity: float) -> tuple[Dividend, ...]:
        """Discrete dividends paid strictly between 0 and the maturity."""
        return tuple(d for d in self.dividends if d.time < maturity)

    def pv_dividends(self, maturity: float) -> float:
        return sum(d.amount*math.exp(-self.rate*d.time) for d in self.dividends_before(maturity))

    def shifted(self, dt: float) -> "Market":
        """Same market seen dt years later, spot unchanged: dividends get closer, paid ones are dropped."""
        return replace(self, dividends=tuple(Dividend(d.time - dt, d.amount) for d in self.dividends if d.time > dt))
