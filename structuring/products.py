"""
products.py - Contract terms and path-wise payoffs of the structures.

Every structure turns simulated paths of the underlying's performance
P_t = S_t / S_0 (P_0 = 1) into CASH FLOWS, in fraction of the notional, paid on
its payment dates. One representation serves the valuation (discounted
expected flows), the scenarios (flows on a given path) and the risk measures.

Structures (single underlying; levels in % of the initial fixing S_0):
- DirectExposure          buy the underlying, income reinvested: pays e^{qT} P_T at T.
- CapitalProtectedNote    protection + participation x (min(P_T, cap) - strike)+ at T
                          (a bearish version pays on the downside). Protection < 100 %
                          gives a partially protected note.
- ReverseConvertible      fixed coupons; at T pays 100 %, or P_T / strike if the
                          conversion condition holds (always below the strike without
                          barrier; with a barrier, only if the barrier was breached).
- Autocallable            conditional (Phoenix) coupons with optional memory, early
                          redemption at 100 % when P >= trigger on an observation
                          date, capital at risk below the capital barrier at maturity.
- OptionPackage           cash deposit + listed-style options and/or the underlying
                          (call spread, put spread, collar). The investor pays the
                          premium out of the capital; the rest earns the rate.

Barrier monitoring: "maturity" (only the final fixing counts, European barrier)
or "daily" (any daily close, American barrier); the simulation grid then holds
every business day (252 per year). Payment dates have no settlement lag.

The flows of notes are promises of an issuer (issuer_risk=True): valuation.py
applies the issuer credit adjustment to them. A package or a direct holding has
no issuer risk (the counterparty risk of OTC options is mentioned, not modelled).
"""
from dataclasses import dataclass, field, replace

import numpy as np

from pricer.european import bsm_price
from pricer.market import Market
from pricer.model import nearest_columns

MONITORING = ("maturity", "daily")
DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class CashFlows:
    times: np.ndarray       # (n_dates,) payment times in years
    amounts: np.ndarray     # (n_paths, n_dates) in fraction of the notional
    issuer_risk: bool

    @property
    def total(self) -> np.ndarray:
        """Undiscounted sum of the flows received on each path."""
        return self.amounts.sum(axis=1)

    @property
    def end_time(self) -> np.ndarray:
        """Time of the last flow on each path (early redemption for an autocallable)."""
        paid = self.amounts != 0
        last = np.where(paid.any(axis=1), paid.shape[1] - 1 - np.argmax(paid[:, ::-1], axis=1), paid.shape[1] - 1)
        return self.times[last]


def _schedule(maturity: float, frequency: int) -> tuple[float, ...]:
    n = max(1, round(maturity*frequency))
    return tuple(maturity*k/n for k in range(1, n + 1))


def _check_monitoring(monitoring: str):
    if monitoring not in MONITORING:
        raise ValueError(f"Barrier monitoring must be one of {MONITORING}.")


@dataclass(frozen=True)
class Product:
    maturity: float
    name: str = field(default="", kw_only=True)

    family = "product"
    complexity = 1                       # 1 simple, 2 intermediate, 3 complex (knowledge check)
    issuer_risk = True

    def __post_init__(self):
        if self.maturity <= 0:
            raise ValueError("The maturity must be positive.")

    @property
    def daily_monitoring(self) -> bool:
        return False

    def observation_times(self) -> tuple[float, ...]:
        return (self.maturity,)

    def payment_times(self) -> tuple[float, ...]:
        return self.observation_times()

    def contractual_protection(self) -> float | None:
        """Minimum redemption at maturity in % of notional if the issuer is solvent (None: capital at risk)."""
        return None

    def pays_income(self) -> str:
        """'fixed', 'conditional' or 'none'."""
        return "none"

    def terms(self) -> list[tuple[str, str]]:
        return [("Maturity", f"{self.maturity:g} years")]

    def cash_flows(self, paths: np.ndarray, times: np.ndarray, market: Market) -> CashFlows:
        raise NotImplementedError

    def _columns(self, times, dates):
        return nearest_columns(np.asarray(times), dates)


@dataclass(frozen=True)
class DirectExposure(Product):
    name: str = field(default="Direct exposure", kw_only=True)
    family = "Direct exposure"
    complexity = 1
    issuer_risk = False

    def terms(self):
        return super().terms() + [("Payoff", "underlying performance, income (dividends, foreign interest or "
                                             "futures collateral interest) reinvested")]

    def cash_flows(self, paths, times, market):
        final = paths[:, self._columns(times, [self.maturity])[0]]
        amounts = (np.exp(market.dividend_yield*self.maturity)*final)[:, None]
        return CashFlows(np.array([self.maturity]), amounts, False)


@dataclass(frozen=True)
class CapitalProtectedNote(Product):
    protection: float = 1.0          # redemption floor at maturity
    participation: float = 1.0
    strike: float = 1.0
    cap: float | None = None         # bullish: performance capped at this level; bearish: floored
    bearish: bool = False
    name: str = field(default="Capital-protected note", kw_only=True)
    family = "Capital protection"
    complexity = 2

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.protection <= 1.5 or self.participation < 0 or self.strike <= 0:
            raise ValueError("Protection in [0, 150 %], participation >= 0 and strike > 0 are required.")
        if self.cap is not None and (self.cap <= self.strike if not self.bearish else self.cap >= self.strike):
            raise ValueError("The cap must lie beyond the strike in the direction of the payoff.")

    def contractual_protection(self):
        return self.protection

    def terms(self):
        direction = "downside" if self.bearish else "upside"
        rows = super().terms() + [("Capital protection at maturity", f"{self.protection:.0%} (issuer risk)"),
                                  ("Participation", f"{self.participation:.0%} of the {direction} beyond {self.strike:.0%}")]
        if self.cap is not None:
            rows.append(("Cap" if not self.bearish else "Floor", f"{self.cap:.0%} of the initial level"))
        return rows

    def option_payoff(self, final: np.ndarray) -> np.ndarray:
        if self.bearish:
            floor = self.cap if self.cap is not None else 0.0
            return np.maximum(self.strike - np.maximum(final, floor), 0.0)
        capped = np.minimum(final, self.cap) if self.cap is not None else final
        return np.maximum(capped - self.strike, 0.0)

    def cash_flows(self, paths, times, market):
        final = paths[:, self._columns(times, [self.maturity])[0]]
        amounts = self.protection + self.participation*self.option_payoff(final)
        return CashFlows(np.array([self.maturity]), amounts[:, None], True)


@dataclass(frozen=True)
class ReverseConvertible(Product):
    coupon_rate: float = 0.08        # per year, paid whatever happens (issuer risk)
    coupon_frequency: int = 1
    barrier: float | None = 0.6      # None: plain reverse convertible (conversion below the strike)
    barrier_monitoring: str = "maturity"
    strike: float = 1.0
    barrier_breached: bool = False   # state of a contract already alive (daily barrier already hit)
    name: str = field(default="Barrier reverse convertible", kw_only=True)
    family = "Reverse convertible"
    complexity = 3

    def __post_init__(self):
        super().__post_init__()
        _check_monitoring(self.barrier_monitoring)
        if self.barrier is not None and not 0 < self.barrier <= self.strike:
            raise ValueError("The barrier must lie in (0, strike].")

    @property
    def daily_monitoring(self):
        return self.barrier is not None and self.barrier_monitoring == "daily"

    def observation_times(self):
        return _schedule(self.maturity, self.coupon_frequency)

    def pays_income(self):
        return "fixed"

    def terms(self):
        barrier = (f"{self.barrier:.0%} ({self.barrier_monitoring} observation)" if self.barrier is not None
                   else "none: conversion whenever the final level is below the strike")
        return super().terms() + [("Coupon", f"{self.coupon_rate:.2%} per year, paid {self.coupon_frequency}x a year, "
                                             "unconditional (issuer risk)"),
                                  ("Strike", f"{self.strike:.0%}"), ("Capital barrier", barrier),
                                  ("Redemption", "100 %, or final level / strike if converted (capital at risk)")]

    def converted(self, paths, times) -> np.ndarray:
        final = paths[:, self._columns(times, [self.maturity])[0]]
        below_strike = final < self.strike
        if self.barrier is None:
            return below_strike
        if self.barrier_monitoring == "daily":
            return below_strike & (self.barrier_breached | (paths[:, 1:].min(axis=1) < self.barrier))
        return below_strike & (final < self.barrier)

    def cash_flows(self, paths, times, market):
        dates = np.array(self.observation_times())
        amounts = np.full((len(paths), len(dates)), self.coupon_rate/self.coupon_frequency)
        final = paths[:, self._columns(times, [self.maturity])[0]]
        amounts[:, -1] += np.where(self.converted(paths, times), final/self.strike, 1.0)
        return CashFlows(dates, amounts, True)


@dataclass(frozen=True)
class Autocallable(Product):
    coupon_rate: float = 0.07        # per year; each observation pays coupon_rate / frequency if conditions hold
    observation_frequency: int = 1
    autocall_trigger: float = 1.0
    coupon_barrier: float = 0.7
    memory: bool = True
    capital_barrier: float = 0.6
    barrier_monitoring: str = "maturity"
    first_call: int = 1              # first observation (1-based) at which early redemption is possible
    missed_coupons: int = 0          # state of a contract already alive: unpaid coupons kept in memory
    barrier_breached: bool = False   # state of a contract already alive: daily capital barrier already hit
    name: str = field(default="Phoenix autocallable", kw_only=True)
    family = "Autocallable"
    complexity = 3

    def __post_init__(self):
        super().__post_init__()
        _check_monitoring(self.barrier_monitoring)
        if not 0 < self.capital_barrier <= 1 or not 0 < self.coupon_barrier or self.autocall_trigger <= 0:
            raise ValueError("Barriers and trigger must be positive, the capital barrier at most 100 %.")
        if self.first_call < 1:
            raise ValueError("first_call counts observations from 1.")

    @property
    def daily_monitoring(self):
        return self.barrier_monitoring == "daily"

    def observation_times(self):
        return _schedule(self.maturity, self.observation_frequency)

    def pays_income(self):
        return "conditional"

    def terms(self):
        n = len(self.observation_times())
        return super().terms() + [
            ("Observations", f"{n} ({self.observation_frequency}x a year)"),
            ("Conditional coupon", f"{self.coupon_rate:.2%} per year if the level is >= {self.coupon_barrier:.0%}"
                                   + (" (with memory of missed coupons)" if self.memory else "")),
            ("Early redemption", f"100 % if the level is >= {self.autocall_trigger:.0%}, from observation {self.first_call}"),
            ("Capital barrier", f"{self.capital_barrier:.0%} ({self.barrier_monitoring} observation)"),
            ("Redemption at maturity", "100 % above the capital barrier, else the final level (capital at risk)")]

    def cash_flows(self, paths, times, market):
        dates = np.array(self.observation_times())
        levels = paths[:, self._columns(times, dates)]
        n_paths, n_obs = levels.shape
        coupon = self.coupon_rate/self.observation_frequency
        amounts = np.zeros((n_paths, n_obs))
        alive = np.ones(n_paths, dtype=bool)
        missed = np.full(n_paths, float(self.missed_coupons))
        for i in range(n_obs):
            level = levels[:, i]
            paid = alive & (level >= self.coupon_barrier)
            amounts[paid, i] += coupon*(1 + (missed[paid] if self.memory else 0))
            missed = np.where(paid, 0, missed + alive)
            if i < n_obs - 1:
                called = alive & (i + 1 >= self.first_call) & (level >= self.autocall_trigger)
                amounts[called, i] += 1.0
                alive &= ~called
        final = levels[:, -1]
        if self.barrier_monitoring == "daily":
            loss = (self.barrier_breached | (paths[:, 1:].min(axis=1) < self.capital_barrier)) & (final < 1.0)
        else:
            loss = final < self.capital_barrier
        amounts[alive, -1] += np.where(loss[alive], final[alive], 1.0)
        return CashFlows(dates, amounts, True)


@dataclass(frozen=True)
class Leg:
    kind: str           # "call", "put" or "underlying"
    strike: float = 1.0
    quantity: float = 1.0   # per unit of notional; negative = sold

    def payoff(self, final: np.ndarray, market: Market, maturity: float) -> np.ndarray:
        if self.kind == "underlying":
            return self.quantity*np.exp(market.dividend_yield*maturity)*final
        sign = 1.0 if self.kind == "call" else -1.0
        return self.quantity*np.maximum(sign*(final - self.strike), 0.0)

    def price(self, market: Market, maturity: float) -> float:
        if self.kind == "underlying":
            return self.quantity*market.spot
        return self.quantity*bsm_price(self.kind == "call", market.spot, self.strike, maturity, market.rate,
                                       market.dividend_yield, market.volatility)


@dataclass(frozen=True)
class OptionPackage(Product):
    legs: tuple[Leg, ...] = ()
    fees: float = 0.0               # upfront costs (commissions, bid-ask), fraction of the capital: hypothetical
    name: str = field(default="Option package", kw_only=True)
    family = "Option strategy"
    complexity = 2
    issuer_risk = False

    def __post_init__(self):
        super().__post_init__()
        if not self.legs:
            raise ValueError("An option package needs at least one leg.")
        for leg in self.legs:
            if leg.kind not in ("call", "put", "underlying"):
                raise ValueError(f"Unknown leg {leg.kind!r}.")

    def premium(self, market: Market) -> float:
        """Model value of the legs at inception (BSM): what the investor pays for them."""
        return sum(leg.price(market, self.maturity) for leg in self.legs)

    def terms(self):
        rows = super().terms()
        for leg in self.legs:
            side = "long" if leg.quantity > 0 else "short"
            what = "underlying" if leg.kind == "underlying" else f"{leg.kind} strike {leg.strike:.0%}"
            rows.append((f"{side.capitalize()} {abs(leg.quantity):.3g} x", what))
        rows.append(("Cash", "capital minus premium and fees, deposited at the risk-free rate"))
        return rows

    def cash_flows(self, paths, times, market):
        final = paths[:, self._columns(times, [self.maturity])[0]]
        cash = (1.0 - self.premium(market) - self.fees)*np.exp(market.rate*self.maturity)
        amounts = cash + sum(leg.payoff(final, market, self.maturity) for leg in self.legs)
        return CashFlows(np.array([self.maturity]), amounts[:, None], False)


def call_spread(maturity: float, lower: float = 1.0, upper: float = 1.2, quantity: float = 1.0,
                fees: float = 0.0) -> OptionPackage:
    return OptionPackage(maturity, (Leg("call", lower, quantity), Leg("call", upper, -quantity)), fees,
                         name=f"Cash + call spread {lower:.0%}/{upper:.0%}")


def put_spread(maturity: float, upper: float = 1.0, lower: float = 0.8, quantity: float = 1.0,
               fees: float = 0.0) -> OptionPackage:
    return OptionPackage(maturity, (Leg("put", upper, quantity), Leg("put", lower, -quantity)), fees,
                         name=f"Cash + put spread {upper:.0%}/{lower:.0%}")


def collar(maturity: float, put_strike: float = 0.9, call_strike: float = 1.1, fees: float = 0.0) -> OptionPackage:
    return OptionPackage(maturity, (Leg("underlying"), Leg("put", put_strike, 1.0), Leg("call", call_strike, -1.0)),
                         fees, name=f"Collar {put_strike:.0%}/{call_strike:.0%} on the underlying")


def with_value(product: Product, field_name: str, value: float) -> Product:
    """The same contract with one term changed (used to solve a coupon or a participation)."""
    return replace(product, **{field_name: value})
