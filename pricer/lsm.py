"""
lsm.py - Least-Squares Monte Carlo of Longstaff and Schwartz (2001) for American options.

Algorithm
---------
1. Simulate N GBM paths with the exact scheme (model.py). The exercise dates
   are t_m = m T / M for m = 1..M, plus each discrete dividend date
   (cum-dividend exercise). Exercise at t = 0 is handled separately.
2. At maturity, each path receives the cash flow g(S_T) at date tau = T.
3. For each exercise date t_m, going backwards from T:
   - keep only the in-the-money paths (g(S_tm) > 0);
   - y = retained cash flow discounted from tau back to t_m;
   - regress y on the basis B(S_tm / K): estimated continuation = B beta_m;
   - exercise if g(S_tm) > continuation, and update (tau, cash flow).
4. Price = average of the cash flows discounted to 0. It is then compared with
   the immediate payoff at t = 0, which is the same on every path.

The three sources of error
--------------------------
- Statistical error: the average of N cash flows fluctuates around its
  expectation. It is the only error measured by `std_error`, which decreases
  as 1/sqrt(N).
- Regression bias: the finite basis and the noise in the coefficients give a
  sub-optimal exercise policy. When it is evaluated on independent paths
  (out_of_sample=True), any policy gives a lower bound, in expectation, of the
  Bermudan price. When it is evaluated on the training paths, the estimate can
  be biased upwards, because the policy has "seen" the paths it evaluates.
- Discretisation error: with M exercise dates, the option priced is a Bermudan
  option, whose price is lower than or equal to the American price. The gap
  shrinks as M grows.

The standard error therefore does NOT guarantee the accuracy of the American
price: a ±2 standard error interval covers neither the regression bias nor the
discretisation error.
"""
from dataclasses import dataclass

import numpy as np

from .contract import Option
from .market import Market
from .model import BlackScholesModel
from .regression import make_basis, least_squares


@dataclass(frozen=True)
class LSMParameters:
    n_paths: int = 100_000
    n_dates: int = 50               # regular exercise dates M on (0, T]
    basis: str = "polynomial"       # "polynomial" or "laguerre"
    degree: int = 2
    seed: int | None = 0            # None: not reproducible
    antithetic: bool = True
    out_of_sample: bool = True      # policy learnt and evaluated on two independent path sets


@dataclass(frozen=True)
class LSMResult:
    price: float
    std_error: float
    in_sample_price: float          # estimate on the training paths
    exercise_now: bool              # True if exercising at t = 0 beats the estimated continuation

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """Statistical interval only: covers neither the regression bias nor the discretisation."""
        return self.price - 1.96*self.std_error, self.price + 1.96*self.std_error


def _learn_policy(option: Option, paths: np.ndarray, times: np.ndarray, rate: float, basis):
    """Backward regression. Returns the coefficients per date and the cash flows discounted to 0."""
    n_paths, n_columns = paths.shape
    last = n_columns - 1
    cash_flow = option.payoff(paths[:, last])
    tau = np.full(n_paths, last)
    coefficients = [None]*last

    for m in range(last - 1, 0, -1):
        exercise = option.payoff(paths[:, m])
        indices = np.flatnonzero(exercise > 0)
        X = basis(paths[indices, m]/option.strike)
        if len(indices) <= X.shape[1]:
            continue   # too few points to regress: no exercise at this date
        y = cash_flow[indices]*np.exp(-rate*(times[tau[indices]] - times[m]))
        beta = least_squares(X, y)
        exercised = indices[exercise[indices] > X @ beta]
        cash_flow[exercised] = exercise[exercised]
        tau[exercised] = m
        coefficients[m] = beta

    return coefficients, cash_flow*np.exp(-rate*times[tau])


def _apply_policy(option: Option, paths: np.ndarray, times: np.ndarray, coefficients, rate: float, basis):
    """Applies an already learnt policy forward in time; cash flows discounted to 0."""
    n_paths, n_columns = paths.shape
    last = n_columns - 1
    cash_flow = np.zeros(n_paths)
    tau = np.full(n_paths, last)
    alive = np.ones(n_paths, dtype=bool)

    for m in range(1, last):
        if coefficients[m] is None:
            continue
        exercise = option.payoff(paths[:, m])
        indices = np.flatnonzero(alive & (exercise > 0))
        continuation = basis(paths[indices, m]/option.strike) @ coefficients[m]
        exercised = indices[exercise[indices] > continuation]
        cash_flow[exercised] = exercise[exercised]
        tau[exercised] = m
        alive[exercised] = False

    cash_flow[alive] = option.payoff(paths[alive, last])
    return cash_flow*np.exp(-rate*times[tau])


def _mean_and_error(cash_flows: np.ndarray, antithetic: bool) -> tuple[float, float]:
    """In antithetic mode, the pairs (i, i + N/2) are averaged: they are the independent draws."""
    if antithetic:
        half = len(cash_flows)//2
        cash_flows = (cash_flows[:half] + cash_flows[half:])/2
    return float(cash_flows.mean()), float(cash_flows.std(ddof=1)/np.sqrt(len(cash_flows)))


def lsm_price(option: Option, market: Market, parameters: LSMParameters = LSMParameters()) -> LSMResult:
    p = parameters
    intrinsic = option.payoff(market.spot)
    if option.maturity == 0:
        return LSMResult(intrinsic, 0.0, intrinsic, intrinsic > 0)
    if p.n_paths < 4:
        raise ValueError("At least 4 paths are needed.")

    basis = make_basis(p.basis, p.degree)
    model = BlackScholesModel(market)
    times = model.simulation_times(option.maturity, p.n_dates)
    training_seed, evaluation_seed = np.random.SeedSequence(p.seed).spawn(2)

    paths = model.simulate(times, p.n_paths, np.random.default_rng(training_seed), p.antithetic)
    coefficients, cash_flows = _learn_policy(option, paths, times, market.rate, basis)
    in_sample_price, error = _mean_and_error(cash_flows, p.antithetic)

    if p.out_of_sample:
        paths = model.simulate(times, p.n_paths, np.random.default_rng(evaluation_seed), p.antithetic)
        cash_flows = _apply_policy(option, paths, times, coefficients, market.rate, basis)
        price, error = _mean_and_error(cash_flows, p.antithetic)
    else:
        price = in_sample_price

    # at t = 0 every path shares S0: the decision is deterministic
    if intrinsic > price:
        return LSMResult(intrinsic, 0.0, in_sample_price, True)
    return LSMResult(price, error, in_sample_price, False)
