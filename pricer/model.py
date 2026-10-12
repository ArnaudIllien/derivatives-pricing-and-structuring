"""
model.py - Underlying model: geometric Brownian motion under the risk-neutral measure.

    dS_t = (r - q) S_t dt + sigma S_t dW_t, with a drop of D at each discrete dividend.

`BlackScholesModel` builds the simulation time grid (regular dates plus the
dividend dates) and simulates paths with the exact GBM scheme, so there is no
discretisation error at the grid dates.

Any other model (local or stochastic volatility...) must expose the same
`simulation_times` and `simulate` methods to be used by the LSM pricer.
"""
import numpy as np

from .market import Market


class BlackScholesModel:
    def __init__(self, market: Market):
        self.market = market

    def simulation_times(self, maturity: float, n_steps: int) -> np.ndarray:
        """Regular grid k T / n_steps, merged with the dividend dates in (0, T)."""
        if n_steps < 1:
            raise ValueError("At least one time step is needed.")
        grid = np.linspace(0, maturity, n_steps + 1)
        dividend_times = np.array([d.time for d in self.market.dividends_before(maturity)])
        if len(dividend_times) == 0:
            return grid
        # a grid date that almost coincides with a dividend date is replaced by it
        tolerance = 1e-10*maturity
        keep = np.all(np.abs(grid[:, None] - dividend_times[None, :]) > tolerance, axis=1)
        return np.unique(np.concatenate([grid[keep], dividend_times]))

    def dividend_payouts(self, times: np.ndarray) -> np.ndarray:
        """Cash amount paid at each date of the grid (zero where there is no dividend)."""
        payouts = np.zeros(len(times))
        for d in self.market.dividends_before(times[-1]):
            payouts[np.argmin(np.abs(times - d.time))] += d.amount
        return payouts

    def simulate(self, times: np.ndarray, n_paths: int, rng: np.random.Generator,
                 antithetic: bool = False) -> np.ndarray:
        """
        Paths of shape (n_paths, len(times)); column k is the cum-dividend price at times[k].

        Between two dates the exact scheme is used:
        S_(t+dt) = (S_t - D_t) exp[(r - q - sigma²/2) dt + sigma sqrt(dt) Z],
        where D_t is the dividend paid at t (zero most of the time). The spot is
        floored at 0 if a dividend exceeds it.

        In antithetic mode, the second half of the paths uses -Z: path i and
        path i + n_paths/2 form a pair.
        """
        if antithetic and n_paths % 2:
            raise ValueError("The number of paths must be even in antithetic mode.")

        m = self.market
        dts = np.diff(times)
        if antithetic:
            z = rng.standard_normal((n_paths//2, len(dts)))
            z = np.concatenate([z, -z])
        else:
            z = rng.standard_normal((n_paths, len(dts)))
        growth = np.exp((m.carry - m.volatility**2/2)*dts + m.volatility*np.sqrt(dts)*z)

        paths = np.empty((n_paths, len(times)))
        paths[:, 0] = m.spot
        payouts = self.dividend_payouts(times)
        if not payouts.any():
            paths[:, 1:] = m.spot*np.cumprod(growth, axis=1)
            return paths
        for k in range(len(dts)):
            paths[:, k + 1] = np.maximum(paths[:, k] - payouts[k], 0.0)*growth[:, k]
        return paths
