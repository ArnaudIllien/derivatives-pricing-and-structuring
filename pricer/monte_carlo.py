"""
monte_carlo.py - Statistics shared by the Monte Carlo pricers (LSM, Asian).

- `independent_draws` averages the antithetic pairs, which are the truly
  independent samples;
- `mean_and_error` returns the sample mean and its standard error;
- `control_variate` corrects a sample with a control of known expectation.

No pricing logic here: the pricers simulate with model.py and pass their
discounted cash flows to these functions.
"""
import numpy as np


def independent_draws(samples: np.ndarray, antithetic: bool) -> np.ndarray:
    """In antithetic mode, sample i and sample i + N/2 form a pair: their average is one independent draw."""
    if not antithetic:
        return samples
    half = len(samples)//2
    return (samples[:half] + samples[half:])/2


def mean_and_error(samples: np.ndarray) -> tuple[float, float]:
    """Sample mean and standard error std / sqrt(N) of independent samples."""
    return float(samples.mean()), float(samples.std(ddof=1)/np.sqrt(len(samples)))


def control_variate(y: np.ndarray, x: np.ndarray, x_mean: float) -> tuple[float, float, float]:
    """
    Control-variate estimate of E[y] from a control x with known expectation x_mean:
        y_cv = y - beta (x - x_mean),  beta = cov(y, x) / var(x),
    which minimises the variance of y_cv. Returns (estimate, standard error, beta).

    beta is estimated on the same sample. This adds a bias of order 1/N,
    negligible next to the standard error, which ignores it. If x is constant,
    there is nothing to correct and beta = 0.
    """
    variance = x.var(ddof=1)
    beta = 0.0 if variance == 0 else float(np.cov(y, x)[0, 1]/variance)
    estimate, error = mean_and_error(y - beta*(x - x_mean))
    return estimate, error, beta
