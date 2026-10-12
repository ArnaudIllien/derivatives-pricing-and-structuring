"""
regression.py - Basis functions and least-squares regression for LSM.

- `make_basis(name, degree)` returns a function that turns x into the
  regression matrix (one column per basis function);
- `least_squares(X, y)` returns the regression coefficients.

The explanatory variable is normalised, x = S / K, so that the columns of the
regression matrix stay of order 1 and the system stays well conditioned.
"""
from collections.abc import Callable

import numpy as np
from numpy.polynomial import laguerre

Basis = Callable[[np.ndarray], np.ndarray]


def make_basis(name: str = "polynomial", degree: int = 2) -> Basis:
    """
    - "polynomial": 1, x, ..., x^degree.
    - "laguerre": 1 and e^{-x/2} L_k(x) for k = 0..degree (weighted Laguerre
      polynomials, as in Longstaff and Schwartz, 2001).
    """
    if degree < 1:
        raise ValueError("The basis degree must be at least 1.")
    if name == "polynomial":
        return lambda x: np.vander(x, degree + 1, increasing=True)
    if name == "laguerre":
        return lambda x: np.column_stack([np.ones_like(x), np.exp(-x/2)[:, None]*laguerre.lagvander(x, degree)])
    raise ValueError(f"Unknown basis: {name!r} (choose 'polynomial' or 'laguerre').")


def least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coefficients beta minimising ||X beta - y||² (SVD, robust to an ill-conditioned matrix)."""
    return np.linalg.lstsq(X, y, rcond=None)[0]
