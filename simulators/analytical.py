"""Analytical black-box test problems with closed-form gradients (for validation)."""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .base import BlackBoxSimulator


class AnalyticalQuadratic(BlackBoxSimulator):
    r"""f(x, xi) = 0.5 * (x - mu - xi)^T A (x - mu - xi).

    With xi having zero mean,
        E_rho[f] = 0.5 * (x - mu)^T A (x - mu) + 0.5 * tr(A * Cov(xi)),
    so the optimum is x* = mu, independent of A and Cov(xi).

    Use ``xi_injection='additive'`` (default) for the form above; or ``'rhs'`` to
    inject as a linear-in-xi term ``f = 0.5 (x-mu)^T A (x-mu) + xi^T B (x - mu)``.
    """

    def __init__(
        self,
        A: np.ndarray,
        mu: np.ndarray,
        B: Optional[np.ndarray] = None,
        xi_injection: Literal["additive", "rhs"] = "additive",
    ):
        super().__init__()
        self.A = np.asarray(A, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.B = B if B is None else np.asarray(B, dtype=float)
        self.xi_injection = xi_injection
        self.n = self.mu.shape[0]
        if self.A.shape != (self.n, self.n):
            raise ValueError("A shape must be (n, n)")

    def evaluate(self, x: np.ndarray, xi: np.ndarray) -> float:
        self.n_calls += 1
        if self.xi_injection == "additive":
            d = x - self.mu - xi
            return 0.5 * float(d @ self.A @ d)
        if self.xi_injection == "rhs":
            d = x - self.mu
            quad = 0.5 * float(d @ self.A @ d)
            if self.B is None:
                lin = float(xi @ d)
            else:
                lin = float(xi @ self.B @ d)
            return quad + lin
        raise ValueError(f"unknown xi_injection={self.xi_injection}")

    # closed-form gradient for tests
    def true_grad(self, x: np.ndarray, xi: np.ndarray) -> np.ndarray:
        if self.xi_injection == "additive":
            return self.A @ (x - self.mu - xi)
        if self.xi_injection == "rhs":
            d = x - self.mu
            base = self.A @ d
            return base + (xi if self.B is None else self.B.T @ xi)
        raise ValueError

    def expected_grad(self, x: np.ndarray, mean_xi: np.ndarray) -> np.ndarray:
        """E_xi[grad_x f] (closed form for tests)."""
        if self.xi_injection == "additive":
            return self.A @ (x - self.mu - mean_xi)
        # rhs: linear-in-xi term contributes (B^T mean_xi or mean_xi)
        d = x - self.mu
        base = self.A @ d
        return base + (mean_xi if self.B is None else self.B.T @ mean_xi)


class NoisyRosenbrock(BlackBoxSimulator):
    r"""Stochastic Rosenbrock used for stress-testing high-d Option 1.

        f(x, xi) = sum_{i=0}^{n-2} 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2
                   + xi^T (x - 1)

    Optimum (without xi term) at x = 1; with zero-mean xi, E_rho[f] same.
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.x_star = np.ones(n)

    def evaluate(self, x: np.ndarray, xi: np.ndarray) -> float:
        self.n_calls += 1
        diff = x[1:] - x[:-1] ** 2
        f = 100.0 * np.sum(diff**2) + np.sum((1.0 - x[:-1]) ** 2)
        f += float(xi @ (x - 1.0))
        return float(f)
