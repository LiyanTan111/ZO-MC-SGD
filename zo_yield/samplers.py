"""Samplers for the process-variation distribution rho(xi).

All samplers share an interface:

    .sample(batch_size: int, rng: Optional[np.random.Generator]) -> ndarray (batch_size, d)

`d` is exposed via the `.dim` attribute. `mean` / `cov` properties expose the analytic
first/second moments where applicable (used by quadrature transforms).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class BaseSampler:
    dim: int

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        raise NotImplementedError


class IndependentGaussianSampler(BaseSampler):
    """xi_j ~ N(mean_j, std_j^2), independent across j."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        if self.mean.shape != self.std.shape:
            raise ValueError("mean and std must share shape")
        self.dim = self.mean.shape[0]

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        z = rng.standard_normal((batch_size, self.dim))
        return self.mean[None, :] + self.std[None, :] * z

    @property
    def cov(self) -> np.ndarray:
        return np.diag(self.std**2)


class CorrelatedGaussianSampler(BaseSampler):
    """xi ~ N(mean, cov)."""

    def __init__(self, mean: Sequence[float], cov: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        self.cov_ = np.asarray(cov, dtype=float)
        if self.cov_.shape != (self.mean.shape[0], self.mean.shape[0]):
            raise ValueError("cov shape mismatch with mean")
        self.dim = self.mean.shape[0]
        # cholesky factor for fast sampling
        self.L = np.linalg.cholesky(self.cov_ + 1e-12 * np.eye(self.dim))

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        z = rng.standard_normal((batch_size, self.dim))
        return self.mean[None, :] + z @ self.L.T

    @property
    def cov(self) -> np.ndarray:
        return self.cov_


class IndependentUniformSampler(BaseSampler):
    """xi_j ~ Unif(low_j, high_j)."""

    def __init__(self, low: Sequence[float], high: Sequence[float]):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        if self.low.shape != self.high.shape:
            raise ValueError("low and high must share shape")
        if np.any(self.high <= self.low):
            raise ValueError("high must be strictly greater than low")
        self.dim = self.low.shape[0]

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        u = rng.uniform(0.0, 1.0, size=(batch_size, self.dim))
        return self.low[None, :] + u * (self.high - self.low)[None, :]

    @property
    def mean(self) -> np.ndarray:
        return 0.5 * (self.low + self.high)

    @property
    def cov(self) -> np.ndarray:
        return np.diag((self.high - self.low) ** 2 / 12.0)


class GaussianMixtureSampler(BaseSampler):
    """sum_k weights_k * N(means_k, covs_k); diagonal or full covs."""

    def __init__(
        self,
        weights: Sequence[float],
        means: Sequence[Sequence[float]],
        covs: Sequence[np.ndarray],
    ):
        w = np.asarray(weights, dtype=float)
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("weights must sum to 1")
        self.weights = w
        self.means = [np.asarray(m, dtype=float) for m in means]
        self.covs = [np.asarray(c, dtype=float) for c in covs]
        d = self.means[0].shape[0]
        for m, c in zip(self.means, self.covs):
            if m.shape != (d,):
                raise ValueError("means must share dimension d")
            if c.shape != (d, d):
                raise ValueError("covs must be d x d")
        self.dim = d
        self.Ls = [np.linalg.cholesky(c + 1e-12 * np.eye(d)) for c in self.covs]

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        comps = rng.choice(len(self.weights), size=batch_size, p=self.weights)
        out = np.empty((batch_size, self.dim))
        for i, k in enumerate(comps):
            z = rng.standard_normal(self.dim)
            out[i] = self.means[k] + self.Ls[k] @ z
        return out
