"""Local GP with Matern-5/2 ARD kernel for TuRBO.

Pure numpy + scipy.linalg/optimize (no gpytorch/torch) — keeps TuRBO
self-contained and dependency-light, per the task's stated philosophy. The GP
is fit by maximum marginal likelihood over per-dimension lengthscales (ARD),
signal variance, and noise variance; it exposes posterior mean/variance and a
joint Thompson sample over a candidate set (the TuRBO acquisition primitive).

Kernel (Matern-5/2, ARD):
  r = sqrt( sum_i ((x_i - x'_i)/λ_i)^2 )
  k(x,x') = σ_f^2 (1 + √5 r + 5/3 r^2) exp(-√5 r)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize

SQRT5 = np.sqrt(5.0)


def _scaled_dist2(Xa: np.ndarray, Xb: np.ndarray, ls: np.ndarray) -> np.ndarray:
    """Pairwise squared distances with per-dim lengthscales -> (na, nb)."""
    Xa = Xa / ls[None, :]
    Xb = Xb / ls[None, :]
    a2 = np.sum(Xa ** 2, axis=1)[:, None]
    b2 = np.sum(Xb ** 2, axis=1)[None, :]
    d2 = a2 + b2 - 2.0 * Xa @ Xb.T
    return np.maximum(d2, 0.0)


def _matern52(Xa: np.ndarray, Xb: np.ndarray, ls: np.ndarray,
              sf2: float) -> np.ndarray:
    d2 = _scaled_dist2(Xa, Xb, ls)
    r = np.sqrt(d2)
    return sf2 * (1.0 + SQRT5 * r + (5.0 / 3.0) * d2) * np.exp(-SQRT5 * r)


class GPModel:
    """Gaussian process regression with a Matern-5/2 ARD kernel.

    y is standardized (zero mean, unit std) internally for numerical stability;
    predictions are returned in the original scale.
    """

    def __init__(self, n_restarts: int = 3, max_iter: int = 60,
                 noise_floor: float = 1e-6, jitter: float = 1e-6):
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.noise_floor = noise_floor
        self.jitter = jitter
        self._fitted = False

    # ---- fitting ---------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray,
            rng: Optional[np.random.Generator] = None) -> "GPModel":
        rng = rng or np.random.default_rng(0)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        self.X = X
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) if y.std() > 1e-12 else 1.0
        self.y = (y - self.y_mean) / self.y_std
        self.d = d

        # log-hyperparameters theta = [log λ_1..λ_d, log σ_f^2, log σ_n^2].
        # Bounds keep the optimizer in a sane region.
        lb = np.array([np.log(1e-2)] * d + [np.log(1e-3), np.log(self.noise_floor)])
        ub = np.array([np.log(1e2)] * d + [np.log(1e3), np.log(1.0)])
        bounds = list(zip(lb, ub))

        def nll(theta):
            ls = np.exp(theta[:d])
            sf2 = np.exp(theta[d])
            sn2 = np.exp(theta[d + 1])
            K = _matern52(X, X, ls, sf2) + (sn2 + self.jitter) * np.eye(n)
            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                return 1e25
            alpha = cho_solve((L, True), self.y)
            # -log marginal likelihood
            val = (0.5 * self.y @ alpha
                   + np.sum(np.log(np.diag(L)))
                   + 0.5 * n * np.log(2.0 * np.pi))
            return float(val)

        best_theta, best_val = None, np.inf
        # First start: median-heuristic lengthscale, moderate signal/noise.
        theta0_list = [np.concatenate([
            np.log(np.full(d, 0.5)), [np.log(1.0)], [np.log(1e-2)]])]
        for _ in range(self.n_restarts - 1):
            theta0_list.append(rng.uniform(lb, ub))
        for theta0 in theta0_list:
            try:
                res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": self.max_iter})
                if res.fun < best_val:
                    best_val, best_theta = float(res.fun), res.x
            except Exception:
                continue
        if best_theta is None:
            best_theta = theta0_list[0]

        self.ls = np.exp(best_theta[:d])
        self.sf2 = float(np.exp(best_theta[d]))
        self.sn2 = float(np.exp(best_theta[d + 1]))

        K = _matern52(X, X, self.ls, self.sf2) + (self.sn2 + self.jitter) * np.eye(n)
        self.L = cholesky(K, lower=True)
        self.alpha = cho_solve((self.L, True), self.y)
        self._fitted = True
        return self

    # ---- prediction ------------------------------------------------------- #
    def predict(self, Xstar: np.ndarray, return_cov: bool = False
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean and (variance | covariance) in the ORIGINAL y scale."""
        if not self._fitted:
            raise RuntimeError("GPModel.predict called before fit")
        Xstar = np.asarray(Xstar, dtype=float)
        Ks = _matern52(Xstar, self.X, self.ls, self.sf2)          # (m, n)
        mu = Ks @ self.alpha                                       # standardized
        v = solve_triangular(self.L, Ks.T, lower=True)            # (n, m)
        if return_cov:
            Kss = _matern52(Xstar, Xstar, self.ls, self.sf2)
            cov = Kss - v.T @ v
            mu = mu * self.y_std + self.y_mean
            cov = cov * (self.y_std ** 2)
            return mu, cov
        Kss_diag = self.sf2 * np.ones(Xstar.shape[0])             # k(x,x)=σ_f^2
        var = Kss_diag - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 0.0)
        mu = mu * self.y_std + self.y_mean
        var = var * (self.y_std ** 2)
        return mu, var

    # ---- Thompson sampling ----------------------------------------------- #
    def thompson_sample(self, Xstar: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
        """One joint posterior realization f ~ N(μ, Σ) over the candidates,
        returned in the original y scale. Deterministic given `rng` state."""
        mu, cov = self.predict(Xstar, return_cov=True)
        m = mu.shape[0]
        cov = 0.5 * (cov + cov.T)                                  # symmetrize
        jit = self.jitter * (self.y_std ** 2)
        for _ in range(6):
            try:
                Lc = cholesky(cov + jit * np.eye(m), lower=True)
                break
            except np.linalg.LinAlgError:
                jit *= 10.0
        else:
            # Fallback: independent marginal sample if covariance is hopeless.
            var = np.clip(np.diag(cov), 0.0, None)
            return mu + np.sqrt(var) * rng.standard_normal(m)
        z = rng.standard_normal(m)
        return mu + Lc @ z
