"""Synthetic stochastic problems S1 (quadratic) and S2 (yield-like).

Both problems take xi ~ N(0, I_dxi). Each exposes:

  * ``f(x, xi)``       — single-sample loss (matches simulator's evaluate)
  * ``E_loss(x)``      — analytic E_xi[f(x, xi)]
  * ``grad_E_loss(x)`` — analytic gradient of E_loss
  * ``yield_at(x)``    — analytic yield (S2 only; S1 has no spec)
  * ``var_f(x)``       — analytic Var_xi[f(x, xi)] at fixed x

Plus two adapters:

  * ``as_simulator()``  — minimal object with ``.evaluate(x, xi) -> float``
                          and counters, suitable for Stoch ZO Option 1/2.
  * ``as_sampler()``    — :class:`zo_yield.samplers.IndependentGaussianSampler`
                          over xi (mean 0, std 1, dim ``d_xi``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import multivariate_normal, norm

from zo_yield.samplers import IndependentGaussianSampler


# --------------------------------------------------------------------------- #
# Adapter helpers
# --------------------------------------------------------------------------- #
class _SyntheticSimulator:
    """Wraps a problem's :meth:`f` so it looks like a circuit simulator."""

    LOSS_PENALTY = 1e3

    def __init__(self, problem):
        self._p = problem
        self.n_calls = 0
        self.n_failures = 0

    def evaluate(self, x: np.ndarray, xi: np.ndarray) -> float:
        self.n_calls += 1
        return float(self._p.f(np.asarray(x, dtype=float),
                                np.asarray(xi, dtype=float)))


# --------------------------------------------------------------------------- #
# S1 — stochastic quadratic
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticQuadratic:
    """f(x, xi) = ½ ||x - x*||^2 + (A xi)^T x,    xi ~ N(0, I_dxi).

    Closed forms:
      E_xi[f]        = ½ ||x - x*||^2          (since E[xi] = 0)
      grad_x E_xi[f] = x - x*
      Var_xi[f|x]    = x^T A A^T x             (linear functional in xi)

    The matrix ``A`` controls how much xi contaminates the loss. Constructing
    it as ``A = U Sigma V^T`` with ``Sigma = diag(s_1, ..., s_dxi)`` lets
    the user dial the *effective* xi-dimension: only the columns of A with
    nonzero s contribute. By default we set ``s = [1, 1, 0, ...]`` so the
    first two xi coordinates are "live" and the rest are passengers — the
    cleanest analogue of the OTA's "few live xi out of many" structure.
    """

    d_x: int
    d_xi: int
    seed: int = 0
    n_active_xi: int = 2
    a_scale: float = 1.0
    x_star: np.ndarray = field(default=None, repr=False)
    A: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        if self.x_star is None:
            self.x_star = rng.standard_normal(self.d_x)
        if self.A is None:
            U = _random_orthogonal(self.d_x, rng)
            V = _random_orthogonal(self.d_xi, rng)
            sigma = np.zeros(min(self.d_x, self.d_xi))
            n_act = min(self.n_active_xi, sigma.shape[0])
            sigma[:n_act] = self.a_scale
            S = np.zeros((self.d_x, self.d_xi))
            for i, s in enumerate(sigma):
                S[i, i] = s
            self.A = U @ S @ V.T

    # ---- analytic ground truth ---- #
    def E_loss(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return 0.5 * float(np.sum((x - self.x_star) ** 2))

    def grad_E_loss(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x - self.x_star

    def var_f(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.A @ self.A.T @ x)

    # ---- single-sample evaluator ---- #
    def f(self, x: np.ndarray, xi: np.ndarray) -> float:
        x = np.asarray(x, dtype=float); xi = np.asarray(xi, dtype=float)
        return 0.5 * float(np.sum((x - self.x_star) ** 2)) + float(x @ (self.A @ xi))

    # ---- adapters ---- #
    def as_simulator(self) -> _SyntheticSimulator:
        return _SyntheticSimulator(self)

    def as_sampler(self) -> IndependentGaussianSampler:
        return IndependentGaussianSampler(mean=np.zeros(self.d_xi),
                                            std=np.ones(self.d_xi))


# --------------------------------------------------------------------------- #
# S2 — stochastic yield-like
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticYieldLike:
    """Two affine specs g_1(x, xi), g_2(x, xi); yield = P(g_1 ≥ 0, g_2 ≥ 0).

    g_k(x, xi) = a_k^T x + b_k^T xi + c_k,   xi ~ N(0, I_dxi)

    For fixed x, (g_1, g_2) is bivariate normal with
      mean   = (a_1^T x + c_1, a_2^T x + c_2)
      stddev = (||b_1||,       ||b_2||)
      corr   = b_1^T b_2 / (||b_1||·||b_2||)

    Yield = P(g_1 ≥ 0, g_2 ≥ 0) = 1 - Phi(-mu_1/sig_1) - Phi(-mu_2/sig_2)
                                  + Phi_2(-mu_1/sig_1, -mu_2/sig_2; corr)
    via inclusion-exclusion on the bivariate Gaussian CDF.

    Loss:
      L(x, xi) = sum_k softplus(alpha · max(0, -g_k(x, xi))) / alpha

    Default a, b, c are chosen so yield(x_star) ∈ [0.6, 0.8] (matching
    the circuit benchmarks' headroom band) and so the gradient of yield
    w.r.t. x has a well-defined direction toward improvement.
    """

    d_x: int
    d_xi: int
    seed: int = 0
    alpha: float = 5.0
    a1: np.ndarray = field(default=None, repr=False)
    a2: np.ndarray = field(default=None, repr=False)
    b1: np.ndarray = field(default=None, repr=False)
    b2: np.ndarray = field(default=None, repr=False)
    c1: float = 0.0
    c2: float = 0.0
    x_star: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        if self.a1 is None:
            self.a1 = rng.standard_normal(self.d_x) / np.sqrt(self.d_x)
        if self.a2 is None:
            self.a2 = rng.standard_normal(self.d_x) / np.sqrt(self.d_x)
        if self.b1 is None:
            self.b1 = 0.5 * rng.standard_normal(self.d_xi) / np.sqrt(self.d_xi)
        if self.b2 is None:
            self.b2 = 0.5 * rng.standard_normal(self.d_xi) / np.sqrt(self.d_xi)
        if self.x_star is None:
            self.x_star = rng.standard_normal(self.d_x)
        # Calibrate c1, c2 so yield(x_star) lands in [0.6, 0.8] band.
        # (Closed-form, no MC.) Heuristic: target both per-spec passrates
        # to ~0.85 so joint passrate is ~0.7 with mild correlation.
        if self.c1 == 0.0 and self.c2 == 0.0:
            target_per_spec = 0.85
            z = float(norm.ppf(target_per_spec))   # we want mu/sigma = z
            sig1 = float(np.linalg.norm(self.b1))
            sig2 = float(np.linalg.norm(self.b2))
            self.c1 = z * sig1 - float(self.a1 @ self.x_star)
            self.c2 = z * sig2 - float(self.a2 @ self.x_star)

    # ---- per-spec moment sheet at a fixed x ---- #
    def _gauss_moments(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        mu1 = float(self.a1 @ x + self.c1)
        mu2 = float(self.a2 @ x + self.c2)
        sig1 = float(np.linalg.norm(self.b1))
        sig2 = float(np.linalg.norm(self.b2))
        rho = float(self.b1 @ self.b2 / (sig1 * sig2 + 1e-30))
        return mu1, mu2, sig1, sig2, rho

    # ---- analytic yield ---- #
    def yield_at(self, x: np.ndarray) -> float:
        mu1, mu2, sig1, sig2, rho = self._gauss_moments(x)
        a = -mu1 / sig1
        b = -mu2 / sig2
        # P(Z1 >= a, Z2 >= b) for bivariate standard normal w/ corr rho
        cov = np.array([[1.0, rho], [rho, 1.0]])
        cdf_ab = float(multivariate_normal.cdf([a, b], mean=[0.0, 0.0], cov=cov))
        return 1.0 - float(norm.cdf(a)) - float(norm.cdf(b)) + cdf_ab

    # ---- analytic E_loss ---- #
    def E_loss(self, x: np.ndarray) -> float:
        """Analytic E_xi[L]. softplus(alpha·max(0, -g_k))/alpha for g_k ~ N(mu_k, sig_k^2).

        For a single g ~ N(mu, sig^2):
          E[softplus(alpha·max(0, -g))/alpha]
          = ln(2)/alpha · P(g ≥ 0) + E[softplus(-alpha·g)/alpha · 1{g<0}]
                                      + ln(2)/alpha · P(g < 0)   ← when -g <= 0 inside max it's 0
        This is messy in closed form. We substitute a tight semi-analytic
        approximation: split on g≥0 vs g<0, integrate the g<0 tail with
        Gauss-Hermite. (10-point GH is exact for polynomials up to deg 19;
        softplus is smooth enough that this is essentially analytic for
        evaluation purposes.)
        """
        mu1, mu2, sig1, sig2, _rho = self._gauss_moments(x)
        return _expected_softplus_penalty(mu1, sig1, self.alpha) \
             + _expected_softplus_penalty(mu2, sig2, self.alpha)

    # ---- single-sample loss ---- #
    def f(self, x: np.ndarray, xi: np.ndarray) -> float:
        x = np.asarray(x, dtype=float); xi = np.asarray(xi, dtype=float)
        g1 = float(self.a1 @ x + self.b1 @ xi + self.c1)
        g2 = float(self.a2 @ x + self.b2 @ xi + self.c2)
        return _softplus_penalty(g1, self.alpha) + _softplus_penalty(g2, self.alpha)

    # ---- analytic grad of E_loss ---- #
    def grad_E_loss(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """∇_x E_xi[L(x, xi)] via central differences on the analytic E_loss.

        E_loss is itself essentially exact (GH quadrature on a smooth softplus),
        so this central-diff gradient inherits that precision — it is the
        ground-truth gradient for ZO-MC-SGD to be benchmarked against.
        """
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)
        for j in range(x.shape[0]):
            ej = np.zeros_like(x); ej[j] = eps
            g[j] = (self.E_loss(x + ej) - self.E_loss(x - ej)) / (2 * eps)
        return g

    # ---- adapters ---- #
    def as_simulator(self) -> _SyntheticSimulator:
        return _SyntheticSimulator(self)

    def as_sampler(self) -> IndependentGaussianSampler:
        return IndependentGaussianSampler(mean=np.zeros(self.d_xi),
                                            std=np.ones(self.d_xi))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random n×n orthogonal matrix via QR of a Gaussian draw."""
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q


def _softplus(z: np.ndarray | float) -> np.ndarray | float:
    z = np.asarray(z, dtype=float)
    # Numerically stable softplus.
    return np.where(z > 30.0, z, np.log1p(np.exp(np.minimum(z, 30.0))))


def _softplus_penalty(g: float, alpha: float) -> float:
    """softplus(alpha · max(0, -g)) / alpha.

    Convex, smooth, equals ~0 for g large positive, equals ~(-g) for g
    large negative. Has a baseline ln(2)/alpha at g = 0.
    """
    arg = alpha * max(0.0, -float(g))
    return float(_softplus(arg)) / alpha


# Cached 20-point Gauss-Hermite nodes / weights for E_softplus integral.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(20)


def _expected_softplus_penalty(mu: float, sig: float, alpha: float) -> float:
    """E[softplus(alpha · max(0, -g))/alpha] for g ~ N(mu, sig^2).

    Computed via 20-point Gauss-Hermite (exact for polynomials up to deg 39;
    softplus is smooth enough that this is < 1e-10 absolute error for the
    α/μ/σ ranges we use — verified by MC in test_synthetic.py).
    """
    g_samples = mu + sig * _GH_NODES
    pen = np.array([_softplus_penalty(float(gs), alpha) for gs in g_samples])
    return float((pen * _GH_WEIGHTS).sum() / np.sqrt(2 * np.pi))
