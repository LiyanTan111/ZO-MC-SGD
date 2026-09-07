"""ZO-MC-SGD -- the stochastic zeroth-order optimizer of the paper.

:class:`ZOMCSGD` estimates the gradient of the smoothed-spec surrogate from
simulation alone. Each step draws a fresh mini-batch of ``batch_size_xi``
process samples from rho(xi), pairs each with its own direction v ~ N(0, I_n),
and averages the central two-point differences. Both evaluations in a pair share
the same xi (implicit per-pair common random numbers), which cancels the leading
process noise in the numerator, so the estimator's accuracy is governed by the
mini-batch size rather than by the process dimension.

:class:`ZOOptimizer` is the parameter-update base (SGD / Adam / SignGD), with
step-size schedule, box projection, and run history.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence

import numpy as np

from .estimators import (
    EstimatorConfig,
    mini_batch_estimate,
    sample_direction,
    phi_factor,
)
from .projection import project_box
from .samplers import BaseSampler


StepRule = Literal["sgd", "adam", "signgd"]
LRSchedule = Literal["const", "1/sqrt(t)", "1/t"]


# --------------------------------------------------------------------------- #
# step-size schedule                                                          #
# --------------------------------------------------------------------------- #
def lr_at(base_lr: float, t: int, schedule: LRSchedule) -> float:
    if schedule == "const":
        return base_lr
    if schedule == "1/sqrt(t)":
        return base_lr / np.sqrt(t + 1.0)
    if schedule == "1/t":
        return base_lr / (t + 1.0)
    raise ValueError(f"unknown schedule={schedule}")


# --------------------------------------------------------------------------- #
# base optimizer (handles parameter update only)                              #
# --------------------------------------------------------------------------- #
@dataclass
class OptimizerConfig:
    rule: StepRule = "adam"
    lr: float = 1e-2
    schedule: LRSchedule = "const"
    # Adam params
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    # box projection (optional)
    x_lo: Optional[np.ndarray] = None
    x_hi: Optional[np.ndarray] = None


class ZOOptimizer:
    """Stateful parameter updater. Knows nothing about ZO or rho(xi)."""

    def __init__(self, x0: np.ndarray, cfg: OptimizerConfig):
        self.x = np.array(x0, dtype=float, copy=True)
        self.cfg = cfg
        self.t = 0
        self._m = np.zeros_like(self.x)
        self._v = np.zeros_like(self.x)

    def step(self, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        lr = lr_at(self.cfg.lr, self.t - 1, self.cfg.schedule)
        if self.cfg.rule == "sgd":
            self.x = self.x - lr * grad
        elif self.cfg.rule == "signgd":
            self.x = self.x - lr * np.sign(grad)
        elif self.cfg.rule == "adam":
            b1, b2, eps = self.cfg.beta1, self.cfg.beta2, self.cfg.eps
            self._m = b1 * self._m + (1.0 - b1) * grad
            self._v = b2 * self._v + (1.0 - b2) * grad * grad
            m_hat = self._m / (1.0 - b1**self.t)
            v_hat = self._v / (1.0 - b2**self.t)
            self.x = self.x - lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            raise ValueError(f"unknown rule={self.cfg.rule}")
        # projection
        if self.cfg.x_lo is not None or self.cfg.x_hi is not None:
            self.x = project_box(self.x, self.cfg.x_lo, self.cfg.x_hi)
        return self.x


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _ground_truth_loss(
    f: Callable[[np.ndarray, np.ndarray], float],
    x: np.ndarray,
    eval_xis: np.ndarray,
) -> float:
    """E[f(x, xi)] estimated by averaging over a fixed evaluation set of xi's."""
    losses = np.array([f(x, xi) for xi in eval_xis])
    return float(losses.mean())


def _zo_grad_at_xi(
    f_at_xi: Callable[[np.ndarray], float],
    x: np.ndarray,
    cfg: EstimatorConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Wrapper around ``mini_batch_estimate`` so caller doesn't repeat the kwargs."""
    return mini_batch_estimate(
        f_at_xi,
        x,
        epsilon=cfg.epsilon,
        n_samples=cfg.n_samples,
        v_dist=cfg.v_dist,
        mode=cfg.mode,
        rng=rng,
    )


# --------------------------------------------------------------------------- #
# Option 1 -- Monte Carlo over xi                                             #
# --------------------------------------------------------------------------- #
@dataclass
class RunHistory:
    iters: list[int] = field(default_factory=list)
    sims: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    x_history: list[np.ndarray] = field(default_factory=list)
    convergence_failures: int = 0
    wallclock_s: float = 0.0


class ZOMCSGD:
    """Mini-batch ZO: expectation over xi via fresh Monte Carlo samples each step.

    added two variance-reduction kwargs:
      ``per_pair_crn``  : True (default) — each
                          per-(xi, v) pair shares its xi between f+ and
                          f- via the closure pattern at line 192. False —
                          draws a fresh ``xi_prime`` for f-, increasing
                          xi noise (degraded baseline that lets us
                          measure CRN's actual lift).
      ``antithetic_xi`` : False (default). True —
                          xi_batch is constructed antithetically:
                          ``[xi_1, …, xi_{K/2}, mirror(xi_1), …, mirror(xi_{K/2})]``
                          where ``mirror(xi) = 2·μ − xi``. Requires
                          ``batch_size_xi`` even.

    Default kwargs (``per_pair_crn=True, antithetic_xi=False``) reproduce
    the the documented default behaviour bit-identically (same RNG consumption order,
    same closure pattern). See ``tests/test_zo_mc_sgd_vr_kwargs.py`` for
    the parity check.
    """

    def __init__(
        self,
        simulator,  # zo_yield.simulators.base.BlackBoxSimulator
        sampler: BaseSampler,
        estimator_cfg: EstimatorConfig,
        optimizer_cfg: OptimizerConfig,
        batch_size_xi: int = 8,
        batch_size_v: int = 1,
        eval_xis: Optional[np.ndarray] = None,
        log_every: int = 1,
        per_pair_crn: bool = True,
        antithetic_xi: bool = False,
    ):
        self.simulator = simulator
        self.sampler = sampler
        self.est_cfg = estimator_cfg
        # use the per-step n_samples for ZO direction count
        self.est_cfg = EstimatorConfig(
            epsilon=estimator_cfg.epsilon,
            n_samples=batch_size_v,
            v_dist=estimator_cfg.v_dist,
            mode=estimator_cfg.mode,
        )
        self.opt_cfg = optimizer_cfg
        self.batch_size_xi = batch_size_xi
        self.batch_size_v = batch_size_v
        self.eval_xis = eval_xis
        self.log_every = log_every
        self.per_pair_crn = per_pair_crn
        self.antithetic_xi = antithetic_xi
        if self.antithetic_xi and self.batch_size_xi % 2 != 0:
            raise ValueError(
                f"antithetic_xi=True requires even batch_size_xi, got "
                f"{self.batch_size_xi}"
            )

    def run(
        self,
        x_init: np.ndarray,
        n_iters: int,
        rng: Optional[np.random.Generator] = None,
        verbose: bool = False,
    ) -> RunHistory:
        rng = rng or np.random.default_rng()
        opt = ZOOptimizer(x_init, self.opt_cfg)
        history = RunHistory()
        sim_count = 0
        t0 = time.time()

        for it in range(n_iters):
            # 1) draw xi mini-batch (antithetic-pair construction if requested)
            if self.antithetic_xi:
                half = self.sampler.sample(self.batch_size_xi // 2, rng=rng)
                mu = getattr(self.sampler, "mean",
                              np.zeros_like(half[0]))
                xi_batch = np.concatenate(
                    [half, 2.0 * mu[None, :] - half], axis=0)
            else:
                xi_batch = self.sampler.sample(self.batch_size_xi, rng=rng)
            # 2) per-xi ZO gradient (per-pair CRN by default; fresh xi_prime
            # for f- when per_pair_crn=False)
            grads = np.zeros_like(opt.x)
            if self.per_pair_crn:
                # ROUND-1.11 BIT-IDENTICAL PATH — do not modify
                for xi in xi_batch:
                    f_at_xi = lambda x_in, _xi=xi: self.simulator.evaluate(x_in, _xi)
                    g, c = _zo_grad_at_xi(f_at_xi, opt.x, self.est_cfg, rng)
                    grads += g
                    sim_count += c
            else:
                # NEW: per-pair gradient with fresh xi_prime for f-
                n_x = opt.x.shape[0]
                phi = phi_factor(n_x, self.est_cfg.v_dist)
                eps = self.est_cfg.epsilon
                for xi in xi_batch:
                    g_xi = np.zeros_like(opt.x)
                    for _ in range(self.batch_size_v):
                        v = sample_direction(n_x, self.est_cfg.v_dist, rng)
                        xi_prime = self.sampler.sample(1, rng=rng)[0]
                        f_p = float(self.simulator.evaluate(opt.x + eps * v, xi))
                        f_m = float(self.simulator.evaluate(opt.x - eps * v, xi_prime))
                        g_xi += (phi / (2.0 * eps)) * (f_p - f_m) * v
                    g_xi /= self.batch_size_v
                    grads += g_xi
                    sim_count += 2 * self.batch_size_v
            grads /= self.batch_size_xi
            # 3) optimizer step
            opt.step(grads)
            # 4) logging
            if (it % self.log_every == 0) or (it == n_iters - 1):
                if self.eval_xis is not None:
                    loss = _ground_truth_loss(self.simulator.evaluate, opt.x, self.eval_xis)
                else:
                    loss = float("nan")
                history.iters.append(it)
                history.sims.append(sim_count)
                history.loss.append(loss)
                history.grad_norm.append(float(np.linalg.norm(grads)))
                history.x_history.append(opt.x.copy())
                if verbose:
                    print(f"[ZO-MC-SGD] it={it:4d} sims={sim_count:6d} loss={loss:.6e} |g|={history.grad_norm[-1]:.3e}")

        history.wallclock_s = time.time() - t0
        history.convergence_failures = int(getattr(self.simulator, "n_failures", 0))
        return history


