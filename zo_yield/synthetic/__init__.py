"""Synthetic stochastic problems with analytic ground truth.

These give ZO-MC-SGD a problem on which the exact gradient, the exact E[loss]
and (S2 only) the exact yield are known in closed form. They are what the
estimator's unbiasedness and variance claims are validated against; a silent
bias bug would not show up against circuit experiments alone.

Two problems exposed:
  * :class:`SyntheticQuadratic` (S1) — smooth quadratic with linear
    coupling to ξ. Closed-form E[f], ∇E[f], Var[f|x].
  * :class:`SyntheticYieldLike` (S2) — two affine specs in ξ; yield is
    the bivariate-Gaussian-CDF probability that both specs pass.
    Closed-form yield via :func:`scipy.stats.multivariate_normal.cdf`.

Both expose :meth:`as_simulator` and :meth:`as_sampler` adapters so
:class:`zo_yield.optimizers.ZOMCSGD` runs on them with no extra wiring.
"""
from .problems import SyntheticQuadratic, SyntheticYieldLike

__all__ = ["SyntheticQuadratic", "SyntheticYieldLike"]
