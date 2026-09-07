"""antithetic ξ sampling.

Pins three properties:
  (i)   sample mean of an antithetic batch is closer to 0 than a naive
        batch at the same seed and n_mc;
  (ii)  for the analytic-yield S2 problem,
        Var[yield_estimate | antithetic] ≤ Var[yield_estimate | naive]
        across 200 fresh seeds at n_mc=8 — antithetic should help on a
        loss whose ξ-dependence is roughly monotonic;
  (iii) requesting odd n_mc with antithetic=True raises a clear error.

The implementation lives in ``zo_yield.variance_reduction.draw_xi_batch``
and ``yield_estimate_with_vr`` rather than as a flag on the sampler
class — the reference pseudocode wraps the sampler instead of
modifying it, which keeps the sampler API stable.
"""
import numpy as np
import pytest

from zo_yield.synthetic import SyntheticYieldLike
from zo_yield.variance_reduction import draw_xi_batch, yield_estimate_with_vr


def test_antithetic_sample_mean_closer_to_zero():
    p = SyntheticYieldLike(d_x=6, d_xi=10, seed=0)
    samp = p.as_sampler()
    naive = draw_xi_batch(samp, n_mc=8, seed=42, antithetic=False)
    anti = draw_xi_batch(samp, n_mc=8, seed=42, antithetic=True)
    # Naive 8-sample mean has stderr 1/√8 ≈ 0.35 per coord; antithetic
    # mean is exactly 0 by construction.
    assert np.linalg.norm(anti.mean(axis=0)) < 1e-12, \
        "antithetic sample mean should be exactly 0 (mirrored pairs)"
    # Naive mean shouldn't be 0 by accident.
    assert np.linalg.norm(naive.mean(axis=0)) > 0.05


def test_antithetic_odd_n_mc_raises():
    p = SyntheticYieldLike(d_x=6, d_xi=10, seed=0)
    samp = p.as_sampler()
    with pytest.raises(ValueError, match="even n_mc"):
        draw_xi_batch(samp, n_mc=7, seed=42, antithetic=True)


def test_antithetic_reduces_yield_estimate_variance_on_s2():
    """At a fixed off-x* point on S2/mid_d, antithetic ξ should make
    the n_mc=8 yield estimate less noisy than naive ξ across seeds.
    """
    p = SyntheticYieldLike(d_x=6, d_xi=10, seed=0)
    samp = p.as_sampler()
    rng_x = np.random.default_rng(7)
    x_check = p.x_star + 0.3 * rng_x.standard_normal(p.d_x)
    # Build a yield predicate from p's specs.
    def predicate(xi):
        g1 = p.a1 @ x_check + p.b1 @ xi + p.c1
        g2 = p.a2 @ x_check + p.b2 @ xi + p.c2
        return g1 >= 0 and g2 >= 0

    n_seeds = 200
    naive = []
    anti = []
    for s in range(n_seeds):
        naive.append(yield_estimate_with_vr(predicate, samp, n_mc=8,
                                              seed=3000 + s, antithetic=False))
        anti.append(yield_estimate_with_vr(predicate, samp, n_mc=8,
                                             seed=3000 + s, antithetic=True))
    var_naive = float(np.array(naive).var(ddof=1))
    var_anti = float(np.array(anti).var(ddof=1))
    ratio = var_naive / max(var_anti, 1e-30)
    assert ratio >= 1.0, \
        f"antithetic should not increase yield-estimate variance: " \
        f"var_naive/var_anti = {ratio:.3f}"
