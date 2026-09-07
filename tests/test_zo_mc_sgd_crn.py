"""CRN switch in the ZO gradient helper.

Pins three properties:
  (i)   crn=True with the same seed gives bit-identical g_hat (the same
        ξ batch is used for both ±FD evaluations);
  (ii)  crn=False with the same seed gives a different g_hat (independent
        ξ batches for ±FD);
  (iii) over 200 fresh seeds at a fixed off-optimum x on S1/mid_d,
        Var[g_hat | crn=True] < Var[g_hat | crn=False].

(iii) is the load-bearing claim — without it the CRN ablation
would be moot.
"""
import numpy as np
import pytest

from zo_yield.synthetic import SyntheticQuadratic
from zo_yield.variance_reduction import zo_grad_with_vr


def _setup(seed=0):
    p = SyntheticQuadratic(d_x=6, d_xi=10, seed=seed)
    sim = p.as_simulator()
    samp = p.as_sampler()
    x_check = p.x_star + 0.5 * np.ones(p.d_x)
    return p, sim, samp, x_check


def test_crn_true_deterministic_at_fixed_seed():
    p, sim, samp, x = _setup()
    rng = np.random.default_rng(7)
    v = rng.standard_normal(p.d_x)   # phi_factor("gaussian")=1 convention
    g_a = zo_grad_with_vr(sim, samp, x, eps=1e-3, n_mc=8, v=v,
                           xi_seed=42, crn=True)
    g_b = zo_grad_with_vr(sim, samp, x, eps=1e-3, n_mc=8, v=v,
                           xi_seed=42, crn=True)
    np.testing.assert_array_equal(g_a, g_b)


def test_crn_false_uses_different_xi_for_plus_minus():
    """With crn=False, f+ and f- average over different ξ batches; that
    introduces additional noise on top of the FD signal, so two calls
    with the same xi_seed (which only seeds the f+ batch) and different
    structure for the f- batch (xi_seed + 1) should still produce a
    deterministic result — but it should differ from the crn=True
    result at the same seed."""
    p, sim, samp, x = _setup()
    rng = np.random.default_rng(7)
    v = rng.standard_normal(p.d_x)   # phi_factor("gaussian")=1 convention
    g_crn = zo_grad_with_vr(sim, samp, x, eps=1e-3, n_mc=8, v=v,
                              xi_seed=42, crn=True)
    g_no_crn = zo_grad_with_vr(sim, samp, x, eps=1e-3, n_mc=8, v=v,
                                 xi_seed=42, crn=False)
    assert not np.allclose(g_crn, g_no_crn), \
        "CRN=False must produce a different g_hat (different xi for f-)"


def test_crn_reduces_variance_on_s1():
    """Headline property: CRN reduces Var[g_hat] on the S1 problem.

    On S1 the loss is f(x, ξ) = ½||x - x*||² + xᵀA·ξ. The ξ-noise
    contribution to f(x+εv, ξ) - f(x-εv, ξ) is (x+εv)ᵀA·ξ -
    (x-εv)ᵀA·ξ = 2·εvᵀA·ξ when ξ is shared; if ξ is independent on the
    two sides, the noise floor is much higher.
    """
    p, sim, samp, x = _setup()
    n_seeds = 200
    eps = 1e-3
    n_mc = 8
    g_crn_all = []
    g_no_crn_all = []
    for s in range(n_seeds):
        rng = np.random.default_rng(1000 + s)
        v = rng.standard_normal(p.d_x)   # phi_factor("gaussian")=1 convention
        g_crn_all.append(zo_grad_with_vr(sim, samp, x, eps=eps, n_mc=n_mc,
                                           v=v, xi_seed=2000 + s, crn=True))
        g_no_crn_all.append(zo_grad_with_vr(sim, samp, x, eps=eps, n_mc=n_mc,
                                              v=v, xi_seed=2000 + s, crn=False))
    var_crn = float(np.array(g_crn_all).var(axis=0, ddof=1).sum())
    var_no_crn = float(np.array(g_no_crn_all).var(axis=0, ddof=1).sum())
    # CRN should give a meaningful reduction on S1 — bar at 1.5× to be
    # robust to seed luck while still failing if CRN is broken.
    ratio = var_no_crn / var_crn
    assert ratio > 1.5, \
        f"CRN didn't reduce variance enough: var_no_crn/var_crn = {ratio:.3f}"
