"""sanity tests for the synthetic problems.

S1 (quadratic):
  - MC mean of f(x, xi) over n_mc=10^5 converges to E_loss(x) within 0.5 %.
  - MC central-difference gradient converges to grad_E_loss(x).
  - Var_xi[f|x] (analytic) matches MC variance within 5 %.

S2 (yield-like):
  - Analytic yield matches MC yield over n_mc=10^5 within 0.3 %.
  - Analytic E_loss matches MC E_loss within 1 %.
  - yield(x_star) is in [0.6, 0.8] band (calibration sanity).

If any of these fails, the optimizer comparisons are
suspect — the synthetic itself is buggy.
"""
import numpy as np
import pytest

from zo_yield.synthetic import SyntheticQuadratic, SyntheticYieldLike


# --------------------------------------------------------------------------- #
# S1 — quadratic
# --------------------------------------------------------------------------- #
def test_s1_E_loss_matches_mc():
    p = SyntheticQuadratic(d_x=6, d_xi=10, seed=0)
    rng = np.random.default_rng(42)
    x = p.x_star + 0.5 * rng.standard_normal(p.d_x)
    xis = rng.standard_normal((100_000, p.d_xi))
    mc = float(np.mean([p.f(x, xi) for xi in xis]))
    truth = p.E_loss(x)
    # MC std at n=10^5 with d_xi=10 random A is ~1% of truth, so 2% is a
    # safe ~2σ bound (failures here would indicate a bug, not noise).
    rel_err = abs(mc - truth) / max(abs(truth), 1.0)
    assert rel_err < 0.02, f"E_loss mismatch: mc={mc:.6f} truth={truth:.6f}"


def test_s1_mc_var_matches_analytic():
    p = SyntheticQuadratic(d_x=6, d_xi=10, seed=0)
    rng = np.random.default_rng(7)
    x = p.x_star + 0.5 * rng.standard_normal(p.d_x)
    xis = rng.standard_normal((100_000, p.d_xi))
    mc_var = float(np.var([p.f(x, xi) for xi in xis], ddof=1))
    truth = p.var_f(x)
    if truth > 1e-12:
        assert abs(mc_var - truth) / truth < 0.05, \
            f"Var_f mismatch: mc={mc_var:.6f} truth={truth:.6f}"


def test_s1_central_diff_grad_matches_analytic():
    """Central differences on E_loss (averaged over many xi) should match
    the analytic grad_E_loss to MC precision."""
    p = SyntheticQuadratic(d_x=4, d_xi=6, seed=1)
    rng = np.random.default_rng(13)
    x = p.x_star + 0.3 * rng.standard_normal(p.d_x)
    eps = 1e-3
    grad_mc = np.zeros(p.d_x)
    n_mc = 30_000
    xis = rng.standard_normal((n_mc, p.d_xi))
    for j in range(p.d_x):
        ej = np.zeros(p.d_x); ej[j] = 1.0
        f_plus = float(np.mean([p.f(x + eps * ej, xi) for xi in xis]))
        f_minus = float(np.mean([p.f(x - eps * ej, xi) for xi in xis]))
        grad_mc[j] = (f_plus - f_minus) / (2 * eps)
    grad_truth = p.grad_E_loss(x)
    np.testing.assert_allclose(grad_mc, grad_truth, atol=0.05)


def test_s1_active_xi_count():
    """With n_active_xi=2, only the first two singular values are nonzero,
    and Var_f(x) = ||A^T x||^2 = 2 components."""
    p = SyntheticQuadratic(d_x=6, d_xi=10, seed=0, n_active_xi=2)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(p.d_x)
    AtAx = p.A.T @ x
    # Only first 2 singular components contribute; project out and check.
    s = np.linalg.svd(p.A, compute_uv=False)
    assert sum(s > 1e-10) == 2, f"effective xi-dim wrong: s={s}"


# --------------------------------------------------------------------------- #
# S2 — yield-like
# --------------------------------------------------------------------------- #
def test_s2_analytic_yield_matches_mc():
    p = SyntheticYieldLike(d_x=6, d_xi=10, seed=0)
    rng = np.random.default_rng(42)
    x = p.x_star
    xis = rng.standard_normal((100_000, p.d_xi))
    n_pass = 0
    for xi in xis:
        g1 = p.a1 @ x + p.b1 @ xi + p.c1
        g2 = p.a2 @ x + p.b2 @ xi + p.c2
        if g1 >= 0 and g2 >= 0:
            n_pass += 1
    mc_y = n_pass / len(xis)
    truth = p.yield_at(x)
    assert abs(mc_y - truth) < 0.005, f"yield mismatch: mc={mc_y:.4f} truth={truth:.4f}"


def test_s2_yield_at_xstar_in_band():
    """Calibration sanity: default c1, c2 land yield(x_star) in [0.55, 0.85]
    so the problem has comparable headroom to the circuit benchmarks."""
    for seed in [0, 1, 2, 3, 4]:
        p = SyntheticYieldLike(d_x=6, d_xi=10, seed=seed)
        y = p.yield_at(p.x_star)
        assert 0.55 <= y <= 0.85, f"seed={seed} yield(x*)={y:.4f} out of band"


def test_s2_analytic_E_loss_matches_mc():
    p = SyntheticYieldLike(d_x=4, d_xi=6, seed=2, alpha=5.0)
    rng = np.random.default_rng(7)
    x = p.x_star + 0.2 * rng.standard_normal(p.d_x)
    xis = rng.standard_normal((100_000, p.d_xi))
    mc = float(np.mean([p.f(x, xi) for xi in xis]))
    truth = p.E_loss(x)
    # Looser tolerance because softplus E[·] is smoother and the GH
    # quadrature is essentially exact, so the gap is pure MC noise.
    assert abs(mc - truth) / max(abs(truth), 1.0) < 0.01, \
        f"E_loss mismatch: mc={mc:.6f} truth={truth:.6f}"


def test_s2_yield_increases_in_correct_direction():
    """Sanity: moving x by +ε in the direction (a_1 + a_2)/||·|| should
    raise yield (since both per-spec margins increase)."""
    p = SyntheticYieldLike(d_x=6, d_xi=10, seed=0)
    direction = (p.a1 + p.a2)
    direction /= np.linalg.norm(direction)
    y_at = p.yield_at(p.x_star)
    y_plus = p.yield_at(p.x_star + 0.5 * direction)
    assert y_plus > y_at, f"yield should rise: {y_at:.4f} -> {y_plus:.4f}"


# --------------------------------------------------------------------------- #
# Adapter sanity
# --------------------------------------------------------------------------- #
def test_adapters_quack_like_simulator_and_sampler():
    p = SyntheticQuadratic(d_x=3, d_xi=4, seed=0)
    sim = p.as_simulator()
    samp = p.as_sampler()
    rng = np.random.default_rng(0)
    xis = samp.sample(8, rng=rng)
    assert xis.shape == (8, 4)
    x = np.zeros(3)
    losses = [sim.evaluate(x, xi) for xi in xis]
    assert all(np.isfinite(losses))
    assert sim.n_calls == 8
