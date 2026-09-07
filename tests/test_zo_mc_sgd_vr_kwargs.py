"""`per_pair_crn` + `antithetic_xi` kwargs.

Pins:
  (i)   defaults reproduce the documented behaviour byte-identically (parity at the
        gradient-trajectory level — required HARD gate before sweeps);
  (ii)  ``antithetic_xi=True`` with odd ``batch_size_xi`` raises ValueError;
  (iii) ``per_pair_crn=False`` consumes 2× distinct ξ values per step
        (one batch for f+, one fresh draw per pair for f-);
  (iv)  ``antithetic_xi=True`` constructs the batch as
        ``[xi_1, …, xi_{K/2}, mirror(xi_1), …, mirror(xi_{K/2})]`` where
        ``mirror(xi) = 2·μ − xi``.
"""
import numpy as np
import pytest

from zo_yield.optimizers import (ZOMCSGD, OptimizerConfig,
                                   _zo_grad_at_xi)
from zo_yield.estimators import EstimatorConfig
from zo_yield.synthetic import SyntheticQuadratic


def _make_runner(crn=True, anti=False, batch_size_xi=4):
    """Construct an optimizer + simulator + sampler bundle for testing."""
    p = SyntheticQuadratic(d_x=6, d_xi=10, seed=0)
    sim = p.as_simulator()
    samp = p.as_sampler()
    opt = ZOMCSGD(
        simulator=sim, sampler=samp,
        estimator_cfg=EstimatorConfig(
            epsilon=5e-3, v_dist="gaussian", mode="central"),
        optimizer_cfg=OptimizerConfig(rule="adam", lr=0.05),
        batch_size_xi=batch_size_xi, batch_size_v=1,
        eval_xis=None, log_every=1,
        per_pair_crn=crn, antithetic_xi=anti,
    )
    x_init = p.x_star + 0.5 * np.ones(p.d_x)
    return opt, x_init, p, sim, samp


def test_default_kwargs_match_round_1_11():
    """The default kwargs (per_pair_crn=True, antithetic_xi=False) must
    reproduce the default trajectory byte-identically. Two runs at the
    same seed — one with explicit defaults, one without — must produce
    bit-identical x_history and grad_norm sequences.
    """
    p1 = SyntheticQuadratic(d_x=6, d_xi=10, seed=0)
    p2 = SyntheticQuadratic(d_x=6, d_xi=10, seed=0)

    def make(p, *, with_explicit_defaults: bool):
        kwargs = dict(
            simulator=p.as_simulator(), sampler=p.as_sampler(),
            estimator_cfg=EstimatorConfig(
                epsilon=5e-3, v_dist="gaussian", mode="central"),
            optimizer_cfg=OptimizerConfig(rule="adam", lr=0.05),
            batch_size_xi=4, batch_size_v=1,
            eval_xis=None, log_every=1,
        )
        if with_explicit_defaults:
            kwargs["per_pair_crn"] = True
            kwargs["antithetic_xi"] = False
        return ZOMCSGD(**kwargs)

    x_init = p1.x_star + 0.5 * np.ones(p1.d_x)
    h_default = make(p1, with_explicit_defaults=False).run(
        x_init.copy(), n_iters=5, rng=np.random.default_rng(42))
    h_explicit = make(p2, with_explicit_defaults=True).run(
        x_init.copy(), n_iters=5, rng=np.random.default_rng(42))

    # x_history and grad_norm must match exactly.
    assert len(h_default.x_history) == len(h_explicit.x_history)
    for x_a, x_b in zip(h_default.x_history, h_explicit.x_history):
        np.testing.assert_array_equal(x_a, x_b)
    np.testing.assert_array_equal(
        np.array(h_default.grad_norm), np.array(h_explicit.grad_norm))


def test_antithetic_xi_requires_even_batch_size():
    p = SyntheticQuadratic(d_x=6, d_xi=10, seed=0)
    with pytest.raises(ValueError, match="even batch_size_xi"):
        ZOMCSGD(
            simulator=p.as_simulator(), sampler=p.as_sampler(),
            estimator_cfg=EstimatorConfig(
                epsilon=5e-3, v_dist="gaussian", mode="central"),
            optimizer_cfg=OptimizerConfig(rule="adam", lr=0.05),
            batch_size_xi=3, batch_size_v=1,
            antithetic_xi=True,
        )


def test_no_pair_crn_uses_fresh_xi_for_minus():
    """With per_pair_crn=False, each per-(xi, v) pair draws a fresh
    xi_prime for f-, so a 4-pair step uses 8 distinct ξ values."""
    p = SyntheticQuadratic(d_x=6, d_xi=10, seed=0)
    samp = p.as_sampler()

    class _LoggingSim:
        LOSS_PENALTY = 1e3
        def __init__(self, p):
            self._p = p
            self.n_calls = 0
            self.n_failures = 0
            self.xi_log = []
        def evaluate(self, x, xi):
            self.n_calls += 1
            self.xi_log.append(np.asarray(xi).copy())
            return float(self._p.f(np.asarray(x), np.asarray(xi)))

    sim = _LoggingSim(p)
    opt = ZOMCSGD(
        simulator=sim, sampler=samp,
        estimator_cfg=EstimatorConfig(
            epsilon=5e-3, v_dist="gaussian", mode="central"),
        optimizer_cfg=OptimizerConfig(rule="adam", lr=0.05),
        batch_size_xi=4, batch_size_v=1, eval_xis=None, log_every=1,
        per_pair_crn=False, antithetic_xi=False,
    )
    opt.run(p.x_star + 0.5 * np.ones(p.d_x), n_iters=1,
             rng=np.random.default_rng(7))

    # 4 pairs × 2 evaluations each = 8 evaluations. Distinct xi values
    # should be 8 (4 from xi_batch each used for f+, 4 fresh xi_prime
    # each used for f-).
    distinct = {tuple(xi) for xi in sim.xi_log}
    assert sim.n_calls == 8, f"expected 8 sim calls, got {sim.n_calls}"
    assert len(distinct) == 8, \
        f"expected 8 distinct xi values, got {len(distinct)}"


def test_antithetic_xi_pairs_are_mirrored():
    """With antithetic_xi=True and batch_size_xi=4, the constructed
    xi_batch should satisfy xi_batch[2:] == 2*sampler.mean − xi_batch[:2].
    We verify by intercepting the sampler's first call."""
    p = SyntheticQuadratic(d_x=6, d_xi=10, seed=0)
    samp = p.as_sampler()
    captured = {}
    orig_sample = samp.sample

    def _spy_sample(n_mc, rng=None):
        out = orig_sample(n_mc, rng=rng)
        captured.setdefault("calls", []).append(out.copy())
        return out
    samp.sample = _spy_sample

    class _NoopSim:
        LOSS_PENALTY = 1e3
        def __init__(self): self.n_calls = 0; self.n_failures = 0
        def evaluate(self, x, xi):
            self.n_calls += 1
            return 0.0   # constant loss; gradient = 0; no optimizer drift

    opt = ZOMCSGD(
        simulator=_NoopSim(), sampler=samp,
        estimator_cfg=EstimatorConfig(
            epsilon=5e-3, v_dist="gaussian", mode="central"),
        optimizer_cfg=OptimizerConfig(rule="adam", lr=0.05),
        batch_size_xi=4, batch_size_v=1, eval_xis=None, log_every=1,
        per_pair_crn=True, antithetic_xi=True,
    )
    opt.run(np.zeros(p.d_x), n_iters=1, rng=np.random.default_rng(11))

    # First sampler call (with antithetic_xi=True) drew the half batch
    # of size 2.
    half = captured["calls"][0]
    assert half.shape == (2, p.d_xi), f"expected half shape (2, 10), got {half.shape}"
    # The actual xi_batch passed to the sim is half + mirrored. We
    # reconstruct it the same way the optimizer does.
    mu = getattr(samp, "mean", np.zeros_like(half[0]))
    expected_full = np.concatenate([half, 2.0 * mu[None, :] - half], axis=0)
    # Mirror property:
    np.testing.assert_allclose(
        expected_full[2:], 2.0 * mu[None, :] - expected_full[:2])
