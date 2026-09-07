"""Unit tests for TuRBO Thompson-sampling acquisition (2 tests)."""
import numpy as np

from methods.turbo.acquisition import generate_candidates, propose
from methods.turbo.config import TuRBOConfig
from methods.turbo.gp_model import GPModel


def _fit_gp(d, rng):
    X = rng.random((12, d))
    y = np.sum((X - 0.5) ** 2, axis=1)
    return GPModel().fit(X, y, rng=rng)


def test_proposed_point_inside_tr_box():
    cfg = TuRBOConfig()
    rng = np.random.default_rng(0)
    d = 5
    gp = _fit_gp(d, rng)
    center = np.full(d, 0.5)
    lo = np.clip(center - 0.2, 0, 1)
    hi = np.clip(center + 0.2, 0, 1)
    cand = generate_candidates(center, lo, hi, cfg.n_cand(d), rng)
    assert np.all(cand >= lo[None, :] - 1e-9) and np.all(cand <= hi[None, :] + 1e-9)
    from methods.turbo.acquisition import thompson_select
    pt = thompson_select(gp, cand, rng)
    assert np.all(pt >= lo - 1e-9) and np.all(pt <= hi + 1e-9)


def test_acquisition_reproducible_given_seed():
    cfg = TuRBOConfig()
    d = 4
    gp = _fit_gp(d, np.random.default_rng(7))
    center = np.full(d, 0.5)
    lo = np.clip(center - 0.3, 0, 1)
    hi = np.clip(center + 0.3, 0, 1)
    p1 = propose(gp, center, lo, hi, cfg, np.random.default_rng(99))
    p2 = propose(gp, center, lo, hi, cfg, np.random.default_rng(99))
    assert np.allclose(p1, p2)
