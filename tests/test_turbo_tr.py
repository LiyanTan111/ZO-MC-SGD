"""Unit tests for TuRBO trust-region management (4 tests)."""
import numpy as np

from methods.turbo.config import TuRBOConfig
from methods.turbo.trust_region import TrustRegion


def test_expand_after_tau_succ():
    cfg = TuRBOConfig()
    tr = TrustRegion(d=6, cfg=cfg)
    L0 = tr.L
    ybest = 1.0
    for k in range(cfg.tau_succ):          # 3 consecutive improvements
        tr.update_after_eval(ybest - (k + 1), ybest)
    assert np.isclose(tr.L, min(cfg.length_max, 2.0 * L0))


def test_shrink_after_tau_fail():
    cfg = TuRBOConfig()
    tr = TrustRegion(d=6, cfg=cfg)
    L0 = tr.L
    for _ in range(cfg.tau_fail(6)):       # 4 consecutive non-improvements
        tr.update_after_eval(5.0, 1.0)
    assert np.isclose(tr.L, L0 / 2.0)


def test_restart_flag_when_below_min():
    cfg = TuRBOConfig()
    tr = TrustRegion(d=6, cfg=cfg)
    # Force many failures to drive L below L_min.
    for _ in range(cfg.tau_fail(6) * 12):
        tr.update_after_eval(5.0, 1.0)
    assert tr.restart_needed and tr.L < cfg.length_min


def test_bounding_box_within_unit_cube():
    cfg = TuRBOConfig()
    tr = TrustRegion(d=4, cfg=cfg)
    tr.set_center(np.array([0.0, 1.0, 0.5, 0.95]))   # near/at boundaries
    lo, hi = tr.get_bounding_box(gp_lengthscales=np.array([1.0, 2.0, 0.5, 1.0]))
    assert np.all(lo >= 0.0) and np.all(hi <= 1.0)
    assert np.all(hi >= lo)
