"""Smoke test for zo_yield.evaluation.evaluate_design."""
import numpy as np
import pytest

from zo_yield.evaluation import Spec, evaluate_design, to_serializable


class FakeSim:
    """Trivial simulator returning gain=70-xi[0], ugbw=1e6 + xi[1]*1e5."""

    def evaluate_with_metrics(self, x, xi):
        gain = 70.0 - xi[0]
        ugbw = 1e6 + xi[1] * 1e5
        m = dict(gain_db=gain, ugbw_hz=ugbw)
        loss = max(0.0, 60 - gain) + max(0.0, 1e6 - ugbw) / 1e6
        return loss, m


class FakeSampler:
    dim = 2

    def sample(self, n, rng=None):
        rng = rng or np.random.default_rng()
        return rng.standard_normal((n, 2))


def test_evaluate_design_basic():
    sim = FakeSim()
    sampler = FakeSampler()
    specs = [
        Spec(
            name="gain_db",
            extract=lambda m: m["gain_db"],
            satisfies=lambda v: v >= 60.0,
            margin=lambda v: 60.0 - v,
        ),
        Spec(
            name="ugbw_hz",
            extract=lambda m: m["ugbw_hz"],
            satisfies=lambda v: v >= 1e6,
            margin=lambda v: 1e6 - v,
        ),
    ]
    rng = np.random.default_rng(0)
    res = evaluate_design(sim, x=np.zeros(3), sampler=sampler, specs=specs, n_mc=2000, rng=rng)
    # gain ~ N(70, 1) => P[gain >= 60] near 1; ugbw ~ N(1e6, 1e5) => P >= 1e6 ~ 0.5
    assert res["per_spec_passrate"]["gain_db"] > 0.95
    assert 0.4 < res["per_spec_passrate"]["ugbw_hz"] < 0.6
    assert 0.4 < res["yield_"] < 0.6   # joint
    # margins arrays have right length
    assert len(res["per_spec_margin"]["gain_db"]) == 2000
    # to_serializable round-trip
    s = to_serializable(res)
    assert "yield" in s and "yield_" not in s
    assert isinstance(s["per_spec_margin"]["gain_db"], list)
