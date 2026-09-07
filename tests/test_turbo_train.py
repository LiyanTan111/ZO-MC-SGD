"""Unit tests for the TuRBO-1 main loop (3 tests)."""
import numpy as np

from methods.turbo.config import TuRBOConfig
from methods.turbo.train import train_turbo


def _quadratic(d, opt):
    def f(u):
        return float(np.sum((np.asarray(u) - opt) ** 2))
    return f


def test_end_to_end_completes_in_unit_cube():
    d = 6
    opt = np.full(d, 0.3)
    res = train_turbo(_quadratic(d, opt), d=d, budget_queries=100,
                      cfg=TuRBOConfig(), seed=0)
    assert res["x_best"].shape == (d,)
    assert np.all(res["x_best"] >= 0.0) and np.all(res["x_best"] <= 1.0)
    # Should make real progress toward the optimum (loose).
    assert res["y_best"] < 0.5


def test_query_count_matches_budget():
    d = 5
    res = train_turbo(_quadratic(d, np.full(d, 0.5)), d=d, budget_queries=60,
                      cfg=TuRBOConfig(), seed=1)
    assert res["n_queries"] == 60


def test_best_so_far_monotone():
    d = 6
    res = train_turbo(_quadratic(d, np.full(d, 0.7)), d=d, budget_queries=80,
                      cfg=TuRBOConfig(), seed=2)
    trace = np.array(res["best_trace"])
    assert np.all(np.diff(trace) <= 1e-12)     # non-increasing
