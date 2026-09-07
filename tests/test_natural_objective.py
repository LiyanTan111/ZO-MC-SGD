"""Unit tests for the natural (yield-MC) objective (task Box 1-2)."""
import numpy as np

from methods.natural_objective.objective import (make_natural_eval_fn,
                                                 natural_objective)


class _Spec:
    def __init__(self, name, thresh):
        self.name = name; self.thresh = thresh
    def extract(self, m):
        return m[self.name]
    def satisfies(self, v):
        return v >= self.thresh
    def margin(self, v):
        return self.thresh - v


class _MockSim:
    """metrics['g'] = mean(x); 'p' = 1.0 (always passes 'p')."""
    def __init__(self):
        self.n_calls = 0
    def evaluate_with_metrics(self, x, xi):
        self.n_calls += 1
        return 0.0, dict(g=float(np.mean(x)) + float(xi[0]), p=1.0)


def test_yield_in_unit_interval_and_spice_equals_n_mc():
    sim = _MockSim()
    specs = [_Spec("g", 0.0), _Spec("p", 0.5)]
    xi = np.random.default_rng(0).standard_normal((10, 2))
    res = natural_objective(sim, specs, np.array([0.0, 0.0]), xi)
    assert 0.0 <= res["yield_"] <= 1.0          # passes/n in [0,1]
    assert res["e_loss"] == -res["yield_"]      # minimize negative yield
    assert sim.n_calls == 10                    # per-call SPICE == n_mc


def test_all_pass_and_all_fail_extremes():
    specs = [_Spec("g", -1e9)]                  # threshold so low: always pass
    sim = _MockSim()
    xi = np.zeros((10, 2))
    assert natural_objective(sim, specs, np.array([5.0, 5.0]), xi)["yield_"] == 1.0
    specs2 = [_Spec("g", 1e9)]                  # threshold so high: never pass
    assert natural_objective(_MockSim(), specs2, np.array([0.0, 0.0]), xi)["yield_"] == 0.0


def test_each_optimizer_completes_one_run():
    # Mock objective: yield peaks at x_norm = 1 (the x_init point). Reuses the
    # real optimizers with a natural-style eval_fn (E_loss = -yield).
    from zo_yield.baselines import bo_optimize, cmaes_optimize, pso_optimize
    n = 4
    x_init = np.ones(n); lo = np.full(n, 0.2); hi = np.full(n, 2.0)

    def eval_fn(x_norm, rng=None, n_mc=10):
        y = float(np.exp(-np.sum((np.asarray(x_norm) - 1.0) ** 2)))  # in (0,1]
        return dict(E_loss=-y, yield_=y)

    for opt in (bo_optimize, cmaes_optimize, pso_optimize):
        res = opt(eval_fn, x_init, lo, hi, budget=100, n_mc=10, seed=0)
        assert res["x_best"].shape == (n,)
        assert res["spice_actual"] <= 100
