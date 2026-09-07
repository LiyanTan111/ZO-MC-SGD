import numpy as np

from simulators.analytical import AnalyticalQuadratic
from zo_yield.estimators import EstimatorConfig
from zo_yield.optimizers import (
    OptimizerConfig,
    ZOMCSGD,
    ZOOptimizer,
)
from zo_yield.samplers import IndependentGaussianSampler


def test_zo_optimizer_adam_quadratic():
    """Deterministic ZO with Adam converges on a SPD quadratic (Task 4.4 acceptance)."""
    rng = np.random.default_rng(0)
    n = 5
    M = rng.standard_normal((n, n))
    A = M @ M.T + np.eye(n)
    mu = rng.standard_normal(n)
    sim = AnalyticalQuadratic(A=A, mu=mu)
    # deterministic xi = 0
    xi = np.zeros(n)

    from zo_yield.estimators import mini_batch_estimate

    opt = ZOOptimizer(
        x0=mu + 1.0, cfg=OptimizerConfig(rule="adam", lr=0.05, schedule="const")
    )
    rng2 = np.random.default_rng(1)
    for _ in range(500):
        g, _ = mini_batch_estimate(
            lambda x: sim.evaluate(x, xi),
            opt.x,
            epsilon=1e-3,
            n_samples=10,
            v_dist="gaussian",
            mode="central",
            rng=rng2,
        )
        opt.step(g)
    err = np.linalg.norm(opt.x - mu)
    assert err < 1e-1, f"ZO+Adam did not converge: ||x - mu|| = {err:.4f}"


def test_option1_converges_on_stochastic_quadratic():
    """Option 1 on f(x, xi) = 0.5(x - mu - xi)^T A (x - mu - xi), xi ~ N(0, sigma^2 I)."""
    rng = np.random.default_rng(0)
    n = 4
    A = np.eye(n)  # easy
    mu = np.array([1.0, -1.0, 2.0, 0.5])
    sim = AnalyticalQuadratic(A=A, mu=mu)
    sampler = IndependentGaussianSampler(mean=np.zeros(n), std=0.1 * np.ones(n))
    eval_xis = sampler.sample(200, rng=np.random.default_rng(99))
    runner = ZOMCSGD(
        simulator=sim,
        sampler=sampler,
        estimator_cfg=EstimatorConfig(epsilon=1e-3, v_dist="gaussian", mode="central"),
        optimizer_cfg=OptimizerConfig(rule="adam", lr=0.05, schedule="const"),
        batch_size_xi=8,
        batch_size_v=4,
        eval_xis=eval_xis,
        log_every=20,
    )
    history = runner.run(x_init=np.zeros(n), n_iters=200, rng=rng)
    err = np.linalg.norm(history.x_history[-1] - mu)
    assert err < 0.15, f"ZO-MC-SGD did not converge: ||x - mu|| = {err:.4f}"

