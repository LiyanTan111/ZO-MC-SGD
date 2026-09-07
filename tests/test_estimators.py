"""Unit tests for two-point ZO gradient estimators (Task 4.1 acceptance)."""
import numpy as np
import pytest

from zo_yield.estimators import (
    EstimatorConfig,
    mini_batch_estimate,
    two_point_central,
    two_point_forward,
    sample_direction,
    phi_factor,
)


def make_quadratic(n=10, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M @ M.T + np.eye(n)  # SPD
    b = rng.standard_normal(n)

    def f(x):
        return 0.5 * x @ A @ x - b @ x

    def grad(x):
        return A @ x - b

    return f, grad, A, b


@pytest.mark.parametrize("v_dist", ["gaussian", "sphere"])
def test_mean_converges_to_true_grad(v_dist):
    rng = np.random.default_rng(123)
    f, grad, _, _ = make_quadratic(n=10)
    x = rng.standard_normal(10)
    g_true = grad(x)
    g_hat, _ = mini_batch_estimate(
        f, x, epsilon=1e-3, n_samples=8000, v_dist=v_dist, mode="central", rng=rng
    )
    rel_err = np.linalg.norm(g_hat - g_true) / np.linalg.norm(g_true)
    # ZO Gaussian has variance O(n) per sample direction; with 8000 samples and n=10
    # we expect rel_err well under 10%. Sphere has higher variance from the phi=n factor.
    tol = 0.08 if v_dist == "gaussian" else 0.20
    assert rel_err < tol, f"v_dist={v_dist}: rel_err={rel_err:.3f}"


def test_variance_decays_as_inverse_n_samples():
    rng = np.random.default_rng(1)
    f, grad, _, _ = make_quadratic(n=10)
    x = rng.standard_normal(10)
    g_true = grad(x)

    sample_counts = [50, 200, 800, 3200]
    errs = []
    for n_s in sample_counts:
        # average variance over a few replicates
        sub_errs = []
        for rep in range(5):
            g_hat, _ = mini_batch_estimate(
                f, x, epsilon=1e-3, n_samples=n_s, v_dist="gaussian",
                mode="central", rng=np.random.default_rng(100 + rep + n_s),
            )
            sub_errs.append(np.sum((g_hat - g_true) ** 2))
        errs.append(np.mean(sub_errs))
    # log-log slope should be ~ -1
    slope, _ = np.polyfit(np.log(sample_counts), np.log(errs), 1)
    assert -1.3 < slope < -0.7, f"variance slope vs n_samples = {slope:.2f}"


def test_two_point_forward_matches_finite_difference():
    rng = np.random.default_rng(2)
    f, grad, _, _ = make_quadratic(n=5)
    x = rng.standard_normal(5)
    v = rng.standard_normal(5)
    eps = 1e-5
    f_x = f(x)
    g, calls = two_point_forward(f, x, eps, v, f_x=f_x, v_dist="gaussian")
    assert calls == 1
    # Hand-compute the same expression
    expected = (f(x + eps * v) - f_x) / eps * v
    np.testing.assert_allclose(g, expected, rtol=1e-10)


def test_phi_factors():
    assert phi_factor(7, "gaussian") == 1.0
    assert phi_factor(7, "sphere") == 7.0
    assert phi_factor(7, "coordinate") == 7.0


def test_sample_direction_shapes_and_norms():
    rng = np.random.default_rng(0)
    n = 20
    g = sample_direction(n, "gaussian", rng)
    s = sample_direction(n, "sphere", rng)
    c = sample_direction(n, "coordinate", rng)
    assert g.shape == (n,)
    assert s.shape == (n,)
    assert c.shape == (n,)
    assert np.isclose(np.linalg.norm(s), 1.0)
    assert np.isclose(c.sum(), 1.0) and np.count_nonzero(c) == 1
