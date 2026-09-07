"""Unit tests for the TuRBO GP model (3 tests)."""
import numpy as np

from methods.turbo.gp_model import GPModel


def test_posterior_mean_recovers_sin():
    rng = np.random.default_rng(0)
    X = np.linspace(0, 2 * np.pi, 24).reshape(-1, 1)
    y = np.sin(X).reshape(-1) + 0.01 * rng.standard_normal(len(X))
    gp = GPModel(n_restarts=3, max_iter=80).fit(X, y, rng=rng)
    Xq = np.array([[0.5], [1.5], [2.5], [3.5], [4.5]])
    mu, _ = gp.predict(Xq)
    assert np.all(np.abs(mu - np.sin(Xq).reshape(-1)) < 0.1)


def test_variance_shrinks_with_data():
    rng = np.random.default_rng(1)
    X = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(3 * X).reshape(-1)
    gp = GPModel().fit(X, y, rng=rng)
    # Variance at an observed location vs a far unobserved location.
    _, var_obs = gp.predict(np.array([[0.0]]))
    _, var_far = gp.predict(np.array([[5.0]]))   # far outside the data range
    assert var_far[0] > var_obs[0]


def test_thompson_sample_deterministic_given_seed():
    rng = np.random.default_rng(2)
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    y = np.cos(4 * X).reshape(-1)
    gp = GPModel().fit(X, y, rng=rng)
    Xc = np.linspace(0, 1, 30).reshape(-1, 1)
    f1 = gp.thompson_sample(Xc, np.random.default_rng(42))
    f2 = gp.thompson_sample(Xc, np.random.default_rng(42))
    assert np.allclose(f1, f2)
    # A different seed gives a different realization.
    f3 = gp.thompson_sample(Xc, np.random.default_rng(43))
    assert not np.allclose(f1, f3)
