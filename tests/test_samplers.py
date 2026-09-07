import numpy as np

from zo_yield.samplers import (
    CorrelatedGaussianSampler,
    GaussianMixtureSampler,
    IndependentGaussianSampler,
    IndependentUniformSampler,
)


def test_independent_gaussian_moments():
    s = IndependentGaussianSampler(mean=[1.0, -2.0, 3.0], std=[0.5, 1.0, 2.0])
    rng = np.random.default_rng(0)
    samples = s.sample(100_000, rng=rng)
    np.testing.assert_allclose(samples.mean(0), [1.0, -2.0, 3.0], atol=2e-2)
    np.testing.assert_allclose(samples.std(0), [0.5, 1.0, 2.0], atol=2e-2)


def test_correlated_gaussian_moments():
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    s = CorrelatedGaussianSampler(mean=[0.0, 0.0], cov=cov)
    rng = np.random.default_rng(0)
    samples = s.sample(200_000, rng=rng)
    emp_cov = np.cov(samples, rowvar=False)
    np.testing.assert_allclose(emp_cov, cov, atol=2e-2)


def test_independent_uniform_moments():
    s = IndependentUniformSampler(low=[-1.0, 0.0], high=[1.0, 4.0])
    rng = np.random.default_rng(0)
    samples = s.sample(200_000, rng=rng)
    np.testing.assert_allclose(samples.mean(0), [0.0, 2.0], atol=1e-2)
    var_expected = np.array([(2.0) ** 2 / 12.0, (4.0) ** 2 / 12.0])
    np.testing.assert_allclose(samples.var(0), var_expected, atol=2e-2)


def test_mixture_sampler_runs():
    s = GaussianMixtureSampler(
        weights=[0.5, 0.5],
        means=[[0.0, 0.0], [3.0, 3.0]],
        covs=[np.eye(2), 0.5 * np.eye(2)],
    )
    rng = np.random.default_rng(0)
    samples = s.sample(10_000, rng=rng)
    # mean should be ~ mid-point
    np.testing.assert_allclose(samples.mean(0), [1.5, 1.5], atol=0.1)
