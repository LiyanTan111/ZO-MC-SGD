"""sanity for perturb_x_init."""
import numpy as np

from zo_yield.init import perturb_x_init


def test_deterministic_per_seed():
    x_nom = np.array([1.0, 2.0, 3.0, 4.0])
    a = perturb_x_init(x_nom, seed=0)
    b = perturb_x_init(x_nom, seed=0)
    np.testing.assert_array_equal(a, b)


def test_seed_changes_output():
    x_nom = np.array([1.0, 2.0, 3.0, 4.0])
    a = perturb_x_init(x_nom, seed=0)
    b = perturb_x_init(x_nom, seed=1)
    assert not np.allclose(a, b)


def test_within_per_var_box():
    x_nom = np.array([1.0, 2.0, 3.0, 4.0])
    a = perturb_x_init(x_nom, seed=0, scale_lo=0.7, scale_hi=1.4)
    factors = a / x_nom
    assert np.all(factors >= 0.7 - 1e-12)
    assert np.all(factors <= 1.4 + 1e-12)


def test_clipping_to_design_box():
    x_nom = np.array([1.0, 2.0])
    x_lo = np.array([0.9, 1.5])
    x_hi = np.array([1.1, 2.5])
    out = perturb_x_init(x_nom, seed=42, scale_lo=0.5, scale_hi=2.0,
                          x_lo=x_lo, x_hi=x_hi)
    assert np.all(out >= x_lo)
    assert np.all(out <= x_hi)


def test_real_circuit_dimensions():
    """Smoke: works with real-circuit X_NOMINAL shapes (n=6 and n=18)."""
    n_csa = 6
    x_csa = np.random.default_rng(0).uniform(1e-6, 1e-4, size=n_csa)
    out_csa = perturb_x_init(x_csa, seed=0)
    assert out_csa.shape == (n_csa,)

    n_3stage = 18
    x_3 = np.random.default_rng(0).uniform(1e-6, 1e-4, size=n_3stage)
    out_3 = perturb_x_init(x_3, seed=0)
    assert out_3.shape == (n_3stage,)
