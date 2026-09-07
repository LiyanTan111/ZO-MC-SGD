"""Unit tests for PCGrad (4 tests)."""
import torch

from methods.robustanalog.pcgrad import project_conflicting


def test_single_task_returns_input_unchanged():
    g = torch.tensor([1.0, -2.0, 3.0])
    out = project_conflicting([g])
    assert torch.allclose(out, g)


def test_fully_conflicting_gradients_cancel():
    # Two anti-parallel gradients (cos = -1): each projects onto the other,
    # leaving ~zero; their sum is ~zero.
    g1 = torch.tensor([1.0, 0.0, 0.0])
    g2 = torch.tensor([-1.0, 0.0, 0.0])
    out = project_conflicting([g1, g2])
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_non_conflicting_gradients_sum():
    # Orthogonal gradients (cos = 0, not < 0): no projection -> plain sum.
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([0.0, 1.0])
    out = project_conflicting([g1, g2])
    assert torch.allclose(out, g1 + g2)


def test_order_invariance_with_fixed_seed():
    # Deterministic given a fixed permutation seed: same inputs + same seed
    # -> identical output across calls.
    g1 = torch.tensor([1.0, -0.5, 0.3])
    g2 = torch.tensor([-0.8, 1.0, -0.2])
    g3 = torch.tensor([0.4, 0.4, -1.0])
    rng_a = torch.Generator(); rng_a.manual_seed(123)
    rng_b = torch.Generator(); rng_b.manual_seed(123)
    out_a = project_conflicting([g1, g2, g3], rng=rng_a)
    out_b = project_conflicting([g1, g2, g3], rng=rng_b)
    assert torch.allclose(out_a, out_b)
