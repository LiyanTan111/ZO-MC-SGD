"""Unit tests for zo_yield/loss.py."""
import math

import pytest

from zo_yield.loss import (
    Penalty,
    _smooth_shortfall,
    combined_loss,
    evaluate_penalty,
)


def test_alpha_none_matches_hard_relu():
    """alpha=None must return max(0, delta) exactly."""
    for delta in [-5.0, -1.0, -1e-9, 0.0, 1e-9, 1.0, 5.0]:
        assert abs(_smooth_shortfall(delta, alpha=None) - max(0.0, delta)) < 1e-12


def test_alpha_inf_limit_matches_hard_relu():
    """As alpha grows, smoothed shortfall approaches max(0, delta)."""
    for delta in [-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0]:
        s = _smooth_shortfall(delta, alpha=1e6)
        h = max(0.0, delta)
        assert abs(s - h) < 1e-4, f"delta={delta}, soft={s}, hard={h}"


def test_delta_zero_residual():
    """At delta=0, softplus = ln(2)/alpha."""
    for alpha in [0.1, 1.0, 10.0]:
        s = _smooth_shortfall(0.0, alpha=alpha)
        expected = math.log(2.0) / alpha
        assert abs(s - expected) < 1e-12, f"alpha={alpha}, s={s}, expected={expected}"


def test_derivative_matches_sigmoid():
    """d/dδ softplus(αδ)/α = sigmoid(αδ). Check numerically at three points."""
    alpha = 1.5
    h = 1e-6
    for delta in [-1.0, 0.0, 0.7]:
        f_plus = _smooth_shortfall(delta + h, alpha=alpha)
        f_minus = _smooth_shortfall(delta - h, alpha=alpha)
        num_deriv = (f_plus - f_minus) / (2 * h)
        analytic = 1.0 / (1.0 + math.exp(-alpha * delta))   # sigmoid(αδ)
        assert abs(num_deriv - analytic) < 1e-5, (
            f"delta={delta}, num={num_deriv}, sigmoid={analytic}"
        )


def test_combined_loss_sums_correctly():
    """combined_loss = objective + Σ weight·shortfall."""
    pens = [
        Penalty("a", lambda s: s["a"], kind="ge", target=10.0, weight=1.0, alpha=None),
        Penalty("b", lambda s: s["b"], kind="le", target=5.0, weight=2.0, alpha=None),
    ]
    sim = {"a": 8.0, "b": 7.0}    # a violated by 2 (target=10), b by 2 (>5)
    expected = 3.14 + 1.0 * 2.0 + 2.0 * 2.0    # objective + 2 + 4 = 9.14
    assert abs(combined_loss(pens, sim, objective=3.14) - expected) < 1e-12


def test_evaluate_penalty_le_kind():
    """'le' kind: positive delta when value > target."""
    p = Penalty("p", lambda s: s["v"], kind="le", target=1.0, weight=3.0, alpha=None)
    # v=2 -> delta=1 -> shortfall=1 -> contribution 3
    assert abs(evaluate_penalty(p, {"v": 2.0}) - 3.0) < 1e-12
    # v=0.5 -> delta=-0.5 -> shortfall=0
    assert abs(evaluate_penalty(p, {"v": 0.5})) < 1e-12


def test_smooth_upper_tail_numerical_stability():
    """Very large alpha*delta should not overflow."""
    s = _smooth_shortfall(delta=100.0, alpha=10.0)
    # alpha*delta = 1000, softplus(1000) ≈ 1000, /alpha = 100
    assert abs(s - 100.0) < 1e-9
