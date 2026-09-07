"""Unit tests for the RobustAnalog reward (paper eqs 2-3, 3 tests)."""
import math

from methods.robustanalog.config import RAConfig
from methods.robustanalog.multi_task_env import Constraint, per_corner_reward

CFG = RAConfig()

CONS = [
    Constraint("gain_db", lambda m: m["gain_db"], target=60.0, direction="ge"),
    Constraint("ugbw_hz", lambda m: m["ugbw_hz"], target=50e6, direction="ge"),
    Constraint("power_w", lambda m: m["power_w"], target=1e-3, direction="le"),
]


def test_all_satisfied_returns_plus_point_two():
    metrics = dict(gain_db=70.0, ugbw_hz=90e6, power_w=0.3e-3)
    r = per_corner_reward(metrics, CONS, CFG)
    assert r == CFG.reward_satisfied   # == +0.2


def test_one_violated_is_negative_and_equals_rel_dist():
    # gain violated (50 < 60), ugbw + power satisfied.
    metrics = dict(gain_db=50.0, ugbw_hz=90e6, power_w=0.3e-3)
    r = per_corner_reward(metrics, CONS, CFG)
    assert r < CFG.reward_sat_threshold
    expected = (50.0 - 60.0) / (abs(50.0) + abs(60.0) + CFG.reward_eps)
    assert math.isclose(r, expected, rel_tol=1e-9)


def test_le_direction_power_over_max_is_negative():
    # power violated (2mW > 1mW upper bound) -> negative contribution.
    metrics = dict(gain_db=70.0, ugbw_hz=90e6, power_w=2e-3)
    r = per_corner_reward(metrics, CONS, CFG)
    assert r < CFG.reward_sat_threshold
    expected = (1e-3 - 2e-3) / (abs(2e-3) + abs(1e-3) + CFG.reward_eps)
    assert math.isclose(r, expected, rel_tol=1e-9)
    # And a feasible power (0.5mW <= 1mW) must NOT contribute a penalty:
    metrics_ok = dict(gain_db=70.0, ugbw_hz=90e6, power_w=0.5e-3)
    assert per_corner_reward(metrics_ok, CONS, CFG) == CFG.reward_satisfied
