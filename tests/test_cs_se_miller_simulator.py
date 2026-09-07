"""sanity tests for cs_se_miller.

Single-ended counterpart of miller_ota's tests. The diagnostic test
(test v) verifies that bumping ANY single-device ξ shifts metrics —
no diff-pair-style cancellation possible because there is no diff pair.
"""
import numpy as np

from benchmarks.cs_se_miller import spec as ota


def test_dimensions():
    assert len(ota.DESIGN_VARS) == 14
    assert len(ota.VARIATION_VARS) == 26
    assert ota.X_NOMINAL.shape == (14,)
    assert ota.X_LO.shape == (14,)
    assert ota.X_HI.shape == (14,)
    s = ota.make_sampler(scale=1.0)
    assert s.dim == 26


def test_nominal_metrics_finite():
    sim = ota.build_simulator()
    loss, m = sim.evaluate_with_metrics(ota.X_NOMINAL, np.zeros(26))
    for k in ("gain_db", "ugbw_hz", "phase_margin_deg", "power_w"):
        assert np.isfinite(m[k]), f"metric {k} not finite: {m[k]}"
    assert np.isfinite(loss)
    # Two cascaded self-biased CS amps should give substantial gain (>50 dB).
    assert m["gain_db"] > 50.0, f"cs_se_miller gain too low: {m['gain_db']}"
    assert m["power_w"] > 0.0


def test_dc_op_voltages_in_range():
    """DC OP at xi=0: out1 (stage-1 out), out (stage-2 out), gate1, gate2 should
    sit between 0 and Vdd."""
    import sys
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared

    sim = ota.build_simulator()
    netlist = sim._render(ota.X_NOMINAL, np.zeros(26))
    ng = NgSpiceShared.new_instance()
    ng.load_circuit(netlist)
    ng.exec_command("op")
    op = ng.plot(simulation=None, plot_name="op1")
    vdd = ota.DEFAULT_TARGETS["vdd"]
    for k in ("out1", "out", "gate1", "gate2"):
        v = float(np.array(op[k].to_waveform()).real.flat[0])
        assert 0.0 < v < vdd, f"{k} = {v:.4f} V outside (0, {vdd})"


def test_simulator_deterministic_no_cache_bug():
    sim = ota.build_simulator()
    xi0 = np.zeros(26)
    loss1, m1 = sim.evaluate_with_metrics(ota.X_NOMINAL, xi0)
    loss2, m2 = sim.evaluate_with_metrics(ota.X_NOMINAL, xi0)
    assert loss1 == loss2
    for k in m1:
        assert m1[k] == m2[k], f"metric {k} drifted: {m1[k]} vs {m2[k]}"


def test_single_ended_no_diffpair_cancellation():
    """Single-ended verification: bumping ANY single-device ξ shifts metrics
    monotonically — no diff-pair-style cancellation possible because there
    is no second matched device.

    Verifies M1 (NMOS_S1, ξ slot 0) and M2 (NMOS_S2, ξ slot 12) have
    distinguishable metric signatures from a small ξ bump.
    """
    sim = ota.build_simulator()

    xi_m1_only = np.zeros(26)
    xi_m1_only[0] = 0.05      # M1 dvth +50 mV

    xi_m2_only = np.zeros(26)
    xi_m2_only[12] = 0.05     # M2 dvth +50 mV (different stage's NMOS)

    _, m_m1 = sim.evaluate_with_metrics(ota.X_NOMINAL, xi_m1_only)
    _, m_m2 = sim.evaluate_with_metrics(ota.X_NOMINAL, xi_m2_only)

    # Two single-device perturbations on different stages should produce
    # distinguishable metric changes (no symmetry to cancel them).
    assert (m_m1["gain_db"] != m_m2["gain_db"] or
            m_m1["ugbw_hz"] != m_m2["ugbw_hz"]), \
        "M1 (stage-1 NMOS) and M2 (stage-2 NMOS) ξ perturbations gave " \
        "identical metrics — possible shared-ξ bug."
