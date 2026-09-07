"""sanity tests for cs_se_miller_3stage."""
import numpy as np

from benchmarks.cs_se_miller_3stage import spec as ota


def test_dimensions():
    assert len(ota.DESIGN_VARS) == 22
    assert len(ota.VARIATION_VARS) == 38
    assert ota.X_NOMINAL.shape == (22,)
    assert ota.X_LO.shape == (22,)
    assert ota.X_HI.shape == (22,)
    s = ota.make_sampler(scale=1.0)
    assert s.dim == 38


def test_nominal_metrics_finite():
    sim = ota.build_simulator()
    loss, m = sim.evaluate_with_metrics(ota.X_NOMINAL, np.zeros(38))
    for k in ("gain_db", "ugbw_hz", "phase_margin_deg", "power_w"):
        assert np.isfinite(m[k]), f"metric {k} not finite: {m[k]}"
    assert np.isfinite(loss)
    # 3-stage cascade should give substantial gain (≥ 60 dB).
    assert m["gain_db"] > 60.0, f"3-stage cascade gain too low: {m['gain_db']}"
    assert m["power_w"] > 0.0


def test_dc_op_voltages_in_range():
    """DC OP at xi=0: stage outputs (out1, out2, out) and gates within (0, Vdd)."""
    import sys
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared

    sim = ota.build_simulator()
    netlist = sim._render(ota.X_NOMINAL, np.zeros(38))
    ng = NgSpiceShared.new_instance()
    ng.load_circuit(netlist)
    ng.exec_command("op")
    op = ng.plot(simulation=None, plot_name="op1")
    vdd = ota.DEFAULT_TARGETS["vdd"]
    for k in ("out1", "out2", "out", "gate1", "gate2", "gate3"):
        v = float(np.array(op[k].to_waveform()).real.flat[0])
        assert 0.0 < v < vdd, f"{k} = {v:.4f} V outside (0, {vdd})"


def test_simulator_deterministic_no_cache_bug():
    sim = ota.build_simulator()
    xi0 = np.zeros(38)
    loss1, m1 = sim.evaluate_with_metrics(ota.X_NOMINAL, xi0)
    loss2, m2 = sim.evaluate_with_metrics(ota.X_NOMINAL, xi0)
    assert loss1 == loss2
    for k in m1:
        assert m1[k] == m2[k], f"metric {k} drifted: {m1[k]} vs {m2[k]}"


def test_per_stage_independence_of_design_vars():
    """Bumping stage-3 design vars must not change stage-1's bias."""
    import sys
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared

    sim = ota.build_simulator()
    x_alt = ota.X_NOMINAL.copy()
    # Stage-3 design-var indices are at positions 12-17.
    x_alt[12] *= 1.5   # bump stage-3 W_n by 50%
    x_alt[16] *= 2.0   # bump stage-3 R_fb by 2×

    netlist0 = sim._render(ota.X_NOMINAL, np.zeros(38))
    netlist_alt = sim._render(x_alt, np.zeros(38))

    v_out1_nom = v_nbias1_nom = None
    for net, label in [(netlist0, "nominal"), (netlist_alt, "alt")]:
        ng = NgSpiceShared.new_instance()
        ng.load_circuit(net)
        ng.exec_command("op")
        op = ng.plot(simulation=None, plot_name="op1")
        v_nbias1 = float(np.array(op["nbias1"].to_waveform()).real.flat[0])
        v_out1 = float(np.array(op["out1"].to_waveform()).real.flat[0])
        if label == "nominal":
            v_nbias1_nom, v_out1_nom = v_nbias1, v_out1
        else:
            assert abs(v_nbias1 - v_nbias1_nom) < 1e-6, \
                f"stage-1 bias drift from stage-3 perturbation: {v_nbias1} vs {v_nbias1_nom}"
            assert abs(v_out1 - v_out1_nom) < 1e-6, \
                f"stage-1 out drift: {v_out1} vs {v_out1_nom}"
