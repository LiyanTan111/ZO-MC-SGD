"""sanity tests for cs_amp_5stage.

Mirrors tests/test_cs_amp_3stage_simulator.py with 5 stages.
"""
import numpy as np

from benchmarks.cs_amp_5stage import spec as cascade


def test_dimensions():
    assert len(cascade.DESIGN_VARS) == 30
    assert len(cascade.VARIATION_VARS) == 42
    assert cascade.X_NOMINAL.shape == (30,)
    assert cascade.X_LO.shape == (30,)
    assert cascade.X_HI.shape == (30,)
    s = cascade.make_sampler(scale=1.0)
    assert s.dim == 42


def test_nominal_metrics_finite():
    sim = cascade.build_simulator()
    loss, m = sim.evaluate_with_metrics(cascade.X_NOMINAL, np.zeros(42))
    for k in ("gain_db", "ugbw_hz", "phase_margin_deg", "power_w"):
        assert np.isfinite(m[k]), f"metric {k} not finite: {m[k]}"
    assert np.isfinite(loss)
    # Cascade should hit substantially higher gain than 3-stage (~75 dB).
    assert m["gain_db"] > 80.0, f"5-stage cascade gain too low: {m['gain_db']}"
    assert m["power_w"] > 0.0


def test_all_five_stages_bias_in_range():
    """Every stage's DC operating point must sit strictly between 0 and Vdd.

    A stage whose node has railed is not amplifying, so the cascade's AC
    response would be meaningless. The per-stage nodes are the five gate nodes
    (each stage's output drives the next stage's gate) plus the final output.
    """
    from simulators.ngspice_subprocess import run_ngspice_subprocess

    sim = cascade.build_simulator()
    netlist = sim._render(cascade.X_NOMINAL, np.zeros(42))
    nodes = ["gate1", "gate2", "gate3", "gate4", "gate5", "out"]

    vectors = run_ngspice_subprocess(
        netlist=netlist, analysis_cmd="op", plot_name="op1", nodes=nodes,
    )

    vdd = getattr(cascade, "VDD", 1.0)
    for k in nodes:
        assert vectors.get(k) is not None, f"node {k} missing from the OP solution"
        v = float(np.real(np.asarray(vectors[k])).flat[0])
        assert 0.0 < v < vdd, f"{k} railed at {v:.4f} V (Vdd = {vdd})"


def test_simulator_deterministic_no_cache_bug():
    sim = cascade.build_simulator()
    xi0 = np.zeros(42)
    loss1, m1 = sim.evaluate_with_metrics(cascade.X_NOMINAL, xi0)
    loss2, m2 = sim.evaluate_with_metrics(cascade.X_NOMINAL, xi0)
    assert loss1 == loss2
    for k in m1:
        assert m1[k] == m2[k], f"metric {k} drifted: {m1[k]} vs {m2[k]}"


def test_per_stage_independence_of_design_vars():
    """Bumping stage-5 design vars must not change stage-1's bias."""
    import sys
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared

    sim = cascade.build_simulator()
    x_alt = cascade.X_NOMINAL.copy()
    # Stage-5 design-var indices are at positions 24-29 (5×6=30, 0-indexed).
    x_alt[24] *= 1.5   # bump stage-5 W_n by 50%
    x_alt[28] *= 2.0   # bump stage-5 R_fb 2×

    netlist0 = sim._render(cascade.X_NOMINAL, np.zeros(42))
    netlist_alt = sim._render(x_alt, np.zeros(42))
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
            assert abs(v_nbias1 - v_nbias1_nom) < 1e-6
            assert abs(v_out1 - v_out1_nom) < 1e-6
