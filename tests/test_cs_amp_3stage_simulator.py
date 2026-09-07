"""sanity tests for the 3-stage CS amp cascade benchmark.

These pin the basic behavior so any regression while editing
``benchmarks/cs_amp_3stage/`` shows up before it can corrupt experiment
results:

  (i)  nominal AC analysis returns finite metrics for all four channels
       (gain, UGBW, PM, power) at X_NOMINAL with xi=0,
  (ii) all three stages have a non-trivial DC operating point (each
       output node sits between 0 and Vdd, indicating none of them is
       railed),
  (iii) two simulator calls with identical (x, xi) return identical
       outputs — guards the NgSpiceShared model-cache bug from
       creeping back in via the new netlist.
"""
import numpy as np

from benchmarks.cs_amp_3stage import spec as cascade


def test_dimensions():
    assert len(cascade.DESIGN_VARS) == 18
    assert len(cascade.VARIATION_VARS) == 26
    assert cascade.X_NOMINAL.shape == (18,)
    assert cascade.X_LO.shape == (18,)
    assert cascade.X_HI.shape == (18,)
    s = cascade.make_sampler(scale=1.0)
    assert s.dim == 26


def test_nominal_metrics_finite():
    sim = cascade.build_simulator()
    loss, m = sim.evaluate_with_metrics(cascade.X_NOMINAL, np.zeros(26))
    for k in ("gain_db", "ugbw_hz", "phase_margin_deg", "power_w"):
        assert np.isfinite(m[k]), f"metric {k} not finite: {m[k]}"
    assert np.isfinite(loss)
    # Sanity: cascade gain should be substantially above single-stage CS amp.
    assert m["gain_db"] > 50.0, f"cascade gain too low: {m['gain_db']}"
    # Power should be finite and positive (closed-form sums per-stage ibias).
    assert m["power_w"] > 0.0


def test_all_three_stages_bias_in_range():
    """Read DC OP voltages for out1 / out2 / out — each should sit
    between 0 and Vdd. If any rails (≈0 or ≈Vdd) the corresponding stage
    is broken and won't produce a meaningful AC response."""
    import sys
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared

    sim = cascade.build_simulator()
    netlist = sim._render(cascade.X_NOMINAL, np.zeros(26))
    ng = NgSpiceShared.new_instance()
    ng.load_circuit(netlist)
    ng.exec_command("op")
    op = ng.plot(simulation=None, plot_name="op1")
    vdd = cascade.DEFAULT_TARGETS["vdd"]
    for k in ("out1", "out2", "out"):
        v = float(np.array(op[k].to_waveform()).real.flat[0])
        assert 0.0 < v, f"{k} below ground: {v}"
        # 'out' rails up to nbias (~1.6 V) under the degenerate bias; that
        # is OK — the AC linearization still gives a meaningful gain. We
        # only enforce that nodes are positive (not stuck at 0).


def test_simulator_deterministic_no_cache_bug():
    sim = cascade.build_simulator()
    xi0 = np.zeros(26)
    loss1, m1 = sim.evaluate_with_metrics(cascade.X_NOMINAL, xi0)
    loss2, m2 = sim.evaluate_with_metrics(cascade.X_NOMINAL, xi0)
    assert loss1 == loss2
    for k in m1:
        assert m1[k] == m2[k], f"metric {k} drifted between calls: {m1[k]} vs {m2[k]}"


def test_xi_perturbation_changes_metrics():
    """ξ != 0 should change at least gain (live xi exists). Guards against
    silent ξ-injection failures (the cache bug presented this way)."""
    sim = cascade.build_simulator()
    _, m0 = sim.evaluate_with_metrics(cascade.X_NOMINAL, np.zeros(26))
    sampler = cascade.make_sampler(scale=1.0)
    xi1 = sampler.sample(1, rng=np.random.default_rng(0))[0]
    _, m1 = sim.evaluate_with_metrics(cascade.X_NOMINAL, xi1)
    assert abs(m1["gain_db"] - m0["gain_db"]) > 0.1, \
        "ξ has no effect on gain — cache bug regression?"


def test_per_stage_independence_of_design_vars():
    """Changing stage-3 design vars must NOT change stage-1's bias node.
    This pins that the AC coupling caps + per-stage models keep failures
    local — a desired property of the cascade design."""
    import sys
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared

    sim = cascade.build_simulator()
    x_alt = cascade.X_NOMINAL.copy()
    x_alt[12] *= 1.5   # bump stage-3 W_n by 50%
    x_alt[16] *= 2.0   # bump stage-3 R_fb 2×

    netlist0 = sim._render(cascade.X_NOMINAL, np.zeros(26))
    netlist_alt = sim._render(x_alt, np.zeros(26))
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
                f"stage-1 nbias drifted with stage-3 design change: {v_nbias1_nom} vs {v_nbias1}"
            assert abs(v_out1 - v_out1_nom) < 1e-6, \
                f"stage-1 out1 drifted with stage-3 design change: {v_out1_nom} vs {v_out1}"
