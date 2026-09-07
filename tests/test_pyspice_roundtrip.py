"""Round-trip test: a resistor divider through the subprocess ngspice executor.

Uses :func:`run_ngspice_subprocess` rather than an in-process
``NgSpiceShared``: the shared library caches ``.model`` cards across
``load_circuit`` calls, so an in-process read is only correct for the first
circuit loaded in a given interpreter. Every simulation in this project goes
through the subprocess executor for that reason, and so does this test.
"""
import numpy as np
import pytest

try:
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared  # noqa: F401

    HAS_NGSPICE = True
except Exception:
    HAS_NGSPICE = False


pytestmark = pytest.mark.skipif(not HAS_NGSPICE, reason="ngspice shared library not available")


def test_resistor_divider():
    """V_in -> R1 -> mid -> R2 -> 0; verify Vmid = vin * R2/(R1+R2)."""
    from simulators.ngspice_subprocess import run_ngspice_subprocess

    vin, r1, r2 = 5.0, 1000.0, 1000.0
    netlist = (
        "resistor_divider\n"
        f"V1 in 0 {vin}\n"
        f"R1 in mid {r1}\n"
        f"R2 mid 0 {r2}\n"
        ".op\n"
        ".end\n"
    )

    vectors = run_ngspice_subprocess(
        netlist=netlist,
        analysis_cmd="op",
        plot_name="op1",
        nodes=["mid"],
    )

    v_mid = float(np.real(np.asarray(vectors["mid"])).flat[0])
    assert abs(v_mid - vin * r2 / (r1 + r2)) < 1e-3
