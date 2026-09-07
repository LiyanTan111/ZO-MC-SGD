"""Subprocess-isolated ngspice executor.

**Why this exists.** PySpice's ``NgSpiceShared`` (and the underlying ngspice DLL
loaded via cffi) caches ``.model`` definitions across ``load_circuit`` calls
within the same Python process. ``remove_circuit`` / ``remcirc`` / ``alter`` /
``altermod`` do **not** invalidate the model cache. As a result, all in-process
calls after the first one return AC results from the FIRST loaded model card,
not the current one.

This was discovered while investigating OTA feasibility —
process variations (especially ΔVth) had measurable expected effects in
fresh-subprocess sims but ZERO effect in repeated in-process sims.

**Fix.** Run each ngspice simulation in a fresh subprocess. The Python-startup
overhead is ~120 ms per call on this machine, which is ~40× the actual sim
time (3 ms). For experiments doing ~10³ sims this adds ~2 minutes of overhead
per experiment — acceptable for correctness.

**Interface.** :func:`run_ngspice_subprocess` takes a complete netlist string,
runs ``op`` + the user's analysis command, and returns selected node-vector
data via a temporary JSON file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Iterable

#: Repository root, passed to the worker so it can import this package even when
#: the package is not installed on the subprocess's default sys.path.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Worker script that the subprocess executes. We embed it inline so callers
# don't need to ship a separate file.
_WORKER_SCRIPT = r'''
import json, sys, os
_repo = os.environ.get("ZO_YIELD_REPO")
if _repo:
    sys.path.insert(0, _repo)

import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceShared

def to_real(arr):
    a = np.asarray(arr)
    return np.real(a)

if __name__ == "__main__":
    in_path = sys.argv[1]
    with open(in_path) as f:
        req = json.load(f)
    netlist = req["netlist"]
    analysis_cmd = req["analysis_cmd"]
    nodes = req["nodes"]
    plot_name = req["plot_name"]
    op_nodes = req.get("op_nodes", [])   # : per-device OP currents

    ng = NgSpiceShared.new_instance()
    ng.load_circuit(netlist)
    ng.exec_command("op")
    ng.exec_command(analysis_cmd)
    plot = ng.plot(simulation=None, plot_name=plot_name)

    out = {"plot_name": plot_name, "available_keys": list(plot.keys()), "vectors": {}}
    for name in nodes:
        if name not in plot.keys():
            out["vectors"][name] = None
            continue
        wave = np.array(plot[name].to_waveform())
        # Save real and imag separately to round-trip via JSON
        out["vectors"][name] = {"real": wave.real.tolist(), "imag": wave.imag.tolist()}

    # Operating-point scalar reads. Read from op1 plot.
    op_out = {}
    if op_nodes:
        op_plot = ng.plot(simulation=None, plot_name="op1")
        for op_name in op_nodes:
            if op_name not in op_plot.keys():
                op_out[op_name] = None
            else:
                wv = np.array(op_plot[op_name].to_waveform())
                # Operating point is a single complex (or real) scalar.
                op_out[op_name] = {"real": float(wv.real.flat[0]),
                                    "imag": float(wv.imag.flat[0])}
    out["op_scalars"] = op_out
    out_path = req.get("out_path") or (in_path + ".out")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(out_path)
'''


def run_ngspice_subprocess(
    netlist: str,
    analysis_cmd: str,
    plot_name: str,
    nodes: Iterable[str],
    timeout_s: float = 30.0,
    op_nodes: Iterable[str] | None = None,
):
    """Execute one netlist+analysis in a fresh ngspice subprocess.

    Args:
      netlist     : full netlist text (title + .model + devices + .control + .end).
      analysis_cmd: e.g. ``"ac dec 20 1 1g"``.
      plot_name   : ngspice plot name to read after the analysis (e.g. "ac1").
      nodes       : iterable of node names to extract from the analysis plot.
      op_nodes    : iterable of operating-point scalars to
                    read from the ``op1`` plot — e.g. ``["i(vdd)"]`` for total
                    supply current. Returned in the second-element dict.
      timeout_s   : kill the subprocess if it runs past this.

    Returns:
      * if ``op_nodes`` is ``None`` (default): vectors dict, same as before
        preserved for back-compat with existing callers.
      * if ``op_nodes`` is provided: ``(vectors_dict, op_scalars_dict)``
        tuple. op scalar values are floats (imag part discarded — OP is DC).
    """
    import numpy as np

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="/tmp"
    ) as in_f:
        json.dump(dict(
            netlist=netlist,
            analysis_cmd=analysis_cmd,
            plot_name=plot_name,
            nodes=list(nodes),
            op_nodes=list(op_nodes) if op_nodes else [],
        ), in_f)
        in_path = in_f.name

    out_path = in_path + ".out"
    try:
        env = os.environ.copy()
        env.setdefault("ZO_YIELD_REPO", REPO_ROOT)
        proc = subprocess.run(
            [sys.executable, "-c", _WORKER_SCRIPT, in_path],
            capture_output=True,
            timeout=timeout_s,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ngspice subprocess failed (rc={proc.returncode}): "
                f"stderr={proc.stderr.decode()[:300]}"
            )
        if not os.path.isfile(out_path):
            raise RuntimeError("ngspice subprocess produced no output JSON")
        with open(out_path) as f:
            res = json.load(f)
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass

    out = {}
    for name, vec in res.get("vectors", {}).items():
        if vec is None:
            out[name] = None
        else:
            out[name] = np.asarray(vec["real"]) + 1j * np.asarray(vec["imag"])

    if not op_nodes:
        return out

    op_out = {}
    for name, sc in res.get("op_scalars", {}).items():
        op_out[name] = None if sc is None else float(sc["real"])
    return out, op_out
