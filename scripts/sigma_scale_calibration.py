"""Calibrate σ_scale per circuit so yield(X_NOMINAL) ∈ [0.30, 0.85].

Sweeps σ_scale ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0} with n_mc=64 fixed ξ-set per
scale, picks the lowest scale that lands yield in the target band.

Usage:
    python scripts/sigma_scale_calibration.py {cs_amp|cs_se_miller|...} [suffix]

The optional ``suffix`` (e.g. ``_v2``) is appended to all output file
basenames (uses this to write `sigma_scale_calibration_v2.*`
without overwriting the existing files).
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np


CIRCUIT_REGISTRY = {
    "cs_amp": "benchmarks.common_source_amp.spec",
    "cs_amp_3stage": "benchmarks.cs_amp_3stage.spec",
    "cs_amp_5stage": "benchmarks.cs_amp_5stage.spec",
    "cs_se_miller": "benchmarks.cs_se_miller.spec",
    "cs_se_miller_3stage": "benchmarks.cs_se_miller_3stage.spec",
}

# Extended downward because the OTA-v2 sampler's per-device ΔW
# is much more aggressive than v1; even scale=0.5 gives yield ~ 0.11. The
# The target band [0.30, 0.85] forces us to sample finer at the low
# end. CS amp / synthetic samplers are unaffected (their make_sampler
# ignores small-scale aliasing because their σ values are tiny in absolute
# terms; the extra scales are just more data points).
SCALES = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
N_MC = 64
TARGET_LOW = 0.30
TARGET_HIGH = 0.85


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CIRCUIT_REGISTRY:
        print(f"usage: {sys.argv[0]} {{cs_amp|cs_se_miller|...}}", file=sys.stderr)
        sys.exit(1)
    circuit = sys.argv[1]
    suffix = sys.argv[2] if len(sys.argv) >= 3 else ""
    mod = __import__(CIRCUIT_REGISTRY[circuit], fromlist=["spec"])

    from zo_yield.evaluation import evaluate_design

    sim = mod.build_simulator()
    specs = mod.build_specs() if hasattr(mod, "build_specs") else mod.SPECS

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments", "results", circuit
    )
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for scale in SCALES:
        sampler = mod.make_sampler(scale=scale)
        rng = np.random.default_rng(555 + int(scale * 100))
        try:
            res = evaluate_design(sim, mod.X_NOMINAL, sampler, specs,
                                   n_mc=N_MC, rng=rng)
            y = res["yield_"]
            psr = res["per_spec_passrate"]
            el = res["E_loss"]
        except Exception as e:
            print(f"  scale={scale}: failed {e}")
            y = float("nan"); psr = {}; el = float("nan")
        rows.append(dict(scale=scale, yield_=y, E_loss=el, per_spec=psr))
        print(f"  scale={scale}: yield={y:.4f}, passrate={psr}", flush=True)

    csv_path = os.path.join(out_dir, f"sigma_scale_calibration{suffix}.csv")
    spec_keys = list(rows[-1]["per_spec"].keys()) if rows[-1]["per_spec"] else []
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scale", "yield", "E_loss"] + [f"pr_{k}" for k in spec_keys])
        for r in rows:
            w.writerow([r["scale"], f"{r['yield_']:.4f}", f"{r['E_loss']:.6f}"]
                       + [f"{r['per_spec'].get(k, ''):.4f}" if r['per_spec'] else ''
                           for k in spec_keys])

    # Decision: lowest scale with yield in [TARGET_LOW, TARGET_HIGH].
    chosen = None
    for r in rows:
        if TARGET_LOW <= r["yield_"] <= TARGET_HIGH:
            chosen = r; break

    md = [f"# σ_scale calibration — {circuit}\n",
          f"Target: yield(X_NOMINAL) ∈ [{TARGET_LOW}, {TARGET_HIGH}], "
          f"n_mc = {N_MC} (fixed ξ-set per scale).\n",
          "## Yield vs σ_scale\n",
          "| σ_scale | yield(X_NOMINAL) | E[loss] |",
          "| ------- | ----------------- | ------- |"]
    for r in rows:
        md.append(f"| {r['scale']:.1f} | {r['yield_']:.4f} | {r['E_loss']:.4f} |")
    md.append("")
    if chosen:
        md.append(f"## Chosen σ_scale: **{chosen['scale']}** (yield = {chosen['yield_']:.4f})")
        md.append(f"\nPer-spec passrate at chosen scale:")
        for k, v in chosen["per_spec"].items():
            md.append(f"  - {k}: {v:.4f}")
        md.append(f"\n**Criterion (c) — Headroom: PASS**")
    else:
        md.append("## No σ_scale qualifies")
        md.append("\nyield(X_NOMINAL) jumps from > 0.85 to < 0.30 between adjacent scales:")
        for i in range(len(rows)-1):
            md.append(f"  - {rows[i]['scale']:.1f} → {rows[i+1]['scale']:.1f}: "
                      f"{rows[i]['yield_']:.3f} → {rows[i+1]['yield_']:.3f}")
        md.append(f"\n**Criterion (c) — Headroom: FAIL** (no scale in target band).")
        md.append(f"The α calibration and clean run for this "
                  f"circuit are skipped. Recommended fix: add per-device W variation "
                  f"(Pelgrom σ_W) so that asymmetric mismatch breaks the OTA's "
                  f"current-mirror symmetry and produces graded yield-vs-design.")
    summary_path = os.path.join(out_dir, f"sigma_scale_calibration{suffix}.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(md))

    out = dict(circuit=circuit, scales=SCALES, rows=[
        dict(scale=r["scale"], yield_=r["yield_"], E_loss=r["E_loss"],
             per_spec=r["per_spec"]) for r in rows],
        chosen_scale=chosen["scale"] if chosen else None,
        criterion_c_pass=(chosen is not None))
    with open(os.path.join(out_dir, f"sigma_scale_calibration{suffix}.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
