"""pick a hard X_init per circuit.

For each of {CS amp, cs_amp_3stage}, generate 8 perturbed candidates
(seeds 0..7), evaluate yield at the calibrated σ_scale and α
on a fixed 64-MC ξ-set, and select the candidate with yield in
[0.10, 0.30] closest to 0.20. Falls back to a tighter [0.8, 1.25]
perturbation scale if no candidate qualifies.

Output per circuit: experiments/results/<circuit>/x_init_hard.json
Plus aggregate: experiments/results/x_init_calibration.md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from zo_yield.evaluation import evaluate_design
from zo_yield.init import perturb_x_init


CIRCUITS = {
    "cs_amp": dict(spec="benchmarks.common_source_amp.spec", scale=3.0),
    "cs_amp_3stage": dict(spec="benchmarks.cs_amp_3stage.spec", scale=1.5),
    "cs_amp_5stage": dict(spec="benchmarks.cs_amp_5stage.spec", scale=0.1),
    "cs_se_miller": dict(spec="benchmarks.cs_se_miller.spec", scale=0.1),
    "cs_se_miller_3stage": dict(spec="benchmarks.cs_se_miller_3stage.spec", scale=1.0),
}

N_CANDIDATES = 8
N_MC = 64
TARGET_LO, TARGET_HI = 0.10, 0.30
TARGET_BEST = 0.20


def calibrate_one(circuit: str, repo: str) -> dict:
    print(f"\n=== {circuit} ===", flush=True)
    cfg = CIRCUITS[circuit]
    mod = __import__(cfg["spec"], fromlist=["spec"])
    sim = mod.build_simulator()
    sampler = mod.make_sampler(scale=cfg["scale"])
    specs = mod.build_specs() if hasattr(mod, "build_specs") else mod.SPECS
    yield_xis = sampler.sample(N_MC, rng=np.random.default_rng(555))

    class FixedSampler:
        dim = sampler.dim
        def sample(self, n_mc, rng=None): return yield_xis[:n_mc]

    def score(seed: int, scale_lo: float, scale_hi: float):
        x = perturb_x_init(mod.X_NOMINAL, seed=seed,
                            scale_lo=scale_lo, scale_hi=scale_hi,
                            x_lo=mod.X_LO, x_hi=mod.X_HI)
        try:
            res = evaluate_design(sim, x, FixedSampler(), specs, n_mc=N_MC)
            return x, float(res["yield_"]), float(res["E_loss"])
        except Exception as e:
            print(f"  seed={seed} eval failed: {e}", flush=True)
            return x, float("nan"), float("nan")

    # Pass 1 — default scale [0.7, 1.4]. If yields too HIGH, widen the
    # perturbation (a tighter [0.8, 1.25] band is the
    # wrong direction for our too-robust CS amp; actual data
    # shows we need WIDER perturbations to land in [0.10, 0.30]).
    SCALE_LADDER = [
        (0.7, 1.4),    # default
        (0.5, 2.0),    # wider — perturb device sizes by ~2× either way
        (0.3, 3.0),    # very wide — multi-x perturbation
        (0.15, 6.0),   # extreme — may trigger many bound-clips
    ]

    candidates = []
    chosen_scale_pair = None
    fallback_used = False

    for level, (slo, shi) in enumerate(SCALE_LADDER):
        if level > 0:
            print(f"  no candidate in [{TARGET_LO}, {TARGET_HI}] at scale="
                  f"[{SCALE_LADDER[level-1][0]}, {SCALE_LADDER[level-1][1]}]; "
                  f"widening to [{slo}, {shi}]", flush=True)
            fallback_used = True
        for s in range(N_CANDIDATES):
            x, y, l = score(s, scale_lo=slo, scale_hi=shi)
            candidates.append(dict(seed=s, scale_lo=slo, scale_hi=shi,
                                    yield_=y, E_loss=l, x=x.tolist()))
            print(f"  seed={s} (scale=[{slo}, {shi}]): yield={y:.4f}", flush=True)
        in_band = [c for c in candidates if TARGET_LO <= c["yield_"] <= TARGET_HI]
        if in_band:
            chosen_scale_pair = (slo, shi)
            break

    in_band = [c for c in candidates if TARGET_LO <= c["yield_"] <= TARGET_HI]

    # Decision
    if in_band:
        chosen = min(in_band, key=lambda c: abs(c["yield_"] - TARGET_BEST))
        verdict = "in_band"
    else:
        chosen = min(candidates, key=lambda c: abs(c["yield_"] - TARGET_BEST))
        verdict = "fallback_closest_to_target"
        print(f"  ⚠ still no candidate in band; picking closest-to-{TARGET_BEST}: "
              f"yield={chosen['yield_']:.4f}", flush=True)

    print(f"  CHOSEN seed={chosen['seed']}, scale=[{chosen['scale_lo']}, "
          f"{chosen['scale_hi']}], yield={chosen['yield_']:.4f}", flush=True)

    # Persist
    out_dir = os.path.join(repo, "experiments", "results", circuit)
    os.makedirs(out_dir, exist_ok=True)
    out = dict(
        circuit=circuit,
        n_mc=N_MC,
        scale=cfg["scale"],
        target_band=[TARGET_LO, TARGET_HI],
        verdict=verdict,
        fallback_used=fallback_used,
        chosen=chosen,
        all_candidates=candidates,
    )
    with open(os.path.join(out_dir, "x_init_hard.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Wrote {out_dir}/x_init_hard.json", flush=True)
    return out


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo, "experiments", "results")
    results = {c: calibrate_one(c, repo) for c in CIRCUITS}

    md = ["# Hard X_init calibration — \n",
          f"Each circuit: 8 candidates from `perturb_x_init(X_NOMINAL, seed)`, "
          f"evaluated with n_mc={N_MC} on a fixed ξ-set (seed=555). Target: "
          f"yield in [{TARGET_LO}, {TARGET_HI}], closest to {TARGET_BEST}.\n",
          "## Results\n",
          "| Circuit | Verdict | Chosen seed | Scale band | yield(X_init) | Fallback used |",
          "| ------- | ------- | ----------- | ---------- | ------------- | ------------- |"]
    for c, r in results.items():
        chosen = r["chosen"]
        md.append(f"| {c} | {r['verdict']} | {chosen['seed']} | "
                  f"[{chosen['scale_lo']}, {chosen['scale_hi']}] | "
                  f"{chosen['yield_']:.4f} | {r['fallback_used']} |")
    md.append("")
    md.append("## Per-candidate yield (default scale [0.7, 1.4])\n")
    md.append("| Circuit | seed | yield |")
    md.append("| ------- | ---- | ----- |")
    for c, r in results.items():
        for cand in r["all_candidates"]:
            if cand["scale_lo"] == 0.7:
                md.append(f"| {c} | {cand['seed']} | {cand['yield_']:.4f} |")
    with open(os.path.join(out_dir, "x_init_calibration.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nWrote {out_dir}/x_init_calibration.md")


if __name__ == "__main__":
    main()
