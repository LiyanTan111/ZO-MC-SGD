"""α calibration: the loss--yield rank-correlation gate.

The smoothed-spec surrogate is the optimization target but yield is the score,
so the two only have to rank designs the same way. This picks the softplus
sharpness α per circuit to make that true.

Build an 80-design calibration set, evaluate yield once per design (cached),
then for each candidate α compute:
  - loss(design) under that α
  - Spearman ρ(loss, yield) across designs
  - mean_yield_top10 = mean yield over the 10% lowest-loss designs

Pick the α with the most negative ρ, tie-broken by mean_yield_top10. A circuit
whose best correlation stays above -0.7 is set aside for redesign rather than
optimized on unreliable footing.

Usage:
    python scripts/loss_yield_calibration.py cs_amp_3stage [suffix]

The optional ``suffix`` (e.g. ``_v2``) is appended to all output file basenames,
and the σ-scale is then read from `sigma_scale_calibration<suffix>.json`. With
no suffix the σ-scale comes from configs/<circuit>.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from zo_yield.circuit_config import CIRCUITS, load_circuit_config

CIRCUIT_REGISTRY = {c: load_circuit_config(c)["spec_module"] for c in CIRCUITS}
CHOSEN_SCALE = {c: float(load_circuit_config(c)["sigma_scale"]) for c in CIRCUITS}

# Candidate α-sets, from near-hard (sharp) to very soft.
ALPHA_CANDIDATES = {
    "cs_amp": [
        dict(name="near_hard",   alpha_gain=5.0,  alpha_ugbw_log=10.0, alpha_power_mW=10.0),
        dict(name="sharp",       alpha_gain=1.0,  alpha_ugbw_log=3.0,  alpha_power_mW=2.0),
        dict(name="default",     alpha_gain=0.5,  alpha_ugbw_log=1.0,  alpha_power_mW=0.5),
        dict(name="soft",        alpha_gain=0.2,  alpha_ugbw_log=0.5,  alpha_power_mW=0.2),
        dict(name="very_soft",   alpha_gain=0.1,  alpha_ugbw_log=0.2,  alpha_power_mW=0.1),
    ],
    # cs_amp_3stage uses the same loss form (gain + UGBW + power_mW) as
    # the single-stage CS amp, so the same five α candidates apply.
    "cs_amp_3stage": [
        dict(name="near_hard",   alpha_gain=5.0,  alpha_ugbw_log=10.0, alpha_power_mW=10.0),
        dict(name="sharp",       alpha_gain=1.0,  alpha_ugbw_log=3.0,  alpha_power_mW=2.0),
        dict(name="default",     alpha_gain=0.5,  alpha_ugbw_log=1.0,  alpha_power_mW=0.5),
        dict(name="soft",        alpha_gain=0.2,  alpha_ugbw_log=0.5,  alpha_power_mW=0.2),
        dict(name="very_soft",   alpha_gain=0.1,  alpha_ugbw_log=0.2,  alpha_power_mW=0.1),
    ],
    # Same loss form as cs_amp / cs_amp_3stage; same α candidates.
    "cs_amp_5stage": [
        dict(name="near_hard",   alpha_gain=5.0,  alpha_ugbw_log=10.0, alpha_power_mW=10.0),
        dict(name="sharp",       alpha_gain=1.0,  alpha_ugbw_log=3.0,  alpha_power_mW=2.0),
        dict(name="default",     alpha_gain=0.5,  alpha_ugbw_log=1.0,  alpha_power_mW=0.5),
        dict(name="soft",        alpha_gain=0.2,  alpha_ugbw_log=0.5,  alpha_power_mW=0.2),
        dict(name="very_soft",   alpha_gain=0.1,  alpha_ugbw_log=0.2,  alpha_power_mW=0.1),
    ],
    # cs_se_miller spec list is gain + UGBW + power (no PM); same α candidates
    # as cs_amp_3stage / cs_amp_5stage.
    "cs_se_miller": [
        dict(name="near_hard",   alpha_gain=5.0,  alpha_ugbw_log=10.0, alpha_power_mW=10.0),
        dict(name="sharp",       alpha_gain=1.0,  alpha_ugbw_log=3.0,  alpha_power_mW=2.0),
        dict(name="default",     alpha_gain=0.5,  alpha_ugbw_log=1.0,  alpha_power_mW=0.5),
        dict(name="soft",        alpha_gain=0.2,  alpha_ugbw_log=0.5,  alpha_power_mW=0.2),
        dict(name="very_soft",   alpha_gain=0.1,  alpha_ugbw_log=0.2,  alpha_power_mW=0.1),
    ],
    # cs_se_miller_3stage shares the gain+UGBW+power spec form.
    "cs_se_miller_3stage": [
        dict(name="near_hard",   alpha_gain=5.0,  alpha_ugbw_log=10.0, alpha_power_mW=10.0),
        dict(name="sharp",       alpha_gain=1.0,  alpha_ugbw_log=3.0,  alpha_power_mW=2.0),
        dict(name="default",     alpha_gain=0.5,  alpha_ugbw_log=1.0,  alpha_power_mW=0.5),
        dict(name="soft",        alpha_gain=0.2,  alpha_ugbw_log=0.5,  alpha_power_mW=0.2),
        dict(name="very_soft",   alpha_gain=0.1,  alpha_ugbw_log=0.2,  alpha_power_mW=0.1),
    ],
}


def build_design_set(mod, n_lognormal=30, n_uniform=30, n_directional=20, seed=42):
    """80-ish designs spanning the design domain."""
    rng = np.random.default_rng(seed)
    designs = [mod.X_NOMINAL.copy()]
    log_nom = np.log(mod.X_NOMINAL)
    log_lo = np.log(mod.X_LO); log_hi = np.log(mod.X_HI)
    sigma = 0.3
    for _ in range(n_lognormal):
        z = np.clip(rng.standard_normal(len(mod.X_NOMINAL)), -2.5, 2.5)
        x = np.exp(log_nom + sigma * z)
        designs.append(np.clip(x, mod.X_LO, mod.X_HI))
    for _ in range(n_uniform):
        u = rng.uniform(0, 1, size=len(mod.X_NOMINAL))
        x = np.exp(log_lo + u * (log_hi - log_lo))
        designs.append(np.clip(x, mod.X_LO, mod.X_HI))
    for _ in range(n_directional):
        k = int(rng.integers(0, len(mod.X_NOMINAL)))
        sign = float(rng.choice([-1.0, 1.0]))
        x = mod.X_NOMINAL.copy()
        x[k] = float(np.clip(x[k] * (2.0 ** (sign * 0.5)), mod.X_LO[k], mod.X_HI[k]))
        designs.append(x)
    return np.array(designs)


def compute_loss(metrics_dict, alpha_set, mod):
    """Compute combined loss given a metrics dict and an α candidate."""
    pens = mod.make_penalties(alphas={k: v for k, v in alpha_set.items() if k.startswith("alpha_")})
    from zo_yield.loss import combined_loss
    objective = 0.0
    if "gain_db" in metrics_dict and hasattr(mod, "_loss_objective_factor"):
        objective = -mod._loss_objective_factor * metrics_dict["gain_db"]
    return float(combined_loss(pens, dict(metrics_dict), objective=objective))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CIRCUIT_REGISTRY:
        print(f"usage: {sys.argv[0]} {{{'|'.join(CIRCUITS)}}}", file=sys.stderr)
        sys.exit(1)
    circuit = sys.argv[1]
    suffix = sys.argv[2] if len(sys.argv) >= 3 else ""
    mod = __import__(CIRCUIT_REGISTRY[circuit], fromlist=["spec"])

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments", "results", circuit
    )
    os.makedirs(out_dir, exist_ok=True)

    # Pick up the matching σ_scale for this suffix if its JSON exists; else
    # fall back to the calibrated σ-scale from configs/.
    sigma_json = os.path.join(out_dir, f"sigma_scale_calibration{suffix}.json")
    if os.path.isfile(sigma_json):
        with open(sigma_json) as f:
            sj = json.load(f)
        if sj.get("chosen_scale") is not None:
            scale = float(sj["chosen_scale"])
            print(f"[{circuit}{suffix}] σ_scale = {scale} (from {sigma_json})")
        else:
            scale = CHOSEN_SCALE[circuit]
            print(f"[{circuit}{suffix}] σ_scale = {scale} (sigma JSON had no chosen_scale; using default)")
    else:
        scale = CHOSEN_SCALE[circuit]
        print(f"[{circuit}{suffix}] σ_scale = {scale} (no sigma JSON; using default)")

    # Build calibration design set
    designs = build_design_set(mod)
    n_designs = len(designs)
    print(f"[{circuit}] calibration set: {n_designs} designs", flush=True)

    # Build fixed ξ-set
    sampler = mod.make_sampler(scale=scale)
    n_mc = 64
    yield_xis = sampler.sample(n_mc, rng=np.random.default_rng(555))
    class FixedSampler:
        dim = sampler.dim
        def sample(self, n_mc, rng=None): return yield_xis[:n_mc]

    sim = mod.build_simulator()
    specs = mod.build_specs() if hasattr(mod, "build_specs") else mod.SPECS

    # For each design: do one sim per xi, derive both yield and metrics for
    # all α-candidates from that single pass (n_designs × n_mc total sims).
    print(f"[{circuit}] evaluating {n_designs} designs × {n_mc} MC = "
          f"{n_designs * n_mc} sims (~{n_designs * n_mc * 0.25 / 60:.1f} min)", flush=True)
    yields = np.zeros(n_designs)
    metrics_per_design = []

    t0 = time.time()
    for i, x in enumerate(designs):
        m_list = []
        n_pass = 0
        for xi in yield_xis[:n_mc]:
            try:
                _, m = sim.evaluate_with_metrics(x, xi)
                m_list.append(m)
                # check all hard specs
                ok = all(s.satisfies(s.extract(m)) for s in specs)
                if ok:
                    n_pass += 1
            except Exception:
                m_list.append(None)
        yields[i] = n_pass / n_mc
        metrics_per_design.append(m_list)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_designs} done, elapsed={time.time()-t0:.1f}s, "
                  f"running yields: min={yields[:i+1].min():.3f} "
                  f"max={yields[:i+1].max():.3f}", flush=True)

    # For each α-set, compute loss per design (mean over xis), then ρ
    rows = []
    for alpha_set in ALPHA_CANDIDATES[circuit]:
        losses = np.zeros(n_designs)
        for i in range(n_designs):
            ls = []
            for m in metrics_per_design[i]:
                if m is None:
                    continue
                try:
                    ls.append(compute_loss(m, alpha_set, mod))
                except Exception:
                    pass
            losses[i] = float(np.mean(ls)) if ls else float("inf")
        rho, pval = spearmanr(losses, yields)
        # mean yield over 10% lowest loss
        k = max(1, n_designs // 10)
        idx_sorted = np.argsort(losses)
        mean_yield_top10 = float(np.mean(yields[idx_sorted[:k]]))
        rows.append(dict(alpha_set=alpha_set, rho=float(rho), pval=float(pval),
                         mean_yield_top10=mean_yield_top10,
                         losses=losses.tolist()))
        print(f"  {alpha_set['name']:12s}: ρ = {rho:+.3f} (p={pval:.2e}), "
              f"top10 yield = {mean_yield_top10:.3f}", flush=True)

    # Decision
    ranked = sorted(rows, key=lambda r: (r["rho"], -r["mean_yield_top10"]))   # most negative ρ first
    chosen = ranked[0]
    chosen_passes = chosen["rho"] < -0.7

    md = [f"# Loss-yield calibration — {circuit}\n",
          f"σ_scale = {scale}, n_designs = {n_designs}, n_mc = {n_mc}.\n",
          f"## Spearman ρ(loss, yield) per α candidate\n",
          "| α set       | ρ          | p-value    | mean yield top-10% |",
          "| ----------- | ---------- | ---------- | ------------------ |"]
    for r in sorted(rows, key=lambda r: r["rho"]):
        md.append(f"| {r['alpha_set']['name']:12s} | {r['rho']:+.3f}     | "
                   f"{r['pval']:.2e}  | {r['mean_yield_top10']:.3f}              |")
    md.append("")
    md.append(f"## Chosen α set: **{chosen['alpha_set']['name']}**")
    md.append(f"  - ρ = {chosen['rho']:+.3f}, top-10 yield = {chosen['mean_yield_top10']:.3f}")
    md.append(f"  - α values: {dict((k,v) for k,v in chosen['alpha_set'].items() if k != 'name')}")
    if chosen_passes:
        md.append(f"\n**Criterion (d) — Loss tracks yield: PASS** (ρ < −0.7)")
    elif chosen["rho"] < -0.4:
        md.append(f"\n**Criterion (d): PARTIAL** (best ρ = {chosen['rho']:+.3f}; "
                   f"target was ρ < −0.7)")
    else:
        md.append(f"\n**Criterion (d): FAIL** (best ρ = {chosen['rho']:+.3f}; "
                   f"target was ρ < −0.7). Loss surface does not track yield "
                   f"adequately at any α tested.")

    with open(os.path.join(out_dir, f"loss_yield_calibration{suffix}.md"), "w") as f:
        f.write("\n".join(md))
    print("\n" + "\n".join(md))

    chosen_alphas = {k: v for k, v in chosen["alpha_set"].items() if k.startswith("alpha_")}
    out = dict(circuit=circuit, scale=scale, n_designs=n_designs, n_mc=n_mc,
               chosen_alphas=chosen_alphas, chosen_name=chosen["alpha_set"]["name"],
               chosen_rho=chosen["rho"], chosen_top10_yield=chosen["mean_yield_top10"],
               criterion_d_pass=chosen_passes,
               all_rhos={r["alpha_set"]["name"]: r["rho"] for r in rows})
    with open(os.path.join(out_dir, f"chosen_alphas{suffix}.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Scatter at chosen α
    fig, ax = plt.subplots(figsize=(7, 5))
    losses = np.array(chosen["losses"])
    ax.scatter(losses, yields, alpha=0.6)
    ax.set_xlabel("E[loss] under chosen α")
    ax.set_ylabel("yield (n_mc=64)")
    ax.set_title(f"{circuit}: loss vs yield, ρ = {chosen['rho']:+.3f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"loss_yield_scatter{suffix}.png"), dpi=140)
    plt.close(fig)
    print(f"Wrote {out_dir}/{{loss_yield_calibration{suffix}.md, chosen_alphas{suffix}.json, loss_yield_scatter{suffix}.png}}")


if __name__ == "__main__":
    main()
