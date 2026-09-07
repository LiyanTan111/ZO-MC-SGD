"""Natural-objective (yield-MC) re-run of BO / CMA-ES / PSO.

Re-runs the EXISTING BO/CMA-ES/PSO optimizers (zo_yield.baselines, unmodified)
on the literature-natural yield-MC objective (Ŷ over n_mc=10 fresh ξ per query,
minimize −Ŷ) instead of the softplus surrogate, to validate that our softplus
framing is the steel-man. 3 methods × 5 circuits × 4 budgets × 5 seeds = 300
runs. Per-cell disk + resume; output is a sidecar (existing softplus data and
the tier1 exports are NOT touched).

Yield protocol parity: final yield at n_mc=80, fixed ξ-set seed 555 — identical
to every baseline; yield_init must reproduce each circuit exactly (HARD GATE).

Usage: python scripts/run_natural_objective_baseline.py [--circuits ...]
       [--budgets 50,100,200,400] [--seeds 5] [--methods bo,cmaes,pso]
       [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from zo_yield.circuit_config import CIRCUITS, load_circuit_config, load_spec_module

BUDGETS = [50, 100, 200, 400]
N_SEEDS = 5
N_MC_NATURAL = 10        # yield-MC samples per query
N_MC_YIELD = 80          # final yield (parity)
YIELD_XI_SEED = 555
ALL_METHODS = ["bo", "cmaes", "pso"]
NAME = {"bo": "BO", "cmaes": "CMAES", "pso": "PSO"}


def _cell_path(circuit, method, budget, seed):
    d = os.path.join(REPO, "experiments", "results", circuit, "natural_cells")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{method}_B{int(budget)}_s{int(seed)}.json")


def build_ctx(circuit):
    """Return (spec module, σ-scale, α, x_init) from configs/<circuit>.json."""
    cfg = load_circuit_config(circuit)
    return (load_spec_module(circuit), float(cfg["sigma_scale"]),
            cfg["alphas"], cfg["x_init"])


def _fixed_yield(mod, sim, x_real, sampler, specs):
    from zo_yield.evaluation import evaluate_design
    yx = sampler.sample(N_MC_YIELD, rng=np.random.default_rng(YIELD_XI_SEED))

    class FixedSampler:
        dim = sampler.dim
        def sample(self, n, rng=None): return yx[:n]
    return float(evaluate_design(sim, np.clip(x_real, mod.X_LO, mod.X_HI),
                                 FixedSampler(), specs, n_mc=N_MC_YIELD)["yield_"])


def run_cell(args_tuple):
    circuit, method, budget, seed = args_tuple
    cell_file = _cell_path(circuit, method, budget, seed)
    t0 = time.time()
    from zo_yield.baselines import bo_optimize, cmaes_optimize, pso_optimize
    from methods.natural_objective.objective import make_natural_eval_fn

    mod, scale, alphas, x_init_real = build_ctx(circuit)
    sampler = mod.make_sampler(scale=scale)
    specs = mod.build_specs() if hasattr(mod, "build_specs") else mod.SPECS
    eval_sim = mod.build_simulator(alphas=alphas)
    yield_init = _fixed_yield(mod, eval_sim, x_init_real, sampler, specs)

    scale_x = x_init_real.copy()
    x_lo_norm = mod.X_LO / scale_x; x_hi_norm = mod.X_HI / scale_x
    x_init_norm = np.ones(len(scale_x))
    method_seed = int(seed) * 100 + int(budget)

    opt_map = {"bo": bo_optimize, "cmaes": cmaes_optimize, "pso": pso_optimize}
    yfin, spice, err = float("nan"), 0, None
    stats = {}
    try:
        sim_opt = mod.build_simulator(alphas=alphas)
        eval_fn = make_natural_eval_fn(sim_opt, sampler, specs, mod.X_LO,
                                       mod.X_HI, scale_x, run_seed=method_seed,
                                       n_mc=N_MC_NATURAL, stats=stats)
        res = opt_map[method](eval_fn, x_init_norm, x_lo_norm, x_hi_norm,
                              budget=int(budget), n_mc=N_MC_NATURAL,
                              seed=method_seed)
        x_best_real = np.clip(res["x_best"] * scale_x, mod.X_LO, mod.X_HI)
        yfin = _fixed_yield(mod, eval_sim, x_best_real, sampler, specs)
        spice = int(res["spice_actual"])
    except Exception as e:
        import traceback; traceback.print_exc(); err = str(e)

    n_sims = stats.get("n_sims", 0)
    n_nan = stats.get("n_nan_total", 0)
    nan_frac = (n_nan / n_sims) if n_sims > 0 else 0.0
    failed = bool(err is not None or nan_frac > 0.30)
    row = dict(circuit=circuit, method=method, objective="natural_yield_mc",
               budget=int(budget), seed=int(seed), yield_init=float(yield_init),
               yield_final=float(yfin),
               delta=float(yfin - yield_init) if np.isfinite(yfin) else float("nan"),
               improvement=float(yfin - yield_init) if np.isfinite(yfin) else float("nan"),
               n_queries_completed=int(stats.get("n_queries", 0)),
               total_spice_used=int(spice), n_mc=N_MC_NATURAL,
               n_nan_total=int(n_nan), nan_frac=float(nan_frac),
               failed=failed, time_wall=float(time.time() - t0))
    if err:
        row["error"] = err
    with open(cell_file, "w") as f:
        json.dump(row, f, indent=2)
    print(f"  [{circuit}] {method} B={budget} s={seed}: y={row['yield_final']:.4f} "
          f"(Δ={row['delta']:+.4f}) q={row['n_queries_completed']} "
          f"spice={spice} nan={nan_frac:.0%}{' FAILED' if failed else ''} "
          f"wall={row['time_wall']:.1f}s", flush=True)
    return row


def write_outputs(circuit, methods, budgets, seeds, suffix=""):
    for m in methods:
        rows = []
        for b in budgets:
            for s in seeds:
                p = _cell_path(circuit, m, b, s)
                if os.path.isfile(p):
                    rows.append(json.load(open(p)))
        if not rows:
            continue
        rows.sort(key=lambda r: (r["budget"], r["seed"]))
        out_dir = os.path.join(REPO, "experiments", "results", circuit)
        yinit = next((r["yield_init"] for r in rows if np.isfinite(r["yield_init"])), float("nan"))
        base = f"round_natural_baseline_{NAME[m]}{suffix}"
        json.dump(dict(circuit=circuit, method=m, objective="natural_yield_mc",
                       yield_init=yinit, n_mc=N_MC_NATURAL, n_mc_yield=N_MC_YIELD,
                       yield_xi_seed=YIELD_XI_SEED, rows=rows),
                  open(os.path.join(out_dir, base + ".json"), "w"), indent=2)
        with open(os.path.join(out_dir, base + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["circuit", "method", "budget", "seed", "yield_init",
                        "yield_final", "delta", "n_queries", "total_spice",
                        "nan_frac", "failed", "wall_s"])
            for r in rows:
                w.writerow([r["circuit"], m, r["budget"], r["seed"],
                            f"{r['yield_init']:.4f}", f"{r['yield_final']:.4f}",
                            f"{r['delta']:+.4f}", r["n_queries_completed"],
                            r["total_spice_used"], f"{r['nan_frac']:.3f}",
                            r["failed"], f"{r['time_wall']:.1f}"])
        print(f"[{circuit}] wrote {base}.{{json,csv}} ({len(rows)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default=",".join(CIRCUITS))
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out-suffix", default="",
                    help="suffix for collected output files (e.g. _highbudget)")
    args = ap.parse_args()
    circuits = [c.strip() for c in args.circuits.split(",") if c.strip()]
    budgets = [int(b) for b in args.budgets.split(",")]
    seeds = list(range(args.seeds))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    t0 = time.time()
    pending = [(c, m, b, s) for c in circuits for m in methods
               for b in budgets for s in seeds
               if not os.path.isfile(_cell_path(c, m, b, s))]
    n_pre = len(circuits) * len(methods) * len(budgets) * len(seeds) - len(pending)
    print(f"Natural-objective sweep: {len(pending)} pending ({n_pre} on disk). "
          f"workers={args.workers}", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as ex:
            futs = {ex.submit(run_cell, c): c for c in pending}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  POOL-ERROR {futs[fut]}: {e}", flush=True)
    for circuit in circuits:
        write_outputs(circuit, methods, budgets, seeds, suffix=args.out_suffix)
    print(f"\nTOTAL wall: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
