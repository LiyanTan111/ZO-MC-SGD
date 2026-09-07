"""RobustAnalog baseline sweep — 5 circuits x 5 budgets x 5 seeds.

Runs the re-implemented RobustAnalog (methods/robustanalog/) as the 6th Tier-1
baseline on the paper's 5-circuit roster, at the same budget grid + seed count
as / so it slots directly into the existing
baseline_comparison data.

Yield protocol:
  The final yield is evaluated EXACTLY as scripts/baseline_comparison.py
  evaluates the 5 existing methods: n_mc=80 at the FIXED ξ-set drawn with
  np.random.default_rng(555). This is mandatory for an apples-to-apples
  6-method comparison (box 13). It satisfies the spec's anti-overfit intent
  (the eval ξ-set is disjoint from the K=20 training corners, which are drawn
  with xi_corner_seed=0), and matches "our standard yield estimation protocol".
  The spec's literal n_eval=200 is adapted to the on-disk standard (80, fixed)

Output (per circuit):
  experiments/results/{circuit}/round_ra_baseline.json
  experiments/results/{circuit}/round_ra_baseline_summary.csv

Usage:
  python scripts/run_robustanalog_baseline.py [--circuits c1,c2] \
      [--budgets 25,50,...] [--seeds 5] [--workers N]
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

ALL_CIRCUITS = ["cs_amp", "cs_amp_3stage", "cs_amp_5stage",
                "cs_se_miller", "cs_se_miller_3stage"]
BUDGETS = [25, 50, 100, 200, 400]
N_SEEDS = 5
N_MC_YIELD = 80          # matches baseline_comparison.py (fixed ξ-set, seed 555)
YIELD_XI_SEED = 555


def _fixed_yield_eval(mod, sim, x_real, sampler, specs, n_mc, seed):
    """Yield at x_real using the FIXED ξ-set (seed) — baseline_comparison parity."""
    from zo_yield.evaluation import evaluate_design
    yield_xis = sampler.sample(n_mc, rng=np.random.default_rng(seed))

    class FixedSampler:
        dim = sampler.dim
        def sample(self, n, rng=None):
            return yield_xis[:n]

    res = evaluate_design(sim, np.clip(x_real, mod.X_LO, mod.X_HI),
                          FixedSampler(), specs, n_mc=n_mc)
    return float(res["yield_"])


def _cell_path(circuit, budget, seed):
    d = os.path.join(REPO, "experiments", "results", circuit, "ra_cells")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"B{int(budget)}_s{int(seed)}.json")


def run_cell(args_tuple):
    """Worker: one (circuit, budget, seed) cell. Writes its row to a per-cell
    JSON on disk (crash-resilient + resumable) and returns it."""
    import torch
    torch.set_num_threads(1)

    circuit, budget, seed, yield_init = args_tuple
    cell_file = _cell_path(circuit, budget, seed)
    from methods.robustanalog.benchmark_adapter import (
        make_env, make_agent, load_circuit_config)
    from methods.robustanalog.config import RAConfig
    from methods.robustanalog.train import train_robustanalog

    cfg = RAConfig()
    t0 = time.time()
    try:
        cc = load_circuit_config(circuit)
        mod = cc["mod"]
        x_init = cc["x_init_real"]
        env = make_env(circuit, cfg)
        agent = make_agent(env, cfg, seed=seed)
        res = train_robustanalog(env, agent, cfg, budget=budget,
                                 x_init_real=x_init, seed=seed)
        # Final yield with the baseline-parity fixed ξ-set.
        sampler = mod.make_sampler(scale=cc["scale"])
        specs = mod.build_specs() if hasattr(mod, "build_specs") else mod.SPECS
        eval_sim = mod.build_simulator(alphas=cc["alphas"])
        yfin = _fixed_yield_eval(mod, eval_sim, res["x_best_real"], sampler,
                                 specs, N_MC_YIELD, YIELD_XI_SEED)
        row = dict(
            circuit=circuit, budget=int(budget), seed=int(seed),
            yield_init=float(yield_init), yield_final=float(yfin),
            delta=float(yfin - yield_init),
            improvement=float(yfin - yield_init),
            n_sims_used=int(res["spice_used"]),
            episodes=int(res["episodes"]),
            best_min_reward=float(res["best_min_reward"]),
            best_mean_reward=float(res["best_mean_reward"]),
            best_source=res["best_source"],
            n_pruned_final=int(res["n_pruned_final"]),
            xi_corner_seed=int(cfg.xi_corner_seed),
            ra_seed=int(seed),
            time_wall=float(time.time() - t0),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        row = dict(circuit=circuit, budget=int(budget), seed=int(seed),
                   yield_init=float(yield_init), yield_final=float("nan"),
                   delta=float("nan"), improvement=float("nan"),
                   n_sims_used=0, episodes=0, best_min_reward=float("nan"),
                   best_mean_reward=float("nan"), best_source="FAILED",
                   n_pruned_final=0, xi_corner_seed=0, ra_seed=int(seed),
                   time_wall=float(time.time() - t0), error=str(e))
    with open(cell_file, "w") as f:
        json.dump(row, f, indent=2)
    print(f"  [{circuit}] B={budget} seed={seed}: "
          f"y={row['yield_final']:.4f} (Δ={row['delta']:+.4f}) "
          f"sims={row['n_sims_used']} wall={row['time_wall']:.1f}s", flush=True)
    return row


def compute_yield_init(circuit):
    """yield_init for a circuit (matches baseline_comparison.py exactly)."""
    from methods.robustanalog.benchmark_adapter import load_circuit_config
    cc = load_circuit_config(circuit)
    mod = cc["mod"]
    sampler = mod.make_sampler(scale=cc["scale"])
    specs = mod.build_specs() if hasattr(mod, "build_specs") else mod.SPECS
    eval_sim = mod.build_simulator(alphas=cc["alphas"])
    return _fixed_yield_eval(mod, eval_sim, cc["x_init_real"], sampler, specs,
                             N_MC_YIELD, YIELD_XI_SEED)


def write_outputs(circuit, rows, meta):
    out_dir = os.path.join(REPO, "experiments", "results", circuit)
    os.makedirs(out_dir, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (r["budget"], r["seed"]))
    with open(os.path.join(out_dir, "round_ra_baseline.json"), "w") as f:
        json.dump(dict(circuit=circuit, method="robustanalog",
                       n_mc_yield=N_MC_YIELD, yield_xi_seed=YIELD_XI_SEED,
                       **meta, rows=rows_sorted), f, indent=2)
    with open(os.path.join(out_dir, "round_ra_baseline_summary.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["circuit", "method", "budget", "seed", "yield_init",
                    "yield_final", "delta", "n_sims_used", "episodes",
                    "best_min_reward", "time_wall"])
        for r in rows_sorted:
            w.writerow([r["circuit"], "robustanalog", r["budget"], r["seed"],
                        f"{r['yield_init']:.4f}", f"{r['yield_final']:.4f}",
                        f"{r['delta']:+.4f}", r["n_sims_used"], r["episodes"],
                        f"{r['best_min_reward']:.4f}", f"{r['time_wall']:.1f}"])
    print(f"[{circuit}] wrote round_ra_baseline.{{json,_summary.csv}} "
          f"({len(rows_sorted)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default=",".join(ALL_CIRCUITS))
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    circuits = [c.strip() for c in args.circuits.split(",") if c.strip()]
    budgets = [int(b) for b in args.budgets.split(",")]
    seeds = list(range(args.seeds))

    t_global = time.time()
    print(f"RobustAnalog sweep: circuits={circuits} budgets={budgets} "
          f"seeds={seeds} workers={args.workers}", flush=True)

    # Gather all pending cells across all circuits (skip ones already on disk).
    yinit_by_circuit = {}
    pending = []
    for circuit in circuits:
        print(f"\n=== {circuit} : computing yield_init ===", flush=True)
        yinit = compute_yield_init(circuit)
        yinit_by_circuit[circuit] = yinit
        print(f"[{circuit}] yield_init = {yinit:.4f}", flush=True)
        for b in budgets:
            for s in seeds:
                if os.path.isfile(_cell_path(circuit, b, s)):
                    continue   # resume: cell already computed
                pending.append((circuit, b, s, yinit))

    n_done_pre = sum(
        1 for circuit in circuits for b in budgets for s in seeds
        if os.path.isfile(_cell_path(circuit, b, s)))
    print(f"\n{len(pending)} cells to run ({n_done_pre} already on disk). "
          f"workers={args.workers}", flush=True)

    if pending:
        # max_tasks_per_child=1 fully isolates each cell (fresh torch state,
        # no leaks, and a crashed cell can't take the pool down with it).
        with ProcessPoolExecutor(max_workers=args.workers,
                                 max_tasks_per_child=1) as ex:
            futs = {ex.submit(run_cell, c): c for c in pending}
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"  [{c[0]}] B={c[1]} seed={c[2]} POOL-ERROR: {e}",
                          flush=True)

    # Collect per-cell files from disk into per-circuit outputs.
    for circuit in circuits:
        rows = []
        for b in budgets:
            for s in seeds:
                p = _cell_path(circuit, b, s)
                if os.path.isfile(p):
                    with open(p) as f:
                        rows.append(json.load(f))
        if rows:
            write_outputs(circuit, rows, meta=dict(
                yield_init=yinit_by_circuit[circuit], budgets=budgets,
                n_seeds=len(seeds)))

    print(f"\nTOTAL wall: {time.time() - t_global:.1f}s", flush=True)


if __name__ == "__main__":
    main()
