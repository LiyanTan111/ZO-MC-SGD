"""TuRBO-1 baseline sweep — 7th Tier-1 method.

Runs TuRBO-1 on the 5-circuit roster at the standard grid B∈{25,50,100,200,400}
(125 runs) and the high-budget extension B∈{800,1600,3200} (75 runs), 5 seeds
each. Same yield protocol as every baseline (n_mc=80, fixed ξ-set seed 555).

Per-query SPICE = n_mc_per_eval (=8), matching the existing BO baseline; budget
B in SPICE maps to floor(B/8) TuRBO queries. Per-cell results are written to
disk for crash-resilient resume (mirrors run_robustanalog_baseline.py).

Output (per circuit):
  experiments/results/{circuit}/round_turbo_baseline.{json,_summary.csv}
  experiments/results/{circuit}/turbo_cells/B{b}_s{s}.json

Usage:
  python scripts/run_turbo_baseline.py [--circuits ...] [--budgets ...]
      [--seeds 5] [--workers N]
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
STD_BUDGETS = [25, 50, 100, 200, 400]
HIGH_BUDGETS = [800, 1600, 3200]
N_SEEDS = 5
N_MC_PER_EVAL = 10       # yield-MC samples/query


def _cell_path(circuit, budget, seed):
    d = os.path.join(REPO, "experiments", "results", circuit, "turbo_cells")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"B{int(budget)}_s{int(seed)}.json")


def run_cell(args_tuple):
    circuit, budget, seed = args_tuple
    cell_file = _cell_path(circuit, budget, seed)
    t0 = time.time()
    from methods.turbo.benchmark_adapter import make_env
    from methods.turbo.config import TuRBOConfig
    from methods.turbo.train import train_turbo

    cfg = TuRBOConfig()
    try:
        env = make_env(circuit, seed=seed)
        budget_queries = int(budget) // N_MC_PER_EVAL
        res = train_turbo(env.objective, d=env.n_design,
                          budget_queries=budget_queries, cfg=cfg, seed=seed,
                          x0_unit=env.x_init_unit)
        yfin = env.evaluate_final_yield(res["x_best"])
        row = dict(
            circuit=circuit, method="turbo", budget=int(budget), seed=int(seed),
            yield_init=float(env.yield_init), yield_final=float(yfin),
            delta=float(yfin - env.yield_init),
            improvement=float(yfin - env.yield_init),
            n_sims_used=int(env.spice_used), n_queries=int(res["n_queries"]),
            best_loss=float(res["y_best"]), n_restarts=int(res["n_restarts"]),
            completed_init=bool(res["completed_init"]),
            n_init_design=int(cfg.n_init(env.n_design)),
            turbo_seed=int(seed), time_wall=float(time.time() - t0),
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        row = dict(circuit=circuit, method="turbo", budget=int(budget),
                   seed=int(seed), yield_init=float("nan"),
                   yield_final=float("nan"), delta=float("nan"),
                   improvement=float("nan"), n_sims_used=0, n_queries=0,
                   best_loss=float("nan"), n_restarts=0, completed_init=False,
                   n_init_design=0, turbo_seed=int(seed),
                   time_wall=float(time.time() - t0), error=str(e))
    with open(cell_file, "w") as f:
        json.dump(row, f, indent=2)
    print(f"  [{circuit}] B={budget} s={seed}: y={row['yield_final']:.4f} "
          f"(Δ={row['delta']:+.4f}) sims={row['n_sims_used']} "
          f"q={row['n_queries']} init_ok={row['completed_init']} "
          f"wall={row['time_wall']:.1f}s", flush=True)
    return row


def write_outputs(circuit, budgets, seeds):
    rows = []
    for b in budgets:
        for s in seeds:
            p = _cell_path(circuit, b, s)
            if os.path.isfile(p):
                with open(p) as f:
                    rows.append(json.load(f))
    if not rows:
        return
    rows.sort(key=lambda r: (r["budget"], r["seed"]))
    out_dir = os.path.join(REPO, "experiments", "results", circuit)
    yinit = next((r["yield_init"] for r in rows if np.isfinite(r["yield_init"])),
                 float("nan"))
    with open(os.path.join(out_dir, "round_turbo_baseline.json"), "w") as f:
        json.dump(dict(circuit=circuit, method="turbo", yield_init=yinit,
                       n_mc_yield=80, yield_xi_seed=555, rows=rows), f, indent=2)
    with open(os.path.join(out_dir, "round_turbo_baseline_summary.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["circuit", "method", "budget", "seed", "yield_init",
                    "yield_final", "delta", "n_sims_used", "n_queries",
                    "completed_init", "time_wall"])
        for r in rows:
            w.writerow([r["circuit"], "turbo", r["budget"], r["seed"],
                        f"{r['yield_init']:.4f}", f"{r['yield_final']:.4f}",
                        f"{r['delta']:+.4f}", r["n_sims_used"], r["n_queries"],
                        r["completed_init"], f"{r['time_wall']:.1f}"])
    print(f"[{circuit}] wrote round_turbo_baseline.{{json,_summary.csv}} "
          f"({len(rows)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default=",".join(ALL_CIRCUITS))
    ap.add_argument("--budgets", default=",".join(map(str, STD_BUDGETS + HIGH_BUDGETS)))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    circuits = [c.strip() for c in args.circuits.split(",") if c.strip()]
    budgets = [int(b) for b in args.budgets.split(",")]
    seeds = list(range(args.seeds))

    t_global = time.time()
    pending = [(c, b, s) for c in circuits for b in budgets for s in seeds
               if not os.path.isfile(_cell_path(c, b, s))]
    n_pre = len(circuits) * len(budgets) * len(seeds) - len(pending)
    print(f"TuRBO sweep: {len(circuits)}circ × {len(budgets)}budg × "
          f"{len(seeds)}seed. {len(pending)} pending ({n_pre} on disk). "
          f"workers={args.workers}", flush=True)

    if pending:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 max_tasks_per_child=1) as ex:
            futs = {ex.submit(run_cell, c): c for c in pending}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  POOL-ERROR {futs[fut]}: {e}", flush=True)

    for circuit in circuits:
        write_outputs(circuit, budgets, seeds)
    print(f"\nTOTAL wall: {time.time() - t_global:.1f}s", flush=True)


if __name__ == "__main__":
    main()
