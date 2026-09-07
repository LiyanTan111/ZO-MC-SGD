"""High-budget comparison — B in {800, 1600, 3200}.

Runs BO, CMA-ES, PSO and ZO-MC-SGD at budgets above the main sweep's ceiling of
400, on the same five circuits, five seeds, and the identical yield protocol
(n_mc=80, fixed xi-set seed 555). TuRBO and RobustAnalog at the same budgets are
produced by run_turbo_baseline.py and run_robustanalog_baseline.py.

Faithful to scripts/baseline_comparison.py (each method's config copied
verbatim) so high-budget cells are protocol-consistent with the existing
B≤400 data and can be appended to the same table/figure.

Output: experiments/results/{circuit}/highbudget_comparison.{json,_summary.csv}
        + per-cell experiments/results/{circuit}/highbudget_cells/{method}_B{b}_s{s}.json
Does NOT touch baseline_comparison/budget_sweep.json (the locked B≤400 data).

Usage: python scripts/run_highbudget_comparison.py [--circuits ...]
       [--budgets 800,1600,3200] [--seeds 5] [--methods bo,cmaes,...]
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

HIGH_BUDGETS = [800, 1600, 3200]
N_SEEDS = 5
N_MC_YIELD = 80
N_MC_PER_EVAL = 8
EVAL_XIS_PER_ITER = 16
YIELD_XI_SEED = 555
ALL_METHODS = ["bo", "cmaes", "pso", "zo_mc_sgd"]


def _cell_path(circuit, method, budget, seed):
    d = os.path.join(REPO, "experiments", "results", circuit, "highbudget_cells")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{method}_B{int(budget)}_s{int(seed)}.json")


def build_circuit_ctx(circuit):
    """Per-circuit shared config (mirrors baseline_comparison.py setup)."""
    cfg = load_circuit_config(circuit)
    return (load_spec_module(circuit), float(cfg["sigma_scale"]),
            cfg["alphas"], cfg["x_init"])


def run_method_cell(args_tuple):
    """Worker: one (circuit, method, budget, seed) cell. Writes per-cell JSON."""
    circuit, method, budget, seed = args_tuple
    cell_file = _cell_path(circuit, method, budget, seed)
    t0 = time.time()

    from zo_yield.evaluation import evaluate_design
    from zo_yield.estimators import EstimatorConfig
    from zo_yield.optimizers import OptimizerConfig, ZOMCSGD
    from zo_yield.baselines import (bo_optimize, cmaes_optimize, pso_optimize,
                                    warm_start_select)

    mod, chosen_scale, chosen_alphas, x_init_real = build_circuit_ctx(circuit)
    sampler = mod.make_sampler(scale=chosen_scale)
    specs = mod.build_specs() if hasattr(mod, "build_specs") else mod.SPECS
    eval_sim = mod.build_simulator(alphas=chosen_alphas)
    yield_xis = sampler.sample(N_MC_YIELD, rng=np.random.default_rng(YIELD_XI_SEED))
    eval_xis_inloop = sampler.sample(EVAL_XIS_PER_ITER, rng=np.random.default_rng(99))

    class FixedSampler:
        dim = sampler.dim
        def sample(self, n_mc, rng=None):
            return yield_xis[:n_mc]

    scale_x = x_init_real.copy()
    n = len(scale_x)
    x_lo_norm = mod.X_LO / scale_x
    x_hi_norm = mod.X_HI / scale_x
    x_init_norm = np.ones(n)

    init_eval = evaluate_design(eval_sim, x_init_real, FixedSampler(), specs,
                                n_mc=N_MC_YIELD)
    yield_init = float(init_eval["yield_"])

    def make_eval_fn(seed_for_sampler):
        sim_local = mod.build_simulator(alphas=chosen_alphas)
        rng_xis = np.random.default_rng(seed_for_sampler + 13_579)
        def _eval(x_norm, _rng=None, n_mc=N_MC_PER_EVAL):
            x_real = np.clip(np.asarray(x_norm) * scale_x, mod.X_LO, mod.X_HI)
            xis = sampler.sample(n_mc, rng=rng_xis)
            class _OneShot:
                dim = sampler.dim
                def sample(self, nn, rng=None): return xis[:nn]
            res = evaluate_design(sim_local, x_real, _OneShot(), specs, n_mc=n_mc)
            return dict(E_loss=float(res["E_loss"]), yield_=float(res["yield_"]))
        return _eval

    class Scaled:
        def __init__(self, b, s):
            self.b = b; self.s = s; self.n_failures = 0
        def evaluate(self, x_norm, xi):
            x = np.clip(x_norm * self.s, mod.X_LO, mod.X_HI)
            v = self.b.evaluate(x, xi); self.n_failures = self.b.n_failures
            return v

    method_seed = int(seed) * 100 + int(budget)
    yfin, spice = float("nan"), 0
    err = None
    try:
        if method == "bo":
            res = bo_optimize(make_eval_fn(method_seed), x_init_norm,
                              x_lo_norm, x_hi_norm, budget=budget,
                              n_mc=N_MC_PER_EVAL, seed=method_seed)
            x_best_real = np.clip(res["x_best"] * scale_x, mod.X_LO, mod.X_HI)
            yfin = float(evaluate_design(eval_sim, x_best_real, FixedSampler(),
                                         specs, n_mc=N_MC_YIELD)["yield_"])
            spice = int(res["spice_actual"])
        elif method == "cmaes":
            res = cmaes_optimize(make_eval_fn(method_seed + 1), x_init_norm,
                                 x_lo_norm, x_hi_norm, budget=budget,
                                 n_mc=N_MC_PER_EVAL, seed=method_seed)
            x_best_real = np.clip(res["x_best"] * scale_x, mod.X_LO, mod.X_HI)
            yfin = float(evaluate_design(eval_sim, x_best_real, FixedSampler(),
                                         specs, n_mc=N_MC_YIELD)["yield_"])
            spice = int(res["spice_actual"])
        elif method == "pso":
            res = pso_optimize(make_eval_fn(method_seed + 2), x_init_norm,
                               x_lo_norm, x_hi_norm, budget=budget,
                               n_mc=N_MC_PER_EVAL, seed=method_seed)
            x_best_real = np.clip(res["x_best"] * scale_x, mod.X_LO, mod.X_HI)
            yfin = float(evaluate_design(eval_sim, x_best_real, FixedSampler(),
                                         specs, n_mc=N_MC_YIELD)["yield_"])
            spice = int(res["spice_actual"])
        elif method == "zo_mc_sgd":
            n_warm = max(1, min(10, budget // 16))
            ws = warm_start_select(make_eval_fn(method_seed + 3), x_init_norm,
                                   x_lo_norm, x_hi_norm, n_warm_pts=n_warm,
                                   seed=method_seed + 3)
            spice_warm = int(ws["spice_actual"]); x_warm = ws["x_best"]
            step_cost = 4 * 1 * 2
            n_iters = max(0, (budget - spice_warm) // step_cost)
            if n_iters > 0:
                s1 = Scaled(mod.build_simulator(alphas=chosen_alphas), scale_x)
                h1 = ZOMCSGD(
                    simulator=s1, sampler=sampler,
                    estimator_cfg=EstimatorConfig(epsilon=5e-3,
                                                  v_dist="gaussian", mode="central"),
                    optimizer_cfg=OptimizerConfig(rule="adam", lr=0.05,
                                                  x_lo=x_lo_norm, x_hi=x_hi_norm),
                    batch_size_xi=4, batch_size_v=1, eval_xis=eval_xis_inloop,
                    log_every=max(1, n_iters // 4),
                ).run(x_warm.copy(), n_iters=int(n_iters),
                      rng=np.random.default_rng(method_seed))
                spice = spice_warm + step_cost * int(n_iters)
                x_final_norm = h1.x_history[-1] if h1.x_history else x_warm
            else:
                spice = spice_warm; x_final_norm = x_warm
            x_best_real = np.clip(x_final_norm * scale_x, mod.X_LO, mod.X_HI)
            yfin = float(evaluate_design(eval_sim, x_best_real, FixedSampler(),
                                         specs, n_mc=N_MC_YIELD)["yield_"])
        else:
            raise ValueError(f"unknown method {method}")
    except Exception as e:
        import traceback; traceback.print_exc()
        err = str(e)

    row = dict(circuit=circuit, method=method, budget=int(budget), seed=int(seed),
               yield_init=yield_init, yield_final=yfin,
               improvement=(yfin - yield_init) if np.isfinite(yfin) else float("nan"),
               spice_actual=int(spice), wall_s=time.time() - t0)
    if err:
        row["error"] = err
    with open(cell_file, "w") as f:
        json.dump(row, f, indent=2)
    print(f"  [{circuit}] {method} B={budget} s={seed}: y={yfin:.4f} "
          f"(Δ={row['improvement']:+.4f}) spice={spice} wall={row['wall_s']:.1f}s",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default=",".join(CIRCUITS))
    ap.add_argument("--budgets", default=",".join(map(str, HIGH_BUDGETS)))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    circuits = [c.strip() for c in args.circuits.split(",") if c.strip()]
    budgets = [int(b) for b in args.budgets.split(",")]
    seeds = list(range(args.seeds))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    t_global = time.time()
    pending = [(c, m, b, s) for c in circuits for m in methods
               for b in budgets for s in seeds
               if not os.path.isfile(_cell_path(c, m, b, s))]
    n_pre = (len(circuits) * len(methods) * len(budgets) * len(seeds)
             - len(pending))
    print(f"High-budget sweep: {len(circuits)}circ × {len(methods)}meth × "
          f"{len(budgets)}budg × {len(seeds)}seed. {len(pending)} pending "
          f"({n_pre} on disk). workers={args.workers}", flush=True)

    if pending:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 max_tasks_per_child=1) as ex:
            futs = {ex.submit(run_method_cell, c): c for c in pending}
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"  POOL-ERROR {c}: {e}", flush=True)

    # Collect per-cell files into per-circuit outputs.
    for circuit in circuits:
        rows = []
        for m in methods:
            for b in budgets:
                for s in seeds:
                    p = _cell_path(circuit, m, b, s)
                    if os.path.isfile(p):
                        with open(p) as f:
                            rows.append(json.load(f))
        if not rows:
            continue
        out_dir = os.path.join(REPO, "experiments", "results", circuit)
        rows.sort(key=lambda r: (r["method"], r["budget"], r["seed"]))
        with open(os.path.join(out_dir, "highbudget_comparison.json"), "w") as f:
            json.dump(dict(circuit=circuit, budgets=budgets, n_seeds=len(seeds),
                           methods=methods, n_mc_yield=N_MC_YIELD,
                           yield_xi_seed=YIELD_XI_SEED, rows=rows), f, indent=2)
        with open(os.path.join(out_dir, "highbudget_comparison_summary.csv"),
                  "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["circuit", "method", "budget", "seed", "yield_init",
                        "yield_final", "improvement", "spice_actual", "wall_s"])
            for r in rows:
                w.writerow([r["circuit"], r["method"], r["budget"], r["seed"],
                            f"{r['yield_init']:.4f}", f"{r['yield_final']:.4f}",
                            f"{r['improvement']:+.4f}", r["spice_actual"],
                            f"{r['wall_s']:.1f}"])
        print(f"[{circuit}] wrote highbudget_comparison.{{json,_summary.csv}} "
              f"({len(rows)} rows)", flush=True)

    print(f"\nTOTAL wall: {time.time() - t_global:.1f}s", flush=True)


if __name__ == "__main__":
    main()
