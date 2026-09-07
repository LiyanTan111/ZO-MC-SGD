"""Budget sweep: ZO-MC-SGD against the black-box baselines on one circuit.

For each budget in BUDGETS and each of N_SEEDS seeds, runs BO, CMA-ES, PSO and
ZO-MC-SGD (with the shared random-search warm start) from the same calibrated
starting design, then scores the returned design at the common evaluation
protocol (n_mc=80 on a fixed ξ-set, seed 555) and records the SPICE actually
consumed.

Output:
  experiments/results/<circuit>/baseline_comparison/budget_sweep.json
  experiments/results/<circuit>/baseline_comparison/budget_sweep_summary.csv

Usage:
    python scripts/baseline_comparison.py cs_amp_3stage
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

from zo_yield.circuit_config import CIRCUITS, load_circuit_config, load_spec_module

BUDGETS = [25, 50, 100, 200, 400]
N_SEEDS = 5
N_MC_YIELD = 80          # final-yield evaluation n_mc (paper Table IV)
N_MC_PER_EVAL = 8        # per-query n_mc for the surrogate objective
EVAL_XIS_PER_ITER = 16   # ξ-set for the in-loop loss estimate (diagnostic only)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CIRCUITS:
        print(f"usage: {sys.argv[0]} {{{'|'.join(CIRCUITS)}}}", file=sys.stderr)
        sys.exit(1)
    circuit = sys.argv[1]
    cfg = load_circuit_config(circuit)
    mod = load_spec_module(circuit)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rs_dir = os.path.join(repo, "experiments", "results", circuit)
    out_dir = os.path.join(rs_dir, "baseline_comparison")
    os.makedirs(out_dir, exist_ok=True)

    # ---- calibrated σ-scale / α / hard X_init (configs/<circuit>.json) ----
    chosen_scale = float(cfg["sigma_scale"])
    chosen_alphas = cfg["alphas"]
    x_init_real = cfg["x_init"]

    # ---- shared simulator infrastructure ----
    sampler = mod.make_sampler(scale=chosen_scale)
    specs = mod.build_specs() if hasattr(mod, "build_specs") else mod.SPECS
    eval_sim = mod.build_simulator(alphas=chosen_alphas)
    yield_xis = sampler.sample(N_MC_YIELD, rng=np.random.default_rng(555))
    eval_xis_inloop = sampler.sample(EVAL_XIS_PER_ITER, rng=np.random.default_rng(99))

    class FixedSampler:
        dim = sampler.dim
        def sample(self, n_mc, rng=None):
            return yield_xis[:n_mc]

    from zo_yield.evaluation import evaluate_design
    from zo_yield.estimators import EstimatorConfig
    from zo_yield.optimizers import OptimizerConfig, ZOMCSGD
    from zo_yield.baselines import (bo_optimize, cmaes_optimize, pso_optimize,
                                      warm_start_select)

    # initial yield at hard X_init (n_mc=80, fixed ξ-set)
    init_eval = evaluate_design(eval_sim, x_init_real, FixedSampler(), specs,
                                  n_mc=N_MC_YIELD)
    yield_init = float(init_eval["yield_"])
    print(f"[{circuit}] yield(X_init) at n_mc=80 = {yield_init:.4f}", flush=True)

    # ---- per-coord scaling: x_norm = x_real / x_init ----
    scale_x = x_init_real.copy()
    n = len(scale_x)
    x_lo_norm = mod.X_LO / scale_x
    x_hi_norm = mod.X_HI / scale_x
    x_init_norm = np.ones(n)

    # ---- baseline eval_fn factory ----
    # Each call creates a fresh subprocess simulator.
    def make_eval_fn(seed_for_sampler):
        """Build an eval_fn that evaluates yield/loss at a design x_norm.

        Uses one persistent simulator instance per outer call (one method
        run). Each yield estimate draws fresh ξ samples from `sampler`.
        """
        sim_local = mod.build_simulator(alphas=chosen_alphas)
        rng_xis = np.random.default_rng(seed_for_sampler + 13_579)

        def _eval(x_norm, _rng=None, n_mc=N_MC_PER_EVAL):
            x_real = np.clip(np.asarray(x_norm) * scale_x, mod.X_LO, mod.X_HI)
            xis = sampler.sample(n_mc, rng=rng_xis)

            class _OneShot:
                dim = sampler.dim
                def sample(self, n, rng=None): return xis[:n]
            res = evaluate_design(sim_local, x_real, _OneShot(), specs, n_mc=n_mc)
            return dict(E_loss=float(res["E_loss"]), yield_=float(res["yield_"]))

        return _eval

    # ---- simulator wrapper: ZO-MC-SGD works in normalized coordinates ----
    class Scaled:
        def __init__(self, b, s):
            self.b = b; self.s = s; self.n_failures = 0
        def evaluate(self, x_norm, xi):
            x = np.clip(x_norm * self.s, mod.X_LO, mod.X_HI)
            v = self.b.evaluate(x, xi); self.n_failures = self.b.n_failures
            return v

    rows = []
    t_global = time.time()

    for budget in BUDGETS:
        print(f"\n--- budget = {budget} sims ---", flush=True)
        for seed in range(N_SEEDS):
            # Each method gets a deterministic seed derived from
            # (budget, seed) so repeated runs are reproducible.
            method_seed = int(seed) * 100 + budget

            # ============================================================
            # BO
            # ============================================================
            t0 = time.time()
            try:
                eval_fn = make_eval_fn(method_seed)
                res = bo_optimize(eval_fn, x_init_norm, x_lo_norm, x_hi_norm,
                                    budget=budget, n_mc=N_MC_PER_EVAL,
                                    seed=method_seed)
                x_best_real = np.clip(res["x_best"] * scale_x, mod.X_LO, mod.X_HI)
                ye = evaluate_design(eval_sim, x_best_real, FixedSampler(),
                                      specs, n_mc=N_MC_YIELD)
                yfin = float(ye["yield_"])
                spice = int(res["spice_actual"])
            except Exception as e:
                print(f"  BO seed={seed} FAILED: {e}", flush=True)
                yfin, spice = float("nan"), 0
            wall = time.time() - t0
            rows.append(dict(circuit=circuit, budget=budget, method="bo",
                              seed=seed, yield_init=yield_init,
                              yield_final=yfin,
                              improvement=yfin - yield_init,
                              spice_actual=spice, wall_s=wall))
            print(f"  budget={budget} bo    seed={seed}: y={yfin:.4f} "
                  f"(Δ={yfin-yield_init:+.4f}) spice={spice} wall={wall:.1f}s",
                  flush=True)

            # ============================================================
            # CMA-ES
            # ============================================================
            t0 = time.time()
            try:
                eval_fn = make_eval_fn(method_seed + 1)
                res = cmaes_optimize(eval_fn, x_init_norm, x_lo_norm, x_hi_norm,
                                       budget=budget, n_mc=N_MC_PER_EVAL,
                                       seed=method_seed)
                x_best_real = np.clip(res["x_best"] * scale_x, mod.X_LO, mod.X_HI)
                ye = evaluate_design(eval_sim, x_best_real, FixedSampler(),
                                      specs, n_mc=N_MC_YIELD)
                yfin = float(ye["yield_"])
                spice = int(res["spice_actual"])
            except Exception as e:
                print(f"  CMA-ES seed={seed} FAILED: {e}", flush=True)
                yfin, spice = float("nan"), 0
            wall = time.time() - t0
            rows.append(dict(circuit=circuit, budget=budget, method="cmaes",
                              seed=seed, yield_init=yield_init,
                              yield_final=yfin,
                              improvement=yfin - yield_init,
                              spice_actual=spice, wall_s=wall))
            print(f"  budget={budget} cmaes seed={seed}: y={yfin:.4f} "
                  f"(Δ={yfin-yield_init:+.4f}) spice={spice} wall={wall:.1f}s",
                  flush=True)

            # ============================================================
            # PSO
            # ============================================================
            t0 = time.time()
            try:
                eval_fn = make_eval_fn(method_seed + 2)
                res = pso_optimize(eval_fn, x_init_norm, x_lo_norm, x_hi_norm,
                                     budget=budget, n_mc=N_MC_PER_EVAL,
                                     seed=method_seed)
                x_best_real = np.clip(res["x_best"] * scale_x, mod.X_LO, mod.X_HI)
                ye = evaluate_design(eval_sim, x_best_real, FixedSampler(),
                                      specs, n_mc=N_MC_YIELD)
                yfin = float(ye["yield_"])
                spice = int(res["spice_actual"])
            except Exception as e:
                print(f"  PSO seed={seed} FAILED: {e}", flush=True)
                yfin, spice = float("nan"), 0
            wall = time.time() - t0
            rows.append(dict(circuit=circuit, budget=budget, method="pso",
                              seed=seed, yield_init=yield_init,
                              yield_final=yfin,
                              improvement=yfin - yield_init,
                              spice_actual=spice, wall_s=wall))
            print(f"  budget={budget} pso   seed={seed}: y={yfin:.4f} "
                  f"(Δ={yfin-yield_init:+.4f}) spice={spice} wall={wall:.1f}s",
                  flush=True)

            # ============================================================
            # ZO-MC-SGD + warm start
            # ============================================================
            t0 = time.time()
            try:
                n_warm = max(1, min(10, budget // 16))
                eval_fn = make_eval_fn(method_seed + 3)
                ws = warm_start_select(eval_fn, x_init_norm, x_lo_norm, x_hi_norm,
                                         n_warm_pts=n_warm,
                                         seed=method_seed + 3)
                spice_warm = int(ws["spice_actual"])
                x_warm = ws["x_best"]
                # ZO-MC-SGD inner loop, batch_size_xi=4, batch_size_v=1, central
                step_cost = 4 * 1 * 2  # = 8 SPICE/step
                zo_budget = budget - spice_warm
                n_iters = max(0, zo_budget // step_cost)
                if n_iters > 0:
                    sim1 = mod.build_simulator(alphas=chosen_alphas)
                    s1 = Scaled(sim1, scale_x)
                    h1 = ZOMCSGD(
                        simulator=s1, sampler=sampler,
                        estimator_cfg=EstimatorConfig(
                            epsilon=5e-3, v_dist="gaussian", mode="central"),
                        optimizer_cfg=OptimizerConfig(
                            rule="adam", lr=0.05,
                            x_lo=x_lo_norm, x_hi=x_hi_norm),
                        batch_size_xi=4, batch_size_v=1,
                        eval_xis=eval_xis_inloop, log_every=max(1, n_iters // 4),
                    ).run(x_warm.copy(), n_iters=int(n_iters),
                           rng=np.random.default_rng(method_seed))
                    spice_zo = step_cost * int(n_iters)
                    # ZO-MC-SGD returns the last iterate x_t
                    x_final_norm = h1.x_history[-1] if h1.x_history else x_warm
                else:
                    spice_zo = 0
                    x_final_norm = x_warm
                spice = spice_warm + spice_zo
                x_best_real = np.clip(x_final_norm * scale_x,
                                        mod.X_LO, mod.X_HI)
                ye = evaluate_design(eval_sim, x_best_real, FixedSampler(),
                                      specs, n_mc=N_MC_YIELD)
                yfin = float(ye["yield_"])
            except Exception as e:
                print(f"  ZO-MC-SGD seed={seed} FAILED: {e}", flush=True)
                yfin, spice = float("nan"), 0
            wall = time.time() - t0
            rows.append(dict(circuit=circuit, budget=budget, method="zo_mc_sgd",
                              seed=seed, yield_init=yield_init,
                              yield_final=yfin,
                              improvement=yfin - yield_init,
                              spice_actual=spice, wall_s=wall))
            print(f"  budget={budget} zo-mc-sgd seed={seed}: y={yfin:.4f} "
                  f"(Δ={yfin-yield_init:+.4f}) spice={spice} wall={wall:.1f}s",
                  flush=True)

    # ---- splice in the random-search reference line, if present ----
    rs_path = os.path.join(rs_dir, "budget_sweep.json")
    if os.path.isfile(rs_path):
        with open(rs_path) as f:
            rs_blob = json.load(f)
        for r in rs_blob["rows"]:
            if r["method"] == "rs":
                rows.append(dict(circuit=circuit,
                                  budget=int(r["budget"]),
                                  method="rs",
                                  seed=int(r["seed"]),
                                  yield_init=float(r["yield_init"]),
                                  yield_final=float(r["yield_final"]),
                                  improvement=float(r["improvement"]),
                                  spice_actual=int(r["budget"]),
                                  wall_s=float(r.get("wall_s", 0.0))))
        print(f"[{circuit}] reused {sum(1 for r in rs_blob['rows'] if r['method']=='rs')} "
              "random-search rows", flush=True)

    # ---- persist ----
    with open(os.path.join(out_dir, "budget_sweep.json"), "w") as f:
        json.dump(dict(circuit=circuit, scale=chosen_scale, alphas=chosen_alphas,
                        x_init=x_init_real.tolist(), yield_init=yield_init,
                        budgets=BUDGETS, n_seeds=N_SEEDS,
                        n_mc_per_eval=N_MC_PER_EVAL,
                        n_mc_yield=N_MC_YIELD,
                        rows=rows), f, indent=2)
    with open(os.path.join(out_dir, "budget_sweep_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["circuit", "method", "budget", "seed",
                     "yield_init", "yield_final", "improvement",
                     "spice_actual", "wall_s"])
        for r in rows:
            w.writerow([r["circuit"], r["method"], r["budget"], r["seed"],
                         f"{r['yield_init']:.4f}", f"{r['yield_final']:.4f}",
                         f"{r['improvement']:+.4f}",
                         r["spice_actual"], f"{r['wall_s']:.1f}"])
    print(f"\nWall total: {time.time() - t_global:.1f}s")
    print(f"Wrote {out_dir}/budget_sweep.{{json,_summary.csv}}")


if __name__ == "__main__":
    main()
