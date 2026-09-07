"""Wall-clock comparison: optimizer-internal vs SPICE wall, per (method, budget, seed).

Separates the algorithm's own compute (GP fits, acquisition, DDPG, ZO gradient
math) from the ngspice simulation cost.

  - optimizer-internal wall: each REAL optimizer is driven by a near-instant
    objective (matched design-dim n, n_mc, budget→#queries). We time
    (total - objective_time); since the objective is ~µs, this is the pure
    algorithm overhead, independent of the SPICE *values* (so no SPICE needed,
    and it's exact). Baselines use mature libs: BO=scikit-optimize,
    CMA-ES=pycma; PSO=lightweight self-rolled; TuRBO/RA = our reimplementations
    (TuRBO GP is pure-numpy → its overhead is an upper bound vs botorch).
  - SPICE wall: (#simulator calls) × (measured median per-call ngspice wall for
    that circuit). #calls is deterministic per (method, budget).

Same machine, fixed seeds; medians reported over seeds.
Output: experiments/results/wallclock_comparison.{md,json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CIRCUITS = {  # circuit -> (spec module, n_design, d_xi)
    "cs_amp": ("benchmarks.common_source_amp.spec", 6, 10),
    "cs_amp_3stage": ("benchmarks.cs_amp_3stage.spec", 18, 26),
    "cs_amp_5stage": ("benchmarks.cs_amp_5stage.spec", 30, 42),
}
N_MC_ZO = 8       # ZO-MC-SGD: batch_xi=4 * 2 (central) per step
N_MC_BB = 10      # BO/CMA/PSO/TuRBO: yield-MC per query


# ----------------------------- fast objective ------------------------------ #
class _Timer:
    def __init__(self): self.t = 0.0; self.n = 0
    def tick(self, dt): self.t += dt; self.n += 1


def _fast_scalar(x, n):
    """Cheap smooth loss in design space (varied so optimizers do real work)."""
    x = np.asarray(x, dtype=float).reshape(-1)[:n]
    return float(np.sum((x - 0.3) ** 2) / n)


class FastSim:
    """Near-instant stand-in for the ngspice simulator (ZO/RA paths)."""
    def __init__(self, n, d_xi, timer): self.n=n; self.d_xi=d_xi; self.tmr=timer; self.n_calls=0; self.n_failures=0
    def evaluate(self, x, xi):
        t0=time.perf_counter(); self.n_calls+=1
        v=_fast_scalar(x,self.n)+1e-3*float(np.sum(np.asarray(xi)))
        self.tmr.tick(time.perf_counter()-t0); return v
    def evaluate_with_metrics(self, x, xi):
        t0=time.perf_counter(); self.n_calls+=1
        s=_fast_scalar(x,self.n)
        m=dict(gain_db=60.0-20*s, ugbw_hz=1e7*(1.0+s), power_w=4e-4+1e-4*s)
        self.tmr.tick(time.perf_counter()-t0); return float(s), m


def fast_eval_fn(n, timer):
    """eval_fn(x_norm)->dict(E_loss,yield_) for BO/CMA/PSO; timed."""
    def _e(x_norm, rng=None, n_mc=N_MC_BB):
        t0=time.perf_counter()
        loss=_fast_scalar(x_norm,n); y=float(np.exp(-loss))
        timer.tick(time.perf_counter()-t0)
        return dict(E_loss=loss, yield_=y)
    return _e


# ----------------------------- per-method timed runs ------------------------ #
def time_zo(n, d_xi, budget, seed):
    from zo_yield.estimators import EstimatorConfig
    from zo_yield.optimizers import OptimizerConfig, ZOMCSGD
    from zo_yield.baselines import warm_start_select
    from zo_yield.samplers import IndependentGaussianSampler
    tmr=_Timer(); sim=FastSim(n,d_xi,tmr)
    sampler=IndependentGaussianSampler(mean=np.zeros(d_xi), std=np.ones(d_xi))
    x_lo=np.full(n,0.2); x_hi=np.full(n,2.0); x0=np.ones(n)
    eval_xis=sampler.sample(16, rng=np.random.default_rng(99))
    ms=seed*100+budget
    ef=fast_eval_fn(n,tmr)
    t0=time.perf_counter()
    n_warm=max(1,min(10,budget//16))
    ws=warm_start_select(ef,x0,x_lo,x_hi,n_warm_pts=n_warm,seed=ms+3)
    n_iters=max(0,(budget-int(ws["spice_actual"]))//N_MC_ZO)
    if n_iters>0:
        class Scaled:
            def __init__(s,b): s.b=b; s.n_failures=0
            def evaluate(s,xn,xi): return s.b.evaluate(np.clip(xn,x_lo,x_hi),xi)
        ZOMCSGD(simulator=Scaled(sim),sampler=sampler,
            estimator_cfg=EstimatorConfig(epsilon=5e-3,v_dist="gaussian",mode="central"),
            optimizer_cfg=OptimizerConfig(rule="adam",lr=0.05,x_lo=x_lo,x_hi=x_hi),
            batch_size_xi=4,batch_size_v=1,eval_xis=eval_xis,log_every=max(1,n_iters//4),
        ).run(ws["x_best"].copy(),n_iters=int(n_iters),rng=np.random.default_rng(ms))
    total=time.perf_counter()-t0
    return total-tmr.t, sim.n_calls


def time_lib_baseline(which, n, budget, seed):
    from zo_yield.baselines import bo_optimize, cmaes_optimize, pso_optimize
    fn={"bo":bo_optimize,"cmaes":cmaes_optimize,"pso":pso_optimize}[which]
    tmr=_Timer(); ef=fast_eval_fn(n,tmr)
    x_lo=np.full(n,0.2); x_hi=np.full(n,2.0); x0=np.ones(n)
    t0=time.perf_counter()
    res=fn(ef,x0,x_lo,x_hi,budget=budget,n_mc=N_MC_BB,seed=seed*100+budget)
    total=time.perf_counter()-t0
    return total-tmr.t, int(res["spice_actual"])


def time_turbo(n, budget, seed):
    from methods.turbo.config import TuRBOConfig
    from methods.turbo.train import train_turbo
    tmr=_Timer()
    def obj(u):
        t0=time.perf_counter(); v=_fast_scalar(u,n); tmr.tick(time.perf_counter()-t0); return v
    cfg=TuRBOConfig()
    t0=time.perf_counter()
    res=train_turbo(obj,d=n,budget_queries=budget//N_MC_BB,cfg=cfg,seed=seed,
                    x0_unit=np.full(n,0.5))
    total=time.perf_counter()-t0
    return total-tmr.t, int(res["n_queries"])*N_MC_BB


def time_ra(circuit, budget, seed):
    from methods.robustanalog.benchmark_adapter import make_env, make_agent
    from methods.robustanalog.config import RAConfig
    from methods.robustanalog.train import train_robustanalog
    cfg=RAConfig(); env=make_env(circuit,cfg)
    tmr=_Timer(); n=env.action_dim; d_xi=env.xi_corners.shape[1]
    env.sim=FastSim(n,d_xi,tmr)          # swap real simulator for fast stub
    x_init=0.5*(env.x_lo+env.x_hi)       # valid midpoint design
    agent=make_agent(env,cfg,seed=seed)
    t0=time.perf_counter()
    res=train_robustanalog(env,agent,cfg,budget=budget,x_init_real=x_init,seed=seed)
    total=time.perf_counter()-t0
    return total-tmr.t, int(res["spice_used"])


def run_cell(args):
    method, circuit, budget, seed = args
    spec, n, d_xi = CIRCUITS[circuit]
    try:
        if method=="zo": ow,nc=time_zo(n,d_xi,budget,seed)
        elif method in ("bo","cmaes","pso"): ow,nc=time_lib_baseline(method,n,budget,seed)
        elif method=="turbo": ow,nc=time_turbo(n,budget,seed)
        elif method=="ra": ow,nc=time_ra(circuit,budget,seed)
        else: raise ValueError(method)
        return dict(method=method,circuit=circuit,budget=budget,seed=seed,
                    optimizer_wall=ow,n_calls=nc,ok=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        return dict(method=method,circuit=circuit,budget=budget,seed=seed,
                    optimizer_wall=float("nan"),n_calls=0,ok=False,err=str(e))


def measure_per_sim_wall(circuit, n_calls=30):
    """Median per-call real ngspice wall at x_init for one circuit."""
    from zo_yield.circuit_config import load_circuit_config, load_spec_module
    ccfg=load_circuit_config(circuit)
    mod=load_spec_module(circuit)
    alphas=ccfg["alphas"]
    sc=mod.make_sampler(scale=float(ccfg["sigma_scale"]))
    x_init=ccfg["x_init"]
    sim=mod.build_simulator(alphas=alphas)
    xis=sc.sample(n_calls,rng=np.random.default_rng(12345))
    ts=[]
    for xi in xis:
        t0=time.perf_counter(); sim.evaluate(x_init,xi); ts.append(time.perf_counter()-t0)
    return float(np.median(ts))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--circuits",default="cs_amp,cs_amp_3stage,cs_amp_5stage")
    ap.add_argument("--budgets",default="100,400,1600")
    ap.add_argument("--seeds",type=int,default=3)
    ap.add_argument("--methods",default="zo,bo,cmaes,pso,turbo,ra")
    ap.add_argument("--workers",type=int,default=12)
    a=ap.parse_args()
    circuits=[c for c in a.circuits.split(",") if c]
    budgets=[int(b) for b in a.budgets.split(",")]
    seeds=list(range(a.seeds)); methods=[m for m in a.methods.split(",") if m]

    print("Measuring per-call ngspice wall per circuit...",flush=True)
    per_sim={c:measure_per_sim_wall(c) for c in circuits}
    for c in circuits: print(f"  {c}: {per_sim[c]*1000:.1f} ms/sim",flush=True)

    cells=[(m,c,b,s) for m in methods for c in circuits for b in budgets for s in seeds]
    print(f"\nTiming {len(cells)} optimizer-internal cells (workers={a.workers})...",flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as ex:
        futs={ex.submit(run_cell,c):c for c in cells}
        for f in as_completed(futs):
            r=f.result(); rows.append(r)
            print(f"  {r['method']:6} {r['circuit']:14} B={r['budget']:5} s={r['seed']}: "
                  f"opt={r['optimizer_wall']*1000:8.1f} ms  calls={r['n_calls']}",flush=True)

    # aggregate: median over seeds
    RS=os.path.join(REPO,"experiments","results")
    agg={"per_sim_wall_ms":{c:per_sim[c]*1000 for c in circuits},"cells":{}}
    md=["# Wall-clock comparison — optimizer-internal vs SPICE (median over seeds)\n",
        "Optimizer-internal = real optimizer driven by a near-instant objective "
        "(matched n, n_mc, #queries); pure algorithm compute. SPICE wall = "
        "#calls × measured median ngspice wall/call. Same machine, fixed seeds. "
        "BO=scikit-optimize, CMA-ES=pycma (mature); PSO=lightweight self-rolled; "
        "TuRBO/RA=our reimpls (TuRBO GP pure-numpy → overhead upper bound).\n",
        "**Per-call ngspice wall:** " + ", ".join(f"{c}={per_sim[c]*1000:.0f}ms" for c in circuits) + "\n"]
    for c in circuits:
        spec,n,d_xi=CIRCUITS[c]
        md.append(f"\n## {c} (n={n}, d_ξ={d_xi})\n")
        md.append("| B | method | optimizer wall (median) | SPICE wall (median) | opt / total |")
        md.append("|---|---|---|---|---|")
        for b in budgets:
            for m in methods:
                sub=[r for r in rows if r["method"]==m and r["circuit"]==c and r["budget"]==b and r["ok"]]
                if not sub: continue
                ow=float(np.median([r["optimizer_wall"] for r in sub]))
                ncalls=int(np.median([r["n_calls"] for r in sub]))
                sw=ncalls*per_sim[c]
                frac=ow/(ow+sw) if (ow+sw)>0 else float("nan")
                agg["cells"][f"{c}|{b}|{m}"]=dict(optimizer_wall_s=ow,spice_wall_s=sw,
                    n_calls=ncalls,opt_frac=frac)
                fmt=lambda s: (f"{s*1000:.1f} ms" if s<1 else f"{s:.2f} s")
                md.append(f"| {b} | {m} | {fmt(ow)} | {fmt(sw)} | {frac*100:.1f}% |")
        md.append("")
    json.dump(agg,open(os.path.join(RS,"wallclock_comparison.json"),"w"),indent=2)
    open(os.path.join(RS,"wallclock_comparison.md"),"w").write("\n".join(md)+"\n")
    print("\nWrote wallclock_comparison.{md,json}",flush=True)


if __name__=="__main__":
    main()
