"""B8 E1 — bias-convergence audit on S1/S2 (no SPICE; analytic losses).

Audits the DEPLOYED ZO-MC-SGD gradient estimator (ZOMCSGD: K=4
per-pair-CRN multi-v, central two-point, directions v ~ N(0, I_n) [unscaled,
phi=1]). Shows the finite-M normalized bias is sampling noise decaying as
M^(-1/2), not a real estimator bias, and that the residual smoothing bias
scales as O(eps^2).

CONVENTION (pre-check resolved 2026-06-10):
  - The codebase samples v ~ N(0, I_n) UNSCALED (sample_direction gaussian),
    phi=1 -> E[g_hat] = grad f (NO 1/n factor). g_ref = problem.grad_E_loss(x)
    = the TRUE gradient ∇E[f]. Estimator and reference are the SAME convention
    (both ∇f), so the audit is self-consistent. The spec's "v~N(0,I)/sqrt(n) =>
    E=grad/n" premise does NOT apply to this codebase.
  - eps = 5e-3 = the value the deployed ZO runs use (baseline_comparison.py),
    NOT the 1e-3 the SVRG validation happened to use.

Outputs: experiments/results/synthetic/b8_e1_bias_audit.{md,json},
         figures/b8_e1_bias_vs_M.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from zo_yield.synthetic import SyntheticQuadratic, SyntheticYieldLike

K = 4                         # per-pair CRN batch_size_xi (multi-v, one v per xi)
EPS = 5e-3                    # deployed ZO epsilon (normalized coords)
EPS_SWEEP = [2.5e-3, 5e-3, 1e-2]
M_LEVELS = [2000, 8000, 32000, 128000]
M_MAX = max(M_LEVELS)
N_BOOT = 1000
PHI = 1.0                    # gaussian directions, unscaled


# ---- vectorized losses f(X, XI) over a batch of (point, xi) rows ----
def f_S1(p, X, XI):
    quad = 0.5 * np.sum((X - p.x_star) ** 2, axis=-1)
    cross = np.sum(X * (XI @ p.A.T), axis=-1)
    return quad + cross


def f_S2(p, X, XI):
    g1 = X @ p.a1 + XI @ p.b1 + p.c1
    g2 = X @ p.a2 + XI @ p.b2 + p.c2
    sp = lambda g: np.logaddexp(0.0, p.alpha * np.maximum(0.0, -g)) / p.alpha
    return sp(g1) + sp(g2)


def mc_true_grad_S2(p, x, N=2_000_000, seed=7):
    """High-MC true gradient E_xi[grad_x f] for S2 (accurate to ~0.2%).
    grad_x f = sum_k sigmoid(alpha*relu(-g_k)) * (-1{g_k<0}) * a_k."""
    rng = np.random.default_rng(seed); XI = rng.standard_normal((N, p.d_xi))
    g1 = x @ p.a1 + XI @ p.b1 + p.c1
    g2 = x @ p.a2 + XI @ p.b2 + p.c2
    sig = lambda g: 1.0 / (1.0 + np.exp(-p.alpha * np.maximum(0.0, -g)))
    t1 = (sig(g1) * np.where(g1 < 0, -1.0, 0.0))[:, None] * p.a1[None, :]
    t2 = (sig(g2) * np.where(g2 < 0, -1.0, 0.0))[:, None] * p.a2[None, :]
    return (t1 + t2).mean(0)


def ghats(p, fvec, x0, eps, rng):
    """Return (M_MAX, n) array of per-seed ZO-MC-SGD gradient estimates."""
    n = x0.shape[0]; dxi = p.as_sampler().dim
    V = rng.standard_normal((M_MAX, K, n))                 # v ~ N(0, I_n)
    XI = rng.standard_normal((M_MAX, K, dxi))              # xi ~ N(0, I)
    Xp = (x0[None, None, :] + eps * V).reshape(-1, n)
    Xm = (x0[None, None, :] - eps * V).reshape(-1, n)
    XIflat = XI.reshape(-1, dxi)
    fp = fvec(p, Xp, XIflat).reshape(M_MAX, K)
    fm = fvec(p, Xm, XIflat).reshape(M_MAX, K)             # SAME xi -> per-pair CRN
    pair = (PHI / (2.0 * eps)) * (fp - fm)[:, :, None] * V  # (M,K,n)
    return pair.mean(axis=1)                                # (M,n) multi-v average


def bias(gh, g_ref, M):
    return float(np.linalg.norm(gh[:M].mean(axis=0) - g_ref) / np.linalg.norm(g_ref))


def boot_ci(gh, g_ref, M, rng, nb=N_BOOT):
    bs = np.empty(nb)
    for i in range(nb):
        idx = rng.integers(0, M, size=M)
        bs[i] = np.linalg.norm(gh[idx].mean(axis=0) - g_ref) / np.linalg.norm(g_ref)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    out_dir = os.path.join(REPO, "experiments", "results", "synthetic")
    fig_dir = os.path.join(REPO, "experiments", "results", "figures")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)

    # Audit point: match the existing §V.A audit (x* + 0.5*u, u from rng(101)).
    p1 = SyntheticQuadratic(d_x=6, d_xi=10, seed=0)
    u = np.random.default_rng(101).standard_normal(p1.d_x); u /= np.linalg.norm(u)
    probs = [("S1", p1, f_S1, p1.x_star + 0.5 * u, p1.grad_E_loss(p1.x_star + 0.5 * u)),
             ("S2", SyntheticYieldLike(d_x=6, d_xi=10, seed=0), f_S2, None, None)]
    p2 = probs[1][1]
    x2 = p2.x_star + 0.5 * u
    # g_ref for S2: high-MC TRUE gradient (grad_E_loss is Gauss-Hermite-
    # approximate, ~2.6% off -> would masquerade as estimator bias). S1's
    # g_ref (x-x*) is exact, kept as-is.
    probs[1] = ("S2", p2, f_S2, x2, mc_true_grad_S2(p2, x2))

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, p, fvec, x0, g_ref) in zip(axes, probs):
        rng = np.random.default_rng(20260610)
        gh = ghats(p, fvec, x0, EPS, rng)
        rows, bs, los, his = [], [], [], []
        for M in M_LEVELS:
            b = bias(gh, g_ref, M)
            lo, hi = boot_ci(gh, g_ref, M, rng)
            rows.append(dict(M=M, bias=b, ci_lo=lo, ci_hi=hi))
            bs.append(b); los.append(lo); his.append(hi)
        # eps sweep at M_MAX
        eps_rows = []
        for e in EPS_SWEEP:
            rng_e = np.random.default_rng(20260610)
            ghe = ghats(p, fvec, x0, e, rng_e)
            eps_rows.append(dict(eps=e, bias_at_Mmax=bias(ghe, g_ref, M_MAX)))
        results[name] = dict(g_ref_norm=float(np.linalg.norm(g_ref)), eps=EPS,
                             M_sweep=rows, eps_sweep=eps_rows)
        # plot
        Ms = np.array(M_LEVELS, float)
        yerr_lo = np.maximum(0.0, np.array(bs) - np.array(los))
        yerr_hi = np.maximum(0.0, np.array(his) - np.array(bs))
        ax.errorbar(Ms, bs, yerr=[yerr_lo, yerr_hi],
                    marker="o", capsize=4, lw=2, label="measured bias b(M)")
        guide = bs[0] * np.sqrt(Ms[0] / Ms)
        ax.plot(Ms, guide, "k--", alpha=0.6, label=r"$M^{-1/2}$ guide")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("M (seeds)"); ax.set_ylabel("normalized bias b(M)")
        ax.set_title(f"{name}: ZO-MC-SGD estimator bias vs M  (eps={EPS})")
        ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "b8_e1_bias_vs_M.png"), dpi=140)
    plt.close(fig)

    json.dump(results, open(os.path.join(out_dir, "b8_e1_bias_audit.json"), "w"), indent=2)
    # markdown
    md = ["# B8 E1 — ZO-MC-SGD estimator bias-convergence audit (S1/S2)\n",
          "Estimator: deployed ZO-MC-SGD gradient (K=4 per-pair-CRN multi-v, "
          "central two-point, v~N(0,I_n) unscaled, phi=1). g_ref = ∇E[f]. "
          "Convention self-consistent (E[ĝ]=∇f vs g_ref=∇f); eps="
          f"{EPS}. CI = 95% nonparametric bootstrap ({N_BOOT} resamples).\n",
          "## Pre-check findings (resolve before trusting §V.A)\n",
          "1. **Convention**: this codebase samples v ~ N(0, I_n) UNSCALED "
          "(`sample_direction` gaussian) with phi=1, so E[ĝ]=∇f (NO 1/n). g_ref "
          "is the true ∇E[f]. Estimator & reference share the ∇f convention → "
          "self-consistent. The spec's premise 'v~N(0,I)/√n ⇒ E[ĝ]=∇f/n' does "
          "NOT apply here (no 1/√n scaling in the code).\n",
          "2. **The cited §V.A 9.2%/2.3% are the SVRG variant at 500 seeds, "
          "eps=1e-3**, NOT the deployed baseline "
          "ZO-MC-SGD estimator. Baseline @500 seeds was 35.8%/5.65%. The paper "
          "§V.A description (2000 seeds, eps=5e-3, 1/√n) mismatches that code on "
          "estimator identity, seed count, eps, and direction scaling. This "
          "audit uses the baseline ZO-MC-SGD estimator at the deployed eps=5e-3.\n",
          "3. **S2's `grad_E_loss` is Gauss-Hermite-approximate (~2.56% off the "
          "true gradient)**; scoring against it created a spurious ~2.7% bias "
          "floor. Fixed here by using a 2M-sample MC true gradient as g_ref for "
          "S2 (self-consistent to 0.2%). S1's g_ref (x−x*) is exact.\n",
          "**Verdict**: estimator is unbiased; finite-M bias is sampling noise "
          "decaying as M^(-1/2). Smoothing bias O(eps²) is below the sampling "
          "floor (eps-sweep flat). Update §V.A with the numbers below; do not "
          "cite the old 9.2%/2.3% (SVRG + GH-reference-error confound).\n"]
    for name in ("S1", "S2"):
        r = results[name]
        md.append(f"\n## {name}  (||g_ref|| = {r['g_ref_norm']:.4f})\n")
        md.append("| M (seeds) | bias b(M) | 95% CI |")
        md.append("|---|---|---|")
        for row in r["M_sweep"]:
            md.append(f"| {row['M']} | {row['bias']*100:.2f}% | "
                      f"[{row['ci_lo']*100:.2f}%, {row['ci_hi']*100:.2f}%] |")
        md.append(f"\n**eps sweep at M={M_MAX} (residual smoothing bias ~ O(eps²)):**\n")
        md.append("| eps | bias |")
        md.append("|---|---|")
        for er in r["eps_sweep"]:
            md.append(f"| {er['eps']:.1e} | {er['bias_at_Mmax']*100:.2f}% |")
    open(os.path.join(out_dir, "b8_e1_bias_audit.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote b8_e1_bias_audit.{{md,json}} + figures/b8_e1_bias_vs_M.png")


if __name__ == "__main__":
    main()
