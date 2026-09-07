"""5-stage CS amplifier cascade benchmark spec.

Vertical extension of cs_amp_3stage. Five byte-for-byte copies of the
validated common-source amplifier, AC-coupled in cascade.
Each stage self-biases independently via a feedback resistor.

Design vars (n = 30):
  Per stage k ∈ {1, 2, 3, 4, 5}: w_n_k, l_n_k, w_p_k, l_p_k, r_fb_k, i_bias_k

Process variations (d = 42):
  Per stage k: dvth_n_k, dkp_n_k, dl_n_k, dw_n_k,
               dvth_p_k, dkp_p_k, dl_p_k, dw_p_k       (8 × 5 = 40)
  Shared globals: dvth_g, dkp_g                       (2 — applied to all stages)

Loss form mirrors cs_amp_3stage: smoothed-spec on gain / UGBW / power.
Spec targets scaled to the 5-stage cascade regime (~5× single-stage gain).
"""
from __future__ import annotations

import math as _math
import os
from typing import Optional

import numpy as np

from simulators.base import BlackBoxSimulator
from zo_yield.loss import Penalty, combined_loss
from zo_yield.samplers import IndependentGaussianSampler


HERE = os.path.dirname(os.path.abspath(__file__))


DEFAULT_ALPHAS = dict(
    alpha_gain=0.5,         # per dB
    alpha_ugbw_log=1.0,     # per decade of log10(Hz)
    alpha_power_mW=0.5,     # per mW
)


# Spec targets — calibrated post-MC-probe at X_NOMINAL.
# Initial guesses scaled from cs_amp_3stage:
#   - cs_amp_3stage nominal gain ≈ 75 dB → 5-stage ≈ 125 dB
#   - cs_amp_3stage nominal UGBW MHz scale → 5-stage UGBW lower
#   - cs_amp_3stage nominal P ≈ 0.3 mW → 5-stage P ≈ 0.5 mW (5/3 × stages)
DEFAULT_TARGETS = dict(
    # Tightened post-sanity (): MC at X_NOMINAL, scale=1, gave gain
    # mean ≈ 94 dB / p10=89.5 / p50=93.2 / p90=99.6. With gain target = 80
    # the yield was ~1.0 (too easy); tightened to 95 so σ_scale calibration
    # in lands yield(X_NOMINAL) in [0.30, 0.85] at scale ∈ {1, 2}.
    gain_target=95.0,        # dB
    ugbw_target=30.0e6,      # 30 MHz; UGBW maxes at sweep limit anyway
    power_target=0.7e-3,     # 0.7 mW; X_NOMINAL P = 0.5 mW so headroom exists
    vdd=1.0,
    rfb=10e6,
    cin=1e-9,
    cc=1e-9,
    cload=1e-12,
)


def make_penalties(targets=None, alphas=None):
    t = targets or DEFAULT_TARGETS
    a = dict(DEFAULT_ALPHAS)
    if alphas:
        a.update(alphas)
    return [
        Penalty(
            name="gain_dB",
            extract=lambda s: s["gain_db"],
            kind="ge",
            target=t["gain_target"],
            weight=1.0,
            alpha=a["alpha_gain"],
        ),
        Penalty(
            name="ugbw_log",
            extract=lambda s: _math.log10(max(s["ugbw_hz"], 1.0)),
            kind="ge",
            target=_math.log10(max(t["ugbw_target"], 1.0)),
            weight=2.0,
            alpha=a["alpha_ugbw_log"],
        ),
        Penalty(
            name="power_mW",
            extract=lambda s: s["power_w"] * 1e3,
            kind="le",
            target=t["power_target"] * 1e3,
            weight=1.0,
            alpha=a["alpha_power_mW"],
        ),
    ]


# ---- Design variable layout (n = 30) -------------------------------------- #
DESIGN_VARS = []
for k in (1, 2, 3, 4, 5):
    DESIGN_VARS += [f"w_n_{k}", f"l_n_{k}", f"w_p_{k}", f"l_p_{k}",
                    f"r_fb_{k}", f"i_bias_{k}"]


# Per-stage X_NOMINAL inherits from cs_amp_3stage (same per-stage pattern).
_PER_STAGE_NOMINAL = np.array([20e-6, 0.18e-6, 40e-6, 0.5e-6, 10e6, 50e-6])
_PER_STAGE_LO = np.array([2e-6, 0.045e-6, 4e-6, 0.045e-6, 1e6, 5e-6])
_PER_STAGE_HI = np.array([200e-6, 2e-6, 400e-6, 4e-6, 100e6, 500e-6])

X_NOMINAL = np.tile(_PER_STAGE_NOMINAL, 5)
X_LO = np.tile(_PER_STAGE_LO, 5)
X_HI = np.tile(_PER_STAGE_HI, 5)


# ---- Variation variable layout (d = 42) ----------------------------------- #
VARIATION_VARS = []
for k in (1, 2, 3, 4, 5):
    VARIATION_VARS += [f"dvth_n_{k}", f"dkp_n_{k}", f"dl_n_{k}", f"dw_n_{k}",
                       f"dvth_p_{k}", f"dkp_p_{k}", f"dl_p_{k}", f"dw_p_{k}"]
VARIATION_VARS += ["dvth_g", "dkp_g"]


def make_sampler(scale: float = 1.0) -> IndependentGaussianSampler:
    """Independent Gaussian sampler over the 42-dim ξ.

    Per-stage sigmas inherit from cs_amp_3stage exactly (per-transistor
    Pelgrom mismatch doesn't change with cascade depth). Shared global
    Vth / Kp shifts (last 2 entries) match the single-stage CS amp's globals.
    No per-device ΔW (cs_amp self-bias breaks symmetry without it; same
    rationale as cs_amp_3stage).
    """
    per_stage = [0.020 * scale,   # dvth_n
                 0.05 * scale,    # dkp_n
                 5e-9 * scale,    # dl_n
                 0.02e-6 * scale, # dw_n
                 0.020 * scale,   # dvth_p
                 0.05 * scale,    # dkp_p
                 5e-9 * scale,    # dl_p
                 0.02e-6 * scale] # dw_p
    sigmas = np.array(per_stage * 5 + [0.010 * scale, 0.02 * scale])
    return IndependentGaussianSampler(mean=np.zeros_like(sigmas), std=sigmas)


# --------------------------------------------------------------------------- #
def _read_template() -> str:
    with open(os.path.join(HERE, "netlist.cir"), "r") as f:
        return f.read()


def _render_kwargs(x: np.ndarray, xi: np.ndarray, targets: dict) -> dict:
    """Build per-stage netlist substitution kwargs."""
    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if x.shape[0] != 30:
        raise ValueError(f"cs_amp_5stage x expected shape (30,), got {x.shape}")
    if xi.shape[0] != 42:
        raise ValueError(f"cs_amp_5stage xi expected shape (42,), got {xi.shape}")

    dvth_g = float(xi[40])
    dkp_g = float(xi[41])

    kw = dict(
        vdd=targets["vdd"],
        rfb=targets["rfb"],
        cin=targets["cin"],
        cc=targets["cc"],
        cload=targets["cload"],
    )
    for k in (1, 2, 3, 4, 5):
        base_x = (k - 1) * 6
        w_n, l_n, w_p, l_p, r_fb, i_bias = (float(x[base_x + j]) for j in range(6))

        base_xi = (k - 1) * 8
        dvth_n = float(xi[base_xi + 0])
        dkp_n  = float(xi[base_xi + 1])
        dl_n   = float(xi[base_xi + 2])
        dw_n   = float(xi[base_xi + 3])
        dvth_p = float(xi[base_xi + 4])
        dkp_p  = float(xi[base_xi + 5])
        dl_p   = float(xi[base_xi + 6])
        dw_p   = float(xi[base_xi + 7])

        vto_n_k = 0.4 + dvth_n + dvth_g
        vto_p_k = -0.4 + dvth_p + dvth_g
        kp_n_k = 300e-6 * (1.0 + dkp_n + dkp_g)
        kp_p_k = 120e-6 * (1.0 + dkp_p + dkp_g)
        wn_eff_k = max(w_n + dw_n, 1e-9)
        ln_eff_k = max(l_n + dl_n, 1e-9)
        wp_eff_k = max(w_p + dw_p, 1e-9)
        lp_eff_k = max(l_p + dl_p, 1e-9)

        kw.update({
            f"vto_n_{k}": vto_n_k, f"vto_p_{k}": vto_p_k,
            f"kp_n_{k}": kp_n_k,   f"kp_p_{k}": kp_p_k,
            f"wn_eff_{k}": wn_eff_k, f"ln_eff_{k}": ln_eff_k,
            f"wp_eff_{k}": wp_eff_k, f"lp_eff_{k}": lp_eff_k,
            f"rfb_{k}": r_fb,        f"ibias_{k}": i_bias,
        })
    return kw


# --------------------------------------------------------------------------- #
class CSAmp5StageSimulator(BlackBoxSimulator):
    """5-stage CS amp cascade — subprocess-isolated ngspice.

    Power model: closed-form ``P = Vdd · 2 · Σ ibias_k`` (the cs_amp_3stage
    OP-current model returned ≈ 0 due to degenerate bias loop; the closed
    form is consistent with cs_amp_3stage and single-stage CS amp).
    """

    def __init__(self, targets: Optional[dict] = None,
                 alphas: Optional[dict] = None):
        super().__init__()
        self.targets = targets or DEFAULT_TARGETS
        self.alphas = alphas or {}
        self.template = _read_template()
        try:
            from simulators.ngspice_subprocess import run_ngspice_subprocess  # noqa
            self._ok = True
        except Exception:
            self._ok = False

    def _render(self, x: np.ndarray, xi: np.ndarray) -> str:
        return self.template.format(**_render_kwargs(x, xi, self.targets))

    def _run_and_extract(self, netlist_str: str, x: np.ndarray) -> dict:
        from simulators.ngspice_subprocess import run_ngspice_subprocess
        res = run_ngspice_subprocess(
            netlist_str,
            analysis_cmd="ac dec 20 1 1g",
            plot_name="ac1",
            nodes=["out", "frequency"],
        )
        vout = res["out"]
        freqs = np.real(res["frequency"])
        mag = np.abs(vout)
        gain_db_arr = 20.0 * np.log10(np.maximum(mag, 1e-30))
        idx_peak = int(np.argmax(gain_db_arr))
        gain_db = float(gain_db_arr[idx_peak])

        ugbw = 0.0
        if gain_db > 0:
            tail = gain_db_arr[idx_peak:]
            below = np.where(tail < 0)[0]
            if below.size == 0:
                ugbw = float(freqs[-1])
            else:
                i = idx_peak + below[0]
                if i == 0:
                    ugbw = float(freqs[0])
                else:
                    f0, f1 = float(freqs[i - 1]), float(freqs[i])
                    g0, g1 = float(gain_db_arr[i - 1]), float(gain_db_arr[i])
                    if g1 == g0:
                        ugbw = f0
                    else:
                        logf = np.log10(f0) + (np.log10(f1) - np.log10(f0)) * (0 - g0) / (g1 - g0)
                        ugbw = 10.0**logf

        phase = np.unwrap(np.angle(vout)) * 180.0 / np.pi
        if ugbw > 0:
            ph_at_ugbw = float(np.interp(np.log10(ugbw),
                                          np.log10(np.maximum(freqs, 1e-30)), phase))
        else:
            ph_at_ugbw = -180.0
        pm = 180.0 + ph_at_ugbw

        # Closed-form per-stage power.
        ibias_sum = sum(float(x[5 + 6 * k]) for k in range(5))
        power = float(self.targets["vdd"]) * 2.0 * ibias_sum
        return dict(gain_db=gain_db, ugbw_hz=ugbw,
                    phase_margin_deg=pm, power_w=power)

    def _loss(self, metrics: dict) -> float:
        pens = make_penalties(targets=self.targets, alphas=self.alphas)
        objective = -0.1 * metrics["gain_db"]
        return float(combined_loss(pens, dict(metrics), objective=objective))

    def evaluate(self, x: np.ndarray, xi: np.ndarray) -> float:
        self.n_calls += 1
        if not self._ok:
            self.n_failures += 1
            return self.LOSS_PENALTY
        try:
            netlist = self._render(x, xi)
        except Exception:
            self.n_failures += 1
            return self.LOSS_PENALTY
        try:
            metrics = self._run_and_extract(netlist, x)
            loss = self._loss(metrics)
            import math
            if not math.isfinite(loss):
                self.n_failures += 1
                return self.LOSS_PENALTY
            return loss
        except Exception:
            self.n_failures += 1
            return self.LOSS_PENALTY

    def evaluate_with_metrics(self, x: np.ndarray, xi: np.ndarray) -> tuple[float, dict]:
        netlist = self._render(x, xi)
        metrics = self._run_and_extract(netlist, x)
        return self._loss(metrics), metrics


def build_simulator(targets: Optional[dict] = None,
                    alphas: Optional[dict] = None) -> CSAmp5StageSimulator:
    return CSAmp5StageSimulator(targets=targets, alphas=alphas)


# --------------------------------------------------------------------------- #
# Yield specs                                                                 #
# --------------------------------------------------------------------------- #
from zo_yield.evaluation import Spec  # noqa: E402


def build_specs(targets: Optional[dict] = None) -> list:
    t = targets or DEFAULT_TARGETS
    g_t = t["gain_target"]
    u_t = t["ugbw_target"]
    p_t = t["power_target"]
    return [
        Spec(
            name="gain_db",
            extract=lambda m: m["gain_db"],
            satisfies=lambda v: v >= g_t,
            margin=lambda v: g_t - v,
        ),
        Spec(
            name="ugbw_hz",
            extract=lambda m: m["ugbw_hz"],
            satisfies=lambda v: v >= u_t,
            margin=lambda v: u_t - v,
        ),
        Spec(
            name="power_w",
            extract=lambda m: m["power_w"],
            satisfies=lambda v: v <= p_t,
            margin=lambda v: v - p_t,
        ),
    ]


SPECS = build_specs()
