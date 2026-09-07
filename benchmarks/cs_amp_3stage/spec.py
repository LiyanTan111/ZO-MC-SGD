"""3-stage CS amplifier cascade benchmark spec.

Three byte-for-byte copies of the validated common-source amplifier,
AC-coupled in cascade. Each stage self-biases independently via a feedback
resistor. The cascade increases parameter count from n=6 to n=18 without
introducing any new failure mode beyond what the single CS amp already
handled.

Design vars (n = 18):
  Per stage k ∈ {1, 2, 3}: w_n_k, l_n_k, w_p_k, l_p_k, r_fb_k, i_bias_k

Process variations (d = 26):
  Per stage k: dvth_n_k, dkp_n_k, dl_n_k, dw_n_k,
               dvth_p_k, dkp_p_k, dl_p_k, dw_p_k       (8 × 3 = 24)
  Shared globals: dvth_g, dkp_g                       (2 — applied to all stages)

Loss form mirrors the single-stage CS amp (zo_yield.loss.combined_loss with
softplus penalties on gain / UGBW / power), but the spec targets are scaled
to the cascade regime (higher gain, lower UGBW, larger power budget).
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


# Default smooth-loss alpha values; will overwrite these
# from the chosen_alphas.json once α calibration runs.
DEFAULT_ALPHAS = dict(
    alpha_gain=0.5,         # per dB
    alpha_ugbw_log=1.0,     # per decade of log10(Hz)
    alpha_power_mW=0.5,     # per mW
)


# Spec targets — calibrated post-MC-probe at X_NOMINAL, scale=1.0:
#   gain mean=65 dB std=3.4 dB; UGBW mean=90 MHz std=38 MHz; P=0.3 mW.
# The targets below put each spec at roughly its MC mean so joint yield
# lands in the [0.30, 0.85] band; the σ_scale sweep will
# fine-tune the operating scale.
DEFAULT_TARGETS = dict(
    gain_target=60.0,        # dB; near MC mean (65) - 1.5σ — most draws clear
    ugbw_target=50.0e6,      # 50 MHz; MC mean (90 MHz) - 1σ — tighter binding
    power_target=0.5e-3,     # 0.5 mW; X_NOMINAL P = 0.3 mW so satisfied by X_NOMINAL
    vdd=1.0,
    rfb=10e6,                # default per-stage feedback resistor (used by X_NOMINAL only)
    cin=1e-9,                # input AC coupling cap
    cc=1e-9,                 # inter-stage AC coupling cap
    cload=1e-12,             # output load cap (treated as fixed in this benchmark)
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


# ---- Design variable layout (n = 18) -------------------------------------- #
# Order: stage 1 (6), stage 2 (6), stage 3 (6).
DESIGN_VARS = []
for k in (1, 2, 3):
    DESIGN_VARS += [f"w_n_{k}", f"l_n_{k}", f"w_p_{k}", f"l_p_{k}",
                    f"r_fb_{k}", f"i_bias_{k}"]


# Per-stage X_NOMINAL inherits from the validated CS amp X_NOMINAL.
# Single-stage was [w_n=20µm, l_n=0.18µm, w_p=40µm, l_p=0.5µm, ibias=50µA, cload=1pF].
# Cload is now a fixed target (not a design var); R_fb takes its slot.
_PER_STAGE_NOMINAL = np.array([20e-6, 0.18e-6, 40e-6, 0.5e-6, 10e6, 50e-6])
_PER_STAGE_LO = np.array([2e-6, 0.045e-6, 4e-6, 0.045e-6, 1e6, 5e-6])
_PER_STAGE_HI = np.array([200e-6, 2e-6, 400e-6, 4e-6, 100e6, 500e-6])

X_NOMINAL = np.tile(_PER_STAGE_NOMINAL, 3)
X_LO = np.tile(_PER_STAGE_LO, 3)
X_HI = np.tile(_PER_STAGE_HI, 3)


# ---- Variation variable layout (d = 26) ----------------------------------- #
# Order: per-stage 8 × 3 + 2 globals = 26. Globals (dvth_g, dkp_g) are at the
# end so the per-stage block can be indexed by [k * 8 : (k + 1) * 8] cleanly.
VARIATION_VARS = []
for k in (1, 2, 3):
    VARIATION_VARS += [f"dvth_n_{k}", f"dkp_n_{k}", f"dl_n_{k}", f"dw_n_{k}",
                       f"dvth_p_{k}", f"dkp_p_{k}", f"dl_p_{k}", f"dw_p_{k}"]
VARIATION_VARS += ["dvth_g", "dkp_g"]


def make_sampler(scale: float = 1.0) -> IndependentGaussianSampler:
    """Independent Gaussian sampler over the 26-dim ξ.

    Per-stage sigmas mirror the CS amp sampler's per-stage sigmas exactly
    (the cascade does not change Pelgrom-class mismatch magnitudes for any
    individual transistor). Shared global Vth / Kp shifts (last 2 entries)
    use the same magnitudes as the single-stage CS amp's globals.
    """
    per_stage = [0.020 * scale,   # dvth_n
                 0.05 * scale,    # dkp_n
                 5e-9 * scale,    # dl_n
                 0.02e-6 * scale, # dw_n
                 0.020 * scale,   # dvth_p
                 0.05 * scale,    # dkp_p
                 5e-9 * scale,    # dl_p
                 0.02e-6 * scale] # dw_p
    sigmas = np.array(per_stage * 3 + [0.010 * scale, 0.02 * scale])
    return IndependentGaussianSampler(mean=np.zeros_like(sigmas), std=sigmas)


# --------------------------------------------------------------------------- #
def _read_template() -> str:
    with open(os.path.join(HERE, "netlist.cir"), "r") as f:
        return f.read()


def _render_kwargs(x: np.ndarray, xi: np.ndarray, targets: dict) -> dict:
    """Build per-stage netlist substitution kwargs."""
    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if x.shape[0] != 18:
        raise ValueError(f"cs_amp_3stage x expected shape (18,), got {x.shape}")
    if xi.shape[0] != 26:
        raise ValueError(f"cs_amp_3stage xi expected shape (26,), got {xi.shape}")

    dvth_g = float(xi[24])
    dkp_g = float(xi[25])

    kw = dict(
        vdd=targets["vdd"],
        rfb=targets["rfb"],         # unused by templated netlist (per-stage rfb_k)
        cin=targets["cin"],
        cc=targets["cc"],
        cload=targets["cload"],
    )
    for k in (1, 2, 3):
        # Design-var slice: per-stage 6 vars starting at (k-1)*6.
        base_x = (k - 1) * 6
        w_n, l_n, w_p, l_p, r_fb, i_bias = (float(x[base_x + j]) for j in range(6))

        # ξ slice: per-stage 8 vars starting at (k-1)*8.
        base_xi = (k - 1) * 8
        dvth_n = float(xi[base_xi + 0])
        dkp_n = float(xi[base_xi + 1])
        dl_n = float(xi[base_xi + 2])
        dw_n = float(xi[base_xi + 3])
        dvth_p = float(xi[base_xi + 4])
        dkp_p = float(xi[base_xi + 5])
        dl_p = float(xi[base_xi + 6])
        dw_p = float(xi[base_xi + 7])

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
class CSAmp3StageSimulator(BlackBoxSimulator):
    """3-stage CS amp cascade simulator — subprocess-isolated ngspice.

    Power model uses the per-device current sum from :
    P = Vdd · |i(Vdd)| read from the OP analysis. Falls back to the
    closed-form ``Vdd · 2 · (Σ ibias_k)`` if the OP read fails.
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
        # OP-current power model from doesn't apply here:
        # the cascade inherits the single-stage CS amp's degenerate-bias OP
        # (i(vdd) ≈ 0, see build notes). Power uses the closed-form
        # ``P = Vdd · 2 · Σ ibias_k`` instead — same convention as the
        # validated common_source_amp.
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

        # UGBW = first 0-dB crossing after the peak (same logic as CS amp).
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

        # Closed-form: each stage draws ~2 × ibias_k (mirror + load branch).
        ibias_sum = float(x[5]) + float(x[11]) + float(x[17])
        power = float(self.targets["vdd"]) * 2.0 * ibias_sum
        return dict(gain_db=gain_db, ugbw_hz=ugbw,
                    phase_margin_deg=pm, power_w=power)

    def _loss(self, metrics: dict) -> float:
        pens = make_penalties(targets=self.targets, alphas=self.alphas)
        # Same gain-bonus convention as the single-stage CS amp.
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
                    alphas: Optional[dict] = None) -> CSAmp3StageSimulator:
    return CSAmp3StageSimulator(targets=targets, alphas=alphas)


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
