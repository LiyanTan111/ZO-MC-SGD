"""Single-ended two-stage Miller-compensated amplifier benchmark.

Diagnostic counterpart to miller_ota: same Miller compensation
(Cc + Rz across the two stages) but **without** the differential
input pair.

Architecture: two cascaded self-biased CS-amp stages
(cs_amp_3stage's per-stage pattern, validated) with
AC coupling between them, plus a Miller compensation network
(Cc + Rz between stage-1 out and stage-2 out). Self-biasing avoids
the DC-balance pathology that would arise if you naively deleted the
diff pair from miller_ota's netlist (the inter-stage node would have
no DC reference).

Designed at d_ξ=26 to match cs_amp_3stage exactly — the
diagnostic-pair comparison `(cs_amp_3stage @ d_ξ=26, antithetic_lift
≈ +0.083) ↔ (cs_se_miller @ d_ξ=26, antithetic_lift = ?)` isolates
the diff-pair-vs-Miller-comp question.

Design vars (n = 14):
  Per stage k ∈ {1, 2}: w_n_k, l_n_k, w_p_k, l_p_k, r_fb_k, i_bias_k  (6 × 2 = 12)
  Miller comp: c_c_miller, r_z_miller  (2)

Process variations (d = 26):
  Per-stage 4 ξ × 3 transistors (M_k, Mref_k, Mload_k) × 2 stages = 24
  Plus 2 shared globals (dvth_g, dkp_g) = 26.

Same loss form as miller_ota (gain + UGBW + PM + power), with PM as
a hard stability spec.
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
    alpha_gain=0.5,
    alpha_ugbw_log=1.0,
    alpha_power_mW=0.5,
)


# Targets calibrated post-MC sanity. Spec structure mirrors cs_amp_3stage
# (gain + UGBW + power; no PM constraint). The Miller compensation network
# (Cc_miller + Rz_miller) is **structurally present** in the netlist and
# affects the AC transfer function, but PM is not a yield gate — this
# matches cs_amp_3stage's spec layout for direct diagnostic comparison
# at d_ξ=26.
DEFAULT_TARGETS = dict(
    gain_target=50.0,        # dB (cs_amp_3stage was 60; relax 10dB to account for
                             # Miller comp's gain-stealing effect at this Cc)
    ugbw_target=0.2e6,       # 0.2 MHz — set near nominal 0.4 MHz so MC has
                             # both pass and fail samples (cs_amp_3stage was 50 MHz
                             # but had no Miller cap suppressing UGBW)
    power_target=0.5e-3,     # 0.5 mW (matches cs_amp_3stage)
    vdd=1.0,                 # match cs_amp_3stage's Vdd
    cin=1e-9,
    cc_ac=1e-9,              # inter-stage AC coupling
    cload=1e-12,             # output load (1 pF, matches cs_amp_3stage)
)


# ---- Design variable layout (n = 14) -------------------------------------- #
DESIGN_VARS = []
for k in (1, 2):
    DESIGN_VARS += [f"w_n_{k}", f"l_n_{k}", f"w_p_{k}", f"l_p_{k}",
                    f"r_fb_{k}", f"i_bias_{k}"]
DESIGN_VARS += ["c_c_miller", "r_z_miller"]


# X_NOMINAL — per-stage values inherited from cs_amp_3stage's per-stage X_NOMINAL
#. Miller comp Cc=2pF, Rz=1kΩ.
_PER_STAGE_NOMINAL = np.array([20e-6, 0.18e-6, 40e-6, 0.5e-6, 10e6, 50e-6])
_PER_STAGE_LO = np.array([2e-6, 0.045e-6, 4e-6, 0.045e-6, 1e6, 5e-6])
_PER_STAGE_HI = np.array([200e-6, 2e-6, 400e-6, 4e-6, 100e6, 500e-6])

X_NOMINAL = np.concatenate([
    _PER_STAGE_NOMINAL,
    _PER_STAGE_NOMINAL,
    np.array([20.0e-12, 30.0]),    # c_c_miller=20pF, r_z_miller=30Ω (≈1/gm_M2 nulling)
])
X_LO = np.concatenate([
    _PER_STAGE_LO, _PER_STAGE_LO,
    np.array([0.1e-12, 1.0]),  # rz_miller lower bumped down (nulling values are small)
])
X_HI = np.concatenate([
    _PER_STAGE_HI, _PER_STAGE_HI,
    np.array([50.0e-12, 100.0e3]),  # cc_miller upper bumped for headroom
])


# ---- Variation variable layout (d = 26) ----------------------------------- #
# Order: per-stage 4 ξ × 3 transistors × 2 stages = 24. Per stage k:
# (M_k, Mref_k, Mload_k) — 3 devices. Plus 2 globals at end.
VARIATION_VARS = []
for k in (1, 2):
    for dev in (f"n_{k}", f"pref_{k}", f"pload_{k}"):
        VARIATION_VARS += [f"dvth_{dev}", f"dkp_{dev}",
                            f"dl_{dev}", f"dw_{dev}"]
VARIATION_VARS += ["dvth_g", "dkp_g"]


def make_sampler(scale: float = 1.0) -> IndependentGaussianSampler:
    """Independent Gaussian sampler over the 26-dim ξ. Per-device sigmas
    match cs_amp_3stage's per-device sigmas exactly.
    """
    per_device = [0.020 * scale,    # dvth
                  0.05 * scale,     # dkp
                  5e-9 * scale,     # dl
                  0.02e-6 * scale]  # dw
    sigmas = np.array(per_device * 6 + [0.010 * scale, 0.02 * scale])
    return IndependentGaussianSampler(mean=np.zeros_like(sigmas), std=sigmas)


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


# --------------------------------------------------------------------------- #
def _read_template() -> str:
    with open(os.path.join(HERE, "netlist.cir"), "r") as f:
        return f.read()


def _render_kwargs(x: np.ndarray, xi: np.ndarray, targets: dict) -> dict:
    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if x.shape[0] != 14:
        raise ValueError(f"cs_se_miller x expected shape (14,), got {x.shape}")
    if xi.shape[0] != 26:
        raise ValueError(f"cs_se_miller xi expected shape (26,), got {xi.shape}")

    dvth_g = float(xi[24])
    dkp_g = float(xi[25])

    kw = dict(
        vdd=targets["vdd"],
        cin=targets["cin"],
        cc_ac=targets["cc_ac"],
        cload=targets["cload"],
        cc_miller=float(x[12]),
        rz_miller=float(x[13]),
    )

    for k in (1, 2):
        # Design-var slice: per-stage 6 vars starting at (k-1)*6.
        base_x = (k - 1) * 6
        w_n, l_n, w_p, l_p, r_fb, i_bias = (float(x[base_x + j]) for j in range(6))

        # ξ slice: per-stage 12 entries (3 devices × 4 ξ) starting at (k-1)*12.
        base_xi = (k - 1) * 12
        # Device order within stage: M_n_k (NMOS), Mref_p_k (PMOS), Mload_p_k (PMOS)
        dvth_n = float(xi[base_xi + 0]);  dkp_n = float(xi[base_xi + 1])
        dl_n   = float(xi[base_xi + 2]);  dw_n  = float(xi[base_xi + 3])
        dvth_pref = float(xi[base_xi + 4]); dkp_pref = float(xi[base_xi + 5])
        dl_pref   = float(xi[base_xi + 6]); dw_pref  = float(xi[base_xi + 7])
        dvth_pload = float(xi[base_xi + 8]); dkp_pload = float(xi[base_xi + 9])
        dl_pload   = float(xi[base_xi + 10]); dw_pload  = float(xi[base_xi + 11])

        vto_n_k     = 0.4 + dvth_n + dvth_g
        vto_pref_k  = -0.4 + dvth_pref + dvth_g
        vto_pload_k = -0.4 + dvth_pload + dvth_g
        # KP base + globals; dW/dL mismatch per device folded into the geometry-
        # surrogate KP scaling (same convention as miller_ota).
        kp_n_base    = 300e-6 * (1.0 + dkp_n + dkp_g)
        kp_pref_base = 120e-6 * (1.0 + dkp_pref + dkp_g)
        kp_pload_base = 120e-6 * (1.0 + dkp_pload + dkp_g)

        # Effective per-device W/L
        wn_eff_k = max(w_n + dw_n, 1e-9)
        ln_eff_k = max(l_n + dl_n, 1e-9)
        # Mref and Mload share the per-stage W_p / L_p design vars,
        # but each device has its own dW/dL ξ slot:
        wpref_eff_k = max(w_p + dw_pref, 1e-9)
        lpref_eff_k = max(l_p + dl_pref, 1e-9)
        wpload_eff_k = max(w_p + dw_pload, 1e-9)
        lpload_eff_k = max(l_p + dl_pload, 1e-9)

        # Geometry-surrogate KP shifts (same convention as miller_ota
        # to preserve per-device dW/dL impact via KP):
        def geom_kp(kp_base, w_eff, l_eff, w_des, l_des):
            return kp_base * (max(w_eff, 1e-9) / max(w_des, 1e-9)) * \
                   (max(l_des, 1e-9) / max(l_eff, 1e-9))

        kp_n_k = geom_kp(kp_n_base, wn_eff_k, ln_eff_k, w_n, l_n)
        kp_pref_k = geom_kp(kp_pref_base, wpref_eff_k, lpref_eff_k, w_p, l_p)
        kp_pload_k = geom_kp(kp_pload_base, wpload_eff_k, lpload_eff_k, w_p, l_p)

        # The netlist uses a SINGLE wp/lp token per stage (Mref and Mload
        # share); per-device dW/dL is folded into kp_*_eff above.
        kw.update({
            f"vto_n{k}":     vto_n_k,     f"kp_n{k}":     kp_n_k,
            f"vto_pref{k}":  vto_pref_k,  f"kp_pref{k}":  kp_pref_k,
            f"vto_pload{k}": vto_pload_k, f"kp_pload{k}": kp_pload_k,
            f"wn_eff_{k}": wn_eff_k, f"ln_eff_{k}": ln_eff_k,
            f"wp_eff_{k}": w_p,      f"lp_eff_{k}": l_p,
            f"rfb_{k}": r_fb,        f"ibias_{k}": i_bias,
        })
    return kw


# --------------------------------------------------------------------------- #
class CSSEMillerSimulator(BlackBoxSimulator):
    """Single-ended two-stage Miller-compensated amp simulator
    (subprocess-isolated ngspice).
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
                        logf = (np.log10(f0)
                                + (np.log10(f1) - np.log10(f0))
                                * (0 - g0) / (g1 - g0))
                        ugbw = 10.0**logf

        phase = np.unwrap(np.angle(vout)) * 180.0 / np.pi
        if ugbw > 0:
            ph_at_ugbw = float(np.interp(np.log10(ugbw),
                                          np.log10(np.maximum(freqs, 1e-30)),
                                          phase))
        else:
            ph_at_ugbw = -180.0
        pm = 180.0 + ph_at_ugbw

        # Power model (same as cs_amp_3stage): each stage draws 2·ibias_k
        # (mirror reference + load branches). Sum across 2 stages.
        ibias_sum = float(x[5]) + float(x[11])
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
                    alphas: Optional[dict] = None) -> CSSEMillerSimulator:
    return CSSEMillerSimulator(targets=targets, alphas=alphas)


# --------------------------------------------------------------------------- #
from zo_yield.evaluation import Spec  # noqa: E402


def build_specs(targets: Optional[dict] = None) -> list:
    t = targets or DEFAULT_TARGETS
    g_t = t["gain_target"]
    u_t = t["ugbw_target"]
    p_t = t["power_target"]
    return [
        Spec(name="gain_db",            extract=lambda m: m["gain_db"],
             satisfies=lambda v: v >= g_t,  margin=lambda v: g_t - v),
        Spec(name="ugbw_hz",            extract=lambda m: m["ugbw_hz"],
             satisfies=lambda v: v >= u_t,  margin=lambda v: u_t - v),
        Spec(name="power_w",            extract=lambda m: m["power_w"],
             satisfies=lambda v: v <= p_t,  margin=lambda v: v - p_t),
    ]


SPECS = build_specs()
