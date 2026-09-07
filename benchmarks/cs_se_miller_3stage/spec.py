"""Single-ended three-stage Miller-compensated amplifier benchmark.

Stacks 3 cs_se_miller per-stage blocks with 2 Miller compensation networks
between consecutive stages. Each stage = self-biased CS amp (cs_amp pattern,
validated). Inter-stage AC coupling.

Design vars (n = 22):
  Per stage k ∈ {1, 2, 3}: w_n_k, l_n_k, w_p_k, l_p_k, r_fb_k, i_bias_k  (6 × 3 = 18)
  Miller comp: c_c1_miller, r_z1_miller, c_c2_miller, r_z2_miller  (4)

Process variations (d = 38):
  Per-stage 4 ξ × 3 transistors (M_k, Mref_k, Mload_k) × 3 stages = 36
  Plus 2 shared globals (dvth_g, dkp_g) = 38.
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


# Specification targets.
DEFAULT_TARGETS = dict(
    gain_target=60.0,        # dB — relaxed from 80 (nominal 78); MC headroom needed
    ugbw_target=0.05e6,      # 0.05 MHz — cushion below nominal 0.1 MHz
    power_target=0.7e-3,     # 0.7 mW (3 stages × cs_se_miller's per-stage power)
    vdd=1.0,                 # match cs_amp_3stage / cs_se_miller
    cin=1e-9,
    cc_ac=1e-9,              # inter-stage AC coupling
    cload=1e-12,             # output load (1 pF)
)


# ---- Design variable layout (n = 22) -------------------------------------- #
DESIGN_VARS = []
for k in (1, 2, 3):
    DESIGN_VARS += [f"w_n_{k}", f"l_n_{k}", f"w_p_{k}", f"l_p_{k}",
                    f"r_fb_{k}", f"i_bias_{k}"]
DESIGN_VARS += ["c_c1_miller", "r_z1_miller", "c_c2_miller", "r_z2_miller"]


# Per-stage X_NOMINAL inherits cs_se_miller's per-stage values.
_PER_STAGE_NOMINAL = np.array([20e-6, 0.18e-6, 40e-6, 0.5e-6, 10e6, 50e-6])
_PER_STAGE_LO = np.array([2e-6, 0.045e-6, 4e-6, 0.045e-6, 1e6, 5e-6])
_PER_STAGE_HI = np.array([200e-6, 2e-6, 400e-6, 4e-6, 100e6, 500e-6])

X_NOMINAL = np.concatenate([
    _PER_STAGE_NOMINAL, _PER_STAGE_NOMINAL, _PER_STAGE_NOMINAL,
    np.array([20.0e-12, 30.0, 20.0e-12, 30.0]),  # 2 Miller-comp pairs
])
X_LO = np.concatenate([
    _PER_STAGE_LO, _PER_STAGE_LO, _PER_STAGE_LO,
    np.array([0.1e-12, 1.0, 0.1e-12, 1.0]),
])
X_HI = np.concatenate([
    _PER_STAGE_HI, _PER_STAGE_HI, _PER_STAGE_HI,
    np.array([50.0e-12, 100.0e3, 50.0e-12, 100.0e3]),
])


# ---- Variation variable layout (d = 38) ----------------------------------- #
# Per-stage 4 ξ × 3 transistors × 3 stages = 36; plus 2 globals.
VARIATION_VARS = []
for k in (1, 2, 3):
    for dev in (f"n_{k}", f"pref_{k}", f"pload_{k}"):
        VARIATION_VARS += [f"dvth_{dev}", f"dkp_{dev}",
                            f"dl_{dev}", f"dw_{dev}"]
VARIATION_VARS += ["dvth_g", "dkp_g"]


def make_sampler(scale: float = 1.0) -> IndependentGaussianSampler:
    """Independent Gaussian sampler over the 38-dim ξ. Per-device sigmas
    match cs_se_miller / cs_amp_3stage Pelgrom-class magnitudes.
    """
    per_device = [0.020 * scale,    # dvth
                  0.05 * scale,     # dkp
                  5e-9 * scale,     # dl
                  0.02e-6 * scale]  # dw
    sigmas = np.array(per_device * 9 + [0.010 * scale, 0.02 * scale])
    return IndependentGaussianSampler(mean=np.zeros_like(sigmas), std=sigmas)


def make_penalties(targets=None, alphas=None):
    t = targets or DEFAULT_TARGETS
    a = dict(DEFAULT_ALPHAS)
    if alphas:
        a.update(alphas)
    return [
        Penalty(name="gain_dB", extract=lambda s: s["gain_db"],
                kind="ge", target=t["gain_target"], weight=1.0,
                alpha=a["alpha_gain"]),
        Penalty(name="ugbw_log",
                extract=lambda s: _math.log10(max(s["ugbw_hz"], 1.0)),
                kind="ge",
                target=_math.log10(max(t["ugbw_target"], 1.0)),
                weight=2.0, alpha=a["alpha_ugbw_log"]),
        Penalty(name="power_mW",
                extract=lambda s: s["power_w"] * 1e3,
                kind="le", target=t["power_target"] * 1e3,
                weight=1.0, alpha=a["alpha_power_mW"]),
    ]


# --------------------------------------------------------------------------- #
def _read_template() -> str:
    with open(os.path.join(HERE, "netlist.cir"), "r") as f:
        return f.read()


def _render_kwargs(x: np.ndarray, xi: np.ndarray, targets: dict) -> dict:
    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if x.shape[0] != 22:
        raise ValueError(f"cs_se_miller_3stage x expected shape (22,), got {x.shape}")
    if xi.shape[0] != 38:
        raise ValueError(f"cs_se_miller_3stage xi expected shape (38,), got {xi.shape}")

    dvth_g = float(xi[36])
    dkp_g = float(xi[37])

    kw = dict(
        vdd=targets["vdd"],
        cin=targets["cin"],
        cc_ac=targets["cc_ac"],
        cload=targets["cload"],
        cc1_miller=float(x[18]),
        rz1_miller=float(x[19]),
        cc2_miller=float(x[20]),
        rz2_miller=float(x[21]),
    )

    for k in (1, 2, 3):
        base_x = (k - 1) * 6
        w_n, l_n, w_p, l_p, r_fb, i_bias = (float(x[base_x + j]) for j in range(6))

        # ξ slice: per-stage 12 entries (3 devices × 4 ξ) starting at (k-1)*12.
        base_xi = (k - 1) * 12
        dvth_n = float(xi[base_xi + 0]);  dkp_n = float(xi[base_xi + 1])
        dl_n   = float(xi[base_xi + 2]);  dw_n  = float(xi[base_xi + 3])
        dvth_pref = float(xi[base_xi + 4]); dkp_pref = float(xi[base_xi + 5])
        dl_pref   = float(xi[base_xi + 6]); dw_pref  = float(xi[base_xi + 7])
        dvth_pload = float(xi[base_xi + 8]); dkp_pload = float(xi[base_xi + 9])
        dl_pload   = float(xi[base_xi + 10]); dw_pload  = float(xi[base_xi + 11])

        vto_n_k     = 0.4 + dvth_n + dvth_g
        vto_pref_k  = -0.4 + dvth_pref + dvth_g
        vto_pload_k = -0.4 + dvth_pload + dvth_g
        kp_n_base    = 300e-6 * (1.0 + dkp_n + dkp_g)
        kp_pref_base = 120e-6 * (1.0 + dkp_pref + dkp_g)
        kp_pload_base = 120e-6 * (1.0 + dkp_pload + dkp_g)

        wn_eff_k = max(w_n + dw_n, 1e-9)
        ln_eff_k = max(l_n + dl_n, 1e-9)
        wpref_eff_k = max(w_p + dw_pref, 1e-9)
        lpref_eff_k = max(l_p + dl_pref, 1e-9)
        wpload_eff_k = max(w_p + dw_pload, 1e-9)
        lpload_eff_k = max(l_p + dl_pload, 1e-9)

        def geom_kp(kp_base, w_eff, l_eff, w_des, l_des):
            return kp_base * (max(w_eff, 1e-9) / max(w_des, 1e-9)) * \
                   (max(l_des, 1e-9) / max(l_eff, 1e-9))

        kp_n_k = geom_kp(kp_n_base, wn_eff_k, ln_eff_k, w_n, l_n)
        kp_pref_k = geom_kp(kp_pref_base, wpref_eff_k, lpref_eff_k, w_p, l_p)
        kp_pload_k = geom_kp(kp_pload_base, wpload_eff_k, lpload_eff_k, w_p, l_p)

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
class CSSEMiller3StageSimulator(BlackBoxSimulator):
    """Single-ended three-stage Miller-compensated amp simulator."""

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

        # Power: per-stage 2·ibias_k summed across 3 stages
        ibias_sum = float(x[5]) + float(x[11]) + float(x[17])
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
                    alphas: Optional[dict] = None) -> CSSEMiller3StageSimulator:
    return CSSEMiller3StageSimulator(targets=targets, alphas=alphas)


# --------------------------------------------------------------------------- #
from zo_yield.evaluation import Spec  # noqa: E402


def build_specs(targets: Optional[dict] = None) -> list:
    t = targets or DEFAULT_TARGETS
    g_t = t["gain_target"]
    u_t = t["ugbw_target"]
    p_t = t["power_target"]
    return [
        Spec(name="gain_db",  extract=lambda m: m["gain_db"],
             satisfies=lambda v: v >= g_t,  margin=lambda v: g_t - v),
        Spec(name="ugbw_hz",  extract=lambda m: m["ugbw_hz"],
             satisfies=lambda v: v >= u_t,  margin=lambda v: u_t - v),
        Spec(name="power_w",  extract=lambda m: m["power_w"],
             satisfies=lambda v: v <= p_t,  margin=lambda v: v - p_t),
    ]


SPECS = build_specs()
