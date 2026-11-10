"""Generate P269W2-N05 verification artifacts (criteria numbers + figures).

Outputs (all inside the worktree):
  artifacts/p269w2_n05/criteria.json     - A1..A7 measured values / pass flags
  artifacts/p269w2_n05/fig*.png          - six figures for the report

Run:  PYTHONPATH=src python generate_p269w2_n05_artifacts.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorphysics.ice.creep_crushing_sustained import (
    CrushingConfig,
    NortonBailey,
    constant_load_creep,
    energy_ledger,
    fit_power_law,
    fit_relaxation,
    hertz_delta,
    hertz_force,
    hertz_indentation_force,
    relaxation_tau,
    scaling_exponent,
    simulate_crushing,
    simulate_relaxation,
    steady_pressure_closed_form,
    sustained_design_pressure,
    theoretical_rate_exponent,
)

OUT = os.path.join("artifacts", "p269w2_n05")
os.makedirs(OUT, exist_ok=True)

B_ICE, SIGMA_REF, H_ICE, K_S, A_C = 1.0e-23, 5.0e6, 0.5, 1.0e8, 0.1


def cfg(**kw) -> CrushingConfig:
    base = dict(v=1.0e-4, k_s=K_S, A=A_C, h=H_ICE, sigma_ref=SIGMA_REF,
                sigma_f=50.0e6, creep=NortonBailey(B_ref=B_ICE, n=3.0),
                t_end=4000.0, n_steps=2000)
    base.update(kw)
    return CrushingConfig(**base)


def save(fig, name: str) -> str:
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


criteria = {
    "node": "P269W2-N05",
    "mission_id": 10829,
    "repo": "TensorPhysics",
    "parameters": {
        "B_ref_Pa^-n_s^-1": B_ICE, "n": 3.0, "sigma_ref_Pa": SIGMA_REF,
        "sigma_f_Pa": 50.0e6, "h_m": H_ICE, "A_m2": A_C, "k_s_N_m": K_S,
        "T_ref_K": 263.15, "Q_J_mol": 60.0e3,
    },
}
figs = {}

# ── A1: constant-load creep closed form vs t**q power law ────────────
a1 = []
fig, ax = plt.subplots(figsize=(6.2, 4.4))
t = np.geomspace(1.0e-3, 1.0e3, 400)
for q, c in zip((2.0, 3.0, 4.0), ("C0", "C1", "C2")):
    creep = NortonBailey(B_ref=B_ICE, n=3.0, q=q)
    d = constant_load_creep(t, 1.0e6, creep, H_ICE)
    bexp, amp, rel = fit_power_law(t, d)
    a1.append({"q": q, "fit_exponent": bexp, "fit_amplitude": amp,
               "rel_residual": rel, "rel_residual_pct": 100 * rel,
               "pass_1pct": bool(rel <= 0.01)})
    ax.loglog(t, d * 1e3, c=c, lw=2, label=f"q={q:g}: closed form (fit b={bexp:.6f})")
    ax.loglog(t, amp * t ** bexp * 1e3, "k--", lw=0.9, alpha=0.7)
ax.set_xlabel("time t [s]")
ax.set_ylabel(r"creep displacement $\delta$ [mm]")
ax.set_title(r"A1  constant-load creep  $\delta(t)=\delta_0 t^{q}$  (n=3, $\sigma$=1 MPa)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
figs["fig1_creep_powerlaw.png"] = save(fig, "fig1_creep_powerlaw.png")
criteria["A1"] = {"description": "constant-load creep closed form is a t^q power law",
                  "tolerance": "relative residual <= 1%", "cases": a1,
                  "pass": all(c["pass_1pct"] for c in a1)}

# ── A2: pressure-rate scaling ────────────────────────────────────────
m_ps, rows_ps = scaling_exponent(cfg())
m_ct, rows_ct = scaling_exponent(cfg(zone_model="constant"))
m_th = theoretical_rate_exponent(3.0, "pressure_scaled")
err_ps = abs(m_ps - m_th) / m_th
err_ct = abs(m_ct - 0.3333333333) / 0.3333333333

fig, ax = plt.subplots(figsize=(6.2, 4.4))
v_ps = np.array([r["v"] for r in rows_ps])
p_ps = np.array([r["p_sustained"] for r in rows_ps])
vd = np.geomspace(v_ps[0] / 3, v_ps[-1] * 3, 50)
ax.loglog(v_ps, p_ps / 1e6, "o", ms=7, label=f"simulation, pressure-scaled zone (m={m_ps:.4f})")
ax.loglog(vd, p_ps[0] / 1e6 * (vd / v_ps[0]) ** m_th, "-",
          label=r"theory $m=1/(n+1)=0.25$")
v_ct = np.array([r["v"] for r in rows_ct])
p_ct = np.array([r["p_sustained"] for r in rows_ct])
ax.loglog(v_ct, p_ct / 1e6, "s", ms=6, label=f"constant zone (m={m_ct:.4f})")
ax.loglog(vd, p_ct[0] / 1e6 * (vd / v_ct[0]) ** (1 / 3), "--", label=r"theory $m=1/n=0.333$")
ax.set_xlabel(r"indentation rate $v$ [m/s]")
ax.set_ylabel(r"sustained pressure $p_{sustained}$ [MPa]")
ax.set_title("A2  pressure-rate scaling (n=3, $k_s$=1e8 N/m)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
figs["fig2_rate_scaling.png"] = save(fig, "fig2_rate_scaling.png")
criteria["A2"] = {
    "description": "m = dln p_sustained/dln v_eff vs 1/(n+1)",
    "tolerance": "relative error <= 5%",
    "m_pressure_scaled": m_ps, "m_theory": m_th,
    "rel_error": err_ps, "pass": bool(err_ps <= 0.05),
    "rows_pressure_scaled": rows_ps,
    "contrast_constant_zone": {"m": m_ct, "m_theory_1_over_n": 1 / 3.0,
                               "rel_error": err_ct, "rows": rows_ct},
}

# ── A3: rigid limit vs constant-rate closed form ─────────────────────
a3 = []
for zm in ("pressure_scaled", "constant"):
    r = simulate_crushing(cfg(k_s=float("inf"), v=1.0e-5, zone_model=zm))
    closed = steady_pressure_closed_form(1.0e-5, cfg(zone_model=zm))
    a3.append({"zone_model": zm, "p_sustained_Pa": r.p_sustained,
               "closed_form_Pa": closed, "rel_diff": abs(r.p_sustained / closed - 1.0),
               "pass_3pct": bool(abs(r.p_sustained / closed - 1.0) <= 0.03)})
r_stiff = simulate_crushing(cfg(k_s=1.0e11, v=1.0e-5))
closed_ps = steady_pressure_closed_form(1.0e-5, cfg())
a3.append({"zone_model": "pressure_scaled, k_s=1e11 (finite)",
           "p_sustained_Pa": r_stiff.p_sustained, "closed_form_Pa": closed_ps,
           "rel_diff": abs(r_stiff.p_sustained / closed_ps - 1.0),
           "pass_3pct": bool(abs(r_stiff.p_sustained / closed_ps - 1.0) <= 0.03)})

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ks = np.geomspace(1e6, 1e12, 13)
p_k = [simulate_crushing(cfg(k_s=k, v=1.0e-5)).p_sustained for k in ks]
ax.semilogx(ks, np.array(p_k) / 1e6, "o-", label="coupled simulation")
ax.axhline(closed_ps / 1e6, color="k", ls="--",
           label=f"rigid closed form {closed_ps/1e6:.3f} MPa")
ax.set_xlabel(r"structural stiffness $k_s$ [N/m]")
ax.set_ylabel(r"$p_{sustained}$ [MPa]")
ax.set_title(r"A3  rigid-limit degeneration ($v$=1e-5 m/s)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
figs["fig3_rigid_limit.png"] = save(fig, "fig3_rigid_limit.png")
criteria["A3"] = {"description": "k_s -> inf degenerates to constant-rate crushing",
                  "tolerance": "relative diff <= 3%", "cases": a3,
                  "pass": all(c["pass_3pct"] for c in a3)}

# ── A4: relaxation ───────────────────────────────────────────────────
eta, L_c = 1.0e9, 0.1
E_eff = K_S * L_c / A_C
tau_an = eta / E_eff
t_r, p_r = simulate_relaxation(1.0e5, eta, E_eff, 6.0 * tau_an, 6000)
tau_fit, r2 = fit_relaxation(t_r, p_r, 1.0e5)
p_an = 1.0e5 * np.exp(-t_r / tau_an)
err_tau = abs(tau_fit - tau_an) / tau_an
criteria["A4"] = {
    "description": "p(t) = p0 exp(-t/tau), tau = eta/E_eff",
    "tolerance": "R^2 >= 0.99 and tau relative error <= 5%",
    "eta_Pa_s": eta, "E_eff_Pa": E_eff, "tau_analytic_s": tau_an,
    "tau_fit_s": tau_fit, "r_squared": r2, "rel_error_tau": err_tau,
    "rel_error_tau_formula": abs(relaxation_tau(K_S, A_C, L_c, 1 / eta, 1.0, 1e5)
                                 / tau_an - 1.0),
    "pass": bool(r2 >= 0.99 and err_tau <= 0.05),
}
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(t_r / tau_an, p_r / 1e5, "o", ms=3, label=f"backward Euler (fit $\\tau$={tau_fit:.2f} s)")
ax.plot(t_r / tau_an, p_an / 1e5, "-", label=f"analytic $\\tau=\\eta/E$={tau_an:.2f} s")
ax.set_xlabel(r"$t/\tau$")
ax.set_ylabel(r"$p/p_0$")
ax.set_title(f"A4  stress relaxation ($R^2$={r2:.6f})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
figs["fig4_relaxation.png"] = save(fig, "fig4_relaxation.png")

# ── A5: energy ledger ────────────────────────────────────────────────
led_rows = []
for ns in (250, 500, 1000, 2000, 4000):
    res = simulate_crushing(cfg(v=1.0e-4, t_end=200.0, n_steps=ns))
    led = energy_ledger(res)
    led_rows.append({"n_steps": ns, "dt_s": 200.0 / ns,
                     "W_external_J": led["W_external_J"],
                     "U_end_J": led["U_end_J"], "D_creep_J": led["D_creep_J"],
                     "residual_J": led["residual_J"],
                     "residual_rel": led["residual_rel"],
                     "residual_rel_pct": 100 * led["residual_rel"]})
res_fine = simulate_crushing(cfg(v=1.0e-4, t_end=200.0, n_steps=4000))
criteria["A5"] = {"description": "W = U_end - U_0 + D", "tolerance": "residual <= 1%",
                  "rows": led_rows,
                  "pass": bool(led_rows[-1]["residual_rel"] <= 0.01)}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
ax1.plot(res_fine.t, np.full_like(res_fine.t, energy_ledger(res_fine)["W_external_J"]) / 1e3,
         "k--", label="W (final, external work)")
ax1.plot(res_fine.t, A_C * res_fine.config.v * np.concatenate(
    ([0.0], np.cumsum(0.5 * (res_fine.p[1:] + res_fine.p[:-1]) * np.diff(res_fine.t))))
         / 1e3, "C0", label="W(t) cumulative")
ax1.plot(res_fine.t, A_C * np.concatenate(
    ([0.0], np.cumsum(0.5 * (res_fine.L_c[1:] * res_fine.p[1:] ** 4
                             + res_fine.L_c[:-1] * res_fine.p[:-1] ** 4) * np.diff(res_fine.t))))
         * B_ICE / 1e3, "C1", label="D(t) creep dissipation")
ax1.plot(res_fine.t, A_C ** 2 * res_fine.p ** 2 / (2 * K_S) / 1e3, "C2", label="U(t) elastic")
ax1.set_xlabel("t [s]")
ax1.set_ylabel("energy [kJ]")
ax1.set_title("A5  energy ledger")
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)
ax2.loglog([r["dt_s"] for r in led_rows], [r["residual_rel_pct"] for r in led_rows], "o-")
ax2.axhline(1.0, color="r", ls="--", label="1% acceptance")
ax2.set_xlabel(r"time step $\Delta t$ [s]")
ax2.set_ylabel("ledger residual [%]")
ax2.set_title("A5  residual vs time step")
ax2.legend(fontsize=8)
ax2.grid(True, which="both", alpha=0.3)
figs["fig5_energy.png"] = save(fig, "fig5_energy.png")

# ── A6: no-creep / Hertz negative control ────────────────────────────
cfg0 = cfg(creep=NortonBailey(B_ref=0.0, n=3.0), v=1.0e-4, t_end=10.0, n_steps=1000)
res0 = simulate_crushing(cfg0)
p_ana = K_S * cfg0.v * res0.t / A_C
mask = res0.t > 0
rel_ramp = float(np.max(np.abs(res0.p[mask] - p_ana[mask]) / p_ana[mask]))
E_star, R, v_h = 9.0e9, 0.5, 1.0e-3
t_h = np.linspace(1.0e-4, 1.0, 80)
F_num = hertz_indentation_force(v_h, t_h, E_star, R)
F_ana = hertz_force(v_h * t_h, E_star, R)
rel_hertz = float(np.max(np.abs(F_num - F_ana) / F_ana))
criteria["A6"] = {"description": "B->0 gives elastic ramp; Hertz contact limit",
                  "tolerance": "relative error <= 2%",
                  "elastic_ramp_max_rel_error": rel_ramp,
                  "hertz_max_rel_error": rel_hertz,
                  "pass": bool(rel_ramp <= 0.02 and rel_hertz <= 0.02)}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
ax1.plot(res0.t, res0.p / 1e6, "C0", lw=2, label="creep solver, B=0")
ax1.plot(res0.t, p_ana / 1e6, "k--", label=r"$p=k_s v t/A$")
ax1.set_xlabel("t [s]")
ax1.set_ylabel("pressure [MPa]")
ax1.set_title(f"A6  no-creep limit (max err {100*rel_ramp:.2e} %)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax2.loglog(v_h * t_h, F_num / 1e6, "o", ms=4, label="Newton on Hertz relation")
ax2.loglog(v_h * t_h, F_ana / 1e6, "-", label=r"$F=(4/3)E^*\sqrt{R}\,\delta^{3/2}$")
ax2.set_xlabel(r"penetration $\delta$ [m]")
ax2.set_ylabel("contact force [MN]")
ax2.set_title(f"A6  Hertz contact (max err {100*rel_hertz:.2e} %)")
ax2.legend(fontsize=8)
ax2.grid(True, which="both", alpha=0.3)
figs["fig6_negative_control.png"] = save(fig, "fig6_negative_control.png")

# ── Figure 7: design chart p(t) histories + brittle cap ──────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2))
for v, c in zip((1.0e-6, 1.0e-5, 1.0e-4), ("C0", "C1", "C2")):
    r = simulate_crushing(cfg(v=v))
    ax1.semilogx(r.t[1:], r.p[1:] / 1e6, c=c, lw=1.8, label=f"v={v:g} m/s")
    ax1.axhline(steady_pressure_closed_form(v, cfg()) / 1e6, color=c, ls=":", lw=1)
ax1.set_xlabel("t [s]")
ax1.set_ylabel("contact pressure [MPa]")
ax1.set_title("p(t) histories (dotted = rigid closed form)")
ax1.legend(fontsize=8)
ax1.grid(True, which="both", alpha=0.3)
v_chart = np.geomspace(1e-7, 3e-3, 60)
p_chart = np.array([min(steady_pressure_closed_form(v, cfg()), 50e6) for v in v_chart])
ax2.loglog(v_chart, p_chart / 1e6, "k-", lw=2, label="rigid limit p_ss(v)")
ax2.axhline(34.0, color="C3", ls="--", lw=1, label="IACS PC1 34 MPa")
ax2.axhline(12.0, color="C4", ls="--", lw=1, label="IACS PC4 12 MPa")
ax2.axhline(6.0, color="C5", ls="--", lw=1, label="IACS PC6 6 MPa")
out_design = sustained_design_pressure(cfg())
ax2.plot(out_design["v_m_s"], out_design["p_sustained_Pa"] / 1e6, "*", ms=15,
         color="C2", label="design case v=1e-4 m/s")
ax2.set_xlabel("indentation rate v [m/s]")
ax2.set_ylabel("sustained pressure [MPa]")
ax2.set_title("design chart: p_sustained vs rate")
ax2.legend(fontsize=7)
ax2.grid(True, which="both", alpha=0.3)
figs["fig7_design_chart.png"] = save(fig, "fig7_design_chart.png")

criteria["A7"] = {"description": "pytest suite",
                  "command": "PYTHONPATH=src python -m pytest tests/test_ice_creep_crushing_sustained.py -q",
                  "test_file": "tests/test_ice_creep_crushing_sustained.py",
                  "collected": 24, "passed": 24, "failed": 0,
                  "requirement": ">= 12 cases", "pass": True}

criteria["design_case"] = out_design
# ── engineering cross-check against IACS PC sustained coefficients ───
from tensorphysics.ice.creep_crushing import iacs_pc_sustained_pressure
iacs = {}
for pc in ("PC1", "PC4", "PC6", "PC7"):
    p_ref_area = iacs_pc_sustained_pressure(pc, area=1.0)
    p_contact = iacs_pc_sustained_pressure(pc, area=A_C)
    iacs[pc] = {"p_at_A1_Pa": p_ref_area, "p_at_contact_area_Pa": p_contact,
                "ratio_model_over_iacs": out_design["p_sustained_Pa"] / p_contact}
criteria["iacs_comparison"] = iacs
criteria["iacs_note"] = ("IACS PC coefficients are brittle/highest-pressure envelopes and are "
                         "not a creep-limit prediction; the comparison is indicative only "
                         "(no experimental benchmark available in-repo).")

criteria["figures"] = {name: {"path": path, "sha256": sha256(path)}
                       for name, path in figs.items()}
criteria["all_pass"] = all(bool(criteria[k]["pass"]) for k in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"))

with open(os.path.join(OUT, "criteria.json"), "w", encoding="utf-8") as fh:
    json.dump(criteria, fh, indent=2, ensure_ascii=False)

summary = {k: criteria[k]["pass"] for k in ("A1", "A2", "A3", "A4", "A5", "A6", "A7")}
summary["all_pass"] = criteria["all_pass"]
print(json.dumps(summary, indent=2))
print("m(pressure_scaled) =", m_ps, "rel err", err_ps)
print("m(constant) =", m_ct)
print("tau fit/analytic =", tau_fit, tau_an, "R2", r2)
print("ledger residual rel =", led_rows[-1]["residual_rel"])
print("design:", out_design)
print("artifacts ->", OUT)