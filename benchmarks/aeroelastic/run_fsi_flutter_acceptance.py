"""Acceptance run for campaign node P21-N04: partitioned FSI, 2-D flutter.

Produces ``artifacts/aeroelastic/acceptance.json`` (every criterion number) and
the figures used by the report.  Runtime is ~1-2 minutes.

Run:  python benchmarks/aeroelastic/run_fsi_flutter_acceptance.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.aeroelastic import (Case, interface_gate,  # noqa: E402
                                       run_coupled, time_domain_run,
                                       theodorsen_c)
from tensorphysics.aeroelastic.flutter import (envelope_growth,  # noqa: E402
                                               flutter_frequency)
from tensorphysics.aeroelastic.interface import panel_grid  # noqa: E402
from tensorphysics.aeroelastic.theodorsen import (  # noqa: E402
    aero_force_harmonic, divergence_speed, flutter_boundary_tracked,
    pk_eigenvalues)
from tensorphysics.aeroelastic.uvlm import GlauertVortexLattice  # noqa: E402
from tensorphysics.aeroelastic.wagner import (A_POLES, BETA_POLES,  # noqa: E402
                                               FIT_ERROR, RationalAero,
                                               c_rational)

OUT = ROOT / "artifacts" / "aeroelastic"
OUT.mkdir(parents=True, exist_ok=True)
RHO = 1.225


def harmonic_compare(aero, k, U, b, a, n_cycles=10, spc=400, h0=1e-4, al0=1e-4):
    """First-harmonic amplitude of the fluid model for prescribed motion."""
    om = k * U / b
    T = 2 * np.pi / om
    dt = T / spc
    aero.reset(U)
    st = aero.solve(h0 * om, 0.0, al0 * om, dt, steady=True)
    aero.commit(st, dt)
    F, ts = [], []
    for n in range(1, int(n_cycles * spc) + 1):
        t = n * dt
        s = aero.solve(h0 * om * np.cos(om * t), al0 * np.sin(om * t),
                       al0 * om * np.cos(om * t), dt)
        aero.commit(s, dt)
        add = aero.added_mass_force(-h0 * om * om * np.sin(om * t),
                                   -al0 * om * om * np.sin(om * t))
        F.append([s.lift + add[0], s.moment + add[1]])
        ts.append(t)
    F = np.array(F)
    ts = np.array(ts)
    m = ts >= 0.5 * ts[-1]
    w = ts[m]

    def amp(y):
        A = np.column_stack([np.ones(w.size), np.cos(om * w), np.sin(om * w)])
        coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        return complex(coef[1], -coef[2])

    return amp(F[:, 0]), amp(F[:, 1])


def main():
    t_start = time.time()
    res = {"node": "P21-N04", "physics": "2-D aeroelastic FSI / flutter",
           "rho": RHO, "gates": {}, "case": {}, "tables": {}}

    # ------------------------------------------------ 1. fluid model checks
    b, U0, a_ea = 0.5, 10.0, -0.5
    uvlm = GlauertVortexLattice(semi_chord=b, n_terms=20, rho=RHO,
                                elastic_axis=a_ea)
    uvlm.reset(U0)
    alpha0 = 0.02
    st = uvlm.solve(0.0, alpha0, 0.0, 1e-3, steady=True)
    lift_th = 2.0 * np.pi * RHO * U0 ** 2 * b * alpha0
    uvlm0 = GlauertVortexLattice(semi_chord=b, n_terms=20, rho=RHO,
                                 elastic_axis=0.0)
    uvlm0.reset(U0)
    st0 = uvlm0.solve(0.0, alpha0, 0.0, 1e-3, steady=True)
    res["tables"]["steady_aero"] = {
        "lift_uvlm": st.lift, "lift_theory": lift_th,
        "lift_rel_err": abs(st.lift - lift_th) / lift_th,
        "moment_ea_quarter_chord": st.moment,
        "moment_mid_chord_over_Lb2": st0.moment / (st.lift * b / 2),
    }

    ks = [0.05, 0.1, 0.2, 0.4, 0.6, 1.0]
    rows = []
    for k in ks:
        ra = RationalAero(semi_chord=b, rho=RHO, elastic_axis=a_ea)
        lh, mh = harmonic_compare(ra, k, U0, b, a_ea)
        th = aero_force_harmonic(k, U0, b, a_ea, RHO, -1e-4j, -1e-4j, 1.0)
        rows.append({"k": k,
                     "lift_rel_err": float(abs(lh - th[0]) / abs(th[0])),
                     "lift_phase_deg": float(np.angle(lh / th[0], deg=True)),
                     "moment_rel_err": float(abs(mh - th[1]) / abs(th[1])),
                     "c_exact": complex(theodorsen_c(k)),
                     "c_rational": complex(c_rational(k))})
    res["tables"]["unsteady_vs_theodorsen"] = rows
    res["tables"]["c_fit_max_error"] = FIT_ERROR

    # UVLM (independent route) at low reduced frequency, pure heave
    uv = GlauertVortexLattice(semi_chord=b, n_terms=20, rho=RHO,
                              elastic_axis=a_ea, core=1.0 * U0 * (2 * np.pi / 50.0
                                                                  / 400))
    lh_uv, _ = harmonic_compare_uvlm(uv, 0.05, U0, b, a_ea)
    th_h = aero_force_harmonic(0.05, U0, b, a_ea, RHO, -1e-4j, 0.0, 1.0)
    res["tables"]["uvlm_heave_k005"] = {
        "lift_rel_err": float(abs(lh_uv - th_h[0]) / abs(th_h[0])),
        "phase_deg": float(np.angle(lh_uv / th_h[0], deg=True))}

    # ------------------------------------------------ 2. interface gates C1/C2
    coeffs = st.coeffs
    gate = interface_gate(uvlm, coeffs, rng=np.random.default_rng(7))
    # C1 is the *operator* conservation gate: the flux the fluid sends is the
    # flux the structure receives (force projection + work conjugacy).  The
    # quadrature error of the 1/sqrt leading-edge singular pressure field on a
    # panel grid is a separate, slowly converging number and is reported below.
    x_mid, dx = panel_grid(b, 200, a_ea)
    dp = uvlm.pressure_jump(x_mid, coeffs)[0]
    from tensorphysics.aeroelastic.interface import (generalised_force,
                                                     work_conjugacy_residual)
    l1, m1 = generalised_force(x_mid, dp, dx, uvlm.x_ea)
    quad = []
    for npan in (100, 200, 400, 800, 1600):
        xm, dxx = panel_grid(b, npan, a_ea)
        dpp = uvlm.pressure_jump(xm, coeffs)[0]
        l2, m2 = generalised_force(xm, dpp, dxx, uvlm.x_ea)
        quad.append({"n_panels": npan,
                     "lift_rel_err": abs(l2 - gate["lift_analytic"])
                     / abs(gate["lift_analytic"]),
                     "work_residual": work_conjugacy_residual(
                         xm, dpp, dxx, uvlm.x_ea, 1e-3, 2e-3)})
    res["tables"]["panel_quadrature_convergence"] = quad
    res["gates"]["C1_momentum_residual"] = float(gate["work_residual"])
    res["gates"]["C1_work_conjugacy_random"] = gate["work_residual_random"]
    res["gates"]["C1_detail"] = {k: float(v) for k, v in gate.items()}
    res["gates"]["C1_panel_field_quadrature_err"] = quad[1]["lift_rel_err"]

    # ------------------------------------------------ 3. section case
    case = Case(name="NACA0012-typical-section", semi_chord=0.5, mass=20.0,
                x_alpha=0.1, r_alpha2=0.25, elastic_axis=0.0,
                omega_h=32.0, omega_alpha=80.0, rho=RHO)
    sec = case.section()
    res["case"]["parameters"] = case.as_dict()
    res["case"]["mass_ratio_mu"] = sec.mass_ratio(RHO)
    res["case"]["added_mass_ratio_lambda"] = sec.added_mass_ratio(RHO)
    res["case"]["natural_frequencies"] = sec.natural_frequencies().tolist()

    # ------------------------------------------------ 4. divergence boundary
    Ud_analytic = divergence_speed(sec, RHO)
    pb = flutter_boundary_tracked(sec, RHO)
    T_ref = 2 * np.pi / (0.5 * (case.omega_h + case.omega_alpha))
    dt = T_ref / 240
    growth = []
    for fac in (0.85, 0.95, 1.05, 1.15):
        U = fac * Ud_analytic
        run = time_domain_run(case, U, dt, 18 * T_ref, scheme="tight")
        growth.append({"U_over_Ud": fac, "U": U, "growth_rate": run.growth,
                       "amplitude": run.amplitude(),
                       "diverged": run.diverged})
    # bisection on the growth rate for the time-domain onset
    lo, hi = 0.7 * Ud_analytic, 1.3 * Ud_analytic

    def g_of(U):
        return time_domain_run(case, U, dt, 14 * T_ref, scheme="tight").growth

    for _ in range(9):
        mid = 0.5 * (lo + hi)
        if g_of(mid) >= 0:
            hi = mid
        else:
            lo = mid
    U_td = 0.5 * (lo + hi)
    res["tables"]["divergence"] = {
        "U_analytic": Ud_analytic, "U_pk_tracked": pb["U_crit"],
        "kind": pb["kind"], "U_time_domain": U_td,
        "err_analytic_vs_pk": abs(pb["U_crit"] - Ud_analytic) / Ud_analytic,
        "err_analytic_vs_time_domain": abs(U_td - Ud_analytic) / Ud_analytic,
        "growth_vs_U": growth,
    }
    res["gates"]["C3_divergence_speed_err"] = \
        abs(U_td - Ud_analytic) / Ud_analytic

    # ------------------------------------------------ 5. flutter branch
    p, kk = pk_eigenvalues(sec, 160.0, RHO)
    top = int(np.argmax(p.real))
    runs_f = []
    for U in (140.0, 150.0, 160.0):
        r = time_domain_run(case, U, dt, 25 * T_ref, scheme="tight")
        runs_f.append({"U": U, "growth_rate": r.growth,
                       "omega_response": flutter_frequency(r),
                       "amplitude": r.amplitude()})
    res["tables"]["flutter_branch"] = {
        "pk_eigen_at_U160": {"p": complex(p[top]), "k": float(kk)},
        "time_domain": runs_f,
        "verdict": "WITHHELD: time-domain growth is negative where the "
                   "frequency-domain p-k predicts positive damping, i.e. the "
                   "flutter branch is not reproduced by the coupled solver",
    }
    res["gates"]["C3_flutter_speed_err"] = None
    res["gates"]["C4_flutter_frequency_err"] = None

    # ------------------------------------------------ 6. energy closure C2
    run_e = time_domain_run(case, 0.95 * Ud_analytic, dt, 12 * T_ref,
                            scheme="tight")
    res["gates"]["C2_energy_residual_tight"] = run_e.energy_residual()
    run_el = time_domain_run(case, 0.95 * Ud_analytic, dt, 12 * T_ref,
                             scheme="loose")
    res["gates"]["C2_energy_residual_loose"] = run_el.energy_residual()

    # ------------------------------------------------ 7. added-mass instability
    light = Case(name="light-section", semi_chord=0.5, mass=0.4,
                 x_alpha=0.1, r_alpha2=0.25, elastic_axis=0.0,
                 omega_h=32.0, omega_alpha=80.0, rho=RHO)
    lsec = light.section()
    res["case"]["light_section"] = {
        "mass": light.mass, "mu": lsec.mass_ratio(RHO),
        "added_mass_ratio": lsec.added_mass_ratio(RHO)}
    U_test = 10.0  # far below the light-section divergence speed
    sweep = []
    for spc in (120, 240, 480, 960, 1920):
        d = T_ref / spc
        rl = time_domain_run(light, U_test, d, 8 * T_ref, scheme="loose")
        rt = time_domain_run(light, U_test, d, 8 * T_ref, scheme="tight")
        sweep.append({"steps_per_cycle": spc, "dt": d,
                      "loose_growth": rl.growth, "loose_diverged": rl.diverged,
                      "loose_amplitude": rl.amplitude(),
                      "tight_growth": rt.growth, "tight_diverged": rt.diverged,
                      "tight_amplitude": rt.amplitude(),
                      "tight_iterations": int(rt.iterations.max()),
                      "tight_converged": bool(rt.converged)})
    res["tables"]["added_mass_sweep"] = sweep
    loose_div = any(s["loose_diverged"] for s in sweep)
    tight_stable = all(not s["tight_diverged"] for s in sweep)
    res["gates"]["C5_loose_dt_limit"] = bool(loose_div)
    res["gates"]["C6_negative_control"] = bool(loose_div and tight_stable)

    # ------------------------------------------------ 8. figures
    make_figures(res, case, ud=Ud_analytic)

    res["runtime_s"] = time.time() - t_start
    (OUT / "acceptance.json").write_text(json.dumps(res, indent=2,
                                                    default=_json_default))
    print(json.dumps({k: v for k, v in res["gates"].items()}, indent=2,
                     default=_json_default))
    print("wrote", OUT / "acceptance.json")


def harmonic_compare_uvlm(aero, k, U, b, aa, n_cycles=10, spc=400,
                          h0=1e-4, al0=0.0):
    om = k * U / b
    dt = 2 * np.pi / om / spc
    aero.reset(U)
    st = aero.solve(h0 * om, 0.0, al0 * om, dt, steady=True)
    aero.commit(st, dt)
    F, ts = [], []
    for n in range(1, int(n_cycles * spc) + 1):
        t = n * dt
        s = aero.solve(h0 * om * np.cos(om * t), al0 * np.sin(om * t),
                       al0 * om * np.cos(om * t), dt)
        aero.commit(s, dt)
        F.append(s.lift)
        ts.append(t)
    F = np.array(F)
    ts = np.array(ts)
    m = ts >= 0.6 * ts[-1]
    w = ts[m]
    c1 = 2.0 / (w[-1] - w[0]) * np.trapezoid(F[m] * np.cos(om * w), w)
    s1 = 2.0 / (w[-1] - w[0]) * np.trapezoid(F[m] * np.sin(om * w), w)
    return complex(c1, -s1), None


def _json_default(o):
    if isinstance(o, complex):
        return {"re": o.real, "im": o.imag}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def make_figures(res, case, ud):
    # fig 1: C(k) and unsteady force agreement
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    kk = np.geomspace(0.01, 5, 300)
    ax[0].semilogx(kk, np.abs(theodorsen_c(kk)), "k-", label="exact $|C(k)|$")
    ax[0].semilogx(kk, np.abs(c_rational(kk)), "r--", label="4-pole fit")
    ax[0].set_xlabel("reduced frequency $k=\\omega b/U$")
    ax[0].set_ylabel("$|C(k)|$")
    ax[0].legend()
    ax[0].set_title("Theodorsen function: exact vs rational fit")
    rows = res["tables"]["unsteady_vs_theodorsen"]
    ks = [r["k"] for r in rows]
    ax[1].loglog(ks, [r["lift_rel_err"] for r in rows], "o-",
                 label="lift amplitude")
    ax[1].loglog(ks, [abs(r["moment_rel_err"]) for r in rows], "s--",
                 label="moment amplitude")
    ax[1].axhline(0.03, color="g", ls=":", label="3 % gate")
    ax[1].set_xlabel("$k$")
    ax[1].set_ylabel("relative error")
    ax[1].legend()
    ax[1].set_title("state-space aero vs exact Theodorsen")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_theodorsen_validation.png", dpi=130)
    plt.close(fig)

    # fig 2: growth rate vs U (time domain) and divergence reference
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    g = res["tables"]["divergence"]["growth_vs_U"]
    ax.plot([x["U"] for x in g], [x["growth_rate"] for x in g], "o-",
            label="coupled FSI (tight)")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(ud, color="r", ls="--",
               label=f"analytic $U_D$={ud:.1f} m/s")
    ax.set_xlabel("free-stream speed $U$ [m/s]")
    ax.set_ylabel("growth rate [1/s]")
    ax.set_title("Static-divergence onset by coupled time marching")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig2_divergence_onset.png", dpi=130)
    plt.close(fig)

    # fig 3: added-mass instability, loose vs tight
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    light = Case(name="light", semi_chord=0.5, mass=0.4, x_alpha=0.1,
                 r_alpha2=0.25, elastic_axis=0.0, omega_h=32.0,
                 omega_alpha=80.0, rho=RHO)
    T_ref = 2 * np.pi / (0.5 * (light.omega_h + light.omega_alpha))
    for scheme, style in (("loose", "r-"), ("tight", "b-")):
        r = time_domain_run(light, 10.0, T_ref / 480, 6 * T_ref,
                            scheme=scheme, max_record=2500)
        ax.semilogy(r.t, np.abs(r.q[:, 1]), style, label=scheme)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("$|\\alpha|$ [rad]")
    ax.set_title("Added-mass instability: explicit vs implicit coupling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig3_loose_vs_tight.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()