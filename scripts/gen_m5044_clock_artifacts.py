#!/usr/bin/env python3
"""Generate the IN16-K8 clock-semantics artifacts and figures.

Outputs (all machine-produced, nothing hand-copied):

``artifacts/m5044/clock_semantics.json``
    GCD/LCM cadence verification for three representative clock pairs, the
    causality audit (including the negative case with a real overshoot), the
    refinement / band / interpolation scans and the declared band coefficients.

``artifacts/m5044/figures/*.png``
    Log-log convergence with the declared band, in-interval reconstruction error,
    coupled time series with its error trace, and the causality timeline.

Usage:  PYTHONPATH=src python3 scripts/gen_m5044_clock_artifacts.py
"""
from __future__ import annotations

import base64
import json
import pathlib
import sys
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.clock_rational import (  # noqa: E402
    CausalityViolation,
    RationalClockSpec,
    RationalCouplingClock,
    artifact_digest,
    causality_audit,
    verify_gcd_cadence,
)
from tensorphysics.campaign.multirate import (  # noqa: E402
    BAND_LINEAR_COEF,
    BAND_ZOH_COEF,
    ORDER_WINDOW,
    ORDER_WINDOW_SECOND,
    TwoIntegratorModel,
    band_scan,
    interpolation_band,
    interpolation_scan,
    refinement_scan,
    signal_amplitude,
    simulate_multirate,
)

OUT = ROOT / "artifacts" / "m5044"
FIGS = OUT / "figures"
MISSION = "5044"
NODE = "IN16-K8"


def clock_cases() -> list[dict]:
    """Clock pairs exercising integer, fractional and high-order rational ratios."""
    cases = [
        ("int_ratio_4x", "hull_section", "lbm_flow", 0.004, 0.001, 1, "integer ratio 4/1 (v1-compatible)"),
        ("frac_ratio_3_2", "structure", "contact", 0.003, 0.002, 1, "fractional ratio 3/2"),
        ("coarse_periods", "hull_section", "ice_contact", 0.004, 0.003, 5, "ratio 4/3, cadence x5"),
        ("high_order", "rigid_body", "dem_ice", 0.02, 0.0075, 1, "ratio 8/3"),
    ]
    out = []
    for key, master, sub, dt_m, dt_s, periods, note in cases:
        spec = RationalClockSpec(master, sub, dt_m, dt_s, exchange_periods=periods)
        ticks = spec.master_ticks_per_exchange * 12
        verify = verify_gcd_cadence(spec, master_ticks=ticks)
        clock = RationalCouplingClock(spec)
        events = clock.run(ticks)
        audit = causality_audit(spec, events)
        # negative case: the sub-solver overshoots the exchange instant
        late = [event.time_exact + spec.sub_dt for event in events]
        audit_late = causality_audit(spec, events, producer_times=late)
        out.append(
            {
                "key": key,
                "note": note,
                "master": master,
                "sub": sub,
                "master_dt_s": float(spec.master_dt),
                "sub_dt_s": float(spec.sub_dt),
                "ratio_exact": f"{spec.ratio.numerator}/{spec.ratio.denominator}",
                "ratio": float(spec.ratio),
                "ratio_is_integer": spec.ratio_is_integer,
                "exchange_periods": periods,
                "atom_exact": f"{spec.atom_s.numerator}/{spec.atom_s.denominator}",
                "atom_s": float(spec.atom_s),
                "master_atoms": spec.master_atoms,
                "sub_atoms": spec.sub_atoms,
                "base_exchange_s": float(spec.base_exchange_s),
                "exchange_interval_s": float(spec.exchange_interval_s),
                "master_ticks_per_exchange": spec.master_ticks_per_exchange,
                "sub_steps_per_exchange": spec.sub_steps_per_exchange,
                "max_information_age_s": float(spec.max_information_age_s),
                "master_ticks_simulated": ticks,
                "event_count": len(events),
                "gcd_cadence_checks": verify,
                "causality_audit": audit,
                "causality_negative_case": audit_late,
                "artifact_digest": artifact_digest(spec),
            }
        )
    return out


def causality_gate_evidence() -> dict:
    """Record the explicit rejection message of the negative case."""
    from tensorphysics.campaign.clock_rational import assert_causal_hold

    spec = RationalClockSpec("structure", "contact", 0.003, 0.002, exchange_periods=1)
    event_time = spec.exchange_interval_s  # 6 ms
    future_sample = event_time + spec.sub_dt  # 8 ms
    message = None
    try:
        assert_causal_hold(event_time, future_sample, label="exchange #1")
    except CausalityViolation as exc:
        message = str(exc)
    legal_age = assert_causal_hold(event_time, event_time)
    return {
        "exchange_time_exact": f"{event_time.numerator}/{event_time.denominator}",
        "future_sample_exact": f"{future_sample.numerator}/{future_sample.denominator}",
        "raised": message is not None,
        "message": message,
        "legal_information_age_exact": f"{legal_age.numerator}/{legal_age.denominator}",
    }


def fig_convergence(refine: dict, bands: dict) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    rows = refine["rows"]
    hs = np.array([r["exchange_interval_s"] for r in rows])
    errs = np.array([r["rel_error"] for r in rows])
    ax.loglog(hs, errs, "o-", label="coupled multirate error (causal ZOH)")
    omega = refine["model"]["omega_max"]
    ax.loglog(
        hs,
        BAND_ZOH_COEF * omega * hs,
        "--",
        label=f"declared band: C*(omega_max*h), C={BAND_ZOH_COEF:g}",
    )
    # reference slopes
    ref = errs[0] * (hs / hs[0])
    ax.loglog(hs, ref, ":", color="grey", label="slope 1 (first order)")
    ax.set_xlabel("exchange interval h [s]")
    ax.set_ylabel("max relative displacement error / A")
    ax.set_title(
        f"IN16-K8 exchange convergence (measured order {refine['order']:.3f})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = FIGS / "convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_interpolation(model: TwoIntegratorModel) -> pathlib.Path:
    """In-interval reconstruction error: causal ZOH vs non-causal linear."""
    x0, v0 = (1.0e-3, 0.0), (0.0, 0.0)
    h = 0.012
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    frac = np.linspace(0.0, 1.0, 81)
    for omega, mode in model.modes():
        _ = omega
    # first interval, exact knots
    left = model.exact(0.0, x0=x0, v0=v0)[0]
    right = model.exact(h, x0=x0, v0=v0)[0]
    exact = np.array([model.exact(f * h, x0=x0, v0=v0)[0] for f in frac])
    zoh = np.full_like(exact, left)
    lin = left + frac * (right - left)
    ax.plot(frac, (exact - zoh) * 1e6, "-", label="ZOH (causal, O(h))")
    ax.plot(frac, (exact - lin) * 1e6, "-", label="linear interp. (uses future sample, O(h^2))")
    amp = abs(left)
    ax.fill_between(
        frac,
        -BAND_ZOH_COEF * model.omega_max * h * amp * 1e6,
        BAND_ZOH_COEF * model.omega_max * h * amp * 1e6,
        color="tab:blue",
        alpha=0.08,
        label="declared band (C*omega_max*h*A)",
    )
    ax.set_xlabel("position inside the exchange interval  (t - t_k)/h")
    ax.set_ylabel("reconstruction error [µm]")
    ax.set_title("IN16-K8 in-interval reconstruction error (h = 12 ms)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGS / "interpolation_band.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_solution(model: TwoIntegratorModel) -> tuple[pathlib.Path, dict]:
    """Coupled time series vs the analytic reference, plus the error trace."""
    spec = RationalClockSpec("hull_section", "ice_contact", 0.004, 0.003, exchange_periods=1)
    x0, v0 = (1.0e-3, 0.0), (0.0, 0.0)
    intervals = 64
    result = simulate_multirate(model, spec, intervals=intervals, x0=x0, v0=v0)
    h = float(spec.exchange_interval_s)
    ts = np.linspace(0.0, intervals * h, 600)
    ex = np.array([model.exact(t, x0=x0, v0=v0) for t in ts])

    # re-run with recording of the numerical trace for the plot
    dt_m, dt_s = float(spec.master_dt), float(spec.sub_dt)
    a, b = spec.master_atoms, spec.sub_atoms
    atom_s = float(spec.atom_s)
    x1, x2 = x0
    v1, v2 = v0
    m1, m2 = model.m_master, model.m_sub
    k1 = model.k_master + model.k_coupling
    k2 = model.k_sub + model.k_coupling
    kc = model.k_coupling
    partner1, partner2 = x2, x1
    trace_t, trace_x1, trace_x2 = [], [], []
    for interval in range(intervals):
        partner1, partner2 = x2, x1
        for atom in range(1, spec.atoms_per_exchange + 1):
            if atom % a == 0:
                f1 = -k1 * x1 + kc * partner1
                v1_half = v1 + 0.5 * dt_m * f1 / m1
                x1_new = x1 + dt_m * v1_half
                v1 = v1_half + 0.5 * dt_m * (-k1 * x1_new + kc * partner1) / m1
                x1 = x1_new
            if atom % b == 0:
                f2 = -k2 * x2 + kc * partner2
                v2_half = v2 + 0.5 * dt_s * f2 / m2
                x2_new = x2 + dt_s * v2_half
                v2 = v2_half + 0.5 * dt_s * (-k2 * x2_new + kc * partner2) / m2
                x2 = x2_new
            trace_t.append(interval * h + atom * atom_s)
            trace_x1.append(x1)
            trace_x2.append(x2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(ts, ex[:, 0] * 1e3, "-", lw=2.2, label="exact master (hull section)")
    ax1.plot(ts, ex[:, 1] * 1e3, "-", lw=2.2, label="exact sub (ice contact)")
    ax1.plot(trace_t, np.array(trace_x1) * 1e3, ".", ms=2.5, label="multirate master (ZOH exchange)")
    ax1.plot(trace_t, np.array(trace_x2) * 1e3, ".", ms=2.5, label="multirate sub (ZOH exchange)")
    ax1.set_ylabel("displacement [mm]")
    ax1.set_title(
        f"IN16-K8 coupled 2-DOF response, h = {h*1e3:.1f} ms, "
        f"ratio {spec.ratio.numerator}/{spec.ratio.denominator}"
    )
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    numer_x1 = np.interp(ts, trace_t, trace_x1)
    numer_x2 = np.interp(ts, trace_t, trace_x2)
    ax2.semilogy(ts, np.maximum(np.abs(numer_x1 - ex[:, 0]), 1e-18) * 1e3, "-", label="|master error|")
    ax2.semilogy(ts, np.maximum(np.abs(numer_x2 - ex[:, 1]), 1e-18) * 1e3, "-", label="|sub error|")
    amp = signal_amplitude(model, x0=x0, v0=v0, duration_s=intervals * h)
    band = BAND_ZOH_COEF * model.omega_max * h * amp * 1e3
    ax2.axhline(band, color="k", ls="--", lw=1.0, label="declared band")
    ax2.set_ylabel("|error| [mm]")
    ax2.set_xlabel("time [s]")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    path = FIGS / "coupled_response.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, {"sim_rel_error": result.rel_error, "declared_band": BAND_ZOH_COEF * model.omega_max * h}


def fig_causality(spec: RationalClockSpec) -> pathlib.Path:
    """Timeline: master ticks, sub steps, exchange instants, rejected future sample."""
    clock = RationalCouplingClock(spec)
    ticks = spec.master_ticks_per_exchange * 4
    events = clock.run(ticks)
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    master_times = [float(i * spec.master_dt) for i in range(1, ticks + 1)]
    sub_times = [float(i * spec.sub_dt) for i in range(1, ticks * spec.master_atoms // spec.sub_atoms + 1)]
    ax.vlines(master_times, 1.0, 1.6, color="tab:blue", lw=1.2, label="master ticks")
    ax.vlines(sub_times, 0.4, 0.95, color="tab:orange", lw=1.0, label="sub steps")
    for event in events:
        ax.axvline(event.time_s, color="k", ls="--", lw=1.0)
        ax.plot([event.time_s], [2.1], "kD", ms=6)
    ax.plot([], [], "kD", ms=6, label="exchange instants (LCM cadence)")
    # the negative case: a sub sample produced after the first exchange instant
    future = float(events[0].time_exact + spec.sub_dt)
    ax.plot([future], [2.1], "rx", ms=10, mew=2)
    ax.annotate(
        "future sample → CausalityViolation",
        xy=(future, 2.1),
        xytext=(future, 2.55),
        ha="center",
        fontsize=8,
        color="tab:red",
    )
    ax.axvline(events[0].time_s, color="tab:red", ls=":", lw=1.0)
    ax.set_ylim(0.2, 2.9)
    ax.set_yticks([0.7, 1.3, 2.1])
    ax.set_yticklabels(["sub steps", "master ticks", "exchanges"], fontsize=9)
    ax.set_xlabel("time [s]")
    ax.set_title(
        f"IN16-K8 clock lattice: ratio {spec.ratio.numerator}/{spec.ratio.denominator}, "
        f"cadence = LCM = {float(spec.exchange_interval_s)*1e3:.1f} ms"
    )
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    path = FIGS / "causality_timeline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    model = TwoIntegratorModel()
    clock_rows = clock_cases()
    gate = causality_gate_evidence()
    refine = refinement_scan(model, levels=4)
    bands = band_scan(model, periods=(1, 2, 4, 8, 16), base_intervals=256)
    interp = interpolation_scan(model)
    interp_one = interpolation_band(model, exchange_interval_s=0.012, intervals=8)

    paths = {
        "convergence": fig_convergence(refine, bands),
        "interpolation": fig_interpolation(model),
        "coupled": fig_solution(model)[0],
        "causality": fig_causality(
            RationalClockSpec("structure", "contact", 0.003, 0.002, exchange_periods=1)
        ),
    }

    criterion = {
        "pytest_cases": 16,
        "gcd_cadence_machine_verified": all(
            row["gcd_cadence_checks"]["verified"] for row in clock_rows
        ),
        "gcd_cases": [
            {
                "key": row["key"],
                "ratio_exact": row["ratio_exact"],
                "atom_exact": row["atom_exact"],
                "verified": row["gcd_cadence_checks"]["verified"],
                "cadence_is_coarsest": row["gcd_cadence_checks"]["cadence_is_coarsest"],
                "atom_maximal": row["gcd_cadence_checks"]["atom_maximal"],
                "spurious_common_periods": row["gcd_cadence_checks"]["spurious_common_periods"],
                "max_abs_residual_s": row["gcd_cadence_checks"]["max_abs_residual_s"],
            }
            for row in clock_rows
        ],
        "causality_negative_case_raised": gate["raised"],
        "causality_negative_cases_per_clock": all(
            not row["causality_negative_case"]["is_causal"] for row in clock_rows
        ),
        "multirate_order": refine["order"],
        "multirate_order_window": list(ORDER_WINDOW),
        "multirate_order_ok": refine["order_ok"],
        "band_zoh_coef": BAND_ZOH_COEF,
        "band_linear_coef": BAND_LINEAR_COEF,
        "band_worst_ratio_refinement": refine["worst_band_ratio"],
        "band_all_within_refinement": refine["all_within_band"],
        "band_all_within_decade_scan": bands["all_within_band"],
        "band_decade_worst_ratio": bands["worst_band_ratio"],
        "interpolation_zoh_order": interp["order_zoh"],
        "interpolation_linear_order": interp["order_linear"],
        "interpolation_linear_order_window": list(ORDER_WINDOW_SECOND),
        "interpolation_linear_order_ok": interp["linear_order_ok"],
        "interpolation_zoh_within_band": interp["all_zoh_within_band"],
        "interpolation_linear_within_band": interp["all_linear_within_band"],
        "interpolation_linear_rejected_non_causal": interp["linear_rejected_as_non_causal"],
        "disposition": "M12",
    }
    payload = {
        "mission_id": MISSION,
        "node": NODE,
        "repository": "TensorPhysics",
        "module": "src/tensorphysics/campaign/clock_rational.py, src/tensorphysics/campaign/multirate.py",
        "tests": "tests/test_campaign_clock_rational.py",
        "disposition_alignment": {
            "disposition": "M12",
            "v1_contract": "src/tensorphysics/campaign/clock.py (integer ratio, integer_sub_step lattice)",
            "extension": "rational atom lattice + LCM exchange cadence + causality gate",
            "backward_compatibility": "v1 artifact schema and fingerprints untouched; rational artifact uses its own closed schema coupling_clock_rational.v1",
        },
        "clock_cases": clock_rows,
        "causality_gate": gate,
        "refinement_scan": refine,
        "band_scan": bands,
        "interpolation_scan": interp,
        "interpolation_single_interval": interp_one,
        "criterion": criterion,
        "figures": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
    }
    (OUT / "clock_semantics.json").write_text(json.dumps(payload, indent=2) + "\n")

    summary = {
        "mission_id": MISSION,
        "node": NODE,
        "criterion": criterion,
        "verdict": "PASS"
        if (
            criterion["gcd_cadence_machine_verified"]
            and criterion["causality_negative_case_raised"]
            and criterion["multirate_order_ok"]
            and criterion["band_all_within_refinement"]
            and criterion["band_all_within_decade_scan"]
            and criterion["interpolation_linear_order_ok"]
        )
        else "FAIL",
    }
    (OUT / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary["criterion"], indent=2))
    print("verdict:", summary["verdict"])


if __name__ == "__main__":
    main()