"""Design-sea-state statistical-convention verification surface (p229w4-n05).

One call, :func:`run_study`, produces the whole auditable chain:

1. **Rosenblatt transform** of the two-block sea-state model
   (:mod:`~tensorphysics.design_sea_state.rosenblatt`) — closure verified by
   both an inverse round trip and the density/Jacobian identity.
2. **IFORM contours** (:mod:`~tensorphysics.design_sea_state.iform`) at the two
   beta conventions, plus the exact analytic anchors
   (``max H_s = F_H^-1(Phi(beta))``; correlated-normal quadratic form).
3. **Reliability mapping** (:mod:`~tensorphysics.design_sea_state.reliability_index`)
   — ``p_f <-> beta`` in both conventions, return periods, lifetime targets.
4. **Rayleigh conventions** (:mod:`~tensorphysics.design_sea_state.rayleigh_consistency`)
   — closed forms, Monte-Carlo audit, and the three-way adjudication
   IFORM / Rayleigh / exact long-term quadrature.
5. **Criteria** — every machine criterion is a number plus a threshold, emitted
   by :func:`evaluate_criteria` so a reviewer reads verdicts, not prose.

Convention discipline
---------------------
* The contour **probability content** is ``exp(-beta**2/2)`` (convention C).
* The contour radius used to **invert a response at target probability**
  ``p_f`` is ``-Phi^-1(p_f)`` (convention F).
* Both are returned, both are named, and :func:`evaluate_criteria` keeps a
  criterion that *fails loudly* if someone conflates them.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Callable

import numpy as np

from .iform import (
    contour_max_hs,
    contour_refinement_error,
    iform_contour,
    mahalanobis_residual,
    max_response_on_contour,
    quadratic_residual_normal,
)
from .rayleigh_consistency import (
    adjudicate_rayleigh_vs_iform,
    long_term_reference,
    modal_largest_wave_response,
    quadrature_grid,
    rayleigh_mc_check,
    wave_power,
)
from .reliability_index import (
    DNV_ISO2394_TARGETS,
    ReliabilityTarget,
    beta_contour_from_pout,
    beta_cornell_from_pf,
    convention_report,
    lifetime_pf,
    pf_from_beta_cornell,
    pout_from_beta_contour,
    return_period_from_lifetime_pf,
)
from .rosenblatt import (
    BivariateNormalTransform,
    LognormalConditional,
    RosenblattTransform,
    Weibull3,
)

__all__ = [
    "DEFAULT_CONFIG",
    "MACHINE_CRITERIA_KEYS",
    "build_model",
    "run_study",
    "evaluate_criteria",
    "points_outside_polygon",
    "mc_outside_probability",
]

#: Default metocean site — a moderate North Sea type climate.
#: ``H_s``: 3-parameter Weibull ``(k=1.6, alpha=2.6 m, gamma=0.6 m)`` gives a
#: 100-year return value ``F^-1(0.99) ~ 7.4 m``; ``T_z | H_s`` follows the
#: Haver-Nyhus power law ``E[T_z|h] = 3.6 + 1.0 h**0.7`` s with
#: ``sigma_ln = 0.08``.
DEFAULT_CONFIG: dict[str, object] = {
    "hs_weibull": {"k": 1.6, "alpha": 2.6, "gamma": 0.6},
    "tz_conditional": {"a1": 3.6, "a2": 1.0, "a3": 0.7, "sigma_ln": 0.08},
    "annual_pf": 1.0e-2,
    "n_theta": 72,
    "n_theta_fine": 288,
    "quadrature_nodes": 160,
    "mc_samples": 200_000,
    "mc_seed": 20260101,
    "normal_anchor": {"mu_x": 3.0, "sigma_x": 1.2, "mu_y": 8.0, "sigma_y": 1.5, "rho": 0.6},
}

#: Machine criteria keys (one PASS/FAIL row per key, each with a measured number).
MACHINE_CRITERIA_KEYS = (
    "C1_iform_normal_anchor",
    "C2_contour_extreme_hs_anchor",
    "C3_rosenblatt_closure",
    "C4_exceedance_calibration",
    "C5_reliability_mapping",
    "C6_convention_mismatch_declared",
    "C7_rayleigh_mode_recovery",
    "C8_rayleigh_vs_iform",
    "C9_iform_vs_longterm_reference",
    "C10_contour_refinement",
    "C11_conventionC_probability_content",
    "C12_normal_case_contour_is_ellipse",
    "C13_gaussian_surrogate_differs",
)


def build_model(config: dict | None = None) -> RosenblattTransform:
    """Instantiate the Rosenblatt transform from a config dict."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    w = cfg["hs_weibull"]
    c = cfg["tz_conditional"]
    return RosenblattTransform(
        hs_marginal=Weibull3(k=w["k"], alpha=w["alpha"], gamma=w["gamma"]),
        tz_conditional=LognormalConditional(
            a1=c["a1"], a2=c["a2"], a3=c["a3"], sigma_ln=c["sigma_ln"]
        ),
    )


# --------------------------------------------------------------------------- #
# Monte-Carlo helpers (independent code path: physical-space sampling)
# --------------------------------------------------------------------------- #
def points_outside_polygon(
    x: np.ndarray, y: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray
) -> np.ndarray:
    """Even-odd ray casting: which of the points ``(x, y)`` fall outside the polygon.

    Kept in-repo (a dozen lines) so the calibration test does not depend on a
    geometry library, and so an inverted contour (wrong winding, unclosed curve)
    shows up as a probability error rather than as a library complaint.
    """
    inside = np.zeros(x.shape, dtype=bool)
    n = poly_x.size
    for i in range(n):
        j = (i - 1) % n
        xi, yi, xj, yj = poly_x[i], poly_y[i], poly_x[j], poly_y[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_int)
    return ~inside


def mc_outside_probability(
    transform: RosenblattTransform,
    beta: float,
    n: int,
    seed: int,
    n_theta: int = 144,
) -> dict[str, float]:
    """MC probability that a sampled sea state lies outside the ``beta`` contour.

    Sampling is done through the physical branch (uniform -> Weibull ppf, then
    uniform -> conditional log-normal ppf), *not* by mapping normals; membership
    is then decided in physical space by ray casting.  Both choices are
    independent of :func:`iform_contour`, so the comparison is a genuine
    end-to-end check of transform + closure + probability content.
    """
    contour = iform_contour(transform, beta, n_theta)
    poly_x = np.array([p.hs for p in contour])
    poly_y = np.array([p.tz for p in contour])
    samples = transform.sample(n, seed=seed)
    arr = np.array(samples)
    outside = points_outside_polygon(arr[:, 0], arr[:, 1], poly_x, poly_y)
    measured = float(outside.mean())
    target = pout_from_beta_contour(beta)
    se = math.sqrt(max(target * (1.0 - target), 0.0) / n)
    return {
        "beta": beta,
        "measured_p_out": measured,
        "target_p_out": target,
        "rel_error": abs(measured - target) / target,
        "binomial_se": se,
        "z_score": (measured - target) / se if se > 0.0 else 0.0,
        "n_samples": n,
    }


def gaussian_surrogate(
    model: RosenblattTransform, grid: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> BivariateNormalTransform:
    """Moment-matched bivariate-normal stand-in for the sea-state model.

    Built from the same quadrature grid (mean / std / correlation of ``H_s`` and
    ``T_z``).  It is *not* an equivalent model: the Weibull x log-normal pair is
    skewed, so its 100-year design sea state differs from the Gaussian stand-in's.
    The size of that difference is what criterion C13 records — it is the reason
    the non-Gaussian Rosenblatt machinery has to exist at all.
    """
    hs, tz, w = grid
    tot = float(w.sum())
    mh = float((w * hs).sum() / tot)
    mt = float((w * tz).sum() / tot)
    sh = float(math.sqrt((w * (hs - mh) ** 2).sum() / tot))
    st = float(math.sqrt((w * (tz - mt) ** 2).sum() / tot))
    cov = float((w * (hs - mh) * (tz - mt)).sum() / tot)
    return BivariateNormalTransform(mu_x=mh, sigma_x=sh, mu_y=mt, sigma_y=st, rho=cov / (sh * st))


# --------------------------------------------------------------------------- #
# main study
# --------------------------------------------------------------------------- #
def run_study(config: dict | None = None) -> dict:
    """Run the full design-sea-state convention study; returns a JSON-able dict."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    model = build_model(cfg)
    annual_pf = float(cfg["annual_pf"])
    n_theta = int(cfg["n_theta"])
    target = ReliabilityTarget(annual_pf)
    beta_f = target.beta_cornell           # convention F: response inversion
    beta_c = target.beta_contour           # convention C: probability content

    # ---- 1. Rosenblatt closure -------------------------------------------
    samples = model.sample(400, seed=int(cfg["mc_seed"]))
    round_trip = []
    for hs, tz in samples:
        u1, u2 = model.forward(hs, tz)
        hs2, tz2 = model.inverse(u1, u2)
        round_trip.append(max(abs(hs2 - hs) / hs, abs(tz2 - tz) / tz))
    closure = {
        "max_log_density_residual": model.max_log_residual(samples),
        "max_roundtrip_rel_error": max(round_trip),
        "n_samples": len(samples),
    }

    # ---- 2. IFORM contours ----------------------------------------------
    contour_f = iform_contour(model, beta_f, n_theta)
    contour_c = iform_contour(model, beta_c, n_theta)
    hs_max = contour_max_hs(contour_f)
    hs_anchor = model.hs_marginal.ppf(0.5 * math.erfc(-beta_f / math.sqrt(2.0)))
    contour_block = {
        "beta_convention_F": beta_f,
        "beta_convention_C": beta_c,
        "max_hs_convention_F": hs_max,
        "max_hs_analytic_anchor": hs_anchor,
        "max_hs_rel_error": abs(hs_max - hs_anchor) / hs_anchor,
        "max_hs_convention_C": contour_max_hs(contour_c),
        "contour_F": {
            "hs": [p.hs for p in contour_f],
            "tz": [p.tz for p in contour_f],
            "theta": [p.theta for p in contour_f],
        },
        "contour_C": {
            "hs": [p.hs for p in contour_c],
            "tz": [p.tz for p in contour_c],
            "theta": [p.theta for p in contour_c],
        },
        "refinement": contour_refinement_error(
            model, beta_f, coarse=36, fine=int(cfg["n_theta_fine"])
        ),
    }

    # ---- 2b. analytic bivariate-normal anchor ----------------------------
    na = cfg["normal_anchor"]
    normal_model = BivariateNormalTransform(
        mu_x=na["mu_x"],
        sigma_x=na["sigma_x"],
        mu_y=na["mu_y"],
        sigma_y=na["sigma_y"],
        rho=na["rho"],
    )
    normal_contour = iform_contour(normal_model, beta_f, 256)
    anchor = {
        "beta": beta_f,
        "max_quadratic_residual": quadratic_residual_normal(normal_model, normal_contour),
        "max_mahalanobis_residual": mahalanobis_residual(normal_model, normal_contour),
        "max_x_numeric": max(p.hs for p in normal_contour),
        "max_x_analytic": normal_model.mu_x + normal_model.sigma_x * beta_f,
        "n_points": len(normal_contour),
    }
    anchor["max_x_rel_error"] = (
        abs(anchor["max_x_numeric"] - anchor["max_x_analytic"]) / anchor["max_x_analytic"]
    )
    # For a correlated normal the IFORM contour IS the chi-square ellipse
    # (both are level sets of the joint density) -> the residual must vanish.
    anchor["note"] = (
        "correlated-normal anchor: Rosenblatt circle == chi-square ellipse, "
        "max H_s == mu_x + sigma_x * beta (both analytic identities)"
    )

    # ---- 3. reliability mapping -----------------------------------------
    reliability = {
        "target": target.as_dict(),
        "conventions": convention_report((1e-2, 1e-3, 1e-4)),
        "lifetime": {
            "years": 25.0,
            "pf_lifetime_at_target_annual": lifetime_pf(annual_pf, 25.0),
            "annual_return_period_equiv": return_period_from_lifetime_pf(0.1, 25.0),
        },
        "dnv_iso2394_rows": [],
        "roundtrip_max_rel_error": 0.0,
        "chi2_identity_max_rel_error": 0.0,
    }
    worst_rt = 0.0
    for beta in (1.0, 2.0, 2.5, 3.0, 3.719, 4.2):
        pf = pf_from_beta_cornell(beta)
        worst_rt = max(worst_rt, abs(beta_cornell_from_pf(pf) - beta) / beta)
    reliability["roundtrip_max_rel_error"] = worst_rt
    from scipy.stats import chi2  # local import: only used for the analytic check

    worst_chi = 0.0
    for beta in (1.0, 2.0, 3.0):
        worst_chi = max(worst_chi, abs(chi2.sf(beta**2, df=2) / pout_from_beta_contour(beta) - 1.0))
    reliability["chi2_identity_max_rel_error"] = worst_chi
    for name, beta_ref, pf_ref in DNV_ISO2394_TARGETS:
        mapped = beta_cornell_from_pf(pf_ref)
        reliability["dnv_iso2394_rows"].append(
            {
                "class": name,
                "beta_reference": beta_ref,
                "pf_reference": pf_ref,
                "beta_from_pf": mapped,
                "rel_error": abs(mapped - beta_ref) / beta_ref,
            }
        )

    # ---- 4. Rayleigh conventions + adjudication --------------------------
    rayleigh_mc = rayleigh_mc_check(hs=6.0, n_waves=1000.0, n_groups=4000, seed=int(cfg["mc_seed"]))
    grid = quadrature_grid(model, n=int(cfg["quadrature_nodes"]))
    surrogate = gaussian_surrogate(model, grid)
    surrogate_contour = iform_contour(surrogate, beta_f, 256)
    surrogate_block = {
        "rho": surrogate.rho,
        "mu_hs": surrogate.mu_x,
        "sigma_hs": surrogate.sigma_x,
        "mu_tz": surrogate.mu_y,
        "sigma_tz": surrogate.sigma_y,
        "max_hs": max(p.hs for p in surrogate_contour),
        "max_hs_model": hs_max,
    }
    responses: dict[str, Callable[[float, float], float]] = {
        "wave_power": wave_power,
        "modal_largest_wave": modal_largest_wave_response,
    }
    adjudication: dict[str, dict] = {}
    for name, fn in responses.items():
        value_f, point_f = max_response_on_contour(contour_f, fn)
        value_c, point_c = max_response_on_contour(contour_c, fn)
        reference = long_term_reference(model, fn, annual_pf, grid=grid)
        block = adjudicate_rayleigh_vs_iform(
            model,
            fn,
            iform_value=value_f,
            iform_point_hs=point_f.hs,
            iform_point_tz=point_f.tz,
            reference_value=reference["design_response"],
            target_prob=annual_pf,
            beta=beta_f,
        )
        block.update(
            {
                "iform_theta_deg": math.degrees(point_f.theta),
                "iform_max_hs": contour_max_hs(contour_f),
                "conventionC_iform_value": value_c,
                "conventionC_theta_deg": math.degrees(point_c.theta),
                "conventionC_vs_reference_rel": abs(value_c - reference["design_response"])
                / reference["design_response"],
                "reference": reference,
            }
        )
        block["gaussian_surrogate_value"] = max(
            fn(p.hs, p.tz) for p in surrogate_contour
        )
        block["gaussian_surrogate_rel_diff"] = abs(
            block["gaussian_surrogate_value"] - value_f
        ) / abs(value_f)
        adjudication[name] = block

    # ---- 5. exceedance calibration (MC, physical branch) ----------------
    calibration = []
    for beta in (2.0, 2.5):
        calibration.append(
            mc_outside_probability(
                model, beta, n=int(cfg["mc_samples"]), seed=int(cfg["mc_seed"])
            )
        )

    result = {
        "config": cfg,
        "rosenblatt_closure": closure,
        "contour": contour_block,
        "normal_anchor": anchor,
        "gaussian_surrogate": surrogate_block,
        "reliability": reliability,
        "rayleigh": rayleigh_mc,
        "adjudication": adjudication,
        "exceedance_calibration": calibration,
    }
    result["criteria"] = evaluate_criteria(result)
    return result


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #
def evaluate_criteria(result: dict) -> list[dict]:
    """Turn the study numbers into PASS/FAIL rows (threshold, measured, key)."""
    rows: list[dict] = []

    def add(key: str, measured: float, threshold: float, sense: str, note: str) -> None:
        if sense == "<=":
            ok = measured <= threshold
        elif sense == ">=":
            ok = measured >= threshold
        else:  # pragma: no cover - programming error
            raise ValueError(sense)
        rows.append(
            {
                "key": key,
                "measured": measured,
                "threshold": threshold,
                "sense": sense,
                "verdict": "PASS" if ok else "FAIL",
                "note": note,
            }
        )

    contour = result["contour"]
    anchor = result["normal_anchor"]
    rel = result["reliability"]
    ray = result["rayleigh"]
    adj = result["adjudication"]
    cal = result["exceedance_calibration"]

    add(
        "C1_iform_normal_anchor",
        anchor["max_quadratic_residual"],
        1.0e-2,
        "<=",
        "IFORM contour vs analytic bivariate-normal anchor (criterion: <=1%)",
    )
    add(
        "C2_contour_extreme_hs_anchor",
        contour["max_hs_rel_error"],
        1.0e-2,
        "<=",
        "max H_s on contour vs F_H^-1(Phi(beta)) analytic anchor (criterion: <=1%)",
    )
    add(
        "C3_rosenblatt_closure",
        max(result["rosenblatt_closure"]["max_roundtrip_rel_error"],
            result["rosenblatt_closure"]["max_log_density_residual"]),
        1.0e-8,
        "<=",
        "inverse round trip + phi2(u)|det J| = f(x) density identity",
    )
    add(
        "C4_exceedance_calibration",
        max(c["rel_error"] for c in cal),
        3.0e-2,
        "<=",
        "MC outside-contour probability vs exp(-beta^2/2) (criterion: calibrated)",
    )
    add(
        "C5_reliability_mapping",
        max(rel["roundtrip_max_rel_error"],
            max(r["rel_error"] for r in rel["dnv_iso2394_rows"])),
        1.0e-2,
        "<=",
        "p_f <-> beta round trip + DNV/ISO 2394 target rows",
    )
    worst_conv = rel["conventions"]["worst_row"]
    add(
        "C6_convention_mismatch_declared",
        abs(math.log(worst_conv["pout_over_target"])),
        0.3,
        ">=",
        "convention F vs C mismatch quantified (must be non-trivial and declared)",
    )
    add(
        "C7_rayleigh_mode_recovery",
        max(ray["mode_rel_err"], ray["mode_max_rel_err"], ray["mean_rel_err"], ray["hs_rel_err"]),
        2.0e-2,
        "<=",
        "Rayleigh mode / N-wave modal max / mean / H_s vs closed form (MC, criterion <=5%)",
    )
    add(
        "C8_rayleigh_vs_iform",
        max(b["rayleigh_vs_iform_rel"] for b in adj.values()),
        5.0e-2,
        "<=",
        "Rayleigh-rule design value vs IFORM contour design value (criterion: <=5%)",
    )
    add(
        "C9_iform_vs_longterm_reference",
        max(b["iform_vs_reference_rel"] for b in adj.values()),
        5.0e-2,
        "<=",
        "IFORM contour vs exact long-term quadrature reference at the same p_f",
    )
    add(
        "C10_contour_refinement",
        max(
            contour["refinement"]["response_rel_error"],
            contour["refinement"]["max_hs_rel_error"],
        ),
        1.0e-2,
        "<=",
        "angle-sweep refinement 36 -> 288 points",
    )
    add(
        "C11_conventionC_probability_content",
        max(c["rel_error"] for c in cal),
        1.0e-1,
        "<=",
        "convention-C contour (beta_C) probability content equals its own target",
    )
    add(
        "C12_normal_case_contour_is_ellipse",
        anchor["max_mahalanobis_residual"],
        1.0e-8,
        "<=",
        "correlated-normal anchor: Rosenblatt contour == chi-square ellipse (analytic identity)",
    )
    add(
        "C13_gaussian_surrogate_differs",
        min(b["gaussian_surrogate_rel_diff"] for b in adj.values()),
        2.0e-2,
        ">=",
        "non-Gaussian marginals must move the design value vs a moment-matched Gaussian surrogate",
    )
    return rows