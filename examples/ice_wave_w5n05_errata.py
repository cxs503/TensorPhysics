"""Errata re-derivation for paper W5-N05 (mission m994) -- fitting / statistics only.

Scope discipline
----------------
This script does **not** re-run the physics pipeline (rule of the erratum).  It
takes ``artifacts/w5n05/results.json`` -- the exact artifact report #823 was
rendered from -- and re-derives, from stored arrays only:

1. every candidate power-law convention for ``alpha ~ n_c^p h_i^q``: per-slice
   log-log slopes on the full 4x8 calibration grid, the joint two-regressor OLS,
   and the same conventions restricted to the 8-point sub-grid the report
   *printed*.  This pins down both the generation convention of the published
   ``p = 0.3840`` / ``q = 1.7530`` and the convention the independent reviewer
   (report #961) must have used (``0.4011 / 1.1225``, slices ``0.6278 / 0.1744``
   and ``1.5288 / 0.8122``);
2. the EnKF error reduction ``1 - RMSE_a / RMSE_b`` from full-precision stored
   arrays, and again from the 4-decimal values as *displayed* in the report
   table, which pins down the 8.65 % vs 8.33 % discrepancy.

The errata document ``docs/papers/w5n05_ice_wave/ERRATA.md`` is rendered from
``ERRATA_TEMPLATE.md`` with every number injected here -- nothing is typed by
hand.

Usage::

    python3 examples/ice_wave_w5n05_errata.py            # recompute + write artifacts
    python3 examples/ice_wave_w5n05_errata.py --check    # assert adjudicated values

Exit code 0 on success, 1 if ``--check`` assertions fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "w5n05" / "results.json"
OUT_DIR = ROOT / "artifacts" / "w5n05" / "errata"
OUT_JSON = OUT_DIR / "errata_w5n05.json"
PAPER = ROOT / "docs" / "papers" / "w5n05_ice_wave"
TEMPLATE = PAPER / "ERRATA_TEMPLATE.md"
OUT_MD = PAPER / "ERRATA.md"

#: Published (report #823) and independently reviewed (report #961) values.
#: Used ONLY to identify which convention reproduces them -- never as a source.
PUBLISHED = {"p": 0.3840, "q": 1.7530, "enkf_reduction": 0.0865}
REVIEW = {
    "p_joint": 0.4011,
    "q_joint": 1.1225,
    "p_slice": (0.6278, 0.1744),
    "q_slice": (1.5288, 0.8122),
    "enkf_reduction": 0.0833,
}


# --------------------------------------------------------------------------- #
# log-log fits
# --------------------------------------------------------------------------- #
def _loglog_slope(x: Any, y: Any) -> dict[str, float]:
    """Slope of ln y versus ln x with intercept, plus its standard error."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    design = np.column_stack([np.ones_like(lx), lx])
    coef, *_ = np.linalg.lstsq(design, ly, rcond=None)
    resid = ly - design @ coef
    sigma2 = float(resid @ resid) / max(lx.size - 2, 1)
    cov = sigma2 * np.linalg.inv(design.T @ design)
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return {
        "slope": float(coef[1]),
        "slope_stderr": float(np.sqrt(cov[1, 1])),
        "intercept": float(coef[0]),
        "r2": 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0.0 else 1.0,
        "n_points": int(lx.size),
    }


def _joint_ols(conc: np.ndarray, thick: np.ndarray, grid: np.ndarray) -> dict[str, Any]:
    """Joint log-log OLS: ln(alpha) = c + p ln(n_c) + q ln(h_i); grid[thick, conc]."""
    n_thick, n_conc = grid.shape
    ln_n = np.tile(np.log(conc), n_thick)
    ln_h = np.repeat(np.log(thick), n_conc)
    alpha = np.asarray(grid, float).reshape(-1)
    ln_a = np.log(alpha)
    design = np.column_stack([np.ones_like(ln_a), ln_n, ln_h])
    coef, *_ = np.linalg.lstsq(design, ln_a, rcond=None)
    resid = ln_a - design @ coef
    sigma2 = float(resid @ resid) / max(ln_a.size - 3, 1)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(design.T @ design)))
    ss_tot = float(((ln_a - ln_a.mean()) ** 2).sum())
    return {
        "p": float(coef[1]),
        "p_stderr": float(se[1]),
        "q": float(coef[2]),
        "q_stderr": float(se[2]),
        "p_ci95": [float(coef[1] - 2 * se[1]), float(coef[1] + 2 * se[1])],
        "q_ci95": [float(coef[2] - 2 * se[2]), float(coef[2] + 2 * se[2])],
        "r2": 1.0 - float(resid @ resid) / ss_tot,
        "rmse_log": float(np.sqrt(sigma2)),
        "n_points": int(ln_a.size),
        "p_single_regressor": _loglog_slope(np.tile(conc, n_thick), alpha)["slope"],
        "q_single_regressor": _loglog_slope(np.repeat(thick, n_conc), alpha)["slope"],
    }


def _slice_tables(conc: np.ndarray, thick: np.ndarray, grid: np.ndarray):
    p_slices = [{"fixed_thickness": float(h), **_loglog_slope(conc, grid[i, :])}
                for i, h in enumerate(thick)]
    q_slices = [{"fixed_concentration": float(n), **_loglog_slope(thick, grid[:, j])}
                for j, n in enumerate(conc)]
    return p_slices, q_slices


# --------------------------------------------------------------------------- #
# MIZ exponent census
# --------------------------------------------------------------------------- #
def miz_exponent_census(miz: dict[str, Any]) -> dict[str, Any]:
    sweep = miz["sweep"]
    conc = np.asarray(sweep["concentrations"], float)
    thick = np.asarray(sweep["ice_thicknesses"], float)
    grid = np.asarray(sweep["alpha_grid"], float)
    assert grid.shape == (thick.size, conc.size), (grid.shape, thick.size, conc.size)

    p_slices, q_slices = _slice_tables(conc, thick, grid)
    i_mid, j_mid = thick.size // 2, conc.size // 2        # code convention
    p_code = _loglog_slope(conc, grid[i_mid, :])
    q_code = _loglog_slope(thick, grid[:, j_mid])
    stored_p = float(sweep["alpha_vs_concentration_exponent"])
    stored_q = float(sweep["alpha_vs_thickness_exponent"])

    # the 8-point sub-grid actually printed in the report (miz.cross_checks)
    sub_n = np.asarray(sorted({float(r["concentration"]) for r in miz["cross_checks"]}))
    sub_h = np.asarray(sorted({float(r["ice_thickness"]) for r in miz["cross_checks"]}))
    idx_n = [int(np.argmin(np.abs(conc - n))) for n in sub_n]
    idx_h = [int(np.argmin(np.abs(thick - h))) for h in sub_h]
    assert all(np.isclose(conc[j], n) for j, n in zip(idx_n, sub_n))
    assert all(np.isclose(thick[i], h) for i, h in zip(idx_h, sub_h))
    subgrid = grid[np.ix_(idx_h, idx_n)]
    printed = {(float(r["concentration"]), float(r["ice_thickness"])): r["alpha"]
               for r in miz["cross_checks"]}
    print_err = max(abs(printed[(float(n), float(h))] - subgrid[i, j])
                    for i, h in enumerate(sub_h) for j, n in enumerate(sub_n))
    p_sub, q_sub = _slice_tables(sub_n, sub_h, subgrid)
    joint_sub = _joint_ols(sub_n, sub_h, subgrid)

    ident = {
        "p_joint_from_subgrid": joint_sub["p"],
        "p_joint_abs_diff": abs(joint_sub["p"] - REVIEW["p_joint"]),
        "q_joint_from_subgrid": joint_sub["q"],
        "q_joint_abs_diff": abs(joint_sub["q"] - REVIEW["q_joint"]),
        "p_slices_from_subgrid": [r["slope"] for r in p_sub],
        "p_slice_abs_diff": [abs(p_sub[0]["slope"] - REVIEW["p_slice"][0]),
                             abs(p_sub[-1]["slope"] - REVIEW["p_slice"][1])],
        "q_slices_from_subgrid": [r["slope"] for r in q_sub],
        "q_slice_abs_diff": [abs(q_sub[0]["slope"] - REVIEW["q_slice"][0]),
                             abs(q_sub[-1]["slope"] - REVIEW["q_slice"][1])],
    }
    ident["max_abs_diff"] = max(ident["p_joint_abs_diff"], ident["q_joint_abs_diff"],
                                *ident["p_slice_abs_diff"], *ident["q_slice_abs_diff"])

    return {
        "grid_shape": [int(thick.size), int(conc.size)],
        "concentrations": conc.tolist(),
        "ice_thicknesses": thick.tolist(),
        "period": float(sweep["period"]),
        "depth": float(sweep["depth"]),
        "loss_factor": float(sweep["loss_factor"]),
        "stored": {"p": stored_p, "q": stored_q},
        "code_convention": {
            "i_mid": i_mid, "j_mid": j_mid,
            "p_slice_thickness": float(thick[i_mid]),
            "q_slice_concentration": float(conc[j_mid]),
            "p": p_code["slope"], "p_abs_err_vs_stored": abs(p_code["slope"] - stored_p),
            "q": q_code["slope"], "q_abs_err_vs_stored": abs(q_code["slope"] - stored_q),
        },
        "p_slices_full_grid": p_slices,
        "q_slices_full_grid": q_slices,
        "p_slice_range": [min(r["slope"] for r in p_slices),
                          max(r["slope"] for r in p_slices)],
        "q_slice_range": [min(r["slope"] for r in q_slices),
                          max(r["slope"] for r in q_slices)],
        "joint_full_grid": _joint_ols(conc, thick, grid),
        "printed_subgrid": {
            "concentrations": sub_n.tolist(), "ice_thicknesses": sub_h.tolist(),
            "grid": subgrid.tolist(),
            "printed_alpha_max_abs_diff_vs_grid": float(print_err),
            "p_slices": p_sub, "q_slices": q_sub, "joint": joint_sub,
        },
        "review_convention_identification": ident,
        # thin-plate flexural anchor: on the k^5 branch k ~ (n_c h^3)^{1/5}
        "theory_thin_plate_flexural": {"p": 0.2, "q": 0.6},
    }


# --------------------------------------------------------------------------- #
# EnKF error reduction
# --------------------------------------------------------------------------- #
def enkf_error_reduction(enkf: dict[str, Any]) -> dict[str, Any]:
    truth = np.asarray(enkf["truth"], float)
    bg = np.asarray(enkf["background_mean"], float)
    an = np.asarray(enkf["analysis_mean"], float)
    rmse_bg = float(np.sqrt(np.mean((bg - truth) ** 2)))
    rmse_an = float(np.sqrt(np.mean((an - truth) ** 2)))
    reduction = 1.0 - rmse_an / rmse_bg
    bg4, an4 = float(f"{rmse_bg:.4f}"), float(f"{rmse_an:.4f}")
    disp = 1.0 - an4 / bg4
    return {
        "rmse_background_recomputed": rmse_bg, "rmse_analysis_recomputed": rmse_an,
        "reduction_recomputed": reduction,
        "rmse_background_stored": float(enkf["rmse_background"]),
        "rmse_analysis_stored": float(enkf["rmse_analysis"]),
        "reduction_stored": float(enkf["rmse_reduction"]),
        "rmse_background_abs_err": abs(rmse_bg - float(enkf["rmse_background"])),
        "rmse_analysis_abs_err": abs(rmse_an - float(enkf["rmse_analysis"])),
        "reduction_abs_err": abs(reduction - float(enkf["rmse_reduction"])),
        "display_4dp": {"rmse_background": bg4, "rmse_analysis": an4},
        "reduction_from_displayed_4dp": disp,
        "reduction_gap_pp": (reduction - disp) * 100.0,
        "reduction_display_abs_diff_vs_review": abs(disp - REVIEW["enkf_reduction"]),
        "n_state": int(truth.size), "n_ensemble": int(enkf["n_ensemble"]),
    }


def adjudicate(results: dict[str, Any]) -> dict[str, Any]:
    census = miz_exponent_census(results["miz"])
    return {
        "mission": results.get("mission"),
        "node": results.get("node"),
        "source_artifact": str(RESULTS.relative_to(ROOT)),
        "miz_census": census,
        "review_convention_identification": census["review_convention_identification"],
        "enkf": enkf_error_reduction(results["enkf"]),
        "criteria_provenance": {
            "miz_alpha_cross_check": float(
                results["criteria"]["d_miz_alpha_error_lt_15pct"]["value"]),
            "miz_alpha_cross_check_note": results["criteria"][
                "d_miz_alpha_error_lt_15pct"]["note"],
        },
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(results: dict[str, Any], adj: dict[str, Any], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    c = adj["miz_census"]
    j = c["joint_full_grid"]
    conc = np.asarray(c["concentrations"])
    thick = np.asarray(c["ice_thicknesses"])
    grid = np.asarray(results["miz"]["sweep"]["alpha_grid"], float)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ps, qs = c["p_slices_full_grid"], c["q_slices_full_grid"]
    axes[0].errorbar(thick, [r["slope"] for r in ps], yerr=[r["slope_stderr"] for r in ps],
                     fmt="o-", color="C0", label="per-slice $p(h_i)$, 4x8 grid")
    axes[0].axhline(c["stored"]["p"], color="C3", ls="--",
                    label="published p = %.4f" % c["stored"]["p"])
    axes[0].axhline(j["p"], color="C2", ls="-.",
                    label="joint OLS p = %.4f $\\pm$ %.3f" % (j["p"], j["p_stderr"]))
    axes[0].axhline(REVIEW["p_joint"], color="C4", ls=":", lw=1.3,
                    label="review joint p = %.4f" % REVIEW["p_joint"])
    axes[0].axhline(0.2, color="0.5", lw=0.9, alpha=0.7, label="flexural anchor p = 1/5")
    axes[0].set_xlabel("ice thickness $h_i$ (m)")
    axes[0].set_ylabel("concentration exponent $p$")
    axes[0].set_title("$\\alpha \\sim n_c^{p}$: p depends on the slice")
    axes[0].legend(fontsize=6.5)
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(conc, [r["slope"] for r in qs], yerr=[r["slope_stderr"] for r in qs],
                     fmt="s-", color="C1", label="per-slice $q(n_c)$, 4x8 grid")
    axes[1].axhline(c["stored"]["q"], color="C3", ls="--",
                    label="published q = %.4f" % c["stored"]["q"])
    axes[1].axhline(j["q"], color="C2", ls="-.",
                    label="joint OLS q = %.4f $\\pm$ %.3f" % (j["q"], j["q_stderr"]))
    axes[1].axhline(REVIEW["q_joint"], color="C4", ls=":", lw=1.3,
                    label="review joint q = %.4f" % REVIEW["q_joint"])
    axes[1].axhline(0.6, color="0.5", lw=0.9, alpha=0.7, label="flexural anchor q = 3/5")
    axes[1].set_xlabel("ice concentration $n_c$")
    axes[1].set_ylabel("thickness exponent $q$")
    axes[1].set_title("$\\alpha \\sim h_i^{q}$: q depends on the slice")
    axes[1].legend(fontsize=6.5)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    written.append(out_dir / "errata_fig1_exponent_conventions.png")
    fig.savefig(written[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for i, h in enumerate(thick):
        ax.loglog(conc, grid[i, :], "o-", ms=4, label="h = %g m" % h)
    sub_n = np.asarray(c["printed_subgrid"]["concentrations"])
    sub_g = np.asarray(c["printed_subgrid"]["grid"])
    for i, h in enumerate(c["printed_subgrid"]["ice_thicknesses"]):
        ax.loglog(sub_n, sub_g[i, :], "k*", ms=13, mfc="none",
                  label="printed in report table" if i == 0 else None)
    ax.loglog(conc, grid[c["code_convention"]["i_mid"], :], color="k", lw=4, alpha=0.22,
              label="p slice (h = %g m)" % c["code_convention"]["p_slice_thickness"])
    ax.set_xlabel("ice concentration $n_c$")
    ax.set_ylabel(r"$\alpha$ (1/m)")
    ax.set_title("Calibration grid (4x8) vs the 8 points printed in the report")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    written.append(out_dir / "errata_fig2_grid_vs_printed.png")
    fig.savefig(written[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    vals = [adj["enkf"]["reduction_recomputed"] * 100.0,
            adj["enkf"]["reduction_from_displayed_4dp"] * 100.0]
    labels = ["full precision\n(results.json)", "4-dp rendered RMSE\n(reviewer input)"]
    bars = ax.bar(labels, vals, color=["C2", "C3"], width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v + 0.08, "%.4f%%" % v, ha="center")
    ax.set_ylabel("RMSE reduction vs background (%)")
    ax.set_ylim(0.0, max(vals) * 1.30)
    ax.set_title("EnKF reduction: %.4f%% (correct) vs %.4f%% (rounding artifact)"
                 % (vals[0], vals[1]))
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    written.append(out_dir / "errata_fig3_enkf_rounding.png")
    fig.savefig(written[-1], dpi=150)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------- #
# Errata document rendering (template + injected numbers)
# --------------------------------------------------------------------------- #
def _pct(num: float, den: float) -> str:
    return "%.2f%%" % (100.0 * num / abs(den))


def render_errata(adj: dict[str, Any], figures: list[Path]) -> str:
    c, e = adj["miz_census"], adj["enkf"]
    j, conv = c["joint_full_grid"], c["code_convention"]
    ident, sub = adj["review_convention_identification"], c["printed_subgrid"]
    th, xchk = c["theory_thin_plate_flexural"], adj["criteria_provenance"]
    p_lo, p_hi = c["p_slice_range"]
    q_lo, q_hi = c["q_slice_range"]
    n_sub = len(sub["concentrations"]) * len(sub["ice_thicknesses"])
    ns = ", ".join("%g" % v for v in sub["concentrations"])
    hs = ", ".join("%g" % v for v in sub["ice_thicknesses"])
    red_ppm = 100.0 * e["reduction_recomputed"]

    def row(*cells: str) -> str:
        return "| " + " | ".join(cells) + " |"

    summary = "\n".join([
        "| 项 | 报告值（#823） | 评审复算值（#961） | 核定正确值 | 判定 |",
        "|---|---|---|---|---|",
        row("MIZ 指数 p", "%.4f" % c["stored"]["p"], "%.4f" % REVIEW["p_joint"],
            "%.4f ± %.4f（32 点联合 OLS；95%% CI [%.3f, %.3f]）"
            % (j["p"], j["p_stderr"], j["p_ci95"][0], j["p_ci95"][1]),
            "报告值是固定冰厚切片斜率，算术无误但口径未声明；"
            "作为「全局标定律」不成立"),
        row("MIZ 指数 q", "%.4f" % c["stored"]["q"], "%.4f" % REVIEW["q_joint"],
            "%.4f ± %.4f（32 点联合 OLS；95%% CI [%.3f, %.3f]）"
            % (j["q"], j["q_stderr"], j["q_ci95"][0], j["q_ci95"][1]),
            "同上；q 强非可分（切片 %.3f–%.3f），单值声明必须附口径" % (q_lo, q_hi)),
        row("EnKF 降幅", "%.2f%%" % (100.0 * PUBLISHED["enkf_reduction"]),
            "%.2f%%" % (100.0 * REVIEW["enkf_reduction"]), "%.4f%%" % red_ppm,
            "报告值**正确**；评审值由 4 位小数舍入后再推算，属渲染精度伪差，"
            "不可作为勘误依据"),
    ])

    convention = "\n".join([
        "- `p := loglog_slope(n_c, alpha_grid[i_mid, :])`，`i_mid = len(h)//2 = %d` → "
        "**固定冰厚 h = %g m** 的 %d 点切片（n_c 扫描）；"
        % (conv["i_mid"], conv["p_slice_thickness"], c["grid_shape"][1]),
        "- `q := loglog_slope(h_i, alpha_grid[:, j_mid])`，`j_mid = len(n)//2 = %d` → "
        "**固定浓度 n_c = %g** 的 %d 点切片（h_i 扫描）。"
        % (conv["j_mid"], conv["q_slice_concentration"], c["grid_shape"][0]),
    ])

    repro = "\n".join([
        "| 存储值（results.json） | 切片口径复算 | |Δ| |", "|---|---|---|",
        row("p = %.17g" % c["stored"]["p"], "%.17g" % conv["p"],
            "%.2e" % conv["p_abs_err_vs_stored"]),
        row("q = %.17g" % c["stored"]["q"], "%.17g" % conv["q"],
            "%.2e" % conv["q_abs_err_vs_stored"]),
    ])

    global_t = "\n".join([
        "| 口径 | p | q | 说明 |", "|---|---|---|---|",
        row("联合双回归 OLS（建议作全局估计量）", "%.4f" % j["p"], "%.4f" % j["q"],
            "R²(log) = %.4f，rmse(log) = %.4f，n = %d"
            % (j["r2"], j["rmse_log"], j["n_points"])),
        row("单变量 OLS（忽略另一变量）", "%.4f" % j["p_single_regressor"],
            "%.4f" % j["q_single_regressor"], "混淆偏差，不作判据"),
        row("固定冰厚切片 p 区间（%d 条）" % c["grid_shape"][0],
            "%.4f – %.4f" % (p_lo, p_hi), "—", "随 h 单调下降"),
        row("固定浓度切片 q 区间（%d 条）" % c["grid_shape"][1], "—",
            "%.4f – %.4f" % (q_lo, q_hi), "随 n_c 单调下降"),
        row("薄板弯曲理论锚（k⁵ 主导分支）", "%.2f" % th["p"], "%.2f" % th["q"],
            "由 alpha ~ (n_c·h³)^{1/5} 得 p = 1/5、q = 3/5"),
    ])

    non_sep = (
        "切片散布：p 跨 %.3f（%.3f–%.3f）、q 跨 %.3f（%.3f–%.3f）；联合 OLS 的对数残差 "
        "rmse(log) = %.4f，即逐点约 %.0f%% 的偏差。**核定为：alpha 不是 n_c、h 的可分幂律，"
        "不存在单一全局指数**——报告要么声明切片口径下的局部指数，要么引用联合 OLS 并给出"
        "置信区间与非可分性说明。「4 位有效数字 + 未声明口径」是伪精度。"
        % (p_hi - p_lo, p_lo, p_hi, q_hi - q_lo, q_lo, q_hi, j["rmse_log"],
           100.0 * (float(np.exp(j["rmse_log"])) - 1.0)))

    subgrid_note = (
        "报告表格只印出 n_c ∈ {%s} × h_i ∈ {%s} 这 %d 个点（`miz.cross_checks`，与 4×8 网格"
        "同值，最大偏差 %.1e）。评审的 6 个数若真来自该表，则在同一**子网格**上应可逐项复现："
        % (ns, hs, n_sub, sub["printed_alpha_max_abs_diff_vs_grid"]))

    subgrid_t = "\n".join([
        "| 评审量 | 评审值 | 子网格复算 | |Δ| |", "|---|---|---|---|",
        row("联合 OLS p", "%.4f" % REVIEW["p_joint"],
            "%.4f" % ident["p_joint_from_subgrid"], "%.1e" % ident["p_joint_abs_diff"]),
        row("联合 OLS q", "%.4f" % REVIEW["q_joint"],
            "%.4f" % ident["q_joint_from_subgrid"], "%.1e" % ident["q_joint_abs_diff"]),
        row("切片 p（h = %g m）" % sub["ice_thicknesses"][0], "%.4f" % REVIEW["p_slice"][0],
            "%.4f" % ident["p_slices_from_subgrid"][0], "%.1e" % ident["p_slice_abs_diff"][0]),
        row("切片 p（h = %g m）" % sub["ice_thicknesses"][-1], "%.4f" % REVIEW["p_slice"][1],
            "%.4f" % ident["p_slices_from_subgrid"][-1], "%.1e" % ident["p_slice_abs_diff"][1]),
        row("切片 q（n_c = %g）" % sub["concentrations"][0], "%.4f" % REVIEW["q_slice"][0],
            "%.4f" % ident["q_slices_from_subgrid"][0], "%.1e" % ident["q_slice_abs_diff"][0]),
        row("切片 q（n_c = %g）" % sub["concentrations"][-1], "%.4f" % REVIEW["q_slice"][1],
            "%.4f" % ident["q_slices_from_subgrid"][-1], "%.1e" % ident["q_slice_abs_diff"][1]),
    ])

    subgrid_conc = (
        "最大偏差 %.1e（评审只给 4 位小数，故量级即舍入）。→ **评审的 6 个数全部可由"
        "「报告印出的 8 点子网格」逐项复现**：评审没有算错，但它出发的数据集是报告可见的"
        "交叉校核表，而非 4×8 标定网格。因此评审「q 与任何拟合口径均不符」成立却不完整："
        "缺的那个口径正是本代码使用的 n_c = %g 的 %d 点全厚度切片（q = %.4f），"
        "在只有 2 个冰厚的子网格上不可能出现。"
        % (ident["max_abs_diff"], conv["q_slice_concentration"], c["grid_shape"][0], conv["q"]))

    enkf_t = "\n".join([
        "| 量 | 全精度复算 | results.json 存储 | |Δ| |", "|---|---|---|---|",
        row("RMSE_b", "%.15f" % e["rmse_background_recomputed"],
            "%.15f" % e["rmse_background_stored"], "%.1e" % e["rmse_background_abs_err"]),
        row("RMSE_a", "%.15f" % e["rmse_analysis_recomputed"],
            "%.15f" % e["rmse_analysis_stored"], "%.1e" % e["rmse_analysis_abs_err"]),
        row("reduction", "%.12f" % e["reduction_recomputed"],
            "%.12f" % e["reduction_stored"], "%.1e" % e["reduction_abs_err"]),
    ])

    enkf_note = (
        "报告正文印 %.2f%%（对全精度核定值 %.4f%% 的偏差 %s，判 Ok）；同一张表把 RMSE 印到 4 位"
        "小数（%.4f → %.4f）。由这两个**渲染后**的数再推降幅得 %.4f%%，与正文相差 %.3f 个"
        "百分点——评审 %.2f%% 由此而来（对评审值 |Δ| = %.1e）。**核定：%.4f%% 正确，%.2f%% 是"
        "伪差**；但伪差暴露真实缺陷：派生量（百分比）有效位数高于其所依赖的原始量。"
        % (100.0 * PUBLISHED["enkf_reduction"], red_ppm,
           _pct(abs(PUBLISHED["enkf_reduction"] - e["reduction_recomputed"]),
                e["reduction_recomputed"]),
           e["display_4dp"]["rmse_background"], e["display_4dp"]["rmse_analysis"],
           100.0 * e["reduction_from_displayed_4dp"], e["reduction_gap_pp"],
           100.0 * REVIEW["enkf_reduction"], e["reduction_display_abs_diff_vs_review"],
           red_ppm, 100.0 * REVIEW["enkf_reduction"]))

    bp = "\n".join([
        "| 层 | 断点 | 证据 |", "|---|---|---|",
        row("results.json",
            "派生指数只有标量，无切片口径字段（固定的是哪条 h、哪条 n_c 查不到）",
            "`alpha_vs_concentration_exponent` 由 i_mid/j_mid 隐式决定"),
        row("报告渲染",
            "正文只印 %d 点交叉校核表，却用 32 点网格算指数；派生百分比用 4 位小数 RMSE 反推"
            % n_sub,
            "正文 %.2f%% vs 表格 %.4f/%.4f 反推 %.4f%%"
            % (100.0 * e["reduction_stored"], e["display_4dp"]["rmse_background"],
               e["display_4dp"]["rmse_analysis"], 100.0 * e["reduction_from_displayed_4dp"])),
        row("语义",
            "`alpha ~ n_c^p` 被写成全局标定律，实为 h = %g m 单切片局部斜率"
            % conv["p_slice_thickness"],
            "p、q 随切片跨 %.2f–%.2f / %.2f–%.2f" % (p_lo, p_hi, q_lo, q_hi)),
    ])

    fig_list = "\n".join(
        "%d. `%s`" % (i + 1, p.name) for i, p in enumerate(figures))
    fig_list += ("\n\n图 1 给出 p、q 的全部切片指数（含标准误）对照报告值 / 联合 OLS / 评审值 / "
                 "薄板弯曲理论锚；图 2 把 32 点标定网格与报告印出的 8 点、以及 p 所用的 "
                 "h = %g m 切片画在一起；图 3 是 %.4f%% 与 %.4f%% 的舍入链。"
                 % (conv["p_slice_thickness"], red_ppm,
                    100.0 * e["reduction_from_displayed_4dp"]))

    err = "\n".join([
        "| 量 | 基准 | 相对误差 | < 3%? |", "|---|---|---|---|",
        row("存储 p vs 切片复算", "同工件切片定义",
            "%.2e" % (100.0 * conv["p_abs_err_vs_stored"] / abs(conv["p"])), "是（机器零）"),
        row("存储 q vs 切片复算", "同工件切片定义",
            "%.2e" % (100.0 * conv["q_abs_err_vs_stored"] / abs(conv["q"])), "是（机器零）"),
        row("存储 EnKF 统计 vs 全精度复算", "同工件数组",
            "%.2e" % (100.0 * e["reduction_abs_err"] / abs(e["reduction_recomputed"])),
            "是（机器零）"),
        row("评审 %.2f%% vs 核定 %.4f%%" % (100.0 * REVIEW["enkf_reduction"], red_ppm),
            "全精度复算",
            _pct(abs(REVIEW["enkf_reduction"] - e["reduction_recomputed"]),
                 e["reduction_recomputed"]), "**否**（渲染精度伪差）"),
        row("报告 p %.4f vs 联合 OLS %.4f" % (c["stored"]["p"], j["p"]), "32 点联合 OLS",
            _pct(abs(c["stored"]["p"] - j["p"]), j["p"]), "**否**（口径不同，非数值误差）"),
        row("报告 q %.4f vs 联合 OLS %.4f" % (c["stored"]["q"], j["q"]), "32 点联合 OLS",
            _pct(abs(c["stored"]["q"] - j["q"]), j["q"]), "**否**（同上）"),
        row("评审 p %.4f vs 联合 OLS %.4f" % (REVIEW["p_joint"], j["p"]), "32 点联合 OLS",
            _pct(abs(REVIEW["p_joint"] - j["p"]), j["p"]), "**否**（用错数据子集）"),
        row("评审 q %.4f vs 联合 OLS %.4f" % (REVIEW["q_joint"], j["q"]), "32 点联合 OLS",
            _pct(abs(REVIEW["q_joint"] - j["q"]), j["q"]), "**否**（同上）"),
    ])

    err_conc = (
        "**结论**：同工件内的算术一致性全部为机器零（|Δ| ≤ 1e-14，远优于 3%%）——报告的两处"
        "数字与其存储工件严格自洽。跨口径比较（切片 vs 全网格联合 OLS、4 位小数再推算）差异"
        "达 %s–%s，全部源于估计量定义而非计算错误。因此**报告无需改数，但必须补口径声明**；"
        "评审的核心质疑（q 不能以单值充当全局指数；8.65%% 与其表格不自洽）方向正确，"
        "但它给出的替代值（用 8 点子网格算的 0.4011/1.1225、由 4 位小数反推的 8.33%%）"
        "不能直接采信。"
        % (_pct(abs(REVIEW["p_joint"] - j["p"]), j["p"]),
           _pct(abs(c["stored"]["q"] - j["q"]), j["q"])))

    artifacts = "\n".join([
        "- `artifacts/w5n05/results.json`：报告渲染所用原始存储（本次一并入库，供口径复现）；",
        "- `artifacts/w5n05/errata/errata_w5n05.json`：本文件全部数字的机读版本；",
        "- `artifacts/w5n05/errata/errata_fig1_exponent_conventions.png`；",
        "- `artifacts/w5n05/errata/errata_fig2_grid_vs_printed.png`；",
        "- `artifacts/w5n05/errata/errata_fig3_enkf_rounding.png`；",
        "- `examples/ice_wave_w5n05_errata.py`：复算器（`--check` 即回归门）；",
        "- `tests/test_ice_wave_errata_w5n05.py`：口径回归测试（切片复现 / 评审口径识别 / "
        "舍入伪差复现）；",
        "- `src/tensorphysics/ice_wave/miz.py`：`sweep_attenuation` 新增口径字段与联合 OLS"
        "（既有键与数值不变）；",
        "- `examples/ice_wave_w5n05_report.py`：正文改为显式声明切片并同印联合 OLS/CI。",
    ])

    text = TEMPLATE.read_text(encoding="utf-8")
    subs = {
        "SUMMARY_TABLE": summary,
        "ROOT_CAUSE_EXTRA": "",
        "MIZ_PARAMS": "浓度 %g–%g（%d 点）× 冰厚 %g–%g m（%d 点）；周期 %g s；水深 %g m；"
                      "tan δ = %g。"
                      % (c["concentrations"][0], c["concentrations"][-1], c["grid_shape"][1],
                         c["ice_thicknesses"][0], c["ice_thicknesses"][-1],
                         c["grid_shape"][0], c["period"], c["depth"], c["loss_factor"]),
        "ENKF_PARAMS": "状态维 %d，集合成员 %d，观测节点 [0, 4, 8]，伪观测取 DEM 均匀化 μ。"
                       % (e["n_state"], e["n_ensemble"]),
        "SOURCE": "`%s`，mission %s / node %s。"
                  % (adj["source_artifact"], adj["mission"], adj["node"]),
        "CONVENTION_LIST": convention,
        "REPRO_TABLE": repro,
        "GLOBAL_TABLE": global_t,
        "NON_SEPARABILITY": non_sep,
        "SUBGRID_NOTE": subgrid_note,
        "SUBGRID_TABLE": subgrid_t,
        "SUBGRID_CONCLUSION": subgrid_conc,
        "ENKF_TABLE": enkf_t,
        "ENKF_NOTE": enkf_note,
        "BREAKPOINT_TABLE": bp,
        "FIG_LIST": fig_list,
        "CROSS_CHECK": "复波数 Newton 解 vs 谱本征时域衰减的最大相对误差 %.1e（判据 d，"
                       "阈值 0.15），故 p/q 之争纯属**统计口径**，不是物理值错误。"
                       % xchk["miz_alpha_cross_check"],
        "SLICE_DRIFT": "p %.3f–%.3f、q %.3f–%.3f" % (p_lo, p_hi, q_lo, q_hi),
        "ERROR_TABLE": err,
        "ERROR_CONCLUSION": err_conc,
        "ARTIFACTS": artifacts,
    }
    for key, val in subs.items():
        text = text.replace("{{%s}}" % key, val)
    leftovers = [ln for ln in text.splitlines() if "{{" in ln]
    if leftovers:
        raise RuntimeError("unfilled placeholders: %s" % leftovers)
    return text


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W5-N05 errata re-derivation")
    parser.add_argument("--check", action="store_true", help="assert adjudicated values")
    args = parser.parse_args(argv)

    results = json.loads(RESULTS.read_text())
    adj = adjudicate(results)
    figures = make_figures(results, adj, OUT_DIR)
    adj["figures"] = [p.name for p in figures]
    OUT_JSON.write_text(json.dumps(adj, indent=1, ensure_ascii=False))
    OUT_MD.write_text(render_errata(adj, figures), encoding="utf-8")

    c, e, ident = adj["miz_census"], adj["enkf"], adj["review_convention_identification"]
    conv, j = c["code_convention"], c["joint_full_grid"]
    print("stored p/q          : %.17g / %.17g" % (c["stored"]["p"], c["stored"]["q"]))
    print("slice p (h=%g)      : %.17g  |delta|=%.2e"
          % (conv["p_slice_thickness"], conv["p"], conv["p_abs_err_vs_stored"]))
    print("slice q (n_c=%g)    : %.17g  |delta|=%.2e"
          % (conv["q_slice_concentration"], conv["q"], conv["q_abs_err_vs_stored"]))
    print("joint OLS (32 pts)  : p=%.6f +- %.6f   q=%.6f +- %.6f   R2=%.4f rmse_log=%.4f"
          % (j["p"], j["p_stderr"], j["q"], j["q_stderr"], j["r2"], j["rmse_log"]))
    print("p slices (full)     :", ["%g m->%.4f" % (r["fixed_thickness"], r["slope"])
                                     for r in c["p_slices_full_grid"]])
    print("q slices (full)     :", ["n=%g->%.4f" % (r["fixed_concentration"], r["slope"])
                                     for r in c["q_slices_full_grid"]])
    print("review conv. max dev: %.1e (p joint %.4f, q joint %.4f)"
          % (ident["max_abs_diff"], ident["p_joint_from_subgrid"],
             ident["q_joint_from_subgrid"]))
    print("enkf reduction      : %.15f (stored %.15f, |delta|=%.2e)"
          % (e["reduction_recomputed"], e["reduction_stored"], e["reduction_abs_err"]))
    print("reduction from 4dp  : %.6f  (gap %.4f pp)" % (
        e["reduction_from_displayed_4dp"], e["reduction_gap_pp"]))
    print("wrote", OUT_JSON.relative_to(ROOT), OUT_MD.relative_to(ROOT),
          "+", len(figures), "figures")

    if args.check:
        ok = True
        checks = [
            ("slice p reproduces stored p", conv["p_abs_err_vs_stored"], 1e-12),
            ("slice q reproduces stored q", conv["q_abs_err_vs_stored"], 1e-12),
            ("EnKF reduction reproduces stored", e["reduction_abs_err"], 1e-12),
            ("EnKF RMSE_b reproduces stored", e["rmse_background_abs_err"], 1e-12),
            ("EnKF RMSE_a reproduces stored", e["rmse_analysis_abs_err"], 1e-12),
            ("review convention identified (<=5e-4)", ident["max_abs_diff"], 5e-4),
            ("4-dp rounding reproduces 8.33%",
             e["reduction_display_abs_diff_vs_review"], 5e-5),
            ("printed table == alpha grid",
             c["printed_subgrid"]["printed_alpha_max_abs_diff_vs_grid"], 0.0),
        ]
        for name, val, tol in checks:
            flag = "ok" if val <= tol else "FAIL"
            print("  [%s] %-42s %.3e <= %.0e" % (flag, name, val, tol))
            ok = ok and val <= tol
        print("CHECK:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())