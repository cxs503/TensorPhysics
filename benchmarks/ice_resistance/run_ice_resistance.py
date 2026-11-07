#!/usr/bin/env python3
"""Full-spectrum ice-resistance benchmark runner (mission 4911, IN10-C1).

Produces, inside the worktree:

* ``artifacts/vp_curves.json``                      -- PC6 speed-power matrix over
  the whole ice-condition spectrum (level ice / brash ice / ridge field),
* ``artifacts/ice_resistance/validation_report.json`` -- anchor-corpus result,
  error table, domain/negative-control record, uncertainty and provenance,
* ``artifacts/ice_resistance/fig_*.png``            -- report figures.

Run::

    PYTHONPATH=src python benchmarks/ice_resistance/run_ice_resistance.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
for p in (str(REPO / "src"), str(REPO), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from anchors_eval import evaluate_corpus  # noqa: E402
from tensorphysics.ice.resistance import (  # noqa: E402
    PARAMETER_PROVENANCE,
    DomainLimits,
    IceState,
    ShipGeometry,
    coefficient_sensitivity,
    coverage_decay,
    ice_resistance,
    ram_capacity,
    ridge_field_mean_resistance,
)

ARTIFACTS = REPO / "artifacts"
OUTDIR = ARTIFACTS / "ice_resistance"

SHIP = ShipGeometry()
LIMITS = DomainLimits()

#: speed grid, m/s (0.5 m/s ... 5.0 m/s == 1 kn ... 9.7 kn)
SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

#: PC6 ice-condition spectrum: level ice thickness / brash coverage / ridge keel
LEVEL_H = [0.3, 0.5, 0.7, 0.9, 1.2]
BRASH = {"thickness_m": 0.6, "coverages": [0.3, 0.6, 0.9]}
RIDGE = {"keels_m": [1.5, 2.5, 3.5], "spacing_m": 200.0, "background_h_m": 0.3,
         "porosity": 0.30}


def build_conditions() -> List[Dict[str, Any]]:
    conds: List[Dict[str, Any]] = []
    for h in LEVEL_H:
        conds.append(
            {
                "id": f"level_h{h:.2f}",
                "kind": "level",
                "label": f"level ice h={h:.2f} m",
                "state": IceState(kind="level", thickness_m=h),
            }
        )
    for c in BRASH["coverages"]:
        conds.append(
            {
                "id": f"brash_h{BRASH['thickness_m']:.2f}_c{c:.2f}",
                "kind": "brash",
                "label": f"brash ice h={BRASH['thickness_m']:.2f} m, coverage c={c:.2f}",
                "state": IceState(kind="brash", thickness_m=BRASH["thickness_m"], coverage=c),
            }
        )
    for hk in RIDGE["keels_m"]:
        conds.append(
            {
                "id": f"ridge_hk{hk:.2f}",
                "kind": "ridge",
                "label": (
                    f"ridge field keel h_k={hk:.2f} m, spacing {RIDGE['spacing_m']:.0f} m, "
                    f"background level ice {RIDGE['background_h_m']:.2f} m"
                ),
                "state": IceState(
                    kind="ridge",
                    thickness_m=RIDGE["background_h_m"],
                    ridge_keel_m=hk,
                    ridge_spacing_m=RIDGE["spacing_m"],
                    ridge_porosity=RIDGE["porosity"],
                ),
            }
        )
    return conds


def speed_power_table(cond: Dict[str, Any]) -> Dict[str, Any]:
    state: IceState = cond["state"]
    points = []
    for V in SPEEDS:
        r = ice_resistance(state, V, SHIP, limits=LIMITS, strict=False)
        p_mw = r["R_total_N"] * V / SHIP.eta_D / 1e6
        point = {
            "V_ms": round(V, 4),
            "V_kn": round(V / 0.514444, 3),
            "R_total_kN": round(r["R_total_kN"], 2),
            "P_delivered_MW": round(p_mw, 4),
            "domain_ok": r["domain_ok"],
            "flags": r["flags"],
        }
        for k in ("R_crush_N", "R_break_N", "R_submersion_N", "R_keel_N", "R_momentum_N",
                  "R_level_N", "R_floe_impact_N", "R_ridge_instantaneous_N", "R_total_N",
                  "decay_factor", "W_per_ridge_J", "ridge_width_m", "ridge_fraction"):
            if k in r:
                point[k] = round(r[k], 4) if isinstance(r[k], float) else r[k]
        points.append(point)
    state_obj = cond["state"]
    return {
        "id": cond["id"],
        "kind": cond["kind"],
        "label": cond["label"],
        "state": {
            "kind": state_obj.kind,
            "thickness_m": state_obj.thickness_m,
            "coverage": state_obj.coverage,
            "ridge_keel_m": state_obj.ridge_keel_m,
            "ridge_spacing_m": state_obj.ridge_spacing_m,
            "ridge_porosity": state_obj.ridge_porosity,
        },
        "points": points,
        "speed_at_installed_power": _max_speed(state, SHIP.installed_power_MW),
        "power_at_5kn_MW": _power_at(state, 5.0 * 0.514444),
        "power_at_max_grid_speed_MW": points[-1]["P_delivered_MW"],
    }


def _power_at(state: IceState, V: float) -> Dict[str, float]:
    r = ice_resistance(state, V, SHIP, limits=LIMITS, strict=False)
    return {
        "V_kn": round(V / 0.514444, 3),
        "R_total_kN": round(r["R_total_kN"], 2),
        "P_delivered_MW": round(r["R_total_N"] * V / SHIP.eta_D / 1e6, 4),
    }


def _max_speed(state: IceState, power_MW: float) -> Dict[str, Any]:
    v_hi = LIMITS.v_limit(SHIP)
    lo, hi = 1e-3, v_hi

    def pw(v: float) -> float:
        r = ice_resistance(state, v, SHIP, limits=LIMITS, strict=False)
        return r["R_total_N"] * v / SHIP.eta_D / 1e6

    if pw(hi) <= power_MW:
        return {"V_ms": round(hi, 4), "V_kn": round(hi / 0.514444, 3),
                "power_MW": round(pw(hi), 4), "capped_by_validity_domain": True}
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if pw(mid) < power_MW:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    v = 0.5 * (lo + hi)
    return {"V_ms": round(v, 4), "V_kn": round(v / 0.514444, 3),
            "power_MW": round(pw(v), 4), "capped_by_validity_domain": False}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _shrink(path: Path, colors: int = 96) -> Path:
    """Palette-quantise a PNG so the figure fits the report payload budget."""

    try:
        from PIL import Image
    except Exception:  # pragma: no cover - PIL always present in this env
        return path
    img = Image.open(path).convert("RGB")
    img = img.convert("P", palette=Image.ADAPTIVE, colors=colors)
    img.save(path, optimize=True)
    return path


def make_figures(tables: List[Dict[str, Any]]) -> List[Path]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    figs: List[Path] = []

    # --- fig 1: resistance vs speed, whole spectrum -----------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = plt.cm.viridis
    for i, t in enumerate(tables):
        v = [p["V_kn"] for p in t["points"]]
        r = [p["R_total_kN"] for p in t["points"]]
        style = "--" if t["kind"] == "brash" else ("-." if t["kind"] == "ridge" else "-")
        ax.plot(v, r, style, color=colors(i / max(1, len(tables) - 1)), lw=1.6, label=t["label"])
    ax.set_xlabel("ship speed V [kn]")
    ax.set_ylabel("ice resistance R [kN]")
    ax.set_title("PC6 reference ship: full ice-condition spectrum (three-state family)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)
    p1 = OUTDIR / "fig1_resistance_spectrum.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    figs.append(_shrink(p1))

    # --- fig 2: delivered power vs speed ---------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, t in enumerate(tables):
        v = [p["V_kn"] for p in t["points"]]
        pw = [p["P_delivered_MW"] for p in t["points"]]
        style = "--" if t["kind"] == "brash" else ("-." if t["kind"] == "ridge" else "-")
        ax.plot(v, pw, style, color=colors(i / max(1, len(tables) - 1)), lw=1.6, label=t["label"])
    ax.axhline(SHIP.installed_power_MW, color="crimson", lw=1.2,
               label=f"installed delivered power {SHIP.installed_power_MW:.0f} MW")
    ax.set_xlabel("ship speed V [kn]")
    ax.set_ylabel("delivered power $P_D$ [MW]  ($\\eta_D$=%.2f)" % SHIP.eta_D)
    ax.set_title("Speed-power curves per ice state (power = R·V/$\\eta_D$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)
    p2 = OUTDIR / "fig2_power_spectrum.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    figs.append(_shrink(p2))

    # --- fig 3: coverage decay law ---------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    cs = [i / 200 for i in range(201)]
    d = [coverage_decay(c)["D"] for c in cs]
    ax1.plot(cs, d, "b-", lw=1.8, label="D(c) decay law")
    ax1.plot(cs, cs, "k:", lw=1.0, label="clearing limit D=c")
    ax1.axvline(0.6, color="crimson", ls="--", lw=1.0, label="$c_{crit}$=0.6")
    ax1.set_xlabel("ice coverage c [-]")
    ax1.set_ylabel("decay factor D(c)")
    ax1.set_title("brash-ice coverage decay law")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)
    r_c = [ice_resistance(IceState(kind="brash", thickness_m=0.6, coverage=c), 2.0, SHIP)
           ["R_total_kN"] for c in cs]
    ax2.plot(cs, r_c, "g-", lw=1.8)
    ax2.axhline(ice_resistance(IceState(kind="level", thickness_m=0.6), 2.0, SHIP)["R_total_kN"],
                color="k", ls=":", lw=1.0, label="intact sheet h=0.6 m")
    ax2.set_xlabel("ice coverage c [-]")
    ax2.set_ylabel("R [kN] at V=2 m/s, h=0.6 m")
    ax2.set_title("brash resistance vs coverage")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    p3 = OUTDIR / "fig3_coverage_decay.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=110)
    plt.close(fig)
    figs.append(_shrink(p3))

    return figs


def make_audit_figure(audit: Dict[str, Any]) -> Path:
    labels = ["benchmark\nmodel", "Keinonen 1983", "Enkvist 1972",
              "legacy Lindqvist\n(ice_anchor A1)", "legacy Riska\n(ice_anchor A5)"]
    vals = [audit["benchmark_model_kN"], audit["keinonen_kN"], audit["enkvist_kN"],
            audit["legacy_lindqvist_kN"], audit["legacy_riska_kN"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(labels, vals, color=["#2b7bba", "#7f7f7f", "#7f7f7f", "#c0392b", "#c0392b"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(audit["reference_mean_kN"] * 1.25, color="orange", ls="--", lw=1.0,
               label="±25% cross-law band")
    ax.axhline(audit["reference_mean_kN"] * 0.75, color="orange", ls="--", lw=1.0)
    ax.set_ylabel("level-ice resistance at h=1 m, V=3 m/s [kN]")
    ax.set_title("audit: legacy stem-angle convention vs corrected family")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "fig4_audit_deviation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _shrink(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tables = [speed_power_table(c) for c in build_conditions()]
    corpus = evaluate_corpus()

    vp = {
        "artifact": "vp_curves",
        "mission_id": 4911,
        "node": "IN10-C1",
        "generated_utc": now,
        "ship": SHIP.as_dict(),
        "model_family": {
            "level": "Lindqvist (1989) three components R_crush + R_break + R_submersion",
            "brash": "coverage decay law D(c): D=c below c_crit=0.6, packing closure above",
            "ridge": "Keinonen-anchored buoyancy/displacement + rubble momentum, "
                     "field-averaged by ridge spacing (energy method)",
            "module": "src/tensorphysics/ice/resistance.py",
        },
        "evidence": {
            "implementation": "REAL (closed-form, component-resolved, no fitted surrogate)",
            "coefficients": "LITERATURE / SEMI-EMPIRICAL (see parameter_provenance)",
            "accuracy_vs_full_scale": "WITHHELD (no digitised full-scale record in repo)",
        },
        "parameter_provenance": PARAMETER_PROVENANCE,
        "domain_limits": LIMITS.as_dict(SHIP),
        "speed_grid_ms": SPEEDS,
        "conditions": tables,
        "ram_capacity": [ram_capacity(SHIP, v) for v in (1.0, 2.0, 3.0)],
        "uncertainty": {
            "coefficient_sensitivity_at_h1m_V3ms": coefficient_sensitivity(1.0, 3.0, SHIP),
            "note": "±50% on each closure constant; the crushing/displacement terms dominate, "
                    "so R_total responds roughly linearly to sigma_c and to the flare angle.",
        },
        "acceptance": {
            "n_reference_anchors": corpus["n_reference_anchors"],
            "n_reference_anchors_passed": corpus["n_reference_anchors_passed"],
            "n_negative_controls": corpus["n_negative_controls"],
            "n_negative_controls_passed": corpus["n_negative_controls_passed"],
            "all_anchors_passed": corpus["all_passed"],
            "criteria": {
                "reference_anchors": ">= 10 all passed",
                "negative_controls": "all out-of-domain requests rejected / truncated+flagged",
                "artifact": "artifacts/vp_curves.json",
                "pytest": "pytest tests/test_ice_resistance.py",
            },
            "vp_table_points": sum(len(t["points"]) for t in tables),
        },
    }
    vp_path = ARTIFACTS / "vp_curves.json"
    vp_path.write_text(json.dumps(vp, ensure_ascii=False, indent=2), encoding="utf-8")

    figs = make_figures(tables)
    figs.append(make_audit_figure(corpus["audit"]))

    report = {
        "mission_id": 4911,
        "node": "IN10-C1",
        "generated_utc": now,
        "benchmark": "benchmarks/ice_resistance",
        "artifact": str(vp_path.relative_to(REPO)),
        "evidence_level": corpus["evidence_level"],
        "anchors": corpus,
        "error_table": _error_table(corpus, tables),
        "figures": [f"artifacts/ice_resistance/{p.name}" for p in figs],
        "domain_and_negative_controls": corpus["negative_controls"],
        "limitations": [
            "绝对精度 vs 实尺度测量 = WITHHELD：仓内无数字化实尺度冰阻力记录；"
            "本报告只给出解析恒等式、已发表定律标度关系、跨定律一致性与推进功率量级带。",
            "Lindqvist 三分量中压溃项与船体宽度无关（σ_c h² 结构），这是原式的特性；"
            "跨定律一致区间接吸收了这一点。",
            "k_bend 闭合常数按跨定律量级标定（非实验标定），其 ±50% 影响已在 uncertainty 中给出。",
            "冰脊项为半静态穿透阻力，动态冲撞峰值可比其高 2-4 倍；能量法 ram 容量忽略推力限制，"
            "因此 ram 容量全部被适用域上限封顶（能量 12-104 MJ 远大于准静态穿脊功 0.2-1.0 MJ），"
            "真实冲撞受推力/动态堆积压力限制，本模型不覆盖。",
            "冰脊场平均阻力按航程加权，可能低于同背景厚度的平整冰阻力（准静态口径）；"
            "冰脊的真实危害是局部峰值载荷与卡船，属峰值/动力问题，不在本阻力谱口径内。",
            "碎冰衰减律的 c_crit=0.6 与堆积闭合系数 A=1 为工程值，无本仓数据标定。",
        ],
    }
    rep_path = OUTDIR / "validation_report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "vp_curves": str(vp_path),
        "validation_report": str(rep_path),
        "anchors": f"{corpus['n_reference_anchors_passed']}/{corpus['n_reference_anchors']}",
        "negative_controls": f"{corpus['n_negative_controls_passed']}/{corpus['n_negative_controls']}",
        "all_passed": corpus["all_passed"],
        "figures": [p.name for p in figs],
        "vp_points": vp["acceptance"]["vp_table_points"],
    }, ensure_ascii=False, indent=2))
    return 0 if corpus["all_passed"] else 1


def _error_table(corpus: Dict[str, Any], tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for a in corpus["anchors"]:
        rows.append({
            "id": a["id"],
            "kind": a["kind"],
            "tolerance": a["tolerance"],
            "error": a["error"] if isinstance(a["error"], (int, float)) else None,
            "passed": a["passed"],
        })
    audit = corpus["audit"]
    rows.append({
        "id": "AUDIT-legacy-lindqvist",
        "kind": "audit_finding",
        "tolerance": None,
        "error": audit["legacy_lindqvist_rel_dev"],
        "passed": None,
        "note": "legacy ice_anchor A1 vs cross-law mean (defect, not an anchor)",
    })
    rows.append({
        "id": "AUDIT-benchmark-vs-crosslaw",
        "kind": "cross_law",
        "tolerance": 0.25,
        "error": audit["benchmark_rel_dev"],
        "passed": abs(audit["benchmark_rel_dev"]) <= 0.25,
    })
    return rows


if __name__ == "__main__":
    raise SystemExit(main())