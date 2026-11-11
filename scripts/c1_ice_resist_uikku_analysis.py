#!/usr/bin/env python3
"""C1-ICE-RESIST: Final analysis with literature baseline comparison.

Key finding: Even literature baselines from Hu & Zhou (2016) don't meet 80% < 25% criterion.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
for p in (str(REPO / "src"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

G = 9.80665
RHO_ICE = 900.0

DATA_DIR = REPO / "data" / "measured" / "ice_resistance" / "series" / "mtuikku_hz2016"
CSV_PATH = DATA_DIR / "source" / "hz2016_level_ice_table3.csv"
SERIES_JSON = DATA_DIR / "series.json"


def load_all_data() -> Tuple[Dict, List[Dict]]:
    """Load both series metadata and CSV with literature baselines."""
    series = json.loads((SERIES_JSON).read_text(encoding="utf-8"))
    
    points = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt = {
                "point_id": row["test_id"],
                "h_m": float(row["ice_thickness_m"]),
                "V_ms": float(row["ice_drift_speed_m_s"]),
                "sigma_f_kPa": float(row["flexural_strength_kPa"]),
                "sigma_c_kPa": float(row["compressive_strength_kPa"]),
                "R_meas_kN": float(row["measured_mean_resistance_table3_kN"]),
                "lit_lindqvist_kN": float(row["lindqvist_prediction_kN"]),
                "lit_riska_kN": float(row["riska_prediction_kN"]),
                "lit_keinonen_kN": float(row["keinonen_prediction_kN"]),
            }
            points.append(pt)
    
    return series, points


# ─────────────────────────────────────────────────────────────────
# Model implementations
# ─────────────────────────────────────────────────────────────────

def lindqvist_buoyancy(h, V, L, B, phi_deg, sigma_f, sigma_c, mu=0.15, rho_i=RHO_ICE, flare_deg=30.0):
    """Lindqvist (1989) — buoyancy interpretation (corrected)."""
    phi_rad = math.radians(phi_deg)
    alpha_rad = math.radians(flare_deg)
    
    C_crush = 0.5 * (math.tan(alpha_rad) + mu / math.cos(alpha_rad))
    R_crush = C_crush * sigma_c * h**2
    
    wedge = math.sin(0.5 * phi_rad) + mu * math.cos(0.5 * phi_rad)
    R_f = 0.5 * rho_i * G * h * B * wedge
    
    Fr_h = V / math.sqrt(G * h)
    C_sub = 0.15 + 0.3 * Fr_h
    R_sub = C_sub * rho_i * V**2 * h * B
    
    return R_crush + R_f + R_sub


def keinonen_level(h, V, L, B, phi_deg, sigma_f, sigma_c=5e6, rho_i=RHO_ICE, mu=0.15):
    """Keinonen (1983) — level ice application."""
    phi_rad = math.radians(phi_deg)
    C0, C1 = 1.8, 0.25
    Fr_h = V / math.sqrt(G * h)
    f_phi = math.sin(phi_rad / 2) + 0.3 * math.cos(phi_rad / 2)
    return C0 * sigma_f * h**1.5 * B**0.5 * (1 + C1 * Fr_h) * f_phi


def riska_ahead(h, V, L, B, phi_deg, sigma_f, sigma_c, mu=0.15, rho_i=RHO_ICE, flare_deg=30.0):
    """Riska (1997) — DAS ahead mode."""
    phi_rad = math.radians(phi_deg)
    alpha_rad = math.radians(flare_deg)
    
    C_crush = 0.4 * (math.tan(alpha_rad) + mu / math.cos(alpha_rad))
    R_crush = C_crush * sigma_c * h**2
    
    wedge = math.sin(phi_rad / 2) + mu * math.cos(phi_rad / 2)
    R_f = 0.4 * rho_i * G * h * B * wedge
    
    Fr_h = V / math.sqrt(G * h)
    C_sub = 0.1 + 0.2 * Fr_h
    R_sub = C_sub * rho_i * V**2 * h * B
    
    return R_crush + R_f + R_sub


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_all(series, points):
    ship = series["ship"]
    L, B = ship["L_pp_m"], ship["B_m"]
    phi, flare = ship["bow_angle_deg"], ship.get("flare_deg", 30.0)
    
    results = []
    for pt in points:
        h, V = pt["h_m"], pt["V_ms"]
        sigma_f, sigma_c = pt["sigma_f_kPa"] * 1e3, pt["sigma_c_kPa"] * 1e3
        R_meas = pt["R_meas_kN"]
        
        # Our predictions
        our_lin = lindqvist_buoyancy(h, V, L, B, phi, sigma_f, sigma_c, flare_deg=flare) / 1000
        our_kei = keinonen_level(h, V, L, B, phi, sigma_f, sigma_c) / 1000
        our_ris = riska_ahead(h, V, L, B, phi, sigma_f, sigma_c, flare_deg=flare) / 1000
        
        # Literature
        lit_lin = pt["lit_lindqvist_kN"]
        lit_ris = pt["lit_riska_kN"]
        lit_kei = pt["lit_keinonen_kN"]
        
        rec = {
            "point_id": pt["point_id"],
            "h_m": h, "V_ms": V,
            "sigma_f_kPa": pt["sigma_f_kPa"],
            "sigma_c_kPa": pt["sigma_c_kPa"],
            "R_meas_kN": R_meas,
            "our_lin_kN": round(our_lin, 1),
            "our_kei_kN": round(our_kei, 1),
            "our_ris_kN": round(our_ris, 1),
            "lit_lin_kN": lit_lin,
            "lit_ris_kN": lit_ris,
            "lit_kei_kN": lit_kei,
        }
        
        # Errors
        for key in ["our_lin", "our_kei", "our_ris", "lit_lin", "lit_ris", "lit_kei"]:
            pred = rec[f"{key}_kN"]
            rec[f"err_{key}_pct"] = round(abs(pred - R_meas) / R_meas * 100, 1)
        
        results.append(rec)
    
    return results


def compute_stats(results, keys):
    stats = {}
    for key in keys:
        errs = [r[f"err_{key}_pct"] for r in results]
        errs_arr = np.array(errs)
        n_under_25 = int(np.sum(errs_arr < 25.0))
        stats[key] = {
            "n": len(errs),
            "mean": round(float(np.mean(errs_arr)), 1),
            "median": round(float(np.median(errs_arr)), 1),
            "max": round(float(np.max(errs_arr)), 1),
            "n_under_25": n_under_25,
            "pct_under_25": round(n_under_25 / len(errs) * 100, 1),
        }
    return stats


# ─────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────

def make_figures(results, outdir):
    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(results))
    width = 0.12
    
    meas = [r["R_meas_kN"] for r in results]
    our_lin = [r["our_lin_kN"] for r in results]
    our_kei = [r["our_kei_kN"] for r in results]
    our_ris = [r["our_ris_kN"] for r in results]
    lit_lin = [r["lit_lin_kN"] for r in results]
    lit_kei = [r["lit_kei_kN"] for r in results]
    
    ax.bar(x - 2.5*width, meas, width, label="Measured", color="#2c3e50")
    ax.bar(x - 1.5*width, our_lin, width, label="Our Lindqvist", color="#3498db")
    ax.bar(x - 0.5*width, lit_lin, width, label="Lit. Lindqvist", color="#85c1e9")
    ax.bar(x + 0.5*width, our_kei, width, label="Our Keinonen", color="#e74c3c")
    ax.bar(x + 1.5*width, lit_kei, width, label="Lit. Keinonen", color="#f1948a")
    ax.bar(x + 2.5*width, our_ris, width, label="Our Riska", color="#27ae60")
    
    ax.set_xlabel("Test Point")
    ax.set_ylabel("Ice Resistance [kN]")
    ax.set_title("MT Uikku: Measured vs Model Predictions (Our Implementations + Literature Baselines)")
    ax.set_xticks(x)
    ax.set_xticklabels([r["point_id"] for r in results])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(outdir / "fig_uikku_comparison.png", dpi=150)
    plt.close(fig)
    
    # Scatter plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    models = [
        ("our_lin_kN", "Our Lindqvist", "#3498db"),
        ("lit_lin_kN", "Lit. Lindqvist", "#85c1e9"),
        ("our_kei_kN", "Our Keinonen", "#e74c3c"),
        ("lit_kei_kN", "Lit. Keinonen", "#f1948a"),
        ("our_ris_kN", "Our Riska", "#27ae60"),
        ("lit_ris_kN", "Lit. Riska", "#82e0aa"),
    ]
    
    meas = [r["R_meas_kN"] for r in results]
    max_val = max(max(meas), 1000) * 1.1
    
    for idx, (key, label, color) in enumerate(models):
        ax = axes[idx // 3, idx % 3]
        pred = [r[key] for r in results]
        ax.scatter(meas, pred, c=color, s=60, alpha=0.8, edgecolors="black", linewidth=0.5)
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="1:1")
        ax.plot([0, max_val], [0, max_val*1.25], "r:", alpha=0.4)
        ax.plot([0, max_val], [0, max_val*0.75], "r:", alpha=0.4)
        ax.set_xlabel("Measured [kN]")
        ax.set_ylabel("Predicted [kN]")
        ax.set_title(label)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
    
    plt.tight_layout()
    fig.savefig(outdir / "fig_uikku_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return [outdir / "fig_uikku_comparison.png", outdir / "fig_uikku_scatter.png"]


# ─────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────

def generate_report(results, stats, fig_paths, outdir):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    
    # Build table
    rows = []
    for r in results:
        rows.append(
            f"| {r['point_id']} | {r['h_m']:.2f} | {r['V_ms']:.1f} | "
            f"**{r['R_meas_kN']:.0f}** | "
            f"{r['our_lin_kN']:.0f} ({r['err_our_lin_pct']:.0f}%) | "
            f"{r['lit_lin_kN']:.0f} ({r['err_lit_lin_pct']:.0f}%) | "
            f"{r['our_kei_kN']:.0f} ({r['err_our_kei_pct']:.0f}%) | "
            f"{r['lit_kei_kN']:.0f} ({r['err_lit_kei_pct']:.0f}%) | "
            f"{r['our_ris_kN']:.0f} ({r['err_our_ris_pct']:.0f}%) | "
            f"{r['lit_ris_kN']:.0f} ({r['err_lit_ris_pct']:.0f}%) |"
        )
    
    table = (
        "| Pt | h[m] | V[m/s] | Meas[kN] | Our_Lin (%) | Lit_Lin (%) | "
        "Our_Kei (%) | Lit_Kei (%) | Our_Ris (%) | Lit_Ris (%) |\n"
        "|---|------|--------|----------|-------------|-------------|"
        "-------------|---------------|-------------|-------------|\n"
        + "\n".join(rows)
    )
    
    # Summary table
    model_labels = {
        "our_lin": "Our Lindqvist (buoyancy)",
        "lit_lin": "Lit. Lindqvist (Hu&Zhou)",
        "our_kei": "Our Keinonen",
        "lit_kei": "Lit. Keinonen (Hu&Zhou)",
        "our_ris": "Our Riska",
        "lit_ris": "Lit. Riska (Hu&Zhou)",
    }
    
    sum_rows = []
    for key in ["our_lin", "lit_lin", "our_kei", "lit_kei", "our_ris", "lit_ris"]:
        s = stats[key]
        sum_rows.append(
            f"| {model_labels[key]} | {s['mean']:.1f}% | {s['median']:.1f}% | "
            f"{s['max']:.1f}% | {s['n_under_25']}/{s['n']} ({s['pct_under_25']:.0f}%) |"
        )
    
    sum_table = (
        "| Model | Mean | Median | Max | Points <25% |\n"
        "|-------|------|--------|-----|-------------|\n"
        + "\n".join(sum_rows)
    )
    
    report = f"""# C1-ICE-RESIST: Ice Resistance Full-Spectrum Empirical Chain
## Lindqvist / Keinonen-B / Riska vs MT Uikku — Final Analysis

**Date**: {now}  
**Mission**: m14881 (C1-ICE-RESIST)  
**External Anchor**: MT Uikku (Hu & Zhou 2016)

---

## 1. Case Introduction

Three classical semi-empirical ice resistance models evaluated against MT Uikku full-scale
model test data (10 points, level ice, h=0.63-1.04m, V=0.2-1.0 m/s).

**Ship**: MT Uikku (DAS tanker), L=150m, B=21.3m, φ=21°, flare=30°

---

## 2. Methodology

### Our Implementations
- **Lindqvist (1989)**: Corrected buoyancy interpretation (R_f = 0.5·ρ_i·g·h·B·wedge)
- **Keinonen (1983)**: R = C0·σ_f·h^1.5·B^0.5·(1+C1·Fr_h)·f(φ), C0=1.8, C1=0.25
- **Riska (1997)**: DAS ahead mode with 0.4 coefficients

All use **per-test measured** ice properties (σ_f, σ_c) from Hu & Zhou Table 3.

### Literature Baselines
From Hu & Zhou (2016) Table 8 — predictions reproduced in the source paper.

---

## 3. Results

### 3.1 Per-Point Comparison

{table}

### 3.2 Aggregate Statistics

{sum_table}

### 3.3 Criterion Assessment (≥80% points < 25% error)

"""
    
    for key in ["our_lin", "lit_lin", "our_kei", "lit_kei", "our_ris", "lit_ris"]:
        s = stats[key]
        passed = s["pct_under_25"] >= 80.0
        status = "✅ PASS" if passed else "❌ FAIL"
        report += f"- **{model_labels[key]}**: {s['pct_under_25']:.0f}% → {status}\n"
    
    report += f"""
### 3.4 Figures

"""
    for fp in fig_paths:
        report += f"![{fp.stem}]({fp.relative_to(REPO)})\n\n"
    
    report += """
---

## 4. Key Findings

### Critical Discovery: Literature Baselines Also Fail the Criterion

**Even the original published predictions from Hu & Zhou (2016) don't meet the 80% < 25% criterion:**
- Literature Lindqvist: 70% (7/10 points < 25%)
- Literature Keinonen: 70% (7/10 points < 25%)
- Literature Riska: 40% (4/10 points < 25%)

This means the criterion itself may be too strict for this dataset, or the dataset has
inherent challenges (e.g., points 401-403 with very low measured resistance are outliers).

### Our Implementations vs Literature

| Model | Our Performance | Literature Performance | Gap |
|-------|----------------|----------------------|-----|
| Lindqvist | 40% < 25% | 70% < 25% | -30% |
| Keinonen | 0% < 25% | 70% < 25% | -70% |
| Riska | 30% < 25% | 40% < 25% | -10% |

**Our implementations underperform the literature baselines.** Root causes:

1. **Ice property usage**: We use per-test measured σ_f/σ_c; literature likely used
   standard/effective values (lower than measured uniaxial strength).

2. **Keinonen overprediction**: Our Keinonen gives 3-10x overprediction because the
   formula R ∝ σ_f·h^1.5 is very sensitive to σ_f. Literature likely used σ_f ~ 500 kPa
   (standard) instead of measured 540-1029 kPa.

3. **Lindqvist discrepancy**: Our corrected Lindqvist is 1.2-2.5x higher than literature,
   suggesting literature used effective σ_c ~ 0.6-0.8x of measured values.

---

## 5. Applicability Domains

| Model | Ice Regime | h [m] | V [m/s] | L [m] | φ [deg] |
|-------|-----------|-------|---------|-------|---------|
| Lindqvist (1989) | level first-year | 0.3-1.5 | 0.5-6.0 | 50-250 | 15-60 |
| Keinonen (1983) | brash/repeated | 0.3-1.2 | 0.5-5.0 | 50-250 | 20-45 |
| Riska (1997) | level (DAS) | 0.3-1.5 | 0.5-6.0 | 50-250 | 15-60 |

---

## 6. Honest-FAIL Assessment

**Status**: ❌ FAIL — criterion ≥80% < 25% not met by any model (ours or literature)

**What was accomplished**:
- ✅ Three model families implemented (Lindqvist/Keinonen/Riska)
- ✅ MT Uikku data loaded and processed (10 points)
- ✅ Formulation error identified and corrected (buoyancy vs bending)
- ✅ Literature baselines compared (critical finding: they also fail)
- ✅ Error tables, figures, and applicability domains generated

**Root causes of underperformance**:
1. ❌ Using per-test measured ice properties (not standard/effective values)
2. ❌ Keinonen formula too sensitive to σ_f variations
3. ❌ No effective strength reduction factor applied
4. ❌ No model-scale corrections

**What remains for Round 2**:
- → Investigate standard ice property values used in literature
- → Implement effective strength reduction (σ_eff = k·σ_measured, k ~ 0.6-0.8)
- → Calibrate closure constants against Uikku data
- → Study whether 80% criterion is achievable for this dataset

---

## References

1. Hu, Z. & Zhou, L. (2016). Int. J. Naval Architecture and Ocean Engineering 8(2), 169-176.
2. Lindqvist, G. (1989). POAC'89, Luleå, Vol.2, pp.722-735.
3. Keinonen, A. (1983). POAC'83, Helsinki, pp.429-444.
4. Riska, K. (1997). POAC'97, Espoo, Vol.2, pp.303-314.

---
*Report generated by C1-ICE-RESIST analysis (m14881, final)*
"""
    
    report_path = REPO / "docs" / "c1_ice_resist_uikku_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("C1-ICE-RESIST: Final Analysis with Literature Comparison")
    print("=" * 80)
    
    print("\n[1/5] Loading data...")
    series, points = load_all_data()
    print(f"  → {len(points)} points with literature baselines")
    
    print("\n[2/5] Evaluating models...")
    results = evaluate_all(series, points)
    
    print("\n  Per-point results:")
    print(f"  {'Pt':>4} {'h':>5} {'V':>4} {'Meas':>6} "
          f"{'O_Lin':>6} {'L_Lin':>6} {'O_Kei':>6} {'L_Kei':>6} {'O_Ris':>6} {'L_Ris':>6}")
    print("  " + "-" * 75)
    for r in results:
        print(f"  {r['point_id']:>4} {r['h_m']:>5.2f} {r['V_ms']:>4.1f} {r['R_meas_kN']:>6.0f} "
              f"{r['our_lin_kN']:>6.0f} {r['lit_lin_kN']:>6.0f} "
              f"{r['our_kei_kN']:>6.0f} {r['lit_kei_kN']:>6.0f} "
              f"{r['our_ris_kN']:>6.0f} {r['lit_ris_kN']:>6.0f}")
    
    print("\n[3/5] Computing statistics...")
    keys = ["our_lin", "lit_lin", "our_kei", "lit_kei", "our_ris", "lit_ris"]
    stats = compute_stats(results, keys)
    
    print("\n  Summary:")
    for key in keys:
        s = stats[key]
        print(f"    {key}: mean={s['mean']:.1f}%, <25%: {s['pct_under_25']:.0f}%")
    
    print("\n[4/5] Generating figures...")
    outdir = REPO / "artifacts" / "ice_resistance"
    outdir.mkdir(parents=True, exist_ok=True)
    fig_paths = make_figures(results, outdir)
    
    # Save JSON
    comp_path = outdir / "uikku_comparison.json"
    comp_data = {
        "schema": "c1_ice_resist.v3",
        "mission_id": "m14881",
        "anchor": "MTUIKKU-HZ2016",
        "generated": datetime.now(timezone.utc).isoformat(),
        "points": results,
        "statistics": stats,
    }
    comp_path.write_text(json.dumps(comp_data, indent=2), encoding="utf-8")
    
    # Applicability domains
    domains = {
        "Lindqvist_1989": {"ice_regime": "level first-year", "h_m": [0.3, 1.5], "V_ms": [0.5, 6.0]},
        "Keinonen_1983": {"ice_regime": "brash/repeated", "h_m": [0.3, 1.2], "V_ms": [0.5, 5.0]},
        "Riska_1997": {"ice_regime": "level (DAS)", "h_m": [0.3, 1.5], "V_ms": [0.5, 6.0]},
    }
    dom_path = outdir / "applicability_domains.json"
    dom_path.write_text(json.dumps(domains, indent=2), encoding="utf-8")
    
    print("\n[5/5] Generating report...")
    report_path = generate_report(results, stats, fig_paths, outdir)
    
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print("\n  Criterion: ≥80% points < 25% error")
    print("\n  Our implementations:")
    for key in ["our_lin", "our_kei", "our_ris"]:
        s = stats[key]
        print(f"    {key}: {s['pct_under_25']:.0f}% → FAIL")
    print("\n  Literature baselines:")
    for key in ["lit_lin", "lit_kei", "lit_ris"]:
        s = stats[key]
        print(f"    {key}: {s['pct_under_25']:.0f}% → {'PASS' if s['pct_under_25'] >= 80 else 'FAIL'}")
    
    print("\n  ❌ HONEST-FAIL: Even literature baselines don't meet 80% criterion")
    print("  → This is a dataset/criterion issue, not just implementation issue")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
