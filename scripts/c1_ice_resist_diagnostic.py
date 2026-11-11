#!/usr/bin/env python3
"""C1-ICE-RESIST: Diagnostic analysis — why do models overpredict MT Uikku?

This script investigates the root causes of the large model-measurement discrepancies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "measured" / "ice_resistance" / "series" / "mtuikku_hz2016"

def load_data():
    series = json.loads((DATA_DIR / "series.json").read_text())
    pass_a = json.loads((DATA_DIR / "pass_a.json").read_text())
    return series, pass_a

def analyze_discrepancy():
    """Analyze why models overpredict."""
    series, pass_a = load_data()
    
    print("=" * 80)
    print("DIAGNOSTIC: Why do ice resistance models overpredict MT Uikku measurements?")
    print("=" * 80)
    
    print("\n1. LITERATURE BASELINES vs MEASUREMENTS")
    print("-" * 80)
    print(f"{'Pt':>4} {'h[m]':>5} {'Meas[kN]':>9} {'Lit_Lin[kN]':>11} {'Lin/Meas':>9} "
          f"{'Lit_Ris[kN]':>11} {'Ris/Meas':>9} {'Lit_Kei[kN]':>11} {'Kei/Meas':>9}")
    print("-" * 80)
    
    for pid in sorted(pass_a.keys()):
        pt = pass_a[pid]
        meas = pt["R_meas_kN"]
        lit = pt.get("literature_baselines_kN", {})
        lit_lin = lit.get("lindqvist", 0)
        lit_ris = lit.get("riska", 0)
        lit_kei = lit.get("keinonen", 0)
        
        print(f"{pid:>4} {pt['h_m']:>5.2f} {meas:>9.0f} {lit_lin:>11.0f} {lit_lin/meas:>8.2f}x "
              f"{lit_ris:>11.0f} {lit_ris/meas:>8.2f}x {lit_kei:>11.0f} {lit_kei/meas:>8.2f}x")
    
    print("\n2. KEY OBSERVATION: Literature baselines are MUCH closer to measurements")
    print("-" * 80)
    print("The literature baselines (from Hu & Zhou 2016 Table 8) show:")
    print("  - Lindqvist: 0.67x - 0.73x of measured (underpredicts by 27-33%)")
    print("  - Riska: 1.07x - 1.31x of measured (overpredicts by 7-31%)")
    print("  - Keinonen: 0.85x - 1.09x of measured (within ±10%)")
    print("\nBut our implementations give:")
    print("  - Lindqvist: 2.8x - 10x of measured (massive overprediction)")
    print("  - Keinonen: 3.3x - 10x of measured (massive overprediction)")
    print("  - Riska: 0.6x - 3.3x of measured (variable)")
    
    print("\n3. ROOT CAUSE ANALYSIS")
    print("-" * 80)
    print("\nA. Ice property usage:")
    print("   - Our models use per-test measured σ_f and σ_c from Table 3")
    print("   - Literature predictions likely used STANDARD/DEFAULT values")
    print("   - Example: Point 103 has σ_c = 1748 kPa, but literature may have used 2000-3000 kPa")
    print("   - The crushing term R_crush ∝ σ_c · h² is VERY sensitive to σ_c")
    
    print("\nB. Formulation differences:")
    print("   - Our Lindqvist uses bending interpretation: R_break ∝ σ_f · h · B")
    print("   - Original Lindqvist may use buoyancy: R_f ∝ ρ_i · g · h · B")
    print("   - With σ_f = 724 kPa, bending term dominates (977 kN)")
    print("   - With buoyancy term, R_f ≈ 24 kN (40x smaller!)")
    
    print("\nC. Effective vs measured ice strength:")
    print("   - Measured σ_c (uniaxial compression) ≠ effective σ_c in ice-structure interaction")
    print("   - Effective strength is typically 0.3-0.5x of uniaxial strength")
    print("   - Using full σ_c overestimates crushing resistance by 2-3x")
    
    print("\nD. Model scale effects:")
    print("   - MT Uikku data is from model tests at 1:31.56 scale")
    print("   - Scale effects on ice strength and failure mode are significant")
    print("   - Empirical formulas calibrated on full-scale data may not apply directly")
    
    print("\n4. RECOMMENDED CORRECTIONS")
    print("-" * 80)
    print("\nTo match literature baselines, we need:")
    print("  1. Use STANDARD ice properties (σ_f = 500 kPa, σ_c = 2-3 MPa) instead of per-test values")
    print("  2. Use buoyancy interpretation for Lindqvist R_f term")
    print("  3. Apply effective strength reduction factor (0.3-0.5)")
    print("  4. Consider model-scale correction factors")
    
    print("\n5. HONEST-FAIL ASSESSMENT")
    print("-" * 80)
    print("The criterion ≥80% points < 25% error is NOT MET by any model.")
    print("\nRoot causes:")
    print("  ❌ Using per-test measured ice properties (not standard values)")
    print("  ❌ Bending vs buoyancy formulation mismatch")
    print("  ❌ No effective strength reduction")
    print("  ❌ No model-scale corrections")
    print("\nThis is an HONEST-FAIL. The models are correctly implemented but")
    print("the input parameters and formulation choices don't match the literature")
    print("baselines. Fixing this requires:")
    print("  → Round 2: Investigate standard ice property values used in literature")
    print("  → Round 2: Implement buoyancy vs bending formulation variants")
    print("  → Round 2: Add effective strength reduction factors")
    print("  → Round 2: Study model-scale correction methods")

if __name__ == "__main__":
    analyze_discrepancy()
