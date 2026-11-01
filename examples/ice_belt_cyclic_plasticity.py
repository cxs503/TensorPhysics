#!/usr/bin/env python3
"""End-to-end example: Ice-belt plating cyclic plasticity analysis.

Demonstrates:
1. M-κ relations (EPP and kinematic hardening)
2. Frame strip response under ISO 19906 ice pressure
3. Melan shakedown bound
4. Incremental cyclic analysis
5. Regime map generation

Run: python examples/ice_belt_cyclic_plasticity.py
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorphysics.ice_plasticity.plastic_section import (
    RectangularSection, EPPMaterial, KinematicHardeningMaterial,
    epp_mk_closed_form, epp_mk_numerical, verify_mk_accuracy,
    two_bar_shakedown_limit
)
from tensorphysics.ice_plasticity.iso19906 import (
    ISO19906Pressure, CyclicIcePressure, typical_ice_belt_params
)
from tensorphysics.ice_plasticity.frame_strip import (
    BeamStrip, PlateStrip, compute_frame_strip_response
)
from tensorphysics.ice_plasticity.shakedown_melan import (
    MelanShakedownBeam, two_bar_melan_shakedown, verify_two_bar_shakedown
)
from tensorphysics.ice_plasticity.incremental_cyclic import (
    KinematicHardening1D, run_cyclic_analysis, classify_regime
)
from tensorphysics.ice_plasticity.regime_map import (
    RegimeMapGenerator, generate_allowable_pressure_map,
    generate_regime_map_with_melan
)


def main():
    print("=" * 70)
    print("Ice-Belt Plating Cyclic Plasticity Analysis")
    print("=" * 70)

    # ============================================================
    # 1. Material and Section Properties
    # ============================================================
    print("\n1. Material and Section Properties")
    print("-" * 40)

    params = typical_ice_belt_params()
    sec = RectangularSection(b=params["stiffener_spacing"], h=params["plate_thickness"])
    mat = EPPMaterial(E=params["E"], sigma_y=params["sigma_y"])

    print(f"  Section: b={sec.b*1000:.0f}mm, h={sec.h*1000:.1f}mm")
    print(f"  Material: E={mat.E/1e9:.0f}GPa, σy={mat.sigma_y/1e6:.0f}MPa")
    print(f"  I = {sec.I:.6e} m⁴")
    print(f"  Z_el = {sec.Z_el:.6e} m³")
    print(f"  Z_pl = {sec.Z_pl:.6e} m³")
    print(f"  Shape factor = {sec.shape_factor:.4f}")

    # ============================================================
    # 2. M-κ Relations
    # ============================================================
    print("\n2. M-κ Relations")
    print("-" * 40)

    kappa_y = 2.0 * mat.sigma_y / (mat.E * sec.h)
    Me = mat.sigma_y * sec.Z_el
    Mp = mat.sigma_y * sec.Z_pl

    print(f"  κy = {kappa_y:.6e} 1/m")
    print(f"  Me = {Me:.2f} N·m")
    print(f"  Mp = {Mp:.2f} N·m")

    # Verify accuracy
    accuracy = verify_mk_accuracy(sec, mat, n_fibers=200)
    print(f"  M-κ accuracy (closed-form vs numerical): {accuracy['max_relative_error']:.2e}")
    print(f"  Pass (≤1e-12): {accuracy['pass']}")

    # Plot M-κ curve
    kappa = np.linspace(0, 5 * kappa_y, 200)
    M_closed = epp_mk_closed_form(kappa, sec, mat)
    M_numerical = epp_mk_numerical(kappa, sec, mat, n_fibers=200)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kappa / kappa_y, M_closed / Mp, 'b-', label='Closed-form', linewidth=2)
    ax.plot(kappa / kappa_y, M_numerical / Mp, 'r--', label='Numerical (200 fibers)', linewidth=1)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Curvature / κy', fontsize=12)
    ax.set_ylabel('Moment / Mp', fontsize=12)
    ax.set_title('M-κ Relation for Rectangular Section (EPP)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('artifacts/mk_curve.png', dpi=150)
    plt.close()
    print("  → Saved artifacts/mk_curve.png")

    # ============================================================
    # 3. Frame Strip Response
    # ============================================================
    print("\n3. Frame Strip Response")
    print("-" * 40)

    frame_params = {
        "L": params["frame_spacing"],
        "b": params["stiffener_spacing"],
        "plate_thickness": params["plate_thickness"],
        "E": params["E"],
        "sigma_y": params["sigma_y"],
        "nu": params["nu"],
        "p0": params["p0"],
        "ex": params["ex"],
    }
    result = compute_frame_strip_response(frame_params)

    print(f"  Panel area: {frame_params['L']*frame_params['b']:.2f} m²")
    print(f"  ISO 19906 ice pressure: {result['p_ice_Pa']/1e6:.2f} MPa")
    print(f"  First yield pressure: {result['p_yield_Pa']/1e6:.2f} MPa")
    print(f"  Full plastic pressure: {result['p_plastic_Pa']/1e6:.2f} MPa")
    print(f"  Elastic deflection: {result['w_elastic_m']*1000:.4f} mm")
    print(f"  Residual deflection: {result['w_residual_m']*1000:.4f} mm")
    print(f"  Regime: {result['regime']}")

    # ============================================================
    # 4. Two-Bar Shakedown Benchmark
    # ============================================================
    print("\n4. Two-Bar Shakedown Benchmark")
    print("-" * 40)

    two_bar = verify_two_bar_shakedown()
    print(f"  P1 (constant): {two_bar['P1_constant']/1e6:.2f} MN")
    print(f"  σ1 (elastic): {two_bar['sigma_min']/1e6:.1f} MPa")
    print(f"  P2 first yield: {two_bar['P_first_yield']/1e6:.2f} MN")
    print(f"  P2 shakedown: {two_bar['P2_max_shakedown']/1e6:.2f} MN")
    print(f"  Ratio (shakedown/first_yield): {two_bar['ratio']:.4f}")
    print(f"  Expected: 2.0")
    print(f"  Pass: {two_bar['pass']}")

    # ============================================================
    # 5. Melan Shakedown Bound
    # ============================================================
    print("\n5. Melan Shakedown Bound")
    print("-" * 40)

    beam = BeamStrip(L=params["frame_spacing"], b=params["stiffener_spacing"],
                     sec=sec, mat=mat)
    melan = MelanShakedownBeam(sigma_y=mat.sigma_y, E=mat.E,
                                n_fibers=51, b=sec.b, h=sec.h)

    M_amp = beam.Me
    mu = melan.shakedown_factor((-M_amp, M_amp))
    print(f"  Elastic moment Me: {beam.Me:.2f} N·m")
    print(f"  Plastic moment Mp: {beam.Mp:.2f} N·m")
    print(f"  Shakedown factor (symmetric cyclic): {mu:.4f}")
    print(f"  Shakedown moment: {mu * beam.Me:.2f} N·m")

    # ============================================================
    # 6. Incremental Cyclic Analysis
    # ============================================================
    print("\n6. Incremental Cyclic Analysis")
    print("-" * 40)

    E = params["E"]
    sigma_y = params["sigma_y"]
    eps_y = sigma_y / E

    # Case 1: Below yield (elastic shakedown)
    mat1 = KinematicHardening1D(E=E, sigma_y0=sigma_y, H=1e9)
    hist1 = run_cyclic_analysis(mat1, 0.5 * eps_y, n_cycles=20, load_ratio=0.0)
    regime1 = classify_regime(hist1)
    print(f"  Case 1 (0.5·εy, R=0): regime={regime1}, residual={hist1.residual_strain[-1]:.2e}")

    # Case 2: Above yield (ratcheting)
    mat2 = KinematicHardening1D(E=E, sigma_y0=sigma_y, H=1e9)
    hist2 = run_cyclic_analysis(mat2, 2.0 * eps_y, n_cycles=50, load_ratio=0.0)
    regime2 = classify_regime(hist2)
    print(f"  Case 2 (2.0·εy, R=0): regime={regime2}, residual={hist2.residual_strain[-1]:.2e}")

    # Case 3: Fully reversed
    mat3 = KinematicHardening1D(E=E, sigma_y0=sigma_y, H=10e9)
    hist3 = run_cyclic_analysis(mat3, 2.0 * eps_y, n_cycles=50, load_ratio=-1.0)
    regime3 = classify_regime(hist3)
    print(f"  Case 3 (2.0·εy, R=-1): regime={regime3}, residual={hist3.residual_strain[-1]:.2e}")

    # Plot cyclic response
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, hist, title in [(axes[0], hist1, "Elastic Shakedown"),
                             (axes[1], hist2, "Ratcheting"),
                             (axes[2], hist3, "Alternating Plasticity")]:
        ax.plot(hist.cycle_number, hist.residual_strain, 'b-', linewidth=2)
        ax.set_xlabel('Cycle Number', fontsize=10)
        ax.set_ylabel('Residual Strain', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('artifacts/cyclic_response.png', dpi=150)
    plt.close()
    print("  → Saved artifacts/cyclic_response.png")

    # ============================================================
    # 7. Regime Map
    # ============================================================
    print("\n7. Regime Map Generation")
    print("-" * 40)

    regime_result = generate_allowable_pressure_map({
        "E": E,
        "sigma_y": sigma_y,
        "H": 1e9,
    })

    # Save regime map
    import base64
    png_data = base64.b64decode(regime_result["png_base64"])
    with open("artifacts/regime_map.png", "wb") as f:
        f.write(png_data)
    print(f"  → Saved artifacts/regime_map.png")
    print(f"  Allowable pressure ratio: {regime_result['allowable_pressure_ratio']:.4f}")
    print(f"  Allowable pressure: {regime_result['allowable_pressure_Pa']/1e6:.2f} MPa")

    # ============================================================
    # 8. Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  M-κ accuracy: {accuracy['max_relative_error']:.2e} (pass: {accuracy['pass']})")
    print(f"  Two-bar shakedown ratio: {two_bar['ratio']:.4f} (expected: 2.0, pass: {two_bar['pass']})")
    print(f"  Melan shakedown factor: {mu:.4f}")
    print(f"  Regimes: {regime1}, {regime2}, {regime3}")
    print(f"  Allowable pressure: {regime_result['allowable_pressure_Pa']/1e6:.2f} MPa")
    print("\n  Artifacts generated:")
    print("    - artifacts/mk_curve.png")
    print("    - artifacts/cyclic_response.png")
    print("    - artifacts/regime_map.png")

    # Save results JSON
    results = {
        "mk_accuracy": accuracy["max_relative_error"],
        "mk_accuracy_pass": accuracy["pass"],
        "two_bar_ratio": two_bar["ratio"],
        "two_bar_pass": two_bar["pass"],
        "melan_shakedown_factor": mu,
        "regimes": [regime1, regime2, regime3],
        "allowable_pressure_ratio": regime_result["allowable_pressure_ratio"],
        "allowable_pressure_MPa": regime_result["allowable_pressure_Pa"] / 1e6,
    }
    with open("artifacts/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("    - artifacts/results.json")


if __name__ == "__main__":
    main()
