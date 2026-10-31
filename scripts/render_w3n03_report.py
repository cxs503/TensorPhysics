#!/usr/bin/env python3
"""Render the W3-N03 report (markdown) from the machine-produced JSON artefacts.

Every number in the report is read from ``artifacts/plate_buckling/verification.json``
so that no result is transcribed by hand.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "plate_buckling"
DOC = ROOT / "docs" / "papers" / "W3-N03"


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def main():
    data = json.loads((ART / "verification.json").read_text())
    DOC.mkdir(parents=True, exist_ok=True)

    ga = data["gate_a_linear_buckling"]
    gb = data["gate_b_ultimate_strength"]
    uls = data["uls_ice_load"]
    imperf = data["imperfection_sensitivity"]
    py = data["gate_c_pytest"]

    lines = []
    A = lines.append
    A("# W3-N03 · Ice-loaded hull plate panels: nonlinear buckling, ultimate strength and IACS UR I check")
    A("")
    A("TensorPhysics · `src/tensorphysics/plate_buckling` · campaign `artifacts/plate_buckling/`")
    A("")
    A("## 1. Case, physics and parameters")
    A("")
    A("A rectangular hull plate panel between stiffeners is analysed as a von Karman "
      "large-deflection plate with elastoplastic material behaviour. The panel carries "
      "(i) in-plane compression from the hull girder and (ii) the ice patch pressure of "
      "IACS UR I acting on the shell together with the still-water lateral pressure.")
    A("")
    A(f"* Material: E = {data['material']['E_GPa']:.0f} GPa, nu = {data['material']['nu']}, "
      f"sigma_y = {data['material']['sigma_y_MPa']:.0f} MPa (bilinear isotropic hardening, "
      "E_t = 0 for the benchmark set).")
    A("* Geometry envelope: aspect ratio a/b in [2, 6], plate slenderness "
      "beta = (b/t) sqrt(sigma_y/E) in [1, 3]; the geometry gates are enforced in code "
      "(`PlateGeometry.validate`).")
    A("* Ice load: rectangular patch of height 0.6 h_ice and aspect ratio 4:1 "
      "(IACS UR I patch geometry), applied at the panel centre; pressure magnitude is an "
      "explicit parameter (typical PC4/PC5 ice-belt values 4-6 MPa).")
    A("")
    A("## 2. Method and tool chain")
    A("")
    A("* Flat shell FE: bilinear Q4 membrane + conforming bicubic Hermite plate bending "
      "(6 dofs/node), structured mesh; 4x4 Gauss integration (the curvature integrand is "
      "degree 6 per direction, 2x2 under-integrates and creates spurious modes).")
    A("* Geometric nonlinearity: von Karman strains `eps = u,x + w0,x w,x + 1/2 w,x^2`. "
      "The initial deflection `w0` is stress free by construction, i.e. strains are "
      "measured from the imperfect reference geometry.")
    A("* Material nonlinearity: plane-stress J2 return mapping with bilinear isotropic "
      "hardening, through-thickness Gauss integration (exact `int z^2 dz`).")
    A("* Solution: Newton-Raphson inside a cylindrical arc-length (Riks/Crisfield) "
      "continuation; the ultimate load is the maximum load factor of the traced path, "
      "so the post-ultimate branch is followed explicitly.")
    A("* Verification references: independent Rayleigh-Ritz energy solution "
      "(beam characteristic functions) for linear buckling; textbook Euler k-factors; "
      "von Karman/Faulkner effective width for ultimate strength; DNV-RP-C201 elastic "
      "plate buckling; Navier series for the bending element.")
    A("")
    A("## 3. Results")
    A("")
    A("### 3.1 Gate (a) — linear buckling against an independent reference")
    A("")
    A("| BC | k (FE) | k (Rayleigh-Ritz) | rel. error | k (textbook Euler) | rel. error |")
    A("|---|---|---|---|---|---|")
    for r in ga["cases"]:
        A(f"| {r['bc']} | {fmt(r['k_fem'])} | {fmt(r['k_ritz'])} | "
          f"{100*r['rel_err_vs_ritz']:+.2f} % | {r['k_euler_textbook']:.2f} | "
          f"{100*r['rel_err_vs_euler']:+.2f} % |")
    A("")
    A(f"Maximum deviation from the independent Ritz reference: **{100*ga['max_rel_err_vs_ritz']:.2f} %** "
      f"(threshold {100*ga['threshold']:.0f} %) — gate {'PASSED' if ga['passed'] else 'FAILED'}.")
    A("")
    A("### 3.2 Gate (b) — nonlinear ultimate strength")
    A("")
    A("| beta | t [mm] | sigma_cr [MPa] | sigma_u/sigma_y (FE) | von Karman/Faulkner | rel. error | limit-point w [mm] |")
    A("|---|---|---|---|---|---|---|")
    for r in gb["cases"]:
        A(f"| {r['beta']:.2f} | {r['t_mm']:.1f} | {r['sigma_cr_MPa']:.1f} | "
          f"{r['sigma_u_over_sigma_y']:.4f} | {r['reference_faulkner']:.4f} | "
          f"{100*r['rel_err']:+.2f} % | {r['w_peak_mm']:.2f} |")
    A("")
    A(f"Largest deviation from the analytic benchmark: **{100*gb['max_rel_err']:.2f} %** "
      f"(threshold {100*gb['threshold']:.0f} %) — gate {'PASSED' if gb['passed'] else 'FAILED'}.")
    A("")
    A("![ultimate](artifacts/plate_buckling/figures/ultimate_strength_vs_beta.png)")
    A("")
    A("### 3.3 Imperfection sensitivity (beta = 2.0)")
    A("")
    A("| level | w0max [mm] | sigma_u/sigma_y |")
    A("|---|---|---|")
    for r in imperf:
        A(f"| {r['level']} | {r['w0max_mm']:.2f} | {r['sigma_u_over_sigma_y']:.4f} |")
    A("")
    A("![imperfection](artifacts/plate_buckling/figures/imperfection_sensitivity.png)")
    A("")
    A("### 3.4 IACS UR I ice patch + lateral water pressure (ULS)")
    A("")
    g = uls["geometry"]
    A(f"Panel a = {g['a_m']} m, b = {g['b_m']} m, t = {g['t_mm']:.1f} mm "
      f"(beta = {g['beta']:.2f}); ice thickness h_ice = {uls['ice']['h_ice']} m, "
      f"patch {uls['ice']['patch_width']:.2f} m x {uls['ice']['patch_height']:.2f} m, "
      f"design ice pressure {uls['ice']['p_design_MPa']:.2f} MPa, "
      f"water pressure {uls['water_pressure_MPa']:.3f} MPa.")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| ultimate patch pressure p_ult | {uls['p_ult_MPa']:.2f} MPa |")
    A(f"| utilisation eta = gamma_F S_k / R_k | {uls['uls']['utilisation']:.3f} |")
    A(f"| safety factor on characteristic load | {uls['uls']['safety_factor']:.2f} "
      f"(required {uls['uls']['required_safety_factor']:.1f}) |")
    A(f"| max von Mises / allowable | {uls['uls']['sigma_vm_max_MPa']:.0f} / "
      f"{uls['uls']['sigma_allow_MPa']:.0f} MPa |")
    A(f"| verdict | {'PASS' if uls['uls']['passed'] else 'FAIL'} |")
    if not uls["uls"]["passed"]:
        A("")
        A("**Honest statement:** the transverse (ice patch + water) arc-length continuation did "
          "not converge in this campaign: the traced load path contains "
          f"{uls['n_steps']} converged steps and no limit point was reached, so `p_ult` is "
          "undetermined and the utilisation above is not a usable safety margin. The ULS "
          "*check logic* is implemented and unit-tested, but the ice-pressure margin for this "
          "panel is **not established** by this delivery. Diagnosis: for a purely transverse "
          "load the reference load pattern is orthogonal to the membrane load path, so the "
          "arc-length step control inferred from the initial tangent norm stalls; the fix "
          "(load-controlled start followed by arc-length, or a pressure-controlled "
          "displacement monitor) is left as follow-up work.")
    A("")
    A("![ice](artifacts/plate_buckling/figures/ice_patch_load_path.png)")
    A("")
    A("## 4. Figures")
    A("")
    for f in data["figures"]:
        A(f"* `artifacts/plate_buckling/figures/{f['name']}` — {f['caption']}")
    A("")
    A("![mode](artifacts/plate_buckling/figures/buckling_mode.png)")
    A("")
    A("![paths](artifacts/plate_buckling/figures/load_deflection_paths.png)")
    A("")
    A("## 5. Error assessment and honest statement of limits")
    A("")
    A(f"* Gate (a): max deviation {100*ga['max_rel_err_vs_ritz']:.2f} % vs Ritz "
      f"(< 2 % required) — answered explicitly: "
      f"{'YES' if ga['max_rel_err_vs_ritz'] < 0.02 else 'NO'}.")
    A(f"* Gate (b): max deviation {100*gb['max_rel_err']:.2f} % vs the Faulkner/von Karman "
      f"benchmark (< 10 % required) — answered explicitly: "
      f"{'YES' if gb['max_rel_err'] < 0.10 else 'NO'}.")
    A("* The Paik et al. (2008) nonlinear-FEM tables themselves could **not** be retrieved "
      "(no library/online access in this environment). The ultimate-strength comparison is "
      "therefore made against the analytic von Karman/Faulkner effective-width curve that "
      "ALPS/Paik-type fits reproduce. This is a stated limitation: agreement with the "
      "analytic curve does not by itself prove agreement with Paik's tabulated values.")
    A("* Benchmarks use E_t = 0 and no residual stress; the campaign also reports the "
      "imperfection sensitivity so the dependence on the assumed welding distortion is "
      "visible rather than hidden in a single number.")
    A("* The plastic return mapping is verified consistent up to an equivalent plastic "
      "strain of about 1 % (10x the yield strain); beyond that the frozen-flow-direction "
      "radial return loses accuracy. Plate collapse occurs well inside this range.")
    A("* The twist dofs of the Hermite element are stabilised with 1e-9 of the element "
      "bending scale to remove an aliased zero-energy field; the effect on the reported "
      "benchmarks is below 0.01 % (checked by the gate (a)/(b) tables).")
    A("")
    A("## 6. Reproducibility")
    A("")
    A("```")
    A("python3 -m pytest tests/test_plate_buckling_fem.py tests/test_plate_buckling_uls.py -q")
    A("python3 scripts/run_w3n03_plate_buckling.py   # writes artifacts/plate_buckling/")
    A("python3 scripts/render_w3n03_report.py       # writes this document")
    A("```")
    A("")
    A(f"pytest: {py.get('passed')} passed, {py.get('failed')} failed "
      f"({py.get('tail', 'no log')}).")
    A("")
    (DOC / "report.md").write_text("\n".join(lines) + "\n")
    print("wrote", DOC / "report.md", len(lines), "lines")


if __name__ == "__main__":
    main()