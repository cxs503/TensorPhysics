"""Golden set: 60 manually annotated contribution items for classifier evaluation.

Composition:
  - C1 verified_case:     12 items (including 2 adversarial)
  - C2 capability_delta:  10 items (including 2 adversarial)
  - C3 defect_fix:        10 items (including 2 adversarial)
  - C4 method_doc:        10 items (including 2 adversarial)
  - C5 negative_result:    8 items (including 1 adversarial)
  - C6 infra:             10 items (including 1 adversarial)
  Total: 60 items, 10 adversarial samples
"""

from typing import Any, Dict, List

from .classifier import ContributionItem
from .ontology import ContributionCategory


def _item(d: Dict) -> Dict:
    """Ensure dict has all required fields."""
    d.setdefault("tags", [])
    d.setdefault("evidence", {})
    d.setdefault("is_adversarial", False)
    return d


GOLDEN_SET: List[Dict[str, Any]] = [
    # ===== C1: Verified Case (12 items) =====
    _item({
        "mission_id": "m103",
        "title": "Holtrop resistance validation against experimental data for KCS hull",
        "summary": "Verified Holtrop 1984 resistance prediction against KRISO KCS experimental data. Geometry validated: Cb=0.6506, L=230m, B=32.2m. Computed Ct=4.012e-3 vs experimental Ct=3.987e-3, error=0.63%.",
        "tags": ["resistance", "validation", "holtrop", "KCS"],
        "evidence": {"geometry_parameters": {"Cb": 0.6506, "L": 230.0, "B": 32.2}, "reference_value": 3.987e-3, "computed_value": 4.012e-3, "error_percent": 0.63},
        "ground_truth": "C1_verified_case",
    }),
    _item({
        "mission_id": "m1046",
        "title": "BEM wave resistance benchmark against Wigley III analytical solution",
        "summary": "TensorBEM wave resistance computation validated against Wigley III parabolic hull analytical solution. L=2.0m, B=0.3m, Cb=0.555. Error in wave resistance coefficient: 2.1% at Fr=0.3.",
        "tags": ["BEM", "wave-resistance", "Wigley", "analytical"],
        "evidence": {"geometry_parameters": {"Cb": 0.555, "L": 2.0, "B": 0.3}, "reference_value": 0.00234, "computed_value": 0.00239, "error_percent": 2.1},
        "ground_truth": "C1_verified_case",
    }),
    _item({
        "mission_id": "m1133",
        "title": "Ice resistance model validation against ACRCL model test data",
        "summary": "Validated ice resistance prediction against ACRCL model basin data for icebreaker hull. Geometry: L=85m, B=20m, Cb=0.58. Computed ice resistance 1250kN vs measured 1180kN, error=5.9%.",
        "tags": ["ice-resistance", "validation", "ACRCL"],
        "evidence": {"geometry_parameters": {"Cb": 0.58, "L": 85.0, "B": 20.0}, "reference_value": 1180.0, "computed_value": 1250.0, "error_percent": 5.9},
        "ground_truth": "C1_verified_case",
    }),
    _item({
        "mission_id": "m1212",
        "title": "Propeller open water characteristics validation against ITTC procedure",
        "summary": "Validated propeller KT, KQ, eta against ITTC standard procedure 7.5-02-03-01.4. Reference: B-series propeller D=0.25m, P/D=1.0. Max error in KT: 1.8% at J=0.6.",
        "tags": ["propeller", "ITTC", "open-water", "validation"],
        "evidence": {"geometry_parameters": {"D": 0.25, "P_D_ratio": 1.0}, "reference_value": 0.285, "computed_value": 0.290, "error_percent": 1.8},
        "ground_truth": "C1_verified_case",
    }),
    _item({
        "mission_id": "m1359",
        "title": "Seakeeping RAO validation against experimental data for S175 container ship",
        "summary": "Computed heave and pitch RAOs for S175 container ship validated against model test data. L=175m, B=25.4m, Cb=0.572. Peak heave RAO error: 4.2% at encounter frequency 0.8 rad/s.",
        "tags": ["seakeeping", "RAO", "S175", "validation"],
        "evidence": {"geometry_parameters": {"Cb": 0.572, "L": 175.0, "B": 25.4}, "reference_value": 1.42, "computed_value": 1.48, "error_percent": 4.2},
        "ground_truth": "C1_verified_case",
    }),
    _item({
        "mission_id": "m1421",
        "title": "Maneuvering simulation validation against PMM test for DTMB 5415",
        "summary": "Validated maneuvering coefficients against PMM test data for DTMB 5415. L=147.5m, B=18.8m, Cb=0.507. Yv' error: 3.1%, Nv' error: 4.7%.",
        "tags": ["maneuvering", "PMM", "DTMB5415", "validation"],
        "evidence": {"geometry_parameters": {"Cb": 0.507, "L": 147.5, "B": 18.8}, "reference_value": -0.42, "computed_value": -0.433, "error_percent": 3.1},
        "ground_truth": "C1_verified_case",
    }),
    _item({
        "mission_id": "m1629",
        "title": "Intact stability calculation verification against IMO criteria",
        "summary": "Verified intact stability calculation for bulk carrier against IMO A.749(18) criteria. GZ curve computed, area 0-30deg = 0.082 m-rad (req > 0.055), area 0-40deg = 0.145 m-rad (req > 0.090).",
        "tags": ["stability", "IMO", "GZ-curve", "verification"],
        "evidence": {"geometry_parameters": {"Cb": 0.82, "L": 190.0, "B": 32.0}, "reference_value": 0.055, "computed_value": 0.082, "error_percent": 0.0},
        "ground_truth": "C1_verified_case",
    }),
    _item({
        "mission_id": "m1830",
        "title": "LBM lid-driven cavity benchmark against Ghia 1982 reference",
        "summary": "TensorLBM lid-driven cavity flow at Re=1000 validated against Ghia et al. 1982 reference solution. Centerline u-velocity max error: 0.8%. Grid: 256x256.",
        "tags": ["LBM", "lid-driven-cavity", "Ghia", "benchmark"],
        "evidence": {"geometry_parameters": {"Re": 1000, "grid": "256x256"}, "reference_value": 0.3876, "computed_value": 0.3845, "error_percent": 0.8},
        "ground_truth": "C1_verified_case",
    }),
    _item({
        "mission_id": "m2089",
        "title": "FEM plate buckling validation against Timoshenko analytical solution",
        "summary": "Validated plate buckling critical load against Timoshenko analytical solution. Simply supported square plate, a/b=1, k=4.0. Computed buckling factor: 4.02 vs analytical 4.0, error=0.5%.",
        "tags": ["FEM", "buckling", "Timoshenko", "analytical"],
        "evidence": {"geometry_parameters": {"a_b_ratio": 1.0, "k": 4.0}, "reference_value": 4.0, "computed_value": 4.02, "error_percent": 0.5},
        "ground_truth": "C1_verified_case",
    }),
    _item({
        "mission_id": "m2111",
        "title": "DEM ice fragment size distribution validation against field data",
        "summary": "Validated DEM ice fragment size distribution against field measurements from Baltic Sea. Mean fragment size: computed 0.45m vs measured 0.42m, error=7.1%.",
        "tags": ["DEM", "ice-fragments", "field-data", "validation"],
        "evidence": {"geometry_parameters": {"ice_thickness": 0.5}, "reference_value": 0.42, "computed_value": 0.45, "error_percent": 7.1},
        "ground_truth": "C1_verified_case",
    }),
    # C1 Adversarial 1: missing geometry validation
    _item({
        "mission_id": "m2114",
        "title": "Resistance computation for unknown hull form",
        "summary": "Computed total resistance using Holtrop method. Ct=3.5e-3 at design speed. No geometry parameters provided, no reference comparison.",
        "tags": ["resistance", "holtrop"],
        "evidence": {"geometry_parameters": None, "reference_value": None, "computed_value": 3.5e-3, "error_percent": None},
        "ground_truth": "C1_verified_case",
        "is_adversarial": True,
    }),
    # C1 Adversarial 2: error > 15%
    _item({
        "mission_id": "m2120",
        "title": "Wave resistance comparison with experiment",
        "summary": "Compared wave resistance prediction with model test. Computed Rw=850N vs experimental Rw=520N. Error=63.5%. Claimed 'reasonable agreement for preliminary design'.",
        "tags": ["wave-resistance", "experiment"],
        "evidence": {"geometry_parameters": {"Cb": 0.6, "L": 100.0, "B": 16.0}, "reference_value": 520.0, "computed_value": 850.0, "error_percent": 63.5},
        "ground_truth": "C1_verified_case",
        "is_adversarial": True,
    }),

    # ===== C2: Capability Delta (10 items) =====
    _item({
        "mission_id": "m4907",
        "title": "LBM GPU acceleration: 12x speedup over CPU implementation",
        "summary": "Ported TensorLBM D2Q9 kernel to CUDA. Baseline CPU: 45 min for 10k steps on 256x256 grid. GPU: 3.7 min on RTX 4090. Speedup: 12.2x.",
        "tags": ["LBM", "GPU", "CUDA", "performance"],
        "evidence": {"baseline_metric": 2700.0, "improved_metric": 222.0, "delta_percent": 1120.0, "metric_name": "time_seconds"},
        "ground_truth": "C2_capability_delta",
    }),
    _item({
        "mission_id": "m4911",
        "title": "BEM panel reduction: 3x faster with adaptive mesh refinement",
        "summary": "Implemented adaptive panel refinement for BEM wave resistance. Baseline: 5000 uniform panels, 120s compute. Adaptive: 1800 panels, 42s compute. Error increase: only 0.3% in Ct.",
        "tags": ["BEM", "adaptive-mesh", "performance"],
        "evidence": {"baseline_metric": 120.0, "improved_metric": 42.0, "delta_percent": 185.7, "metric_name": "compute_time_seconds"},
        "ground_truth": "C2_capability_delta",
    }),
    _item({
        "mission_id": "m4919",
        "title": "FEM convergence improvement: higher-order elements reduce DOF by 60%",
        "summary": "Replaced linear elements with quadratic serendipity elements in plate buckling FEM. Baseline: 10000 linear DOF for 1% error. Improved: 4000 quadratic DOF for same error.",
        "tags": ["FEM", "higher-order", "convergence"],
        "evidence": {"baseline_metric": 10000.0, "improved_metric": 4000.0, "delta_percent": 150.0, "metric_name": "degrees_of_freedom"},
        "ground_truth": "C2_capability_delta",
    }),
    _item({
        "mission_id": "m4920",
        "title": "Ice resistance model extended to multi-year ice conditions",
        "summary": "Extended ice resistance prediction from first-year to multi-year ice. Baseline model valid for hi<1.5m. Extended model valid for hi<3.0m.",
        "tags": ["ice-resistance", "multi-year", "extended-scope"],
        "evidence": {"baseline_metric": 1.5, "improved_metric": 3.0, "delta_percent": 100.0, "metric_name": "max_ice_thickness_m"},
        "ground_truth": "C2_capability_delta",
    }),
    _item({
        "mission_id": "m4934",
        "title": "DEM contact detection optimized: 5x faster with spatial hashing",
        "summary": "Replaced brute-force O(n²) contact detection with spatial hashing O(n). Baseline: 500 particles, 25s/step. Improved: 500 particles, 5s/step.",
        "tags": ["DEM", "contact-detection", "spatial-hashing", "performance"],
        "evidence": {"baseline_metric": 25.0, "improved_metric": 5.0, "delta_percent": 400.0, "metric_name": "time_per_step_seconds"},
        "ground_truth": "C2_capability_delta",
    }),
    _item({
        "mission_id": "m5044",
        "title": "Propeller cavitation prediction accuracy improved by 15%",
        "summary": "Improved cavitation inception prediction by adding viscous correction to BEM. Baseline error in cavitation number: 22%. Improved error: 7%.",
        "tags": ["propeller", "cavitation", "accuracy"],
        "evidence": {"baseline_metric": 22.0, "improved_metric": 7.0, "delta_percent": 68.2, "metric_name": "cavitation_number_error_percent"},
        "ground_truth": "C2_capability_delta",
    }),
    _item({
        "mission_id": "m5192",
        "title": "Seakeeping computation extended to forward speed effects",
        "summary": "Added forward speed capability to seakeeping BEM. Baseline: zero-speed RAOs only. Extended: speed up to Fr=0.3.",
        "tags": ["seakeeping", "forward-speed", "capability"],
        "evidence": {"baseline_metric": 0.0, "improved_metric": 0.3, "delta_percent": 100.0, "metric_name": "max_froude_number"},
        "ground_truth": "C2_capability_delta",
    }),
    _item({
        "mission_id": "m5207",
        "title": "Structural reliability analysis: Monte Carlo variance reduced 8x",
        "summary": "Implemented Latin Hypercube Sampling for structural reliability. Baseline MC: 100k samples, COV=5%. LHS: 12.5k samples, COV=5%.",
        "tags": ["structural", "reliability", "LHS", "variance-reduction"],
        "evidence": {"baseline_metric": 100000.0, "improved_metric": 12500.0, "delta_percent": 700.0, "metric_name": "samples_for_COV_5pct"},
        "ground_truth": "C2_capability_delta",
    }),
    # C2 Adversarial 1: delta < 5%
    _item({
        "mission_id": "m5208",
        "title": "Minor optimization in resistance calculation",
        "summary": "Optimized loop order in resistance computation. Baseline: 12.3s. Improved: 12.1s. 1.6% speedup.",
        "tags": ["resistance", "optimization"],
        "evidence": {"baseline_metric": 12.3, "improved_metric": 12.1, "delta_percent": 1.6, "metric_name": "compute_time_seconds"},
        "ground_truth": "C2_capability_delta",
        "is_adversarial": True,
    }),
    # C2 Adversarial 2: no baseline
    _item({
        "mission_id": "m5212",
        "title": "Faster ice resistance computation",
        "summary": "Improved ice resistance model performance. Now runs in 8 seconds. Much faster than before.",
        "tags": ["ice-resistance", "performance"],
        "evidence": {"baseline_metric": None, "improved_metric": 8.0, "delta_percent": None, "metric_name": "compute_time_seconds"},
        "ground_truth": "C2_capability_delta",
        "is_adversarial": True,
    }),

    # ===== C3: Defect Fix (10 items) =====
    _item({
        "mission_id": "m5213",
        "title": "Fix: BEM free-surface Green function singularity at source point",
        "summary": "Fixed singularity handling in free-surface Green function evaluation. Bug: division by zero when source and field point coincide. Fix: added analytical limit for r→0. Regression test added.",
        "tags": ["BEM", "bug-fix", "green-function", "singularity"],
        "evidence": {"bug_description": "Division by zero at coincident source/field points", "fix_description": "Added analytical limit evaluation for r→0", "regression_test": "test_green_function_singularity_limit"},
        "ground_truth": "C3_defect_fix",
    }),
    _item({
        "mission_id": "m579",
        "title": "Fix: LBM boundary condition wrong sign for bounce-back",
        "summary": "Fixed sign error in bounce-back boundary condition. Bug: velocity was being added instead of subtracted, causing slip at walls. Fix: corrected sign.",
        "tags": ["LBM", "bug-fix", "boundary-condition"],
        "evidence": {"bug_description": "Wrong sign in bounce-back: velocity added instead of subtracted", "fix_description": "Corrected sign in distribution function reflection", "regression_test": "test_bounce_back_no_slip"},
        "ground_truth": "C3_defect_fix",
    }),
    _item({
        "mission_id": "m580",
        "title": "Fix: FEM stiffness matrix assembly missing factor of 2 for symmetric terms",
        "summary": "Fixed stiffness matrix assembly bug. Bug: symmetric off-diagonal terms were not doubled, leading to 15% error in displacement. Fix: added factor of 2.",
        "tags": ["FEM", "bug-fix", "stiffness-matrix"],
        "evidence": {"bug_description": "Missing factor of 2 for symmetric off-diagonal terms", "fix_description": "Added factor of 2 in assembly loop", "regression_test": "test_cantilever_beam_deflection"},
        "ground_truth": "C3_defect_fix",
    }),
    _item({
        "mission_id": "m598",
        "title": "Fix: DEM contact force direction inverted for overlapping particles",
        "summary": "Fixed contact force direction bug. Bug: repulsive force was pointing inward instead of outward for overlapping particles, causing particle collapse.",
        "tags": ["DEM", "bug-fix", "contact-force"],
        "evidence": {"bug_description": "Repulsive force direction inverted for overlapping particles", "fix_description": "Inverted force direction vector", "regression_test": "test_particle_stack_stability"},
        "ground_truth": "C3_defect_fix",
    }),
    _item({
        "mission_id": "m605",
        "title": "Fix: Ice resistance model using wrong ice density value",
        "summary": "Fixed ice density constant. Bug: used 800 kg/m³ (snow density) instead of 917 kg/m³ (sea ice). Fix: corrected constant.",
        "tags": ["ice-resistance", "bug-fix", "constants"],
        "evidence": {"bug_description": "Wrong ice density: 800 instead of 917 kg/m³", "fix_description": "Corrected ice density constant to 917 kg/m³", "regression_test": "test_ice_density_constant"},
        "ground_truth": "C3_defect_fix",
    }),
    _item({
        "mission_id": "m611",
        "title": "Fix: Propeller open water diagram interpolation out of bounds",
        "summary": "Fixed interpolation crash when J value outside B-series table range. Bug: no bounds checking, caused segfault. Fix: added extrapolation with warning.",
        "tags": ["propeller", "bug-fix", "interpolation"],
        "evidence": {"bug_description": "No bounds checking on J interpolation, caused segfault", "fix_description": "Added bounds checking with linear extrapolation", "regression_test": "test_propeller_interpolation_bounds"},
        "ground_truth": "C3_defect_fix",
    }),
    _item({
        "mission_id": "m667",
        "title": "Fix: Seakeeping RAO phase angle wrapped incorrectly",
        "summary": "Fixed RAO phase angle wrapping. Bug: phase was in [-π, π] but should be [0, 2π] for consistency with ITTC procedure.",
        "tags": ["seakeeping", "bug-fix", "phase-angle"],
        "evidence": {"bug_description": "Phase angle range [-π,π] instead of [0,2π]", "fix_description": "Added phase unwrapping to [0, 2π]", "regression_test": "test_rao_phase_convention"},
        "ground_truth": "C3_defect_fix",
    }),
    _item({
        "mission_id": "m668",
        "title": "Fix: Maneuvering hydrodynamic derivatives sign convention mismatch",
        "summary": "Fixed sign convention for hydrodynamic derivatives. Bug: used MMG convention but code expected ITTC convention.",
        "tags": ["maneuvering", "bug-fix", "sign-convention"],
        "evidence": {"bug_description": "MMG vs ITTC sign convention mismatch", "fix_description": "Added convention flag with conversion", "regression_test": "test_hydrodynamic_derivative_signs"},
        "ground_truth": "C3_defect_fix",
    }),
    # C3 Adversarial 1: cosmetic change
    _item({
        "mission_id": "m671",
        "title": "Fix: reformatted resistance calculation code",
        "summary": "Reformatted the resistance calculation module for better readability. Changed variable names and added comments. No functional changes.",
        "tags": ["resistance", "refactoring"],
        "evidence": {"bug_description": "Code was hard to read", "fix_description": "Reformatted and renamed variables", "regression_test": None},
        "ground_truth": "C3_defect_fix",
        "is_adversarial": True,
    }),
    # C3 Adversarial 2: feature disguised as fix
    _item({
        "mission_id": "m680",
        "title": "Fix: added new ice model option",
        "summary": "Fixed the ice resistance module by adding a new ice model option (Lindqvist). Users can now choose between different ice models.",
        "tags": ["ice-resistance", "new-feature"],
        "evidence": {"bug_description": "Only one ice model available", "fix_description": "Added Lindqvist model as option", "regression_test": None},
        "ground_truth": "C3_defect_fix",
        "is_adversarial": True,
    }),

    # ===== C4: Method/Doc (10 items) =====
    _item({
        "mission_id": "m685",
        "title": "Tutorial: How to run TensorLBM lid-driven cavity benchmark",
        "summary": "Step-by-step tutorial for running the lid-driven cavity benchmark with TensorLBM. Includes: (1) grid setup, (2) boundary conditions, (3) relaxation time selection, (4) convergence criteria, (5) post-processing.",
        "tags": ["LBM", "tutorial", "benchmark", "documentation"],
        "evidence": {"method_description": "Lid-driven cavity benchmark setup and execution", "steps_or_workflow": "5 steps: grid setup, BCs, tau selection, convergence, post-processing", "expected_output": "Velocity profiles matching Ghia 1982 reference"},
        "ground_truth": "C4_method_doc",
    }),
    _item({
        "mission_id": "m686",
        "title": "Guide: BEM panel mesh generation for ship hulls",
        "summary": "Documentation on generating panel meshes for ship hull BEM analysis. Covers: (1) hull surface discretization, (2) panel aspect ratio guidelines, (3) free-surface panel extent, (4) convergence study procedure.",
        "tags": ["BEM", "mesh-generation", "guide", "documentation"],
        "evidence": {"method_description": "Panel mesh generation for ship hull BEM", "steps_or_workflow": "4 topics: discretization, aspect ratio, free-surface extent, convergence", "expected_output": "Validated panel mesh for KCS hull"},
        "ground_truth": "C4_method_doc",
    }),
    _item({
        "mission_id": "m687",
        "title": "Workflow: Complete ship resistance prediction pipeline",
        "summary": "End-to-end workflow documentation for ship resistance prediction. Steps: (1) hull geometry import, (2) mesh generation, (3) resistance computation, (4) validation against reference, (5) report generation.",
        "tags": ["resistance", "workflow", "pipeline", "documentation"],
        "evidence": {"method_description": "Complete ship resistance prediction pipeline", "steps_or_workflow": "5 steps: geometry import, mesh, computation, validation, report", "expected_output": "Validated resistance prediction with report"},
        "ground_truth": "C4_method_doc",
    }),
    _item({
        "mission_id": "m690",
        "title": "Protocol: Ice resistance model selection guide",
        "summary": "Step-by-step protocol for selecting appropriate ice resistance model based on ice conditions, ship type, and accuracy requirements. Covers: Lindqvist, Riska, ACRCL models.",
        "tags": ["ice-resistance", "protocol", "model-selection", "documentation"],
        "evidence": {"method_description": "Ice resistance model selection protocol", "steps_or_workflow": "Decision tree based on ice conditions, ship type, accuracy needs", "expected_output": "Model recommendation with justification"},
        "ground_truth": "C4_method_doc",
    }),
    _item({
        "mission_id": "m704",
        "title": "How-to: Set up FEM plate buckling analysis with TensorFEM",
        "summary": "Tutorial for plate buckling analysis using TensorFEM. Steps: (1) define plate geometry and material, (2) apply boundary conditions, (3) set up eigenvalue buckling analysis, (4) interpret results.",
        "tags": ["FEM", "buckling", "tutorial", "how-to"],
        "evidence": {"method_description": "FEM plate buckling analysis setup", "steps_or_workflow": "4 steps: geometry/material, BCs, eigenvalue analysis, interpretation", "expected_output": "Buckling load factor and mode shape"},
        "ground_truth": "C4_method_doc",
    }),
    _item({
        "mission_id": "m707",
        "title": "Documentation: Propeller design workflow with BEM",
        "summary": "Complete documentation for propeller design using BEM. Covers: (1) design requirements, (2) blade geometry parameterization, (3) BEM iteration procedure, (4) cavitation check, (5) off-design performance.",
        "tags": ["propeller", "BEM", "design", "documentation"],
        "evidence": {"method_description": "Propeller design workflow with BEM", "steps_or_workflow": "5 steps: requirements, geometry, iteration, cavitation, off-design", "expected_output": "Designed propeller with performance curves"},
        "ground_truth": "C4_method_doc",
    }),
    _item({
        "mission_id": "m709",
        "title": "Guide: Seakeeping analysis procedure for preliminary design",
        "summary": "Step-by-step guide for seakeeping analysis in preliminary ship design. Covers: (1) wave spectrum selection, (2) RAO computation, (3) response statistics, (4) operability assessment.",
        "tags": ["seakeeping", "guide", "preliminary-design", "documentation"],
        "evidence": {"method_description": "Seakeeping analysis for preliminary design", "steps_or_workflow": "4 steps: wave spectrum, RAO, statistics, operability", "expected_output": "Response statistics and operability assessment"},
        "ground_truth": "C4_method_doc",
    }),
    _item({
        "mission_id": "m712",
        "title": "Walkthrough: DEM ice-structure interaction simulation setup",
        "summary": "Detailed walkthrough for setting up DEM ice-structure interaction. Steps: (1) ice particle generation, (2) contact model selection, (3) structure definition, (4) simulation parameters, (5) result analysis.",
        "tags": ["DEM", "ice-structure", "walkthrough", "documentation"],
        "evidence": {"method_description": "DEM ice-structure interaction setup", "steps_or_workflow": "5 steps: particles, contact model, structure, parameters, analysis", "expected_output": "Ice-structure interaction simulation results"},
        "ground_truth": "C4_method_doc",
    }),
    # C4 Adversarial 1: auto-generated API docs
    _item({
        "mission_id": "m721",
        "title": "API documentation for resistance module",
        "summary": "Auto-generated API documentation from docstrings. Lists all functions, parameters, and return types. No explanatory content or usage examples.",
        "tags": ["resistance", "api-docs", "auto-generated"],
        "evidence": {"method_description": "Auto-generated API docs", "steps_or_workflow": None, "expected_output": None},
        "ground_truth": "C4_method_doc",
        "is_adversarial": True,
    }),
    # C4 Adversarial 2: vague description
    _item({
        "mission_id": "m722",
        "title": "Notes on resistance calculation",
        "summary": "Some notes about how resistance calculation works. It uses the Holtrop method and you need to input ship parameters. The output is resistance value.",
        "tags": ["resistance", "notes"],
        "evidence": {"method_description": "Vague notes on resistance calculation", "steps_or_workflow": "Input parameters, get output", "expected_output": "Resistance value"},
        "ground_truth": "C4_method_doc",
        "is_adversarial": True,
    }),

    # ===== C5: Negative Result (8 items) =====
    _item({
        "mission_id": "m723",
        "title": "Negative result: LBM turbulence model fails for high Re ship flows",
        "summary": "Attempted to apply Smagorinsky LES model in LBM for ship resistance at Re=10^7. Failed: numerical instability at Re>10^6 despite grid refinement. Diagnostic: kinetic energy grows unbounded, CFL condition violated even with dt=1e-6.",
        "tags": ["LBM", "turbulence", "negative-result", "high-Re"],
        "evidence": {"attempt_description": "Smagorinsky LES in LBM for ship resistance at Re=10^7", "failure_mode": "Numerical instability, unbounded kinetic energy growth", "diagnostic_data": "CFL violation at dt=1e-6, energy spectrum shows accumulation at high k"},
        "ground_truth": "C5_negative_result",
    }),
    _item({
        "mission_id": "m735",
        "title": "Negative result: BEM free-surface panel method diverges for high-speed planing hulls",
        "summary": "Attempted BEM wave resistance prediction for planing hull at Fr=0.8. Failed: iterative solver diverged after 500 iterations. Diagnostic: residual oscillates between 0.1 and 10, no convergence trend.",
        "tags": ["BEM", "planing-hull", "divergence", "negative-result"],
        "evidence": {"attempt_description": "BEM wave resistance for planing hull at Fr=0.8", "failure_mode": "Iterative solver divergence, residual oscillation", "diagnostic_data": "Residual oscillates 0.1-10, 500 iterations, no convergence"},
        "ground_truth": "C5_negative_result",
    }),
    _item({
        "mission_id": "m760",
        "title": "Negative result: DEM ice model cannot reproduce ridge formation",
        "summary": "Attempted to simulate ice ridge formation using DEM with cohesive bonds. Failed: bonds break too easily, no ridge structure forms. Diagnostic: bond strength calibrated from lab tests but field-scale ridges require different scaling.",
        "tags": ["DEM", "ice-ridge", "negative-result", "scaling"],
        "evidence": {"attempt_description": "DEM ice ridge formation with cohesive bonds", "failure_mode": "Bonds break too easily, no ridge structure", "diagnostic_data": "Bond strength from lab tests insufficient for field scale"},
        "ground_truth": "C5_negative_result",
    }),
    _item({
        "mission_id": "m761",
        "title": "Negative result: FEM fatigue life prediction unreliable for welded joints",
        "summary": "Attempted FEM-based fatigue life prediction for ship hull welded joints. Failed: predicted life varied by factor of 10x with mesh refinement. Diagnostic: stress singularity at weld toe.",
        "tags": ["FEM", "fatigue", "welded-joints", "negative-result"],
        "evidence": {"attempt_description": "FEM fatigue life prediction for welded joints", "failure_mode": "Mesh-dependent results, stress singularity at weld toe", "diagnostic_data": "Life prediction varies 10x with mesh refinement"},
        "ground_truth": "C5_negative_result",
    }),
    _item({
        "mission_id": "m762",
        "title": "Negative result: Maneuvering simulation unstable for large rudder angles",
        "summary": "Attempted maneuvering simulation with rudder angles >35°. Failed: simulation becomes unstable, ship motion diverges. Diagnostic: hydrodynamic derivatives extrapolated beyond validation range.",
        "tags": ["maneuvering", "rudder", "instability", "negative-result"],
        "evidence": {"attempt_description": "Maneuvering simulation with rudder angles >35°", "failure_mode": "Simulation instability, divergent ship motion", "diagnostic_data": "Hydrodynamic derivatives extrapolated beyond validation range"},
        "ground_truth": "C5_negative_result",
    }),
    _item({
        "mission_id": "m763",
        "title": "Negative result: Ice vibration prediction does not match full-scale measurements",
        "summary": "Attempted to predict ice-induced hull vibration using coupled ice-structure model. Failed: predicted vibration amplitude 5x higher than full-scale measurements.",
        "tags": ["ice-vibration", "scaling", "negative-result", "full-scale"],
        "evidence": {"attempt_description": "Ice-induced hull vibration prediction with coupled model", "failure_mode": "Predicted amplitude 5x higher than measurements", "diagnostic_data": "Ice load spectrum scaling issue, resonance frequency mismatch"},
        "ground_truth": "C5_negative_result",
    }),
    _item({
        "mission_id": "m764",
        "title": "Negative result: SPH free-surface capture produces spurious oscillations",
        "summary": "Attempted SPH simulation of ship wave pattern. Failed: spurious pressure oscillations at free surface. Diagnostic: kernel deficiency near free surface, particle deficiency causes pressure noise.",
        "tags": ["SPH", "free-surface", "oscillations", "negative-result"],
        "evidence": {"attempt_description": "SPH ship wave pattern simulation", "failure_mode": "Spurious pressure oscillations at free surface", "diagnostic_data": "Kernel deficiency, particle deficiency near free surface"},
        "ground_truth": "C5_negative_result",
    }),
    # C5 Adversarial 1: vague failure
    _item({
        "mission_id": "m765",
        "title": "Simulation didn't work",
        "summary": "Tried to run the resistance simulation but it didn't work. Got some errors. Will try again later.",
        "tags": ["resistance", "failed"],
        "evidence": {"attempt_description": "Ran resistance simulation", "failure_mode": "Got errors", "diagnostic_data": None},
        "ground_truth": "C5_negative_result",
        "is_adversarial": True,
    }),

    # ===== C6: Infrastructure (10 items) =====
    _item({
        "mission_id": "m779",
        "title": "CI pipeline: added automated regression testing for all solvers",
        "summary": "Added CI pipeline that runs regression tests for all solvers on every PR. Baseline: manual testing, ~2 days turnaround. Improved: automated, 30 min turnaround.",
        "tags": ["CI", "regression-testing", "automation", "infrastructure"],
        "evidence": {"infra_component": "CI pipeline", "improvement_description": "Automated regression testing for all solvers on PR", "impact_metric": "Turnaround from 2 days to 30 min, caught 3 regressions in week 1"},
        "ground_truth": "C6_infra",
    }),
    _item({
        "mission_id": "m788",
        "title": "Test harness: unified benchmark framework across all Tensor* libraries",
        "summary": "Created unified benchmark framework that runs standard test cases across TensorLBM, TensorFEM, TensorBEM, TensorDEM. Impact: reduced test setup time by 70%.",
        "tags": ["test-harness", "benchmark", "unified", "infrastructure"],
        "evidence": {"infra_component": "Test harness / benchmark framework", "improvement_description": "Unified framework across all Tensor* libraries", "impact_metric": "70% reduction in test setup time, cross-library comparison enabled"},
        "ground_truth": "C6_infra",
    }),
    _item({
        "mission_id": "m789",
        "title": "Deployment: containerized solver images for reproducible runs",
        "summary": "Created Docker images for all solvers with pinned dependencies. Impact: eliminated 90% of environment-related support tickets.",
        "tags": ["docker", "deployment", "reproducibility", "infrastructure"],
        "evidence": {"infra_component": "Docker container images", "improvement_description": "Containerized solvers with pinned dependencies", "impact_metric": "90% reduction in environment-related support tickets"},
        "ground_truth": "C6_infra",
    }),
    _item({
        "mission_id": "m790",
        "title": "Monitoring: added performance dashboards for solver runs",
        "summary": "Added monitoring dashboards tracking solver performance metrics. Impact: reduced debugging time by 50%.",
        "tags": ["monitoring", "dashboards", "performance", "infrastructure"],
        "evidence": {"infra_component": "Performance monitoring dashboards", "improvement_description": "Real-time dashboards with alerting for solver metrics", "impact_metric": "50% reduction in debugging time"},
        "ground_truth": "C6_infra",
    }),
    _item({
        "mission_id": "m791",
        "title": "Build system: migrated to pyproject.toml with proper dependency management",
        "summary": "Migrated all Tensor* libraries from setup.py to pyproject.toml. Impact: build success rate from 60% to 95%.",
        "tags": ["build-system", "pyproject", "dependencies", "infrastructure"],
        "evidence": {"infra_component": "Build system", "improvement_description": "Migrated to pyproject.toml with dependency management", "impact_metric": "Build success rate from 60% to 95%"},
        "ground_truth": "C6_infra",
    }),
    _item({
        "mission_id": "m792",
        "title": "Agent harness: added task dispatch and result collection system",
        "summary": "Built agent task dispatch system that distributes computation tasks across available agents. Impact: 3x throughput for parameter sweeps.",
        "tags": ["agent-harness", "task-dispatch", "automation", "infrastructure"],
        "evidence": {"infra_component": "Agent task dispatch system", "improvement_description": "Automated task distribution and result collection", "impact_metric": "3x throughput for parameter sweeps"},
        "ground_truth": "C6_infra",
    }),
    _item({
        "mission_id": "m793",
        "title": "Data pipeline: automated report generation from solver outputs",
        "summary": "Created automated report generation pipeline. Baseline: manual report writing, ~4 hours per report. Improved: automated, 5 min per report.",
        "tags": ["reporting", "automation", "pipeline", "infrastructure"],
        "evidence": {"infra_component": "Report generation pipeline", "improvement_description": "Automated report generation from solver outputs", "impact_metric": "48x faster: 4 hours to 5 minutes per report"},
        "ground_truth": "C6_infra",
    }),
    _item({
        "mission_id": "m794",
        "title": "Version control: added pre-commit hooks for code quality",
        "summary": "Added pre-commit hooks for linting, formatting, and type checking. Impact: reduced review cycle time by 40%.",
        "tags": ["pre-commit", "code-quality", "linting", "infrastructure"],
        "evidence": {"infra_component": "Pre-commit hooks", "improvement_description": "Automated linting, formatting, type checking before commit", "impact_metric": "40% reduction in review cycle time"},
        "ground_truth": "C6_infra",
    }),
    _item({
        "mission_id": "m799",
        "title": "Artifact store: centralized storage for simulation outputs and figures",
        "summary": "Built centralized artifact store for simulation outputs, figures, and reports. Baseline: scattered files across machines. Improved: single searchable store with metadata.",
        "tags": ["artifact-store", "storage", "metadata", "infrastructure"],
        "evidence": {"infra_component": "Artifact store", "improvement_description": "Centralized storage with metadata indexing", "impact_metric": "Search time from hours to seconds, 100% artifact retention"},
        "ground_truth": "C6_infra",
    }),
    # C6 Adversarial 1: minor tweak without measurable impact
    _item({
        "mission_id": "m800",
        "title": "Updated CI config file",
        "summary": "Updated the CI configuration file to use a newer Python version. No measurable impact on build time or success rate.",
        "tags": ["CI", "config", "python"],
        "evidence": {"infra_component": "CI config", "improvement_description": "Updated Python version", "impact_metric": "No measurable change"},
        "ground_truth": "C6_infra",
        "is_adversarial": True,
    }),
]


def load_golden_set() -> List[ContributionItem]:
    """Load golden set as ContributionItem objects."""
    items = []
    for d in GOLDEN_SET:
        items.append(ContributionItem(
            mission_id=d["mission_id"],
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            tags=d.get("tags", []),
            evidence=d.get("evidence", {}),
            metadata={"ground_truth": d["ground_truth"], "is_adversarial": d.get("is_adversarial", False)},
        ))
    return items
