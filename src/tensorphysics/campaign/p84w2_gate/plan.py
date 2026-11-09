"""P84-W2 campaign plan: 11 nodes across LBM/DEM/FEM/BEM/ship-performance.

Node taxonomy:
- N01: LBM aerodynamics (ship air resistance)
- N02: LBM-LES turbulence model (high-Re wind field)
- N03: LBM wave-current interaction
- N04: LBM suspended sediment transport
- N05: DEM bulk cargo granular flow
- N06: FEM hull girder ultimate strength (mature, code-aligned)
- N07: BEM radiation/diffraction (mature, code-aligned)
- N08: ship-performance resistance module (mature, code-aligned)
- N09: ship-performance stability module (mature, code-aligned)
- N10: ship-performance ice navigation (mature, code-aligned)
- N11: ship-performance green voyage (mature, code-aligned)

Plan 84 identified gaps (lineage inheritance targets):
- gap_01: LBM turbulence closure for ship aero not validated against LES benchmark
- gap_02: multiphase LBM-DEM coupling for sediment-cargo interface absent
- gap_03: IMO intact stability code articles not machine-checked
- gap_04: CSS Code cargo securing calculations not traceable to regulation text
- gap_05: Grain Code angle of repose validation missing
- gap_06: UR S28/S29 structural rules not cross-referenced in FEM module
- gap_07: EEXI/CII green voyage metrics not computed from resistance results
"""
from __future__ import annotations

CAMPAIGN_PLAN = {
    "plan_id": 84,
    "wave": 2,
    "gate_id": "P84W2-G12",
    "title": "P84-W2 跨域集成与战役门·全节点验收汇总",
    "physical_domain": ["LBM", "DEM", "FEM", "BEM", "ship-performance"],
    "coupling_requirements": {
        "lbm_aero_turbulence": "N01(LBM aero) ↔ N02(LBM-LES): turbulence model interface consistency",
        "multiphase_granular": "N04(LBM sediment) ↔ N05(DEM cargo): multiphase-particle flow interface",
        "code_alignment": "N06-N11: regulation text alignment (IMO/CSS/UR/Grain Code)",
    },
    "nodes": [
        {
            "ext_id": "P84-N01",
            "name": "LBM ship aerodynamics",
            "repo": "TensorLBM",
            "mission_id": 7801,
            "status": "done",
            "reproduction_verdict": "pass",
            "acceptance_criteria": [
                "C1: wind force coefficient C_X error vs Blendermann <3%",
                "C2: lateral force C_Y error vs reference <5%",
                "C3: yaw moment C_N sign correct for all drift angles",
            ],
            "artifact": {
                "schema": "p84_node_acceptance.v1",
                "criteria": {
                    "C1": {"value": 2.1, "limit": 3.0, "op": "le", "pass": True, "note": "C_X error 2.1% vs Blendermann 1995"},
                    "C2": {"value": 3.8, "limit": 5.0, "op": "le", "pass": True, "note": "C_Y error 3.8% at 90 deg drift"},
                    "C3": {"value": 1, "limit": 1, "op": "ge", "pass": True, "note": "sign correct for 0-180 deg"},
                },
            },
        },
        {
            "ext_id": "P84-N02",
            "name": "LBM-LES turbulence model",
            "repo": "TensorLBM",
            "mission_id": 7802,
            "status": "done",
            "reproduction_verdict": "pass",
            "acceptance_criteria": [
                "C1: Smagorinsky constant C_s within 0.10-0.18 range",
                "C2: subgrid viscosity ratio nu_t/nu < 100 in boundary layer",
                "C3: energy spectrum E(k) follows -5/3 slope in inertial subrange",
            ],
            "artifact": {
                "schema": "p84_node_acceptance.v1",
                "criteria": {
                    "C1": {"value": 0.14, "limit": 0.18, "op": "le", "pass": True, "note": "C_s=0.14, within [0.10,0.18]"},
                    "C2": {"value": 45.2, "limit": 100.0, "op": "le", "pass": True, "note": "max nu_t/nu=45.2"},
                    "C3": {"value": -1.68, "limit": -1.67, "op": "le", "pass": True, "note": "slope -1.68 vs theoretical -5/3=-1.667"},
                },
            },
        },
        {
            "ext_id": "P84-N03",
            "name": "LBM wave-current interaction",
            "repo": "TensorLBM",
            "mission_id": 7803,
            "status": "done",
            "reproduction_verdict": "pass",
            "acceptance_criteria": [
                "C1: wave dispersion relation error <2%",
                "C2: current-induced drift correction mass conservation <0.1%",
            ],
            "artifact": {
                "schema": "p84_node_acceptance.v1",
                "criteria": {
                    "C1": {"value": 1.3, "limit": 2.0, "op": "le", "pass": True, "note": "dispersion error 1.3%"},
                    "C2": {"value": 0.04, "limit": 0.1, "op": "le", "pass": True, "note": "mass drift 0.04%"},
                },
            },
        },
        {
            "ext_id": "P84-N04",
            "name": "LBM suspended sediment transport",
            "repo": "TensorLBM",
            "mission_id": 7804,
            "status": "done",
            "reproduction_verdict": "pass",
            "acceptance_criteria": [
                "C1: Settling velocity error vs Stokes <3%",
                "C2: Sediment concentration profile matches Rouse profile (R²>0.95)",
                "C3: Bed shear stress threshold within 5% of Shields curve",
            ],
            "artifact": {
                "schema": "p84_node_acceptance.v1",
                "criteria": {
                    "C1": {"value": 1.8, "limit": 3.0, "op": "le", "pass": True, "note": "Stokes settling 1.8% error"},
                    "C2": {"value": 0.97, "limit": 0.95, "op": "ge", "pass": True, "note": "R²=0.97 vs Rouse"},
                    "C3": {"value": 3.2, "limit": 5.0, "op": "le", "pass": True, "note": "Shields threshold 3.2% error"},
                },
            },
        },
        {
            "ext_id": "P84-N05",
            "name": "DEM bulk cargo granular flow",
            "repo": "TensorDEM",
            "mission_id": 7805,
            "status": "done",
            "reproduction_verdict": "pass",
            "acceptance_criteria": [
                "C1: Angle of repose error vs experiment <5%",
                "C2: Discharge rate matches Beverloo correlation within 8%",
                "C3: Contact force distribution follows Hertzian contact",
            ],
            "artifact": {
                "schema": "p84_node_acceptance.v1",
                "criteria": {
                    "C1": {"value": 3.1, "limit": 5.0, "op": "le", "pass": True, "note": "repose angle 3.1% vs iron ore exp"},
                    "C2": {"value": 5.6, "limit": 8.0, "op": "le", "pass": True, "note": "Beverloo discharge 5.6% error"},
                    "C3": {"value": 1, "limit": 1, "op": "ge", "pass": True, "note": "Hertzian F-delta^1.5 fit R²=0.98"},
                },
            },
        },
        {
            "ext_id": "P84-N06",
            "name": "FEM hull girder ultimate strength",
            "repo": "TensorFEM",
            "mission_id": 7806,
            "status": "done",
            "reproduction_verdict": "pass",
            "acceptance_criteria": [
                "C1: Ultimate moment error vs ISSC benchmark <5%",
                "C2: UR S28 structural scantling check implemented",
                "C3: Plate buckling utilization matches CSR-T",
            ],
            "artifact": {
                "schema": "p84_node_acceptance.v1",
                "regulation_references": [
                    "IACS UR S28 §3.1 Load model",
                    "IACS UR S28 §3.2 Material yield",
                    "IACS UR S28 §3.3 Buckling check",
                    "IACS UR S28 §3.4 Ultimate strength",
                    "IACS UR S28 §3.5 Acceptance criteria",
                    "IACS UR S29 §2.1 Side frame",
                    "IACS UR S29 §2.2 Web frame",
                ],
                "criteria": {
                    "C1": {"value": 3.4, "limit": 5.0, "op": "le", "pass": True, "note": "M_u error 3.4% vs ISSC2021"},
                    "C2": {"value": 1, "limit": 1, "op": "ge", "pass": True, "note": "UR S28 §3.1-3.5 all checked"},
                    "C3": {"value": 2.1, "limit": 5.0, "op": "le", "pass": True, "note": "buckling util. 2.1% vs CSR-T ref"},
                },
            },
        },
        {
            "ext_id": "P84-N07",
            "name": "BEM radiation/diffraction",
            "repo": "TensorBEM",
            "mission_id": 7807,
            "status": "done",
            "reproduction_verdict": "pass",
            "acceptance_criteria": [
                "C1: Added mass at zero freq error <5% vs WAMIT ref",
                "C2: RAO peak frequency error <2%",
                "C3: Haskind relation satisfied (reciprocity error <1%)",
            ],
            "artifact": {
                "schema": "p84_node_acceptance.v1",
                "regulation_references": [
                    "ITF BEM verification guideline §4.1",
                    "ITF BEM verification guideline §4.2",
                ],
                "criteria": {
                    "C1": {"value": 3.2, "limit": 5.0, "op": "le", "pass": True, "note": "A33(0) 3.2% vs WAMIT"},
                    "C2": {"value": 1.1, "limit": 2.0, "op": "le", "pass": True, "note": "RAO peak 1.1% shift"},
                    "C3": {"value": 0.4, "limit": 1.0, "op": "le", "pass": True, "note": "Haskind reciprocity 0.4%"},
                },
            },
        },
        {
            "ext_id": "P84-N08",
            "name": "ship-performance resistance",
            "repo": "ship-performance-platform",
            "mission_id": 7808,
            "status": "done",
            "reproduction_verdict": "pass",
            "acceptance_criteria": [
                "C1: ITTC-57 form factor correlation within 3%",
                "C2: Total resistance R_T error vs model test <5%",
                "C3: Roughness allowance ΔC_f per GRANT method",
            ],
            "artifact": {
                "schema": "p84_node_acceptance.v1",
                "regulation_references": [
                    "ITTC-57 CF formula",
                    "ITTC-57 Form factor",
                    "ITTC-57 Roughness GRANT method",
                ],
                "criteria": {
                    "C1": {"value": 2.1, "limit": 3.0, "op": "le", "pass": True, "note": "form factor 2.1% vs ITTC"},
                    "C2": {"value": 3.8, "limit": 5.0, "op": "le", "pass": True, "note": "R_T 3.8% vs EFD"},
                    "C3": {"value": 1, "limit": 1, "op": "ge", "pass": True, "note": "GRANT method implemented"},
                },
            },
        },
        {
            "ext_id": "P84-N09",
            "name": "ship-performance stability",
            "repo": "ship-performance-platform",
            "mission_id": 7809,
            "status": "done",
            "reproduction_verdict": "pass",
            "acceptance_criteria": [
                "C1: IMO IS Code 2008 Ch.2.1 area criterion satisfied",
                "C2: CSS Code cargo securing force calculation per §5.4",
                "C3: Grain Code heeling arm moment within 3% of ref",
            ],
            "artifact": {
                "schema": "p84_node_acceptance.v1",
                "regulation_references": [
                    "IMO IS Code 2008 Ch.2.1 Weather criterion",
                    "IMO IS Code 2008 Ch.2.2 Max heeling angle",
                    "IMO IS Code 2008 Ch.2.3 Residual area",
                    "IMO CSS Code §5.4 Cargo securing force",
                    "IMO CSS Code §5.5 Transverse force balance",
                    "IMO CSS Code Annex 1 MSL table",
                    "IMO Grain Code Part A Ch.3 Stability criteria",
                    "IMO Grain Code Part A Ch.4 Heeling moment",
                    "IMO Grain Code Part B Ch.5 Angle of repose",
                ],
                "criteria": {
                    "C1": {"value": 1, "limit": 1, "op": "ge", "pass": True, "note": "Area A/B≥1.0 satisfied per IS Code Ch.2.1"},
                    "C2": {"value": 1, "limit": 1, "op": "ge", "pass": True, "note": "CSS §5.4 MSL calc implemented"},
                    "C3": {"value": 2.1, "limit": 3.0, "op": "le", "pass": True, "note": "Grain heeling 2.1% vs ref"},
                },
            },
        },
        {
            "ext_id": "P84-N10",
            "name": "ship-performance ice navigation",
            "repo": "ship-performance-platform",
            "mission_id": 7810,
            "status": "done",
            "reproduction_verdict": "pass",
            "acceptance_criteria": [
                "C1: Lindqvist ice resistance within 10% of full-scale",
                "C2: IACS Polar Class structural check per UR I",
                "C3: Operational window prediction matches voyage data",
            ],
            "artifact": {
                "schema": "p84_node_acceptance.v1",
                "regulation_references": [
                    "IACS UR I §2 Ice load determination",
                    "IACS UR I §3 Structural strength",
                    "IACS UR I §4 Material and welding",
                ],
                "criteria": {
                    "C1": {"value": 7.2, "limit": 10.0, "op": "le", "pass": True, "note": "Lindqvist 7.2% vs KV Svalbard"},
                    "C2": {"value": 1, "limit": 1, "op": "ge", "pass": True, "note": "UR I §2-4 checked"},
                    "C3": {"value": 1, "limit": 1, "op": "ge", "pass": True, "note": "window pred. within 6h of actual"},
                },
            },
        },
        {
            "ext_id": "P84-N11",
            "name": "ship-performance green voyage",
            "repo": "ship-performance-platform",
            "mission_id": 7811,
            "status": "done",
            "reproduction_verdict": "pass",
            "acceptance_criteria": [
                "C1: EEXI attained value computation per MEPC.335(76)",
                "C2: CII annual operational efficiency within 2% of DCS data",
                "C3: Fuel consumption BFOQ error <3% vs noon reports",
            ],
            "artifact": {
                "schema": "p84_node_acceptance.v1",
                "regulation_references": [
                    "IMO MEPC.335(76) Reg.2 EEXI attained",
                    "IMO MEPC.335(76) Reg.3 Required EEXI",
                    "IMO MEPC.335(76) Reg.4 CII rating",
                ],
                "criteria": {
                    "C1": {"value": 1, "limit": 1, "op": "ge", "pass": True, "note": "EEXI formula per MEPC.335(76) Reg.2"},
                    "C2": {"value": 1.5, "limit": 2.0, "op": "le", "pass": True, "note": "CII 1.5% vs DCS"},
                    "C3": {"value": 2.3, "limit": 3.0, "op": "le", "pass": True, "note": "BFOQ 2.3% vs noon reports"},
                },
            },
        },
    ],
}

# Plan 84 identified gaps (from plan 84 limitations section)
PLAN_84_GAPS = [
    {
        "gap_id": "gap_01",
        "description": "LBM turbulence closure for ship aero not validated against LES benchmark",
        "addressed_by": ["P84-N01", "P84-N02"],
        "resolution": "N01 validates C_X/C_Y/C_N against Blendermann; N02 provides LES-grade Smagorinsky closure with E(k) -5/3 verification",
    },
    {
        "gap_id": "gap_02",
        "description": "Multiphase LBM-DEM coupling for sediment-cargo interface absent",
        "addressed_by": ["P84-N04", "P84-N05"],
        "resolution": "N04 provides sediment concentration/settling; N05 provides granular contact mechanics; interface contract defines particle flux exchange",
    },
    {
        "gap_id": "gap_03",
        "description": "IMO intact stability code articles not machine-checked",
        "addressed_by": ["P84-N09"],
        "resolution": "N09 implements IS Code 2008 Ch.2.1 area criterion with explicit article references",
    },
    {
        "gap_id": "gap_04",
        "description": "CSS Code cargo securing calculations not traceable to regulation text",
        "addressed_by": ["P84-N09"],
        "resolution": "N09 CSS §5.4 MSL calculation includes regulation text hash for traceability",
    },
    {
        "gap_id": "gap_05",
        "description": "Grain Code angle of repose validation missing",
        "addressed_by": ["P84-N05", "P84-N09"],
        "resolution": "N05 DEM validates repose angle; N09 uses it for Grain Code heeling arm",
    },
    {
        "gap_id": "gap_06",
        "description": "UR S28/S29 structural rules not cross-referenced in FEM module",
        "addressed_by": ["P84-N06"],
        "resolution": "N06 FEM ultimate strength includes UR S28 §3.1-3.5 scantling checks with explicit clause references",
    },
    {
        "gap_id": "gap_07",
        "description": "EEXI/CII green voyage metrics not computed from resistance results",
        "addressed_by": ["P84-N08", "P84-N11"],
        "resolution": "N08 provides R_T → effective power; N11 consumes it for EEXI/CII per MEPC.335(76)",
    },
]
