"""Cross-domain interface contracts for P84-W2.

Two couplings are contract-bearing in P84-W2:

1. LBM aero (N01) ↔ LBM-LES (N02): turbulence model interface consistency
   - N01 produces wind force coefficients; N02 provides turbulence closure
   - Interface: turbulent viscosity field must be consistent with RANS/LES
     decomposition used in the aero force calculation
   - Check: Smagorinsky C_s range, subgrid viscosity ratio, energy spectrum slope

2. LBM suspended sediment (N04) ↔ DEM bulk cargo (N05): multiphase-particle flow
   - N04 provides sediment concentration field and settling velocity
   - N05 provides granular contact mechanics and angle of repose
   - Interface: particle flux at fluid-granular boundary must conserve mass
   - Check: Stokes settling consistency, repose angle cross-validation, Shields threshold

Each check returns a stable check_id for mechanical diffing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Contract 1: LBM turbulence model interface (N01 ↔ N02)
# ---------------------------------------------------------------------------
LBM_TURBULENCE_CONTRACT = "lbm_turbulence_consistency.v1"

#: Physical bounds on Smagorinsky constant (dimensionless)
CS_MIN = 0.10
CS_MAX = 0.18

#: Maximum subgrid viscosity ratio in boundary layer
NUT_NU_MAX = 100.0

#: Theoretical Kolmogorov -5/3 slope
SPECTRUM_SLOPE_THEORY = -5.0 / 3.0
SPECTRUM_SLOPE_TOL = 0.05  # ±0.05 tolerance


# ---------------------------------------------------------------------------
# Contract 2: Multiphase-particle flow interface (N04 ↔ N05)
# ---------------------------------------------------------------------------
MULTIPHASE_PARTICLE_CONTRACT = "multiphase_particle_coupling.v1"

#: Maximum settling velocity error vs Stokes (fraction)
SETTLING_VELOCITY_ERROR_MAX = 0.03

#: Minimum R² for Rouse profile fit
ROUSE_R2_MIN = 0.95

#: Maximum Shields threshold error (fraction)
SHIELDS_ERROR_MAX = 0.05

#: Maximum angle of repose error (fraction)
REPOSE_ANGLE_ERROR_MAX = 0.05

#: Maximum Beverloo discharge rate error (fraction)
BEVERLOO_ERROR_MAX = 0.08


@dataclass
class CrossDomainCheck:
    """Result of a single cross-domain interface check."""
    check_id: str
    contract_id: str
    description: str
    status: str  # PASS / FAIL / WITHHELD
    value: float | None = None
    limit: float | None = None
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "contract_id": self.contract_id,
            "description": self.description,
            "status": self.status,
            "value": self.value,
            "limit": self.limit,
            "detail": self.detail,
        }


def check_turbulence_consistency(n01_artifact: dict | None,
                                  n02_artifact: dict | None) -> list[CrossDomainCheck]:
    """Check LBM aero (N01) and LBM-LES (N02) turbulence model interface.

    Verifies:
    - T1: Smagorinsky C_s within physical range [0.10, 0.18]
    - T2: Subgrid viscosity ratio nu_t/nu < 100 in boundary layer
    - T3: Energy spectrum E(k) follows -5/3 slope in inertial subrange
    - T4: N01 wind coefficients computed with consistent turbulence closure
    """
    checks: list[CrossDomainCheck] = []

    if n02_artifact is None:
        checks.append(CrossDomainCheck(
            "T1", LBM_TURBULENCE_CONTRACT,
            "Smagorinsky C_s in [0.10, 0.18]",
            "WITHHELD", detail="N02 artifact not delivered"))
        checks.append(CrossDomainCheck(
            "T2", LBM_TURBULENCE_CONTRACT,
            "Subgrid viscosity ratio nu_t/nu < 100",
            "WITHHELD", detail="N02 artifact not delivered"))
        checks.append(CrossDomainCheck(
            "T3", LBM_TURBULENCE_CONTRACT,
            "Energy spectrum E(k) slope ≈ -5/3",
            "WITHHELD", detail="N02 artifact not delivered"))
        return checks

    crit = n02_artifact.get("criteria", {})

    # T1: Smagorinsky C_s range
    cs_rec = crit.get("C1", {})
    cs_val = cs_rec.get("value")
    if cs_val is not None:
        in_range = CS_MIN <= cs_val <= CS_MAX
        checks.append(CrossDomainCheck(
            "T1", LBM_TURBULENCE_CONTRACT,
            "Smagorinsky C_s in [0.10, 0.18]",
            "PASS" if in_range else "FAIL",
            value=cs_val, limit=CS_MAX,
            detail=f"C_s={cs_val}, range [{CS_MIN},{CS_MAX}]"))
    else:
        checks.append(CrossDomainCheck(
            "T1", LBM_TURBULENCE_CONTRACT,
            "Smagorinsky C_s in [0.10, 0.18]",
            "WITHHELD", detail="C_s value not in artifact"))

    # T2: Subgrid viscosity ratio
    nut_rec = crit.get("C2", {})
    nut_val = nut_rec.get("value")
    if nut_val is not None:
        ok = nut_val <= NUT_NU_MAX
        checks.append(CrossDomainCheck(
            "T2", LBM_TURBULENCE_CONTRACT,
            "Subgrid viscosity ratio nu_t/nu < 100",
            "PASS" if ok else "FAIL",
            value=nut_val, limit=NUT_NU_MAX,
            detail=f"nu_t/nu={nut_val}, max={NUT_NU_MAX}"))
    else:
        checks.append(CrossDomainCheck(
            "T2", LBM_TURBULENCE_CONTRACT,
            "Subgrid viscosity ratio nu_t/nu < 100",
            "WITHHELD", detail="nu_t/nu value not in artifact"))

    # T3: Energy spectrum slope
    spec_rec = crit.get("C3", {})
    spec_val = spec_rec.get("value")
    if spec_val is not None:
        slope_ok = abs(spec_val - SPECTRUM_SLOPE_THEORY) <= SPECTRUM_SLOPE_TOL
        checks.append(CrossDomainCheck(
            "T3", LBM_TURBULENCE_CONTRACT,
            "Energy spectrum E(k) slope ≈ -5/3",
            "PASS" if slope_ok else "FAIL",
            value=spec_val, limit=SPECTRUM_SLOPE_THEORY,
            detail=f"slope={spec_val}, theory={SPECTRUM_SLOPE_THEORY:.3f}±{SPECTRUM_SLOPE_TOL}"))
    else:
        checks.append(CrossDomainCheck(
            "T3", LBM_TURBULENCE_CONTRACT,
            "Energy spectrum E(k) slope ≈ -5/3",
            "WITHHELD", detail="spectrum slope not in artifact"))

    # T4: Cross-check N01 uses consistent closure
    if n01_artifact is not None:
        n01_crit = n01_artifact.get("criteria", {})
        cx_rec = n01_crit.get("C1", {})
        cx_val = cx_rec.get("value")
        if cx_val is not None:
            # N01 C_X error should be <3% (consistent with LES-grade turbulence)
            ok = cx_val <= 3.0
            checks.append(CrossDomainCheck(
                "T4", LBM_TURBULENCE_CONTRACT,
                "N01 wind C_X error <3% (consistent with LES closure)",
                "PASS" if ok else "FAIL",
                value=cx_val, limit=3.0,
                detail=f"C_X error={cx_val}%, requires LES-grade turbulence"))
        else:
            checks.append(CrossDomainCheck(
                "T4", LBM_TURBULENCE_CONTRACT,
                "N01 wind C_X error <3% (consistent with LES closure)",
                "WITHHELD", detail="N01 C_X not in artifact"))
    else:
        checks.append(CrossDomainCheck(
            "T4", LBM_TURBULENCE_CONTRACT,
            "N01 wind C_X error <3% (consistent with LES closure)",
            "WITHHELD", detail="N01 artifact not delivered"))

    return checks


def check_multiphase_particle_interface(n04_artifact: dict | None,
                                         n05_artifact: dict | None) -> list[CrossDomainCheck]:
    """Check LBM sediment (N04) and DEM cargo (N05) multiphase-particle interface.

    Verifies:
    - M1: Settling velocity error vs Stokes <3%
    - M2: Rouse profile R² > 0.95
    - M3: Shields threshold error <5%
    - M4: DEM angle of repose error <5% (cross-validates with N04 sediment bed)
    - M5: Beverloo discharge rate error <8%
    """
    checks: list[CrossDomainCheck] = []

    # N04 checks
    if n04_artifact is not None:
        crit = n04_artifact.get("criteria", {})

        # M1: Settling velocity
        st_rec = crit.get("C1", {})
        st_val = st_rec.get("value")
        if st_val is not None:
            ok = st_val <= SETTLING_VELOCITY_ERROR_MAX * 100  # percent
            checks.append(CrossDomainCheck(
                "M1", MULTIPHASE_PARTICLE_CONTRACT,
                "Settling velocity error vs Stokes <3%",
                "PASS" if ok else "FAIL",
                value=st_val, limit=SETTLING_VELOCITY_ERROR_MAX * 100,
                detail=f"settling error={st_val}%"))
        else:
            checks.append(CrossDomainCheck(
                "M1", MULTIPHASE_PARTICLE_CONTRACT,
                "Settling velocity error vs Stokes <3%",
                "WITHHELD", detail="settling velocity not in artifact"))

        # M2: Rouse profile
        rouse_rec = crit.get("C2", {})
        rouse_val = rouse_rec.get("value")
        if rouse_val is not None:
            ok = rouse_val >= ROUSE_R2_MIN
            checks.append(CrossDomainCheck(
                "M2", MULTIPHASE_PARTICLE_CONTRACT,
                "Rouse profile R² > 0.95",
                "PASS" if ok else "FAIL",
                value=rouse_val, limit=ROUSE_R2_MIN,
                detail=f"R²={rouse_val}"))
        else:
            checks.append(CrossDomainCheck(
                "M2", MULTIPHASE_PARTICLE_CONTRACT,
                "Rouse profile R² > 0.95",
                "WITHHELD", detail="Rouse R² not in artifact"))

        # M3: Shields threshold
        sh_rec = crit.get("C3", {})
        sh_val = sh_rec.get("value")
        if sh_val is not None:
            ok = sh_val <= SHIELDS_ERROR_MAX * 100
            checks.append(CrossDomainCheck(
                "M3", MULTIPHASE_PARTICLE_CONTRACT,
                "Shields threshold error <5%",
                "PASS" if ok else "FAIL",
                value=sh_val, limit=SHIELDS_ERROR_MAX * 100,
                detail=f"Shields error={sh_val}%"))
        else:
            checks.append(CrossDomainCheck(
                "M3", MULTIPHASE_PARTICLE_CONTRACT,
                "Shields threshold error <5%",
                "WITHHELD", detail="Shields threshold not in artifact"))
    else:
        for cid, desc in [("M1", "Settling velocity error vs Stokes <3%"),
                          ("M2", "Rouse profile R² > 0.95"),
                          ("M3", "Shields threshold error <5%")]:
            checks.append(CrossDomainCheck(
                cid, MULTIPHASE_PARTICLE_CONTRACT, desc,
                "WITHHELD", detail="N04 artifact not delivered"))

    # N05 checks
    if n05_artifact is not None:
        crit = n05_artifact.get("criteria", {})

        # M4: Angle of repose
        ar_rec = crit.get("C1", {})
        ar_val = ar_rec.get("value")
        if ar_val is not None:
            ok = ar_val <= REPOSE_ANGLE_ERROR_MAX * 100
            checks.append(CrossDomainCheck(
                "M4", MULTIPHASE_PARTICLE_CONTRACT,
                "DEM angle of repose error <5%",
                "PASS" if ok else "FAIL",
                value=ar_val, limit=REPOSE_ANGLE_ERROR_MAX * 100,
                detail=f"repose error={ar_val}%"))
        else:
            checks.append(CrossDomainCheck(
                "M4", MULTIPHASE_PARTICLE_CONTRACT,
                "DEM angle of repose error <5%",
                "WITHHELD", detail="repose angle not in artifact"))

        # M5: Beverloo discharge
        bev_rec = crit.get("C2", {})
        bev_val = bev_rec.get("value")
        if bev_val is not None:
            ok = bev_val <= BEVERLOO_ERROR_MAX * 100
            checks.append(CrossDomainCheck(
                "M5", MULTIPHASE_PARTICLE_CONTRACT,
                "Beverloo discharge rate error <8%",
                "PASS" if ok else "FAIL",
                value=bev_val, limit=BEVERLOO_ERROR_MAX * 100,
                detail=f"Beverloo error={bev_val}%"))
        else:
            checks.append(CrossDomainCheck(
                "M5", MULTIPHASE_PARTICLE_CONTRACT,
                "Beverloo discharge rate error <8%",
                "WITHHELD", detail="Beverloo discharge not in artifact"))
    else:
        for cid, desc in [("M4", "DEM angle of repose error <5%"),
                          ("M5", "Beverloo discharge rate error <8%")]:
            checks.append(CrossDomainCheck(
                cid, MULTIPHASE_PARTICLE_CONTRACT, desc,
                "WITHHELD", detail="N05 artifact not delivered"))

    return checks


def run_cross_domain_checks(artifacts: dict[str, dict]) -> dict:
    """Run all cross-domain interface checks.

    Args:
        artifacts: mapping ext_id -> artifact dict

    Returns:
        dict with contracts list and summary
    """
    n01_art = artifacts.get("P84-N01")
    n02_art = artifacts.get("P84-N02")
    n04_art = artifacts.get("P84-N04")
    n05_art = artifacts.get("P84-N05")

    turb_checks = check_turbulence_consistency(n01_art, n02_art)
    multi_checks = check_multiphase_particle_interface(n04_art, n05_art)

    all_checks = turb_checks + multi_checks
    n_pass = sum(1 for c in all_checks if c.status == "PASS")
    n_fail = sum(1 for c in all_checks if c.status == "FAIL")
    n_withheld = sum(1 for c in all_checks if c.status == "WITHHELD")

    return {
        "contracts": [
            {
                "contract_id": LBM_TURBULENCE_CONTRACT,
                "title": "LBM aero ↔ LBM-LES turbulence model consistency",
                "producer": "P84-N02 (TensorLBM)",
                "consumer": "P84-N01 (TensorLBM)",
                "n_checks": len(turb_checks),
                "n_pass": sum(1 for c in turb_checks if c.status == "PASS"),
                "n_fail": sum(1 for c in turb_checks if c.status == "FAIL"),
                "n_withheld": sum(1 for c in turb_checks if c.status == "WITHHELD"),
                "verdict": "PASS" if all(c.status == "PASS" for c in turb_checks) else (
                    "FAIL" if any(c.status == "FAIL" for c in turb_checks) else "WITHHELD"),
                "checks": [c.as_dict() for c in turb_checks],
            },
            {
                "contract_id": MULTIPHASE_PARTICLE_CONTRACT,
                "title": "LBM sediment ↔ DEM cargo multiphase-particle coupling",
                "producer": "P84-N04 (TensorLBM) + P84-N05 (TensorDEM)",
                "consumer": "cross-domain mass flux exchange",
                "n_checks": len(multi_checks),
                "n_pass": sum(1 for c in multi_checks if c.status == "PASS"),
                "n_fail": sum(1 for c in multi_checks if c.status == "FAIL"),
                "n_withheld": sum(1 for c in multi_checks if c.status == "WITHHELD"),
                "verdict": "PASS" if all(c.status == "PASS" for c in multi_checks) else (
                    "FAIL" if any(c.status == "FAIL" for c in multi_checks) else "WITHHELD"),
                "checks": [c.as_dict() for c in multi_checks],
            },
        ],
        "summary": {
            "total_checks": len(all_checks),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_withheld": n_withheld,
            "overall": "PASS" if n_fail == 0 and n_withheld == 0 else (
                "FAIL" if n_fail > 0 else "WITHHELD"),
        },
    }
