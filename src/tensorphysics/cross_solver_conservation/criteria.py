"""Machine-adjudicable criteria for the cross-solver conservation audit.

Every criterion is frozen with an id, the physical statement it encodes, a
threshold and a direction, so a stored audit JSON can be re-gated without
re-running the physics (same pattern as ``coupling_energy.acceptance``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict

__all__ = ["Criterion", "CRITERIA", "CRITERIA_DIGEST", "evaluate", "format_table"]


@dataclass(frozen=True)
class Criterion:
    cid: str
    statement: str
    metric: str
    threshold: float
    direction: str  # "<=" or ">="
    unit: str = "-"
    stage: str = ""


CRITERIA = (
    Criterion(
        "C1_fem_dem_mapping",
        "FEM strain energy delivered to DEM equals the DEM chain elastic energy "
        "(conservative field mapping)",
        "mapping_rel_error",
        0.01,
        "<=",
        unit="rel",
        stage="fem_dem",
    ),
    Criterion(
        "C2_fem_dem_energy_closure",
        "FEM->DEM fracture handoff: U_FEM = KE + residual elastic + cohesive "
        "dissipation + damping + handoff deficit",
        "closure_rel_error",
        0.01,
        "<=",
        unit="rel",
        stage="fem_dem",
    ),
    Criterion(
        "C3_cohesive_energy_bookkeeping",
        "analytic cohesive dissipation matches the numerically integrated path "
        "work int M dtheta",
        "diss_path_vs_analytic_rel",
        0.01,
        "<=",
        unit="rel",
        stage="fem_dem",
    ),
    Criterion(
        "C4_crack_arrest_resolution",
        "DEM chain resolves the cohesive process zone (l_b / l_pz) so the crack "
        "arrest length is mesh independent",
        "crack_length_spread_across_mesh",
        0.10,
        "<=",
        unit="rel",
        stage="fem_dem",
    ),
    Criterion(
        "C5_dem_lbm_momentum_closure",
        "DEM->LBM: ice momentum deficit equals the discrete wall impulse "
        "(Ladd exchange)",
        "closure_ice_vs_ladd",
        0.02,
        "<=",
        unit="rel",
        stage="dem_lbm",
    ),
    Criterion(
        "C6_lbm_wall_flux_consistency",
        "LBM moving-wall momentum exchange equals the macroscopic wall shear "
        "stress integral",
        "closure_shear_vs_ladd",
        0.02,
        "<=",
        unit="rel",
        stage="dem_lbm",
    ),
    Criterion(
        "C7_lbm_momentum_identity",
        "discrete momentum balance P(t)-P(0) = force - wall impulse (boundary "
        "implementation audit)",
        "balance_error_rel",
        1e-8,
        "<=",
        unit="rel",
        stage="dem_lbm",
    ),
    Criterion(
        "C8_lbm_poiseuille_profile",
        "LBM channel velocity profile against the analytic Poiseuille solution",
        "profile_rel_error",
        0.03,
        "<=",
        unit="rel",
        stage="dem_lbm",
    ),
    Criterion(
        "C9_lbm_wall_shear_vs_analytic",
        "steady wall shear from the discrete exchange against rho g h / 2",
        "tau_ladd_vs_analytic_rel",
        0.02,
        "<=",
        unit="rel",
        stage="dem_lbm",
    ),
    Criterion(
        "C10_multirate_work_error",
        "multirate field exchange (DEM 1 kHz -> LBM 10 kHz) preserves the "
        "exchanged work within the energy gate",
        "work_rel_cubic",
        0.01,
        "<=",
        unit="rel",
        stage="multirate",
    ),
    Criterion(
        "C11_multirate_impulse_error",
        "multirate field exchange preserves the exchanged impulse within the "
        "momentum gate",
        "impulse_rel_cubic",
        0.02,
        "<=",
        unit="rel",
        stage="multirate",
    ),
    Criterion(
        "C12_chain_energy_closure",
        "end-to-end FEM->DEM->LBM energy ledger closes",
        "chain_closure_rel",
        0.01,
        "<=",
        unit="rel",
        stage="chain",
    ),
)


def _digest(criteria) -> str:
    blob = json.dumps([asdict(c) for c in criteria], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


CRITERIA_DIGEST = _digest(CRITERIA)


def evaluate(metrics: dict, criteria=CRITERIA) -> list:
    """Return one outcome record per criterion (PASS / FAIL / WITHHELD)."""
    out = []
    for c in criteria:
        val = metrics.get(c.metric)
        if val is None:
            out.append(
                {"cid": c.cid, "status": "WITHHELD", "value": None,
                 "threshold": c.threshold, "statement": c.statement}
            )
            continue
        ok = val <= c.threshold if c.direction == "<=" else val >= c.threshold
        out.append(
            {
                "cid": c.cid,
                "status": "PASS" if ok else "FAIL",
                "value": float(val),
                "threshold": c.threshold,
                "direction": c.direction,
                "statement": c.statement,
                "stage": c.stage,
            }
        )
    return out


def format_table(outcomes) -> str:
    lines = ["| criterion | status | value | threshold |", "|---|---|---|---|"]
    for o in outcomes:
        v = "-" if o["value"] is None else f"{o['value']:.3e}"
        lines.append(f"| {o['cid']} | {o['status']} | {v} | {o['threshold']:.1e} |")
    return "\n".join(lines)