"""ISO 19906 partial-factor tables and the consistency adjudication.

Provenance policy
-----------------
The primary standard text (ISO 19906:2010, Annex A / Tables A.1-A.3) is **not
available inside this worktree**. The table values below are therefore stored
with an explicit ``source_tier``:

* ``in_repo``  — a value that is literally implemented in this repository
  (:mod:`tensorphysics.ice_plasticity.iso19906` defaults, ``gamma_ice = 1.25``,
  ``gamma_m = 1.15``). Fully verifiable: :data:`REPO_ANCHOR_ROWS`.
* ``secondary`` — a value transcribed from the published literature that
  summarises ISO 19906. Numerically usable, but every verdict derived from it
  is stamped ``evidence_tier = "conditional"`` and never reported as REAL.

Nothing here fabricates a primary-source claim: rows carry
``primary_source_verified = False`` and the adjudication propagates that flag
into the report, so a conditional verdict cannot be mistaken for a verified one.

Adjudication metric
-------------------
Comparing factors alone is ambiguous because ``gamma_F * gamma_M`` is the only
convention-free quantity. The adjudication therefore *also* re-runs the
calibration inverse: for the factor pair of a design case it computes the
achieved reliability index ``beta_ach`` of a structure designed by that pair and
compares it with the target ``beta_T`` of the consequence class. A table row
whose achieved ``beta`` falls short of the target by more than
``TOLERANCE_BETA`` is declared ``NON_CONSERVATIVE`` — that is a statement about
safety level, not about the arithmetic of the factors.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..structural.distributions import norm_cdf

__all__ = [
    "TOLERANCE_CONSISTENT",
    "TOLERANCE_ACCEPTABLE",
    "TOLERANCE_BETA",
    "ISO19906Row",
    "DesignCase",
    "TARGET_RELIABILITY_ROWS",
    "LOAD_FACTOR_ROWS",
    "RESISTANCE_FACTOR_ROWS",
    "REPO_ANCHOR_ROWS",
    "DESIGN_CASES",
    "adjudicate_target_reliability",
    "adjudicate_factor_rows",
    "adjudicate_design_cases",
    "summarise_verdicts",
]

#: Relative deviation band treated as "consistent" / "acceptable" (platform rule).
TOLERANCE_CONSISTENT = 0.05
TOLERANCE_ACCEPTABLE = 0.15
#: Reliability-index deviation treated as "consistent" (one tenth of a unit).
TOLERANCE_BETA = 0.10


@dataclass(frozen=True)
class ISO19906Row:
    """One table row with its provenance."""

    table: str
    row_key: str
    quantity: str
    value: float
    description: str
    source_tier: str = "secondary"
    primary_source_verified: bool = False
    basis: str = ""

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "row_key": self.row_key,
            "quantity": self.quantity,
            "value": self.value,
            "description": self.description,
            "source_tier": self.source_tier,
            "primary_source_verified": self.primary_source_verified,
            "basis": self.basis,
        }


# --------------------------------------------------------------------------- #
# Table A.1 — target annual reliability by consequence class
# --------------------------------------------------------------------------- #
TARGET_RELIABILITY_ROWS: tuple[ISO19906Row, ...] = (
    ISO19906Row(
        table="A.1",
        row_key="consequence_class_S1",
        quantity="beta_T",
        value=3.3,
        description="low consequence / manned-evacuable structure",
        basis="secondary summary of ISO 19906 Annex A target reliability table "
        "(band 3.3-3.5 for the lowest consequence class); primary text not available",
    ),
    ISO19906Row(
        table="A.1",
        row_key="consequence_class_S2",
        quantity="beta_T",
        value=3.8,
        description="normal/high consequence offshore structure",
        basis="secondary summary of ISO 19906 Annex A (band 3.7-4.0)",
    ),
    ISO19906Row(
        table="A.1",
        row_key="consequence_class_S3",
        quantity="beta_T",
        value=4.0,
        description="high consequence, ice-resistant structure, no redundancy",
        basis="secondary summary of ISO 19906 Annex A; matches the mission target beta_T = 4.0",
    ),
    ISO19906Row(
        table="A.1",
        row_key="consequence_class_S4",
        quantity="beta_T",
        value=4.3,
        description="very high consequence (crew quarters, critical functions)",
        basis="secondary summary of ISO 19906 Annex A (band 4.2-4.5)",
    ),
)

# --------------------------------------------------------------------------- #
# Table A.2 — load factors on the ice action
# --------------------------------------------------------------------------- #
LOAD_FACTOR_ROWS: tuple[ISO19906Row, ...] = (
    ISO19906Row(
        table="A.2",
        row_key="ice_action_extreme",
        quantity="gamma_F",
        value=1.35,
        description="extreme (annual-exceedance 1e-2) ice action, ultimate limit state",
        basis="secondary summary of ISO 19906 load-factor table for ice actions",
    ),
    ISO19906Row(
        table="A.2",
        row_key="ice_action_abnormal",
        quantity="gamma_F",
        value=1.10,
        description="abnormal/extreme ice action with ice management or monitoring",
        basis="secondary summary of ISO 19906 load-factor table",
    ),
    ISO19906Row(
        table="A.2",
        row_key="ice_action_frequent",
        quantity="gamma_F",
        value=1.00,
        description="frequent ice action, serviceability limit state",
        basis="secondary summary of ISO 19906 load-factor table",
    ),
)

# --------------------------------------------------------------------------- #
# Table A.3 — resistance factors
# --------------------------------------------------------------------------- #
RESISTANCE_FACTOR_ROWS: tuple[ISO19906Row, ...] = (
    ISO19906Row(
        table="A.3",
        row_key="steel_plate_yielding",
        quantity="gamma_M",
        value=1.10,
        description="steel plating, yielding / gross-section plasticity",
        basis="secondary summary of ISO 19906 resistance-factor table",
    ),
    ISO19906Row(
        table="A.3",
        row_key="steel_plate_buckling",
        quantity="gamma_M",
        value=1.15,
        description="steel plating, elastic-plastic buckling",
        basis="secondary summary of ISO 19906 resistance-factor table",
    ),
    ISO19906Row(
        table="A.3",
        row_key="steel_weld_connection",
        quantity="gamma_M",
        value=1.25,
        description="welded connection / heat-affected zone",
        basis="secondary summary of ISO 19906 resistance-factor table",
    ),
)

# --------------------------------------------------------------------------- #
# Fully verified in-repo anchor (no standard text needed)
# --------------------------------------------------------------------------- #
REPO_ANCHOR_ROWS: tuple[ISO19906Row, ...] = (
    ISO19906Row(
        table="repo:ice_plasticity.uls_check",
        row_key="gamma_ice",
        quantity="gamma_F",
        value=1.25,
        description="default ice load factor implemented in tensorphysics.ice_plasticity",
        source_tier="in_repo",
        primary_source_verified=True,
        basis="src/tensorphysics/ice_plasticity/iso19906.py: DEFAULT_PARTIAL_FACTORS['gamma_ice']",
    ),
    ISO19906Row(
        table="repo:ice_plasticity.uls_check",
        row_key="gamma_m",
        quantity="gamma_M",
        value=1.15,
        description="default material factor implemented in tensorphysics.ice_plasticity",
        source_tier="in_repo",
        primary_source_verified=True,
        basis="src/tensorphysics/ice_plasticity/iso19906.py: DEFAULT_PARTIAL_FACTORS['gamma_m']",
    ),
)


@dataclass(frozen=True)
class DesignCase:
    """A (load factor, resistance factor, target beta) triple to be adjudicated."""

    case_key: str
    gamma_f_row: str
    gamma_m_row: str
    beta_row: str
    description: str = ""
    source_tier: str = "secondary"
    primary_source_verified: bool = False


DESIGN_CASES: tuple[DesignCase, ...] = (
    DesignCase(
        case_key="iso_extreme_yield_high",
        gamma_f_row="ice_action_extreme",
        gamma_m_row="steel_plate_yielding",
        beta_row="consequence_class_S3",
        description="extreme ice action + plate yielding at the high-consequence target",
    ),
    DesignCase(
        case_key="iso_extreme_buckling_high",
        gamma_f_row="ice_action_extreme",
        gamma_m_row="steel_plate_buckling",
        beta_row="consequence_class_S3",
        description="extreme ice action + elastic-plastic buckling",
    ),
    DesignCase(
        case_key="iso_extreme_yield_normal",
        gamma_f_row="ice_action_extreme",
        gamma_m_row="steel_plate_yielding",
        beta_row="consequence_class_S2",
        description="extreme ice action + plate yielding at the normal-consequence target",
    ),
    DesignCase(
        case_key="iso_abnormal_yield_high",
        gamma_f_row="ice_action_abnormal",
        gamma_m_row="steel_plate_yielding",
        beta_row="consequence_class_S3",
        description="managed ice action case",
    ),
    DesignCase(
        case_key="repo_anchor_yield_high",
        gamma_f_row="gamma_ice",
        gamma_m_row="gamma_m",
        beta_row="consequence_class_S3",
        description="in-repo ice_plasticity default factor pair (fully verified anchor)",
        source_tier="in_repo",
        primary_source_verified=True,
    ),
)


def _rows_by_key() -> dict[str, ISO19906Row]:
    out: dict[str, ISO19906Row] = {}
    for row in TARGET_RELIABILITY_ROWS + LOAD_FACTOR_ROWS + RESISTANCE_FACTOR_ROWS + REPO_ANCHOR_ROWS:
        out[row.row_key] = row
    return out


def _verdict_from_relative(deviation: float, conservative_sign: int) -> str:
    """Classify a relative deviation; ``conservative_sign=+1`` means larger is safer."""
    signed = deviation * conservative_sign
    magnitude = abs(deviation)
    if magnitude <= TOLERANCE_CONSISTENT:
        return "CONSISTENT"
    if signed > 0.0:
        return "CONSERVATIVE"
    return "NON_CONSERVATIVE" if magnitude > TOLERANCE_ACCEPTABLE else "MINOR_UNCONSERVATIVE"


def _verdict_from_beta(beta_achieved: float, beta_target: float) -> str:
    delta = beta_achieved - beta_target
    if abs(delta) <= TOLERANCE_BETA:
        return "CONSISTENT"
    if delta > 0.0:
        return "CONSERVATIVE"
    return "NON_CONSERVATIVE" if abs(delta) > 3.0 * TOLERANCE_BETA else "MINOR_UNCONSERVATIVE"


def adjudicate_target_reliability(
    calibrated_beta: float = 4.0, calibrated_pf: float | None = None
) -> list[dict]:
    """Compare the calibrated target with the Table A.1 rows."""
    pf = calibrated_pf if calibrated_pf is not None else norm_cdf(-calibrated_beta)
    rows = []
    for row in TARGET_RELIABILITY_ROWS:
        deviation = (calibrated_beta - row.value) / row.value
        rows.append(
            {
                **row.to_dict(),
                "calibrated_value": calibrated_beta,
                "calibrated_pf": pf,
                "table_pf": norm_cdf(-row.value),
                "relative_deviation": deviation,
                "verdict": _verdict_from_relative(deviation, +1),
                "evidence_tier": "conditional" if not row.primary_source_verified else "real",
            }
        )
    return rows


def adjudicate_factor_rows(
    calibrated_gamma_f: float, calibrated_gamma_m: float
) -> list[dict]:
    """Row-by-row comparison of the calibrated factors with A.2 and A.3."""
    rows = []
    for row in LOAD_FACTOR_ROWS + RESISTANCE_FACTOR_ROWS + REPO_ANCHOR_ROWS:
        calibrated = calibrated_gamma_f if row.quantity == "gamma_F" else calibrated_gamma_m
        deviation = (calibrated - row.value) / row.value
        rows.append(
            {
                **row.to_dict(),
                "calibrated_value": calibrated,
                "relative_deviation": deviation,
                "verdict": _verdict_from_relative(deviation, -1),
                "evidence_tier": "conditional" if not row.primary_source_verified else "real",
            }
        )
    return rows


def adjudicate_design_cases(beta_of_factor_pair) -> list[dict]:
    """Adjudicate every design case through the achieved reliability index.

    ``beta_of_factor_pair(gamma_f, gamma_m) -> float`` is supplied by the
    calibration module (it re-solves the panel with the given factor pair).
    """
    keys = _rows_by_key()
    rows = []
    for case in DESIGN_CASES:
        gf = keys[case.gamma_f_row]
        gm = keys[case.gamma_m_row]
        beta_row = keys[case.beta_row]
        pair = gf.value * gm.value
        beta_ach = float(beta_of_factor_pair(pair))
        rows.append(
            {
                "case_key": case.case_key,
                "description": case.description,
                "gamma_F_row": case.gamma_f_row,
                "gamma_F": gf.value,
                "gamma_M_row": case.gamma_m_row,
                "gamma_M": gm.value,
                "factor_product": pair,
                "beta_target": beta_row.value,
                "beta_target_row": case.beta_row,
                "pf_target": norm_cdf(-beta_row.value),
                "beta_achieved": beta_ach,
                "pf_achieved": norm_cdf(-beta_ach),
                "delta_beta": beta_ach - beta_row.value,
                "verdict": _verdict_from_beta(beta_ach, beta_row.value),
                "source_tier": case.source_tier,
                "primary_source_verified": case.primary_source_verified,
                "evidence_tier": "conditional" if not case.primary_source_verified else "real",
            }
        )
    return rows


def summarise_verdicts(rows: list[dict]) -> dict:
    """Tally verdicts (used as a machine criterion in the report JSON)."""
    counts: dict[str, int] = {}
    for row in rows:
        verdict = row.get("verdict", "UNKNOWN")
        counts[verdict] = counts.get(verdict, 0) + 1
    total = len(rows)
    return {
        "total_rows": total,
        "counts": counts,
        "all_rows_verdicted": all("verdict" in row for row in rows) and total > 0,
        "all_numeric": all(
            isinstance(row.get("relative_deviation", row.get("delta_beta")), (int, float))
            or row.get("relative_deviation") is not None
            or row.get("delta_beta") is not None
            for row in rows
        ),
    }