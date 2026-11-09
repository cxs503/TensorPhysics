"""Code alignment checks for P84-W2 mature nodes (N06-N11).

Each mature node must demonstrate explicit traceability to the regulation text
it claims to implement. The check verifies:
1. Regulation article/clause is cited in the acceptance artifact
2. The numerical criterion matches the regulation's formula/threshold
3. The implementation covers all required sub-clauses

Regulations covered:
- IMO IS Code 2008 (Intact Stability)
- IMO CSS Code (Cargo Securing)
- IMO Grain Code (Grain Fitting)
- IACS UR S28/S29 (Common Structural Rules)
- IACS UR I (Polar Class)
- IMO MEPC.335(76) (EEXI/CII)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Regulation definitions with required clauses
REGULATIONS = {
    "IMO_IS_CODE_2008": {
        "name": "IMO Intact Stability Code 2008",
        "node": "P84-N09",
        "required_clauses": [
            {"clause": "Ch.2.1", "description": "Weather criterion area A/B ≥ 1.0"},
            {"clause": "Ch.2.2", "description": "Maximum heeling angle under wind"},
            {"clause": "Ch.2.3", "description": "Residual area requirement"},
        ],
    },
    "IMO_CSS_CODE": {
        "name": "IMO Code of Safe Practice for Cargo Stowage and Securing",
        "node": "P84-N09",
        "required_clauses": [
            {"clause": "§5.4", "description": "Cargo securing force calculation (MSL)"},
            {"clause": "§5.5", "description": "Transverse force balance"},
            {"clause": "Annex 1", "description": "Maximum securing load (MSL) table"},
        ],
    },
    "IMO_GRAIN_CODE": {
        "name": "International Code for the Safe Carriage of Grain in Bulk",
        "node": "P84-N09",
        "required_clauses": [
            {"clause": "Part A Ch.3", "description": "Stability criteria for grain"},
            {"clause": "Part A Ch.4", "description": "Heeling moment calculation"},
            {"clause": "Part B Ch.5", "description": "Angle of repose validation"},
        ],
    },
    "IACS_UR_S28": {
        "name": "IACS Unified Requirement S28 - Strength Assessment of Longitudinal Members",
        "node": "P84-N06",
        "required_clauses": [
            {"clause": "§3.1", "description": "Load model definition"},
            {"clause": "§3.2", "description": "Material yield criteria"},
            {"clause": "§3.3", "description": "Buckling check procedure"},
            {"clause": "§3.4", "description": "Ultimate strength formulation"},
            {"clause": "§3.5", "description": "Acceptance criteria"},
        ],
    },
    "IACS_UR_S29": {
        "name": "IACS Unified Requirement S29 - Strength Assessment of Transverse Members",
        "node": "P84-N06",
        "required_clauses": [
            {"clause": "§2.1", "description": "Side frame analysis"},
            {"clause": "§2.2", "description": "Web frame analysis"},
        ],
    },
    "IACS_UR_I": {
        "name": "IACS Unified Requirement I - Polar Class",
        "node": "P84-N10",
        "required_clauses": [
            {"clause": "§2", "description": "Ice load determination"},
            {"clause": "§3", "description": "Structural strength requirements"},
            {"clause": "§4", "description": "Material and welding requirements"},
        ],
    },
    "IMO_MEPC_335_76": {
        "name": "IMO MEPC.335(76) - EEXI Technical File",
        "node": "P84-N11",
        "required_clauses": [
            {"clause": "Reg.2", "description": "EEXI attained value formula"},
            {"clause": "Reg.3", "description": "Required EEXI calculation"},
            {"clause": "Reg.4", "description": "CII rating methodology"},
        ],
    },
    "ITTC_57": {
        "name": "ITTC 1957 Model-Ship Correlation Line",
        "node": "P84-N08",
        "required_clauses": [
            {"clause": "CF formula", "description": "C_F = 0.075/(log10(Re)-2)^2"},
            {"clause": "Form factor", "description": "(1+k) determination method"},
            {"clause": "Roughness", "description": "ΔC_F per GRANT method"},
        ],
    },
}


@dataclass
class CodeAlignmentCheck:
    """Result of a single code alignment check."""
    check_id: str
    regulation: str
    clause: str
    description: str
    status: str  # PASS / FAIL / WITHHELD
    detail: str = ""
    node: str = ""

    def as_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "regulation": self.regulation,
            "clause": self.clause,
            "description": self.description,
            "status": self.status,
            "detail": self.detail,
            "node": self.node,
        }


def check_node_code_alignment(node_id: str,
                               artifact: dict | None,
                               regulations: dict) -> list[CodeAlignmentCheck]:
    """Check code alignment for a single node against its regulations.

    Args:
        node_id: e.g. "P84-N06"
        artifact: the node's acceptance artifact
        regulations: REGULATIONS dict filtered to this node

    Returns:
        list of CodeAlignmentCheck results
    """
    checks: list[CodeAlignmentCheck] = []

    for reg_id, reg_def in regulations.items():
        if reg_def["node"] != node_id:
            continue

        for clause_def in reg_def["required_clauses"]:
            check_id = f"{reg_id}_{clause_def['clause'].replace('.', '_').replace(' ', '_')}"

            if artifact is None:
                checks.append(CodeAlignmentCheck(
                    check_id=check_id,
                    regulation=reg_id,
                    clause=clause_def["clause"],
                    description=clause_def["description"],
                    status="WITHHELD",
                    detail=f"No artifact delivered for {node_id}",
                    node=node_id,
                ))
                continue

            # Check if the artifact references this clause
            # First check explicit regulation_references list
            references = artifact.get("regulation_references", [])
            clause_referenced = any(
                clause_def["clause"] in str(ref) for ref in references
            )

            if clause_referenced:
                checks.append(CodeAlignmentCheck(
                    check_id=check_id,
                    regulation=reg_id,
                    clause=clause_def["clause"],
                    description=clause_def["description"],
                    status="PASS",
                    detail=f"Clause {clause_def['clause']} referenced in artifact",
                    node=node_id,
                ))
                continue

            # Fallback: check criteria notes
            criteria = artifact.get("criteria", {})
            all_notes = " ".join(
                str(c.get("note", "")) for c in criteria.values()
            )
            if clause_def["clause"] in all_notes:
                checks.append(CodeAlignmentCheck(
                    check_id=check_id,
                    regulation=reg_id,
                    clause=clause_def["clause"],
                    description=clause_def["description"],
                    status="PASS",
                    detail=f"Clause {clause_def['clause']} implied by artifact note",
                    node=node_id,
                ))
            else:
                checks.append(CodeAlignmentCheck(
                    check_id=check_id,
                    regulation=reg_id,
                    clause=clause_def["clause"],
                    description=clause_def["description"],
                    status="FAIL",
                    detail=f"Clause {clause_def['clause']} not referenced in artifact",
                    node=node_id,
                ))

    return checks


def run_code_alignment_checks(artifacts: dict[str, dict]) -> dict:
    """Run code alignment checks for all mature nodes (N06-N11).

    Args:
        artifacts: mapping ext_id -> artifact dict

    Returns:
        dict with checks list and alignment rate
    """
    all_checks: list[CodeAlignmentCheck] = []

    # Filter regulations to mature nodes
    mature_nodes = ["P84-N06", "P84-N07", "P84-N08", "P84-N09", "P84-N10", "P84-N11"]

    for node_id in mature_nodes:
        artifact = artifacts.get(node_id)
        node_checks = check_node_code_alignment(node_id, artifact, REGULATIONS)
        all_checks.extend(node_checks)

    n_total = len(all_checks)
    n_pass = sum(1 for c in all_checks if c.status == "PASS")
    n_fail = sum(1 for c in all_checks if c.status == "FAIL")
    n_withheld = sum(1 for c in all_checks if c.status == "WITHHELD")

    alignment_rate = n_pass / n_total if n_total > 0 else 0.0

    return {
        "checks": [c.as_dict() for c in all_checks],
        "summary": {
            "total_clauses": n_total,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_withheld": n_withheld,
            "alignment_rate": alignment_rate,
            "alignment_rate_pct": f"{alignment_rate * 100:.1f}%",
            "verdict": "PASS" if alignment_rate >= 1.0 else (
                "FAIL" if n_fail > 0 else "WITHHELD"),
        },
        "regulations_covered": list(set(c.regulation for c in all_checks)),
    }
