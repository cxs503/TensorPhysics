"""Chain-readiness verdict: the machine-readable gate the campaign consumes.

The gate is *declarative*: the criteria table below is the single source of
truth, the evaluator is written against it, and the verdict follows three rules

1. **WITHHELD** if any required interface product is absent (silence must never
   read as PASS - the chain has no verdict until all faces have spoken),
2. **FAIL** if a face is present but one of its numbers misses its threshold
   (``honest FAIL`` is never overridden),
3. **PASS** only when every face passes every criterion.

The output is ``chain_readiness.v1`` JSON with the per-face verdicts, the
measured numbers, the thresholds they were compared against and a sha256 of each
product so that the verdict is traceable to the exact evidence it judged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "SCHEMA",
    "GATE_ID",
    "REQUIRED_PRODUCTS",
    "CRITERIA",
    "Criterion",
    "evaluate_chain",
    "write_readiness",
]

SCHEMA = "chain_readiness.v1"
GATE_ID = "P35-N10-CHAIN"

#: Every one of these must be present before the chain can be judged.
REQUIRED_PRODUCTS = ("A", "B", "C", "E", "T", "N")

FACE_TITLES = {
    "A": "LBM free-surface pressure -> 6-DOF resultant force",
    "B": "6-DOF motion -> LBM moving/immersed boundary (geometry + normal velocity)",
    "C": "slamming pressure -> beam modal generalised force (Duhamel)",
    "E": "energy ledger over one wave period",
    "T": "time-synchronisation ledger",
    "N": "negative controls (no stabilisation / zero-order interpolation)",
}


@dataclass(frozen=True)
class Criterion:
    """One machine-checkable acceptance criterion."""

    face: str
    key: str
    op: str            # "<", "<=", ">", ">=", "=="
    threshold: float
    label: str

    def check(self, value: float) -> bool:
        v = float(value)
        if self.op == "<":
            return v < self.threshold
        if self.op == "<=":
            return v <= self.threshold
        if self.op == ">":
            return v > self.threshold
        if self.op == ">=":
            return v >= self.threshold
        if self.op == "==":
            return abs(v - self.threshold) <= 1e-12
        raise ValueError(f"unknown operator {self.op!r}")


#: The campaign contract (numbers as specified in the P35-N10 goal).
CRITERIA: tuple[Criterion, ...] = (
    Criterion("A", "force_rel_err", "<", 0.02, "|F_A - F_ref|/|F_ref| < 2 %"),
    Criterion("A", "observed_order", ">=", 1.0, "mesh refinement order >= 1"),
    Criterion("A", "reference_self_consistency", "<", 0.02, "reference is itself converged"),
    Criterion("B", "closed_flux_rel", "<", 1e-8, "closed-surface zero-flux identity"),
    Criterion("B", "volume_rel_err", "<", 0.02, "dV/dt = -int u.n dA within 2 %"),
    Criterion("B", "normal_velocity_rel_err", "<", 0.02, "delivered normal velocity within 2 %"),
    Criterion("B", "power_rel_err", "<", 0.02, "power reciprocity within 2 %"),
    Criterion("C", "modal_force_rel_err", "<", 0.03, "virtual work vs strip generalised force < 3 %"),
    Criterion("C", "duhamel_rel_err", "<", 0.03, "Duhamel vs RK4 modal response < 3 %"),
    Criterion("E", "wave_ledger_rel_err", "<", 0.03, "input work - (kinetic+potential+dissipation) < 3 %"),
    Criterion("E", "body_ledger_rel_err", "<", 0.03, "body work ledger closure < 3 %"),
    Criterion("T", "zero_offset_rel_err", "<", 0.03, "zero-offset residual < 3 %"),
    Criterion("T", "offset_residual", ">", 0.10, "dt/2 offset residual > 10 % (detected)"),
    Criterion("T", "zero_order_force_err", ">", 0.10, "zero-order interpolation force error > 10 %"),
    Criterion("T", "offset_detected", "==", 1.0, "dt/2 offset detector fires"),
    Criterion("T", "zero_order_detected", "==", 1.0, "zero-order detector fires"),
    Criterion("N", "divergence_detection_rate", "==", 1.0, "unstabilised coupling divergence detected 100 %"),
    Criterion("N", "stabilised_converged_rate", "==", 1.0, "Aitken/IQN-ILS converge 100 %"),
)


def _sha256(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def evaluate_chain(products: Mapping[str, Mapping | None]) -> dict:
    """Evaluate the chain readiness from the six interface products.

    ``products`` maps face keys (``"A"``/``"B"``/``"C"``/``"E"``/``"T"``/``"N"``)
    to the measured numbers, or ``None`` when the product is absent.
    """
    missing = [f for f in REQUIRED_PRODUCTS if products.get(f) is None]
    faces: dict[str, dict] = {}

    for face in REQUIRED_PRODUCTS:
        crits = [c for c in CRITERIA if c.face == face]
        prod = products.get(face)
        rows = []
        for c in crits:
            if prod is None or c.key not in prod:
                rows.append(
                    {
                        "key": c.key,
                        "label": c.label,
                        "op": c.op,
                        "threshold": c.threshold,
                        "value": None,
                        "passed": False,
                        "status": "missing",
                    }
                )
                continue
            val = float(prod[c.key])
            rows.append(
                {
                    "key": c.key,
                    "label": c.label,
                    "op": c.op,
                    "threshold": c.threshold,
                    "value": val,
                    "passed": bool(c.check(val)),
                    "status": "measured",
                }
            )
        if prod is None:
            verdict = "WITHHELD"
        elif all(r["passed"] for r in rows):
            verdict = "PASS"
        else:
            verdict = "FAIL"
        faces[face] = {
            "title": FACE_TITLES[face],
            "verdict": verdict,
            "present": prod is not None,
            "evidence_sha256": None if prod is None else _sha256(prod),
            "criteria": rows,
        }

    if missing:
        overall = "WITHHELD"
    elif any(f["verdict"] == "FAIL" for f in faces.values()):
        overall = "FAIL"
    else:
        overall = "PASS"

    blocking = [
        f"{face}:{r['key']}"
        for face, rec in faces.items()
        for r in rec["criteria"]
        if not r["passed"]
    ]
    return {
        "schema": SCHEMA,
        "gate_id": GATE_ID,
        "overall_status": overall,
        "chain_ready": overall == "PASS",
        "required_products": list(REQUIRED_PRODUCTS),
        "missing_products": missing,
        "blocking_criteria": blocking,
        "faces": faces,
    }


def write_readiness(verdict: dict, path) -> str:
    """Write the verdict JSON (deterministic ordering) and return the path."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(verdict, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(p)