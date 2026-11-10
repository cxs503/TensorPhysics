"""Cross-repository benchmark JSON adapters: solver evidence as a UQ input.

The UQ pipeline treats a solver as a black box ``points (n, d) -> values (n,)``.
For the *validation* half of a certificate there is a second, complementary
source of evidence: the benchmark artefacts a solver repository already
publishes (``TensorLBM`` marine/GPU validation JSON, ``TensorFVM``
verification summaries, ``TensorFEM`` benchmark evidence).  Those files carry

* a measured quantity of interest,
* the reference / analytic value it is compared with,
* the repository's own pass/fail gate,

so they are exactly the ``relative_error`` term that
:func:`tensorphysics.uq.gate.engineering_error_gate` needs next to the UQ band.

This module parses those artefacts **without importing the solver repositories**
(paths in, JSON out), pins each read with the file's sha256, and provides two
compositions:

``combine_budget``
    GUM-style budget: the UQ band (input spread + surrogate error, converted to
    a standard uncertainty) and the benchmark comparison error (uniform
    assumption, ``u = e/sqrt(3)``) combined in quadrature, then expanded with
    ``k = 2``.
``gate_with_benchmark``
    Hands the parsed comparison error to the existing pre-evidence gate and
    attaches the budget and the artefact digests, so the gate decision is
    traceable to the exact benchmark file it consumed.

Supported artefacts (schema-detected, no configuration needed)
--------------------------------------------------------------
``TensorLBM``
    * ``benchmarks/bench_marine.py`` result lists (``st_sim``/``st_ref``,
      ``cd_sim``/``cd_ref``, ``omega_measured``/``omega_theory``);
    * ``artifacts/sdaa_*.json`` hydrostatics blocks (``results.<grid>.<coef>``
      against ``analytic.<coef>``).
``TensorFVM``
    * ``scripts/verification-*/summary.json`` with ``meta`` + ``cases``
      (velocity-profile L2 error against the Ghia reference);
    * ``scripts/verification-mesh-quality/results.json`` (closed-form shear /
      non-orthogonality comparison).
``TensorFEM``
    * ``benchmarks/finite_width_hole_audit.json`` (``kt_net`` against the
      Howland reference, per case).

Anything else raises :class:`UnknownBenchmarkError` instead of being guessed at.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .gate import GateDecision, engineering_error_gate
from .verdict import DEFAULT_THRESHOLD

__all__ = [
    "BenchmarkArtifact",
    "BenchmarkQoI",
    "BudgetComponent",
    "CombinedBudget",
    "SUPPORTED_SOLVERS",
    "UnknownBenchmarkError",
    "combine_budget",
    "detect_solver",
    "gate_with_benchmark",
    "parse_artifact",
    "parse_benchmarks",
    "relative_error",
    "select",
]

SUPPORTED_SOLVERS = ("TensorLBM", "TensorFVM", "TensorFEM")

#: Two-sided 95 % normal quantile, used to convert a 95 % interval width to a
#: standard uncertainty (GUM: ``u = U / (2 * 1.959964)``).
_Z95 = 1.959963984540054
#: Standard uncertainty of a uniform distribution over a half-width ``e``.
_UNIFORM_STD = 1.0 / math.sqrt(3.0)


class UnknownBenchmarkError(ValueError):
    """The JSON payload does not match any supported benchmark schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, str):
        try:
            v = float(value)
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return None


@dataclass(frozen=True)
class BenchmarkQoI:
    """One measured-vs-reference pair extracted from a benchmark artefact."""

    solver: str
    artifact: str
    sha256: str
    qoi_id: str
    quantity: str
    value: float | None
    reference: float | None
    absolute_error: float | None
    relative_error: float | None
    passed: bool | None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "artifact": self.artifact,
            "sha256": self.sha256,
            "qoi_id": self.qoi_id,
            "quantity": self.quantity,
            "value": self.value,
            "reference": self.reference,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class BenchmarkArtifact:
    """A parsed benchmark artefact, digest-pinned."""

    path: str
    sha256: str
    solver: str
    payload: Any = field(repr=False)

    @classmethod
    def load(cls, path: str | Path, *, solver: str | None = None) -> "BenchmarkArtifact":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"benchmark artefact not found: {p}")
        raw = p.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - bad input
            raise UnknownBenchmarkError(f"{p} is not valid JSON: {exc}") from exc
        detected = solver or detect_solver(payload)
        if detected not in SUPPORTED_SOLVERS:
            raise UnknownBenchmarkError(
                f"{p}: unsupported solver {detected!r}; expected one of {SUPPORTED_SOLVERS}"
            )
        return cls(path=str(p), sha256=_sha256(p), solver=detected, payload=payload)

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "solver": self.solver}


def detect_solver(payload: Any) -> str:
    """Identify the publishing repository from the artefact's schema."""
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            keys = set().union(*(set(item) for item in payload if isinstance(item, dict)))
            if {"st_ref", "st_sim"} & keys or {"cd_ref", "cd_sim"} & keys:
                return "TensorLBM"
            if "omega_theory" in keys:
                return "TensorLBM"
        raise UnknownBenchmarkError("list payload does not match the TensorLBM marine schema")
    if not isinstance(payload, dict):
        raise UnknownBenchmarkError(f"unsupported payload type {type(payload)!r}")
    keys = set(payload)
    if {"analytic", "results"} <= keys and "case" in keys:
        return "TensorLBM"
    meta = payload.get("meta")
    if isinstance(meta, dict) and str(meta.get("code", "")).startswith("tensorfvm"):
        return "TensorFVM"
    if {"curved_channel", "shear_closed_form"} <= keys:
        return "TensorFVM"
    if {"cases", "length_study"} <= keys:
        return "TensorFEM"
    if {"ladder", "reference"} <= keys:
        return "TensorFEM"
    raise UnknownBenchmarkError(
        f"unrecognised benchmark schema, top-level keys: {sorted(keys)[:12]}"
    )


def _record(
    *,
    artifact: BenchmarkArtifact,
    qoi_id: str,
    quantity: str,
    value: Any,
    reference: Any,
    passed: bool | None = None,
    absolute_error: float | None = None,
    detail: dict[str, Any] | None = None,
) -> BenchmarkQoI | None:
    v = _as_float(value)
    r = _as_float(reference)
    if v is None:
        return None
    if absolute_error is None and r is not None:
        absolute_error = abs(v - r)
    rel = None
    # a reference that is numerically zero cannot carry a relative error: the
    # ratio would be dominated by rounding noise, so only the absolute error is
    # reported (same reason ``None`` is used for missing references).
    if r is not None and abs(r) > 1e-12 * max(1.0, abs(v)):
        rel = float(abs(v - r) / abs(r))
    return BenchmarkQoI(
        solver=artifact.solver,
        artifact=artifact.path,
        sha256=artifact.sha256,
        qoi_id=qoi_id,
        quantity=quantity,
        value=v,
        reference=r,
        absolute_error=None if absolute_error is None else float(absolute_error),
        relative_error=rel,
        passed=None if passed is None else bool(passed),
        detail=detail or {},
    )


def _from_lbm_marine(art: BenchmarkArtifact) -> list[BenchmarkQoI]:
    out: list[BenchmarkQoI] = []
    for idx, item in enumerate(art.payload):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", f"case{idx}"))
        for qty, key, ref_key in (
            ("strouhal", "st_sim", "st_ref"),
            ("drag_coefficient", "cd_sim", "cd_ref"),
            ("oscillation_frequency", "omega_measured", "omega_theory"),
        ):
            if key in item:
                rec = _record(
                    artifact=art,
                    qoi_id=f"lbm.marine.{name}.{qty}",
                    quantity=qty,
                    value=item.get(key),
                    reference=item.get(ref_key),
                    detail={k: item[k] for k in ("elapsed_s", "run_dir") if k in item},
                )
                if rec is not None:
                    out.append(rec)
        if "relative_error" in item and not any(
            k in item for k in ("st_sim", "cd_sim", "omega_measured")
        ):
            rel = _as_float(item["relative_error"])
            if rel is not None:
                out.append(
                    BenchmarkQoI(
                        solver=art.solver,
                        artifact=art.path,
                        sha256=art.sha256,
                        qoi_id=f"lbm.marine.{name}.relative_error",
                        quantity=str(item.get("quantity", name)),
                        value=None,
                        reference=None,
                        absolute_error=None,
                        relative_error=float(abs(rel)),
                        passed=None,
                        detail={},
                    )
                )
    return out


def _from_lbm_hydrostatics(art: BenchmarkArtifact) -> list[BenchmarkQoI]:
    payload = art.payload
    analytic = payload.get("analytic") or {}
    results = payload.get("results") or {}
    case = str(payload.get("case", "case"))
    out: list[BenchmarkQoI] = []
    if not isinstance(results, dict):
        return out
    for grid, block in results.items():
        if not isinstance(block, dict):
            continue
        for coef, ref in analytic.items():
            if coef not in block:
                continue
            rec = _record(
                artifact=art,
                qoi_id=f"lbm.{case}.{coef}@{grid}",
                quantity=str(coef),
                value=block.get(coef),
                reference=ref,
                passed=payload.get("overall_PASS"),
                detail={"grid": str(grid), "overall_pass": payload.get("overall_PASS")},
            )
            if rec is not None:
                out.append(rec)
    return out


def _from_fvm_profile(art: BenchmarkArtifact) -> list[BenchmarkQoI]:
    payload = art.payload
    meta = payload.get("meta") or {}
    code = str(meta.get("code", "tensorfvm"))
    out: list[BenchmarkQoI] = []
    for idx, case in enumerate(payload.get("cases") or []):
        if not isinstance(case, dict):
            continue
        re = case.get("reynolds", f"case{idx}")
        for comp in ("u", "v"):
            prof = (case.get(comp) or {}).get("profiles") or {}
            computed = prof.get("computed")
            reference = prof.get("reference")
            if not isinstance(computed, list) or not isinstance(reference, list):
                continue
            c = np.asarray(computed, dtype=float)
            r = np.asarray(reference, dtype=float)
            if c.size != r.size or c.size == 0:
                continue
            diff = float(np.linalg.norm(c - r))
            norm = float(np.linalg.norm(r))
            rel = diff / norm if norm > 0.0 else None
            out.append(
                BenchmarkQoI(
                    solver=art.solver,
                    artifact=art.path,
                    sha256=art.sha256,
                    qoi_id=f"fvm.{code}.Re{re}.{comp}_profile_l2",
                    quantity=f"{comp}_velocity_profile_l2_relative",
                    value=diff,
                    reference=norm,
                    absolute_error=None,
                    relative_error=rel,
                    passed=None,
                    detail={
                        "reynolds": re,
                        "grid": case.get("n"),
                        "residual": case.get("residual"),
                        "norm": code,
                    },
                )
            )
    return out


def _from_fvm_mesh_quality(art: BenchmarkArtifact) -> list[BenchmarkQoI]:
    payload = art.payload
    out: list[BenchmarkQoI] = []
    for idx, row in enumerate(payload.get("shear_closed_form") or []):
        if not isinstance(row, dict):
            continue
        rec = _record(
            artifact=art,
            qoi_id=f"fvm.mesh_quality.shear_closed_form[{idx}].nonorthogonality_deg",
            quantity="nonorthogonality_deg",
            value=row.get("measured_nonorthogonality_deg"),
            reference=row.get("shear_deg"),
            passed=row.get("gate_passed"),
            detail={"shear_deg": row.get("shear_deg")},
        )
        if rec is not None:
            out.append(rec)
    for idx, row in enumerate(payload.get("rotation_invariance") or []):
        if not isinstance(row, dict):
            continue
        rec = _record(
            artifact=art,
            qoi_id=f"fvm.mesh_quality.rotation_invariance[{idx}].skewness",
            quantity="internal_skewness_max",
            value=row.get("internal_skewness_max"),
            reference=payload.get("orthogonal_reference", {}).get("internal_skewness_max"),
            detail={"angle_deg": row.get("angle_deg")},
        )
        if rec is not None:
            out.append(rec)
    return out


def _from_fem_hole(art: BenchmarkArtifact) -> list[BenchmarkQoI]:
    payload = art.payload
    out: list[BenchmarkQoI] = []
    for idx, case in enumerate(payload.get("cases") or []):
        if not isinstance(case, dict):
            continue
        levels = case.get("levels") or []
        if not levels:
            continue
        finest = levels[-1]
        rho = case.get("rho", idx)
        rec = _record(
            artifact=art,
            qoi_id=f"fem.finite_width_hole.rho{rho}.kt_net_finest",
            quantity="kt_net",
            value=finest.get("kt_net"),
            reference=case.get("howland_kt_net"),
            passed=bool(payload.get("passes_3pct")) if "passes_3pct" in payload else None,
            detail={
                "rho": rho,
                "dofs": finest.get("dofs"),
                "criterion": payload.get("criterion"),
            },
        )
        if rec is not None:
            out.append(rec)
    for idx, row in enumerate(payload.get("length_study") or []):
        if not isinstance(row, dict) or "relative_error_vs_howland" not in row:
            continue
        rel = _as_float(row["relative_error_vs_howland"])
        if rel is None:
            continue
        out.append(
            BenchmarkQoI(
                solver=art.solver,
                artifact=art.path,
                sha256=art.sha256,
                qoi_id=f"fem.finite_width_hole.length_study[{idx}].relative_error",
                quantity="relative_error_vs_howland",
                value=None,
                reference=None,
                absolute_error=None,
                relative_error=float(abs(rel)),
                passed=bool(payload.get("passes_3pct")) if "passes_3pct" in payload else None,
                detail={"half_length_over_half_width": row.get("half_length_over_half_width")},
            )
        )
    return out


_EXTRACTORS = {
    ("TensorLBM", "marine"): _from_lbm_marine,
    ("TensorLBM", "hydrostatics"): _from_lbm_hydrostatics,
    ("TensorFVM", "profile"): _from_fvm_profile,
    ("TensorFVM", "mesh"): _from_fvm_mesh_quality,
    ("TensorFEM", "hole"): _from_fem_hole,
}


def parse_artifact(artifact: BenchmarkArtifact) -> tuple[BenchmarkQoI, ...]:
    """Extract every measured-vs-reference pair from a loaded artefact."""
    payload = artifact.payload
    if artifact.solver == "TensorLBM":
        if isinstance(payload, list):
            return tuple(_from_lbm_marine(artifact))
        return tuple(_from_lbm_hydrostatics(artifact))
    if artifact.solver == "TensorFVM":
        meta = payload.get("meta") or {}
        if isinstance(meta, dict) and str(meta.get("code", "")).startswith("tensorfvm"):
            return tuple(_from_fvm_profile(artifact))
        return tuple(_from_fvm_mesh_quality(artifact))
    return tuple(_from_fem_hole(artifact))


def parse_benchmarks(
    path: str | Path, *, solver: str | None = None
) -> tuple[BenchmarkArtifact, tuple[BenchmarkQoI, ...]]:
    """Load one artefact and return ``(artifact, records)``."""
    artifact = BenchmarkArtifact.load(path, solver=solver)
    records = parse_artifact(artifact)
    if not records:
        raise UnknownBenchmarkError(f"{path}: recognised schema but no QoI pair was extracted")
    return artifact, records


def select(
    records: Iterable[BenchmarkQoI],
    *,
    solver: str | None = None,
    qoi_id: str | None = None,
    quantity: str | None = None,
    with_reference: bool = True,
) -> tuple[BenchmarkQoI, ...]:
    """Filter records by repository / id / quantity."""
    out = []
    for rec in records:
        if solver is not None and rec.solver != solver:
            continue
        if qoi_id is not None and rec.qoi_id != qoi_id:
            continue
        if quantity is not None and rec.quantity != quantity:
            continue
        if with_reference and rec.relative_error is None and rec.absolute_error is None:
            continue
        out.append(rec)
    return tuple(out)


def relative_error(
    records: Sequence[BenchmarkQoI], *, qoi_id: str | None = None, solver: str | None = None
) -> float:
    """The worst (conservative) relative error among the selected records."""
    chosen = [r for r in records if r.relative_error is not None]
    if solver is not None:
        chosen = [r for r in chosen if r.solver == solver]
    if qoi_id is not None:
        chosen = [r for r in chosen if r.qoi_id == qoi_id]
    if not chosen:
        raise ValueError("no record with a relative error was selected")
    return float(max(abs(r.relative_error) for r in chosen))


@dataclass(frozen=True)
class BudgetComponent:
    """One term of the uncertainty budget."""

    name: str
    relative_standard: float
    basis: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "relative_standard": float(self.relative_standard),
            "basis": self.basis,
            "source": self.source,
        }


@dataclass(frozen=True)
class CombinedBudget:
    """Quadrature combination of the UQ band and the benchmark errors."""

    uq_relative_width: float
    components: tuple[BudgetComponent, ...]
    combined_standard: float
    expansion_factor: float
    expanded_relative: float
    threshold: float
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "uq_relative_width": float(self.uq_relative_width),
            "components": [c.as_dict() for c in self.components],
            "combined_standard": float(self.combined_standard),
            "expansion_factor": float(self.expansion_factor),
            "expanded_relative": float(self.expanded_relative),
            "threshold": float(self.threshold),
            "status": self.status,
        }


def combine_budget(
    uq_relative_width: float,
    records: Sequence[BenchmarkQoI],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    expansion_factor: float = 2.0,
    solver_filter: str | None = None,
) -> CombinedBudget:
    """Combine the UQ interval width and benchmark errors (GUM, first order).

    * the UQ term enters as a standard uncertainty, ``u = U / (2 * 1.96)``,
      which assumes the 95 % band is approximately normal;
    * each benchmark comparison error ``e`` enters as ``u = e / sqrt(3)``, the
      standard uncertainty of a uniform distribution over ``+/- e`` (the
      comparison error is a bound, not a standard deviation);
    * the components are independent by construction (different solvers,
      different references), so they combine in quadrature and are expanded
      with ``k = 2``.
    """
    if not (uq_relative_width >= 0.0 and math.isfinite(uq_relative_width)):
        raise ValueError(f"uq_relative_width must be finite and >= 0, got {uq_relative_width!r}")
    chosen = [r for r in records if r.relative_error is not None]
    if solver_filter is not None:
        chosen = [r for r in chosen if r.solver == solver_filter]
    if not chosen:
        raise ValueError("no benchmark record with a relative error was supplied")
    components = [
        BudgetComponent(
            name="uq_band",
            relative_standard=float(uq_relative_width) / (2.0 * _Z95),
            basis="GUM: u = U95 / (2 * 1.959964)",
            source="tensorphysics.uq.workflow/UQStudy",
        )
    ]
    for rec in chosen:
        components.append(
            BudgetComponent(
                name=f"benchmark:{rec.solver}:{rec.qoi_id}",
                relative_standard=float(abs(rec.relative_error)) * _UNIFORM_STD,
                basis="GUM: u = e / sqrt(3) (uniform bound)",
                source=f"{rec.artifact}#sha256={rec.sha256[:12]}",
            )
        )
    combined = math.sqrt(sum(c.relative_standard**2 for c in components))
    expanded = expansion_factor * combined
    return CombinedBudget(
        uq_relative_width=float(uq_relative_width),
        components=tuple(components),
        combined_standard=float(combined),
        expansion_factor=float(expansion_factor),
        expanded_relative=float(expanded),
        threshold=float(threshold),
        status="PASS" if expanded <= threshold else "FAIL",
    )


def gate_with_benchmark(
    *,
    study: Any,
    records: Sequence[BenchmarkQoI],
    threshold: float = DEFAULT_THRESHOLD,
    qoi_id: str | None = None,
    solver: str | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Run the pre-evidence gate on a benchmark comparison error.

    The comparison error is taken from the parsed artefact (worst selected
    record unless ``qoi_id`` pins one), so the number the gate adjudicates is
    the number the solver repository published -- not a re-typed value.
    """
    err = relative_error(records, qoi_id=qoi_id, solver=solver)
    decision: GateDecision = engineering_error_gate(
        study=study, relative_error=err, threshold=threshold, label=label
    )
    rel_width = None
    budget = getattr(study, "budget", None)
    if budget is not None:
        rel_width = float(budget.relative_width)
    combined = (
        combine_budget(rel_width, records, threshold=threshold, solver_filter=solver)
        if rel_width is not None
        else None
    )
    chosen = select(records, solver=solver, qoi_id=qoi_id)
    return {
        "label": label,
        "relative_error": float(err),
        "gate": decision.as_dict() if hasattr(decision, "as_dict") else {
            "status": decision.status,
            "reasons": [r.as_dict() for r in decision.reasons],
            "evidence_digest": decision.evidence_digest,
        },
        "status": decision.status,
        "combined_budget": None if combined is None else combined.as_dict(),
        "provenance": [r.as_dict() for r in chosen],
    }