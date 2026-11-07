"""ASME V&V 20 validation uncertainty and the ``|E| <= U_val`` acceptance test.

``U_val`` combines the numerical uncertainty ``U_num`` (normally the GCI of the
finest grid, see :mod:`tensorphysics.vv.gci`) with the input/experimental
uncertainty ``U_input``:

    U_val = sqrt(U_num**2 + U_input**2)

A model prediction ``f_sim`` is *validated* against a reference ``f_exp`` when
the comparison error ``E = f_sim - f_exp`` satisfies ``|E| <= U_val``.  The
gate reports the ratio ``|E| / U_val`` so that a marginal pass is visible; a
ratio above 1 is a hard FAIL and must not be overwritten by a nicer number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BOUNDARY_SLACK",
    "ValidationResult",
    "evaluate_validation_table",
    "validation_uncertainty",
    "validate",
]


#: Relative slack on the ``|E| <= U_val`` acceptance boundary.  ``U_val`` and
#: ``E`` are both computed from floating point numbers, so an exact boundary
#: case (``|E|`` *equal* to ``U_val`` by construction) can land a few ulp on
#: either side; the slack keeps such a case a PASS without loosening anything
#: measurable (1e-12 relative is ~4 orders below the 1e-8 precision of the
#: inputs themselves).  The reported ``ratio`` is *not* slacked.
BOUNDARY_SLACK = 1e-12


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of one ASME V&V 20 comparison."""

    f_sim: float
    f_exp: float
    u_num: float
    u_input: float
    u_val: float
    error: float
    ratio: float
    passed: bool
    label: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def as_dict(self) -> dict[str, Any]:
        out = {
            "f_sim": self.f_sim,
            "f_exp": self.f_exp,
            "u_num": self.u_num,
            "u_input": self.u_input,
            "u_val": self.u_val,
            "error": self.error,
            "relative_error": self.relative_error,
            "ratio": self.ratio,
            "status": self.status,
        }
        if self.label:
            out["label"] = self.label
        return out

    @property
    def relative_error(self) -> float | None:
        if self.f_exp == 0.0:
            return None
        return self.error / self.f_exp


def validation_uncertainty(u_num: float, u_input: float = 0.0) -> float:
    """``U_val = sqrt(U_num**2 + U_input**2)`` (ASME V&V 20-2009)."""
    for name, value in (("u_num", u_num), ("u_input", u_input)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return math.sqrt(u_num * u_num + u_input * u_input)


def validate(
    f_sim: float,
    f_exp: float,
    *,
    u_num: float,
    u_input: float = 0.0,
    label: str = "",
) -> ValidationResult:
    """Compare one simulation result with a reference under ASME V&V 20."""
    if not math.isfinite(f_sim) or not math.isfinite(f_exp):
        raise ValueError("f_sim and f_exp must be finite")
    u_val = validation_uncertainty(u_num, u_input)
    error = f_sim - f_exp
    if u_val == 0.0:
        # zero declared uncertainty: only an exact match can pass
        ratio = 0.0 if error == 0.0 else math.inf
        passed = error == 0.0
    else:
        ratio = abs(error) / u_val
        passed = abs(error) <= u_val + BOUNDARY_SLACK * max(u_val, abs(error))
    return ValidationResult(
        f_sim=f_sim,
        f_exp=f_exp,
        u_num=u_num,
        u_input=u_input,
        u_val=u_val,
        error=error,
        ratio=ratio,
        passed=passed,
        label=label,
    )


def evaluate_validation_table(
    rows: Iterable[Mapping[str, Any] | Sequence[float]],
    *,
    keys: Sequence[str] = ("f_sim", "f_exp", "u_num", "u_input"),
) -> list[ValidationResult]:
    """Run the V&V 20 comparison over a table of cases.

    ``rows`` may be mappings (field names, optional ``label``) or plain
    sequences ordered as ``keys``.
    """
    results: list[ValidationResult] = []
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            missing = [k for k in keys if k not in row]
            if missing:
                raise ValueError(f"row {index} is missing fields {missing}")
            results.append(
                validate(
                    row["f_sim"],
                    row["f_exp"],
                    u_num=row["u_num"],
                    u_input=row.get("u_input", 0.0),
                    label=str(row.get("label", "")),
                )
            )
        else:
            values = list(row)
            if len(values) < len(keys):
                raise ValueError(f"row {index} needs {len(keys)} values, got {len(values)}")
            results.append(
                validate(
                    values[0],
                    values[1],
                    u_num=values[2],
                    u_input=values[3] if len(values) > 3 else 0.0,
                )
            )
    return results