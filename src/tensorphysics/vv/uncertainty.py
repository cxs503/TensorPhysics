"""ASME V&V 20 uncertainty ledger: ``U_num``, ``U_input``, ``U_val`` and ``U_ref``.

ASME V&V 20-2009 defines the validation uncertainty of a simulation result as

.. math::  U_{val} = \\sqrt{U_{num}^2 + U_{input}^2}

with ``U_num`` the numerical uncertainty (normally the GCI of the finest grid,
see :mod:`tensorphysics.vv.gci`) and ``U_input`` the uncertainty of the input
parameters.  The model is *validated* against a reference when the comparison
error :math:`E = f_{sim} - f_{ref}` satisfies ``|E| <= U_val``.

What the plain scalar form hides
--------------------------------
The acceptance test is only meaningful when the comparison has something to
compare *with*.  Every reference carries its own uncertainty ``U_ref``
(experimental scatter, sea-trial correction uncertainty, a literature value
quoted to three digits).  Two statements therefore have to be kept apart:

* **is the model within U_val?** -- the ASME test;
* **could this reference have told the two apart at all?** -- the
  discriminability of the reference.

If ``U_ref`` is not smaller than ``U_val``, a PASS says nothing: a model that is
off by many ``U_num`` would still pass because the reference is that blurry.
:func:`compare` returns ``WITHHELD`` for exactly this case, and the ledger keeps
every term so the reader can see which one dominated.

The ledger is also the audit record: :attr:`UncertaintyLedger.digest` hashes the
canonical ledger so a published number can be traced back to its uncertainty
terms.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "DISCRIMINATION_FACTOR",
    "UncertaintyLedger",
    "LedgerVerdict",
    "compare",
    "ledger_from_study",
    "u_num_from_study",
]

#: A reference is discriminative when ``U_ref < factor * U_val``.  The default
#: of 1 reproduces the ASME statement "the reference must be sharper than the
#: validation uncertainty"; raising it demands a sharper benchmark.
DISCRIMINATION_FACTOR = 1.0

PASS = "PASS"
FAIL = "FAIL"
WITHHELD = "WITHHELD"


def _check_term(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return value


@dataclass(frozen=True)
class UncertaintyLedger:
    """The complete uncertainty budget of one comparison.

    ``u_num``       numerical uncertainty (GCI of the finest grid)
    ``u_input``     input-parameter uncertainty
    ``u_ref``       reference / benchmark uncertainty (experiment, sea trial,
                    literature value) -- **not** part of ``U_val`` in ASME V&V 20,
                    but decisive for the discriminability of the test
    """

    u_num: float
    u_input: float = 0.0
    u_ref: float = 0.0
    u_num_source: str = "GCI (Roache, Fs=1.25)"
    u_input_source: str = "declared input-parameter uncertainty"
    u_ref_source: str = "declared reference uncertainty"
    components: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_term("u_num", self.u_num)
        _check_term("u_input", self.u_input)
        _check_term("u_ref", self.u_ref)

    # -- ASME V&V 20 -------------------------------------------------------
    @property
    def u_val(self) -> float:
        """``U_val = sqrt(U_num**2 + U_input**2)`` (ASME V&V 20-2009)."""
        return math.sqrt(self.u_num**2 + self.u_input**2)

    @property
    def u_val_total(self) -> float:
        """``sqrt(U_num**2 + U_input**2 + U_ref**2)`` -- reported for transparency only."""
        return math.sqrt(self.u_num**2 + self.u_input**2 + self.u_ref**2)

    @property
    def reference_ratio(self) -> float | None:
        """``U_ref / U_val``: how blurry the reference is relative to the simulation."""
        u_val = self.u_val
        if u_val == 0.0:
            return None if self.u_ref == 0.0 else math.inf
        return self.u_ref / u_val

    @property
    def u_num_fraction(self) -> float | None:
        """Share of ``U_val`` carried by the numerical term (diagnostic)."""
        u_val = self.u_val
        return None if u_val == 0.0 else self.u_num / u_val

    def discriminative(self, factor: float = DISCRIMINATION_FACTOR) -> bool:
        ratio = self.reference_ratio
        if ratio is None:
            # no uncertainty at all: the comparison is exact, hence discriminative
            return True
        return ratio < factor

    def dominant_term(self) -> str:
        terms = {"u_num": self.u_num, "u_input": self.u_input, "u_ref": self.u_ref}
        return max(terms, key=lambda k: terms[k])

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "tensorphysics.vv.uncertainty-ledger/1",
            "u_num": self.u_num,
            "u_input": self.u_input,
            "u_ref": self.u_ref,
            "u_val": self.u_val,
            "u_val_total": self.u_val_total,
            "reference_ratio": self.reference_ratio,
            "u_num_fraction": self.u_num_fraction,
            "dominant_term": self.dominant_term(),
            "u_num_source": self.u_num_source,
            "u_input_source": self.u_input_source,
            "u_ref_source": self.u_ref_source,
            "components": list(self.components),
            "discrimination_factor": DISCRIMINATION_FACTOR,
            "discriminative": self.discriminative(),
        }
        payload["digest"] = ledger_digest(payload)
        return payload


@dataclass(frozen=True)
class LedgerVerdict:
    """Outcome of an ASME V&V 20 comparison with a reference-uncertainty check."""

    status: str
    f_sim: float
    f_ref: float
    error: float
    ledger: UncertaintyLedger
    ratio: float | None
    passed: bool
    discriminative: bool
    detail: str = ""
    label: str = ""

    @property
    def relative_error(self) -> float | None:
        return None if self.f_ref == 0.0 else self.error / self.f_ref

    def as_dict(self) -> dict[str, Any]:
        out = {
            "schema": "tensorphysics.vv.vv20-ledger-verdict/1",
            "status": self.status,
            "f_sim": self.f_sim,
            "f_ref": self.f_ref,
            "error": self.error,
            "relative_error": self.relative_error,
            "ratio": self.ratio,
            "passed": self.passed,
            "discriminative": self.discriminative,
            "detail": self.detail,
            "ledger": self.ledger.as_dict(),
        }
        if self.label:
            out["label"] = self.label
        return out


def ledger_digest(payload: Mapping[str, Any]) -> str:
    """sha256 of the canonical ledger (audit-chain fingerprint)."""
    body = {k: v for k, v in payload.items() if k != "digest"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(canonical.encode()).hexdigest()


def u_num_from_study(study: Any) -> tuple[float, str]:
    """Absolute ``U_num`` of a :class:`~tensorphysics.vv.gci.GridConvergenceStudy`.

    The GCI may be relative (fraction of the finest value); the ledger always
    carries an absolute uncertainty, so the relative band is scaled by
    ``|f_fine|`` here and the provenance string records which form was used.
    """
    if study is None or getattr(study, "gci_fine", None) is None:
        raise ValueError("the study carries no GCI: U_num is not available")
    gci_value = study.gci_fine
    if getattr(study, "gci_is_relative", False):
        return abs(gci_value * study.f_fine), (
            f"GCI_fine={gci_value:.6g} (relative, Fs={study.safety_factor:g}) x |f_fine|={abs(study.f_fine):.6g}"
        )
    return abs(gci_value), f"GCI_fine={gci_value:.6g} (absolute, Fs={study.safety_factor:g})"


def ledger_from_study(
    study: Any,
    *,
    u_input: float = 0.0,
    u_ref: float = 0.0,
    u_input_source: str = "declared input-parameter uncertainty",
    u_ref_source: str = "declared reference uncertainty",
    components: Sequence[Mapping[str, Any]] = (),
) -> UncertaintyLedger:
    """Build the ledger for a converged study, taking ``U_num`` from its GCI."""
    u_num, source = u_num_from_study(study)
    return UncertaintyLedger(
        u_num=u_num,
        u_input=u_input,
        u_ref=u_ref,
        u_num_source=source,
        u_input_source=u_input_source,
        u_ref_source=u_ref_source,
        components=tuple(dict(c) for c in components),
    )


def compare(
    f_sim: float,
    f_ref: float,
    ledger: UncertaintyLedger,
    *,
    label: str = "",
    discrimination_factor: float = DISCRIMINATION_FACTOR,
) -> LedgerVerdict:
    """ASME V&V 20 acceptance test plus the reference-discriminability guard.

    * ``PASS``      -- ``|E| <= U_val`` **and** the reference is discriminative.
    * ``FAIL``      -- ``|E| > U_val`` with a discriminative reference.
    * ``WITHHELD``  -- the reference is not sharper than ``U_val``: the test
      cannot separate a good model from a bad one, so no grade is issued.
      ``ratio`` is still reported so the reader sees the raw numbers.
    """
    if not isinstance(f_sim, (int, float)) or not math.isfinite(f_sim):
        raise ValueError(f"f_sim must be finite, got {f_sim!r}")
    if not isinstance(f_ref, (int, float)) or not math.isfinite(f_ref):
        raise ValueError(f"f_ref must be finite, got {f_ref!r}")
    if discrimination_factor <= 0.0:
        raise ValueError("discrimination_factor must be positive")

    error = float(f_sim) - float(f_ref)
    u_val = ledger.u_val
    if u_val == 0.0:
        ratio = 0.0 if error == 0.0 else math.inf
        within = error == 0.0
    else:
        ratio = abs(error) / u_val
        within = abs(error) <= u_val + 1e-12 * max(u_val, abs(error))

    discriminative = ledger.discriminative(discrimination_factor)
    if not discriminative:
        status = WITHHELD
        detail = (
            f"reference uncertainty U_ref={ledger.u_ref:.6g} is not sharper than U_val={u_val:.6g} "
            f"(U_ref/U_val={ledger.reference_ratio:.3f} >= {discrimination_factor:g}): the comparison "
            f"cannot discriminate (raw |E|/U_val={ratio:.4f}, within={within})"
        )
    elif within:
        status = PASS
        detail = (
            f"|E|={abs(error):.6g} <= U_val={u_val:.6g} (ratio {ratio:.4f}); dominant term "
            f"{ledger.dominant_term()} (U_ref/U_val={ledger.reference_ratio})"
        )
    else:
        status = FAIL
        detail = (
            f"|E|={abs(error):.6g} > U_val={u_val:.6g} (ratio {ratio:.4f}): the model is not validated "
            f"against this reference, dominant term {ledger.dominant_term()}"
        )

    return LedgerVerdict(
        status=status,
        f_sim=float(f_sim),
        f_ref=float(f_ref),
        error=error,
        ledger=ledger,
        ratio=ratio,
        passed=within,
        discriminative=discriminative,
        detail=detail,
        label=label,
    )


def evaluate_ledger_table(
    rows: Iterable[Mapping[str, Any]],
    *,
    keys: Sequence[str] = ("f_sim", "f_ref", "u_num"),
) -> list[LedgerVerdict]:
    """Run :func:`compare` over a table of ``f_sim``/``f_ref``/``u_num``/... rows."""
    out: list[LedgerVerdict] = []
    for index, row in enumerate(rows):
        missing = [k for k in keys if k not in row]
        if missing:
            raise ValueError(f"row {index} is missing fields {missing}")
        ledger = UncertaintyLedger(
            u_num=row["u_num"],
            u_input=row.get("u_input", 0.0),
            u_ref=row.get("u_ref", 0.0),
            u_num_source=str(row.get("u_num_source", "declared")),
            u_ref_source=str(row.get("u_ref_source", "declared reference uncertainty")),
        )
        out.append(
            compare(
                row["f_sim"],
                row["f_ref"],
                ledger,
                label=str(row.get("label", "")),
                discrimination_factor=row.get("discrimination_factor", DISCRIMINATION_FACTOR),
            )
        )
    return out