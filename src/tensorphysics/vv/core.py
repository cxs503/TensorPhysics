"""Core V&V gate: one entry point composing the functional sub-gates.

This is the ``core`` of the V&V machine gate (mission m7729, original PR #199,
re-opened as m8768).  It deliberately re-implements **no** numerics: every
number is produced by :mod:`tensorphysics.vv.gci`,
:mod:`tensorphysics.vv.validation` or :mod:`tensorphysics.vv.cross_solver`
and carried here with provenance.

Stages of :meth:`VVCore.run`
----------------------------
1. **convergence** -- :func:`tensorphysics.vv.gci.evaluate` on the supplied
   grid levels (observed order, Richardson extrapolation, GCI, asymptotic
   range, noise certificate);
2. **validation** -- the ASME V&V 20 ``|E| <= U_val`` test that
   :func:`tensorphysics.vv.gci.evaluate` runs against ``f_exp`` when a
   reference is supplied (``U_num`` defaults to the study's GCI band);
3. **cross-solver** --
   :func:`tensorphysics.vv.cross_solver.cross_solver_agreement` on the
   optional set of per-solver samples.

The fused status is the worst of the stage statuses under the gate ordering
``PASS < WARN < WITHHELD < UNRESOLVED < FAIL``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .cross_solver import (
    AgreementPolicy,
    AgreementResult,
    SolverSample,
    cross_solver_agreement,
)
from .gci import (
    FAIL,
    PASS,
    UNRESOLVED,
    WARN,
    WITHHELD,
    GatePolicy,
    GateVerdict,
    Observation,
    evaluate,
)
from .validation import ValidationResult, validate

__all__ = ["CoreVerdict", "VVCore", "evaluate_core"]

# Same ordering the convergence gate uses (tensorphysics.vv.gci._STATUS_ORDER).
_STATUS_ORDER = {PASS: 0, WARN: 1, WITHHELD: 2, UNRESOLVED: 3, FAIL: 4}


def _worst_status(statuses: Iterable[str]) -> str:
    """Worst status under the gate ordering (FAIL dominates)."""
    worst = PASS
    for status in statuses:
        if _STATUS_ORDER[status] > _STATUS_ORDER[worst]:
            worst = status
    return worst


@dataclass(frozen=True)
class CoreVerdict:
    """Fused machine verdict of the V&V core gate.

    ``convergence`` is the full :class:`~tensorphysics.vv.gci.GateVerdict`
    (stage 1, with the embedded validation stage 2); ``validation`` is the
    :class:`~tensorphysics.vv.validation.ValidationResult` of stage 2
    (``None`` when no reference was supplied); ``agreement`` is the
    :class:`~tensorphysics.vv.cross_solver.AgreementResult` of stage 3
    (``None`` when no samples were supplied).
    """

    status: str
    quantity: str
    convergence: GateVerdict | None
    validation: ValidationResult | None
    agreement: AgreementResult | None
    checks: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(c["detail"] for c in self.checks if c["status"] != PASS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "tensorphysics.vv.core/1",
            "status": self.status,
            "quantity": self.quantity,
            "convergence": None if self.convergence is None else self.convergence.as_dict(),
            "validation": None if self.validation is None else self.validation.as_dict(),
            "agreement": None if self.agreement is None else self.agreement.as_dict(),
            "checks": list(self.checks),
            "reasons": list(self.reasons),
        }


class VVCore:
    """Unified V&V machine gate (convergence + validation + cross-solver).

    Parameters
    ----------
    gate_policy:
        Policy of the convergence/validation stage (defaults to the
        documented ASME/ITTC set of :class:`~tensorphysics.vv.gci.GatePolicy`).
    agreement_policy:
        Policy of the cross-solver stage (defaults to
        :class:`~tensorphysics.vv.cross_solver.AgreementPolicy`).
    """

    def __init__(
        self,
        gate_policy: GatePolicy | None = None,
        agreement_policy: AgreementPolicy | None = None,
    ) -> None:
        self.gate_policy = gate_policy if gate_policy is not None else GatePolicy()
        self.agreement_policy = (
            agreement_policy if agreement_policy is not None else AgreementPolicy()
        )

    # -- stage 1 (+ embedded stage 2) ---------------------------------------
    def convergence(
        self,
        levels: Sequence[Observation],
        *,
        quantity: str = "observable",
        **gate_kwargs: Any,
    ) -> GateVerdict:
        """Run the convergence stage.

        Extra keyword arguments (``f_exp``, ``u_num``, ``u_input``, ``f_sim``,
        ``features``, ``u_ref``, ...) are forwarded to
        :func:`tensorphysics.vv.gci.evaluate`; when ``f_exp`` is given the
        ASME V&V 20 validation stage runs inside it.
        """
        return evaluate(levels, quantity=quantity, policy=self.gate_policy, **gate_kwargs)

    # -- stage 2 standalone --------------------------------------------------
    def validation(
        self,
        f_sim: float,
        f_exp: float,
        *,
        u_num: float,
        u_input: float = 0.0,
        label: str = "",
    ) -> ValidationResult:
        """Standalone ASME V&V 20 comparison (no convergence study needed)."""
        return validate(f_sim, f_exp, u_num=u_num, u_input=u_input, label=label)

    # -- stage 3 --------------------------------------------------------------
    def cross_solver(
        self,
        samples: Iterable[SolverSample],
        *,
        quantity: str = "",
        policy: AgreementPolicy | None = None,
    ) -> AgreementResult:
        """Run the cross-solver agreement stage."""
        return cross_solver_agreement(
            samples, quantity=quantity, policy=policy or self.agreement_policy
        )

    # -- full pipeline ---------------------------------------------------------
    def run(
        self,
        levels: Sequence[Observation],
        *,
        quantity: str = "observable",
        f_exp: float | None = None,
        u_num: float | None = None,
        u_input: float | None = None,
        samples: Iterable[SolverSample] | None = None,
        **gate_kwargs: Any,
    ) -> CoreVerdict:
        """Run all stages and fuse the verdicts.

        ``f_exp`` / ``u_num`` / ``u_input`` are forwarded to the convergence
        stage, which performs the ASME V&V 20 validation when a reference is
        given.  ``samples`` (optional) adds the cross-solver agreement stage;
        its checks are appended to the account under the ``cross-solver/``
        prefix so the fused verdict stays a single flat, JSON-serialisable
        record.
        """
        conv = self.convergence(
            levels,
            quantity=quantity,
            f_exp=f_exp,
            u_num=u_num,
            u_input=u_input,
            **gate_kwargs,
        )
        agr: AgreementResult | None = None
        if samples is not None:
            agr = self.cross_solver(samples, quantity=quantity)

        checks: list[dict[str, Any]] = list(conv.checks)
        if agr is not None:
            checks.extend({**c, "check": f"cross-solver/{c['check']}"} for c in agr.checks)

        statuses = [conv.status]
        if agr is not None:
            statuses.append(agr.status)
        return CoreVerdict(
            status=_worst_status(statuses),
            quantity=quantity,
            convergence=conv,
            validation=conv.validation,
            agreement=agr,
            checks=tuple(checks),
        )


def evaluate_core(
    levels: Sequence[Observation],
    *,
    quantity: str = "observable",
    f_exp: float | None = None,
    u_num: float | None = None,
    u_input: float | None = None,
    samples: Iterable[SolverSample] | None = None,
    gate_policy: GatePolicy | None = None,
    agreement_policy: AgreementPolicy | None = None,
    **gate_kwargs: Any,
) -> CoreVerdict:
    """One-call V&V core gate (module-level convenience for :class:`VVCore`)."""
    core = VVCore(gate_policy=gate_policy, agreement_policy=agreement_policy)
    return core.run(
        levels,
        quantity=quantity,
        f_exp=f_exp,
        u_num=u_num,
        u_input=u_input,
        samples=samples,
        **gate_kwargs,
    )
