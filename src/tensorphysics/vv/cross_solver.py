"""Cross-solver agreement gate: mutual consistency of a solver ensemble.

The V&V machine gate has two halves.  :mod:`tensorphysics.vv.validation`
judges *one* solver against a *reference* (experiment or analytic solution);
:mod:`tensorphysics.vv.gci` judges *one* solver's own grid convergence.  This
module answers the third question, the one that has no reference at all:

    *N solvers each report a value for the same observable on the same
    problem.  Do they agree, and what consensus value -- with an uncertainty
    -- may be published?*

It is the cross-solver complement of the single-solver grid-convergence gate:
no experiment, no analytic solution, only the mutual consistency of the
ensemble.  A solver that disagrees with the ensemble is flagged as an
outlier; the consensus value and its uncertainty are what may be published.
A FAIL is never overwritten by a nicer number.

Conventions
-----------
``samples`` is a collection of :class:`SolverSample`, one per solver.  Each
sample carries the solver name, the reported value ``f``, and an optional
1-sigma uncertainty ``u`` (normally that solver's own finest-grid GCI).

The **consensus** is the inverse-variance weighted mean

    f_c = sum_i w_i f_i / sum_i w_i,      w_i = 1 / u_i**2

with uncertainty ``u_c = 1 / sqrt(sum_i w_i)``.  When no sample declares a
positive uncertainty the consensus is the plain arithmetic mean and its
uncertainty is left ``None`` (the gate then relies on the spread band alone).
A study that mixes declared and undeclared uncertainties is rejected: the
caller must be explicit about which regime it is in.

The **agreement band** is the largest pairwise disagreement

    spread     = max_{i<j} |f_i - f_j|
    spread_rel = spread / |f_c|

and the gate is deliberately conservative: a band wider than
``fail_spread`` (or a solver more than ``outlier_sigma`` of its own standard
deviations from the consensus) is a hard FAIL, a band wider than
``warn_spread`` is a WARN, otherwise PASS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from enum import Enum

from .gci import FAIL, PASS, WARN

__all__ = [
    "PASS",
    "WARN",
    "FAIL",
    "AgreementPolicy",
    "AgreementResult",
    "SolverSample",
    "agreement_spread",
    "consensus",
    "cross_solver_agreement",
    "CanonicalCase",
    "ComparisonResult",
    "ComparisonStatus",
    "CrossSolverAdapter",
    "create_boundary_layer_case",
    "create_cylinder_drag_case",
    "create_lid_driven_cavity_case",
]


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverSample:
    """One solver's result for one observable on one problem.

    ``u`` is the solver's own 1-sigma uncertainty on ``f`` (normally its
    finest-grid GCI).  ``h`` is optional context (grid size) and is carried
    for provenance but not used by the gate.
    """

    solver: str
    f: float
    u: float | None = None
    h: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.solver, str) or not self.solver.strip():
            raise ValueError(f"solver name must be a non-empty string, got {self.solver!r}")
        if not isinstance(self.f, (int, float)) or not math.isfinite(self.f):
            raise ValueError(f"f must be finite, got {self.f!r}")
        if self.u is not None and (not math.isfinite(self.u) or self.u < 0.0):
            raise ValueError(f"u must be finite and non-negative, got {self.u!r}")
        if self.h is not None and (not math.isfinite(self.h) or self.h <= 0.0):
            raise ValueError(f"h must be finite and positive, got {self.h!r}")

    def declares_uncertainty(self) -> bool:
        return self.u is not None and self.u > 0.0

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"solver": self.solver, "f": self.f}
        if self.u is not None:
            out["u"] = self.u
        if self.h is not None:
            out["h"] = self.h
        if self.label:
            out["label"] = self.label
        return out

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SolverSample":
        diff: dict[str, Any] = {}
        solver = payload.get("solver", payload.get("name", payload.get("method")))
        if solver is not None:
            diff["solver"] = str(solver)
        f = payload.get("f", payload.get("value", payload.get("metric")))
        if f is not None:
            diff["f"] = f
        for key in ("u", "h", "label"):
            if payload.get(key) is not None:
                diff[key] = payload[key]
        if "solver" not in diff:
            raise ValueError(f"sample needs a solver name, got keys {sorted(payload)}")
        if "f" not in diff:
            raise ValueError(f"sample needs a value, got keys {sorted(payload)}")
        return cls(**diff)


@dataclass(frozen=True)
class AgreementPolicy:
    """Thresholds of the cross-solver agreement gate.

    ``warn_spread`` / ``fail_spread`` are bands on the largest pairwise
    disagreement: relative (a fraction of ``|f_c|``) by default, absolute
    when ``relative`` is False or the consensus is zero.  ``outlier_sigma``
    flags a solver whose value lies more than that many of its own standard
    deviations from the consensus.
    """

    warn_spread: float = 0.02
    fail_spread: float = 0.05
    outlier_sigma: float = 3.0
    relative: bool = True

    def __post_init__(self) -> None:
        for name in ("warn_spread", "fail_spread"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
        if self.fail_spread < self.warn_spread:
            raise ValueError(
                f"fail_spread ({self.fail_spread}) must be >= warn_spread ({self.warn_spread})"
            )
        if not math.isfinite(self.outlier_sigma) or self.outlier_sigma <= 0.0:
            raise ValueError(f"outlier_sigma must be finite and positive, got {self.outlier_sigma!r}")

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class AgreementResult:
    """Machine verdict of the cross-solver agreement gate."""

    status: str
    quantity: str
    n_solvers: int
    solvers: tuple[str, ...]
    consensus: float
    consensus_u: float | None
    spread: float
    spread_rel: float | None
    widest_pair: tuple[str, str]
    deviations: tuple[dict[str, Any], ...]
    outliers: tuple[str, ...]
    checks: tuple[dict[str, Any], ...]
    policy: AgreementPolicy = field(default_factory=AgreementPolicy)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(c["detail"] for c in self.checks if c["status"] != PASS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "tensorphysics.vv.cross_solver/1",
            "status": self.status,
            "quantity": self.quantity,
            "n_solvers": self.n_solvers,
            "solvers": list(self.solvers),
            "consensus": self.consensus,
            "consensus_u": self.consensus_u,
            "spread": self.spread,
            "spread_rel": self.spread_rel,
            "widest_pair": list(self.widest_pair),
            "deviations": list(self.deviations),
            "outliers": list(self.outliers),
            "checks": list(self.checks),
            "reasons": list(self.reasons),
            "policy": self.policy.as_dict(),
        }


# ---------------------------------------------------------------------------
# numerical helpers
# ---------------------------------------------------------------------------


def _validated_samples(samples: Iterable[SolverSample]) -> tuple[SolverSample, ...]:
    ordered = tuple(samples)
    if len(ordered) < 2:
        raise ValueError("a cross-solver comparison needs at least two solvers")
    names = [s.solver for s in ordered]
    if len(set(names)) != len(names):
        raise ValueError(f"solver names must be unique, got {names}")
    declaring = [s for s in ordered if s.declares_uncertainty()]
    if declaring and len(declaring) != len(ordered):
        raise ValueError(
            "either every sample declares a positive uncertainty or none do; "
            f"declaring={sorted(s.solver for s in declaring)}, "
            f"silent={sorted(s.solver for s in ordered if not s.declares_uncertainty())}"
        )
    return ordered


def consensus(samples: Iterable[SolverSample]) -> tuple[float, float | None]:
    """Inverse-variance weighted mean and its 1-sigma uncertainty.

    Returns ``(f_c, u_c)``.  ``u_c`` is ``None`` when no sample declares a
    positive uncertainty (plain mean, no uncertainty attached).
    """
    ordered = _validated_samples(samples)
    if all(s.declares_uncertainty() for s in ordered):
        weights = [1.0 / (s.u * s.u) for s in ordered]
        total = math.fsum(weights)
        f_c = math.fsum(w * s.f for w, s in zip(weights, ordered)) / total
        u_c = 1.0 / math.sqrt(total)
        return f_c, u_c
    # no sample declares a positive uncertainty: plain mean, no uncertainty
    f_c = math.fsum(s.f for s in ordered) / len(ordered)
    return f_c, None


def agreement_spread(
    samples: Iterable[SolverSample],
) -> tuple[float, float | None, tuple[str, str]]:
    """Largest pairwise disagreement.

    Returns ``(spread, spread_rel, widest_pair)`` where ``spread`` is the
    maximum absolute pairwise difference, ``spread_rel`` is that divided by
    ``|f_c|`` (``None`` when the consensus is zero), and ``widest_pair`` names
    the two solvers achieving the maximum.
    """
    ordered = _validated_samples(samples)
    f_c, _ = consensus(ordered)
    spread = 0.0
    widest = (ordered[0].solver, ordered[1].solver)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            d = abs(ordered[i].f - ordered[j].f)
            if d > spread:
                spread = d
                widest = (ordered[i].solver, ordered[j].solver)
    spread_rel = spread / abs(f_c) if f_c != 0.0 else None
    return spread, spread_rel, widest


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def cross_solver_agreement(
    samples: Iterable[SolverSample],
    *,
    quantity: str = "",
    policy: AgreementPolicy | None = None,
) -> AgreementResult:
    """Judge the mutual consistency of a set of solver results.

    ``samples`` is one :class:`SolverSample` per solver.  The gate reports the
    consensus value and its uncertainty, the agreement band, and a
    PASS / WARN / FAIL verdict with a per-check account.
    """
    ordered = _validated_samples(samples)
    pol = policy if policy is not None else AgreementPolicy()
    f_c, u_c = consensus(ordered)
    spread, spread_rel, widest = agreement_spread(ordered)

    band = spread_rel if (pol.relative and spread_rel is not None) else spread
    band_kind = "relative" if (pol.relative and spread_rel is not None) else "absolute"

    checks: list[dict[str, Any]] = []
    if band > pol.fail_spread:
        spread_status = FAIL
    elif band > pol.warn_spread:
        spread_status = WARN
    else:
        spread_status = PASS
    checks.append(
        {
            "check": "spread",
            "status": spread_status,
            "band": band,
            "band_kind": band_kind,
            "warn": pol.warn_spread,
            "fail": pol.fail_spread,
            "detail": (
                f"widest pair {widest[0]}/{widest[1]} disagrees by {band:.6g} "
                f"({band_kind}) against warn={pol.warn_spread:g} fail={pol.fail_spread:g}"
            ),
        }
    )

    deviations: list[dict[str, Any]] = []
    outliers: list[str] = []
    has_u = all(s.declares_uncertainty() for s in ordered)
    for s in ordered:
        dev = abs(s.f - f_c)
        dev_sigma = dev / s.u if (has_u and s.u > 0.0) else None
        entry: dict[str, Any] = {
            "solver": s.solver,
            "f": s.f,
            "deviation": dev,
            "deviation_sigma": dev_sigma,
        }
        deviations.append(entry)
        if dev_sigma is not None and dev_sigma > pol.outlier_sigma:
            outliers.append(s.solver)
    outlier_status = FAIL if outliers else PASS
    checks.append(
        {
            "check": "outlier",
            "status": outlier_status,
            "outlier_sigma": pol.outlier_sigma,
            "detail": (
                "no solver exceeds "
                f"{pol.outlier_sigma:g} sigma of its own uncertainty"
                if not outliers
                else f"outliers beyond {pol.outlier_sigma:g} sigma: {', '.join(outliers)}"
            ),
        }
    )

    if FAIL in (c["status"] for c in checks):
        status = FAIL
    elif WARN in (c["status"] for c in checks):
        status = WARN
    else:
        status = PASS

    return AgreementResult(
        status=status,
        quantity=quantity,
        n_solvers=len(ordered),
        solvers=tuple(s.solver for s in ordered),
        consensus=f_c,
        consensus_u=u_c,
        spread=spread,
        spread_rel=spread_rel,
        widest_pair=widest,
        deviations=tuple(deviations),
        outliers=tuple(outliers),
        checks=tuple(checks),
        policy=pol,
    )


# ---------------------------------------------------------------------------
# high-level adapter API (complementary to the functional gate)
# ---------------------------------------------------------------------------


class ComparisonStatus(Enum):
    """Status of a cross-solver comparison (high-level API)."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class ComparisonResult:
    """Result of comparing one quantity across solvers (high-level API).

    ``solver_values`` maps solver name -> reported value.  ``max_deviation``
    is the largest deviation from ``reference_value`` (percent, when a
    reference is available) or from the ensemble mean (when not).
    """

    quantity_name: str
    solver_values: Dict[str, float]
    reference_value: Optional[float] = None
    reference_source: str = ""
    mean_value: float = 0.0
    std_value: float = 0.0
    max_deviation: float = 0.0
    max_deviation_solver: str = ""
    pairwise_deviations: Dict[str, float] = field(default_factory=dict)
    status: ComparisonStatus = ComparisonStatus.INSUFFICIENT_DATA
    threshold: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary."""
        return {
            "quantity_name": self.quantity_name,
            "solver_values": self.solver_values,
            "reference_value": self.reference_value,
            "reference_source": self.reference_source,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "max_deviation": self.max_deviation,
            "max_deviation_solver": self.max_deviation_solver,
            "pairwise_deviations": self.pairwise_deviations,
            "status": self.status.value,
            "threshold": self.threshold,
        }


@dataclass
class CanonicalCase:
    """Definition of a canonical test case for cross-solver comparison.

    ``reference_quantities`` lists the observables to compare; ``reference_values``
    (optional) holds the analytical / experimental reference for each.
    """

    name: str
    description: str
    physics: str
    geometry: Dict[str, Any]
    parameters: Dict[str, float]
    reference_quantities: List[str]
    reference_values: Optional[Dict[str, float]] = None
    reference_source: str = ""


class CrossSolverAdapter:
    """Cross-solver benchmark adapter (high-level API).

    Registers canonical cases and per-solver results, then compares the
    ensemble on each case.  ``deviation_threshold`` is the maximum acceptable
    deviation (percent) from the reference / ensemble mean; ``warning_threshold``
    flags near-threshold deviations as WARN.
    """

    def __init__(
        self,
        deviation_threshold: float = 5.0,
        warning_threshold: float = 3.0,
    ):
        if not math.isfinite(deviation_threshold) or deviation_threshold <= 0.0:
            raise ValueError(
                f"deviation_threshold must be finite and positive, got {deviation_threshold!r}"
            )
        if not math.isfinite(warning_threshold) or warning_threshold <= 0.0:
            raise ValueError(
                f"warning_threshold must be finite and positive, got {warning_threshold!r}"
            )
        if warning_threshold >= deviation_threshold:
            raise ValueError(
                f"warning_threshold ({warning_threshold}) must be < "
                f"deviation_threshold ({deviation_threshold})"
            )
        self.deviation_threshold = deviation_threshold
        self.warning_threshold = warning_threshold
        self._registered_cases: Dict[str, CanonicalCase] = {}
        self._solver_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    def register_canonical_case(self, case: CanonicalCase) -> None:
        """Register a canonical test case."""
        if not isinstance(case, CanonicalCase):
            raise TypeError(f"case must be a CanonicalCase, got {type(case).__name__}")
        self._registered_cases[case.name] = case

    def register_solver_result(
        self,
        solver_name: str,
        case_name: str,
        quantities: Dict[str, float],
    ) -> None:
        """Register one solver's results for one canonical case."""
        if case_name not in self._registered_cases:
            raise ValueError(f"case {case_name!r} is not registered")
        for q, v in quantities.items():
            if not math.isfinite(v):
                raise ValueError(f"quantity {q!r} for {solver_name!r} must be finite")
        self._solver_results.setdefault(solver_name, {})[case_name] = dict(quantities)

    def compare_case(self, case_name: str) -> List[ComparisonResult]:
        """Compare all registered solvers on one canonical case.

        Returns one :class:`ComparisonResult` per reference quantity.
        """
        if case_name not in self._registered_cases:
            raise ValueError(f"case {case_name!r} is not registered")
        case = self._registered_cases[case_name]

        # Collect per-solver values for each reference quantity.
        results: List[ComparisonResult] = []
        for qty in case.reference_quantities:
            values: Dict[str, float] = {}
            for solver, cases in self._solver_results.items():
                if case_name in cases and qty in cases[case_name]:
                    values[solver] = cases[case_name][qty]
            if len(values) < 2:
                results.append(
                    ComparisonResult(
                        quantity_name=qty,
                        solver_values=values,
                        reference_value=(
                            case.reference_values.get(qty)
                            if case.reference_values
                            else None
                        ),
                        reference_source=case.reference_source,
                        status=ComparisonStatus.INSUFFICIENT_DATA,
                        threshold=self.deviation_threshold,
                    )
                )
                continue

            # Reference value (analytical / experimental) or ensemble mean.
            ref = (
                case.reference_values.get(qty)
                if case.reference_values and qty in case.reference_values
                else None
            )
            ref_source = case.reference_source if ref is not None else "ensemble-mean"

            # Statistics.
            vals = list(values.values())
            mean = math.fsum(vals) / len(vals)
            var = math.fsum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var)

            # Deviations (percent).
            base = ref if ref is not None else mean
            if abs(base) < 1e-14:
                max_dev = float("inf")
                max_solver = ""
            else:
                devs = {s: 100.0 * abs(v - base) / abs(base) for s, v in values.items()}
                max_solver = max(devs, key=devs.get)
                max_dev = devs[max_solver]

            # Pairwise deviations (percent).
            pairwise: Dict[str, float] = {}
            solvers = list(values.keys())
            for i in range(len(solvers)):
                for j in range(i + 1, len(solvers)):
                    a, b = solvers[i], solvers[j]
                    if abs(values[a]) < 1e-14 or abs(values[b]) < 1e-14:
                        pairwise[f"{a}-{b}"] = float("inf")
                    else:
                        pairwise[f"{a}-{b}"] = (
                            100.0 * abs(values[a] - values[b]) / abs(values[a])
                        )

            # Status.
            if max_dev > self.deviation_threshold:
                status = ComparisonStatus.FAIL
            elif max_dev > self.warning_threshold:
                status = ComparisonStatus.WARN
            else:
                status = ComparisonStatus.PASS

            results.append(
                ComparisonResult(
                    quantity_name=qty,
                    solver_values=values,
                    reference_value=ref,
                    reference_source=ref_source,
                    mean_value=mean,
                    std_value=std,
                    max_deviation=max_dev,
                    max_deviation_solver=max_solver,
                    pairwise_deviations=pairwise,
                    status=status,
                    threshold=self.deviation_threshold,
                )
            )
        return results

    def compare_all(self) -> Dict[str, List[ComparisonResult]]:
        """Compare all registered solvers on every registered case."""
        return {name: self.compare_case(name) for name in self._registered_cases}


# ---------------------------------------------------------------------------
# canonical case factories
# ---------------------------------------------------------------------------


def create_boundary_layer_case() -> CanonicalCase:
    """Flat-plate laminar boundary layer (Blasius) canonical case.

    Reference: Blasius boundary-layer thickness ``delta = 4.64 / sqrt(Re_L) * L``
    at ``Re_L = 1e5``, ``L = 1`` m.
    """
    return CanonicalCase(
        name="flat_plate_bl",
        description="Flat-plate laminar boundary layer (Blasius)",
        physics="laminar boundary layer",
        geometry={"type": "flat_plate", "L": 1.0},
        parameters={"Re_L": 1e5, "U_inf": 1.0},
        reference_quantities=["boundary_layer_thickness"],
        reference_values={"boundary_layer_thickness": 4.64e-3},
        reference_source="Blasius (1908)",
    )


def create_cylinder_drag_case() -> CanonicalCase:
    """Cylinder drag at ``Re = 40`` canonical case.

    Reference: steady laminar cylinder drag coefficient ``C_D = 1.54``
    (Tritton 1959 / Schlichting).
    """
    return CanonicalCase(
        name="cylinder_drag_re40",
        description="Steady laminar flow past a circular cylinder at Re=40",
        physics="cylinder drag",
        geometry={"type": "cylinder", "D": 1.0},
        parameters={"Re": 40.0, "U_inf": 1.0},
        reference_quantities=["drag_coefficient"],
        reference_values={"drag_coefficient": 1.54},
        reference_source="Tritton (1959)",
    )


def create_lid_driven_cavity_case() -> CanonicalCase:
    """Lid-driven cavity at ``Re = 100`` canonical case.

    Reference: centreline ``u`` velocity at the cavity mid-height
    ``u_center = 0.0547`` (Ghia et al. 1982).
    """
    return CanonicalCase(
        name="lid_driven_cavity_re100",
        description="Lid-driven square cavity at Re=100",
        physics="incompressible cavity flow",
        geometry={"type": "square_cavity", "L": 1.0},
        parameters={"Re": 100.0, "U_lid": 1.0},
        reference_quantities=["centerline_u"],
        reference_values={"centerline_u": 0.0547},
        reference_source="Ghia et al. (1982)",
    )
