"""Interface-operator contract for cross-solver exchange (FVM <-> LBM <-> DEM).

Physics motivation
------------------
Gluing two solvers at an interface is, mathematically, applying an *interface
operator*: a map from the source-side interface discretisation to the
target-side one.  A coupling is trustworthy only if the operator is

1. **conservative** -- the integral of the exchanged quantity sent by the donor
   equals the integral received by the acceptor (``conservation_residual``);
2. **consistent** -- the map reproduces the solver's own approximation order on
   smooth data (``observed_order``);
3. **symmetric** -- the result does not depend on the interface labelling
   (reflection equivariance) nor on the visiting order (round trip)
   (``symmetry_flip``).

The module is deliberately geometry-light: interfaces are one-dimensional
(:class:`SegmentMesh`) or point clouds (:class:`PointSet`).  Overlap integrals
are then analytic, so the acceptance numbers are not polluted by quadrature
noise, and every criterion is exactly reproducible.

Ownership: cross-solver coordination semantics + numerical verification
criteria only; no solver kernels, no agent-runtime primitives.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "SegmentMesh",
    "PointSet",
    "AnalyticField",
    "ANALYTIC_FIELDS",
    "exact_values",
    "conservation_residual",
    "OrderEstimate",
    "SymmetryFlip",
    "OperatorReport",
    "InterfaceOperator",
    "uniform_mesh",
    "graded_mesh",
    "reflect_about",
]


# ---------------------------------------------------------------------------
# discretisations
# ---------------------------------------------------------------------------

_GL_X = np.array([-0.8611363115940526, -0.3399810435848563,
                  0.3399810435848563, 0.8611363115940526])
_GL_W = np.array([0.3478548451374538, 0.6521451548625461,
                  0.6521451548625461, 0.3478548451374538])


@dataclass(frozen=True)
class SegmentMesh:
    """1D control-volume partition (FVM cells / LBM lattice cells)."""

    edges: np.ndarray
    label: str = "mesh"

    def __post_init__(self) -> None:
        e = np.asarray(self.edges, dtype=float)
        if e.ndim != 1 or e.size < 2:
            raise ValueError("edges must be a 1D array with at least 2 entries")
        if not np.all(np.diff(e) > 0.0):
            raise ValueError("edges must be strictly increasing")
        object.__setattr__(self, "edges", e)

    # ---- geometry -------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.edges.size - 1)

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def measures(self) -> np.ndarray:
        return self.widths

    @property
    def bounds(self) -> tuple[float, float]:
        return float(self.edges[0]), float(self.edges[-1])

    @property
    def length(self) -> float:
        a, b = self.bounds
        return b - a

    # ---- operations -----------------------------------------------------
    def cell_averages(self, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Cell mean via 4-point Gauss-Legendre (exact for polynomials <= 7)."""
        a = self.edges[:-1]
        h = self.widths
        x = a[:, None] + h[:, None] * 0.5 * (1.0 + _GL_X[None, :])
        vals = np.asarray(f(x), dtype=float)
        return 0.5 * (vals @ _GL_W)

    def reflect(self) -> "SegmentMesh":
        a, b = self.bounds
        return SegmentMesh((a + b) - self.edges[::-1], label=f"reflect({self.label})")

    def submesh(self, lo: float, hi: float) -> "SegmentMesh":
        """Cells strictly inside [lo, hi]; endpoints snapped to lo/hi."""
        e = self.edges
        keep = np.where((e >= lo - 1e-14) & (e <= hi + 1e-14))[0]
        if keep.size < 2:
            raise ValueError("window contains fewer than one cell")
        sub = e[keep[0]: keep[-1] + 1].copy()
        sub[0], sub[-1] = lo, hi
        return SegmentMesh(sub, label=f"{self.label}[{lo},{hi}]")

    def __len__(self) -> int:
        return self.n


@dataclass(frozen=True)
class PointSet:
    """Point cloud (DEM particle centres, lattice nodes used as samples)."""

    centers: np.ndarray
    measures: np.ndarray | None = None
    label: str = "points"
    kind: str = "point"

    def __post_init__(self) -> None:
        c = np.asarray(self.centers, dtype=float)
        if c.ndim != 1 or c.size < 1:
            raise ValueError("centers must be a non-empty 1D array")
        object.__setattr__(self, "centers", c)
        if self.measures is None:
            object.__setattr__(self, "measures", np.ones_like(c))
        else:
            m = np.asarray(self.measures, dtype=float)
            if m.shape != c.shape:
                raise ValueError("measures must match centers")
            object.__setattr__(self, "measures", m)

    @property
    def n(self) -> int:
        return int(self.centers.size)

    def cell_averages(self, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        return np.asarray(f(self.centers), dtype=float)

    def reflect(self) -> "PointSet":
        a, b = float(self.centers.min()), float(self.centers.max())
        return PointSet((a + b) - self.centers[::-1],
                        self.measures[::-1], label=f"reflect({self.label})",
                        kind=self.kind)

    def __len__(self) -> int:
        return self.n


def uniform_mesh(x0: float, x1: float, n: int, label: str = "uniform") -> SegmentMesh:
    return SegmentMesh(np.linspace(x0, x1, n + 1), label=f"{label}[{n}]")


def graded_mesh(x0: float, x1: float, n: int, ratio: float = 1.25,
                label: str = "graded") -> SegmentMesh:
    """Geometric grading, finest cell at x0 (ratio>1 means growing to x1)."""
    h = np.log(ratio)
    xi = np.linspace(0.0, 1.0, n + 1)
    g = (np.exp(h * xi) - 1.0) / (np.exp(h) - 1.0)
    return SegmentMesh(x0 + (x1 - x0) * g, label=f"{label}[{n}]")


def measure(disc) -> np.ndarray:
    return np.asarray(disc.measures, dtype=float)


def reflect_about(disc, axis: tuple[float, float]):
    """Mirror a discretisation about the axis ``(a, b)`` (``x -> a + b - x``)."""
    a, b = float(axis[0]), float(axis[1])
    if isinstance(disc, SegmentMesh):
        return SegmentMesh((a + b) - disc.edges[::-1],
                           label=f"reflect({disc.label})")
    return PointSet((a + b) - np.asarray(disc.centers)[::-1],
                    np.asarray(disc.measures)[::-1],
                    label=f"reflect({disc.label})", kind=disc.kind)


# ---------------------------------------------------------------------------
# analytic test fields (the five field families required by the campaign)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalyticField:
    name: str
    func: Callable[[np.ndarray], np.ndarray]
    note: str = ""


def _boundary_layer(x: np.ndarray, delta: float = 0.06) -> np.ndarray:
    return np.exp(-x / delta)


ANALYTIC_FIELDS: tuple[AnalyticField, ...] = (
    AnalyticField("constant", lambda x: np.ones_like(x) * 3.5,
                  "u=3.5, exact reproduction test (C1)"),
    AnalyticField("linear", lambda x: 0.4 + 1.25 * x,
                  "reproduces the gradient; a 2nd-order operator is exact (C2)"),
    AnalyticField("quadratic", lambda x: 0.5 - 0.75 * x + 2.0 * x * x,
                  "measures the second-order curvature error"),
    AnalyticField("harmonic", lambda x: np.sin(2.0 * np.pi * x),
                  "sign changing field; catches positivity clippers"),
    AnalyticField("boundary_layer", _boundary_layer,
                  "exp(-x/0.06): steep gradient, worst case for local stencils"),
)

FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in ANALYTIC_FIELDS)


def exact_values(disc, field_: AnalyticField) -> np.ndarray:
    """Reference data on ``disc``: cell mean for meshes, point value for clouds."""
    return disc.cell_averages(field_.func)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def conservation_residual(sent_values: np.ndarray, sent_measure: np.ndarray,
                         recv_values: np.ndarray, recv_measure: np.ndarray) -> float:
    """|int_recv - int_sent| / sum |flux|  (dimensionless, C1/C3 form)."""
    sent = float(np.sum(np.asarray(sent_values) * np.asarray(sent_measure)))
    recv = float(np.sum(np.asarray(recv_values) * np.asarray(recv_measure)))
    scale = float(np.sum(np.abs(np.asarray(sent_values)) * np.asarray(sent_measure)))
    if scale <= 0.0:
        scale = 1.0
    return abs(recv - sent) / scale


@dataclass(frozen=True)
class OrderEstimate:
    """Observed order from a three-level refinement study.

    ``order`` is ``None`` when the transfer is exact to round-off on every
    level: a scheme that reproduces the field to machine precision has *no*
    measurable slope, which is the limiting case of ``p -> inf``.  This is the
    mathematically honest bookkeeping for linear fields, on which every
    >=2nd-order consistent operator is exact.
    """

    field: str
    resolutions: tuple[int, ...]
    errors: tuple[float, ...]
    order: float | None
    exact: bool

    def passes(self, p_min: float) -> bool:
        return bool(self.exact or (self.order is not None and self.order >= p_min))

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "resolutions": list(self.resolutions),
            "l2_errors": [float(e) for e in self.errors],
            "order_fit": None if self.order is None else float(self.order),
            "exact_to_roundoff": bool(self.exact),
            "passes_p_min": None,
        }


def estimate_order(field_name: str, resolutions: Sequence[int],
                   errors: Sequence[float], h: Sequence[float],
                   exact_tol: float = 1e-11) -> OrderEstimate:
    errs = np.asarray(errors, dtype=float)
    hs = np.asarray(h, dtype=float)
    exact = bool(np.all(errs <= exact_tol))
    order: float | None
    if exact or np.any(errs <= 0.0):
        order = None
    else:
        slope = np.polyfit(np.log(hs), np.log(np.maximum(errs, 1e-300)), 1)[0]
        order = None if not np.isfinite(slope) else float(slope)
    return OrderEstimate(field_name, tuple(int(r) for r in resolutions),
                         tuple(float(e) for e in errs), order, exact)


@dataclass(frozen=True)
class SymmetryFlip:
    """Direction/labelling sensitivity of an interface operator.

    reflection : ||T(Ru) - R(Tu)|| / ||T(Ru)||   (is the operator mirror-symmetric?)
    roundtrip  : ||T_BA(T_AB u) - u|| / ||u||    (is the exchange reversible?)
    """

    reflection: float
    roundtrip: float

    @property
    def value(self) -> float:
        """Headline number: labelling/mirror sensitivity of the operator.

        ``roundtrip`` is reported next to it but is *not* a defect for dual
        pairs (an exchange operator and its adjoint are not inverses: for the
        LBM-DEM ME the spread side is a smoother by construction).
        """
        return float(self.reflection)

    def to_dict(self) -> dict:
        return {"reflection_defect": float(self.reflection),
                "roundtrip_defect": float(self.roundtrip),
                "value": self.value}


@dataclass
class OperatorReport:
    """Per-(operator, field) result row."""

    operator: str
    field: str
    conservation_residual: float
    l2_error: float
    observed_order: OrderEstimate | None = None
    symmetry_flip: SymmetryFlip | None = None

    def to_dict(self) -> dict:
        return {
            "operator": self.operator,
            "field": self.field,
            "conservation_residual": float(self.conservation_residual),
            "l2_error": float(self.l2_error),
            "observed_order": None if self.observed_order is None
            else self.observed_order.to_dict(),
            "symmetry_flip": None if self.symmetry_flip is None
            else self.symmetry_flip.to_dict(),
        }


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------

class InterfaceOperator(ABC):
    """Contract every cross-solver exchange operator must satisfy.

    Subclasses implement :meth:`transfer`; the three diagnostics required by
    the campaign are provided here so that they are defined identically for
    every operator (single source of truth for the acceptance gates).
    """

    name: str = "interface-operator"
    conservative: bool = True

    # -- the only mandatory method ---------------------------------------
    @abstractmethod
    def transfer(self, src, src_values: np.ndarray, tgt) -> np.ndarray:
        """Map interface data ``src_values`` on ``src`` onto ``tgt``."""

    # -- helpers ---------------------------------------------------------
    def dense_matrix(self, src, tgt) -> np.ndarray:
        """Matrix representation (valid for linear operators; used for audits)."""
        n = src.n
        eye = np.eye(n)
        out = np.zeros((tgt.n, n))
        for i in range(n):
            out[:, i] = self.transfer(src, eye[i], tgt)
        return out

    def conservation_residual(self, src, src_values: np.ndarray, tgt,
                              tgt_values: np.ndarray | None = None) -> float:
        if tgt_values is None:
            tgt_values = self.transfer(src, src_values, tgt)
        return conservation_residual(src_values, measure(src),
                                     tgt_values, measure(tgt))

    def l2_error(self, tgt, tgt_values: np.ndarray,
                 field_: AnalyticField) -> float:
        exact = exact_values(tgt, field_)
        m = measure(tgt)
        num = float(np.sum((tgt_values - exact) ** 2 * m))
        den = float(np.sum(exact ** 2 * m))
        if den <= 0.0:
            den = 1.0
        return float(np.sqrt(num / den))

    def observed_order(self, field_: AnalyticField,
                       pairs: Sequence[tuple], exact_tol: float = 1e-11) -> OrderEstimate:
        """Refinement study over ``pairs`` = [(src, tgt), ...] (coarse -> fine).

        The L2 error is measured against the analytic cell mean on the target
        side, i.e. resolvability, not just pointwise sampling.
        """
        errors, hs, resolutions = [], [], []
        for src, tgt in pairs:
            vals = self.transfer(src, exact_values(src, field_), tgt)
            errors.append(self.l2_error(tgt, vals, field_))
            hs.append(float(np.mean(measure(tgt))))
            resolutions.append(int(tgt.n))
        return estimate_order(field_.name, resolutions, errors, hs,
                              exact_tol=exact_tol)

    def symmetry_flip(self, src, src_values: np.ndarray, tgt,
                      tgt_values: np.ndarray | None = None) -> SymmetryFlip:
        if tgt_values is None:
            tgt_values = self.transfer(src, src_values, tgt)
        # (1) reflection equivariance: T(Ru) ?= R(Tu)   (common mirror axis)
        axis = src.bounds if isinstance(src, SegmentMesh) else \
            (float(np.min(src.centers)), float(np.max(src.centers)))
        rs, rt = reflect_about(src, axis), reflect_about(tgt, axis)
        a = self.transfer(rs, np.asarray(src_values)[::-1], rt)
        b = np.asarray(tgt_values)[::-1]
        den = float(np.sum(a ** 2 * measure(rt))) or 1.0
        refl = float(np.sqrt(float(np.sum((a - b) ** 2 * measure(rt))) / den))
        # (2) round trip tgt -> src -> tgt
        back = self.transfer(tgt, tgt_values, src)
        den2 = float(np.sum(np.asarray(src_values) ** 2 * measure(src))) or 1.0
        rt_def = float(np.sqrt(
            float(np.sum((back - np.asarray(src_values)) ** 2 * measure(src))) / den2))
        return SymmetryFlip(reflection=refl, roundtrip=rt_def)

    # -- reporting -------------------------------------------------------
    def evaluate(self, field_: AnalyticField, src, tgt,
                 pairs: Sequence[tuple] | None = None) -> OperatorReport:
        vals = self.transfer(src, exact_values(src, field_), tgt)
        res = self.conservation_residual(src, exact_values(src, field_), tgt, vals)
        l2 = self.l2_error(tgt, vals, field_)
        flip = self.symmetry_flip(src, exact_values(src, field_), tgt, vals)
        order = None if pairs is None else self.observed_order(field_, pairs)
        return OperatorReport(self.name, field_.name, res, l2, order, flip)