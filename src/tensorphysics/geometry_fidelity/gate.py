"""The cross-solver geometry gate: CAD integrals vs the grid that will be used.

Campaign node **IN50-N08**.  The provenance contract
(:mod:`.contract`, ``geometry_fidelity_contract.v1``) answers *whether* a grid
may enter a physics chain.  This module is the numeric front door that the
mission asks for: a **GeometryGate** that takes the CAD geometry parameters of a
hull, integrates them independently, integrates the *actual discretisation* that
a solver will consume, and returns the five deviations a chain cannot tolerate

    L, B, Cb, nabla (displacement), S (wetted area)

with hard tolerances (0.5 % by default).  Nothing downstream may look at the grid
before :meth:`GeometryGate.evaluate` has passed it.

Why these five: ``Cb``, ``L``, ``B`` fix the resistance / propulsion / stability
denominators, ``nabla`` fixes the mass and the hydrostatics, and ``S`` fixes the
friction.  A grid whose wetted area is 2 % off silently biases every
ITTC-1957 friction number by 2 % -- far outside the 15 % acceptance band on the
total resistance once it is amplified through the form factor.

The discrete side is deliberately *only* nodal geometry: the gate integrates the
station/section grid the solver would read (trapezoid in the section parameter,
Simpson along the hull), never the CAD formula.  Two independent CAD routes
(transverse-slice Simpson and stratified Monte Carlo) live in
:class:`~tensorphysics.geometry_fidelity.hulls.HullForm`, so the CAD number is
itself audited.

The corpus is five hulls: KCS, KVLCC2 and SUBOFF (the published contract hulls)
plus a Series-60-like and a Panamax-bulk-carrier-like form added here, because a
gate validated on one hull is a curve fit, not a contract.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import cad_sha256, canonical_json
from .hulls import HullForm, contract_hulls, kcs_hull, kvlcc2_hull, suboff_hull

__all__ = [
    "C1_TOLERANCES",
    "GateReport",
    "GeometryGate",
    "bulk_carrier_hull",
    "corrupted_discretisation",
    "discretise",
    "gate_corpus",
    "series60_hull",
]

#: Acceptance criterion C1 of IN50-N08: every deviation below 0.5 %.
C1_TOLERANCES: dict[str, float] = {
    "l_rel": 5e-3,
    "b_rel": 5e-3,
    "nabla_rel": 5e-3,
    "s_rel": 5e-3,
    "cb_rel": 5e-3,
}


# ---------------------------------------------------------------------------
# two additional wall-sided hulls (five-hull corpus)
# ---------------------------------------------------------------------------


def _solve_section_exponent(base: Mapping[str, float], cb_target: float) -> float:
    """Section exponent ``m`` that closes ``Cb`` for a ``wall_sided_power`` hull.

    Uses only the public waterplane quadrature: ``Cb = Cwp * m / (m + 1)``.
    """
    probe = HullForm("probe", "wall_sided_power", 1.0, 1.0, 1.0, dict(base))
    cwp = probe.cad_waterplane_area()
    ratio = cb_target / cwp
    if not 0.05 < ratio < 0.995:
        raise ValueError(f"Cb/Cwp = {ratio:.4f} outside the realisable band")
    return ratio / (1.0 - ratio)


def series60_hull() -> HullForm:
    """Series 60 (Cb 0.600, L/B 7.5): the classic fine-form resistance hull."""
    base = {
        "mid": 0.49,
        "transom": 0.08,
        "fore_a": 4.5,
        "fore_b": 1.05,
        "aft_a": 3.2,
        "aft_b": 1.00,
        "m": 1.0,
    }
    m = _solve_section_exponent(base, 0.600)
    return HullForm("series60", "wall_sided_power", 121.92, 16.256, 6.502,
                    dict(base, m=m), cb_contract=0.600)


def bulk_carrier_hull() -> HullForm:
    """Panamax bulk carrier (Cb 0.830): full form, flat-bottomed sections."""
    base = {
        "mid": 0.50,
        "transom": 0.22,
        "fore_a": 6.5,
        "fore_b": 1.25,
        "aft_a": 4.5,
        "aft_b": 1.15,
        "m": 1.0,
    }
    m = _solve_section_exponent(base, 0.830)
    return HullForm("bulk_carrier", "wall_sided_power", 225.0, 32.26, 13.80,
                    dict(base, m=m), cb_contract=0.830)


def gate_corpus() -> dict[str, HullForm]:
    """The five-hull corpus of the geometry gate (KCS, KVLCC2, SUBOFF + 2)."""
    hulls = dict(contract_hulls())          # kcs, kvlcc2, suboff
    hulls["series60"] = series60_hull()
    hulls["bulk_carrier"] = bulk_carrier_hull()
    return hulls


# ---------------------------------------------------------------------------
# discretisation: the nodal grid a solver would actually read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Discretisation:
    """Nodal parametrisation of the wetted surface, ``(n_x+1, n_s+1)``.

    ``x`` are station positions (cosine clustered), ``s`` the section parameter
    (``z = T s^m`` for a wall-sided hull, the circular angle for a body of
    revolution), ``y``/``z`` the resulting half-breadth and height.  These nodes
    are the only input of the discrete integrals.
    """

    xs: np.ndarray
    ss: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    label: str = "grid"

    @property
    def n_x(self) -> int:
        return int(self.xs.size - 1)

    @property
    def n_s(self) -> int:
        return int(self.ss.size - 1)

    @property
    def length(self) -> float:
        return float(self.xs[-1] - self.xs[0])

    @property
    def max_breadth(self) -> float:
        return 2.0 * float(np.max(self.ys))

    def section_areas(self) -> np.ndarray:
        """Section area from the nodal section polygon (trapezoid in ``s``)."""
        y, z = self.ys, self.zs
        dz = np.diff(z, axis=1)
        return 2.0 * np.sum(0.5 * (y[:, :-1] + y[:, 1:]) * dz, axis=1)

    def volume(self) -> float:
        """Displacement from the nodal section areas: trapezoid along the hull.

        The stations are cosine clustered, so the *actual* spacings
        ``diff(xs)`` are used (a uniform-``h`` Simpson rule on clustered nodes
        is simply wrong -- that mistake inflated the displacement by 20 % on the
        first version of this gate and is kept as a recorded pitfall).
        """
        a = self.section_areas()
        dx = np.diff(self.xs)
        return float(np.sum(0.5 * (a[:-1] + a[1:]) * dx))

    def side_area(self) -> float:
        """Wetted side area from nodal quads (both sides), no end planes."""
        p = np.stack([self.xs[:, None] * np.ones_like(self.ys), self.ys, self.zs], axis=-1)
        # quad (i,k)-(i+1,k)-(i+1,k+1)-(i,k+1) split into two triangles
        q00, q10 = p[:-1, :-1], p[1:, :-1]
        q11, q01 = p[1:, 1:], p[:-1, 1:]
        a1 = 0.5 * np.linalg.norm(np.cross(q10 - q00, q11 - q00), axis=-1)
        a2 = 0.5 * np.linalg.norm(np.cross(q11 - q00, q01 - q00), axis=-1)
        return 2.0 * float(np.sum(a1 + a2))

    def end_area(self) -> float:
        """Area of the two transverse end planes (transom / bow, both sides)."""
        a = self.section_areas()
        return float(a[0] + a[-1])

    def surface(self) -> float:
        """Total wetted area = sides + both end planes (same convention as CAD)."""
        return self.side_area() + self.end_area()

    def digest(self) -> str:
        payload = {
            "label": self.label,
            "xs": [float(v) for v in self.xs],
            "ss": [float(v) for v in self.ss],
            "ys": [float(v) for v in self.ys.ravel()],
            "zs": [float(v) for v in self.zs.ravel()],
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def discretise(
    hull: HullForm,
    n_x: int = 120,
    n_s: int = 48,
    *,
    stretch: float = 1.0,
    section_shift: float = 0.0,
    truncate: float = 0.0,
    label: str | None = None,
) -> Discretisation:
    """Build the nodal grid of ``hull`` (cosine stations x uniform sections).

    ``n_x`` must be even so that Simpson and the cosine spacing both hit
    ``x = L/2`` (the section that carries ``B``).  The keyword arguments are the
    deliberate corruptions used by the gate's negative controls.
    """
    if n_x % 2 or n_s % 2:
        raise ValueError("n_x and n_s must be even")
    base = np.linspace(0.0, 1.0, n_x + 1)
    xi = 0.5 * (1.0 - np.cos(np.pi * base))            # cosine clustering
    if stretch != 1.0:
        xi = 0.5 + (xi - 0.5) * stretch
    if truncate > 0.0:
        xi = xi * (1.0 - truncate)
    xs = hull.length_m * xi
    ss = np.linspace(0.0, 1.0, n_s + 1)
    s_eff = np.clip(ss + section_shift, 0.0, 1.0)
    ys = np.zeros((n_x + 1, n_s + 1))
    zs = np.zeros((n_x + 1, n_s + 1))
    for i, x in enumerate(xs):
        for k, s in enumerate(ss):
            ys[i, k], zs[i, k] = hull.surface_yz(float(x), float(s_eff[k]))
    return Discretisation(xs, ss, ys, zs,
                          label=label or f"body_fitted[{(n_x, n_s)}]")


#: Deliberate grid corruptions (the gate must reject all of them).
CORRUPTIONS: Mapping[str, dict[str, float]] = {
    "truncate_1pct": {"truncate": 0.01},
    "stretch_2pct": {"stretch": 1.02},
    "section_shift_half_cell": {"section_shift": 0.5 / 48.0},
}


def corrupted_discretisation(hull: HullForm, kind: str, n_x: int = 120,
                             n_s: int = 48) -> Discretisation:
    if kind not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {kind!r}; expected one of {sorted(CORRUPTIONS)}")
    kwargs = dict(CORRUPTIONS[kind])
    if "section_shift" in kwargs:
        kwargs["section_shift"] = kwargs["section_shift"] * (48.0 / n_s)
    return discretise(hull, n_x, n_s, label=f"{kind}", **kwargs)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateReport:
    """The five deviations plus the provenance digests, JSON-ready."""

    geometry_id: str
    cad: Mapping[str, float]
    discrete: Mapping[str, float]
    deviations: Mapping[str, float]
    tolerances: Mapping[str, float]
    checks: tuple[Mapping[str, Any], ...]
    verdict: str
    provenance: Mapping[str, str]
    grid_label: str

    @property
    def passed(self) -> bool:
        return self.verdict == "GATE_PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_id": self.geometry_id,
            "cad": {k: float(v) for k, v in self.cad.items()},
            "discrete": {k: float(v) for k, v in self.discrete.items()},
            "deviations": {k: float(v) for k, v in self.deviations.items()},
            "tolerances": {k: float(v) for k, v in self.tolerances.items()},
            "checks": [dict(c) for c in self.checks],
            "verdict": self.verdict,
            "provenance": dict(self.provenance),
            "grid_label": self.grid_label,
        }

    def as_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class GeometryGate:
    """CAD-vs-grid gate for the five principal integral quantities.

    ``evaluate(hull)`` builds the solver grid, integrates it from nodal geometry,
    integrates the CAD truth source on its *own* quadrature, and compares.  The
    gate is the only sanctioned entry point: a chain that skips it has no
    provenance.
    """

    def __init__(
        self,
        n_x: int = 120,
        n_s: int = 48,
        tolerances: Mapping[str, float] | None = None,
        cad_quadrature: tuple[int, int] = (240, 160),
    ) -> None:
        self.n_x = int(n_x)
        self.n_s = int(n_s)
        self.tolerances = dict(C1_TOLERANCES if tolerances is None else tolerances)
        self.cad_quadrature = (int(cad_quadrature[0]), int(cad_quadrature[1]))

    # -- the two sides ----------------------------------------------------
    def cad_integrals(self, hull: HullForm) -> dict[str, float]:
        n_x, n_s = self.cad_quadrature
        volume = hull.cad_volume_slice(n_x, n_s)
        area = hull.cad_surface_area(n_x, n_s)
        return {
            "l_m": hull.length_m,
            "b_m": hull.breadth_m,
            "nabla_m3": volume,
            "s_wetted_m2": area,
            "cb": volume / (hull.length_m * hull.breadth_m * hull.draught_m),
        }

    def discrete_integrals(self, disc: Discretisation, draught: float) -> dict[str, float]:
        length = disc.length
        breadth = disc.max_breadth
        volume = disc.volume()
        area = disc.surface()
        return {
            "l_m": length,
            "b_m": breadth,
            "nabla_m3": volume,
            "s_wetted_m2": area,
            "cb": volume / (length * breadth * draught),
        }

    # -- the gate ---------------------------------------------------------
    def evaluate(self, hull: HullForm, disc: Discretisation | None = None) -> GateReport:
        disc = discretise(hull, self.n_x, self.n_s) if disc is None else disc
        cad = self.cad_integrals(hull)
        dis = self.discrete_integrals(disc, hull.draught_m)
        deviations = {
            "l_rel": abs(dis["l_m"] - cad["l_m"]) / cad["l_m"],
            "b_rel": abs(dis["b_m"] - cad["b_m"]) / cad["b_m"],
            "nabla_rel": abs(dis["nabla_m3"] - cad["nabla_m3"]) / cad["nabla_m3"],
            "s_rel": abs(dis["s_wetted_m2"] - cad["s_wetted_m2"]) / cad["s_wetted_m2"],
            "cb_rel": abs(dis["cb"] - cad["cb"]) / cad["cb"],
        }
        checks = tuple(
            {
                "id": key,
                "value": float(deviations[key]),
                "limit": float(self.tolerances[key]),
                "pass": bool(deviations[key] <= self.tolerances[key]),
            }
            for key in ("l_rel", "b_rel", "nabla_rel", "s_rel", "cb_rel")
        )
        passed = all(c["pass"] for c in checks)
        provenance = {
            "cad_sha256": cad_sha256(hull, *self.cad_quadrature),
            "grid_sha256": disc.digest(),
        }
        return GateReport(
            geometry_id=hull.geometry_id,
            cad=cad,
            discrete=dis,
            deviations=deviations,
            tolerances=self.tolerances,
            checks=checks,
            verdict="GATE_PASS" if passed else "GATE_REJECT",
            provenance=provenance,
            grid_label=disc.label,
        )

    def evaluate_corpus(
        self,
        corpus: Mapping[str, HullForm] | None = None,
        resolutions: Sequence[tuple[int, int]] = ((120, 48), (96, 40)),
    ) -> dict[str, Any]:
        """Evaluate every hull at every resolution; the worse deviation rules.

        A grid-independence probe: a gate that only passes at one resolution is a
        resolution artefact, so the corpus is run twice and the per-quantity
        maximum deviation is what gets compared with the tolerance.  The default
        pair is the production window; the acceptance harness additionally runs a
        deliberately coarse probe (60x24) and reports that the fullest hull is
        then *rejected* -- honesty over convenience.
        """
        corpus = gate_corpus() if corpus is None else corpus
        reports: list[GateReport] = []
        for name, hull in corpus.items():
            for n_x, n_s in resolutions:
                reports.append(self.evaluate(hull, discretise(hull, n_x, n_s)))
        per_hull: dict[str, Any] = {}
        for report in reports:
            entry = per_hull.setdefault(report.geometry_id, {
                "verdict": "GATE_PASS",
                "grids": [],
                "worst": {k: 0.0 for k in report.deviations},
                "cad": dict(report.cad),
                "provenance": dict(report.provenance),
            })
            entry["grids"].append({
                "label": report.grid_label,
                "deviations": dict(report.deviations),
                "verdict": report.verdict,
            })
            for key, value in report.deviations.items():
                entry["worst"][key] = max(entry["worst"][key], float(value))
            if not report.passed:
                entry["verdict"] = "GATE_REJECT"
        worst = {k: max(v["worst"][k] for v in per_hull.values()) for k in C1_TOLERANCES}
        verdict = ("GATE_PASS" if all(v <= self.tolerances[k] for k, v in worst.items())
                   else "GATE_REJECT")
        return {
            "criterion": "C1",
            "tolerances": dict(self.tolerances),
            "n_hulls": len(per_hull),
            "hulls": per_hull,
            "worst_deviation": worst,
            "verdict": verdict,
        }

    def negative_controls(self, hull: HullForm | None = None,
                          kinds: Sequence[str] | None = None) -> dict[str, Any]:
        """The gate must reject every deliberately damaged grid."""
        hull = kcs_hull() if hull is None else hull
        kinds = tuple(CORRUPTIONS) if kinds is None else tuple(kinds)
        rows = []
        for kind in kinds:
            disc = corrupted_discretisation(hull, kind, self.n_x, self.n_s)
            report = self.evaluate(hull, disc)
            rows.append({
                "corruption": kind,
                "verdict": report.verdict,
                "deviations": dict(report.deviations),
                "rejected": not report.passed,
            })
        return {
            "geometry_id": hull.geometry_id,
            "controls": rows,
            "n_rejected": sum(1 for r in rows if r["rejected"]),
            "n_controls": len(rows),
        }