"""Plate limit-load verification surface (yield-line / mechanism upper bounds).

Companion of :mod:`tensorphysics.plasticity.beam_hinge`: the beam module gives
the incremental elastoplastic path (M-kappa, hinge formation, springback) while
this module covers the *plate* side of the same J2 limit-strength surface with
classical limit analysis -- rigid-perfectly-plastic mechanisms whose internal
work is dissipated on plastic hinge lines and whose external work is the load
times the volume under the collapse field.

Mechanisms implemented
----------------------
``rectangular_plate_mechanism`` 4 planar facets, each spanning one supported
    edge and the plate centre; hinges along the four corner-to-centre lines.
    For a square plate this is the classical pyramid mechanism.
``circular_plate_mechanism`` conical collapse field hinged at the support
    circle (the exact mechanism of the simply supported circular plate).

References (closed forms the code is checked against)
-----------------------------------------------------
* Simply supported square plate, uniform pressure: ``p_u = 24 M_p / a**2``
  (yield-line pyramid; see Chen & Han 1988, ch. 6, or Jones 1989 ch. 4).
* Simply supported circular plate, uniform pressure: ``p_u = 6 M_p / R**2``
  (exact solution; Hopkins & Wang 1955; also the conical mechanism above).
* Long rectangular plate (``a >> b``): ``p_u -> 8 M_p / b**2`` (cylindrical
  bending limit of the same 4-facet family).

Notes
-----
Upper-bound values, ``M_p = sigma_y t**2 / 4`` per unit width for a von Mises /
Tresca plate strip.  Membrane action and finite deflections are outside the
envelope of this module (limit analysis is infinitesimal).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = [
    "Facet",
    "MechanismResult",
    "rectangular_plate_mechanism",
    "circular_plate_mechanism",
    "plastic_moment_per_width",
    "square_plate_limit",
    "strip_plate_mechanism",
]


def plastic_moment_per_width(sigma_y: float, thickness: float) -> float:
    """Fully plastic moment per unit width, ``M_p = sigma_y t^2 / 4``."""
    return sigma_y * thickness**2 / 4.0


@dataclass(frozen=True)
class Facet:
    """Planar facet of a rigid collapse mechanism.

    ``vertices`` are the in-plane corner coordinates (3, 2) and ``grad`` the
    constant in-plane deflection gradient (deflection per unit length) with the
    facet deflection ``delta(x) = grad . (x - x0)`` and ``x0`` the facet origin.
    """

    vertices: np.ndarray
    grad: np.ndarray
    origin: np.ndarray

    def delta(self, x: np.ndarray) -> float:
        return float(np.dot(self.grad, np.asarray(x, dtype=float) - self.origin))

    @property
    def area(self) -> float:
        """Polygon area (shoelace formula; works for triangles and quads)."""
        v = np.asarray(self.vertices, dtype=float)
        x, y = v[:, 0], v[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def centroid(self) -> np.ndarray:
        v = np.asarray(self.vertices, dtype=float)
        x, y = v[:, 0], v[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        a = cross.sum()
        if abs(a) < 1e-30:  # pragma: no cover - defensive
            return v.mean(axis=0)
        return np.array(
            [
                float(((x + np.roll(x, -1)) * cross).sum() / (3.0 * a)),
                float(((y + np.roll(y, -1)) * cross).sum() / (3.0 * a)),
            ]
        )

    def external_work_per_unit_pressure(self) -> float:
        """``int delta dA`` over the facet (linear field -> value at centroid)."""
        return float(self.area * self.delta(self.centroid))


@dataclass
class MechanismResult:
    pressure: float
    internal_work: float
    external_work_per_pressure: float
    hinges: list[dict]
    name: str

    def as_dict(self) -> dict:
        return {
            "mechanism": self.name,
            "collapse_pressure": self.pressure,
            "internal_work": self.internal_work,
            "external_work_unit_pressure": self.external_work_per_pressure,
            "hinges": [
                {
                    "start": [float(v) for v in h["start"]],
                    "end": [float(v) for v in h["end"]],
                    "length": h["length"],
                    "rotation": h["rotation"],
                    "work": h["work"],
                }
                for h in self.hinges
            ],
        }


def _shared_vertices(f1: Facet, f2: Facet, tol: float = 1e-9) -> list[np.ndarray]:
    shared = []
    for a in f1.vertices:
        for b in f2.vertices:
            if np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)) < tol:
                shared.append(np.asarray(a, float))
                break
    return shared


def _hinge_record(f1: Facet, f2: Facet, M_p: float) -> dict | None:
    shared = _shared_vertices(f1, f2)
    if len(shared) != 2:
        return None
    p1, p2 = shared
    d = p2 - p1
    length = float(np.linalg.norm(d))
    if length < 1e-12:
        return None
    n = np.array([-d[1], d[0]]) / length  # in-plane normal to the hinge line
    rotation = abs(float(np.dot(f1.grad - f2.grad, n)))
    return {
        "start": p1,
        "end": p2,
        "length": length,
        "rotation": rotation,
        "work": M_p * length * rotation,
    }


def _mechanism_result(
    facets: Iterable[Facet], M_p: float, name: str
) -> MechanismResult:
    facets = list(facets)
    hinges: list[dict] = []
    for i in range(len(facets)):
        for j in range(i + 1, len(facets)):
            rec = _hinge_record(facets[i], facets[j], M_p)
            if rec is not None:
                hinges.append(rec)
    internal = float(sum(h["work"] for h in hinges))
    external = float(sum(f.external_work_per_unit_pressure() for f in facets))
    if external <= 0.0:  # pragma: no cover - defensive
        raise ValueError("mechanism has no external work")
    return MechanismResult(
        pressure=internal / external,
        internal_work=internal,
        external_work_per_pressure=external,
        hinges=hinges,
        name=name,
    )


def rectangular_plate_mechanism(
    a: float, b: float, M_p: float, delta0: float = 1.0
) -> MechanismResult:
    """Four-facet (corner-hinged pyramid) mechanism of a simply supported plate.

    The plate occupies ``[0, a] x [0, b]``; each facet is a triangle formed by
    one supported edge and the plate centre, so the deflection vanishes on the
    edges and equals ``delta0`` at the centre.
    """
    centre = np.array([0.5 * a, 0.5 * b])
    corners = [
        np.array([0.0, 0.0]),
        np.array([a, 0.0]),
        np.array([a, b]),
        np.array([0.0, b]),
    ]
    facets: list[Facet] = []
    # edge i is spanned by corners[i] -> corners[i+1]
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        edge_mid = 0.5 * (p + q)
        normal = centre - edge_mid
        dist = float(np.linalg.norm(normal))
        facets.append(
            Facet(
                vertices=np.array([p, q, centre]),
                grad=normal / dist * (delta0 / dist),
                origin=edge_mid,
            )
        )
    return _mechanism_result(facets, M_p, f"rectangular 4-facet a={a:g} b={b:g}")


def square_plate_limit(a: float, M_p: float, delta0: float = 1.0) -> float:
    """Collapse pressure of a simply supported square plate (upper bound)."""
    return rectangular_plate_mechanism(a, a, M_p, delta0).pressure


def strip_plate_mechanism(a: float, b: float, M_p: float, delta0: float = 1.0) -> MechanismResult:
    """Cylindrical-bending (mid-line hinge) mechanism of a simply supported strip.

    Two rectangular facets hinged along ``y = b/2``; gives the long-plate limit
    ``p_u = 8 M_p / b**2`` (the beam-strip collapse load per unit width).
    """
    facets = [
        Facet(
            vertices=np.array([[0.0, 0.0], [a, 0.0], [a, 0.5 * b], [0.0, 0.5 * b]]),
            grad=np.array([0.0, 2.0 * delta0 / b]),
            origin=np.array([0.0, 0.0]),
        ),
        Facet(
            vertices=np.array([[0.0, 0.5 * b], [a, 0.5 * b], [a, b], [0.0, b]]),
            grad=np.array([0.0, -2.0 * delta0 / b]),
            origin=np.array([0.0, b]),
        ),
    ]
    return _mechanism_result(facets, M_p, f"strip midline a={a:g} b={b:g}")


def circular_plate_mechanism(
    R: float, M_p: float, delta0: float = 1.0, n_theta: int = 720, n_r: int = 400
) -> MechanismResult:
    """Conical mechanism of a simply supported circular plate, hinge at ``r=R``.

    ``delta(r) = delta0 (1 - r/R)`` -- internal work on the support hinge
    circle ``2 pi R`` with rotation ``delta0/R``; external work by radial
    quadrature.  Gives the exact limit pressure ``6 M_p / R**2``.
    """
    rotation = delta0 / R
    length = 2.0 * math.pi * R
    internal = M_p * length * rotation
    r = (np.arange(n_r) + 0.5) * (R / n_r)
    delta = delta0 * (1.0 - r / R)
    external = float(np.sum(2.0 * math.pi * r * delta * (R / n_r)))
    return MechanismResult(
        pressure=internal / external,
        internal_work=internal,
        external_work_per_pressure=external,
        hinges=[
            {
                "start": np.array([R, 0.0]),
                "end": np.array([R, 0.0]),
                "length": length,
                "rotation": rotation,
                "work": internal,
            }
        ],
        name=f"circular conical R={R:g}",
    )