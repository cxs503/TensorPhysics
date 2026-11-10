"""Analytic potential-flow fields around the bluff sections of a superstructure.

The droplet-impingement model needs a *physical* air-velocity field around the
structures that catch ice (masts, pipes, crane booms, blunt superstructure
edges).  Two exact solutions of the Laplace equation are provided:

``CylinderFlow``
    Uniform flow past a circular cylinder of radius ``R``
    (complex potential ``w = U (z + R^2/z)``, Milne-Thomson).

``EllipseFlow``
    Uniform flow past an ellipse of semi-axes ``A`` (streamwise) and ``B``
    (cross-stream) obtained from the Joukowski map ``z = zeta + c^2/zeta`` of a
    circle of radius ``mu``.  Reduces to a flat plate as ``B/A -> 0`` and to the
    cylinder as ``B/A -> 1``, so the same kernel covers pipes, crane booms and
    (as a blunt-edge surrogate) the superstructure front face.

Both classes expose the same duck-typed interface used by
:mod:`tensorphysics.superstructure_icing.droplets`:

``velocity(x, y)``            air velocity (vectorised)
``contains(x, y)``            inside-body mask
``surface(n)``                upper-half surface polyline ``(xs, ys, ss)`` with
                              arc length ``s`` measured from the stagnation point
``project(x, y)``             nearest point on that polyline -> ``(s, dist)``
``s_stag``                    arc-length of the stagnation point (always 0)
``height``                    half-height normal to the flow (used to normalise
                              the total collection efficiency)
``diameter``                  characteristic length (used in Reynolds/Stokes numbers)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class BluffBody:
    """Common interface for the analytic sections (see module docstring)."""

    name: str = "body"
    U: float = 15.0

    # -- geometry ---------------------------------------------------------
    @property
    def height(self) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def diameter(self) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def velocity(self, x, y):  # pragma: no cover - overridden
        raise NotImplementedError

    def contains(self, x, y):  # pragma: no cover - overridden
        raise NotImplementedError

    def surface(self, n: int = 4001):
        raise NotImplementedError

    def gap(self, x, y):
        """Normalised signed gap to the wall (``<0`` inside the body)."""
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------
    def project(self, x, y, n: int = 8001):
        """Return ``(s, dist)`` of the nearest point of the upper surface."""
        xs, ys, ss = self.surface(n)
        d2 = (xs[None, :] - np.atleast_1d(x)[:, None]) ** 2 + (
            ys[None, :] - np.atleast_1d(y)[:, None]
        ) ** 2
        j = np.argmin(d2, axis=1)
        return ss[j], np.sqrt(d2[np.arange(j.size), j])


@dataclass
class CylinderFlow(BluffBody):
    """Uniform flow past a circular cylinder."""

    R: float = 0.1
    U: float = 15.0
    name: str = "cylinder"

    @property
    def height(self) -> float:
        return float(self.R)

    @property
    def diameter(self) -> float:
        return 2.0 * float(self.R)

    # -- flow -------------------------------------------------------------
    def velocity(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r2 = np.maximum(x * x + y * y, (1e-6 * self.R) ** 2)
        r4 = r2 * r2
        R2 = self.R * self.R
        u = self.U * (1.0 - R2 * (x * x - y * y) / r4)
        v = self.U * (-2.0 * R2 * x * y / r4)
        return u, v

    def contains(self, x, y):
        return (np.asarray(x) ** 2 + np.asarray(y) ** 2) < self.R**2

    def gap(self, x, y):
        return np.hypot(np.asarray(x, dtype=float), np.asarray(y, dtype=float)) / self.R - 1.0

    def surface(self, n: int = 4001):
        phi = np.linspace(0.0, np.pi, n)
        xs = -self.R * np.cos(phi)
        ys = self.R * np.sin(phi)
        return xs, ys, self.R * phi

    def surface_point(self, s):
        """Point and outward normal at arc length ``s`` from stagnation."""
        phi = np.asarray(s, dtype=float) / self.R
        return -self.R * np.cos(phi), self.R * np.sin(phi)


@dataclass
class EllipseFlow(BluffBody):
    """Uniform flow past an ellipse (Joukowski map of a circle)."""

    A: float = 0.5  # semi-axis along the flow [m]
    B: float = 0.1  # semi-axis normal to the flow [m]
    U: float = 15.0
    name: str = "ellipse"

    def __post_init__(self):
        mu = 0.5 * (self.A + self.B)
        self.mu = mu
        self.c2 = 0.25 * (self.A**2 - self.B**2)

    @property
    def height(self) -> float:
        return float(self.B)

    @property
    def diameter(self) -> float:
        return 2.0 * float(self.B)

    def _zeta(self, x, y):
        z = np.asarray(x, dtype=complex) + 1j * np.asarray(y, dtype=complex)
        root = np.sqrt(z * z - 4.0 * self.c2 + 0j)
        z1 = 0.5 * (z + root)
        z2 = 0.5 * (z - root)
        # the physical domain is the *exterior* of the circle |zeta| = mu, so the
        # pre-image is always the root with the larger modulus (picking "the first
        # root with |zeta| > mu" fails by round-off exactly on the wall, where the
        # velocity is then evaluated on the wrong branch)
        return np.where(np.abs(z1) >= np.abs(z2), z1, z2)

    def velocity(self, x, y):
        zeta = self._zeta(x, y)
        dwdz = self.U * (1.0 - self.mu**2 / zeta**2) / (1.0 - self.c2 / zeta**2)
        return dwdz.real, -dwdz.imag

    def contains(self, x, y):
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        return (xa / self.A) ** 2 + (ya / self.B) ** 2 < 1.0

    def gap(self, x, y):
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        return np.hypot(xa / self.A, ya / self.B) - 1.0

    def surface(self, n: int = 4001):
        theta = np.linspace(np.pi, 0.0, n)  # stagnation (x=-A) -> trailing edge
        xs = self.A * np.cos(theta)
        ys = self.B * np.sin(theta)
        ds = np.hypot(np.diff(xs), np.diff(ys))
        ss = np.concatenate([[0.0], np.cumsum(ds)])
        return xs, ys, ss

    def surface_point(self, s):
        xs, ys, ss = self.surface(20001)
        phi = np.linspace(np.pi, 0.0, 20001)
        return np.interp(s, ss, xs), np.interp(s, ss, ys)


def section_flow(kind: str, **kwargs) -> BluffBody:
    """Factory used by the superstructure geometry model."""
    if kind == "cylinder":
        return CylinderFlow(**{k: v for k, v in kwargs.items() if k in {"R", "U"}})
    if kind == "ellipse":
        return EllipseFlow(**{k: v for k, v in kwargs.items() if k in {"A", "B", "U"}})
    raise ValueError(f"unknown section kind: {kind!r}")