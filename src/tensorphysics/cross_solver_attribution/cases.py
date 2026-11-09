"""Analytic reference solutions driving the cross-solver arbitration protocol.

Every case in this module is an *exact* solution of the equation it is used to
test, so that the same driving data can be fed to two different discretisations
(DG and FVM, or SPH and LBM) and the difference between the two answers can be
compared with the distance of each to the truth.  That triple
``(e_DG, e_FVM, D_DG-FVM)`` is what the attribution ledger decomposes.

Honesty notes (recorded here, not in a footnote):

* ``advection_sine`` is an exact solution of ``u_t + a u_x = 0``.
* ``diffusion_erf`` uses the similarity solution ``u = (1 - erf(eta))/2`` of the
  heat equation, ``eta = (x - xc) / sqrt(4 nu (t + t0))``.  A ``tanh`` front is
  **not** an exact heat-equation solution (it solves a different similarity
  problem), so the erf form is used and the shape is reported as what it is: a
  smooth tanh-type front.
* ``swe_lake_*``: still water, ``u = 0`` and ``h + b = const``; exact steady
  solution of the shallow-water equations with a bed.
* ``swe_ritter``: Ritter (1892) dry-bed dam break, exact rarefaction solution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import erf

# 8-point Gauss-Legendre on [-1, 1]; exact for polynomials up to degree 15, so
# cell averages of smooth fields are exact to round-off.
GL_X, GL_W = np.polynomial.legendre.leggauss(8)


def cell_averages(func: Callable[[np.ndarray], np.ndarray], x0: float, x1: float,
                  n: int) -> np.ndarray:
    """Exact (to round-off) cell averages of ``func`` on ``n`` uniform cells."""
    edges = np.linspace(float(x0), float(x1), n + 1)
    h = edges[1] - edges[0]
    xq = edges[:-1, None] + 0.5 * h * (GL_X[None, :] + 1.0)
    vals = np.asarray(func(xq), dtype=float)
    return 0.5 * (vals * GL_W[None, :]).sum(axis=1)


def relative_l2(numeric: np.ndarray, reference: np.ndarray) -> float:
    """Discrete relative L2 norm on cell averages (uniform cell weight cancels)."""
    num = np.asarray(numeric, dtype=float)
    ref = np.asarray(reference, dtype=float)
    den = float(np.sqrt(np.mean(ref ** 2)))
    if not np.all(np.isfinite(num)):
        return float("inf")
    diff = float(np.sqrt(np.mean((num - ref) ** 2)))
    if den == 0.0:
        return diff
    return diff / den


# ---------------------------------------------------------------------------
# scalar advection / diffusion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScalarCase:
    """Scalar advection-diffusion reference case.

    ``kind`` selects the analytic solution; both kinds are exact solutions of
    ``u_t + a u_x = nu u_xx`` on the stated domain and time window.
    """

    name: str
    kind: str
    x0: float
    x1: float
    t_end: float
    a: float = 0.0
    nu: float = 0.0
    k: float = 1.0
    t0: float = 0.02

    def exact(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.kind == "advection_sine":
            return np.sin(2.0 * math.pi * self.k * (x - self.a * t))
        if self.kind == "diffusion_erf":
            xc = 0.5 * (self.x0 + self.x1)
            eta = (x - xc) / math.sqrt(4.0 * self.nu * (t + self.t0))
            return 0.5 * (1.0 - erf(eta))
        raise ValueError(f"unknown scalar case kind {self.kind!r}")

    def initial(self, x):
        return self.exact(x, 0.0)

    def exact_averages(self, n: int) -> np.ndarray:
        return cell_averages(lambda x: self.exact(x, self.t_end), self.x0, self.x1, n)

    def initial_averages(self, n: int) -> np.ndarray:
        return cell_averages(self.initial, self.x0, self.x1, n)

    def boundary_values(self, t: float) -> tuple[float, float]:
        return float(self.exact(self.x0, t)), float(self.exact(self.x1, t))


def advection_sine(k: int = 1, t_end: float = 0.5, a: float = 1.0) -> ScalarCase:
    """Periodic travelling sine wave (exact solution of pure advection)."""
    return ScalarCase(name=f"advection_sine_k{k}", kind="advection_sine",
                      x0=0.0, x1=1.0, t_end=t_end, a=a, nu=0.0, k=k)


def diffusion_front(nu: float = 0.05, t_end: float = 0.2) -> ScalarCase:
    """Smooth tanh-type front, the erf similarity solution of the heat equation."""
    return ScalarCase(name="diffusion_erf_front", kind="diffusion_erf",
                      x0=0.0, x1=2.0, t_end=t_end, a=0.0, nu=nu, t0=0.02)


# ---------------------------------------------------------------------------
# shallow water
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SWECase:
    """Shallow-water reference case on a bed ``b(x)``.

    ``kind`` is one of ``lake_flat`` (still lake, flat bed), ``lake_bump``
    (still lake over a smooth Gaussian bump -- the well-balancedness test) or
    ``ritter`` (dry-bed dam break, exact rarefaction).
    """

    name: str
    kind: str
    x0: float
    x1: float
    t_end: float
    g: float = 9.81
    h0: float = 1.0
    dam_x: float = 2.0
    bump_height: float = 0.2
    bump_width: float = 0.25

    def bed(self, x):
        x = np.asarray(x, dtype=float)
        if self.kind == "lake_bump":
            xc = 0.5 * (self.x0 + self.x1)
            return self.bump_height * np.exp(-((x - xc) / self.bump_width) ** 2)
        return np.zeros_like(x)

    def exact(self, x, t):
        """Return ``(h, u)`` at ``(x, t)``."""
        x = np.asarray(x, dtype=float)
        if self.kind in ("lake_flat", "lake_bump"):
            b = self.bed(x)
            return self.h0 - b, np.zeros_like(x)
        if self.kind == "ritter":
            h0, g = self.h0, self.g
            c0 = math.sqrt(g * h0)
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = (x - self.dam_x) / max(t, 1e-12)
            h = np.zeros_like(x)
            u = np.zeros_like(x)
            up = xi <= -c0
            fan = (xi > -c0) & (xi < 2.0 * c0)
            h[up] = h0
            h[fan] = (2.0 * c0 - xi[fan]) ** 2 / (9.0 * g)
            u[fan] = (2.0 / 3.0) * (c0 + xi[fan])
            return h, u
        raise ValueError(f"unknown swe case kind {self.kind!r}")

    def exact_averages(self, n: int):
        """Cell averages ``(h, hu)``."""
        h = cell_averages(lambda x: self.exact(x, self.t_end)[0], self.x0, self.x1, n)

        def hu(x):
            hh, uu = self.exact(x, self.t_end)
            return hh * uu

        return h, cell_averages(hu, self.x0, self.x1, n)

    def initial_state(self, x):
        """Return ``(h, hu)`` at ``t = 0`` (use a tiny time offset for Ritter)."""
        h, u = self.exact(x, 1e-6 if self.kind == "ritter" else 0.0)
        return h, h * u


def lake_at_rest(bump: bool = True, t_end: float = 1.0, h0: float = 1.0) -> SWECase:
    """Still lake; with ``bump`` the bed is a Gaussian and the test is exact
    well-balancedness, without it the bed is flat."""
    return SWECase(name="lake_bump" if bump else "lake_flat",
                   kind="lake_bump" if bump else "lake_flat",
                   x0=0.0, x1=2.0, t_end=t_end, h0=h0)


def ritter_dry_bed(h0: float = 0.5, t_end: float = 0.3) -> SWECase:
    """Ritter dry-bed dam break; the rarefaction must stay inside the domain."""
    return SWECase(name="ritter_dry_bed", kind="ritter", x0=0.0, x1=4.0,
                   t_end=t_end, h0=h0, dam_x=2.0)


# ---------------------------------------------------------------------------
# small-amplitude gravity wave (celerity anchor, SPH and LBM legs)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LinearWaveCase:
    """Long gravity wave in a periodic channel: exact celerity ``sqrt(g h0)``.

    ``eta(x, t) = A cos(k x - omega t)`` with the non-dispersive (shallow-water)
    ``omega = k sqrt(g h0)``.  The case is used with ``A / h0 << 1`` so that the
    nonlinear terms are negligible and the shallow-water equations admit this as
    an exact solution in the long-wave limit.
    """

    h0: float = 1.0
    length: float = 20.0
    amplitude: float = 0.01
    g: float = 9.81
    n_wave: int = 1

    @property
    def k(self) -> float:
        return 2.0 * math.pi * self.n_wave / self.length

    @property
    def c_exact(self) -> float:
        return math.sqrt(self.g * self.h0)

    @property
    def omega(self) -> float:
        return self.k * self.c_exact

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    def depth(self, x, t: float = 0.0):
        x = np.asarray(x, dtype=float)
        return self.h0 + self.amplitude * np.cos(self.k * x - self.omega * t)

    def velocity(self, x, t: float = 0.0):
        x = np.asarray(x, dtype=float)
        # linear shallow-water: u = (g/c) * eta' integrated in time ->
        # for a right-going wave u = (g h0 / c) * eta / h0 = (c) * eta / h0
        return self.c_exact * self.amplitude / self.h0 * np.cos(self.k * x - self.omega * t)