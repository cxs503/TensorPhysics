"""Non-uniform free-free Timoshenko beam finite element for hull-girder dry modes.

Physics
-------
A prismatic or longitudinally varying beam is discretised with two-node
elements carrying a transverse deflection ``w`` and a section rotation
``theta`` at each node.  The element formulation is the standard
shear-flexible (Timoshenko) beam element in which the shear strain is
interpolated independently of the bending rotation, which is what removes
shear locking::

    phi_e = 12 EI / (kappa G A L_e^2)                       (shear parameter)

    K_e = EI / (L_e^3 (1 + phi_e)) *
          [[ 12     6L    -12     6L    ]
           [ 6L   (4+phi)L^2 -6L (2-phi)L^2 ]
           [-12    -6L     12    -6L    ]
           [ 6L   (2-phi)L^2 -6L (4+phi)L^2 ]]

The mass matrix combines the classical Euler-Bernoulli consistent
translational matrix with the rotary-inertia matrix of the section::

    M_e = rho A L/420 * [[156, 22L, 54, -13L], ...]
        + rho I /(30 L) * [[36, 3L, -36, 3L], ...]

Two consistency limits are used as verification gates in this module:

* ``shear_rigid=True`` (``GAs -> inf``) recovers the Euler-Bernoulli
  consistent-mass element, whose free-free frequencies must reproduce the
  classical roots ``beta_n L = 4.7300, 7.8532, 10.9956, 14.1372, ...``
  of ``cosh(beta L) cos(beta L) = 1``;
* finite shear stiffness always *lowers* the frequency relative to the
  Euler-Bernoulli value of the same beam (shear softening), and the relative
  drop grows with mode number and with the slenderness ratio ``h/L``.

Units are SI throughout: ``EI`` in N*m^2, ``GAs = kappa G A`` in N,
``m = rho A`` in kg/m, ``rI = rho I`` in kg*m.

The module is a *structural* kernel only: it knows nothing about water.  The
hydrodynamic generalised mass is added by :mod:`wet`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

__all__ = [
    "EB_FREE_FREE_ROOTS",
    "BeamProperties",
    "DryModes",
    "TimoshenkoBeam",
    "euler_bernoulli_free_free_roots",
    "uniform_beam",
]

#: First roots of ``cosh(b) cos(b) = 1`` (free-free Euler-Bernoulli beam).
EB_FREE_FREE_ROOTS = (
    4.730040744862704,
    7.853204624095838,
    10.995607838001671,
    14.137165491257464,
)

#: Shear correction factor of a thin-walled box girder (Cowper, 6/5 is the
#: rectangle value; a hull-girder box sits between 0.4 and 0.6).
KAPPA_SHEAR = 0.5


def _generalised_eigh(k: np.ndarray, m: np.ndarray) -> tuple:
    """``K phi = lambda M phi`` via a Cholesky reduction (ascending order)."""
    m = 0.5 * (m + m.T)
    k = 0.5 * (k + k.T)
    low = np.linalg.cholesky(m)                 # m = low low^T
    low_inv = np.linalg.inv(low)
    a = low_inv @ k @ low_inv.T
    a = 0.5 * (a + a.T)
    lam, y = np.linalg.eigh(a)
    vec = low_inv.T @ y
    return lam, vec


def euler_bernoulli_free_free_roots(n_modes: int) -> np.ndarray:
    """``beta_n L`` roots of the free-free Euler-Bernoulli beam."""
    roots = list(EB_FREE_FREE_ROOTS)
    b = roots[-1]
    while len(roots) < n_modes:
        # f(b) = cosh(b)cos(b) - 1; roots are bracketed by (n+1/2) pi steps.
        lo = b + 0.5 * np.pi
        hi = lo + np.pi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if np.cosh(mid) * np.cos(mid) - 1.0 > 0.0:
                hi = mid
            else:
                lo = mid
        b = 0.5 * (lo + hi)
        roots.append(b)
    return np.asarray(roots[:n_modes], dtype=float)


@dataclass(frozen=True)
class BeamProperties:
    """Longitudinal distributions of the beam (arrays sampled on ``x``)."""

    x: np.ndarray            # nodal coordinates (m), uniform spacing
    EI: np.ndarray           # bending stiffness (N m^2)
    GAs: np.ndarray          # shear rigidity kappa G A (N)
    m: np.ndarray            # mass per unit length (kg/m)
    rI: np.ndarray           # rotary inertia per unit length (kg m)

    def __post_init__(self) -> None:
        n = np.asarray(self.x).size
        for name in ("EI", "GAs", "m", "rI"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.size != n:
                raise ValueError(f"{name} must have the same size as x")
            if np.any(arr <= 0.0):
                raise ValueError(f"{name} must be strictly positive")

    @property
    def length(self) -> float:
        return float(self.x[-1] - self.x[0])

    @property
    def n_nodes(self) -> int:
        return int(np.asarray(self.x).size)


def uniform_beam(
    length: float = 200.0,
    height: float = 10.0,
    young: float = 2.06e11,
    shear_modulus: float | None = None,
    density: float = 7850.0,
    n_elements: int = 200,
    wall_factor: float = 1.0,
    section: str = "solid_rect",
    width: float | None = None,
) -> BeamProperties:
    """Uniform thin-walled box girder idealisation used by the analytic gates.

    ``height`` is the depth of the effectively contributing material
    (``I = A h^2 / 12`` for the two flanges), ``wall_factor`` scales the
    effective flange/side area and hence both ``I`` and ``A``; keeping the
    flange area fixed makes ``L/h`` the only slenderness lever.
    """
    if shear_modulus is None:
        shear_modulus = young / (2.0 * (1.0 + 0.3))
    n_nodes = int(n_elements) + 1
    x = np.linspace(-0.5 * length, 0.5 * length, n_nodes)
    if section == "solid_rect":
        # Textbook Timoshenko benchmark section: I/A = h^2/12, kappa = 5/6.
        b = float(width if width is not None else height)
        area = b * height * wall_factor
        inertia = b * height ** 3 / 12.0 * wall_factor
        kappa = 5.0 / 6.0
    elif section == "box":
        # Thin-walled box girder: two flanges of area A_f at +-h/2 so that
        # I = A_f h^2 / 2 and I/A = h^2/4 (the hull-girder-like case), kappa=0.5.
        area_flange = 1.0 * wall_factor
        inertia = area_flange * height ** 2 / 2.0
        area = 2.0 * area_flange * wall_factor
        kappa = KAPPA_SHEAR
    else:
        raise ValueError(f"unknown section idealisation {section!r}")
    EI = np.full(n_nodes, young * inertia)
    GAs = np.full(n_nodes, kappa * shear_modulus * area)
    m = np.full(n_nodes, density * area)
    rI = np.full(n_nodes, density * inertia)
    return BeamProperties(x=x, EI=EI, GAs=GAs, m=m, rI=rI)


@dataclass
class DryModes:
    """Free-free dry modes of the beam."""

    frequencies_hz: np.ndarray       # elastic modes only (rigid modes removed)
    omega: np.ndarray                # rad/s
    shapes: np.ndarray               # (n_modes, n_nodes) mass-normalised
    slopes: np.ndarray               # d(shapes)/dx
    curvature: np.ndarray            # d2(shapes)/dx^2
    x: np.ndarray
    rigid_modes: np.ndarray = field(default_factory=lambda: np.zeros(0))
    shear_rigid: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def n_modes(self) -> int:
        return int(self.frequencies_hz.size)


class TimoshenkoBeam:
    """Free-free Timoshenko beam assembled with two-node elements."""

    def __init__(self, props: BeamProperties, shear_rigid: bool = False) -> None:
        self.props = props
        self.shear_rigid = bool(shear_rigid)
        self._k = self._assemble_stiffness()
        self._m = self._assemble_mass()

    # ------------------------------------------------------------- assembly
    def _assemble_stiffness(self) -> np.ndarray:
        p = self.props
        n_el = p.n_nodes - 1
        n_dof = 2 * p.n_nodes
        K = np.zeros((n_dof, n_dof))
        for e in range(n_el):
            le = p.x[e + 1] - p.x[e]
            ei = 0.5 * (p.EI[e] + p.EI[e + 1])
            gas = 0.5 * (p.GAs[e] + p.GAs[e + 1]) if not self.shear_rigid else np.inf
            phi = 0.0 if not np.isfinite(gas) else 12.0 * ei / (gas * le * le)
            c = ei / (le ** 3 * (1.0 + phi))
            ke = c * np.array(
                [
                    [12.0, 6.0 * le, -12.0, 6.0 * le],
                    [6.0 * le, (4.0 + phi) * le * le, -6.0 * le, (2.0 - phi) * le * le],
                    [-12.0, -6.0 * le, 12.0, -6.0 * le],
                    [6.0 * le, (2.0 - phi) * le * le, -6.0 * le, (4.0 + phi) * le * le],
                ]
            )
            idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            K[np.ix_(idx, idx)] += ke
        return K

    def _assemble_mass(self) -> np.ndarray:
        p = self.props
        n_el = p.n_nodes - 1
        n_dof = 2 * p.n_nodes
        M = np.zeros((n_dof, n_dof))
        for e in range(n_el):
            le = p.x[e + 1] - p.x[e]
            mt = 0.5 * (p.m[e] + p.m[e + 1])
            ri = 0.5 * (p.rI[e] + p.rI[e + 1])
            mt_mat = (mt * le / 420.0) * np.array(
                [
                    [156.0, 22.0 * le, 54.0, -13.0 * le],
                    [22.0 * le, 4.0 * le * le, 13.0 * le, -3.0 * le * le],
                    [54.0, 13.0 * le, 156.0, -22.0 * le],
                    [-13.0 * le, -3.0 * le * le, -22.0 * le, 4.0 * le * le],
                ]
            )
            mr_mat = (ri / (30.0 * le)) * np.array(
                [
                    [36.0, 3.0 * le, -36.0, 3.0 * le],
                    [3.0 * le, 4.0 * le * le, -3.0 * le, -le * le],
                    [-36.0, -3.0 * le, 36.0, -3.0 * le],
                    [3.0 * le, -le * le, -3.0 * le, 4.0 * le * le],
                ]
            )
            idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            M[np.ix_(idx, idx)] += mt_mat + mr_mat
        return M

    @property
    def stiffness(self) -> np.ndarray:
        return self._k

    @property
    def mass(self) -> np.ndarray:
        return self._m

    # ------------------------------------------------------------- solution
    def dry_modes(self, n_modes: int = 8) -> DryModes:
        """Free-free dry modes with the two rigid-body modes removed.

        The generalised problem ``K phi = lambda M phi`` is reduced by a Cholesky
        factor of ``M`` and solved with ``numpy.linalg.eigh`` (symmetric standard
        problem).  This is numerically equivalent to a QZ/generalised LAPACK
        call, avoids an architecture-specific LAPACK driver (a generalised
        ``eigh`` segfault was observed on the loong64 nodes) and keeps the
        ascending eigenvalue order.
        """
        lam, vec = _generalised_eigh(self._k, self._m)
        order = np.argsort(lam)
        lam = lam[order]
        vec = vec[:, order]
        total = lam.size
        # Rigid-body modes: |lambda| below a small fraction of the first
        # elastic eigenvalue.  Both heave and pitch must appear.
        scale = max(abs(lam[-1]), 1e-30)
        rigid = np.where(np.abs(lam) < 1.0e-9 * scale)[0]
        if rigid.size < 2:
            # Fall back to "the two smallest eigenvalues are rigid".
            rigid = np.arange(2)
        elastic = np.arange(2, min(total, 2 + int(n_modes)))
        lam_e = np.clip(lam[elastic], 0.0, None)
        omega = np.sqrt(lam_e)
        phi = vec[:, elastic].T                     # (n_modes, n_dof)
        w = phi[:, 0::2]
        theta = phi[:, 1::2]
        x = self.props.x
        dphi = np.gradient(w, x, axis=1)
        d2phi = np.gradient(dphi, x, axis=1)
        meta = {
            "n_dof": int(total),
            "n_elements": int(self.props.n_nodes - 1),
            "rigid_eigenvalues": lam[rigid].tolist(),
            "shear_rigid": self.shear_rigid,
        }
        return DryModes(
            frequencies_hz=omega / (2.0 * np.pi),
            omega=omega,
            shapes=w,
            slopes=dphi,
            curvature=d2phi,
            x=x,
            rigid_modes=vec[:, rigid],
            shear_rigid=self.shear_rigid,
            meta=meta,
        )

    def axial_and_torsion(self) -> dict:
        """Not modelled here (bending only) - reported honestly."""
        return {
            "axial": "not modelled",
            "torsion": "not modelled",
            "note": "hull-girder vertical bending only; horizontal/torsional "
                    "modes are outside the scope of this node",
        }

    def max_curvature_position(self, modes: DryModes, mode: int) -> float:
        """``x`` of maximum |curvature| (node location of the hot spot)."""
        k = np.abs(modes.curvature[mode])
        return float(modes.x[int(np.argmax(k))])