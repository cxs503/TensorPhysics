"""Two-dimensional strip hydrodynamics for the hull-girder wet modes.

What is computed, and with what physics
---------------------------------------
The generalised hydrodynamic mass of a hull-girder bending mode is built from
the *sectional* (two-dimensional) added mass ``a33(x)``.  Hull-girder vertical
bending frequencies are high (0.5-2 Hz on a 200-300 m hull, i.e. encounter
periods of 0.5-2 s), so the appropriate strip coefficient is the
**infinite-frequency limit**: the free-surface condition
``-omega^2 phi + g dphi/dz = 0`` degenerates to ``phi = 0`` on ``z = 0`` as
``omega -> inf`` and no wave is radiated.  This is the coefficient used by
springing/whipping practice, and it is the one piece of the radiation problem
that can be obtained from a *Dirichlet* boundary-value problem (no
free-surface Green function required).

Numerics
--------
Constant-strength 2-D source panels on the wetted contour with collocation at
the panel midpoints:

* ``phi(x) = (1/2 pi) Sum_j sigma_j Int_panel_j ln|x - x'| ds'`` (exact panel
  integral of the logarithmic kernel, so the self-panel singularity is
  integrated analytically instead of being sampled);
* the Neumann condition ``dphi/dn = v_n`` gives the Fredholm equation of the
  first kind for ``sigma``; its kernel is the *exact* normal derivative of the
  log kernel, approximated by midpoint collocation, and the diagonal carries
  the jump term ``+1/2`` of the single layer;
* the ``phi = 0`` free-surface condition is imposed by the method of images
  (image sources of opposite sign at ``-z``), so only the wetted contour is
  discretised;
* the added-mass matrix follows from the fluid kinetic energy,
  ``m_jk = -rho Int phi_k n_j ds`` (the minus sign is the outward-from-fluid
  normal convention).

Verification (:mod:`tests.test_hull_hydroelastic`): the heave added mass of a
deeply submerged circular cylinder must be ``rho pi a^2`` per unit length and
that of a deeply submerged flat plate of chord ``2a`` moved normal to itself
must also be ``rho pi a^2``; both are exact 2-D potential-flow results.  A
floating section gains added mass relative to the submerged case because
``phi = 0`` on the free surface reduces the flow domain asymmetry.

Evidence level: the added mass is REAL (boundary-value problem solved and
checked against closed forms).  Section *damping* is **not** solved here - it
requires the radiating free-surface Green function, which this node does not
implement; :func:`column_damping_estimate` documents the order-of-magnitude
engineering estimate that is used instead and :mod:`wet` states the
consequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Section2D",
    "circle_section",
    "superellipse_section",
    "panelize",
    "added_mass_matrix",
    "heave_added_mass",
    "column_damping_estimate",
    "RHO_WATER",
]

RHO_WATER = 1025.0  # kg/m^3
GRAVITY = 9.81


@dataclass
class Section2D:
    """Wetted contour of a transverse section (closed by the waterline)."""

    y: np.ndarray  # horizontal coordinate (m), 0 = centreline
    z: np.ndarray  # vertical coordinate (m), 0 = waterline, negative down
    name: str = "section"
    #: floating sections end on the waterline and must NOT be closed by a lid:
    #: the waterline is free surface (phi = 0), not wetted body surface.
    open_ends: bool = False

    @property
    def half_beam(self) -> float:
        return float(np.max(np.abs(self.y)))

    @property
    def draught(self) -> float:
        return float(-np.min(self.z))

    @property
    def area(self) -> float:
        y = np.asarray(self.y, dtype=float)
        z = np.asarray(self.z, dtype=float)
        if not self.open_ends:
            y = np.append(y, y[0])
            z = np.append(z, z[0])
        return float(abs(0.5 * np.sum(y[:-1] * z[1:] - y[1:] * z[:-1])))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "half_beam_m": self.half_beam,
            "draught_m": self.draught,
            "n_points": int(np.asarray(self.y).size),
        }


def _ordered(y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if y.size > 1 and abs(y[0] - y[-1]) < 1e-12 and abs(z[0] - z[-1]) < 1e-12:
        y, z = y[:-1], z[:-1]
    return y, z


def circle_section(radius: float = 1.0, centre_z: float = -10.0, n: int = 240) -> Section2D:
    """Full circular cylinder (used for the ``rho pi a^2`` analytic gate)."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return Section2D(y=radius * np.cos(th), z=centre_z + radius * np.sin(th), name="circle")


def superellipse_section(
    half_beam: float, draught: float, exponent: float = 4.0, n: int = 160
) -> Section2D:
    """Lewis-like section: superellipse from the keel to the waterline.

    ``(|y|/b)^n + (|z|/T)^n = 1`` with ``b`` the waterline half beam and ``T``
    the draught.  ``n = 2`` is an ellipse, ``n -> inf`` rectangular, ``n = 4..8``
    covers tanker/bulker midbody fullness.  The waterplane width is exactly
    ``2b`` so the strip idealisation stays consistent with the hull-girder
    waterplane distribution used for the wave loads.
    """
    if half_beam <= 0.0 or draught <= 0.0:
        raise ValueError("half beam and draught must be positive")
    if exponent < 2.0:
        raise ValueError("exponent must be >= 2")
    phi = np.linspace(0.0, 0.5 * np.pi, max(8, n // 2))
    y_half = half_beam * np.abs(np.sin(phi)) ** (2.0 / exponent)
    z_half = -draught * np.abs(np.cos(phi)) ** (2.0 / exponent)
    y = np.concatenate([y_half, -y_half[::-1]])
    z = np.concatenate([z_half, z_half[::-1]])
    return Section2D(y=y, z=z, name="superellipse", open_ends=True)


@dataclass
class Panels:
    """Constant-strength panels of a section contour."""

    my: np.ndarray      # panel midpoints
    mz: np.ndarray
    y0: np.ndarray      # panel start points
    z0: np.ndarray
    y1: np.ndarray      # panel end points
    z1: np.ndarray
    length: np.ndarray
    ty: np.ndarray      # unit tangent
    tz: np.ndarray
    ny: np.ndarray      # outward unit normal (body -> fluid)
    nz: np.ndarray

    @property
    def n(self) -> int:
        return int(self.my.size)


def panelize(section: Section2D, n_panels: int | None = None) -> Panels:
    """Resample the contour by arc length into ``n_panels`` constant panels."""
    y, z = _ordered(section.y, section.z)
    if y.size < 3:
        raise ValueError("a section needs at least three points")
    if section.open_ends:
        yc, zc = y, z
    else:
        yc = np.append(y, y[0])
        zc = np.append(z, z[0])
    seg = np.hypot(np.diff(yc), np.diff(zc))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    n_panels = int(n_panels or max(60, y.size))
    edges = np.linspace(0.0, total, n_panels + 1)
    ey = np.interp(edges, s, yc)
    ez = np.interp(edges, s, zc)
    y0, z0, y1, z1 = ey[:-1], ez[:-1], ey[1:], ez[1:]
    my = 0.5 * (y0 + y1)
    mz = 0.5 * (z0 + z1)
    ty = y1 - y0
    tz = z1 - z0
    length = np.hypot(ty, tz)
    ty = ty / length
    tz = tz / length
    ny = tz
    nz = -ty
    cy, cz = float(np.mean(my)), float(np.mean(mz))
    flip = (my - cy) * ny + (mz - cz) * nz < 0.0
    ny = np.where(flip, -ny, ny)
    nz = np.where(flip, -nz, nz)
    return Panels(my, mz, y0, z0, y1, z1, length, ty, tz, ny, nz)


def _log_panel_integral(fy: np.ndarray, fz: np.ndarray, panels: Panels) -> np.ndarray:
    """``G[i, j] = Int_{panel j} ln|x_i - x'(s)| ds`` (exact, per panel).

    Straight-panel analytic integral; the logarithmic singularity is integrable
    and the closed form is used for ``h -> 0`` as well.
    """
    # local panel coordinates of the field points
    dy = fy[:, None] - panels.my[None, :]
    dz = fz[:, None] - panels.mz[None, :]
    s0 = dy * panels.ty[None, :] + dz * panels.tz[None, :]
    h = np.abs(-dy * panels.tz[None, :] + dz * panels.ty[None, :])
    half = 0.5 * panels.length[None, :]
    u1 = -half - s0
    u2 = half - s0
    hh = np.maximum(h, 1.0e-14)

    def prim(u: np.ndarray, h_: np.ndarray) -> np.ndarray:
        r2 = u * u + h_ * h_
        return 0.5 * (u * np.log(r2) - 2.0 * u + 2.0 * h_ * np.arctan(u / h_))

    integral = prim(u2, hh) - prim(u1, hh)
    # exact collinear limit h = 0: Int ln|u| du = u ln|u| - u
    collinear = h < 1.0e-13
    if np.any(collinear):
        u1c = np.where(collinear, u1, 1.0)
        u2c = np.where(collinear, u2, 1.0)
        lim = np.where(
            collinear,
            u2c * np.log(np.maximum(np.abs(u2c), 1e-30))
            - u2c
            - (u1c * np.log(np.maximum(np.abs(u1c), 1e-30)) - u1c),
            0.0,
        )
        integral = np.where(collinear, lim, integral)
    return integral


def _normal_derivative_matrix(panels: Panels, free_surface: bool) -> np.ndarray:
    """``A[i, j] = dphi_i/dn`` per unit ``sigma_j`` (jump term included).

    Exact constant-panel influence coefficient: for a straight panel the
    normal derivative of the log kernel integrates to
    ``[arctan(u/h)]`` with ``u`` the tangential coordinate along the panel and
    ``h`` the signed normal distance from the collocation point to the panel
    line, so no midpoint approximation is involved.
    """
    dy = panels.my[:, None] - panels.my[None, :]
    dz = panels.mz[:, None] - panels.mz[None, :]
    s0 = dy * panels.ty[None, :] + dz * panels.tz[None, :]
    h = dz * panels.nz[None, :] + dy * panels.ny[None, :]
    half = 0.5 * panels.length[None, :]
    hh = np.where(np.abs(h) < 1.0e-12, 1.0e-12, h)
    A = (np.arctan((-half - s0) / hh) - np.arctan((half - s0) / hh)) / (2.0 * np.pi)
    A = np.where(np.abs(h) < 1.0e-12, 0.0, A)
    np.fill_diagonal(A, 0.5)  # jump of the single layer
    if free_surface:
        # image sources of opposite sign at -z give phi = 0 on z = 0
        fy = panels.my
        fz = -panels.mz
        dy_i = fy[:, None] - panels.my[None, :]
        dz_i = fz[:, None] - panels.mz[None, :]
        s0_i = dy_i * panels.ty[None, :] + dz_i * panels.tz[None, :]
        h_i = dz_i * panels.nz[None, :] + dy_i * panels.ny[None, :]
        hh_i = np.where(np.abs(h_i) < 1.0e-12, 1.0e-12, h_i)
        Ai = (np.arctan((-half - s0_i) / hh_i) - np.arctan((half - s0_i) / hh_i)) / (2.0 * np.pi)
        A = A - np.where(np.abs(h_i) < 1.0e-12, 0.0, Ai)
    return A


def _mode_velocity(mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "heave":
        return np.array([0.0, 1.0]), "heave"
    if mode == "sway":
        return np.array([1.0, 0.0]), "sway"
    if mode == "roll":
        return np.array([np.nan, np.nan]), "roll"
    raise ValueError(f"unknown mode {mode!r}")


def _normal_velocity(mode: str, panels: Panels) -> np.ndarray:
    if mode == "heave":
        return panels.nz
    if mode == "sway":
        return panels.ny
    if mode == "roll":
        # in-plane rotation about the origin with unit angular velocity
        return -panels.mz * panels.ny + panels.my * panels.nz
    raise ValueError(f"unknown mode {mode!r}")


def added_mass_matrix(
    section: Section2D,
    rho: float = RHO_WATER,
    modes: tuple[str, ...] = ("heave", "sway", "roll"),
    n_panels: int | None = None,
    free_surface: bool = False,
) -> dict:
    """Sectional added-mass matrix ``a_jk`` per unit length of hull.

    ``free_surface=True`` imposes ``phi = 0`` on ``z = 0`` (infinite-frequency
    condition for a floating section); ``False`` is an unbounded fluid, which is
    the configuration of the analytic checks.
    """
    panels = panelize(section, n_panels)
    A = _normal_derivative_matrix(panels, free_surface)
    G = _log_panel_integral(panels.my, panels.mz, panels)
    if free_surface:
        G = G - _log_panel_integral(panels.my, -panels.mz, panels)

    potentials = {}
    for mode in modes:
        sigma = np.linalg.solve(A, _normal_velocity(mode, panels))
        potentials[mode] = (sigma @ G.T) / (2.0 * np.pi)

    coeff = np.zeros((len(modes), len(modes)))
    for j, mj in enumerate(modes):
        vn_j = _normal_velocity(mj, panels) * panels.length
        for k, mk in enumerate(modes):
            coeff[j, k] = -rho * float(vn_j @ potentials[mk])
    coeff = 0.5 * (coeff + coeff.T)  # reciprocity
    return {
        "modes": list(modes),
        "matrix": coeff,
        "free_surface": bool(free_surface),
        "n_panels": panels.n,
        "section": section.as_dict(),
    }


def heave_added_mass(
    section: Section2D,
    rho: float = RHO_WATER,
    n_panels: int | None = None,
    free_surface: bool = False,
) -> float:
    return float(added_mass_matrix(section, rho, ("heave",), n_panels, free_surface)["matrix"][0, 0])


def column_damping_estimate(draught: float, omega: float, rho: float = RHO_WATER) -> float:
    """Order-of-magnitude 2-D radiation damping of a heaving section (kg/(m s)).

    .. warning::
       **Engineering estimate, not a solved radiation problem.**  It is the
       long-wave wave-making limit of a heaving section: the displaced-volume
       flux ``q = B_eq * w_dot`` radiates two deep-water waves whose total
       energy flux balances ``b33 w_dot^2 / 2``, giving

           b33 = rho g^2 B_eq^2 / (2 omega^3),

       with ``B_eq = 2 A_section / T`` (twice the mean wetted width) as the
       equivalent piston width of a Lewis-type section.  Used only to size the
       (small) hydrodynamic share of the modal damping; the structural share is
       an explicit input.  Evidence grade: APPROXIMATE.
    """
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    b_eq = 2.0 * draught
    return float(rho * GRAVITY ** 2 * b_eq ** 2 / (2.0 * omega ** 3))