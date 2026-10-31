"""Core kernels for ice-loaded hull plate buckling analysis.

This module contains the three ingredients that the nonlinear plate solver
builds on:

* :class:`PlateGeometry` -- rectangular plate geometry with machine-checkable
  validation (aspect ratio ``a/b``, plate slenderness
  ``beta = (b/t) * sqrt(sigma_y/E)``, mesh aspect ratio).  No analysis is
  allowed to run on a geometry that fails validation, mirroring the
  "verify geometry before solving" discipline used for whole-ship work.
* :class:`BilinearMaterial` + :func:`plane_stress_return_map` -- plane-stress
  von Mises elastoplasticity with bilinear isotropic hardening, vectorised so
  that all (element, Gauss point, through-thickness layer) points are updated
  in one call.
* :func:`element_kernels` -- Gauss-point kernels of a flat rectangular shell
  finite element made of a bilinear ``Q4`` membrane and a conforming bicubic
  Hermite plate-bending rectangle (16 dofs).  The kernels also expose
  ``dw/dx`` and ``dw/dy`` at the same Gauss points so the von Karman
  large-deflection strain terms can be formed.

Conventions
-----------
* Plate plane is ``x`` (length ``a``, longitudinal) - ``y`` (breadth ``b``).
* Six dofs per node: ``u``, ``v``, ``w``, ``theta_x = dw/dy``,
  ``theta_y = -dw/dx``, ``w_xy`` (Hermite twist dof).
* Curvatures are ``kappa = [-w,xx, -w,yy, -2 w,xy]`` so that the strain at a
  through-thickness coordinate ``z`` is ``eps(z) = eps_m + z * kappa`` with
  engineering shear components.
* Compression is negative; membrane stress resultants are per unit width.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

#: 4x4 Gauss points in natural coordinates.  The bicubic Hermite bending
#: element has curvature fields that are cubic along the orthogonal direction,
#: so the strain-energy integrand is of degree 6 per direction; 4x4 Gauss
#: integration is required (a 2x2 rule under-integrates and produces spurious
#: zero-energy modes).
_gx, _gw = np.polynomial.legendre.leggauss(4)
GAUSS_POINTS = np.array([[_gx[i], _gx[j]] for j in range(4) for i in range(4)])
GAUSS_WEIGHTS_1D = _gw


class GeometryError(ValueError):
    """Raised when a plate geometry fails validation."""


@dataclass(frozen=True)
class PlateGeometry:
    """Rectangular unstiffened plate (plate panel between stiffeners).

    Parameters
    ----------
    a, b, t:
        Length (x), breadth (y) and thickness in metres.
    E, nu, sigma_y, E_t:
        Elastic modulus, Poisson ratio, yield stress and *hardening modulus*
        (``E_t`` is the slope of the bilinear uniaxial stress-strain curve;
        ``E_t = 0`` gives elastic-perfectly-plastic behaviour).
    nx, ny:
        Number of elements along x and y.
    """

    a: float
    b: float
    t: float
    E: float = 2.06e11
    nu: float = 0.3
    sigma_y: float = 3.13e8
    E_t: float = 0.0
    nx: int = 24
    ny: int = 8

    # ------------------------------------------------------------------ props
    @property
    def aspect(self) -> float:
        """Aspect ratio ``a/b``."""
        return self.a / self.b

    @property
    def beta(self) -> float:
        """Plate slenderness ``beta = (b/t) sqrt(sigma_y/E)``."""
        return (self.b / self.t) * np.sqrt(self.sigma_y / self.E)

    @property
    def flexural_rigidity(self) -> float:
        """Plate flexural rigidity ``D = E t^3 / (12 (1 - nu^2))``."""
        return self.E * self.t**3 / (12.0 * (1.0 - self.nu**2))

    @property
    def element_aspect(self) -> float:
        """Largest element aspect ratio (mesh quality gate)."""
        ex, ey = self.a / self.nx, self.b / self.ny
        return max(ex, ey) / min(ex, ey)

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def n_dof(self) -> int:
        return 6 * self.n_nodes

    @property
    def n_elem(self) -> int:
        return self.nx * self.ny

    # ------------------------------------------------------------- validation
    def validate(self, aspect_range: Tuple[float, float] = (2.0, 6.0),
                 beta_range: Tuple[float, float] = (1.0, 3.0),
                 max_elem_aspect: float = 3.0) -> Dict[str, float]:
        """Validate geometry against the design envelope; raise on failure.

        Returns a dictionary of the checked quantities so callers can persist
        them as provenance (numbers are never re-typed by hand in reports).
        """
        checks = {
            "a_over_b": float(self.aspect),
            "beta": float(self.beta),
            "element_aspect": float(self.element_aspect),
        }
        if not (self.a > 0 and self.b > 0 and self.t > 0):
            raise GeometryError("a, b, t must be positive")
        lo, hi = aspect_range
        if not (lo - 1e-12 <= self.aspect <= hi + 1e-12):
            raise GeometryError(
                f"aspect ratio a/b={self.aspect:.3f} outside [{lo}, {hi}]")
        lo, hi = beta_range
        if not (lo - 1e-12 <= self.beta <= hi + 1e-12):
            raise GeometryError(
                f"plate slenderness beta={self.beta:.3f} outside [{lo}, {hi}]")
        if self.element_aspect > max_elem_aspect:
            raise GeometryError(
                f"element aspect ratio {self.element_aspect:.2f} > "
                f"{max_elem_aspect}: mesh is not close to square")
        return checks


@dataclass(frozen=True)
class BilinearMaterial:
    """Isotropic-hardening bilinear von Mises material (plane stress)."""

    E: float
    nu: float = 0.3
    sigma_y: float = 3.13e8
    E_t: float = 0.0

    @property
    def shear_modulus(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def plastic_modulus(self) -> float:
        """Uniaxial plastic (hardening) modulus ``H = E E_t / (E - E_t)``."""
        if self.E_t <= 0.0:
            return 0.0
        return self.E * self.E_t / (self.E - self.E_t)

    @property
    def elastic_matrix(self) -> np.ndarray:
        c = self.E / (1.0 - self.nu**2)
        return c * np.array(
            [[1.0, self.nu, 0.0],
             [self.nu, 1.0, 0.0],
             [0.0, 0.0, 0.5 * (1.0 - self.nu)]]
        )

    @classmethod
    def from_geometry(cls, geom: PlateGeometry) -> "BilinearMaterial":
        return cls(E=geom.E, nu=geom.nu, sigma_y=geom.sigma_y, E_t=geom.E_t)


# ---------------------------------------------------------------------------
# Plane-stress elastoplastic return mapping
# ---------------------------------------------------------------------------
def _deviator(lam: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deviatoric components of a plane-stress state (sigma_zz = 0)."""
    mean = (lam[..., 0] + lam[..., 1]) / 3.0
    return lam[..., 0] - mean, lam[..., 1] - mean, lam[..., 2]


def _von_mises(lam: np.ndarray) -> np.ndarray:
    """von Mises equivalent stress of a plane-stress state."""
    sxx, syy, sxy = _deviator(lam)
    szz = -(lam[..., 0] + lam[..., 1]) / 3.0
    return np.sqrt(np.maximum(
        1.5 * (sxx**2 + syy**2 + szz**2 + 2.0 * sxy**2), 0.0))


def plane_stress_return_map(
    strain: np.ndarray,
    eps_p: np.ndarray,
    alpha: np.ndarray,
    material: BilinearMaterial,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised plane-stress J2 return mapping with isotropic hardening.

    In plane stress the elastic matrix does not map the flow direction onto
    itself, so the linear-hardening return is *not* a one-step closed form; the
    consistency condition ``q(dg) = sigma_y + H (alpha + dg)`` is solved by a
    Newton iteration on the plastic multiplier ``dg`` (with a Picard update of
    the flow direction), using the analytic derivative
    ``dq/ddg = -n^T C n`` (engineering shear counted twice).

    Parameters
    ----------
    strain, eps_p, alpha:
        Total strain, committed plastic strain (``(..., 3)`` with engineering
        shear) and committed equivalent plastic strain.
    material:
        :class:`BilinearMaterial`.

    Returns
    -------
    sigma, C_ep, eps_p_new, alpha_new
        Consistent tangent ``C_ep = C - (C n)(C n)^T / (H + n^T C n)``.
    """
    strain = np.asarray(strain, dtype=float)
    eps_p = np.asarray(eps_p, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    C = material.elastic_matrix
    G = material.shear_modulus
    H = material.plastic_modulus

    sig_tr = (strain - eps_p) @ C.T
    q_tr = _von_mises(sig_tr)
    fy = material.sigma_y + H * alpha
    plastic = q_tr > fy * (1.0 + 1e-12)
    if not np.any(plastic):
        return sig_tr, np.broadcast_to(C, sig_tr.shape + (3,)).copy(),             eps_p.copy(), alpha.copy()

    q_safe = np.maximum(q_tr, 1e-30)
    n = 1.5 * np.stack(_deviator(sig_tr), axis=-1) / q_safe[..., None]
    # Radial return with the flow direction frozen at the elastic trial
    # deviator.  q(dg) is then strictly decreasing, so a bracketed bisection
    # converges unconditionally and enforces the consistency condition
    # q = sigma_y + H (alpha + dg) by construction (a Newton iteration on the
    # multiplier can overshoot in plane stress, where C n is not collinear
    # with n).
    Cn = n @ C.T
    # Safe upper bracket: the deviatoric stress vanishes at
    # dg* = q_tr / alpha_q with alpha_q = n^T C n >= 2G (1 - nu) in plane stress,
    # so the consistency root always lies below q_tr / (2G (1 - nu)).
    lo = np.zeros_like(q_tr)
    hi = np.where(plastic,
                  np.maximum(q_tr / (2.0 * G * (1.0 - material.nu)), 1e-12), 0.0)
    for _ in range(80):
        dg = 0.5 * (lo + hi)
        q_m = _von_mises(sig_tr - dg[..., None] * Cn)
        high = plastic & (q_m > material.sigma_y + H * (alpha + dg))
        lo = np.where(high, dg, lo)
        hi = np.where(plastic & ~high, dg, hi)
    dg = 0.5 * (lo + hi)
    dg = np.where(plastic, dg, 0.0)

    Cn = n @ C.T
    sigma = sig_tr - dg[..., None] * Cn
    eps_p_new = eps_p + dg[..., None] * n
    alpha_new = alpha + dg

    denom = H + (n[..., 0] * Cn[..., 0] + n[..., 1] * Cn[..., 1]
                 + 2.0 * n[..., 2] * Cn[..., 2])
    C_ep = np.broadcast_to(C, sig_tr.shape + (3,)).copy()
    C_ep[plastic] = (C[None, :, :]
                     - np.einsum("...i,...j->...ij", Cn, Cn)[plastic]
                     / denom[plastic][:, None, None])
    return sigma, C_ep, eps_p_new, alpha_new


# ---------------------------------------------------------------------------
# Element kernels (Q4 membrane + ACM bending, identical for every element of a
# uniform rectangular mesh, hence precomputed once and shared).
# ---------------------------------------------------------------------------
def _hermite_1d(t: np.ndarray):
    """Cubic Hermite basis on t in [0, 1] (value/slope at both ends)."""
    h = np.array([1 - 3 * t**2 + 2 * t**3,
                  t - 2 * t**2 + t**3,
                  3 * t**2 - 2 * t**3,
                  -t**2 + t**3])
    dh = np.array([-6 * t + 6 * t**2,
                   1 - 4 * t + 3 * t**2,
                   6 * t - 6 * t**2,
                   -2 * t + 3 * t**2])
    ddh = np.array([-6 + 12 * t, -4 + 6 * t, 6 - 12 * t, -2 + 6 * t])
    return h, dh, ddh


def _hermite_to_nodal(ex: float, ey: float) -> np.ndarray:
    """Map the 16 nodal bending dofs to the 16 Hermite generalised dofs.

    Nodal dof order per node: ``[w, theta_x, theta_y, w_xy]`` with
    ``theta_x = dw/dy`` and ``theta_y = -dw/dx``; node order
    ``(0,0), (ex,0), (ex,ey), (0,ey)``.  Hermite generalised dofs are ordered
    ``A[i, j]`` with ``i, j`` in ``{0, 1, 2, 3}`` (value / t-slope at t=0,
    value / t-slope at t=1) and ``w_t = ex w,x``, ``w_s = ey w,y``,
    ``w_ts = ex ey w,xy``.
    """
    M = np.zeros((16, 16))
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for c, (t, s) in enumerate(corners):
        it, js = int(round(2 * t)), int(round(2 * s))
        base = 4 * c
        M[it * 4 + js, base + 0] = 1.0            # w
        M[(it + 1) * 4 + js, base + 2] = -ex      # w_t = ex w,x = -ex theta_y
        M[it * 4 + (js + 1), base + 1] = ey       # w_s = ey w,y = ey theta_x
        M[(it + 1) * 4 + (js + 1), base + 3] = ex * ey   # w_ts
    return M


def element_kernels(ex: float, ey: float) -> Dict[str, np.ndarray]:
    """Gauss-point kernels of the rectangular flat shell element.

    Membrane: bilinear Q4.  Bending: conforming bicubic Hermite rectangle
    (16 dofs: ``w, theta_x, theta_y, w_xy`` per node).

    Returns a dict of arrays indexed ``[gauss_point, ...]``:

    ``Bm`` (16, 3, 8)  membrane strain-displacement matrix
    ``Bb`` (16, 3, 16) curvature-displacement matrix (kappa = [-w,xx,-w,yy,-2w,xy])
    ``Nwx`` (16, 16)   dw/dx shape derivative row
    ``Nwy`` (16, 16)   dw/dy shape derivative row
    ``Nw`` (16, 16)    w interpolation row (for output / imperfection checks)
    ``weight`` (16,)   Gauss weight x Jacobian determinant
    """
    M = _hermite_to_nodal(ex, ey)

    n_g = GAUSS_POINTS.shape[0]
    Bm = np.zeros((n_g, 3, 8))
    Bb = np.zeros((n_g, 3, 16))
    Nwx = np.zeros((n_g, 16))
    Nwy = np.zeros((n_g, 16))
    Nw = np.zeros((n_g, 16))
    weight = np.zeros(n_g)
    det_j = 0.25 * ex * ey

    for g, (xi, eta) in enumerate(GAUSS_POINTS):
        t = 0.5 * (1.0 + xi)
        s = 0.5 * (1.0 + eta)
        # ---- membrane Q4
        dndxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dndeta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        dndx = dndxi * (2.0 / ex)
        dndy = dndeta * (2.0 / ey)
        for i in range(4):
            Bm[g, 0, 2 * i] = dndx[i]
            Bm[g, 1, 2 * i + 1] = dndy[i]
            Bm[g, 2, 2 * i] = dndy[i]
            Bm[g, 2, 2 * i + 1] = dndx[i]
        # ---- bending bicubic Hermite
        h, dh, ddh = _hermite_1d(t)
        hs, dhs, ddhs = _hermite_1d(s)
        row_xx = np.array([[(1.0 / ex**2) * ddh[i] * hs[j] for j in range(4)]
                           for i in range(4)]).ravel()
        row_yy = np.array([[(1.0 / ey**2) * h[i] * ddhs[j] for j in range(4)]
                           for i in range(4)]).ravel()
        row_xy = np.array([[(1.0 / (ex * ey)) * dh[i] * dhs[j] for j in range(4)]
                           for i in range(4)]).ravel()
        row_x = np.array([[(1.0 / ex) * dh[i] * hs[j] for j in range(4)]
                          for i in range(4)]).ravel()
        row_y = np.array([[(1.0 / ey) * h[i] * dhs[j] for j in range(4)]
                          for i in range(4)]).ravel()
        row_w = np.array([[(h[i] * hs[j]) for j in range(4)]
                          for i in range(4)]).ravel()
        Bb[g, 0, :] = -(row_xx @ M)
        Bb[g, 1, :] = -(row_yy @ M)
        Bb[g, 2, :] = -2.0 * (row_xy @ M)
        Nwx[g] = row_x @ M
        Nwy[g] = row_y @ M
        Nw[g] = row_w @ M
        gi = g % 4
        gj = g // 4
        weight[g] = det_j * GAUSS_WEIGHTS_1D[gi] * GAUSS_WEIGHTS_1D[gj]

    return {"Bm": Bm, "Bb": Bb, "Nwx": Nwx, "Nwy": Nwy, "Nw": Nw,
            "weight": weight}


def consistent_pressure_load(ex: float, ey: float, q: float) -> np.ndarray:
    """Consistent nodal load vector (16 values) of a uniform pressure ``q``.

    For the bicubic Hermite rectangle the consistent load includes nodal
    moments (``q ex ey^2 / 24`` type terms) in addition to the corner forces
    ``q ex ey / 4``; omitting the moment terms makes the element
    inconsistent and systematically too flexible.
    """
    M = _hermite_to_nodal(ex, ey)
    wts = np.array([0.5, 1.0 / 12.0, 0.5, -1.0 / 12.0])
    L_A = q * ex * ey * np.array([[wts[i] * wts[j] for j in range(4)]
                                  for i in range(4)]).ravel()
    return M.T @ L_A


def navier_simply_supported_deflection(geom: PlateGeometry, q: float,
                                       n_terms: int = 41) -> float:
    """Analytic Navier solution: centre deflection of an SSSS plate under
    uniform pressure ``q`` (independent reference for the bending element)."""
    D = geom.flexural_rigidity
    a, b = geom.a, geom.b
    w = 0.0
    for m in range(1, n_terms + 1, 2):
        for n in range(1, n_terms + 1, 2):
            w += (16.0 * q / (np.pi**6 * D)) * (
                1.0 / (m * n * (m * m / a**2 + n * n / b**2) ** 2)
            ) * np.sin(m * np.pi / 2.0) * np.sin(n * np.pi / 2.0)
    return float(w)