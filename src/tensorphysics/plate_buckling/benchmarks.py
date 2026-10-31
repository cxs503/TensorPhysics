"""Reference solutions used to verify the nonlinear plate-buckling solver.

Three independent references are provided:

``ritz_buckling_stress``
    Rayleigh-Ritz solution of the elastic plate buckling problem using a
    double series of beam characteristic functions (sine for simply supported
    edges, clamped-clamped beam functions for clamped edges).  This is an
    independent method (energy minimisation, no finite elements) and is used as
    the reference for the linear-buckling gate for the four boundary-condition
    combinations.

``euler_buckling_stress``
    Classical closed-form Euler-type plate buckling stress
    ``sigma_cr = k pi^2 D / (b^2 t)`` with the textbook ``k`` factors.

``faulkner_ultimate``
    von Karman / Faulkner effective-width ultimate strength of a wide plate in
    uniform compression, ``sigma_u/sigma_y = 2/beta - 1/beta^2`` for
    ``beta >= 1``.  It is the analytic benchmark curve that ALPS-type empirical
    fits (Paik et al.) reproduce, and is used as the ultimate-strength reference
    because the tabulated values of Paik et al. (2008) are not available in this
    offline environment (stated explicitly in the report).

``dnv_rp_c201_elastic_buckling``
    DNV-RP-C201 Sec. 5 elastic plate buckling stress, used as the design-code
    cross-check of the critical stress.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .core import PlateGeometry

__all__ = [
    "ritz_buckling_stress",
    "euler_buckling_stress",
    "dnv_rp_c201_elastic_buckling",
    "faulkner_ultimate",
    "EULER_K_FACTORS",
]

#: Textbook buckling coefficients ``k`` for a long plate under uniform
#: longitudinal compression (a/b >= 1), Timoshenko & Gere.
EULER_K_FACTORS = {
    "SSSS": 4.0,
    "CCCC": 6.97,
    "SCSC": 6.97,   # loaded edges clamped, unloaded edges simply supported
    "CSCS": 6.97,   # unloaded edges clamped, loaded edges simply supported
}


def _beam_functions(kind: str, n_terms: int, length: float):
    """Return callables ``f(x)``, ``f'(x)``, ``f''(x)`` for each basis term."""
    if kind == "SS":
        roots = [m * np.pi / length for m in range(1, n_terms + 1)]

        def val(i, x):
            return np.sin(roots[i] * x)

        def d1(i, x):
            return roots[i] * np.cos(roots[i] * x)

        def d2(i, x):
            return -roots[i] ** 2 * np.sin(roots[i] * x)

        return [val, d1, d2], n_terms

    if kind == "C":
        # clamped-clamped beam characteristic functions
        betas = np.array([4.73004074, 7.85320462, 10.9956078, 14.1371655,
                          17.2787597, 20.4203522, 23.5619449, 26.7035376,
                          29.8451302, 32.9867229])[:n_terms]
        coeffs = (np.cosh(betas) - np.cos(betas)) / (np.sinh(betas) - np.sin(betas))

        def val(i, x):
            b = betas[i] / length
            return (np.cosh(b * x) - np.cos(b * x)
                    - coeffs[i] * (np.sinh(b * x) - np.sin(b * x)))

        def d1(i, x):
            b = betas[i] / length
            return b * (np.sinh(b * x) + np.sin(b * x)
                        - coeffs[i] * (np.cosh(b * x) - np.cos(b * x)))

        def d2(i, x):
            b = betas[i] / length
            return b**2 * (np.cosh(b * x) + np.cos(b * x)
                           - coeffs[i] * (np.sinh(b * x) + np.sin(b * x)))

        return [val, d1, d2], len(betas)

    raise ValueError("kind must be 'SS' or 'C'")


def ritz_buckling_stress(geom: PlateGeometry, bc: str = "SSSS",
                         n_terms: int = 8, n_gauss: int = 32
                         ) -> Dict[str, float]:
    """Rayleigh-Ritz elastic buckling stress under uniform axial compression.

    Returns ``sigma_cr`` (Pa), the critical buckling coefficient ``k`` and the
    number of Ritz terms used.  Independent of the finite element solver.
    """
    if bc not in ("SSSS", "CCCC", "SCSC", "CSCS"):
        raise ValueError(f"unsupported bc {bc!r}")
    x_kind = "C" if bc in ("CCCC", "SCSC") else "SS"
    y_kind = "C" if bc in ("CCCC", "CSCS") else "SS"

    fx, nfx = _beam_functions(x_kind, n_terms, geom.a)
    fy, nfy = _beam_functions(y_kind, n_terms, geom.b)

    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    x = 0.5 * geom.a * (xg + 1.0)
    wx = 0.5 * geom.a * wg
    y = 0.5 * geom.b * (xg + 1.0)
    wy = 0.5 * geom.b * wg
    XX, YY = np.meshgrid(x, y, indexing="ij")
    WGT = np.outer(wx, wy)

    basis = [(i, j) for i in range(nfx) for j in range(nfy)]
    nb = len(basis)

    def field(fs, axis, order):
        vals = []
        for (i, j) in basis:
            if axis == "x":
                vals.append(fs[order](i, XX) * fy[0](j, YY))
            else:
                vals.append(fy[order](j, YY) * fx[0](i, XX))
        return np.stack(vals, axis=-1)   # (nxg, nyg, nb)

    phi = np.stack([fx[0](i, XX) * fy[0](j, YY) for (i, j) in basis], axis=-1)
    phi_xx = field(fx, "x", 2)
    phi_yy = field(fy, "y", 2)
    phi_x = field(fx, "x", 1)
    phi_y = field(fy, "y", 1)

    D = geom.flexural_rigidity
    nu = geom.nu

    def integrate(a_field, b_field):
        return np.einsum("ijp,ij,ijq->pq", a_field, WGT, b_field)

    Kb = D * (integrate(phi_xx, phi_xx) + integrate(phi_yy, phi_yy)
              + nu * (integrate(phi_xx, phi_yy) + integrate(phi_yy, phi_xx))
              + 2.0 * (1.0 - nu) * integrate(phi_x * 0.0 + phi_y, phi_x * 0.0 + phi_y))
    # the last term must be the mixed derivative product; build it explicitly
    phi_xy = np.stack([fx[1](i, XX) * fy[1](j, YY) for (i, j) in basis], axis=-1)
    Kb = D * (integrate(phi_xx, phi_xx) + integrate(phi_yy, phi_yy)
              + nu * (integrate(phi_xx, phi_yy) + integrate(phi_yy, phi_xx))
              + 2.0 * (1.0 - nu) * integrate(phi_xy, phi_xy))
    Kg = integrate(phi_x, phi_x)          # unit axial membrane force N_x = 1 N/m

    w, v = np.linalg.eig(np.linalg.solve(Kb, Kg)) if False else (None, None)
    from scipy.linalg import eig
    w, v = eig(Kb, Kg)
    w = np.real(w)
    w = w[np.isfinite(w)]
    n_cr = np.min(w[w > 0.0])
    k = n_cr * geom.b**2 / (np.pi**2 * D)
    return {"sigma_cr": float(n_cr / geom.t), "n_cr": float(n_cr),
            "k": float(k), "n_terms": int(nb)}


def euler_buckling_stress(geom: PlateGeometry, bc: str = "SSSS") -> Dict[str, float]:
    """Closed-form Euler plate buckling stress with the textbook ``k``."""
    k = EULER_K_FACTORS[bc]
    D = geom.flexural_rigidity
    sigma = k * np.pi**2 * D / (geom.b**2 * geom.t)
    return {"sigma_cr": float(sigma), "k": float(k)}


def dnv_rp_c201_elastic_buckling(geom: PlateGeometry, bc: str = "SSSS"
                                 ) -> Dict[str, float]:
    """DNV-RP-C201 (Sec. 5) elastic plate buckling stress.

    ``sigma_cr = C pi^2 E / (12 (1 - nu^2)) (t/b)^2`` with ``C = 4`` for a
    simply supported longitudinally compressed plate (aspect ratio >= 1) and
    ``C = 6.97`` for clamped edges; the reduced slenderness
    ``lambda = sqrt(sigma_y / sigma_cr)`` is also returned.
    """
    C = EULER_K_FACTORS[bc]
    sigma = C * np.pi**2 * geom.E / (12.0 * (1.0 - geom.nu**2)) * (geom.t / geom.b) ** 2
    return {"sigma_cr": float(sigma), "C": float(C),
            "lambda": float(np.sqrt(geom.sigma_y / sigma))}


def faulkner_ultimate(beta: float) -> float:
    """von Karman / Faulkner effective-width ultimate strength ``sigma_u/sigma_y``."""
    if beta < 1.0:
        return 1.0
    return float(2.0 / beta - 1.0 / beta**2)