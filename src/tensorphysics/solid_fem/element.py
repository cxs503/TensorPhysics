"""Isoparametric 3-D solid element kernel: B-matrix, stiffness, hourglass control.

Formulations provided for the *volumetric* response of the element:

``full``
    Straight 2x2x2 (Hex8) / 3x3x3 (Hex20) Gauss integration.  Locks for
    nu -> 0.5.
``bbar``
    Mean-dilatation (Hughes) B-bar: the dilatational part of the
    strain-displacement operator is replaced by its volume average over the
    element, the deviatoric part is untouched.  Removes volumetric locking
    while keeping all strain modes (no spurious modes).
``sri``
    Selective reduced integration: the dilatational part of the stiffness is
    integrated with the 1-point rule, the deviatoric part fully.  Linear
    analysis only (the nonlinear path uses ``bbar``).

Hourglass control (``hex8`` + 1-point rule only) follows the
Flanagan-Belytschko orthogonal hourglass approach: four hourglass base
vectors are projected orthogonal to the element's linear displacement field,
yielding per-direction gamma vectors; the stabilisation stiffness is
``kappa * gamma gamma^T`` with ``kappa = alpha_hg * G * V^(1/3)``
(dimensionally a stiffness, physically the shear stiffness of the element
volume, so ``alpha_hg = O(1)`` is the physical scale and no magic tuning is
needed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shape import GAUSS_RULES, default_rule, n_nodes, shape

__all__ = [
    "jacobian",
    "b_matrix",
    "ElementKinematics",
    "element_kinematics",
    "element_volume",
    "element_stiffness",
    "element_strain_energy",
    "hourglass_gamma",
    "hourglass_stiffness",
    "hourglass_energy",
    "element_internal_force",
    "FLANAGAN_BELYTSCKO_H",
]

#: Flanagan-Belytschko hourglass base vectors for the Hex8 node ordering used
#: by :mod:`tensorphysics.solid_fem.shape` (each row sums to zero).
FLANAGAN_BELYTSCKO_H = np.array(
    [
        [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0],
    ]
)


def jacobian(dN_dxi: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Jacobian ``J[i, j] = d x_j / d xi_i`` (3x3)."""
    return dN_dxi @ coords


def b_matrix(dN_dx: np.ndarray) -> np.ndarray:
    """Strain-displacement matrix for Voigt strain ``[exx, eyy, ezz, gxy, gyz, gxz]``.

    ``dN_dx`` has shape (3, n_nodes) with rows d/dx, d/dy, d/dz.
    """
    nn = dN_dx.shape[1]
    B = np.zeros((6, 3 * nn))
    for a in range(nn):
        bx, by, bz = dN_dx[0, a], dN_dx[1, a], dN_dx[2, a]
        B[0, 3 * a + 0] = bx
        B[1, 3 * a + 1] = by
        B[2, 3 * a + 2] = bz
        B[3, 3 * a + 0] = by
        B[3, 3 * a + 1] = bx
        B[4, 3 * a + 1] = bz
        B[4, 3 * a + 2] = by
        B[5, 3 * a + 0] = bz
        B[5, 3 * a + 2] = bx
    return B


@dataclass
class ElementKinematics:
    """Everything an element assembly needs to know point-wise."""

    points: np.ndarray        # (ngp, 3) natural coordinates
    weights: np.ndarray       # (ngp,) Gauss weights (parent-space)
    det_j: np.ndarray         # (ngp,) Jacobian determinant
    B: np.ndarray             # (ngp, 6, ndof) strain-displacement matrices
    volume: float             # element volume
    dN_dx: np.ndarray         # (ngp, 3, n_nodes) world shape derivatives

    @property
    def ndof(self) -> int:
        return self.B.shape[2]

    @property
    def integration_points(self) -> int:
        return self.B.shape[0]


def element_kinematics(
    element: str,
    coords: np.ndarray,
    rule: str | None = None,
    formulation: str = "full",
) -> ElementKinematics:
    """Pre-compute Jacobians and B-matrices at the Gauss points.

    Parameters
    ----------
    element : ``"hex8"`` / ``"hex20"``
    coords : (n_nodes, 3) node coordinates
    rule : Gauss rule name (``"1pt"``, ``"2x2x2"``, ``"3x3x3"``)
    formulation : ``"full"``, ``"bbar"`` or ``"sri"`` (B-matrices are B-bar for
        ``bbar``; ``sri`` is handled by :func:`element_stiffness` directly).
    """
    key = element.lower()
    nn = n_nodes(key)
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (nn, 3):
        raise ValueError(f"{key} expects coords of shape ({nn}, 3), got {coords.shape}")
    rule = rule or default_rule(key)
    if rule not in GAUSS_RULES:
        raise ValueError(f"unknown Gauss rule {rule!r}; known: {sorted(GAUSS_RULES)}")
    pts, wts = GAUSS_RULES[rule]
    ngp = pts.shape[0]

    B = np.zeros((ngp, 6, 3 * nn))
    dN_dx_all = np.zeros((ngp, 3, nn))
    det_j = np.zeros(ngp)
    for g in range(ngp):
        _N, dN_dxi = shape(key, *pts[g])
        J = jacobian(dN_dxi, coords)
        det = float(np.linalg.det(J))
        if det <= 0.0:
            raise ValueError(f"non-positive Jacobian ({det:.3e}) for {key} element")
        dN_dx = np.linalg.solve(J, dN_dxi)
        dN_dx_all[g] = dN_dx
        det_j[g] = det
        B[g] = b_matrix(dN_dx)

    volume = float(np.sum(wts * det_j))
    if volume <= 0.0:
        raise ValueError("non-positive element volume")

    if formulation in {"bbar", "bbar_stab"}:
        B = _apply_bbar(B, wts, det_j, volume)
    elif formulation not in {"full", "sri", "hourglass"}:
        raise ValueError(f"unknown formulation {formulation!r}")

    return ElementKinematics(
        points=pts, weights=wts, det_j=det_j, B=B, volume=volume, dN_dx=dN_dx_all
    )


def _apply_bbar(B: np.ndarray, wts: np.ndarray, det_j: np.ndarray, volume: float) -> np.ndarray:
    """Replace the dilatational strain rows by their volume average."""
    ngp = B.shape[0]
    dv = wts * det_j
    # dilatational row: B_v = sum of the three normal-strain rows, B_v @ u = tr(eps)
    B_v = B[:, 0, :] + B[:, 1, :] + B[:, 2, :]
    B_v_bar = np.sum((dv / volume)[:, None] * B_v, axis=0)
    out = B.copy()
    for g in range(ngp):
        for r in range(3):
            out[g, r, :] = B[g, r, :] - B_v[g] / 3.0 + B_v_bar / 3.0
    return out


def _fluctuation_volumetric_stiffness(
    kin: "ElementKinematics",
    kin_full: "ElementKinematics",
    C: np.ndarray,
    dv: np.ndarray,
) -> np.ndarray:
    """Volumetric stiffness acting on the dilatation *fluctuation*.

    ``B_v`` is the point-wise dilatation row and ``B_v_bar`` its volume average
    (the B-bar part).  ``int (B_v - B_v_bar)(B_v - B_v_bar)^T dV`` is
    semi-positive definite with a nullspace that includes every *uniform*
    dilatation mode, so adding ``stab * K_bulk`` times this matrix removes the
    spurious zero-energy modes of the mean-dilatation element without
    stiffening the physical (nearly uniform) volumetric response.
    """
    m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    bulk = float(m @ C @ m) / 9.0
    B_v = kin_full.B[:, 0, :] + kin_full.B[:, 1, :] + kin_full.B[:, 2, :]
    volume = float(np.sum(dv))
    B_v_bar = np.sum((dv / volume)[:, None] * B_v, axis=0)
    k = np.zeros_like(B_v_bar[:, None] * B_v_bar[None, :])
    for g in range(B_v.shape[0]):
        fluct = B_v[g] - B_v_bar
        k += dv[g] * np.outer(fluct, fluct)
    return bulk * k


def element_volume(element: str, coords: np.ndarray, rule: str | None = None) -> float:
    """Element volume by Gauss integration of the Jacobian determinant."""
    return element_kinematics(element, coords, rule).volume


def element_stiffness(
    element: str,
    coords: np.ndarray,
    C: np.ndarray,
    rule: str | None = None,
    formulation: str = "full",
    hourglass: float = 0.0,
    shear_modulus: float | None = None,
    vol_stab: float = 1.0e-3,
) -> np.ndarray:
    """Element stiffness matrix (ndof x ndof).

    Parameters
    ----------
    C : (6, 6) material tangent in Voigt form.
    hourglass : hourglass coefficient ``alpha_hg``.  Only active for Hex8 with
        the 1-point rule; ``0.0`` disables stabilisation (pure 1-point
        reduced integration, rank deficient).
    shear_modulus : G used for the hourglass stiffness scale (default: taken
        from ``C`` via ``C[3, 3]``).
    vol_stab : relative weight of the dilatation-fluctuation stabilisation used
        by the ``bbar_stab`` formulation (see
        :func:`_fluctuation_volumetric_stiffness`).
    """
    key = element.lower()
    nn = n_nodes(key)
    rule = rule or default_rule(key)
    k = np.zeros((3 * nn, 3 * nn))

    if formulation in {"bbar", "bbar_stab"}:
        if formulation == "bbar" and key == "hex20":
            # plain mean dilatation leaves three spurious zero-energy modes for
            # the 20-node serendipity brick
            raise ValueError(
                "B-bar is not stable for hex20 (3 spurious zero-energy modes); "
                "use formulation='bbar_stab' or 'sri'"
            )
        kin = element_kinematics(key, coords, rule, "bbar")
        kin_full = element_kinematics(key, coords, rule, "full")
        dv = kin.weights * kin.det_j
        for g in range(kin.integration_points):
            k += dv[g] * (kin.B[g].T @ C @ kin.B[g])
        if formulation == "bbar_stab":
            k += vol_stab * _fluctuation_volumetric_stiffness(kin, kin_full, C, dv)
        if hourglass and key == "hex8" and rule == "1pt":
            G = float(C[3, 3]) if shear_modulus is None else float(shear_modulus)
            k += hourglass_stiffness(coords, hourglass, G)
        return 0.5 * (k + k.T)

    if formulation == "sri":
        kin = element_kinematics(key, coords, rule, "full")
        dv = kin.weights * kin.det_j
        m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        bulk_modulus = float(m @ C @ m) / 9.0          # m^T C m = 9 K
        C_bulk = bulk_modulus * np.outer(m, m)
        C_dev = C - C_bulk
        for g in range(kin.integration_points):
            k += dv[g] * (kin.B[g].T @ C_dev @ kin.B[g])
        # dilatational part with the reduced rule: 1 point for Hex8, 2x2x2 for
        # Hex20 (3x3x3 deviatoric / 2x2x2 volumetric is the standard stable
        # pairing for the 20-node brick)
        kin1 = element_kinematics(key, coords, "1pt" if key == "hex8" else "2x2x2", "full")
        dv1 = kin1.weights * kin1.det_j
        for g1 in range(kin1.integration_points):
            k += dv1[g1] * (kin1.B[g1].T @ C_bulk @ kin1.B[g1])
    else:
        kin = element_kinematics(key, coords, rule, formulation)
        dv = kin.weights * kin.det_j
        for g in range(kin.integration_points):
            k += dv[g] * (kin.B[g].T @ C @ kin.B[g])

    if hourglass and key == "hex8" and rule == "1pt":
        G = float(C[3, 3]) if shear_modulus is None else float(shear_modulus)
        k += hourglass_stiffness(coords, hourglass, G)
    return 0.5 * (k + k.T)


def element_strain_energy(
    element: str,
    coords: np.ndarray,
    u_el: np.ndarray,
    C: np.ndarray,
    rule: str | None = None,
    formulation: str = "full",
    hourglass: float = 0.0,
    shear_modulus: float | None = None,
) -> float:
    """Strain energy ``0.5 u^T K u`` of one element."""
    k = element_stiffness(
        element,
        coords,
        C,
        rule=rule,
        formulation=formulation,
        hourglass=hourglass,
        shear_modulus=shear_modulus,
    )
    return 0.5 * float(u_el @ k @ u_el)


def hourglass_gamma(coords: np.ndarray) -> np.ndarray:
    """Orthogonal hourglass gamma vectors, shape (4, 8).

    ``gamma = h - (h . x_i) B_i`` makes each gamma vector orthogonal to both
    the constant and the linear displacement field, so hourglass stabilisation
    cannot pollute rigid-body or constant-strain response.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (8, 3):
        raise ValueError("hourglass control is defined for Hex8 (8, 3) coordinates")
    kin = element_kinematics("hex8", coords, "1pt", "full")
    B_lin = kin.dN_dx[0]                      # (3, 8) derivatives at the centre
    gamma = np.zeros((4, 8))
    for a in range(4):
        h = FLANAGAN_BELYTSCKO_H[a]
        g = h.copy()
        for i in range(3):
            g -= float(h @ coords[:, i]) * B_lin[i]
        gamma[a] = g
    return gamma


def hourglass_stiffness(coords: np.ndarray, alpha_hg: float, shear_modulus: float) -> np.ndarray:
    """Flanagan-Belytschko hourglass stabilisation stiffness (24x24).

    ``kappa = alpha_hg * G * V^(1/3)`` -- the shear stiffness of a cube of the
    element volume, i.e. the natural physical scale for an hourglass mode of
    unit generalised amplitude.
    """
    coords = np.asarray(coords, dtype=float)
    gamma = hourglass_gamma(coords)
    volume = element_kinematics("hex8", coords, "1pt").volume
    kappa = float(alpha_hg) * float(shear_modulus) * volume ** (1.0 / 3.0)
    k = np.zeros((24, 24))
    for a in range(4):
        for i in range(3):
            dofs = 3 * np.arange(8) + i
            k[np.ix_(dofs, dofs)] += kappa * np.outer(gamma[a], gamma[a])
    return k


def hourglass_energy(
    coords: np.ndarray, u_el: np.ndarray, alpha_hg: float, shear_modulus: float
) -> float:
    """Stabilisation energy ``0.5 u^T K_hg u`` of one Hex8 element."""
    k = hourglass_stiffness(coords, alpha_hg, shear_modulus)
    u_el = np.asarray(u_el, dtype=float)
    return 0.5 * float(u_el @ k @ u_el)


def element_internal_force(
    element: str,
    coords: np.ndarray,
    stress: list[np.ndarray] | np.ndarray,
    rule: str | None = None,
    formulation: str = "full",
) -> np.ndarray:
    """Internal force vector ``f = sum_gp w detJ B^T sigma``.

    ``stress`` is either a (6,) constant stress (linear analysis) or a list /
    (ngp, 6) array of stresses at the Gauss points.
    """
    key = element.lower()
    rule = rule or default_rule(key)
    kin = element_kinematics(key, coords, rule, formulation)
    dv = kin.weights * kin.det_j
    stress = np.asarray(stress, dtype=float)
    if stress.ndim == 1:
        stress = np.tile(stress, (kin.integration_points, 1))
    f = np.zeros(kin.ndof)
    for g in range(kin.integration_points):
        f += dv[g] * (kin.B[g].T @ stress[g])
    return f