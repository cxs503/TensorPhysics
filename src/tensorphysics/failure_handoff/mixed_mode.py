"""Mixed-mode crack-tip characterisation for the FEM→DEM handoff.

Why this module exists
----------------------
The frozen ``failure_faces`` v3 packet carries surface geometry and cohesive
energy, but no crack-tip state.  A handoff protocol that claims to convert a
*continuous* crack into discrete fragments has to answer two questions that only
crack-tip mechanics can answer:

1. **Which way does the crack go?**  The maximum circumferential stress (MTS)
   criterion of Erdogan & Sih (1963) turns the mixed-mode stress intensity
   factors ``K_I, K_II`` into a kink angle.  The FEM prediction is obtained
   independently, from a full-field scan of the element stresses
   (:func:`ring_kink_angle`), so the comparison is between two *different*
   numerical routes, not between a formula and itself.
2. **Is the FEM field good enough to be the handoff source?**  The
   displacement-correlation estimate of ``K_I`` (:func:`stress_intensity_factors`)
   is checked on an un-notched-offset three-point-bend specimen against the
   published single-edge-notch-bend (SENB) geometry function of ASTM E399 /
   Tada, which is an external reference and not part of this code base.

Conventions
-----------
``x`` along the span, ``y`` up, ``z`` through the thickness; the notch is cut
from the bottom face at mid-span, so the crack plane is ``x = const`` and the
crack grows in ``+y``.  The local crack-tip frame is therefore
``(e_X, e_Y) = (e_y, -e_x)`` (right handed with ``e_z``), and the kink angle
``theta`` is measured from ``+y`` towards ``+x``:

    e_r = (sin theta, cos theta, 0),   e_theta = (cos theta, -sin theta, 0)

Displacement correlation
------------------------
For a crack along ``X`` with faces ``Y = +-r`` the standard near-tip relations
(plane strain ``kappa = 3 - 4 nu``) are

    K_I  = mu/(kappa+1) sqrt(2 pi / r) [u_Y(r, +pi) - u_Y(r, -pi)]
    K_II = mu/(kappa+1) sqrt(2 pi / r) [u_X(r, +pi) - u_X(r, -pi)]

Rotating ``X -> y`` and ``Y -> -x`` gives the working form used here,

    K_I  = mu/(kappa+1) sqrt(2 pi / r) (u_x^right - u_x^left)
    K_II = mu/(kappa+1) sqrt(2 pi / r) (u_y^left  - u_y^right)

with ``left``/``right`` the two notch faces at ``x = x0`` and ``x = x0 + gap``
and ``r`` the distance below the notch tip.  ``K(r)`` is extrapolated linearly
in ``r`` to ``r = 0``; the raw ``K(r)`` values are kept in the returned
diagnostics so the extrapolation can be audited instead of trusted.

Scope
-----
Linear elastic, small strain, through-thickness straight crack front, element
deletion for crack advance.  Not a cohesive-zone or XFEM solver: the kink angle
is a *prediction* of the local criterion applied to a linear-elastic field, and
the advance is discretised on the hex mesh (documented discretisation error).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable, Sequence

import numpy as np

from ..solid_fem.element import element_kinematics
from ..solid_fem.material import isotropic_elasticity
from ..solid_fem.solver import ElasticSolution, solve_elastic

from .three_point_bend import (
    BendModel,
    BeamSpec,
    _load_patch_dofs,
    _prune,
    _support_dofs,
    element_average_stress,
)

__all__ = [
    "EFFECTIVE_KAPPA_CASES",
    "notch_face_node_pairs",
    "stress_intensity_factors",
    "mts_kink_angle_deg",
    "asymptotic_sigma_theta",
    "asymptotic_kink_angle_deg",
    "ring_kink_angle",
    "williams_stress_fit",
    "element_centroids",
    "senb_k1_reference",
    "solve_step",
    "principal_stress",
    "propagate_crack",
]

#: Poisson-ratio dependent ``kappa`` used by the displacement correlation.
#: ``plane strain`` is the standard choice for a through-thickness crack front;
#: ``plane stress`` is reported as a sensitivity because a 20 mm specimen is thin.
EFFECTIVE_KAPPA_CASES: dict[str, str] = {"plane_strain": "3 - 4 nu", "plane_stress": "(3 - nu)/(1 + nu)"}


def _kappa(plane: str, nu: float) -> float:
    if plane == "plane_strain":
        return 3.0 - 4.0 * nu
    if plane == "plane_stress":
        return (3.0 - nu) / (1.0 + nu)
    raise ValueError(f"unknown plane assumption {plane!r}; expected 'plane_strain' or 'plane_stress'")


def _key(y: float, z: float, digits: int = 9) -> tuple[float, float]:
    return (round(float(y), digits), round(float(z), digits))


def notch_face_node_pairs(
    model: BendModel, *, z_rel: float = 0.5, tol_rel: float = 1e-6
) -> list[tuple[float, int, int]]:
    """Node pairs straddling the notch at the same ``y`` and through-thickness station.

    Returns ``[(r, node_left, node_right), ...]`` sorted by increasing ``r`` (the
    distance below the notch tip ``y = notch_depth_effective_m``).
    """
    nodes = model.mesh.nodes
    x_left = model.crack_plane_x_m
    x_right = x_left + model.notch_gap_effective_m
    span = max(model.spec.length_m, model.spec.height_m)
    tol = tol_rel * span
    z_mid = z_rel * model.spec.thickness_m
    # Through-thickness station: the node plane closest to z_rel * t.  The plane
    # is taken from the node coordinates themselves (no rounding), and the match
    # tolerance is a fraction of the specimen size so it is mesh-independent.
    z_values = np.unique(nodes[:, 2])
    z_near = float(z_values[np.argmin(np.abs(z_values - z_mid))])
    z_tol = 1e-6 * max(span, model.spec.thickness_m)

    left = np.nonzero((np.abs(nodes[:, 0] - x_left) <= tol) & (np.abs(nodes[:, 2] - z_near) <= z_tol))[0]
    right = np.nonzero((np.abs(nodes[:, 0] - x_right) <= tol) & (np.abs(nodes[:, 2] - z_near) <= z_tol))[0]
    if left.size == 0 or right.size == 0:
        raise RuntimeError(
            "could not find nodes on both notch faces at the mid-thickness station; "
            f"x_left={x_left:.6g}, x_right={x_right:.6g}, z={z_near:.6g}"
        )
    right_by_key = {_key(nodes[i, 1], nodes[i, 2]): int(i) for i in right}
    tip_y = float(model.notch_depth_effective_m)
    pairs: list[tuple[float, int, int]] = []
    for i in left:
        y = float(nodes[i, 1])
        if y > tip_y + 1e-9:
            continue
        key = _key(y, nodes[i, 2])
        j = right_by_key.get(key)
        if j is None:
            continue
        pairs.append((tip_y - y, int(i), int(j)))
    if not pairs:
        raise RuntimeError(
            "no matched node pairs below the notch tip; the notch faces are not "
            "node-aligned (check the notch gap snapping)"
        )
    return sorted(pairs)


def stress_intensity_factors(
    model: BendModel,
    u: np.ndarray,
    *,
    nu: float | None = None,
    plane: str = "plane_strain",
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Displacement-correlation ``K_I`` / ``K_II`` extrapolated to the notch tip."""
    u = np.asarray(u, dtype=float).reshape(-1, 3)
    nu = float(model.spec.nu if nu is None else nu)
    E = float(model.spec.E_pa)
    mu = E / (2.0 * (1.0 + nu))
    kappa = _kappa(plane, nu)
    pairs = notch_face_node_pairs(model)
    if max_pairs is not None:
        pairs = pairs[: int(max_pairs)]
    rows: list[dict[str, float]] = []
    # The first node row sits *at* the notch tip, where the asymptotic field is
    # not defined (and 1/sqrt(r) would blow up): it is used for the geometry only.
    r_floor = 1e-3 * float(model.dy_m)
    for r, i, j in pairs:
        if r <= r_floor:
            continue
        du_x = float(u[j, 0] - u[i, 0])  # opening of the notch
        du_y = float(u[i, 1] - u[j, 1])  # sliding of the notch faces
        factor = mu / (kappa + 1.0) * math.sqrt(2.0 * math.pi / r)
        rows.append(
            {
                "r_m": r,
                "du_x_m": du_x,
                "du_y_m": du_y,
                "k1_r": factor * du_x,
                "k2_r": factor * du_y,
            }
        )
    if len(rows) < 2:
        raise RuntimeError("at least two node pairs are needed to extrapolate K to the tip")
    r = np.array([row["r_m"] for row in rows])
    k1 = np.array([row["k1_r"] for row in rows])
    k2 = np.array([row["k2_r"] for row in rows])

    def _extrapolate(rr: np.ndarray, kk: np.ndarray) -> tuple[float, float]:
        # Linear least squares of K(r) = K_tip + slope * r  -> intercept is K at r=0.
        A = np.vstack([np.ones_like(rr), rr]).T
        coef, *_ = np.linalg.lstsq(A, kk, rcond=None)
        resid = kk - A @ coef
        ss_tot = float(np.sum((kk - kk.mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0.0 else 1.0
        return float(coef[0]), float(r2)

    k1_tip, k1_r2 = _extrapolate(r, k1)
    k2_tip, k2_r2 = _extrapolate(r, k2)
    return {
        "plane": plane,
        "kappa": kappa,
        "shear_modulus_pa": mu,
        "pair_count": len(rows),
        "r_range_m": [float(r.min()), float(r.max())],
        "k1_pa_sqrt_m": k1_tip,
        "k2_pa_sqrt_m": k2_tip,
        "k1_fit_r2": k1_r2,
        "k2_fit_r2": k2_r2,
        "samples": rows,
    }


def williams_stress_fit(
    model: BendModel,
    sigma_element: np.ndarray,
    *,
    tip: Sequence[float] | None = None,
    r_min_m: float | None = None,
    r_max_m: float | None = None,
) -> dict[str, Any]:
    """Least-squares first-order Williams fit of ``K_I, K_II`` to the FE stress field.

    Every element centroid in the annulus ``r_min <= r <= r_max`` around the notch
    tip contributes three equations (``sigma_XX, sigma_YY, tau_XY`` in the local
    crack frame ``X = +y``, ``Y = -x``); the unknowns are the two stress intensity
    factors plus the constant T-stress.  This estimator uses the *stress* field over a whole annulus of
    elements, so it is independent of the *displacement* correlation of
    :func:`stress_intensity_factors` — which is what makes the C1 comparison a
    real cross-check rather than a tautology.
    """
    if tip is None:
        tip = (
            model.crack_plane_x_m + 0.5 * model.notch_gap_effective_m,
            model.notch_depth_effective_m,
        )
    x_tip, y_tip = float(tip[0]), float(tip[1])
    # The K-dominant annulus must stay well inside the ligament (r << a); with a
    # 5 mm mesh and a 20 mm notch this means r in [1, 2.5] element sizes, which is
    # the smallest annulus that still contains enough element centroids to fit two
    # unknowns.  Larger annuli fall outside the asymptotic zone (fit R^2 drops).
    r_min = float(r_min_m if r_min_m is not None else 1.0 * model.dy_m)
    r_max = float(r_max_m if r_max_m is not None else 2.5 * model.dy_m)
    centroids = element_centroids(model.mesh)
    A: list[list[float]] = []
    b: list[float] = []
    used = 0
    for e in range(model.mesh.n_elem):
        dx_e = float(centroids[e, 0] - x_tip)
        dy_e = float(centroids[e, 1] - y_tip)
        r = math.hypot(dx_e, dy_e)
        if r < r_min or r > r_max:
            continue
        X, Y = dy_e, -dx_e
        theta = math.atan2(Y, X)
        sxx, syy, _szz, sxy = (float(v) for v in np.asarray(sigma_element[e]).reshape(6)[:4])
        s_xx, s_yy, t_xy = syy, sxx, -sxy
        c = math.cos(0.5 * theta)
        s = math.sin(0.5 * theta)
        c3 = math.cos(1.5 * theta)
        s3 = math.sin(1.5 * theta)
        scale = 1.0 / math.sqrt(2.0 * math.pi * r)
        # Third unknown: the T-stress (a constant sigma_XX), which is the
        # dominant non-singular term over a finite annulus.  Without it the K
        # fit absorbs part of T and biases both factors low.
        A.append([scale * c * (1.0 - s * s3), -scale * s * (2.0 + c * c3), 1.0])
        b.append(s_xx)
        A.append([scale * c * (1.0 + s * s3), scale * s * c * c3, 0.0])
        b.append(s_yy)
        A.append([scale * c * s * c3, scale * c * (1.0 - s * s3), 0.0])
        b.append(t_xy)
        used += 1
    if used < 6:
        raise RuntimeError(
            f"only {used} element centroids in the annulus {r_min:.4g}..{r_max:.4g} m; "
            "widen the annulus or refine the mesh"
        )
    A_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    coef, *_ = np.linalg.lstsq(A_arr, b_arr, rcond=None)
    resid = b_arr - A_arr @ coef
    ss_tot = float(np.sum((b_arr - b_arr.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0.0 else 1.0
    return {
        "k1_pa_sqrt_m": float(coef[0]),
        "k2_pa_sqrt_m": float(coef[1]),
        "t_stress_pa": float(coef[2]),
        "fit_r2": r2,
        "element_count": used,
        "r_range_m": [r_min, r_max],
        "rms_residual_pa": float(np.sqrt(np.mean(resid**2))),
        "stress_rms_pa": float(np.sqrt(np.mean(b_arr**2))),
    }


def mts_kink_angle_deg(k1: float, k2: float) -> float:
    """Erdogan–Sih maximum circumferential stress kink angle, in degrees.

    ``theta = 2 atan[(K_I - sqrt(K_I^2 + 8 K_II^2)) / (4 K_II)]``, with the
    limiting cases ``K_II -> 0`` (``theta -> 0``) and ``K_I -> 0``
    (``theta -> -70.53 deg``) handled explicitly so the formula is not evaluated
    at a removable singularity.
    """
    k1 = float(k1)
    k2 = float(k2)
    scale = max(abs(k1), abs(k2), 1e-30)
    if abs(k2) <= 1e-12 * scale:
        # Pure mode I: the MTS angle is 0 for K_I > 0 and 180 for K_I < 0;
        # only the K_I > 0 branch is physical for a tensile crack.
        return 0.0
    root = math.sqrt(k1 * k1 + 8.0 * k2 * k2)
    theta = 2.0 * math.atan((k1 - root) / (4.0 * k2))
    return math.degrees(theta)


def asymptotic_sigma_theta(k1: float, k2: float, r: float, theta_rad: float) -> float:
    """First-order Williams circumferential stress at ``(r, theta)``."""
    if r <= 0.0:
        raise ValueError("r must be positive")
    c = math.cos(0.5 * theta_rad)
    return c / math.sqrt(2.0 * math.pi * r) * (
        k1 * c * c - 1.5 * k2 * math.sin(theta_rad)
    )


def asymptotic_kink_angle_deg(k1: float, k2: float, *, n: int = 720) -> float:
    """Argmax of :func:`asymptotic_sigma_theta` over ``theta in [-90, 90] deg``.

    Numerically independent of the closed-form MTS angle: the scan only uses the
    Williams field, so agreement between the two is a real check of the code path
    that feeds the criterion.
    """
    thetas = np.radians(np.linspace(-90.0, 90.0, n))
    values = [asymptotic_sigma_theta(k1, k2, 1.0, float(t)) for t in thetas]
    best = int(np.argmax(values))
    return float(np.degrees(thetas[best]))


def element_centroids(mesh) -> np.ndarray:
    """Element centroids as an ``(n_elem, 3)`` array (one vectorised pass)."""
    return np.asarray(mesh.nodes, dtype=float)[np.asarray(mesh.connectivity, dtype=int)].mean(axis=1)


def _nearest_element_stress(
    mesh, sigma_element: np.ndarray, point: np.ndarray, centroids: np.ndarray | None = None
) -> np.ndarray:
    """Element-average stress of the element whose centroid is closest to ``point``."""
    if centroids is None:
        centroids = element_centroids(mesh)
    d2 = np.sum((centroids - point) ** 2, axis=1)
    return np.asarray(sigma_element[int(np.argmin(d2))], dtype=float).reshape(6)


def ring_kink_angle(
    model: BendModel,
    sigma_element: np.ndarray,
    *,
    tip: Sequence[float] | None = None,
    radius_m: float | None = None,
    n_angles: int = 721,
) -> dict[str, Any]:
    """Full-field FEM kink-angle prediction: argmax of ``sigma_theta_theta``.

    ``sigma_theta_theta = sigma_xx cos^2 t + sigma_yy sin^2 t - 2 sigma_xy sin t cos t``
    is evaluated at points on a circular arc of radius ``radius_m`` around the
    notch tip, using the element-average stress of the nearest element.  This is
    a *finite-radius* evaluation of the criterion on the raw FE field: it does not
    use ``K_I``/``K_II`` at all.
    """
    nodes = model.mesh.nodes
    if tip is None:
        x_tip = model.crack_plane_x_m + 0.5 * model.notch_gap_effective_m
        y_tip = model.notch_depth_effective_m
    else:
        x_tip, y_tip = float(tip[0]), float(tip[1])
    radius = float(radius_m if radius_m is not None else 2.0 * model.dy_m)
    thetas = np.radians(np.linspace(-90.0, 90.0, int(n_angles)))
    values = np.empty_like(thetas)
    centroids = element_centroids(model.mesh)
    z_mid = 0.5 * (float(nodes[:, 2].min()) + float(nodes[:, 2].max()))
    for index, theta in enumerate(thetas):
        # Local crack frame: X = +y (crack axis), Y = -x (face normal), so a
        # positive theta rotates from +y towards -x, the same sign convention as
        # the Erdogan-Sih formula and the Williams fit.
        point = np.array([x_tip - radius * math.sin(theta), y_tip + radius * math.cos(theta), z_mid])
        stress = _nearest_element_stress(model.mesh, sigma_element, point, centroids)
        sxx, syy, _szz, sxy = stress[0], stress[1], stress[2], stress[3]
        values[index] = (
            sxx * math.cos(theta) ** 2
            + syy * math.sin(theta) ** 2
            + 2.0 * sxy * math.sin(theta) * math.cos(theta)
        )
    best = int(np.argmax(values))
    # 3-point parabolic refinement so the reported angle is not quantised by n_angles.
    angle = float(np.degrees(thetas[best]))
    if 0 < best < len(values) - 1:
        y0, y1, y2 = values[best - 1], values[best], values[best + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 0.0:
            delta = 0.5 * (y0 - y2) / denom
            step = float(np.degrees(thetas[1] - thetas[0]))
            angle += float(np.clip(delta, -1.0, 1.0)) * step
    return {
        "radius_m": radius,
        "theta_deg": angle,
        "peak_sigma_theta_pa": float(values[best]),
        "min_sigma_theta_pa": float(values.min()),
        "max_sigma_theta_pa": float(values.max()),
        "scan_deg": [float(np.degrees(thetas[0])), float(np.degrees(thetas[-1]))],
    }


def senb_k1_reference(
    load_n: float,
    span_m: float,
    width_m: float,
    thickness_m: float,
    crack_length_m: float,
) -> dict[str, float]:
    """ASTM E399 / Tada single-edge-notch-bend ``K_I`` (``S/W = 4`` geometry).

        K_I = 3 P S / (2 t W^2) * sqrt(pi a) * F(alpha)
        F(alpha) = 1.99 - alpha (1-alpha)(2.15 - 3.93 alpha + 2.7 alpha^2) /
                   ((1 + 2 alpha)(1 - alpha)^{3/2})
    """
    alpha = float(crack_length_m) / float(width_m)
    if not 0.0 < alpha < 1.0:
        raise ValueError("crack_length/width must be in (0, 1)")
    F = 1.99 - alpha * (1.0 - alpha) * (2.15 - 3.93 * alpha + 2.7 * alpha * alpha) / (
        (1.0 + 2.0 * alpha) * (1.0 - alpha) ** 1.5
    )
    prefactor = 3.0 * load_n * span_m / (2.0 * thickness_m * width_m * width_m)
    return {
        "alpha": alpha,
        "geometry_function": float(F),
        "k1_pa_sqrt_m": float(prefactor * math.sqrt(math.pi * crack_length_m) * F),
    }


def solve_step(
    model: BendModel,
    *,
    applied_displacement_m: float,
    load_offset_m: float = 0.0,
    formulation: str = "full",
) -> tuple[ElasticSolution, np.ndarray]:
    """Solve one linear-elastic step on ``model.mesh`` (offset load allowed)."""
    mesh = model.mesh
    spec = model.spec
    C = isotropic_elasticity(spec.E_pa, spec.nu)
    fixed_dofs, _support_ids = _support_dofs(mesh, spec)
    patch = 2.0 * model.dx_m
    load_dofs = _load_patch_dofs(mesh, spec, patch, load_offset_m)
    prescribed: dict[int, float] = {int(d): 0.0 for d in fixed_dofs}
    for dof in load_dofs:
        if int(dof) in prescribed:
            raise RuntimeError(f"dof {int(dof)} is both supported and loaded")
        prescribed[int(dof)] = -abs(applied_displacement_m)
    dofs = np.array(sorted(prescribed), dtype=int)
    values = np.array([prescribed[int(d)] for d in dofs], dtype=float)
    solution = solve_elastic(mesh, C, dofs, fixed_values=values, formulation=formulation)
    sigma = element_average_stress(mesh, solution.u, C, formulation=formulation)
    return solution, sigma


def principal_stress(sigma_voigt: np.ndarray) -> np.ndarray:
    """Principal stresses (ascending) of a Voigt ``[xx, yy, zz, xy, yz, zx]`` stress."""
    sxx, syy, szz, sxy, syz, szx = (float(v) for v in np.asarray(sigma_voigt).reshape(6))
    tensor = np.array([[sxx, sxy, szx], [sxy, syy, syz], [szx, syz, szz]])
    return np.linalg.eigvalsh(tensor)


def propagate_crack(
    model: BendModel,
    *,
    applied_displacement_m: float,
    load_offset_m: float = 0.0,
    steps: int = 3,
    advance_radius_m: float | None = None,
    initial_keep: np.ndarray | None = None,
) -> dict[str, Any]:
    """Advance the crack by element deletion and measure the resulting path angle.

    At every step the element in front of the tip with the largest first principal
    stress is deleted and the model re-solved, so the path is driven by the
    *element-level* stress state and not by the MTS formula.  The path angle is
    measured from the initial notch tip to the volume centroid of the deleted
    element set, which is the direction the crack actually took in the discrete
    model.

    This is deliberately a crude discrete-crack model (element deletion, no
    cohesive softening) — it is used as an *independent* FEM-side direction
    estimate next to :func:`ring_kink_angle`, and its discretisation error is
    reported (``step_centroids_m``) instead of hidden.
    """
    base_mesh = model.base_mesh if model.base_mesh is not None else model.mesh
    keep = (
        np.asarray(initial_keep, dtype=bool).copy()
        if initial_keep is not None
        else (
            np.asarray(model.notch_mask, dtype=bool).copy()
            if model.notch_mask is not None
            else np.ones(base_mesh.n_elem, dtype=bool)
        )
    )
    radius = float(advance_radius_m if advance_radius_m is not None else 3.0 * model.dy_m)
    tip = np.array(
        [model.crack_plane_x_m + 0.5 * model.notch_gap_effective_m, model.notch_depth_effective_m]
    )
    path: list[list[float]] = []
    removed: list[int] = []
    step_stress_max: list[float] = []
    for _step in range(int(steps)):
        mesh_step, original_ids = _prune(base_mesh, keep)
        model_step = replace(
            model,
            mesh=mesh_step,
            element_ids_original=original_ids,
            notch_element_count=int((~keep).sum()),
        )
        _solution, sigma = solve_step(
            model_step,
            applied_displacement_m=applied_displacement_m,
            load_offset_m=load_offset_m,
        )
        centroids = element_centroids(mesh_step)
        best_index = -1
        best_sigma1 = -np.inf
        for e in range(mesh_step.n_elem):
            centroid = centroids[e]
            # Only elements strictly ahead of the current tip may be removed: a
            # crack cannot advance sideways into material it already passed.
            if centroid[1] < tip[1] + 0.5 * model.dy_m - 1e-12:
                continue
            if centroid[1] > tip[1] + radius:
                continue
            if abs(centroid[0] - tip[0]) > radius:
                continue
            sigma1 = float(principal_stress(sigma[e])[-1])
            if sigma1 > best_sigma1:
                best_sigma1 = sigma1
                best_index = e
        if best_index < 0:
            raise RuntimeError(
                f"no intact element found in front of the tip (y={tip[1]:.4g}, radius={radius:.4g})"
            )
        centroid = centroids[best_index]
        tip = np.array([float(centroid[0]), float(centroid[1])])
        original = int(original_ids[best_index])
        keep[original] = False
        removed.append(original)
        step_stress_max.append(best_sigma1)
        path.append([float(centroid[0]), float(centroid[1]), float(centroid[2])])
    tip0 = np.array(
        [model.crack_plane_x_m + 0.5 * model.notch_gap_effective_m, model.notch_depth_effective_m]
    )
    # Path direction from the *volume centroid* of the removed set (not from the
    # last element), which averages out the single-element discretisation noise.
    centroids_removed = np.array([p[:2] for p in path], dtype=float)
    first = centroids_removed[0] - tip0
    theta_first = math.degrees(math.atan2(-float(first[0]), float(first[1])))
    end = centroids_removed.mean(axis=0)
    delta = end - tip0
    # Same sign convention as the MTS angle: positive rotates from +y towards -x.
    theta = math.degrees(math.atan2(-float(delta[0]), float(delta[1])))
    return {
        "steps": len(path),
        "advance_radius_m": radius,
        "initial_tip_m": [float(v) for v in tip0],
        "final_tip_m": [float(v) for v in end],
        "path_centroids_m": path,
        "removed_elements": removed,
        "max_principal_stress_pa": [float(v) for v in step_stress_max],
        "theta_path_deg": float(theta),
        "theta_path_first_deg": float(theta_first),
        # Element deletion cannot resolve a kink finer than one element: the first
        # advance is quantised by atan(dx/dy), reported so the path number is read
        # with its band instead of as a precise angle.
        "quantisation_deg": float(math.degrees(math.atan2(model.dx_m, model.dy_m))),
        "path_length_m": float(np.linalg.norm(delta)),
        "notch_keep_mask": keep,
    }