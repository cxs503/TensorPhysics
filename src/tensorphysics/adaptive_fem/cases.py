"""Three verification anchors for adaptive h-refinement.

1. **Cook membrane** — tapered panel, clamped left edge, uniform shear on the
   right edge; shear-dominated bending with two stress singularities at the
   clamped corners.  No closed-form solution: the reference is a high-resolution
   numerical solution (and the classical tip deflection 23.9 for E = 1,
   nu = 1/3, t = 1, total load 6.25).
2. **Thick-walled cylinder (Lamé)** — plane strain, internal pressure; exact
   radial displacement/stress available, so both the actual error and the ZZ
   effectivity index are measurable.
3. **Crack tip** — centre-crack geometry in a quarter model with a free crack
   face on the symmetry line; the r^(-1/2) stress singularity is the classic
   test of whether an adaptive mesh can capture a non-smooth solution.  The
   analytic mode-I stress intensity factor of a centre crack in a finite-width
   plate is the reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .elasticity2d import Material2D
from .mesh2d import TriMesh, nodal_weights, quad_to_tris
from .problem import DirichletBC, PlaneProblem, TractionBC

__all__ = [
    "structured_quad_mesh",
    "cook_mesh",
    "cook_problem",
    "cook_tip_displacement",
    "annulus_mesh",
    "annulus_projector",
    "LameFields",
    "lame_problem",
    "crack_mesh",
    "crack_problem",
    "center_crack_KI",
    "ki_from_cod",
    "ki_from_stress",
    "ki_from_jintegral",
    "displacement_l2_error",
]


# ---------------------------------------------------------------------------
# generic structured generator
# ---------------------------------------------------------------------------
def structured_quad_mesh(x: np.ndarray, y: np.ndarray, map_fn=None) -> TriMesh:
    """Structured grid over ``x`` × ``y`` with optional coordinate mapping."""
    from .mesh2d import structured_quad_mesh as _sqm

    return _sqm(x, y, map_fn)


# ---------------------------------------------------------------------------
# 1. Cook membrane
# ---------------------------------------------------------------------------
COOK_H_LEFT = 60.0
COOK_H_RIGHT = 44.0
COOK_LENGTH = 48.0
COOK_RISE = COOK_H_LEFT - COOK_H_RIGHT  # 16: bottom edge rises over the length
COOK_LOAD = 6.25
COOK_TIP_REFERENCE = 23.9  # classical reference, E=1, nu=1/3, t=1, P=6.25


def cook_mesh(n: int = 8) -> TriMesh:
    """Structured mesh on the Cook trapezoid (``n`` divisions per side)."""
    x = np.linspace(0.0, COOK_LENGTH, n + 1)
    v = np.linspace(0.0, 1.0, n + 1)

    def mapper(p: np.ndarray) -> np.ndarray:
        u = p[:, 0] / COOK_LENGTH
        vv = p[:, 1]
        return np.stack(
            [COOK_LENGTH * u, COOK_RISE * u * (1.0 - vv) + COOK_H_LEFT * vv], axis=1
        )

    return structured_quad_mesh(x, v, mapper)


def cook_problem(E: float = 1.0, nu: float = 1.0 / 3.0, load: float = COOK_LOAD) -> PlaneProblem:
    mat = Material2D(E=E, nu=nu, thickness=1.0, plane="stress")
    tol = 1e-9 * COOK_LENGTH
    traction = load / COOK_H_RIGHT  # uniform shear on the 44-long right edge
    return PlaneProblem(
        name="cook_membrane",
        material=mat,
        dirichlet=(
            DirichletBC("x", lambda p: np.abs(p[:, 0]) < tol, 0.0, "clamped-x"),
            DirichletBC("y", lambda p: np.abs(p[:, 0]) < tol, 0.0, "clamped-y"),
        ),
        tractions=(
            TractionBC(
                lambda p0, p1: (np.abs(p0[:, 0] - COOK_LENGTH) < tol)
                & (np.abs(p1[:, 0] - COOK_LENGTH) < tol),
                (0.0, traction),
                "shear",
            ),
        ),
        description={
            "geometry": "trapezoid 48 long, 60 high at clamped edge, 44 at loaded edge",
            "material": f"E={E}, nu={nu}, plane stress, t=1",
            "load": f"uniform shear traction {traction} on right edge (total {load})",
            "reference": f"tip vertical displacement {COOK_TIP_REFERENCE} (literature)",
        },
    )


def cook_tip_displacement(mesh: TriMesh, u: np.ndarray, at: tuple[float, float] = (48.0, 60.0)) -> float:
    """Vertical displacement of the node closest to ``at`` (upper-right corner)."""
    d = np.linalg.norm(mesh.nodes - np.asarray(at, float), axis=1)
    k = int(np.argmin(d))
    if d[k] > 1e-9:
        raise ValueError("no node at the Cook tip corner")
    return float(u[2 * k + 1])


# ---------------------------------------------------------------------------
# 2. thick-walled cylinder (Lamé)
# ---------------------------------------------------------------------------
def annulus_projector(a: float, b: float, tol: float = 1e-6):
    """Snap midpoints of arc edges back onto the exact circular boundary.

    New nodes created by h-refinement on a curved boundary would otherwise sit on
    the chord (radius ``a cos(dtheta/2) < a``), which both perturbs the geometry
    and detaches pressure boundary conditions.  ``project`` receives the two edge
    endpoints and returns the new node position; interior edges keep the plain
    midpoint.
    """

    def project(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        p0 = np.atleast_2d(np.asarray(p0, float))
        p1 = np.atleast_2d(np.asarray(p1, float))
        mid = 0.5 * (p0 + p1)
        r0 = np.hypot(p0[:, 0], p0[:, 1])
        r1 = np.hypot(p1[:, 0], p1[:, 1])
        for radius in (a, b):
            sel = (np.abs(r0 - radius) < tol * radius) & (np.abs(r1 - radius) < tol * radius)
            if np.any(sel):
                rr = np.hypot(mid[sel, 0], mid[sel, 1])
                mid[sel] = mid[sel] * (radius / rr)[:, None]
        return mid

    return project


def annulus_mesh(nr: int = 6, nt: int = 12, a: float = 0.1, b: float = 0.25) -> TriMesh:
    """Quarter annulus ``a <= r <= b``, ``0 <= theta <= pi/2``."""
    r = np.linspace(a, b, nr + 1)
    t = np.linspace(0.0, 0.5 * math.pi, nt + 1)
    R, T = np.meshgrid(r, t, indexing="ij")
    nodes = np.stack([(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel()], axis=1)
    idx = np.arange((nr + 1) * (nt + 1)).reshape(nr + 1, nt + 1)
    i, j = np.meshgrid(np.arange(nr), np.arange(nt), indexing="ij")
    q = np.stack(
        [idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]], axis=-1
    ).reshape(-1, 4)
    mesh = TriMesh(nodes, quad_to_tris(nodes, q))
    mesh.meta["project"] = annulus_projector(a, b)
    return mesh


@dataclass
class LameFields:
    a: float
    b: float
    p: float
    material: Material2D

    @property
    def A(self) -> float:
        return self.p * self.a**2 / (self.b**2 - self.a**2)

    @property
    def B(self) -> float:
        return self.p * self.a**2 * self.b**2 / (self.b**2 - self.a**2)

    def u_r(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, float)
        E, nu = self.material.E, self.material.nu
        if self.material.plane == "strain":
            return (1.0 + nu) / E * ((1.0 - 2.0 * nu) * self.A * r + self.B / r)
        return 1.0 / E * ((1.0 - nu) * self.A * r + (1.0 + nu) * self.B / r)

    def displacement(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float)
        r = np.hypot(pts[:, 0], pts[:, 1])
        ur = self.u_r(r)
        return np.stack([ur * pts[:, 0] / r, ur * pts[:, 1] / r], axis=1)

    def stress(self, pts: np.ndarray) -> np.ndarray:
        """Voigt ``(sigma_xx, sigma_yy, sigma_xy)`` at cartesian points."""
        pts = np.asarray(pts, float)
        r = np.hypot(pts[:, 0], pts[:, 1])
        ct = pts[:, 0] / r
        st = pts[:, 1] / r
        sr = self.A - self.B / r**2
        stt = self.A + self.B / r**2
        sxx = sr * ct**2 + stt * st**2
        syy = sr * st**2 + stt * ct**2
        sxy = (sr - stt) * st * ct
        return np.stack([sxx, syy, sxy], axis=1)


def lame_problem(
    a: float = 0.1,
    b: float = 0.25,
    p: float = 100e6,
    E: float = 200e9,
    nu: float = 0.3,
    plane: str = "strain",
) -> PlaneProblem:
    """Quarter-annulus Lamé problem: symmetry planes + internal pressure."""
    mat = Material2D(E=E, nu=nu, thickness=1.0, plane=plane)
    tol = 1e-9 * b

    def inner_arc(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        r0 = np.hypot(p0[:, 0], p0[:, 1])
        r1 = np.hypot(p1[:, 0], p1[:, 1])
        # both endpoints on (or, pre-projection, just inside) the inner circle
        return (r0 <= a * (1.0 + 1e-9)) & (r1 <= a * (1.0 + 1e-9))

    def pressure(mid: np.ndarray) -> np.ndarray:
        r = np.hypot(mid[:, 0], mid[:, 1])
        return p * np.stack([mid[:, 0] / r, mid[:, 1] / r], axis=1)

    return PlaneProblem(
        name=f"thick_wall_cylinder_{plane}",
        material=mat,
        dirichlet=(
            DirichletBC("y", lambda q: np.abs(q[:, 1]) < tol, 0.0, "symmetry-y0"),
            DirichletBC("x", lambda q: np.abs(q[:, 0]) < tol, 0.0, "symmetry-x0"),
        ),
        tractions=(TractionBC(inner_arc, pressure, "internal-pressure"),),
        description={
            "geometry": f"quarter annulus a={a}, b={b}",
            "material": f"E={E}, nu={nu}, plane {plane}",
            "load": f"internal pressure p={p} on r=a",
            "reference": "Lamé thick-wall cylinder closed form",
        },
    )


# ---------------------------------------------------------------------------
# 3. crack tip
# ---------------------------------------------------------------------------
def crack_mesh(
    a: float = 1.0,
    W: float = 10.0,
    H: float = 10.0,
    n_crack: int = 2,
    n_right: int = 9,
    n_y: int = 10,
) -> TriMesh:
    """Quarter model of a centre-cracked plate; ``(a, 0)`` is a mesh vertex.

    A quasi-uniform, well-shaped start mesh (min angle > 25 deg): the r^(-1/2)
    grading near the tip is produced by the adaptive refinement, not baked into a
    hand-tuned graded grid, so that the demonstration really exercises the
    estimator + marker + refiner chain.
    """
    xs = np.concatenate(
        [np.linspace(0.0, a, n_crack + 1), np.linspace(a, W, n_right + 1)[1:]]
    )
    ys = np.linspace(0.0, H, n_y + 1)
    return structured_quad_mesh(xs, ys)


def crack_problem(
    a: float = 1.0,
    W: float = 10.0,
    H: float = 10.0,
    sigma: float = 1.0,
    E: float = 1000.0,
    nu: float = 0.3,
    plane: str = "stress",
) -> PlaneProblem:
    mat = Material2D(E=E, nu=nu, thickness=1.0, plane=plane)
    tol = 1e-9 * max(W, H)
    return PlaneProblem(
        name="crack_tip_singularity",
        material=mat,
        dirichlet=(
            DirichletBC("x", lambda q: np.abs(q[:, 0]) < tol, 0.0, "symmetry-x0"),
            # the crack tip lies on the symmetry plane, where the exact
            # solution has u_y = 0 (the faces close), so it is constrained too
            DirichletBC(
                "y",
                lambda q: (np.abs(q[:, 1]) < tol) & (q[:, 0] > a - 1e-12),
                0.0,
                "symmetry-crack-line",
            ),
        ),
        tractions=(
            TractionBC(
                lambda p0, p1: (np.abs(p0[:, 1] - H) < tol) & (np.abs(p1[:, 1] - H) < tol),
                (0.0, sigma),
                "remote-tension",
            ),
        ),
        description={
            "geometry": f"quarter model W={W}, H={H}, crack half-length a={a}",
            "material": f"E={E}, nu={nu}, plane {plane}, t=1",
            "load": f"remote tension sigma={sigma} on y={H}",
            "reference": "K_I = sigma*sqrt(pi*a)*sqrt(sec(pi*a/(2W)))",
        },
    )


def center_crack_KI(a: float, W: float, sigma: float) -> float:
    """Mode-I SIF of a centre crack in a finite-width plate (2W wide)."""
    return float(sigma * math.sqrt(math.pi * a) * math.sqrt(1.0 / math.cos(0.5 * math.pi * a / W)))


def _lstsq2(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares fit ``y ~ c0 + c1 * x2`` with basis columns ``[x1, x2]``."""
    A = np.stack([np.asarray(x1, float), np.asarray(x2, float)], axis=1)
    AtA = A.T @ A
    AtY = A.T @ np.asarray(y, float)
    det = AtA[0, 0] * AtA[1, 1] - AtA[0, 1] * AtA[1, 0]
    inv = np.array([[AtA[1, 1], -AtA[0, 1]], [-AtA[1, 0], AtA[0, 0]]]) / det
    return inv @ AtY


def crack_face_nodes(mesh: TriMesh, a: float, tol: float = 1e-9):
    """Nodes on the free crack face ``y = 0, x < a`` with their tip distance."""
    nodes = mesh.nodes
    mask = (np.abs(nodes[:, 1]) < tol * max(1.0, a)) & (nodes[:, 0] < a - tol)
    idx = np.flatnonzero(mask)
    r = a - nodes[idx, 0]
    order = np.argsort(r)
    return idx[order], r[order]


def ki_from_cod(
    mesh: TriMesh,
    u: np.ndarray,
    a: float,
    material: Material2D,
    *,
    n_fit: int = 6,
    max_r_fraction: float = 0.25,
    skip_nearest: int = 1,
) -> tuple[float, dict]:
    """Mode-I SIF from the near-tip crack-opening displacement.

    ``u_y(theta=pi) = 4 K_I / E' * sqrt(r / 2 pi)`` so that
    ``K_I = lim_{r->0} E' u_y sqrt(2 pi) / (4 sqrt(r))``; the limit is taken by
    a least-squares fit ``K(r) = K_I + c sqrt(r)`` over the ``n_fit`` crack-face
    nodes closest to the tip.
    """
    idx, r = crack_face_nodes(mesh, a)
    uy = u[2 * idx + 1]
    sel = r <= max_r_fraction * a
    idx, r, uy = idx[sel], r[sel], uy[sel]
    if r.size < 3:
        raise ValueError("not enough crack-face nodes near the tip")
    # the crack-face node adjacent to the tip is contaminated by the tip-node
    # closure (CST cannot represent r^(1/2) inside one element); it is skipped
    r = r[skip_nearest:]
    uy = uy[skip_nearest:]
    if r.size < 3:
        raise ValueError("not enough crack-face nodes near the tip")
    n_fit = int(min(n_fit, r.size))
    r = r[:n_fit]
    uy = uy[:n_fit]
    K_r = material.E_eff * uy * math.sqrt(2.0 * math.pi) / (4.0 * np.sqrt(r))
    coef = _lstsq2(np.ones_like(r), np.sqrt(r), K_r)
    return float(coef[0]), {
        "r": r.tolist(),
        "K_profile": K_r.tolist(),
        "linear_coeff": float(coef[1]),
        "n_fit": n_fit,
    }


def ki_from_stress(
    mesh: TriMesh,
    stress: np.ndarray,
    a: float,
    *,
    n_fit: int = 6,
    max_r_fraction: float = 0.25,
) -> float:
    """Independent cross-check: ``sigma_yy(theta=0) = K_I / sqrt(2 pi r)``."""
    cent = mesh.centroids()
    mask = (np.abs(cent[:, 1]) < 1e-9) & (cent[:, 0] > a)
    r = cent[mask, 0] - a
    syy = stress[mask, 1]
    sel = r <= max_r_fraction * a
    r, syy = r[sel], syy[sel]
    if r.size < 3:
        raise ValueError("not enough elements ahead of the tip")
    order = np.argsort(r)
    r, syy = r[order][:n_fit], syy[order][:n_fit]
    K_r = syy * np.sqrt(2.0 * math.pi * r)
    coef = _lstsq2(np.ones_like(r), r, K_r)
    return float(coef[0])


# ---------------------------------------------------------------------------
# error metrics
# ---------------------------------------------------------------------------
def displacement_l2_error(
    mesh: TriMesh, u: np.ndarray, exact: callable, thickness: float = 1.0
) -> dict:
    """Relative discrete L2 error of the displacement field vs an exact field."""
    w = nodal_weights(mesh.nodes, mesh.tris) * thickness
    uh = np.stack([u[0::2], u[1::2]], axis=1)
    ue = exact(mesh.nodes)
    diff = uh - ue
    err = float(np.sqrt(np.sum(w * np.sum(diff**2, axis=1))))
    ref = float(np.sqrt(np.sum(w * np.sum(ue**2, axis=1))))
    return {
        "abs": err,
        "ref": ref,
        "rel": err / ref if ref > 0 else float("nan"),
        "max_nodal": float(np.max(np.linalg.norm(diff, axis=1))),
    }

def ki_from_jintegral(
    mesh: TriMesh,
    u: np.ndarray,
    stress: np.ndarray,
    a: float,
    material: Material2D,
    *,
    r_in_fraction: float = 0.15,
    r_out_fraction: float = 0.5,
    tip: tuple[float, float] | None = None,
) -> tuple[float, dict]:
    """Mode-I SIF from the domain form of the J-integral.

    ``J = int_A (sigma_ij du_j/dx1 - W delta_1i) dq/dx_i dA`` with a nodal weight
    ``q`` that is 1 on ``r <= r_in`` and 0 beyond ``r_out`` (so no boundary flux
    is picked up: the crack faces are traction free and the symmetry segment has
    ``sigma_xy = u_y = 0``).  For the quarter (half-contour) model the contour
    carries half of the total energy release, hence ``K_I = sqrt(2 J E')``.

    The J-integral is an energy functional of the whole annulus rather than a
    pointwise near-tip fit, which makes it markedly more robust on meshes that
    are not graded in a controlled way.
    """
    nodes = mesh.nodes
    tris = mesh.tris
    tip = np.array([a, 0.0]) if tip is None else np.asarray(tip, float)
    r_nodes = np.hypot(nodes[:, 0] - tip[0], nodes[:, 1] - tip[1])
    r_in = r_in_fraction * a
    r_out = r_out_fraction * a
    if r_out <= r_in:
        raise ValueError("r_out must exceed r_in")
    q = np.clip((r_out - r_nodes) / (r_out - r_in), 0.0, 1.0)

    x = nodes[tris, 0]
    y = nodes[tris, 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    area2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    area = 0.5 * area2
    inv2a = 1.0 / area2

    ue = u[np.stack([2 * tris, 2 * tris + 1], axis=2).reshape(len(tris), 6)]
    ux = ue[:, 0::2]
    uy = ue[:, 1::2]
    qe = q[tris]

    dux_dx = np.einsum("mi,mi->m", b, ux) * inv2a
    dux_dy = np.einsum("mi,mi->m", c, ux) * inv2a
    duy_dx = np.einsum("mi,mi->m", b, uy) * inv2a
    dq_dx = np.einsum("mi,mi->m", b, qe) * inv2a
    dq_dy = np.einsum("mi,mi->m", c, qe) * inv2a

    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    W = 0.5 * np.einsum("mi,ij,mj->m", stress, material.compliance, stress)
    integrand = (sxx * dux_dx + sxy * duy_dx - W) * dq_dx + (sxy * dux_dx + syy * duy_dx) * dq_dy
    J_half = float(np.sum(integrand * area))
    K = float(np.sqrt(max(2.0 * J_half * material.E_eff, 0.0)))
    return K, {
        "J_half": J_half,
        "G_total": 2.0 * J_half,
        "r_in": r_in,
        "r_out": r_out,
        "n_elements_in_domain": int(np.sum((r_nodes[tris].max(axis=1) <= r_out))),
    }
