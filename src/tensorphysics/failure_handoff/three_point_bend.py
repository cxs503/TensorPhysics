"""Notched three-point-bend FEM model that feeds the fracture-face handoff.

The model is deliberately small and closed-form-checkable: a prismatic beam meshed
with Hex8 solid elements from :mod:`tensorphysics.solid_fem`, a mid-span notch
created by **element removal** (the notch surfaces are therefore mesh-aligned,
which is what a crack plane needs), two support rollers at the beam ends, and a
displacement-controlled load patch at mid-span.

Two uses:

* :func:`solve_bend` returns the linear-elastic solution, element-average stresses
  and the total support reaction — the input of the fracture-face extraction;
* :func:`verify_against_beam_theory` compares the *plain* (un-notched) beam
  deflection and bottom-fibre stress against Euler-Bernoulli / Timoshenko beam
  theory, so the FEM field used by the handoff is not taken on faith.

Axis convention: ``x`` along the span, ``y`` up, ``z`` through the thickness.
Bending is about ``z``; the notch is cut from the bottom face at mid-span.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..solid_fem.element import element_kinematics
from ..solid_fem.material import isotropic_elasticity
from ..solid_fem.mesh import SolidMesh, box_mesh
from ..solid_fem.solver import ElasticSolution, solve_elastic

__all__ = [
    "BeamSpec",
    "BendModel",
    "build_bend_model",
    "solve_bend",
    "element_average_stress",
    "second_moment_of_area",
    "euler_bernoulli_deflection",
    "timoshenko_deflection",
    "bending_stress_bottom_fibre",
    "verify_against_beam_theory",
]


@dataclass(frozen=True)
class BeamSpec:
    """Geometry and elasticity of a three-point-bend specimen."""

    length_m: float = 0.40
    height_m: float = 0.06
    thickness_m: float = 0.02
    notch_depth_m: float = 0.0
    notch_gap_m: float = 0.0
    E_pa: float = 30.0e9
    nu: float = 0.20

    def __post_init__(self) -> None:
        for name in ("length_m", "height_m", "thickness_m", "E_pa"):
            value = getattr(self, name)
            if not (value > 0.0):
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.notch_depth_m < self.height_m:
            raise ValueError("notch_depth_m must be in [0, height_m)")
        if self.notch_gap_m < 0.0:
            raise ValueError("notch_gap_m must be non-negative")

    @property
    def G_pa(self) -> float:
        return self.E_pa / (2.0 * (1.0 + self.nu))

    @property
    def I_m4(self) -> float:
        return second_moment_of_area(self.thickness_m, self.height_m)


@dataclass
class BendModel:
    """Meshed bend specimen plus the bookkeeping needed for the crack plane."""

    mesh: SolidMesh
    spec: BeamSpec
    notch_depth_effective_m: float
    dx_m: float
    dy_m: float
    notch_element_count: int
    crack_plane_x_m: float
    notch_gap_effective_m: float
    element_ids_original: np.ndarray  # mesh element e -> original box element id
    #: Mesh *before* the notch was cut (needed to add elements back, e.g. crack advance).
    base_mesh: SolidMesh | None = None
    #: Boolean mask over ``base_mesh`` elements: True where the element survived the notch cut.
    notch_mask: np.ndarray | None = None


def second_moment_of_area(width_m: float, height_m: float) -> float:
    """``I = b h^3 / 12`` about the horizontal centroidal axis."""
    return width_m * height_m**3 / 12.0


# ---------------------------------------------------------------------------
# Meshing (with a mesh-aligned notch)
# ---------------------------------------------------------------------------

def _prune(mesh: SolidMesh, keep: np.ndarray) -> tuple[SolidMesh, np.ndarray]:
    """Drop elements and renumber the nodes they still use."""
    conn = mesh.connectivity[keep]
    used = np.unique(conn)
    remap = -np.ones(mesh.n_nodes, dtype=int)
    remap[used] = np.arange(used.size)
    pruned = SolidMesh(
        element="hex8", nodes=mesh.nodes[used], connectivity=remap[conn]
    )
    return pruned, np.nonzero(keep)[0]


def build_bend_model(
    spec: BeamSpec, *, dims: tuple[int, int, int] = (32, 12, 4)
) -> BendModel:
    """Mesh the specimen, cut the notch by element removal, snap to the grid."""
    nx, ny, nz = (int(d) for d in dims)
    dx = spec.length_m / nx
    dy = spec.height_m / ny
    mesh = box_mesh(
        dims=(nx, ny, nz),
        lengths=(spec.length_m, spec.height_m, spec.thickness_m),
        origin=(0.0, 0.0, 0.0),
    )
    # Snap the notch depth / gap onto element boundaries; report the snapped value.
    layers = int(round(spec.notch_depth_m / dy)) if spec.notch_depth_m > 0 else 0
    notch_depth = layers * dy
    if spec.notch_gap_m > 0.0:
        gap = int(round(spec.notch_gap_m / dx)) * dx
    else:
        gap = 0.0
    if spec.notch_gap_m > 0.0 and gap <= 0.0:
        raise ValueError(
            f"requested notch_gap_m={spec.notch_gap_m:.6g} m snapped to zero on a "
            f"dx={dx:.6g} m mesh; a silent plain beam would be tested instead of a "
            "notched specimen (choose notch_gap_m >= 0.5 * dx)"
        )
    if gap > 0.0:
        gap = max(gap, dx)

    keep = np.ones(mesh.n_elem, dtype=bool)
    if layers > 0 and gap > 0.0:
        x_mid = 0.5 * spec.length_m
        for e in range(mesh.n_elem):
            coords = mesh.element_coords(e)
            x_center = float(coords[:, 0].mean())
            y_max = float(coords[:, 1].max())
            in_band = abs(x_center - x_mid) <= 0.5 * gap + 1e-12
            below_root = y_max <= notch_depth + 1e-12
            if in_band and below_root:
                keep[e] = False
    pruned, original_ids = _prune(mesh, keep)

    # The crack plane is the *mesh face* where the notch band starts: the removed
    # elements may snap to a different width than requested, so the plane must be
    # derived from the mesh, never from the requested gap.
    removed = np.nonzero(~keep)[0]
    if removed.size:
        crack_plane_x = float(
            min(mesh.element_coords(int(e))[:, 0].min() for e in removed)
        )
        notch_gap_effective = float(
            max(mesh.element_coords(int(e))[:, 0].max() for e in removed)
        ) - crack_plane_x
    else:
        crack_plane_x = 0.5 * spec.length_m
        notch_gap_effective = 0.0
    return BendModel(
        mesh=pruned,
        spec=spec,
        notch_depth_effective_m=notch_depth,
        dx_m=dx,
        dy_m=dy,
        notch_element_count=int((~keep).sum()),
        notch_gap_effective_m=notch_gap_effective,
        crack_plane_x_m=crack_plane_x,
        element_ids_original=original_ids,
        base_mesh=mesh,
        notch_mask=keep,
    )


# ---------------------------------------------------------------------------
# Boundary conditions and solve
# ---------------------------------------------------------------------------

def _support_dofs(mesh: SolidMesh, spec: BeamSpec) -> tuple[np.ndarray, np.ndarray]:
    """Roller supports on the bottom face at both ends; minimal rigid-body fix."""
    tol = 1e-9 * max(spec.length_m, spec.thickness_m)
    nodes = mesh.nodes
    bottom = np.abs(nodes[:, 1]) <= tol
    left = bottom & (np.abs(nodes[:, 0]) <= tol)
    right = bottom & (np.abs(nodes[:, 0] - spec.length_m) <= tol)
    left_ids = np.nonzero(left)[0]
    right_ids = np.nonzero(right)[0]
    # Vertical restraint at both supports, spanwise restraint at the left roller,
    # out-of-plane restraint along the whole bottom line (2-D bending kinematics).
    support_ids = np.unique(np.concatenate([left_ids, right_ids]))
    dofs = list(3 * support_ids + 1)  # u_y
    dofs += list(3 * left_ids + 0)  # u_x at x = 0
    dofs += list(3 * support_ids + 2)  # u_z
    dofs = np.array(sorted(set(dofs)), dtype=int)
    return dofs, support_ids


def _load_patch_dofs(
    mesh: SolidMesh,
    spec: BeamSpec,
    patch_width_m: float,
    load_offset_m: float = 0.0,
) -> np.ndarray:
    tol = 1e-9 * spec.height_m
    nodes = mesh.nodes
    top = np.abs(nodes[:, 1] - spec.height_m) <= tol
    x_load = 0.5 * spec.length_m + float(load_offset_m)
    mid = np.abs(nodes[:, 0] - x_load) <= 0.5 * patch_width_m + 1e-12
    ids = np.nonzero(top & mid)[0]
    if ids.size == 0:
        raise ValueError("the load patch selected no nodes; widen patch_width_m")
    return 3 * ids + 1  # u_y


@dataclass
class BendSolution:
    """Everything the extraction step and the verification step need."""

    model: BendModel
    C: np.ndarray
    solution: ElasticSolution
    sigma_element: np.ndarray
    load_pa_total_n: float
    applied_displacement_m: float
    midspan_deflection_m: float
    strain_energy_j: float
    support_reaction_n: float


def solve_bend(
    model: BendModel,
    *,
    applied_displacement_m: float,
    patch_width_m: float | None = None,
    formulation: str = "full",
    load_offset_m: float = 0.0,
) -> BendSolution:
    """Displacement-controlled linear-elastic solve of the notched specimen.

    The total transverse load is recovered from the support reactions, so the
    "load level" reported downstream is a measured quantity, not an input.
    """
    mesh = model.mesh
    spec = model.spec
    C = isotropic_elasticity(spec.E_pa, spec.nu)
    fixed_dofs, support_ids = _support_dofs(mesh, spec)
    patch = patch_width_m if patch_width_m is not None else 2.0 * model.dx_m
    load_dofs = _load_patch_dofs(mesh, spec, patch, load_offset_m)
    # A dof must never be both supported and loaded; resolving the two sets through
    # one dof->value map keeps the dof order and the value order in lockstep.
    prescribed: dict[int, float] = {int(d): 0.0 for d in fixed_dofs}
    for dof in load_dofs:
        if int(dof) in prescribed:
            raise RuntimeError(f"dof {int(dof)} is both supported and loaded")
        prescribed[int(dof)] = -abs(applied_displacement_m)
    all_dofs = np.array(sorted(prescribed), dtype=int)
    all_values = np.array([prescribed[int(d)] for d in all_dofs], dtype=float)
    solution = solve_elastic(
        mesh, C, all_dofs, fixed_values=all_values, formulation=formulation
    )
    sigma = element_average_stress(mesh, solution.u, C, formulation=formulation)

    u = solution.u.reshape(-1, 3)
    tol = 1e-9 * spec.height_m
    top = np.abs(mesh.nodes[:, 1] - spec.height_m) <= tol
    mid = np.abs(mesh.nodes[:, 0] - (0.5 * spec.length_m + float(load_offset_m))) <= 0.5 * patch + 1e-12
    midspan_deflection = float(-np.mean(u[top & mid, 1]))
    # Support reactions are +y for a downward applied displacement, so their sum is
    # the magnitude of the transverse load carried by the specimen.
    reaction = float(np.sum(solution.reactions[3 * support_ids + 1]))
    return BendSolution(
        model=model,
        C=C,
        solution=solution,
        sigma_element=sigma,
        load_pa_total_n=reaction,
        applied_displacement_m=abs(applied_displacement_m),
        midspan_deflection_m=midspan_deflection,
        strain_energy_j=float(solution.energy),
        support_reaction_n=reaction,
    )


def element_average_stress(
    mesh: SolidMesh, u: np.ndarray, C: np.ndarray, *, formulation: str = "full"
) -> np.ndarray:
    """Element-average Cauchy stress, Voigt ``[xx, yy, zz, xy, yz, zx]``, in Pa."""
    n_elem = mesh.n_elem
    out = np.zeros((n_elem, 6))
    for e in range(n_elem):
        coords = mesh.element_coords(e)
        kin = element_kinematics("hex8", coords, None, formulation)
        u_e = u[mesh.element_dofs(e)]
        stresses = np.array([C @ (kin.B[g] @ u_e) for g in range(kin.integration_points)])
        out[e] = stresses.mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# Beam-theory baselines
# ---------------------------------------------------------------------------

def euler_bernoulli_deflection(P: float, L: float, E: float, I: float) -> float:
    """Centre deflection of a simply supported beam, central point load."""
    return P * L**3 / (48.0 * E * I)


def timoshenko_deflection(
    P: float, L: float, E: float, G: float, I: float, A: float, k_shear: float = 5.0 / 6.0
) -> float:
    """Centre deflection including transverse shear (rectangular section)."""
    bending = P * L**3 / (48.0 * E * I)
    shear = P * L / (4.0 * G * k_shear * A)
    return bending + shear


def bending_stress_bottom_fibre(P: float, L: float, h: float, t: float) -> float:
    """``sigma = M c / I`` at mid-span bottom fibre, ``M = P L / 4`` (tension +)."""
    return bending_stress_at_height(P, L, h, t, 0.0)


def bending_stress_at_height(
    P: float, L: float, h: float, t: float, y_above_bottom_m: float
) -> float:
    """Beam-theory ``sigma_xx`` at height ``y`` above the bottom fibre (tension +).

    ``sigma(y) = M (h/2 - y) / I`` with ``M = P L / 4`` at mid-span.
    """
    M = P * L / 4.0
    return M * (h / 2.0 - y_above_bottom_m) / second_moment_of_area(t, h)


def verify_against_beam_theory(
    spec: BeamSpec,
    *,
    dims: tuple[int, int, int] = (48, 6, 3),
    applied_displacement_m: float = 2.0e-5,
    patch_width_m: float | None = None,
) -> dict:
    """Plain (un-notched) beam: FEM vs Euler-Bernoulli and Timoshenko.

    The FEM stress used downstream by the fracture-face extraction is only as good
    as the displacement field it comes from, so this baseline is the gate that
    certifies the field: mid-span deflection and bottom-layer bending stress are
    compared with beam theory, and the two theory variants are reported separately
    (the Euler-Bernoulli value neglects transverse shear; at ``L/h = 20`` the shear
    contribution is well under 1 %, so the reported error is not an artefact of the
    baseline choice).
    """
    plain = BeamSpec(
        length_m=spec.length_m,
        height_m=spec.height_m,
        thickness_m=spec.thickness_m,
        notch_depth_m=0.0,
        notch_gap_m=0.0,
        E_pa=spec.E_pa,
        nu=spec.nu,
    )
    model = build_bend_model(plain, dims=dims)
    sol = solve_bend(
        model, applied_displacement_m=applied_displacement_m, patch_width_m=patch_width_m
    )
    P = abs(sol.load_pa_total_n)
    A = spec.height_m * spec.thickness_m
    delta_eb = euler_bernoulli_deflection(P, spec.length_m, spec.E_pa, spec.I_m4)
    delta_ti = timoshenko_deflection(P, spec.length_m, spec.E_pa, spec.G_pa, spec.I_m4, A)

    # Deflection is measured on the *neutral axis* at mid-span, not on the loaded
    # top surface: the imposed value there is a boundary condition, so using it
    # would compare the answer with itself.
    u = sol.solution.u.reshape(-1, 3)
    nodes = model.mesh.nodes
    tol = 1e-9 * spec.height_m
    midspan = np.abs(nodes[:, 0] - 0.5 * spec.length_m) <= 1e-9
    neutral = np.abs(nodes[:, 1] - 0.5 * spec.height_m) <= tol
    if not np.any(midspan & neutral):
        raise RuntimeError("no mid-span node on the neutral axis for the deflection check")
    fem_delta = float(abs(np.mean(u[midspan & neutral, 1])))

    # Bottom element layer at mid-span: element-average stress = bending stress at
    # the element centroid height (linear field → exact), one element each side.
    y_centroid = 0.5 * model.dy_m
    midspan_layer = []
    for e in range(model.mesh.n_elem):
        coords = model.mesh.element_coords(e)
        x_center = 0.5 * (coords[:, 0].min() + coords[:, 0].max())
        if abs(x_center - 0.5 * spec.length_m) <= 0.5 * model.dx_m + 1e-12 and coords[:, 1].min() <= 1e-12:
            midspan_layer.append(e)
    if not midspan_layer:
        raise RuntimeError("no bottom-layer element found at mid-span")
    sigma_fem = float(np.mean([abs(sol.sigma_element[e, 0]) for e in midspan_layer]))
    sigma_an = abs(
        bending_stress_at_height(P, spec.length_m, spec.height_m, spec.thickness_m, y_centroid)
    )

    return {
        "spec": {
            "length_m": spec.length_m,
            "height_m": spec.height_m,
            "thickness_m": spec.thickness_m,
            "E_pa": spec.E_pa,
            "nu": spec.nu,
            "dims": list(dims),
            "L_over_h": spec.length_m / spec.height_m,
        },
        "mesh": {
            "element_count": int(model.mesh.n_elem),
            "node_count": int(model.mesh.n_nodes),
            "dof_count": int(model.mesh.n_dof),
        },
        "applied_displacement_m": sol.applied_displacement_m,
        "total_load_n": P,
        "stiffness_n_per_m": P / fem_delta,
        "deflection": {
            "fem_m": fem_delta,
            "euler_bernoulli_m": delta_eb,
            "timoshenko_m": delta_ti,
            "relative_error_vs_euler_bernoulli": abs(fem_delta - delta_eb) / abs(delta_eb),
            "relative_error_vs_timoshenko": abs(fem_delta - delta_ti) / abs(delta_ti),
            "shear_contribution_ratio": abs(delta_ti - delta_eb) / abs(delta_eb),
        },
        "bottom_fibre_stress": {
            "fem_pa": sigma_fem,
            "analytic_pa": sigma_an,
            "fem_height_m": y_centroid,
            "element_count_used": len(midspan_layer),
            "relative_error": abs(sigma_fem - sigma_an) / abs(sigma_an),
        },
        "strain_energy_j": sol.strain_energy_j,
        # External work is P * u at the *loaded* dofs: the imposed patch
        # displacement, not the neutral-axis deflection.
        "half_P_delta_applied_j": 0.5 * P * sol.applied_displacement_m,
        "strain_energy_relative_error": abs(
            sol.strain_energy_j - 0.5 * P * sol.applied_displacement_m
        )
        / max(0.5 * P * sol.applied_displacement_m, 1e-30),
        "half_P_delta_neutral_axis_j": 0.5 * P * fem_delta,
    }
