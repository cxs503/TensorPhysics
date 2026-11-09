"""Verification problems for the 3-D solid element kernel.

Contents
--------
``patch_test``
    Constant-strain / constant-stress patch test on (optionally distorted)
    Hex8 and Hex20 patches; returns the exactness errors.
``thick_cylinder``
    Thick-walled cylinder under internal pressure: mesh generation, plane
    strain boundary conditions, consistent pressure loading, and the metrics
    compared against the Lame solution.
``lame_*``
    Closed-form Lame stresses, radial displacement, axial stress (plane
    strain) and the elastic-perfectly-plastic limit pressure.
``hourglass_study``
    Rank / spurious-mode counts and the stabilisation energy fraction.
``uniaxial_plasticity``
    Single-element uniaxial tension vs the analytic bilinear curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse.linalg import eigsh

from .element import element_kinematics, element_stiffness
from .material import J2Material, PlasticState, isotropic_elasticity
from .mesh import SolidMesh, annulus_mesh, box_mesh, pressure_load
from .solver import solve_elastic, solve_plastic
from .shape import default_rule

__all__ = [
    "patch_test",
    "lame_stresses",
    "lame_radial_displacement",
    "lame_sigma_z",
    "lame_limit_pressure",
    "thick_cylinder",
    "CylinderSetup",
    "hourglass_rank",
    "hourglass_study",
    "uniaxial_plasticity",
    "strain_voigt_linear",
]


def strain_voigt_linear(A: np.ndarray) -> np.ndarray:
    """Voigt strain of the linear displacement field ``u = A x``."""
    return np.array(
        [
            A[0, 0],
            A[1, 1],
            A[2, 2],
            A[0, 1] + A[1, 0],
            A[1, 2] + A[2, 1],
            A[0, 2] + A[2, 0],
        ]
    )


# ---------------------------------------------------------------------------
# Patch test
# ---------------------------------------------------------------------------


def patch_test(
    element: str = "hex8",
    dims: Sequence[int] = (2, 2, 2),
    lengths: Sequence[float] = (1.0, 1.0, 1.0),
    distort: float = 0.0,
    seed: int = 0,
    E: float = 210.0e9,
    nu: float = 0.3,
    A: np.ndarray | None = None,
    formulation: str = "full",
) -> dict:
    """Constant-strain patch test.

    The linear displacement field ``u = A x`` is prescribed on every boundary
    node; interior nodes must reproduce it exactly and every Gauss point must
    carry the constant stress ``C : eps(A)``.
    """
    mesh = box_mesh(dims=dims, lengths=lengths, element=element, distort=distort, seed=seed)
    A = (
        np.array([[1.0e-3, 4.0e-4, -3.0e-4],
                  [2.0e-4, -6.0e-4, 5.0e-4],
                  [-1.0e-4, 3.0e-4, 8.0e-4]])
        if A is None
        else np.asarray(A, dtype=float)
    )
    eps_exact = strain_voigt_linear(A)
    C = isotropic_elasticity(E, nu)
    sig_exact = C @ eps_exact

    lo = np.min(mesh.nodes, axis=0)
    hi = np.max(mesh.nodes, axis=0)
    tol = 1e-10 * np.max(np.abs(hi - lo))
    on_boundary = np.zeros(mesh.n_nodes, dtype=bool)
    for d in range(3):
        on_boundary |= np.abs(mesh.nodes[:, d] - lo[d]) < tol
        on_boundary |= np.abs(mesh.nodes[:, d] - hi[d]) < tol
    boundary = np.nonzero(on_boundary)[0]
    interior = np.nonzero(~on_boundary)[0]

    u_exact = mesh.nodes @ A.T
    fixed = mesh.dofs_of_nodes(boundary)
    sol = solve_elastic(
        mesh, C, fixed_dofs=fixed, fixed_values=u_exact[boundary].ravel(),
        formulation=formulation,
    )
    span = np.max(np.abs(hi - lo))
    scale_u = np.max(np.abs(u_exact)) * span
    udisp_err = float(np.max(np.abs(sol.u - u_exact.ravel())) / scale_u)

    # stress error at every Gauss point
    err_s = 0.0
    scale_s = float(np.max(np.abs(sig_exact)))
    for e in range(mesh.n_elem):
        coords = mesh.element_coords(e)
        kin = element_kinematics(mesh.element, coords, None, formulation)
        u_e = sol.u[mesh.element_dofs(e)]
        for g in range(kin.integration_points):
            sig = C @ (kin.B[g] @ u_e)
            err_s = max(err_s, float(np.max(np.abs(sig - sig_exact))) / scale_s)

    return {
        "element": mesh.element,
        "dims": list(dims),
        "distort": float(distort),
        "formulation": formulation,
        "n_nodes": mesh.n_nodes,
        "n_elem": mesh.n_elem,
        "n_interior_nodes": int(interior.size),
        "displacement_error": udisp_err,
        "stress_error": err_s,
        "eps_exact": eps_exact.tolist(),
        "sig_exact": sig_exact.tolist(),
    }


# ---------------------------------------------------------------------------
# Lame thick-walled cylinder
# ---------------------------------------------------------------------------


def lame_stresses(r, a: float, b: float, p: float):
    """Lame radial / hoop stresses (plane strain, internal pressure ``p``)."""
    r = np.asarray(r, dtype=float)
    K = p * a**2 / (b**2 - a**2)
    return K * (1.0 - b**2 / r**2), K * (1.0 + b**2 / r**2)


def lame_sigma_z(sigma_r, sigma_theta, nu: float):
    """Plane-strain axial stress ``nu (sigma_r + sigma_theta)``."""
    return nu * (np.asarray(sigma_r) + np.asarray(sigma_theta))


def lame_radial_displacement(r, a: float, b: float, p: float, E: float, nu: float):
    """Plane-strain radial displacement of the Lame cylinder."""
    r = np.asarray(r, dtype=float)
    C = (1.0 + nu) * p * a**2 / (E * (b**2 - a**2))
    return C * ((1.0 - 2.0 * nu) * r + b**2 / r)


def lame_limit_pressure(a: float, b: float, sigma_y: float) -> float:
    """Elastic-perfectly-plastic limit pressure, plane strain, von Mises.

    ``p_L = (2 sigma_y / sqrt(3)) ln(b/a)``  (from ``sigma_theta - sigma_r =
    2 sigma_y / sqrt(3)`` in the fully plastic state with ``sigma_z =
    (sigma_r + sigma_theta)/2``).
    """
    return 2.0 * sigma_y / np.sqrt(3.0) * np.log(b / a)


@dataclass
class CylinderSetup:
    mesh: SolidMesh
    a: float
    b: float
    p: float
    E: float
    nu: float
    formulation: str
    rule: str
    fixed_dofs: np.ndarray
    loads: np.ndarray
    u: np.ndarray
    energy: float

    def theta_ref(self) -> float:
        """Peak hoop stress of the Lame solution (error normalisation)."""
        return self.p * (self.b**2 + self.a**2) / (self.b**2 - self.a**2)

    # -- metrics -------------------------------------------------------
    def inner_displacement_error(self) -> dict:
        tol = 1e-9 * self.b
        r = np.hypot(self.mesh.nodes[:, 0], self.mesh.nodes[:, 1])
        inner = np.abs(r - self.a) < tol
        d = self.mesh.nodes[inner]
        u_xy = self.u.reshape(-1, 3)[inner][:, :2]
        ur_fe = (u_xy * (d[:, :2] / r[inner, None])).sum(axis=1)
        ur_ex = lame_radial_displacement(self.a, self.a, self.b, self.p, self.E, self.nu)
        return {
            "ur_fe_mean": float(np.mean(ur_fe)),
            "ur_exact": float(ur_ex),
            "rel_error": float(abs(np.mean(ur_fe) - ur_ex) / abs(ur_ex)),
            "ur_scatter": float(np.std(ur_fe)),
        }

    def gauss_field_errors(self) -> dict:
        """Radial/hoop stress and volumetric strain errors at the Gauss points."""
        from .element import element_kinematics
        from .material import isotropic_elasticity

        C = isotropic_elasticity(self.E, self.nu)
        sr_err, st_err = [], []
        ev_fe, ev_ex = [], []
        for e in range(self.mesh.n_elem):
            coords = self.mesh.element_coords(e)
            kin = element_kinematics(self.mesh.element, coords, self.rule, self.formulation)
            u_e = self.u[self.mesh.element_dofs(e)]
            for g in range(kin.integration_points):
                from .shape import shape as _shape

                N, _ = _shape(self.mesh.element, *kin.points[g])
                x = N @ coords
                r = float(np.hypot(x[0], x[1]))
                er, et = x[0] / r, x[1] / r  # radial unit vector
                sig = C @ (kin.B[g] @ u_e)
                # rotate the Cartesian stress into the (r, theta, z) frame
                T = np.array(
                    [
                        [er * er, et * et, 0.0, 2 * er * et, 0.0, 0.0],
                        [et * et, er * er, 0.0, -2 * er * et, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                )
                sig_rt = T @ sig
                sr_ex, st_ex = lame_stresses(r, self.a, self.b, self.p)
                sr_err.append(abs(sig_rt[0] - sr_ex) / self.p)
                st_err.append(abs(sig_rt[1] - st_ex) / self.theta_ref())
                strain = kin.B[g] @ u_e
                ev_fe.append(float(strain[0] + strain[1] + strain[2]))
                ev_ex.append(
                    float(
                        (1.0 - 2.0 * self.nu) * (1.0 + self.nu) / self.E
                        * (sr_ex + st_ex)
                    )
                )
        ev_fe = np.array(ev_fe)
        ev_ex = np.array(ev_ex)
        return {
            "sigma_r_max_rel_error": float(np.max(sr_err)),
            "sigma_theta_max_rel_error": float(np.max(st_err)),
            "eps_v_fe_mean": float(np.mean(ev_fe)),
            "eps_v_exact": float(np.mean(ev_ex)),
            "eps_v_rel_error": float(
                abs(np.mean(ev_fe) - np.mean(ev_ex)) / abs(np.mean(ev_ex))
            ),
            "eps_v_pointwise_max_rel_error": float(
                np.max(np.abs(ev_fe - ev_ex)) / abs(np.mean(ev_ex))
            ),
        }


def thick_cylinder(
    a: float = 1.0,
    b: float = 2.0,
    p: float = 50.0e6,
    E: float = 210.0e9,
    nu: float = 0.3,
    n_radial: int = 4,
    n_circ: int = 8,
    element: str = "hex8",
    formulation: str = "full",
    rule: str | None = None,
    length: float = 1.0,
    theta_span: float = np.pi / 2.0,
    hourglass: float = 0.0,
) -> CylinderSetup:
    """Quarter annulus in plane strain under internal pressure.

    Boundary conditions: ``u_z = 0`` everywhere (plane strain), symmetry
    ``u_y = 0`` on the ``theta = 0`` face and ``u_x = 0`` on the
    ``theta = theta_span`` face (``theta_span = pi/2``).
    """
    mesh = annulus_mesh(
        a=a, b=b, length=length, n_radial=n_radial, n_circ=n_circ, n_axial=1,
        element=element, theta_span=theta_span,
    )
    tol = 1e-9 * b
    rad = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    theta = np.arctan2(mesh.nodes[:, 1], mesh.nodes[:, 0])

    fixed = [mesh.dofs_of_nodes(np.arange(mesh.n_nodes), component=2)]
    on_theta0 = np.abs(theta) < tol
    fixed.append(mesh.dofs_of_nodes(np.nonzero(on_theta0)[0], component=1))
    if abs(theta_span - np.pi / 2.0) < 1e-9:
        on_theta90 = np.abs(theta - np.pi / 2.0) < tol
        fixed.append(mesh.dofs_of_nodes(np.nonzero(on_theta90)[0], component=0))
    else:  # keep one node pinned tangentially for other spans
        fixed.append(mesh.dofs_of_nodes(np.nonzero(on_theta0 & (rad > b - tol))[0], component=0))
    fixed_dofs = np.unique(np.concatenate(fixed))

    def inner_face(centroid, normal, corners):
        r = np.hypot(corners[:, 0], corners[:, 1])
        return bool(np.max(r) <= a + 1e-9 * b)

    loads = pressure_load(mesh, p, inner_face)
    C = isotropic_elasticity(E, nu)
    sol = solve_elastic(
        mesh, C, fixed_dofs=fixed_dofs, loads=loads, formulation=formulation,
        rule=rule, hourglass=hourglass, shear_modulus=E / (2.0 * (1.0 + nu)),
    )
    return CylinderSetup(
        mesh=mesh, a=a, b=b, p=p, E=E, nu=nu,
        formulation=formulation, rule=rule or default_rule(mesh.element),
        fixed_dofs=fixed_dofs, loads=loads, u=sol.u, energy=sol.energy,
    )


# ---------------------------------------------------------------------------
# Hourglass control
# ---------------------------------------------------------------------------


def hourglass_rank(
    alpha_hg: float = 0.0, E: float = 1000.0, nu: float = 0.3, distort: float = 0.0, seed: int = 1
) -> dict:
    """Zero-energy (rigid plus spurious) mode count of a single Hex8 element."""
    mesh = box_mesh(dims=(1, 1, 1), lengths=(1.0, 1.0, 1.0), element="hex8",
                    distort=distort, seed=seed)
    C = isotropic_elasticity(E, nu)
    k = element_stiffness(
        "hex8", mesh.element_coords(0), C, rule="1pt", formulation="full",
        hourglass=alpha_hg, shear_modulus=E / (2 * (1 + nu)),
    )
    w = np.linalg.eigvalsh(0.5 * (k + k.T))
    scale = max(abs(w).max(), 1e-30)
    zero = int(np.sum(np.abs(w) < 1e-8 * scale))
    return {"alpha_hg": float(alpha_hg), "zero_modes": zero, "eigenvalues_min": float(w.min()),
            "eigenvalues_max": float(w.max())}


def hourglass_study(
    alpha_hg: float = 1.0,
    E: float = 210.0e9,
    nu: float = 0.3,
    n_radial: int = 4,
    n_circ: int = 8,
    element: str = "hex8",
) -> dict:
    """Hourglass energy fraction of a well-posed problem (pressurised cylinder).

    The cylinder field is smooth (``~1/r^2``), so a properly stabilised 1-point
    mesh should keep the spurious (hourglass) energy fraction small; the
    unstabilised mesh is rank deficient.
    """
    setup = thick_cylinder(
        a=1.0, b=2.0, p=50e6, E=E, nu=nu, n_radial=n_radial, n_circ=n_circ,
        element="hex8", formulation="full", rule="1pt",
    )
    C = isotropic_elasticity(E, nu)
    from .solver import hourglass_energy_ratio

    e_hg, e_tot = hourglass_energy_ratio(
        setup.mesh, setup.u, C, hourglass=alpha_hg, shear_modulus=E / (2 * (1 + nu)),
        rule="1pt", formulation="full",
    )
    ur = setup.inner_displacement_error()
    return {
        "alpha_hg": float(alpha_hg),
        "n_radial": n_radial,
        "n_circ": n_circ,
        "element": element,
        "hourglass_energy": float(e_hg),
        "total_energy": float(e_tot),
        "hourglass_fraction": float(abs(e_hg) / e_tot) if e_tot else float("inf"),
        "inner_displacement_rel_error": ur["rel_error"],
    }


# ---------------------------------------------------------------------------
# J2 plasticity: uniaxial tension
# ---------------------------------------------------------------------------


def uniaxial_plasticity(
    E: float = 210.0e9,
    nu: float = 0.3,
    sigma_y0: float = 250.0e6,
    hardening: float = 20.0e9,
    n_steps: int = 40,
    max_strain: float = 0.002,
    element: str = "hex8",
    dims: Sequence[int] = (1, 1, 1),
    rule: str | None = None,
    formulation: str = "full",
) -> dict:
    """Uniaxial tension of a unit cube, compared with the analytic curve.

    Loading: prescribed ``u_x`` on the ``x = L`` face, ``u_x = 0`` on ``x = 0``,
    rollers on the other faces (lateral faces traction free) and one node pinned
    laterally -> homogeneous uniaxial stress state.
    """
    mesh = box_mesh(dims=dims, lengths=(1.0, 1.0, 1.0), element=element)
    tol = 1e-9
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    z = mesh.nodes[:, 2]
    left = np.nonzero(np.abs(x) < tol)[0]
    right = np.nonzero(np.abs(x - 1.0) < tol)[0]
    # rollers: lateral displacement of the symmetry planes y=0, z=0 is zero
    fixed = [mesh.dofs_of_nodes(left, component=0)]
    fixed.append(mesh.dofs_of_nodes(right, component=0))
    fixed.append(mesh.dofs_of_nodes(np.nonzero(np.abs(y) < tol)[0], component=1))
    fixed.append(mesh.dofs_of_nodes(np.nonzero(np.abs(z) < tol)[0], component=2))
    fixed_dofs = np.unique(np.concatenate(fixed))

    # displacement control: prescribe u_x = max_strain on the loaded face
    prescribed = np.zeros(mesh.n_dof)
    prescribed[mesh.dofs_of_nodes(right, component=0)] = max_strain
    material = J2Material(E=E, nu=nu, sigma_y0=sigma_y0, hardening=hardening)
    sol = solve_plastic(
        mesh, material, fixed_dofs=fixed_dofs, load_vector=np.zeros(mesh.n_dof),
        fixed_values=prescribed[fixed_dofs], n_steps=n_steps, rule=rule,
        formulation=formulation, tol=1e-10,
        reaction_dofs=mesh.dofs_of_nodes(right, component=0),
    )
    # axial stress = total reaction on the loaded face / area
    from .solver import gauss_stresses_plastic

    gs = gauss_stresses_plastic(
        mesh, sol.u, material, sol.states,
        rule=rule or default_rule(mesh.element), formulation=formulation,
    )
    sig_xx = float(np.mean([s[:, 0].mean() for _pts, s in gs]))
    eps_xx = float(np.mean(sol.u[mesh.dofs_of_nodes(right, component=0)]))  # L = 1

    # full curve from a 1-element strain-driven sweep
    strains = np.linspace(0.0, max_strain, 201)
    from .material import uniaxial_tension_analytic

    sig_ref = uniaxial_tension_analytic(strains, E, sigma_y0, hardening)
    sig_at = float(np.interp(eps_xx, strains, sig_ref))
    alpha_max = max(st.alpha for el in sol.states for st in el)
    area = 1.0  # unit cube: lateral faces are traction free
    curve = [
        {
            "eps_xx": float(h["load_factor"]) * max_strain,
            "sigma_xx": float(h["reaction_sum"]) / area,
        }
        for h in sol.history
        if h.get("reaction_sum") is not None
    ]
    return {
        "curve": curve,
        "element": mesh.element,
        "eps_xx": eps_xx,
        "sigma_xx_fe": sig_xx,
        "sigma_xx_analytic": sig_at,
        "rel_error": float(abs(sig_xx - sig_at) / sig_at),
        "alpha_max": float(alpha_max),
        "converged": sol.converged,
        "iterations": [h["iterations"] for h in sol.history],
        "eps_yield": sigma_y0 / E,
        "dims": list(dims),
    }

# ---------------------------------------------------------------------------
# Cantilever bending (hourglass / bending accuracy study)
# ---------------------------------------------------------------------------


def timoshenko_tip_deflection(P: float, L: float, b: float, h: float, E: float, nu: float) -> float:
    """Timoshenko cantilever tip deflection (end load ``P``, rectangular section)."""
    G = E / (2.0 * (1.0 + nu))
    I = b * h**3 / 12.0
    A = b * h
    k_shear = 5.0 / 6.0
    return P * L**3 / (3.0 * E * I) + P * L / (k_shear * G * A)


def cantilever_tip_deflection(
    L: float = 10.0,
    b: float = 1.0,
    h: float = 1.0,
    P: float = 1.0e5,
    E: float = 210.0e9,
    nu: float = 0.3,
    n_length: int = 8,
    n_thick: int = 1,
    n_width: int = 1,
    element: str = "hex8",
    rule: str | None = None,
    formulation: str = "full",
    hourglass: float = 0.0,
) -> dict:
    """Cantilever with an end shear load: bending accuracy / spurious modes.

    The shear load is spread over the end face so that no local pressure
    singularity is introduced; the tip deflection is compared with the
    Timoshenko closed form.
    """
    mesh = box_mesh(
        dims=(n_length, n_thick, n_width), lengths=(L, h, b), element=element
    )
    tol = 1e-9 * max(L, h, b)
    x = mesh.nodes[:, 0]
    clamp = np.nonzero(np.abs(x) < tol)[0]
    fixed_dofs = mesh.dofs_of_nodes(clamp)

    tip = np.nonzero(np.abs(x - L) < tol)[0]
    # only the face with y in [0, h] (thick side) is loaded; every node on the
    # tip face shares the load
    load = np.zeros(mesh.n_dof)
    load[mesh.dofs_of_nodes(tip, component=1)] = P / tip.size

    C = isotropic_elasticity(E, nu)
    sol = solve_elastic(
        mesh, C, fixed_dofs=fixed_dofs, loads=load, rule=rule,
        formulation=formulation, hourglass=hourglass,
        shear_modulus=E / (2.0 * (1.0 + nu)),
    )
    uy_tip = float(np.mean(sol.u[mesh.dofs_of_nodes(tip, component=1)]))
    ref = timoshenko_tip_deflection(P, L, b, h, E, nu)
    hg_energy = None
    total_energy = None
    if hourglass:
        from .solver import hourglass_energy_ratio

        hg_energy, total_energy = hourglass_energy_ratio(
            mesh, sol.u, C, hourglass=hourglass, shear_modulus=E / (2.0 * (1.0 + nu)),
            rule=rule, formulation=formulation,
        )
    return {
        "element": mesh.element,
        "n_length": n_length,
        "n_thick": n_thick,
        "n_width": n_width,
        "rule": rule or default_rule(mesh.element),
        "formulation": formulation,
        "hourglass": float(hourglass),
        "uy_tip_fe": uy_tip,
        "uy_tip_timoshenko": ref,
        "rel_error": float(abs(uy_tip - ref) / ref),
        "hourglass_energy": None if hg_energy is None else float(hg_energy),
        "total_energy": None if total_energy is None else float(total_energy),
        "hourglass_fraction": (
            None if hg_energy is None or not total_energy else float(abs(hg_energy) / total_energy)
        ),
        "u": sol.u,
        "mesh": mesh,
    }


# ---------------------------------------------------------------------------
# J2 plasticity: thick cylinder limit pressure
# ---------------------------------------------------------------------------


def cylinder_limit_pressure(
    a: float = 1.0,
    b: float = 2.0,
    E: float = 210.0e9,
    nu: float = 0.3,
    sigma_y0: float = 250.0e6,
    hardening: float = 0.0,
    n_radial: int = 4,
    n_circ: int = 8,
    n_steps: int = 40,
    p_max_factor: float = 1.15,
    element: str = "hex8",
    formulation: str = "bbar",
) -> dict:
    """Pressurise the plane-strain cylinder incrementally to plastic collapse.

    The finite element limit pressure is defined as the pressure at which the
    outermost radial layer first goes plastic, i.e. the instant the plastic
    zone traverses the whole wall (the physical definition of the
    elastic-perfectly-plastic limit pressure).
    """
    from .solver import solve_plastic

    p_limit = lame_limit_pressure(a, b, sigma_y0)
    p_max = p_max_factor * p_limit
    mesh = annulus_mesh(
        a=a, b=b, length=1.0, n_radial=n_radial, n_circ=n_circ, n_axial=1,
        element=element, theta_span=np.pi / 2.0,
    )
    tol = 1e-9 * b
    theta = np.arctan2(mesh.nodes[:, 1], mesh.nodes[:, 0])
    fixed = [mesh.dofs_of_nodes(np.arange(mesh.n_nodes), component=2)]
    fixed.append(mesh.dofs_of_nodes(np.nonzero(np.abs(theta) < tol)[0], component=1))
    fixed.append(
        mesh.dofs_of_nodes(np.nonzero(np.abs(theta - np.pi / 2.0) < tol)[0], component=0)
    )
    fixed_dofs = np.unique(np.concatenate(fixed))

    def inner_face(centroid, normal, corners):
        r = np.hypot(corners[:, 0], corners[:, 1])
        return bool(np.max(r) <= a + 1e-9 * b)

    unit_load = pressure_load(mesh, 1.0, inner_face)
    outer = [
        e for e in range(mesh.n_elem)
        if float(np.max(np.hypot(mesh.element_coords(e)[:, 0],
                                mesh.element_coords(e)[:, 1]))) > b - 1e-9 * b
    ]
    material = J2Material(E=E, nu=nu, sigma_y0=sigma_y0, hardening=hardening)

    u = np.zeros(mesh.n_dof)
    states = None
    curve = []
    residual_traces: list[list[float]] = []
    p_traverse = None
    dlam = p_max / n_steps
    lam = 0.0
    for _ in range(n_steps):
        lam_new = lam + dlam
        # NOTE: when continuing from a converged state the load vector passed
        # to the solver is the *total* pressure, not the increment.
        sol = solve_plastic(
            mesh, material, fixed_dofs=fixed_dofs,
            load_vector=unit_load * lam_new, n_steps=1,
            formulation=formulation, tol=1e-9, max_iter=60,
            u0=u, states0=states,
        )
        if not sol.converged:
            break
        u, states = sol.u, [[st.copy() for st in el] for el in sol.states]
        lam = lam_new
        trace = sol.history[0].get("residual_trace") if sol.history else None
        if trace:
            residual_traces.append([float(t) for t in trace])
        alpha_outer = max(st.alpha for e in outer for st in states[e])
        alpha_inner = max(st.alpha for el in states for st in el)
        u_xyz = u.reshape(-1, 3)
        r_nodes = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
        inner_nodes = np.abs(r_nodes - a) < tol
        ur_inner = float(
            np.mean(
                (u_xyz[inner_nodes][:, :2]
                 * (mesh.nodes[inner_nodes][:, :2] / r_nodes[inner_nodes, None])).sum(axis=1)
            )
        )
        curve.append(
            {"pressure": lam, "pressure_over_p_limit": lam / p_limit,
             "alpha_outer": alpha_outer, "alpha_max": alpha_inner,
             "ur_inner": ur_inner, "converged": True}
        )
        _ = ur_inner
        if p_traverse is None and alpha_outer > 1e-8:
            p_traverse = lam
    return {
        "a": a, "b": b, "sigma_y0": sigma_y0, "hardening": hardening,
        "p_limit_analytic": float(p_limit),
        "p_limit_fe": None if p_traverse is None else float(p_traverse),
        "rel_error": (
            None if p_traverse is None else float(abs(p_traverse - p_limit) / p_limit)
        ),
        "load_curve": curve,
        "residual_traces": residual_traces,
        "last_converged_pressure": float(lam),
        "n_radial": n_radial, "n_circ": n_circ,
        "formulation": formulation, "element": element,
    }
