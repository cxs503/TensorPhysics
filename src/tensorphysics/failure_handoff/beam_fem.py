"""Metre-scale beam FEM driver for the ``failure_faces`` hand-off (node p495w4-n07).

Euler–Bernoulli beam finite elements (Hermite cubics, consistent mass) on a
simply supported bar driven in three-point bending by an imposed midspan
displacement, either quasi-statically (static path) or at a dynamic cross-head
rate (Newmark average-acceleration path, ``gamma = 1/2``, ``beta = 1/4`` → no
algorithmic damping, energy conserving for linear systems).

Fracture is detected when the tensile bending stress at the extreme fibre first
reaches ``sigma_t``.  The post-fracture release is integrated on the two free
halves without any fracture sink (the run is energy conserving and its closure
is reported as ``mechanical_closure``); the surface energy ``G_f * A_crack`` is
a consumer-side accounting term that the DEM draws from the *dispersive* part of
the release motion, so that the total momentum is untouched.  Velocities,
displacements, per-element strain energies, the centre-of-mass velocity and the
fluctuation kinetic energy at the kinetic-energy peak form the hand-off state.

The module writes nothing; :mod:`tensorphysics.failure_handoff.beam_cases`
consumes it, builds the packet and runs the conservation audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = [
    "GRAVITY_M_S2",
    "BeamFractureSpec",
    "SIX_CASES",
    "BeamMesh",
    "BeamState",
    "FractureEvent",
    "ReleaseState",
    "assemble_beam",
    "self_weight_vector",
    "nodal_lumped_masses",
    "tension_stress",
    "element_strain_energy",
    "element_velocity",
    "element_displacement",
    "analytic_midspan_deflection",
    "analytic_max_stress",
    "solve_static",
    "newmark_dynamic_load",
    "resolve_fracture",
    "release_analysis",
    "packet_from_fracture",
    "release_sidecar",
]

GRAVITY_M_S2 = 9.81


# ---------------------------------------------------------------------------
# Case specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BeamFractureSpec:
    """Geometry, material, mesh and loading rate of one three-point-bend case."""

    length_m: float
    height_m: float = 0.80
    width_m: float = 0.60
    youngs_modulus_pa: float = 9.0e9
    poisson_ratio: float = 0.33
    density_kg_m3: float = 917.0
    tensile_strength_pa: float = 1.5e6
    fracture_energy_j_m2: float = 5.0
    element_length_m: float = 0.05
    load_rate_m_s: float = 1.0e-4
    rise_time_s: float = 2.0e-3
    rate_kind: str = "quasi_static"
    include_self_weight: bool = True
    gravity_m_s2: float = GRAVITY_M_S2
    release_period_fraction: float = 0.01
    release_periods: float = 6.0
    dem_dt_s: float = 2.0e-3
    dem_duration_s: float = 0.06
    dem_restitution: float = 0.30
    dem_friction: float = 0.40
    drop_height_m: float = 0.02
    case_id: str = ""
    timestamp: str = "2026-11-11T00:00:00Z"

    def __post_init__(self) -> None:
        for name in (
            "length_m",
            "height_m",
            "width_m",
            "youngs_modulus_pa",
            "density_kg_m3",
            "tensile_strength_pa",
            "fracture_energy_j_m2",
            "element_length_m",
            "load_rate_m_s",
            "rise_time_s",
        ):
            value = float(getattr(self, name))
            if not value > 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in [0, 0.5)")
        if self.rate_kind not in ("quasi_static", "dynamic"):
            raise ValueError(
                f"rate_kind must be 'quasi_static' or 'dynamic', got {self.rate_kind!r}"
            )
        if self.element_length_m > 0.5 * self.height_m:
            raise ValueError(
                "element_length_m is too coarse for the beam depth: the fragment cells "
                "would not resolve the cross-section"
            )
        n = self.length_m / self.element_length_m
        if abs(n - round(n)) > 1e-9 or int(round(n)) % 2:
            raise ValueError(f"length_m / element_length_m must be an even integer, got {n}")
        for axis, size in (("height_m", self.height_m), ("width_m", self.width_m)):
            cells = size / self.element_length_m
            if abs(cells - round(cells)) > 1e-9:
                raise ValueError(
                    f"{axis} / element_length_m must be an integer so that fragment cells "
                    f"are cubes of the element size, got {cells}"
                )

    # -- mesh ------------------------------------------------------------
    @property
    def n_elements(self) -> int:
        return int(round(self.length_m / self.element_length_m))

    @property
    def n_cells_y(self) -> int:
        return int(round(self.height_m / self.element_length_m))

    @property
    def n_cells_z(self) -> int:
        return int(round(self.width_m / self.element_length_m))

    @property
    def cells_per_element(self) -> int:
        return self.n_cells_y * self.n_cells_z

    @property
    def n_fragments(self) -> int:
        return self.n_elements * self.cells_per_element

    # -- geometry / material --------------------------------------------
    @property
    def area_m2(self) -> float:
        return self.height_m * self.width_m

    @property
    def inertia_m4(self) -> float:
        return self.width_m * self.height_m**3 / 12.0

    @property
    def flexural_rigidity_pa_m4(self) -> float:
        return self.youngs_modulus_pa * self.inertia_m4

    @property
    def mass_per_length_kg_m(self) -> float:
        return self.density_kg_m3 * self.area_m2

    @property
    def mass_kg(self) -> float:
        return self.mass_per_length_kg_m * self.length_m

    @property
    def weight_per_length_n_m(self) -> float:
        return self.mass_per_length_kg_m * self.gravity_m_s2

    @property
    def crack_area_m2(self) -> float:
        return self.area_m2

    @property
    def surface_energy_j(self) -> float:
        return self.fracture_energy_j_m2 * self.crack_area_m2

    @property
    def failure_moment_n_m(self) -> float:
        return 2.0 * self.tensile_strength_pa * self.inertia_m4 / self.height_m

    @property
    def static_failure_displacement_m(self) -> float:
        return self.failure_moment_n_m * self.length_m**2 / (12.0 * self.flexural_rigidity_pa_m4)

    @property
    def static_failure_load_n(self) -> float:
        """Central point load that first reaches ``sigma_t`` (self weight included)."""
        return (self.failure_moment_n_m - 0.125 * self.weight_per_length_n_m * self.length_m**2) * 4.0 / self.length_m

    @property
    def half_first_frequency_rad_s(self) -> float:
        half = 0.5 * self.length_m
        return 22.3733 * math.sqrt(
            self.flexural_rigidity_pa_m4 / (self.mass_per_length_kg_m * half**4)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "length_m": self.length_m,
            "height_m": self.height_m,
            "width_m": self.width_m,
            "youngs_modulus_pa": self.youngs_modulus_pa,
            "poisson_ratio": self.poisson_ratio,
            "density_kg_m3": self.density_kg_m3,
            "tensile_strength_pa": self.tensile_strength_pa,
            "fracture_energy_j_m2": self.fracture_energy_j_m2,
            "element_length_m": self.element_length_m,
            "load_rate_m_s": self.load_rate_m_s,
            "rise_time_s": self.rise_time_s,
            "static_failure_load_n": self.static_failure_load_n,
            "static_failure_load_rate_n_per_s": self.static_failure_load_n / self.rise_time_s,
            "rate_kind": self.rate_kind,
            "include_self_weight": self.include_self_weight,
            "n_elements": self.n_elements,
            "cells_per_element": self.cells_per_element,
            "n_fragments": self.n_fragments,
            "mass_kg": self.mass_kg,
            "surface_energy_j": self.surface_energy_j,
            "failure_moment_n_m": self.failure_moment_n_m,
            "static_failure_displacement_m": self.static_failure_displacement_m,
        }


def _make_cases() -> tuple[BeamFractureSpec, ...]:
    out = []
    for length in (2.0, 4.0, 6.0):
        for kind, rate, tag in (
            ("quasi_static", 1.0e-4, "qs"),
            ("dynamic", 1.0e-4, "dyn"),  # cross-head rate is not used by the force ramp
        ):
            out.append(
                BeamFractureSpec(
                    length_m=length,
                    load_rate_m_s=rate,
                    rate_kind=kind,
                    case_id=f"L{int(length)}m_{tag}",
                )
            )
    return tuple(out)


#: Campaign sweep: span L = 2/4/6 m × two load rates (quasi-static / dynamic).
SIX_CASES = _make_cases()


# ---------------------------------------------------------------------------
# Beam finite elements
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BeamMesh:
    """One-dimensional beam mesh; dofs are ``(w, theta)`` per node, ``w`` down."""

    length_m: float
    n_elements: int

    @property
    def n_nodes(self) -> int:
        return self.n_elements + 1

    @property
    def n_dof(self) -> int:
        return 2 * self.n_nodes

    @property
    def element_length_m(self) -> float:
        return self.length_m / self.n_elements

    @property
    def node_x_m(self) -> np.ndarray:
        return np.linspace(0.0, self.length_m, self.n_nodes)

    def dof(self, node: int, component: int) -> int:
        return 2 * int(node) + int(component)

    def element_dofs(self, element: int) -> np.ndarray:
        return np.array([2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3], dtype=int)


def _element_matrices(le: float, EI: float, rhoA: float) -> tuple[np.ndarray, np.ndarray]:
    k = EI / le**3 * np.array(
        [
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, 4.0 * le * le, -6.0 * le, 2.0 * le * le],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, 2.0 * le * le, -6.0 * le, 4.0 * le * le],
        ]
    )
    m = rhoA * le / 420.0 * np.array(
        [
            [156.0, 22.0 * le, 54.0, -13.0 * le],
            [22.0 * le, 4.0 * le * le, 13.0 * le, -3.0 * le * le],
            [54.0, 13.0 * le, 156.0, -22.0 * le],
            [-13.0 * le, -3.0 * le * le, -22.0 * le, 4.0 * le * le],
        ]
    )
    return k, m


def assemble_beam(mesh: BeamMesh, spec: BeamFractureSpec) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the dense stiffness and consistent mass matrices of the bar."""
    EI = spec.flexural_rigidity_pa_m4
    rhoA = spec.mass_per_length_kg_m
    le = mesh.element_length_m
    k_e, m_e = _element_matrices(le, EI, rhoA)
    K = np.zeros((mesh.n_dof, mesh.n_dof))
    M = np.zeros((mesh.n_dof, mesh.n_dof))
    for element in range(mesh.n_elements):
        dofs = mesh.element_dofs(element)
        K[np.ix_(dofs, dofs)] += k_e
        M[np.ix_(dofs, dofs)] += m_e
    return K, M


def self_weight_vector(mesh: BeamMesh, spec: BeamFractureSpec) -> np.ndarray:
    """Consistent nodal load of the self weight (``w`` positive downward)."""
    f = np.zeros(mesh.n_dof)
    if not spec.include_self_weight:
        return f
    q = spec.weight_per_length_n_m
    le = mesh.element_length_m
    local = q * le / 2.0 * np.array([1.0, le / 6.0, 1.0, -le / 6.0])
    for element in range(mesh.n_elements):
        f[mesh.element_dofs(element)] += local
    return f


def nodal_lumped_masses(mesh: BeamMesh, spec: BeamFractureSpec) -> np.ndarray:
    """Lumped translational nodal masses ``rho A le / 2`` (ends half)."""
    m = np.full(mesh.n_nodes, spec.mass_per_length_kg_m * mesh.element_length_m)
    m[0] *= 0.5
    m[-1] *= 0.5
    return m


def tension_stress(mesh: BeamMesh, spec: BeamFractureSpec, u: np.ndarray) -> np.ndarray:
    """Tensile bending stress at the extreme fibre of every element (Pa).

    ``w`` is positive downward, so a sagging (concave-up in the physical frame)
    element has negative curvature and the bottom fibre carries ``-E kappa h/2``.
    """
    le = mesh.element_length_m
    E = spec.youngs_modulus_pa
    out = np.empty(mesh.n_elements)
    for element in range(mesh.n_elements):
        dofs = mesh.element_dofs(element)
        kappa = 0.0
        for xi in (0.0, 0.5, 1.0):
            d2 = np.array(
                [
                    (-6.0 + 12.0 * xi) / le**2,
                    (-4.0 + 6.0 * xi) / le,
                    (6.0 - 12.0 * xi) / le**2,
                    (-2.0 + 6.0 * xi) / le,
                ]
            )
            kappa += float(d2 @ u[dofs]) / 3.0
        out[element] = -E * kappa * 0.5 * spec.height_m
    return out


def element_strain_energy(mesh: BeamMesh, spec: BeamFractureSpec, u: np.ndarray) -> np.ndarray:
    """Per-element elastic strain energy ``1/2 u_e^T k_e u_e``."""
    k_e, _ = _element_matrices(
        mesh.element_length_m, spec.flexural_rigidity_pa_m4, spec.mass_per_length_kg_m
    )
    out = np.empty(mesh.n_elements)
    for element in range(mesh.n_elements):
        ue = u[mesh.element_dofs(element)]
        out[element] = 0.5 * float(ue @ k_e @ ue)
    return out


def _consistent_average(mesh: BeamMesh, values: np.ndarray) -> np.ndarray:
    le = mesh.element_length_m
    weights = np.array([0.5, le / 12.0, 0.5, -le / 12.0])
    return np.array(
        [float(weights @ values[mesh.element_dofs(e)]) for e in range(mesh.n_elements)]
    )


def element_velocity(mesh: BeamMesh, spec: BeamFractureSpec, v: np.ndarray) -> np.ndarray:
    """Consistent-mass average vertical velocity of every element (positive up)."""
    return _consistent_average(mesh, v)


def element_displacement(mesh: BeamMesh, spec: BeamFractureSpec, u: np.ndarray) -> np.ndarray:
    """Consistent-mass average vertical displacement of every element (positive down)."""
    return _consistent_average(mesh, u)


@dataclass
class BeamState:
    """A beam state plus the energies used by the conservation ledger."""

    u: np.ndarray
    v: np.ndarray
    mesh: BeamMesh
    strain_energy_j: float
    kinetic_energy_j: float
    potential_energy_j: float
    max_tension_pa: float
    element_energy_j: np.ndarray
    time_s: float = 0.0

    @property
    def mechanical_energy_j(self) -> float:
        return self.strain_energy_j + self.kinetic_energy_j + self.potential_energy_j


def _state(
    mesh: BeamMesh,
    spec: BeamFractureSpec,
    u: np.ndarray,
    v: np.ndarray | None,
    time_s: float,
) -> BeamState:
    _K, M = assemble_beam(mesh, spec)
    v = np.zeros_like(u) if v is None else v
    ue = element_strain_energy(mesh, spec, u)
    masses = nodal_lumped_masses(mesh, spec)
    potential = -float(np.sum(masses * spec.gravity_m_s2 * u[0::2]))
    return BeamState(
        u=u,
        v=v,
        mesh=mesh,
        strain_energy_j=float(ue.sum()),
        kinetic_energy_j=0.5 * float(v @ M @ v),
        potential_energy_j=potential,
        max_tension_pa=float(np.max(tension_stress(mesh, spec, u))),
        element_energy_j=ue,
        time_s=time_s,
    )


# ---------------------------------------------------------------------------
# Pre-fracture analysis
# ---------------------------------------------------------------------------

@dataclass
class FractureEvent:
    """State of the beam at the instant the tensile strength is reached."""

    state: BeamState
    mode: str
    load_n: float
    imposed_displacement_m: float
    time_to_failure_s: float
    static_reference_load_n: float
    dynamic_amplification: float
    solver_note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "load_n": self.load_n,
            "imposed_displacement_m": self.imposed_displacement_m,
            "time_to_failure_s": self.time_to_failure_s,
            "max_tension_pa": self.state.max_tension_pa,
            "static_reference_load_n": self.static_reference_load_n,
            "dynamic_amplification": self.dynamic_amplification,
            "strain_energy_j": self.state.strain_energy_j,
            "kinetic_energy_j": self.state.kinetic_energy_j,
            "potential_energy_j": self.state.potential_energy_j,
            "solver_note": self.solver_note,
        }


def analytic_midspan_deflection(spec: BeamFractureSpec, load_n: float) -> float:
    """Euler–Bernoulli midspan deflection of a three-point bend bar."""
    return load_n * spec.length_m**3 / (48.0 * spec.flexural_rigidity_pa_m4)


def analytic_max_moment(spec: BeamFractureSpec, load_n: float) -> float:
    strength = 0.125 * spec.weight_per_length_n_m * spec.length_m**2
    return load_n * spec.length_m / 4.0 + strength


def analytic_max_stress(spec: BeamFractureSpec, load_n: float) -> float:
    return analytic_max_moment(spec, load_n) * 0.5 * spec.height_m / spec.inertia_m4


def _solve_partition(K: np.ndarray, f: np.ndarray, imposed: Mapping[int, float]) -> np.ndarray:
    u = np.zeros(K.shape[0])
    constrained = np.array(sorted(imposed), dtype=int)
    prescribed = np.array([imposed[int(d)] for d in constrained], dtype=float)
    u[constrained] = prescribed
    free = np.setdiff1d(np.arange(K.shape[0]), constrained)
    if free.size:
        u[free] = np.linalg.solve(
            K[np.ix_(free, free)], f[free] - K[np.ix_(free, constrained)] @ prescribed
        )
    return u


def solve_static(spec: BeamFractureSpec, displacement_m: float) -> tuple[BeamState, float]:
    """Static displacement-controlled solution; returns the state and the load."""
    mesh = BeamMesh(spec.length_m, spec.n_elements)
    K, _M = assemble_beam(mesh, spec)
    f = self_weight_vector(mesh, spec)
    mid = mesh.n_nodes // 2
    imposed = {
        mesh.dof(0, 0): 0.0,
        mesh.dof(mesh.n_nodes - 1, 0): 0.0,
        mesh.dof(mid, 0): float(displacement_m),
    }
    u = _solve_partition(K, f, imposed)
    load = float((K @ u - f)[mesh.dof(mid, 0)])
    return _state(mesh, spec, u, None, 0.0), load


def _static_fracture(spec: BeamFractureSpec) -> FractureEvent:
    lo = 0.0
    hi = max(4.0 * spec.static_failure_displacement_m, 1e-6)
    state_hi, load_hi = solve_static(spec, hi)
    guard = 0
    while state_hi.max_tension_pa < spec.tensile_strength_pa:
        hi *= 2.0
        state_hi, load_hi = solve_static(spec, hi)
        guard += 1
        if guard > 40:
            raise RuntimeError("static fracture search did not bracket the tensile strength")
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        state_mid, load_mid = solve_static(spec, mid)
        if state_mid.max_tension_pa >= spec.tensile_strength_pa:
            hi, state_hi, load_hi = mid, state_mid, load_mid
        else:
            lo = mid
    return FractureEvent(
        state=state_hi,
        mode="quasi_static",
        load_n=load_hi,
        imposed_displacement_m=hi,
        time_to_failure_s=hi / spec.load_rate_m_s,
        static_reference_load_n=spec.static_failure_load_n,
        dynamic_amplification=1.0,
        solver_note="static displacement-controlled path; inertia neglected",
    )


def newmark_dynamic_load(
    spec: BeamFractureSpec,
    *,
    beta: float = 0.25,
    gamma: float = 0.5,
    max_steps: int = 400000,
) -> dict[str, Any]:
    """Ramp the central *force* up to the static failure load in ``rise_time_s``.

    Force control (not displacement control) is used on purpose: an imposed
    cross-head velocity on a rigid three-point-bend rig concentrates curvature
    under the load and produces non-physical local stresses.  A force ramp lets
    the bar's own inertia set the dynamic amplification.
    """
    mesh = BeamMesh(spec.length_m, spec.n_elements)
    K, M = assemble_beam(mesh, spec)
    f_self = self_weight_vector(mesh, spec)
    mid_node = mesh.n_nodes // 2
    mid_dof = mesh.dof(mid_node, 0)
    constrained = np.array([mesh.dof(0, 0), mesh.dof(mesh.n_nodes - 1, 0)], dtype=int)
    free = np.setdiff1d(np.arange(mesh.n_dof), constrained)

    force_rate = spec.static_failure_load_n / spec.rise_time_s
    omega_beam = math.pi**2 / spec.length_m**2 * math.sqrt(
        spec.flexural_rigidity_pa_m4 / spec.mass_per_length_kg_m
    )
    dt = min(1.0e-4, spec.rise_time_s / 400.0, 2.0 * math.pi / omega_beam / 500.0)
    dt = max(dt, 5.0e-9)
    K_ff = K[np.ix_(free, free)]
    M_ff = M[np.ix_(free, free)]
    inv = np.linalg.inv(M_ff + beta * dt * dt * K_ff)

    u = np.zeros(mesh.n_dof)
    v = np.zeros(mesh.n_dof)
    a = np.zeros(mesh.n_dof)
    a[free] = inv @ (f_self[free] - K_ff @ u[free])

    prev: dict[str, Any] | None = None
    step = 0
    while step < max_steps:
        step += 1
        time = step * dt
        force = force_rate * time
        f_ext = f_self.copy()
        f_ext[mid_dof] += force
        u_pred = u[free] + dt * v[free] + dt * dt * (0.5 - beta) * a[free]
        v_pred = v[free] + dt * (1.0 - gamma) * a[free]
        a_new = inv @ (f_ext[free] - K_ff @ u_pred)
        u_new = u_pred + beta * dt * dt * a_new
        v_new = v_pred + gamma * dt * a_new
        u = u.copy()
        v = v.copy()
        a = a.copy()
        u[free] = u_new
        v[free] = v_new
        a[free] = a_new

        sigma = float(np.max(tension_stress(mesh, spec, u)))
        if sigma >= spec.tensile_strength_pa:
            if prev is None:
                blend_u, blend_v, t_f, load_f = u.copy(), v.copy(), time, force
            else:
                s0 = prev["sigma"]
                frac = (spec.tensile_strength_pa - s0) / max(sigma - s0, 1e-30)
                frac = min(max(frac, 0.0), 1.0)
                blend_u = prev["u"] + frac * (u - prev["u"])
                blend_v = prev["v"] + frac * (v - prev["v"])
                t_f = prev["time"] + frac * dt
                load_f = force_rate * t_f
            state = _state(mesh, spec, blend_u, blend_v, t_f)
            return {
                "state": state,
                "time_s": t_f,
                "imposed_displacement_m": float(blend_u[mid_dof]),
                "load_n": float(load_f),
                "dt_s": dt,
                "steps": step,
                "force_rate_n_per_s": force_rate,
                "dynamic_amplification": float(load_f / spec.static_failure_load_n),
            }
        prev = {"u": u.copy(), "v": v.copy(), "sigma": sigma, "time": time}
    raise RuntimeError("dynamic force ramp did not reach the tensile strength")


def _dynamic_fracture(spec: BeamFractureSpec) -> FractureEvent:
    out = newmark_dynamic_load(spec)
    return FractureEvent(
        state=out["state"],
        mode="dynamic",
        load_n=out["load_n"],
        imposed_displacement_m=out["imposed_displacement_m"],
        time_to_failure_s=out["time_s"],
        static_reference_load_n=spec.static_failure_load_n,
        dynamic_amplification=out["dynamic_amplification"],
        solver_note=(
            "force-controlled ramp with Newmark average acceleration "
            f"(beta=0.25, gamma=0.5), dP/dt={out['force_rate_n_per_s']:.3e} N/s, "
            f"dt={out['dt_s']:.3e} s, {out['steps']} steps, no algorithmic damping"
        ),
    )


def resolve_fracture(spec: BeamFractureSpec) -> FractureEvent:
    """Drive the bar to fracture at the prescribed loading rate."""
    if spec.rate_kind == "quasi_static":
        return _static_fracture(spec)
    return _dynamic_fracture(spec)


# ---------------------------------------------------------------------------
# Post-fracture release dynamics (two free halves, energy conserving)
# ---------------------------------------------------------------------------

@dataclass
class ReleaseState:
    """Result of the (energy-conserving) release integration at hand-off."""

    time_s: float
    strain_energy_j: float
    kinetic_energy_j: float
    potential_energy_j: float
    mechanical_closure: float
    history: dict[str, list[float]]
    element_velocity_m_s: np.ndarray
    element_displacement_m: np.ndarray
    element_strain_energy_j: np.ndarray
    element_kinetic_energy_j: np.ndarray
    element_subgrid_kinetic_j: np.ndarray
    crack_face_velocity_m_s: tuple[float, float]
    centre_of_mass_velocity_m_s: float
    fluctuation_kinetic_j: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "handoff_time_s": self.time_s,
            "strain_energy_j": self.strain_energy_j,
            "kinetic_energy_j": self.kinetic_energy_j,
            "potential_energy_j": self.potential_energy_j,
            "mechanical_closure": self.mechanical_closure,
            "crack_face_velocity_m_s": list(self.crack_face_velocity_m_s),
            "centre_of_mass_velocity_m_s": self.centre_of_mass_velocity_m_s,
            "fluctuation_kinetic_j": self.fluctuation_kinetic_j,
            "element_velocity_m_s": [float(x) for x in self.element_velocity_m_s],
            "element_displacement_m": [float(x) for x in self.element_displacement_m],
            "element_strain_energy_j": [float(x) for x in self.element_strain_energy_j],
            "element_kinetic_energy_j": [float(x) for x in self.element_kinetic_energy_j],
            "element_subgrid_kinetic_j": [float(x) for x in self.element_subgrid_kinetic_j],
            "subgrid_kinetic_total_j": float(np.sum(self.element_subgrid_kinetic_j)),
        }


def element_kinetic_energy(
    mesh: BeamMesh, spec: BeamFractureSpec, v: np.ndarray
) -> np.ndarray:
    """Exact kinetic energy of every element, ``1/2 v_e^T m_e v_e`` (consistent mass)."""
    _k, m_e = _element_matrices(
        mesh.element_length_m, spec.flexural_rigidity_pa_m4, spec.mass_per_length_kg_m
    )
    return np.array(
        [
            0.5 * float(v[mesh.element_dofs(e)] @ m_e @ v[mesh.element_dofs(e)])
            for e in range(mesh.n_elements)
        ]
    )


def _release_energy(K, M, masses, u, v, spec) -> float:
    return (
        0.5 * float(u @ K @ u)
        + 0.5 * float(v @ M @ v)
        - float(np.sum(masses * spec.gravity_m_s2 * u[0::2]))
    )


def release_analysis(spec: BeamFractureSpec, event: FractureEvent) -> ReleaseState:
    """Integrate the two free halves (energy conserving) to the kinetic-energy peak.

    No fracture sink is applied here: the surface energy is a *consumer-side*
    accounting term (see :func:`tensorphysics.failure_handoff.beam_cases.
    build_fragment_cloud`), where it is drawn from the dispersive part of the
    release motion so that the total momentum is untouched.
    """
    half = BeamMesh(0.5 * spec.length_m, spec.n_elements // 2)
    K_h, M_h = assemble_beam(half, spec)
    f_h = self_weight_vector(half, spec)
    n = K_h.shape[0]
    K = np.zeros((2 * n, 2 * n))
    M = np.zeros((2 * n, 2 * n))
    K[:n, :n] = K_h
    K[n:, n:] = K_h
    M[:n, :n] = M_h
    M[n:, n:] = M_h
    f = np.concatenate([f_h, f_h])
    left_face = 2 * (half.n_nodes - 1)
    right_face = n

    u0 = np.zeros(2 * n)
    v0 = np.zeros(2 * n)
    half_elements = spec.n_elements // 2
    for offset, node0 in ((0, 0), (n, half_elements)):
        for local in range(half.n_nodes):
            node = node0 + local
            for component in (0, 1):
                u0[2 * local + component + offset] = event.state.u[2 * node + component]
                v0[2 * local + component + offset] = event.state.v[2 * node + component]

    omega = spec.half_first_frequency_rad_s
    dt = spec.release_period_fraction * 2.0 * math.pi / omega
    n_steps = max(int(math.ceil(spec.release_periods * 2.0 * math.pi / omega / dt)), 200)
    beta, gamma = 0.25, 0.5
    masses = np.concatenate([nodal_lumped_masses(half, spec), nodal_lumped_masses(half, spec)])
    e0 = _release_energy(K, M, masses, u0, v0, spec)

    inv = np.linalg.inv(M + beta * dt * dt * K)
    u = u0.copy()
    v = v0.copy()
    a = np.linalg.solve(M, f - K @ u)
    hist = {k: [] for k in ("times", "strain", "kinetic", "potential", "total")}
    best: dict[str, Any] = {"kinetic": -1.0}
    for step in range(n_steps + 1):
        strain = 0.5 * float(u @ K @ u)
        kinetic = 0.5 * float(v @ M @ v)
        potential = -float(np.sum(masses * spec.gravity_m_s2 * u[0::2]))
        total = strain + kinetic + potential
        if step % 4 == 0 or step == n_steps:
            hist["times"].append(step * dt)
            hist["strain"].append(strain)
            hist["kinetic"].append(kinetic)
            hist["potential"].append(potential)
            hist["total"].append(total)
        if kinetic > best["kinetic"]:
            best = {
                "kinetic": kinetic,
                "strain": strain,
                "potential": potential,
                "u": u.copy(),
                "v": v.copy(),
                "time": step * dt,
                "total": total,
            }
        if step == n_steps:
            break
        u_pred = u + dt * v + dt * dt * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a
        a_new = inv @ (f - K @ u_pred)
        u = u_pred + beta * dt * dt * a_new
        v = v_pred + gamma * dt * a_new
        a = a_new

    u_star = best["u"]
    v_star = best["v"]
    left_vel = element_velocity(half, spec, v_star[:n])
    right_vel = element_velocity(half, spec, v_star[n:])
    element_vel = np.concatenate([left_vel, right_vel])
    element_mass = spec.mass_per_length_kg_m * spec.element_length_m
    centre_velocity = float(np.sum(element_mass * element_vel) / (element_mass * element_vel.size))
    fluctuation = float(
        0.5 * element_mass * float(np.sum((element_vel - centre_velocity) ** 2))
    )
    return ReleaseState(
        time_s=float(best["time"]),
        strain_energy_j=float(best["strain"]),
        kinetic_energy_j=float(best["kinetic"]),
        potential_energy_j=float(best["potential"]),
        mechanical_closure=float(abs(best["total"] - e0) / max(abs(e0), 1e-30)),
        history=hist,
        element_velocity_m_s=element_vel,
        element_displacement_m=np.concatenate(
            [
                element_displacement(half, spec, u_star[:n]),
                element_displacement(half, spec, u_star[n:]),
            ]
        ),
        element_strain_energy_j=np.concatenate(
            [
                element_strain_energy(half, spec, u_star[:n]),
                element_strain_energy(half, spec, u_star[n:]),
            ]
        ),
        element_kinetic_energy_j=np.concatenate(
            [
                element_kinetic_energy(half, spec, v_star[:n]),
                element_kinetic_energy(half, spec, v_star[n:]),
            ]
        ),
        element_subgrid_kinetic_j=np.maximum(
            np.concatenate(
                [
                    element_kinetic_energy(half, spec, v_star[:n]),
                    element_kinetic_energy(half, spec, v_star[n:]),
                ]
            )
            - 0.5 * element_mass * element_vel**2,
            0.0,
        ),
        crack_face_velocity_m_s=(float(v_star[left_face]), float(v_star[right_face])),
        centre_of_mass_velocity_m_s=centre_velocity,
        fluctuation_kinetic_j=fluctuation,
    )


# ---------------------------------------------------------------------------
# failure_faces packet + release sidecar
# ---------------------------------------------------------------------------

def packet_from_fracture(
    spec: BeamFractureSpec, event: FractureEvent, release: ReleaseState
) -> dict[str, Any]:
    """Build and seal the frozen ``failure_faces`` v3 packet of the crack plane."""
    ny, nz = spec.n_cells_y, spec.n_cells_z
    dy = spec.height_m / ny
    dz = spec.width_m / nz
    x = 0.5 * spec.length_m
    y0 = -0.5 * spec.height_m
    z0 = -0.5 * spec.width_m
    faces = []
    for j in range(ny):
        for k in range(nz):
            ya, yb = y0 + j * dy, y0 + (j + 1) * dy
            za, zb = z0 + k * dz, z0 + (k + 1) * dz
            area = dy * dz
            faces.append(
                {
                    "face_id": f"CP{j:02d}{k:02d}",
                    "element_id": spec.n_elements // 2,
                    "vertices_m": [
                        [float(x), float(ya), float(za)],
                        [float(x), float(yb), float(za)],
                        [float(x), float(yb), float(zb)],
                        [float(x), float(ya), float(zb)],
                    ],
                    "normal": [1.0, 0.0, 0.0],
                    "centroid_m": [float(x), float(0.5 * (ya + yb)), float(0.5 * (za + zb))],
                    "area_m2": float(area),
                    "specific_energy_j_m2": float(spec.fracture_energy_j_m2),
                    "energy_j": float(spec.fracture_energy_j_m2 * area),
                    "failure_mode": "mode_I",
                    "mode_mixity": 0.0,
                    "normal_traction_pa": float(spec.tensile_strength_pa),
                    "shear_traction_pa": 0.0,
                }
            )
    packet = {
        "schema_name": "failure_faces",
        "schema_version": 3,
        "version": "3.0.0",
        "source_solver": "TensorPhysics/beam-EB-fem",
        "source_model_id": (
            f"beam:{spec.case_id or 'case'}:L{spec.length_m:g}:{event.mode}:"
            f"n{spec.n_elements}x{ny}x{nz}"
        ),
        "timestamp": spec.timestamp,
        "units": {"length": "m", "area": "m2", "energy": "J", "traction": "Pa"},
        "fracture_layer_thickness_m": float(spec.element_length_m),
        "face_count": len(faces),
        "area_total_m2": float(sum(f["area_m2"] for f in faces)),
        "energy_total_j": float(sum(f["energy_j"] for f in faces)),
        "crack_front_m": [
            [float(x), float(y0), float(z0)],
            [float(x), float(y0), float(z0 + spec.width_m)],
        ],
        "faces": faces,
    }
    from .schema import seal as _seal

    return _seal(packet)


def release_sidecar(
    spec: BeamFractureSpec,
    event: FractureEvent,
    release: ReleaseState,
    packet: dict[str, Any],
) -> dict[str, Any]:
    """Sealed sidecar with the kinematic release state (outside the frozen schema)."""
    from .schema import seal as _seal

    body = {
        "case_id": spec.case_id,
        "packet_sha256": packet["sha256"],
        "pre_fracture": {
            "strain_energy_j": event.state.strain_energy_j,
            "kinetic_energy_j": event.state.kinetic_energy_j,
            "potential_energy_j": event.state.potential_energy_j,
            "max_tension_pa": event.state.max_tension_pa,
            "load_n": event.load_n,
            "imposed_displacement_m": event.imposed_displacement_m,
            "time_to_failure_s": event.time_to_failure_s,
            "mode": event.mode,
            "dynamic_amplification": event.dynamic_amplification,
        },
        "handoff": {
            "time_s": release.time_s,
            "strain_energy_j": release.strain_energy_j,
            "kinetic_energy_j": release.kinetic_energy_j,
            "potential_energy_j": release.potential_energy_j,
            "surface_energy_j": spec.surface_energy_j,
            "mechanical_closure": release.mechanical_closure,
            "crack_face_velocity_m_s": list(release.crack_face_velocity_m_s),
            "centre_of_mass_velocity_m_s": release.centre_of_mass_velocity_m_s,
            "fluctuation_kinetic_j": release.fluctuation_kinetic_j,
        },
        "element_velocity_m_s": [float(x) for x in release.element_velocity_m_s],
        "element_displacement_m": [float(x) for x in release.element_displacement_m],
        "element_strain_energy_j": [float(x) for x in release.element_strain_energy_j],
        "element_kinetic_energy_j": [float(x) for x in release.element_kinetic_energy_j],
        "element_subgrid_kinetic_j": [float(x) for x in release.element_subgrid_kinetic_j],
        "units": {"velocity": "m/s", "displacement": "m", "energy": "J"},
    }
    return _seal(body)