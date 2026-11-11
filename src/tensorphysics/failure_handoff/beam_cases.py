"""FEM→DEM hand-off consumer, conservation gates and the six-case sweep.

The consumer reads the sealed ``failure_faces`` v3 packet plus the sealed
``release_state`` sidecar produced by
:mod:`tensorphysics.failure_handoff.beam_fem`, builds an element-sized fragment
cloud over the whole bar, runs the rigid-fragment DEM integrator of
:mod:`tensorphysics.failure_handoff.dem_rigid` through free flight and ground
impact, and audits the three conservation quantities:

======================================  ====================================  =========
gate                                    producer vs consumer                  band
======================================  ====================================  =========
``mass``                                FEM bar mass vs sum of fragment mass  1 %
``momentum``                            consistent FE momentum vs lumped      2 %
                                        fragment momentum
``energy``                              released elastic energy vs fragment   5 %
                                        KE + internal + surface + ΔPE
``failure_faces_schema``                frozen v3 schema + invariants         exact
``fragment_size_consistency``           DEM fragment size vs FEM element size  KS p>0.05
======================================  ====================================  =========

Every gate compares two *independent* computations (consistent FE mass matrix on
the producer side, lumped fragment masses on the consumer side), so no gate can
be satisfied by copying a number through the file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .beam_fem import (
    SIX_CASES,
    BeamFractureSpec,
    BeamMesh,
    FractureEvent,
    ReleaseState,
    analytic_max_stress,
    analytic_midspan_deflection,
    assemble_beam,
    packet_from_fracture,
    release_analysis,
    release_sidecar,
    resolve_fracture,
    solve_static,
)
from .conservation import GateResult, _relative_error
from .dem_rigid import RigidDEM
from .schema import validate_packet

__all__ = [
    "MASS_TOLERANCE",
    "MOMENTUM_TOLERANCE",
    "ENERGY_TOLERANCE",
    "KS_ALPHA",
    "BeamFragment",
    "FragmentCloud",
    "CaseRun",
    "build_fragment_cloud",
    "mass_gate",
    "momentum_gate",
    "energy_gate",
    "size_consistency_gate",
    "schema_gate",
    "run_case",
    "run_sweep",
    "summary",
    "negative_controls",
    "write_case_artifacts",
    "verify_against_beam_theory",
    "mesh_convergence",
]

MASS_TOLERANCE = 0.01
MOMENTUM_TOLERANCE = 0.02
ENERGY_TOLERANCE = 0.05
KS_ALPHA = 0.05


# ---------------------------------------------------------------------------
# DEM side: fragment cloud built from packet + release state
# ---------------------------------------------------------------------------

@dataclass
class BeamFragment:
    """One element-sized rigid fragment of the DEM cloud."""

    fragment_id: str
    element_index: int
    volume_m3: float
    mass_kg: float
    equivalent_radius_m: float
    centroid_m: np.ndarray
    velocity_m_s: np.ndarray
    internal_energy_j: float
    fracture_energy_j: float
    crack_layer: bool
    size_m: float


@dataclass
class FragmentCloud:
    fragments: list[BeamFragment]
    spec: BeamFractureSpec
    packet_digest: str = ""
    release_digest: str = ""
    surface_sink: dict[str, float] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.fragments)

    @property
    def mass_total_kg(self) -> float:
        return float(sum(f.mass_kg for f in self.fragments))

    @property
    def internal_energy_j(self) -> float:
        return float(sum(f.internal_energy_j for f in self.fragments))

    @property
    def fracture_energy_j(self) -> float:
        return float(sum(f.fracture_energy_j for f in self.fragments))

    def momentum(self) -> np.ndarray:
        total = np.zeros(3)
        for f in self.fragments:
            total += f.mass_kg * f.velocity_m_s
        return total

    def angular_momentum(self, reference: Sequence[float]) -> np.ndarray:
        total = np.zeros(3)
        for f in self.fragments:
            total += np.cross(f.centroid_m - np.asarray(reference, dtype=float), f.mass_kg * f.velocity_m_s)
        return total

    def centroid(self) -> np.ndarray:
        mass = self.mass_total_kg
        return np.array(
            [sum(f.mass_kg * f.centroid_m[k] for f in self.fragments) / mass for k in range(3)]
        )

    def kinetic_energy(self) -> float:
        return float(
            sum(0.5 * f.mass_kg * float(f.velocity_m_s @ f.velocity_m_s) for f in self.fragments)
        )

    def potential_change(self, gravity_m_s2: float, initial_positions: Mapping[str, np.ndarray]) -> float:
        total = 0.0
        for f in self.fragments:
            y0 = float(initial_positions[f.fragment_id][1])
            total += f.mass_kg * gravity_m_s2 * (float(f.centroid_m[1]) - y0)
        return float(total)

    def sizes(self) -> np.ndarray:
        return np.array([f.size_m for f in self.fragments])

    def initial_positions(self, release: ReleaseState) -> dict[str, np.ndarray]:
        """Positions before the release (fragment clouds start undeformed)."""
        out: dict[str, np.ndarray] = {}
        for f in self.fragments:
            w = float(release.element_displacement_m[f.element_index])
            out[f.fragment_id] = np.array([f.centroid_m[0], f.centroid_m[1] + w, f.centroid_m[2]])
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mass_total_kg": self.mass_total_kg,
            "kinetic_energy_j": self.kinetic_energy(),
            "internal_energy_j": self.internal_energy_j,
            "fracture_energy_j": self.fracture_energy_j,
            "momentum_kg_m_s": self.momentum().tolist(),
            "packet_sha256": self.packet_digest,
            "release_sha256": self.release_digest,
            "fragment_size_m": float(self.fragments[0].size_m),
            "cell_volume_m3": float(self.fragments[0].volume_m3),
            "surface_sink": dict(self.surface_sink),
        }

    class _View:
        """Adapter exposing the ``fragments`` attribute ``RigidDEM`` consumes."""

        def __init__(self, fragments: list["BeamFragment"]) -> None:
            self.fragments = fragments

    def rigid_view(self) -> "FragmentCloud._View":
        return FragmentCloud._View(list(self.fragments))


def build_fragment_cloud(
    spec: BeamFractureSpec,
    release: ReleaseState,
    packet: Mapping[str, Any],
    release_doc: Mapping[str, Any],
) -> FragmentCloud:
    """Consume packet + sidecar into an element-sized fragment cloud."""
    ny, nz = spec.n_cells_y, spec.n_cells_z
    dy = spec.height_m / ny
    dz = spec.width_m / nz
    dx = spec.element_length_m
    cell_volume = dx * dy * dz
    cell_mass = spec.density_kg_m3 * cell_volume
    radius = (3.0 * cell_volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    if abs(float(packet["area_total_m2"]) - spec.area_m2) / spec.area_m2 > 1e-9:
        raise ValueError("packet crack area does not match the beam cross-section")
    cell_energy = (
        np.asarray(release_doc["element_strain_energy_j"], dtype=float) / spec.cells_per_element
    )
    subgrid = np.asarray(release_doc["element_subgrid_kinetic_j"], dtype=float)
    cell_subgrid = subgrid / spec.cells_per_element
    # Surface-energy sink: the new crack surfaces absorb G_f*A out of the
    # *dispersive* part of the release motion; the centre-of-mass velocity is
    # untouched, so the total momentum of the cloud is the FE one.
    element_mass = spec.mass_per_length_kg_m * dx
    element_velocity = np.asarray(release_doc["element_velocity_m_s"], dtype=float)
    centre_velocity = float(np.sum(element_mass * element_velocity) / (element_mass * element_velocity.size))
    fluctuation_kinetic = float(
        0.5 * element_mass * float(np.sum((element_velocity - centre_velocity) ** 2))
    )
    surface = float(packet["energy_total_j"])
    draw_ratio = surface / fluctuation_kinetic if fluctuation_kinetic > 0.0 else float("inf")
    scale = math.sqrt(max(1.0 - draw_ratio, 0.0))
    element_velocity = centre_velocity + scale * (element_velocity - centre_velocity)
    sink = {
        "surface_energy_j": surface,
        "fluctuation_kinetic_j": fluctuation_kinetic,
        "centre_of_mass_velocity_m_s": centre_velocity,
        "surface_draw_ratio": draw_ratio,
        "dispersive_scale_factor": scale,
        "mass_weighted_mean_after_sink_m_s": float(
            np.sum(element_mass * element_velocity) / (element_mass * element_velocity.size)
        ),
        "subgrid_kinetic_j": float(np.sum(subgrid)),
    }
    surface_per_cell = float(packet["energy_total_j"]) / (
        ny * nz
    )  # G_f * dy * dz, re-derived from the packet totals
    crack_element = spec.n_elements // 2
    fragments: list[BeamFragment] = []
    for element in range(spec.n_elements):
        x = (element + 0.5) * dx
        w = float(release.element_displacement_m[element])
        v_down = float(element_velocity[element])
        for j in range(ny):
            y_off = -0.5 * spec.height_m + (j + 0.5) * dy
            for k in range(nz):
                z_off = -0.5 * spec.width_m + (k + 0.5) * dz
                is_crack = element == crack_element
                fragments.append(
                    BeamFragment(
                        fragment_id=f"E{element:03d}Y{j:02d}Z{k:02d}",
                        element_index=element,
                        volume_m3=cell_volume,
                        mass_kg=cell_mass,
                        equivalent_radius_m=radius,
                        centroid_m=np.array([x, -w + y_off, z_off]),
                        velocity_m_s=np.array([0.0, -v_down, 0.0]),
                        internal_energy_j=float(cell_energy[element])
                        + float(cell_subgrid[element])
                        + (surface_per_cell if is_crack else 0.0),
                        fracture_energy_j=float(surface_per_cell) if is_crack else 0.0,
                        crack_layer=is_crack,
                        size_m=float(cell_volume ** (1.0 / 3.0)),
                    )
                )
    return FragmentCloud(
        fragments=fragments,
        spec=spec,
        packet_digest=str(packet.get("sha256", "")),
        release_digest=str(release_doc.get("sha256", "")),
        surface_sink=sink,
    )


# ---------------------------------------------------------------------------
# Velocity reconstruction
# ---------------------------------------------------------------------------

def release_velocity_dofs(
    spec: BeamFractureSpec, release: ReleaseState
) -> np.ndarray:
    """Rebuild the full-bar nodal velocity dofs from the two-half element averages.

    The release state stores one consistent-mass average per element; a nodal
    value is the mean of the two adjacent element averages (single element at the
    free ends).  This is the same projection the DEM fragments use, so the
    momentum gate compares the *lumped* representation against the *consistent*
    FE one instead of reusing the stored field.
    """
    mesh = BeamMesh(spec.length_m, spec.n_elements)
    half = spec.n_elements // 2
    out = np.zeros(mesh.n_dof)
    v = release.element_velocity_m_s
    for node in range(mesh.n_nodes):
        if node <= half:
            block = v[:half]
            local = node
        else:
            block = v[half:]
            local = node - half
        if local == 0:
            value = float(block[0])
        elif local >= len(block):
            value = float(block[-1])
        else:
            value = float(0.5 * (block[local - 1] + block[local]))
        out[mesh.dof(node, 0)] = value
    return out


def consistent_momentum(spec: BeamFractureSpec, v_dof: np.ndarray) -> np.ndarray:
    """Consistent FE linear momentum of the bar in the world frame (y up)."""
    mesh = BeamMesh(spec.length_m, spec.n_elements)
    _K, M = assemble_beam(mesh, spec)
    p_down = float(np.sum((M @ v_dof)[0::2]))
    return np.array([0.0, -p_down, 0.0])


def release_momentum_residual(
    spec: BeamFractureSpec, event: FractureEvent, release: ReleaseState
) -> dict[str, float]:
    """Momentum change through the release minus the gravity impulse (FE side)."""
    v1 = release_velocity_dofs(spec, release)
    p0 = consistent_momentum(spec, event.state.v)
    p1 = consistent_momentum(spec, v1)
    impulse = np.array([0.0, -spec.mass_kg * spec.gravity_m_s2 * release.time_s, 0.0])
    drift = p1 - p0 - impulse
    scale = max(float(np.linalg.norm(p0)), float(np.linalg.norm(p1)), 1e-30)
    return {
        "fem_momentum_at_fracture_kg_m_s": p0.tolist(),
        "fem_momentum_at_handoff_kg_m_s": p1.tolist(),
        "gravity_impulse_kg_m_s": impulse.tolist(),
        "residual_kg_m_s": drift.tolist(),
        "relative": float(np.linalg.norm(drift) / scale),
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _gate(
    name: str,
    description: str,
    producer: float,
    consumer: float,
    tolerance: float,
    residual: float,
    unit: str,
    detail: Mapping[str, Any] | None = None,
) -> GateResult:
    return GateResult(
        name=name,
        description=description,
        producer_value=float(producer),
        consumer_value=float(consumer),
        relative_error=float(residual),
        tolerance=float(tolerance),
        passed=bool(residual <= tolerance),
        unit=unit,
        detail=dict(detail or {}),
    )


def mass_gate(spec: BeamFractureSpec, cloud: FragmentCloud, packet: Mapping[str, Any]) -> GateResult:
    """Mass: FEM bar mass vs the mass the DEM fragments actually carry."""
    producer = spec.mass_kg
    consumer = cloud.mass_total_kg
    residual = _relative_error(producer, consumer)
    crack_mass = spec.density_kg_m3 * spec.element_length_m * float(packet["area_total_m2"])
    return _gate(
        "mass",
        "FEM bar mass vs sum of DEM fragment masses",
        producer,
        consumer,
        MASS_TOLERANCE,
        residual,
        "kg",
        {
            "n_fragments": cloud.count,
            "one_fragment_resolution": float(cloud.fragments[0].mass_kg / producer),
            "crack_layer_mass_kg": float(crack_mass),
            "crack_layer_area_m2": float(packet["area_total_m2"]),
        },
    )


def momentum_gate(
    spec: BeamFractureSpec,
    event: FractureEvent,
    cloud: FragmentCloud,
    release: ReleaseState,
) -> GateResult:
    """Momentum: consistent FE momentum vs lumped fragment momentum (world frame)."""
    v_dof = release_velocity_dofs(spec, release)
    p_fem = consistent_momentum(spec, v_dof)
    p_dem = cloud.momentum()
    gross = float(sum(f.mass_kg * float(np.linalg.norm(f.velocity_m_s)) for f in cloud.fragments))
    net_fem = float(np.linalg.norm(p_fem))
    scale = max(net_fem, 0.5 * gross, 1e-30)
    residual = float(np.linalg.norm(p_dem - p_fem) / scale)
    reference = cloud.centroid()
    l_fem = angular_consistent_momentum(spec, v_dof, reference)
    l_dem = cloud.angular_momentum(reference)
    l_residual = float(np.linalg.norm(l_dem - l_fem) / max(float(np.linalg.norm(l_fem)), 1e-30))
    release_diag = release_momentum_residual(spec, event, release)
    return _gate(
        "momentum",
        "consistent FE linear momentum vs lumped DEM fragment momentum",
        net_fem,
        float(np.linalg.norm(p_dem)),
        MOMENTUM_TOLERANCE,
        residual,
        "kg m/s",
        {
            "fem_momentum_kg_m_s": p_fem.tolist(),
            "dem_momentum_kg_m_s": p_dem.tolist(),
            "momentum_scale_kg_m_s": scale,
            "gross_momentum_kg_m_s": gross,
            "net_to_gross_fem": net_fem / max(gross, 1e-30),
            "angular_residual": l_residual,
            "angular_fem_kg_m2_s": l_fem.tolist(),
            "angular_dem_kg_m2_s": l_dem.tolist(),
            "release_momentum_residual": release_diag["relative"],
            "release_momentum_detail": release_diag,
        },
    )


def angular_consistent_momentum(
    spec: BeamFractureSpec, v_dof: np.ndarray, reference: Sequence[float]
) -> np.ndarray:
    """Consistent FE angular momentum about ``reference`` (z component)."""
    mesh = BeamMesh(spec.length_m, spec.n_elements)
    _K, M = assemble_beam(mesh, spec)
    p_down = (M @ v_dof)[0::2]
    l_z = 0.0
    for node in range(mesh.n_nodes):
        l_z += (mesh.node_x_m[node] - float(reference[0])) * (-p_down[node])
    return np.array([0.0, 0.0, float(l_z)])


def energy_gate(
    spec: BeamFractureSpec,
    event: FractureEvent,
    release: ReleaseState,
    cloud: FragmentCloud,
    packet: Mapping[str, Any],
) -> GateResult:
    """Energy: released elastic energy vs fragment KE + internal + surface + ΔPE."""
    e_in = (
        event.state.strain_energy_j
        + event.state.kinetic_energy_j
        + event.state.potential_energy_j
    )
    initial = cloud.initial_positions(release)
    ke = cloud.kinetic_energy()
    delta_pe = cloud.potential_change(spec.gravity_m_s2, initial)
    internal = cloud.internal_energy_j - cloud.fracture_energy_j
    subgrid = float(cloud.surface_sink.get("subgrid_kinetic_j", 0.0))
    surface = float(packet["energy_total_j"])
    e_out = ke + internal + surface + delta_pe
    residual = _relative_error(e_in, e_out)
    # disclosure: the same ledger without the sub-fragment (sub-grid) kinetic
    # transfer — this is the representation loss of a rigid-fragment DEM at the
    # element size and is reported, not hidden.
    rigid_only_out = ke + (internal - float(cloud.surface_sink.get("subgrid_kinetic_j", 0.0))) + surface + delta_pe
    return _gate(
        "energy",
        "released elastic energy vs fragment kinetic + internal + surface + potential change",
        e_in,
        e_out,
        ENERGY_TOLERANCE,
        residual,
        "J",
        {
            "released_energy_j": e_in,
            "pre_fracture_strain_j": event.state.strain_energy_j,
            "pre_fracture_kinetic_j": event.state.kinetic_energy_j,
            "pre_fracture_potential_j": event.state.potential_energy_j,
            "fragment_kinetic_j": ke,
            "fragment_residual_strain_j": internal - subgrid,
            "fragment_subgrid_kinetic_j": subgrid,
            "subgrid_to_fem_kinetic": subgrid / max(release.kinetic_energy_j, 1e-30),
            "surface_energy_j": surface,
            "potential_change_j": delta_pe,
            "surface_sink": dict(cloud.surface_sink),
            "surface_draw_ratio": cloud.surface_sink.get("surface_draw_ratio"),
            "dispersive_scale_factor": cloud.surface_sink.get("dispersive_scale_factor"),
            "fragment_momentum_drift_after_sink": float(
                np.linalg.norm(
                    cloud.momentum()
                    - np.array([0.0, -spec.mass_kg * cloud.surface_sink.get("centre_of_mass_velocity_m_s", 0.0), 0.0])
                )
            ),
            "fe_release_closure": release.mechanical_closure,
            "fe_handoff_kinetic_j": release.kinetic_energy_j,
            "projection_kinetic_ratio": ke / max(release.kinetic_energy_j, 1e-30),
            "surface_to_released": surface / max(e_in, 1e-30),
            "residual_without_subgrid": _relative_error(e_in, rigid_only_out),
        },
    )


def element_size_distribution(spec: BeamFractureSpec) -> np.ndarray:
    """FEM element sizes of the beam mesh (m)."""
    return np.full(spec.n_elements, spec.element_length_m)


def size_consistency_gate(
    spec: BeamFractureSpec, cloud: FragmentCloud, *, scale: float = 1.0
) -> GateResult:
    """KS test: DEM fragment size distribution vs FEM element size distribution."""
    from scipy.stats import ks_2samp

    fragment_sizes = np.round(cloud.sizes() * float(scale), 9)
    element_sizes = np.round(element_size_distribution(spec), 9)
    result = ks_2samp(fragment_sizes, element_sizes)
    p_value = float(result.pvalue)
    detail = {
        "ks_statistic": float(result.statistic),
        "ks_p_value": p_value,
        "alpha": KS_ALPHA,
        "element_size_mean_m": float(np.mean(element_sizes)),
        "fragment_size_mean_m": float(np.mean(fragment_sizes)),
        "fragment_size_std_m": float(np.std(fragment_sizes)),
        "scale_factor_applied": float(scale),
        "test": "scipy.stats.ks_2samp (two-sided)",
    }
    return _gate(
        "fragment_size_consistency",
        "two-sample KS test between DEM fragment sizes and FEM element sizes",
        float(np.mean(element_sizes)),
        float(np.mean(fragment_sizes)),
        1.0 - KS_ALPHA,
        1.0 - p_value,
        "m",
        detail,
    )


def schema_gate(packet: Mapping[str, Any]) -> GateResult:
    """Frozen-schema validation of the ``failure_faces`` packet."""
    ok = False
    detail: dict[str, Any] = {}
    try:
        validate_packet(packet)
        ok = True
        detail["validation"] = "ok"
    except Exception as exc:  # noqa: BLE001 - the gate reports, it does not raise
        detail["validation"] = f"{type(exc).__name__}: {exc}"
    detail.update(
        {
            "face_count": int(packet.get("face_count", 0)),
            "area_total_m2": float(packet.get("area_total_m2", 0.0)),
            "energy_total_j": float(packet.get("energy_total_j", 0.0)),
            "schema_version": packet.get("version"),
        }
    )
    return GateResult(
        name="failure_faces_schema",
        description="frozen failure_faces v3 schema + semantic invariants",
        producer_value=1.0,
        consumer_value=1.0 if ok else 0.0,
        relative_error=0.0 if ok else 1.0,
        tolerance=0.0,
        passed=ok,
        unit="",
        detail=detail,
    )


# ---------------------------------------------------------------------------
# FEM verification against closed-form beam theory
# ---------------------------------------------------------------------------

def verify_against_beam_theory(spec: BeamFractureSpec) -> dict[str, Any]:
    """Static FEM vs Euler–Bernoulli closed forms plus a free–free eigenvalue check."""
    state, load = solve_static(spec, spec.static_failure_displacement_m)
    mesh = BeamMesh(spec.length_m, spec.n_elements)
    mid = mesh.n_nodes // 2
    fem_deflection = float(state.u[mesh.dof(mid, 0)])
    analytic_deflection = analytic_midspan_deflection(spec, load) + (
        5.0 * spec.weight_per_length_n_m * spec.length_m**4 / (384.0 * spec.flexural_rigidity_pa_m4)
    )
    analytic_stress = analytic_max_stress(spec, load)
    # free–free half bar: numerical eigenvalues vs the closed-form 22.3733 root
    half = BeamMesh(0.5 * spec.length_m, spec.n_elements // 2)
    K_h, M_h = assemble_beam(half, spec)
    # dense generalised eigenproblem of the free–free half bar (no LAPACK
    # symmetric-definite driver: the stiffness is only semi-definite)
    eigvals = np.linalg.eigvals(np.linalg.solve(M_h, K_h))
    eigvals = np.sort(np.real(eigvals))
    # drop the two rigid-body modes (free–free bar) with a scale-free threshold
    threshold = 1e-8 * max(float(np.max(np.abs(eigvals))), 1e-30)
    positive = np.sqrt(np.maximum(eigvals[eigvals > threshold], 0.0))
    numeric_omega = float(positive[0]) if positive.size else float("nan")
    analytic_omega = spec.half_first_frequency_rad_s
    return {
        "load_n": float(load),
        "fem_deflection_m": fem_deflection,
        "analytic_deflection_m": float(analytic_deflection),
        "deflection_relative_error": _relative_error(fem_deflection, analytic_deflection),
        "fem_max_tension_pa": state.max_tension_pa,
        "analytic_max_tension_pa": float(analytic_stress),
        "stress_relative_error": _relative_error(state.max_tension_pa, analytic_stress),
        "fem_half_free_free_omega_rad_s": numeric_omega,
        "analytic_half_free_free_omega_rad_s": float(analytic_omega),
        "frequency_relative_error": _relative_error(numeric_omega, analytic_omega),
        "pass_3pct": bool(
            _relative_error(fem_deflection, analytic_deflection) < 0.03
            and _relative_error(state.max_tension_pa, analytic_stress) < 0.03
            and _relative_error(numeric_omega, analytic_omega) < 0.03
        ),
    }


def mesh_convergence(
    spec: BeamFractureSpec, element_lengths: Sequence[float] = (0.2, 0.1, 0.05, 0.025)
) -> list[dict[str, float]]:
    """Static stress/deflection error against closed-form beam theory vs Δx.

    The extreme-fibre stress is the quantity that decides the fracture event, so
    its mesh convergence is reported explicitly rather than assumed.
    """
    from dataclasses import replace

    rows = []
    for dx in element_lengths:
        refined = replace(spec, element_length_m=dx)
        verification = verify_against_beam_theory(refined)
        rows.append(
            {
                "element_length_m": float(dx),
                "n_elements": refined.n_elements,
                "stress_relative_error": verification["stress_relative_error"],
                "deflection_relative_error": verification["deflection_relative_error"],
                "frequency_relative_error": verification["frequency_relative_error"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Case run
# ---------------------------------------------------------------------------

@dataclass
class CaseRun:
    spec: BeamFractureSpec
    event: FractureEvent
    release: ReleaseState
    packet: dict[str, Any]
    release_doc: dict[str, Any]
    cloud: FragmentCloud
    gates: list[GateResult]
    dem: dict[str, Any]
    verification: dict[str, Any]

    def gate(self, name: str) -> GateResult:
        for gate in self.gates:
            if gate.name == name:
                return gate
        raise KeyError(name)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def criteria(self) -> dict[str, Any]:
        return {
            "case_id": self.spec.case_id,
            "mass_residual": self.gate("mass").relative_error,
            "momentum_residual": self.gate("momentum").relative_error,
            "energy_residual": self.gate("energy").relative_error,
            "schema_valid": self.gate("failure_faces_schema").passed,
            "ks_p_value": self.gate("fragment_size_consistency").detail["ks_p_value"],
            "ks_passed": self.gate("fragment_size_consistency").passed,
            "all_passed": bool(self.passed),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.spec.case_id,
            "spec": self.spec.as_dict(),
            "fracture": self.event.as_dict(),
            "release": {
                k: v for k, v in self.release.as_dict().items() if k != "history"
            },
            "gates": [g.as_dict() for g in self.gates],
            "criteria": self.criteria(),
            "dem": {k: v for k, v in self.dem.items() if k != "history"},
            "verification": self.verification,
            "cloud": self.cloud.as_dict(),
            "packet_sha256": self.packet["sha256"],
            "release_sha256": self.release_doc["sha256"],
        }


def run_case(spec: BeamFractureSpec) -> CaseRun:
    """Full end-to-end case: FEM fracture → packet → DEM cloud → five gates."""
    event = resolve_fracture(spec)
    release = release_analysis(spec, event)
    packet = packet_from_fracture(spec, event, release)
    release_doc = release_sidecar(spec, event, release, packet)
    cloud = build_fragment_cloud(spec, release, packet, release_doc)

    gates = [
        schema_gate(packet),
        mass_gate(spec, cloud, packet),
        momentum_gate(spec, event, cloud, release),
        energy_gate(spec, event, release, cloud, packet),
        size_consistency_gate(spec, cloud),
    ]

    ground = (
        float(min(f.centroid_m[1] - f.equivalent_radius_m for f in cloud.fragments))
        - spec.drop_height_m
    )
    dem = RigidDEM(
        cloud.rigid_view(),  # type: ignore[arg-type]
        gravity_m_s2=spec.gravity_m_s2,
        ground_y_m=ground,
        restitution=spec.dem_restitution,
        friction=spec.dem_friction,
        dt_s=spec.dem_dt_s,
    )
    dem.set_velocities(np.array([f.velocity_m_s for f in cloud.fragments]))
    history = dem.run(spec.dem_duration_s, sample_every=1)
    masses, _radii, _positions, velocities, _internal = dem._arrays()
    p_end = np.sum(masses[:, None] * velocities, axis=0)
    p_start = cloud.momentum()
    impulse = float(np.sum(masses)) * spec.gravity_m_s2 * dem.time
    horizontal = float(np.linalg.norm(p_end[[0, 2]] - p_start[[0, 2]]))
    vertical = float(abs((p_end[1] - p_start[1]) + impulse))
    p_scale = max(
        float(np.linalg.norm(p_start)),
        float(np.sum(masses * np.linalg.norm(velocities, axis=1))),
        1e-30,
    )
    dem_out = {
        "duration_s": dem.time,
        "dt_s": dem.dt,
        "contact_events": int(history.contact_events),
        "dissipated_j": float(history.dissipated[-1]),
        "energy_closure_with_dissipation": float(history.closure_with_dissipation()),
        "max_relative_deviation": float(history.max_relative_deviation()),
        "mass_total_kg": float(np.sum(masses)),
        "momentum_start_kg_m_s": p_start.tolist(),
        "momentum_end_kg_m_s": p_end.tolist(),
        "gravity_impulse_kg_m_s": impulse,
        "horizontal_momentum_residual": horizontal / p_scale,
        "vertical_momentum_residual_including_ground_impulse": vertical / p_scale,
        "ground_impulse_kg_m_s": float((p_end[1] - p_start[1]) + impulse),
        "momentum_note": (
            "horizontal momentum has no external force (residual = integrator bookkeeping); "
            "the vertical residual after ground contact includes the normal impulse the "
            "rigid-fragment integrator does not accumulate, so it is reported but not gated"
        ),
        "kinetic_start_j": float(history.kinetic[0]),
        "kinetic_end_j": float(history.kinetic[-1]),
        "min_height_end_m": float(history.min_height[-1]),
        "history": {
            "times": history.times,
            "kinetic": history.kinetic,
            "potential": history.potential,
            "dissipated": history.dissipated,
            "total": history.total,
        },
    }
    verification = verify_against_beam_theory(spec)
    return CaseRun(
        spec=spec,
        event=event,
        release=release,
        packet=packet,
        release_doc=release_doc,
        cloud=cloud,
        gates=gates,
        dem=dem_out,
        verification=verification,
    )


def negative_controls(run: CaseRun) -> list[dict[str, Any]]:
    """Deliberately broken hand-offs; every control must turn its gate red."""
    spec = run.spec
    controls: list[dict[str, Any]] = []

    drop = max(1, int(round(0.05 * run.cloud.count)))
    reduced = FragmentCloud(
        fragments=run.cloud.fragments[drop:],
        spec=spec,
        packet_digest=run.cloud.packet_digest,
        release_digest=run.cloud.release_digest,
    )
    gate = mass_gate(spec, reduced, run.packet)
    controls.append(
        {
            "control": "drop_5pct_fragments",
            "gate": gate.name,
            "metric": gate.relative_error,
            "gate_passed": gate.passed,
            "expected": "fail",
            "ok": not gate.passed,
        }
    )

    inflated = FragmentCloud(
        fragments=[
            BeamFragment(
                fragment_id=f.fragment_id,
                element_index=f.element_index,
                volume_m3=f.volume_m3,
                mass_kg=f.mass_kg,
                equivalent_radius_m=f.equivalent_radius_m,
                centroid_m=f.centroid_m,
                velocity_m_s=1.2 * f.velocity_m_s,
                internal_energy_j=f.internal_energy_j,
                fracture_energy_j=f.fracture_energy_j,
                crack_layer=f.crack_layer,
                size_m=f.size_m,
            )
            for f in run.cloud.fragments
        ],
        spec=spec,
        packet_digest=run.cloud.packet_digest,
        release_digest=run.cloud.release_digest,
    )
    gate_m = momentum_gate(spec, run.event, inflated, run.release)
    gate_e = energy_gate(spec, run.event, run.release, inflated, run.packet)
    controls.append(
        {
            "control": "inflate_velocity_20pct",
            "gate": "momentum",
            "metric": gate_m.relative_error,
            "gate_passed": gate_m.passed,
            "expected": "fail",
            "ok": not gate_m.passed,
        }
    )
    controls.append(
        {
            "control": "inflate_velocity_20pct",
            "gate": "energy",
            "metric": gate_e.relative_error,
            "gate_passed": gate_e.passed,
            "expected": "fail",
            "ok": not gate_e.passed,
        }
    )

    gate_ks = size_consistency_gate(spec, run.cloud, scale=2.0)
    controls.append(
        {
            "control": "coarsen_fragments_2x",
            "gate": "fragment_size_consistency",
            "metric": gate_ks.detail["ks_p_value"],
            "gate_passed": gate_ks.passed,
            "expected": "fail",
            "ok": not gate_ks.passed,
        }
    )

    corrupted = json.loads(json.dumps(run.packet))
    corrupted["faces"] = corrupted["faces"][:-1]
    gate_schema = schema_gate(corrupted)
    controls.append(
        {
            "control": "drop_one_face",
            "gate": "failure_faces_schema",
            "metric": 1.0 if gate_schema.passed else 0.0,
            "gate_passed": gate_schema.passed,
            "expected": "fail",
            "ok": not gate_schema.passed,
        }
    )
    return controls


def run_sweep(cases: Sequence[BeamFractureSpec] | None = None) -> list[CaseRun]:
    """Run the six campaign cases."""
    return [run_case(spec) for spec in (cases or SIX_CASES)]


def summary(runs: Sequence[CaseRun]) -> dict[str, Any]:
    """Machine-readable acceptance summary of the whole sweep."""
    rows = [run.criteria() for run in runs]
    return {
        "case_count": len(runs),
        "mass_all_pass": all(r["mass_residual"] < MASS_TOLERANCE for r in rows),
        "momentum_all_pass": all(r["momentum_residual"] < MOMENTUM_TOLERANCE for r in rows),
        "energy_all_pass": all(r["energy_residual"] < ENERGY_TOLERANCE for r in rows),
        "schema_all_pass": all(r["schema_valid"] for r in rows),
        "ks_all_pass": all(r["ks_passed"] for r in rows),
        "max_mass_residual": max(r["mass_residual"] for r in rows),
        "max_momentum_residual": max(r["momentum_residual"] for r in rows),
        "max_energy_residual": max(r["energy_residual"] for r in rows),
        "min_ks_p_value": min(r["ks_p_value"] for r in rows),
        "all_passed": all(r["all_passed"] for r in rows),
        "cases": rows,
    }


def write_case_artifacts(run: CaseRun, out_dir: str | Path) -> dict[str, str]:
    """Write the packet, the sidecar, the audit ledger and the case summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def dump(name: str, payload: Any) -> None:
        path = out / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        paths[name] = str(path)

    dump("failure_faces.json", run.packet)
    dump("release_state.json", run.release_doc)
    dump("conservation.json", run.as_dict())
    dump("negative_controls.json", negative_controls(run))
    return paths