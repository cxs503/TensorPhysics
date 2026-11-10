"""End-to-end **continuous crack → discrete fragment** handoff protocol.

This is the executable protocol behind node ``p273w3-n03``:

    FEM (mixed-mode notched ice beam)                 [three_point_bend / mixed_mode]
      -> crack-tip K_I, K_II by displacement correlation
      -> MTS kink angle (criterion) + independent stress-side Williams fit
      -> crack advance by element deletion (independent path estimate)
      -> sealed ``failure_faces`` v3 packet           [fem_export / schema, frozen]
    DEM initial condition                              [transfer]
      -> polyhedral fragments reconstructed from the packet vertices
      -> mass, inertia tensor, inherited velocity + spin (rigid-body fit)
      -> volume / momentum / energy budgets
    DEM continuation                                   [dem_rigid]
      -> rigid-fragment free fall + impact with the inherited velocities

The machine-checkable criteria of the node are evaluated inline and returned as
numbers by :meth:`ProtocolRun.criteria`:

* ``C1`` kink angle: MTS applied to the stress-side Williams fit of the FE field
  vs the closed-form MTS angle from the displacement-correlation ``K_I, K_II`` —
  tolerance 10 deg.
* ``C2`` volume: reconstructed polyhedron volume vs FEM failure-region volume
  (element Jacobian volumes) — tolerance 3 %.
* ``C3`` momentum: fragment momentum vs FEM crack-face nodal momentum —
  tolerance 2 % (angular momentum reported as a secondary check).
* ``C4`` energy: fracture + kinetic + potential ledger, FEM side vs DEM side —
  tolerance 5 %.

The frozen packet contract is *not* touched: everything the handoff adds rides in
a companion bundle keyed by the packet SHA-256 (see :func:`run_protocol`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dem_fragments import FragmentSet, fragments_from_packet
from .dem_rigid import EnergyHistory, RigidDEM
from .fem_export import (
    FEMFractureResult,
    FractureMaterial,
    extract_failure_faces,
    packet_from_result,
    write_packet,
)
from .mixed_mode import (
    mts_kink_angle_deg,
    propagate_crack,
    ring_kink_angle,
    senb_k1_reference,
    solve_step,
    stress_intensity_factors,
    williams_stress_fit,
)
from .pipeline import calibrate_load
from .schema import packet_digest, validate_packet
from .three_point_bend import BeamSpec, build_bend_model, solve_bend
from .transfer import (
    TransferFragment,
    build_transfer_fragment,
    element_volumes,
    energy_budget,
    momentum_budget,
    rigid_body_fit,
    volume_budget,
)

__all__ = [
    "ICE_SPEC",
    "ICE_MATERIAL",
    "ICE_DENSITY_KG_M3",
    "ProtocolRun",
    "build_fragments",
    "gate_sensitivity",
    "run_protocol",
    "write_protocol_artifacts",
]

#: Ice three-point-bend specimen: span/width = 4 so the ASTM E399 SENB reference
#: applies directly as an external ``K_I`` anchor.
ICE_SPEC = BeamSpec(
    length_m=0.24,
    height_m=0.06,
    thickness_m=0.02,
    notch_depth_m=0.02,
    notch_gap_m=0.005,
    E_pa=9.0e9,
    nu=0.33,
)

#: Columnar freshwater ice, quasi-brittle tensile behaviour (typical laboratory
#: range: E = 5-10 GPa, sigma_t = 0.5-2 MPa, G_Ic = 3-15 J/m^2).
ICE_MATERIAL = FractureMaterial(
    E_pa=9.0e9,
    nu=0.33,
    sigma_t_pa=1.2e6,
    G_Ic_j_m2=8.0,
    G_IIc_j_m2=16.0,
    eta_bk=1.5,
)
ICE_DENSITY_KG_M3 = 917.0

DEFAULT_TIMESTAMP = "2026-11-10T00:00:00Z"


def build_fragments(
    packet: Mapping,
    *,
    density_kg_m3: float,
    element_volume_by_id: Mapping[int, float],
    thickness_m: float | None = None,
) -> list[TransferFragment]:
    """Packet faces -> fragments with reconstructed geometry, mass and inertia."""
    thickness = float(
        thickness_m if thickness_m is not None else packet["fracture_layer_thickness_m"]
    )
    fragments: list[TransferFragment] = []
    for i, face in enumerate(packet["faces"]):
        element_id = int(face.get("element_id", -1))
        fragments.append(
            build_transfer_fragment(
                face,
                density_kg_m3=density_kg_m3,
                thickness_m=thickness,
                index=i,
                fem_element_volume_m3=float(element_volume_by_id[element_id]),
            )
        )
    return fragments


def _attach_velocity(
    fragments: Sequence[TransferFragment],
    *,
    mesh,
    u: np.ndarray,
    rate_scale: float,
) -> None:
    """Carry the FEM mass distribution and the rigid-body part of the velocity.

    The producer state stored with each fragment is the **element** nodal set
    (8 nodes, lumped mass ``m_e/8``) because that is how the FEM stores its mass;
    the FEM-side momentum and kinetic energy are then computed on exactly the mass
    distribution the DEM fragment represents.  The rigid-body fit itself uses only
    the four fracture-face nodes, because only that surface is inside the packet.
    """
    displacement = np.asarray(u, dtype=float).reshape(-1, 3)
    nodes = np.asarray(mesh.nodes, dtype=float)
    for fragment in fragments:
        element = int(fragment.element_id)
        if element < 0 or element >= mesh.n_elem:
            raise ValueError(f"fragment {fragment.fragment_id}: bad element id {element}")
        conn = [int(v) for v in np.asarray(mesh.connectivity[element], dtype=int)]
        positions = nodes[conn]
        nodal_v = displacement[conn] * float(rate_scale)
        fragment.node_ids = conn
        fragment.nodal_positions_m = positions
        fragment.nodal_velocity_m_s = nodal_v
        face_idx = [int(np.argmin(np.linalg.norm(positions - vertex, axis=1))) for vertex in fragment.vertices_m]
        fragment.face_node_indices = face_idx
        # Fit the rigid motion on all element nodes: that is the mass distribution
        # the fragment stands for, and the packet-only (4 face node) fit biases the
        # translated velocity towards the tensile face.
        fit = rigid_body_fit(positions, nodal_v, fragment.centroid_m)
        fragment.velocity_m_s = np.asarray(fit["velocity_m_s"], dtype=float)
        fragment.omega_rad_s = np.asarray(fit["omega_rad_s"], dtype=float)
        fragment.fit_residual_m_s = float(fit["fit_residual_m_s"])


@dataclass
class ProtocolRun:
    """Everything one protocol execution produced, with the criteria inline."""

    spec: BeamSpec
    material: FractureMaterial
    settings: dict[str, Any]
    k1_anchor: dict[str, Any]
    tip_state: dict[str, Any]
    propagation: dict[str, Any]
    fem: dict[str, Any]
    result: FEMFractureResult
    packet: dict
    fragments: list[TransferFragment]
    volume: dict[str, Any]
    momentum: dict[str, Any]
    energy: dict[str, Any]
    dem_history: EnergyHistory
    dem_initial: dict[str, Any]

    def criteria(self) -> dict[str, Any]:
        c1 = abs(float(self.tip_state["theta_fem_deg"]) - float(self.tip_state["theta_mts_deg"]))
        c1_path = abs(float(self.propagation["theta_path_deg"]) - float(self.tip_state["theta_mts_deg"]))
        return {
            "C1_kink_angle_fem_deg": float(self.tip_state["theta_fem_deg"]),
            "C1_kink_angle_mts_deg": float(self.tip_state["theta_mts_deg"]),
            "C1_kink_angle_error_deg": c1,
            "C1_tolerance_deg": 10.0,
            "C1_pass": bool(c1 <= 10.0),
            "C1_path_angle_deg": float(self.propagation["theta_path_deg"]),
            "C1_path_angle_error_deg": c1_path,
            "C1_path_within_tolerance": bool(c1_path <= 10.0),
            "C1_ring_scan_mean_deg": float(self.tip_state["ring_scan_mean_deg"]),
            "C2_volume_relative_error": float(self.volume["relative_error"]),
            "C2_worst_fragment_relative_error": float(self.volume["worst_fragment_relative_error"]),
            "C2_tolerance": float(self.volume["tolerance"]),
            "C2_pass": bool(self.volume["passed"]),
            "C3_momentum_relative_error": float(self.momentum["momentum_relative_error"]),
            "C3_angular_momentum_relative_error": float(self.momentum["angular_relative_error"]),
            "C3_tolerance": float(self.momentum["tolerance"]),
            "C3_pass": bool(self.momentum["passed"]),
            "C3_angular_pass": bool(self.momentum["angular_passed"]),
            "C4_energy_relative_error": float(self.energy["relative_error"]),
            "C4_kinetic_relative_error": float(self.energy["kinetic_relative_error"]),
            "C4_tolerance": float(self.energy["tolerance"]),
            "C4_pass": bool(self.energy["passed"]),
            "C4_dem_continuation_closure": float(self.dem_history.closure_with_dissipation()),
            "C4_dem_continuation_pass": bool(self.dem_history.closure_with_dissipation() <= 0.05),
            "C6_capability": {
                "capability_id": "handoff.fem_to_dem",
                "claimed_level": "Implemented",
                "evidence": {
                    "C1_kink_angle_error_deg": c1,
                    "C2_volume_relative_error": float(self.volume["relative_error"]),
                    "C3_momentum_relative_error": float(self.momentum["momentum_relative_error"]),
                    "C4_energy_relative_error": float(self.energy["relative_error"]),
                    "C1": c1 <= 10.0,
                    "C2": bool(self.volume["passed"]),
                    "C3": bool(self.momentum["passed"]),
                    "C4": bool(self.energy["passed"]),
                },
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "k1_anchor": self.k1_anchor,
            "tip_state": self.tip_state,
            "propagation": {k: v for k, v in self.propagation.items() if k != "notch_keep_mask"},
            "fem": self.fem,
            "packet": {
                "version": self.packet["version"],
                "face_count": self.packet["face_count"],
                "area_total_m2": self.packet["area_total_m2"],
                "energy_total_j": self.packet["energy_total_j"],
                "sha256": self.packet["sha256"],
            },
            "fragments": [f.as_dict() for f in self.fragments],
            "volume": self.volume,
            "momentum": self.momentum,
            "energy": self.energy,
            "dem_initial": self.dem_initial,
            "dem_continuation": self.dem_history.as_dict(),
            "criteria": self.criteria(),
        }


def run_protocol(
    *,
    spec: BeamSpec = ICE_SPEC,
    material: FractureMaterial = ICE_MATERIAL,
    dims: tuple[int, int, int] = (48, 12, 4),
    density_kg_m3: float = ICE_DENSITY_KG_M3,
    load_offset_m: float = 0.024,
    target_ratio: float = 1.40,
    velocity_rate_m_s: float = 1.0,
    propagation_steps: int = 3,
    release_height_m: float = 0.10,
    duration_s: float = 0.05,
    dt_s: float = 1.0e-5,
    timestamp: str = DEFAULT_TIMESTAMP,
    source_solver: str = "TensorFEM/hex8-3pb-mixed-mode",
    sample_every: int = 10,
    keep_fields: bool = False,
) -> ProtocolRun:
    """Run the full FEM→DEM handoff protocol and evaluate C1-C4."""
    model = build_bend_model(spec, dims=dims)
    element_volumes_fem = element_volumes(model.mesh)

    # ---- (a) external K_I anchor: symmetric SENB vs the ASTM E399 geometry function
    sym_solution = solve_bend(model, applied_displacement_m=1.0e-5, load_offset_m=0.0)
    sym = stress_intensity_factors(model, sym_solution.solution.u, plane="plane_strain")
    reference = senb_k1_reference(
        load_n=sym_solution.load_pa_total_n,
        span_m=spec.length_m,
        width_m=spec.height_m,
        thickness_m=spec.thickness_m,
        crack_length_m=model.notch_depth_effective_m,
    )
    k1_anchor = {
        "kind": "SENB three-point bend, span/width = 4, ASTM E399 geometry function",
        "load_n": float(sym_solution.load_pa_total_n),
        "fem_k1_pa_sqrt_m": float(sym["k1_pa_sqrt_m"]),
        "fem_k2_pa_sqrt_m": float(sym["k2_pa_sqrt_m"]),
        "reference_k1_pa_sqrt_m": float(reference["k1_pa_sqrt_m"]),
        "relative_error": abs(sym["k1_pa_sqrt_m"] - reference["k1_pa_sqrt_m"])
        / abs(reference["k1_pa_sqrt_m"]),
        "geometry_function": float(reference["geometry_function"]),
        "alpha": float(reference["alpha"]),
        "kappa": float(sym["kappa"]),
        "extrapolation_r2": float(sym["k1_fit_r2"]),
        "samples": sym["samples"],
        "tolerance": 0.15,
        "note": (
            "displacement correlation on a finite-width (mesh-snapped) notch, so the "
            "ASTM sharp-crack reference is approached but not matched exactly"
        ),
    }
    k1_anchor["passed"] = bool(k1_anchor["relative_error"] <= k1_anchor["tolerance"])

    # ---- (b) mixed-mode crack-tip state
    calibration = calibrate_load(
        model,
        material,
        probe_displacement_m=1.0e-5,
        target_ratio=target_ratio,
        load_offset_m=load_offset_m,
    )
    mixed_solution, sigma_mixed = solve_step(
        model,
        applied_displacement_m=calibration["applied_displacement_m"],
        load_offset_m=load_offset_m,
    )
    u_mixed = mixed_solution.u
    tip = stress_intensity_factors(model, u_mixed, plane="plane_strain")
    theta_mts = mts_kink_angle_deg(tip["k1_pa_sqrt_m"], tip["k2_pa_sqrt_m"])
    stress_fit = williams_stress_fit(model, sigma_mixed)
    theta_fem = mts_kink_angle_deg(stress_fit["k1_pa_sqrt_m"], stress_fit["k2_pa_sqrt_m"])
    rings = [
        ring_kink_angle(model, sigma_mixed, radius_m=r)
        for r in (1.5 * model.dy_m, 2.0 * model.dy_m, 3.0 * model.dy_m)
    ]
    tip_state = {
        "k1_pa_sqrt_m": tip["k1_pa_sqrt_m"],
        "k2_pa_sqrt_m": tip["k2_pa_sqrt_m"],
        "k2_over_k1": (
            tip["k2_pa_sqrt_m"] / tip["k1_pa_sqrt_m"] if tip["k1_pa_sqrt_m"] else float("nan")
        ),
        "kappa": tip["kappa"],
        "k1_fit_r2": tip["k1_fit_r2"],
        "k2_fit_r2": tip["k2_fit_r2"],
        "samples": tip["samples"],
        "theta_mts_deg": theta_mts,
        "theta_fem_deg": theta_fem,
        "theta_fem_source": "MTS applied to the Williams stress-field fit (stress side)",
        "stress_fit": stress_fit,
        "stress_fit_k1_pa_sqrt_m": stress_fit["k1_pa_sqrt_m"],
        "stress_fit_k2_pa_sqrt_m": stress_fit["k2_pa_sqrt_m"],
        "stress_fit_r2": stress_fit["fit_r2"],
        "ring_scan_angles_deg": [r["theta_deg"] for r in rings],
        "ring_scan_mean_deg": float(np.mean([r["theta_deg"] for r in rings])),
        "ring_radii_m": [r["radius_m"] for r in rings],
        "load_offset_m": load_offset_m,
        "applied_displacement_m": calibration["applied_displacement_m"],
    }

    # ---- (c) independent FEM-side direction: element-deletion crack advance
    propagation = propagate_crack(
        model,
        applied_displacement_m=calibration["applied_displacement_m"],
        load_offset_m=load_offset_m,
        steps=propagation_steps,
    )

    # ---- (d) failure faces -> frozen v3 packet
    result = extract_failure_faces(
        model.mesh,
        sigma_mixed,
        material=material,
        axis="x",
        coord=model.crack_plane_x_m,
        timestamp=timestamp,
        source_solver=source_solver,
        source_model_id=(
            f"ice-3pb-L{spec.length_m:.3f}-h{spec.height_m:.3f}-a{spec.notch_depth_m:.3f}"
            f"-offset{load_offset_m:.4f}"
        ),
        fracture_layer_thickness_m=model.dx_m,
        ligament_min_coord=("y", model.notch_depth_effective_m),
    )
    packet = packet_from_result(result)
    validate_packet(packet)

    # ---- (e) DEM initial condition
    volume_by_id = {int(e): float(element_volumes_fem[e]) for e in range(model.mesh.n_elem)}
    fragments = build_fragments(
        packet,
        density_kg_m3=density_kg_m3,
        element_volume_by_id=volume_by_id,
        thickness_m=model.dx_m,
    )
    rate_scale = velocity_rate_m_s / max(calibration["applied_displacement_m"], 1e-30)
    _attach_velocity(fragments, mesh=model.mesh, u=u_mixed, rate_scale=rate_scale)

    volume = volume_budget(fragments)
    momentum = momentum_budget(fragments)
    energy = energy_budget(
        fragments,
        producer_internal_energy_j=packet["energy_total_j"],
        reference_height_m=0.0,
    )

    # ---- (f) DEM continuation with the inherited velocities
    fragment_set: FragmentSet = fragments_from_packet(
        packet, density_kg_m3=density_kg_m3, layer_thickness_m=model.dx_m
    )
    dem = RigidDEM(fragment_set, restitution=1.0, friction=0.0, dt_s=dt_s)
    dem.set_velocities([f.velocity_m_s for f in fragments])
    dem.lift_to(release_height_m)
    dem_history = dem.run(duration_s, sample_every=sample_every)

    fem_info = {
        "dims": list(dims),
        "element_count": int(model.mesh.n_elem),
        "node_count": int(model.mesh.n_nodes),
        "dx_m": model.dx_m,
        "dy_m": model.dy_m,
        "notch_depth_effective_m": model.notch_depth_effective_m,
        "notch_gap_effective_m": model.notch_gap_effective_m,
        "crack_plane_x_m": model.crack_plane_x_m,
        "total_load_n": float(sym_solution.load_pa_total_n),
        "calibration": calibration,
        "strain_energy_j": float(sym_solution.strain_energy_j),
        "ligament_failed_fraction": result.diagnostics.get("ligament_failed_fraction"),
        "failures": {
            "max_normal_traction_pa": result.diagnostics.get("max_normal_traction_pa"),
            "failed_face_count": result.diagnostics.get("failed_face_count"),
            "candidate_face_count": result.diagnostics.get("candidate_face_count"),
            "mean_mode_mixity_failed": result.diagnostics.get("mean_mode_mixity_failed"),
        },
    }
    if keep_fields:
        from .mixed_mode import element_centroids

        fem_info["element_centroid_m"] = element_centroids(model.mesh).tolist()
        fem_info["element_stress_voigt_pa"] = np.asarray(sigma_mixed, dtype=float).tolist()
        fem_info["element_volume_m3"] = [float(v) for v in element_volumes_fem]
    dem_initial = {
        "fragment_count": len(fragments),
        "total_mass_kg": float(math.fsum(f.mass_kg for f in fragments)),
        "velocity_rate_m_s": velocity_rate_m_s,
        "rate_scale_1_per_s": rate_scale,
        "initial_ke_dem_j": energy["ke_dem_j"],
        "initial_ke_fem_j": energy["ke_fem_j"],
        "release_height_m": release_height_m,
        "duration_s": duration_s,
        "dt_s": dt_s,
        "max_speed_m_s": float(max(np.linalg.norm(f.velocity_m_s) for f in fragments)),
    }
    settings = {
        "load_offset_m": load_offset_m,
        "target_ratio": target_ratio,
        "density_kg_m3": density_kg_m3,
        "thickness_m": model.dx_m,
        "velocity_rate_m_s": velocity_rate_m_s,
        "propagation_steps": propagation_steps,
        "release_height_m": release_height_m,
        "duration_s": duration_s,
        "timestamp": timestamp,
        "source_solver": source_solver,
    }
    return ProtocolRun(
        spec=spec,
        material=material,
        settings=settings,
        k1_anchor=k1_anchor,
        tip_state=tip_state,
        propagation=propagation,
        fem=fem_info,
        result=result,
        packet=packet,
        fragments=fragments,
        volume=volume,
        momentum=momentum,
        energy=energy,
        dem_history=dem_history,
        dem_initial=dem_initial,
    )


def gate_sensitivity(run: ProtocolRun) -> list[dict[str, Any]]:
    """Probe that the C1-C4 gates actually discriminate.

    A gate that cannot fail is worthless, so every criterion is re-evaluated
    against a deliberately corrupted copy of the same handoff.  Each case reports
    the corrupted quantity, the observed error and whether the gate turned red
    (``detected``).  The first case is the untouched control and must stay green.
    """
    import copy

    from .transfer import TransferFragment

    base = run.criteria()
    cases: list[dict[str, Any]] = []

    def record(name: str, quantity: str, value: float, tolerance: float, control: bool = False) -> None:
        cases.append(
            {
                "case": name,
                "control": control,
                "quantity": quantity,
                "observed_error": float(value),
                "tolerance": float(tolerance),
                "detected": bool(value > tolerance),
                "expected_detected": not control,
            }
        )

    record("control_unchanged", "C1 kink angle error (deg)", base["C1_kink_angle_error_deg"], 10.0, True)
    record("control_unchanged", "C2 volume relative error", base["C2_volume_relative_error"], 0.03, True)
    record("control_unchanged", "C3 momentum relative error", base["C3_momentum_relative_error"], 0.02, True)
    record("control_unchanged", "C4 energy relative error", base["C4_energy_relative_error"], 0.05, True)

    # (1) the consumer reconstructs with a 5 % thicker fracture layer -> volume gate
    thick = [
        build_transfer_fragment(
            {
                "face_id": f.face_id,
                "element_id": f.element_id,
                "vertices_m": f.vertices_m.tolist(),
                "normal": (f.normal / np.linalg.norm(f.normal)).tolist(),
                "energy_j": f.fracture_energy_j,
            },
            density_kg_m3=ICE_DENSITY_KG_M3,
            thickness_m=f.thickness_m * 1.05,
            index=i,
            fem_element_volume_m3=f.fem_element_volume_m3,
        )
        for i, f in enumerate(run.fragments)
    ]
    record("layer_thickness_plus_5pct", "C2 volume relative error", volume_budget(thick)["relative_error"], 0.03)

    # (2) one fragment is silently dropped on the consumer side -> momentum gate
    dropped = copy.deepcopy(list(run.fragments)[:-1])
    reference = run.momentum["momentum_magnitude_fem"]
    kept = sum(f.mass_kg * f.velocity_m_s for f in dropped)
    record(
        "fragment_dropped",
        "C3 momentum relative error",
        float(np.linalg.norm(kept - np.asarray(run.momentum["fem_momentum_kg_m_s"]))) / reference,
        0.02,
    )

    # (3) the consumer forgets 20 % of the cohesive energy -> energy gate
    lossy = []
    for i, f in enumerate(run.fragments):
        clone = copy.copy(f)
        clone.fracture_energy_j = f.fracture_energy_j * 0.8
        lossy.append(clone)
    record(
        "fracture_energy_minus_20pct",
        "C4 energy relative error",
        energy_budget(
            lossy,
            producer_internal_energy_j=run.packet["energy_total_j"],
            reference_height_m=0.0,
        )["relative_error"],
        0.05,
    )

    # (4) the crack-tip angle is perturbed by 15 deg -> kink gate
    record(
        "kink_angle_off_by_15deg",
        "C1 kink angle error (deg)",
        abs(float(run.tip_state["theta_fem_deg"]) + 15.0 - float(run.tip_state["theta_mts_deg"])),
        10.0,
    )
    return cases


def write_protocol_artifacts(run: ProtocolRun, out_dir: str | Path) -> dict[str, str]:
    """Write packet, handoff bundle, budgets and criteria to ``out_dir``."""
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def dump(name: str, payload: Any) -> None:
        target = out / name
        target.write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, default=float) + "\n",
            encoding="utf-8",
        )
        paths[name.replace(".json", "")] = str(target)

    paths["packet"] = str(out / "failure_faces.json")
    write_packet(run.packet, paths["packet"])
    dump(
        "handoff_bundle.json",
        {
            "packet_sha256": packet_digest(run.packet),
            "packet_version": run.packet["version"],
            "contract": "failure_faces.v3 (frozen, extended by this companion bundle only)",
            "crack_tip": {k: v for k, v in run.tip_state.items() if k != "samples"},
            "crack_tip_samples": run.tip_state["samples"],
            "propagation": {k: v for k, v in run.propagation.items() if k != "notch_keep_mask"},
            "k1_anchor": run.k1_anchor,
            "fragments": [f.as_dict() for f in run.fragments],
        },
    )
    dump("volume_budget.json", run.volume)
    dump("momentum_budget.json", run.momentum)
    dump("energy_budget.json", run.energy)
    dump("dem_continuation.json", run.dem_history.as_dict())
    dump("criteria_report.json", run.criteria())
    dump("gate_sensitivity.json", gate_sensitivity(run))
    dump("protocol_run.json", run.as_dict())
    return paths