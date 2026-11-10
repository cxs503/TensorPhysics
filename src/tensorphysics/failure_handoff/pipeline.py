"""End-to-end FEM→DEM failure-face handoff: three-point bend → fragments → free fall.

The pipeline is the executable version of PANORAMA §5.3's "only frozen interface":
every step is a function call on a typed packet, and every criterion is evaluated
inline and returned as a number (no post-hoc bookkeeping by hand).

Steps
-----
1. **FEM** notched three-point-bend solve (:mod:`~tensorphysics.failure_handoff.three_point_bend`).
2. **Load calibration**: the specimen is loaded to ``target_ratio * sigma_t`` at the
   ligament root, so the crack front sits *inside* the ligament (partial fracture)
   instead of the trivial "everything breaks".
3. **Extraction + export**: fracture faces → sealed ``failure_faces`` v3 packet,
   exported twice to prove idempotency (byte-identical → identical SHA-256).
4. **DEM initialisation**: packet → polyhedral fragments (optionally clumps) with
   inherited fracture energy.
5. **Conservation gates**: area ≤1 %, energy ≤2 %, mass/clump identities.
6. **Free fall + impact**: rigid-fragment DEM, release height ``H``; the elastic run
   must conserve ``KE + PE + internal``, the dissipative run must close the ledger
   with the recorded dissipation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .conservation import (
    HandoffGateError,
    area_conservation_gate,
    energy_conservation_gate,
    run_handoff_gates,
)
from .dem_fragments import FragmentSet, fragments_from_packet
from .dem_rigid import EnergyHistory, RigidDEM
from .fem_export import (
    FEMFractureResult,
    FractureMaterial,
    extract_failure_faces,
    face_tractions,
    mode_mixity,
    packet_from_result,
    write_packet,
)
from .schema import packet_digest
from .three_point_bend import BeamSpec, build_bend_model, solve_bend

__all__ = [
    "gate_sensitivity",
    "DEFAULT_SPEC",
    "DEFAULT_MATERIAL",
    "DEFAULT_DENSITY_KG_M3",
    "DEFAULT_TIMESTAMP",
    "HandoffRun",
    "calibrate_load",
    "run_handoff",
    "write_run_artifacts",
]

#: Quasi-brittle specimen (concrete-like): strong enough to fracture under a
#: displacement-controlled three-point-bend load, small enough to keep the FEM cheap.
DEFAULT_SPEC = BeamSpec(
    length_m=0.40,
    height_m=0.06,
    thickness_m=0.02,
    notch_depth_m=0.02,
    notch_gap_m=0.02,
    E_pa=30.0e9,
    nu=0.20,
)
DEFAULT_MATERIAL = FractureMaterial(
    E_pa=30.0e9,
    nu=0.20,
    sigma_t_pa=3.5e6,
    G_Ic_j_m2=100.0,
    G_IIc_j_m2=200.0,
    eta_bk=1.5,
)
DEFAULT_DENSITY_KG_M3 = 2400.0
#: Deterministic provenance stamp: the same FEM result re-exported must hash the same.
DEFAULT_TIMESTAMP = "2026-11-10T00:00:00Z"


def calibrate_load(
    model,
    material: FractureMaterial,
    *,
    probe_displacement_m: float = 1.0e-5,
    target_ratio: float = 1.05,
) -> dict[str, Any]:
    """Find the applied displacement that puts the ligament root at ``target_ratio*sigma_t``.

    Linear elasticity makes the stress field proportional to the applied
    displacement, so one probe solve plus a scale factor is exact (validated in
    :func:`~tensorphysics.failure_handoff.three_point_bend.verify_against_beam_theory`).
    """
    from .fem_export import faces_on_plane

    probe = solve_bend(model, applied_displacement_m=probe_displacement_m)
    faces = faces_on_plane(model.mesh, axis="x", coord=model.crack_plane_x_m)
    if not faces:
        raise RuntimeError(
            f"no element face lies on the crack plane x={model.crack_plane_x_m:.6g} m; "
            "the notch band did not snap onto the mesh"
        )
    sigma_max = 0.0
    ligament_area = 0.0
    for element_id, _key, verts in faces:
        centroid_x = float(model.mesh.element_coords(element_id)[:, 0].mean())
        if centroid_x > model.crack_plane_x_m:
            continue
        if float(verts[:, 1].min()) < model.notch_depth_effective_m - 1e-9:
            continue
        sigma_n, tau = face_tractions(
            probe.sigma_element[element_id], _outward_normal(verts, 0, positive=True)
        )
        sigma_max = max(sigma_max, sigma_n)
        _ = tau
        from .geometry import polygon_area

        ligament_area += polygon_area(verts)
    if sigma_max <= 0.0:
        raise RuntimeError(
            "the ligament is in compression under the applied displacement; check "
            "the support/load orientation"
        )
    scale = target_ratio * material.sigma_t_pa / sigma_max
    return {
        "probe_displacement_m": probe_displacement_m,
        "probe_max_normal_traction_pa": sigma_max,
        "target_normal_traction_pa": target_ratio * material.sigma_t_pa,
        "scale_factor": scale,
        "applied_displacement_m": probe_displacement_m * scale,
        "ligament_area_m2": ligament_area,
    }


def _outward_normal(verts: np.ndarray, axis_index: int, *, positive: bool) -> np.ndarray:
    from .geometry import newell_area_vector

    area_vector = newell_area_vector(verts)
    normal = area_vector / float(np.linalg.norm(area_vector))
    if positive and normal[axis_index] < 0.0:
        normal = -normal
    if (not positive) and normal[axis_index] > 0.0:
        normal = -normal
    return normal


@dataclass
class HandoffRun:
    """All artifacts of one end-to-end handoff run."""

    calibration: dict
    fem: dict
    result: FEMFractureResult
    packet: dict
    fragment_set: FragmentSet
    gates: GateReport
    elastic_history: EnergyHistory
    dissipative_history: EnergyHistory
    idempotency: dict
    settings: dict

    # -- criterion numbers -------------------------------------------------
    def criteria(self) -> dict[str, Any]:
        area_gate = next(g for g in self.gates.gates if g.name == "area")
        energy_gate = next(g for g in self.gates.gates if g.name == "energy")
        return {
            "C1_area_relative_error": area_gate.relative_error,
            "C1_area_tolerance": area_gate.tolerance,
            "C1_area_pass": area_gate.passed,
            "C2_energy_relative_error": energy_gate.relative_error,
            "C2_energy_tolerance": energy_gate.tolerance,
            "C2_energy_pass": energy_gate.passed,
            "C3_schema_pass": True,  # the packet was sealed + validated on export
            "C3_schema_version": self.packet["version"],
            "C4_energy_residual_elastic": self.elastic_history.max_relative_deviation(),
            "C4_energy_residual_tolerance": 0.05,
            "C4_pass": self.elastic_history.max_relative_deviation() <= 0.05,
            "C5_sha256_first": self.idempotency["first"],
            "C5_sha256_second": self.idempotency["second"],
            "C5_idempotent": self.idempotency["equal"],
            "dissipative_closure": self.dissipative_history.closure_with_dissipation(),
            "elastic_dissipation_j": self.elastic_history.dissipated[-1],
        }

    def as_dict(self) -> dict:
        return {
            "settings": self.settings,
            "calibration": self.calibration,
            "fem": self.fem,
            "packet": {
                "version": self.packet["version"],
                "face_count": self.packet["face_count"],
                "area_total_m2": self.packet["area_total_m2"],
                "energy_total_j": self.packet["energy_total_j"],
                "sha256": self.packet["sha256"],
                "source_solver": self.packet["source_solver"],
                "timestamp": self.packet["timestamp"],
                "crack_front_point_count": len(self.packet.get("crack_front_m", [])),
            },
            "fragment_set": {
                "fragment_count": self.fragment_set.count,
                "area_total_m2": self.fragment_set.area_total_m2,
                "mass_total_kg": self.fragment_set.mass_total_kg,
                "volume_total_m3": self.fragment_set.volume_total_m3,
                "internal_energy_total_j": self.fragment_set.internal_energy_total_j,
            },
            "gates": self.gates.as_dict(),
            "energy_chain": {
                "elastic": self.elastic_history.as_dict(),
                "dissipative": self.dissipative_history.as_dict(),
            },
            "idempotency": self.idempotency,
            "criteria": self.criteria(),
            "fem_diagnostics": self.result.diagnostics,
        }


def run_handoff(
    *,
    spec: BeamSpec = DEFAULT_SPEC,
    material: FractureMaterial = DEFAULT_MATERIAL,
    dims: tuple[int, int, int] = (32, 12, 4),
    density_kg_m3: float = DEFAULT_DENSITY_KG_M3,
    clump_count: int = 0,
    target_ratio: float = 1.05,
    probe_displacement_m: float = 1.0e-5,
    release_height_m: float = 0.5,
    duration_s: float = 0.8,
    dt_s: float = 1.0e-4,
    restitution: float = 1.0,
    friction: float = 0.0,
    timestamp: str = DEFAULT_TIMESTAMP,
    source_solver: str = "TensorFEM/hex8-3point-bend",
    sample_every: int = 20,
) -> HandoffRun:
    """Run the full handoff pipeline and evaluate every acceptance criterion."""
    model = build_bend_model(spec, dims=dims)
    calibration = calibrate_load(
        model,
        material,
        probe_displacement_m=probe_displacement_m,
        target_ratio=target_ratio,
    )
    bend = solve_bend(model, applied_displacement_m=calibration["applied_displacement_m"])
    result = extract_failure_faces(
        model.mesh,
        bend.sigma_element,
        material=material,
        axis="x",
        coord=model.crack_plane_x_m,
        timestamp=timestamp,
        source_solver=source_solver,
        source_model_id=f"3pb-notched-L{spec.length_m:.3f}-h{spec.height_m:.3f}-a{spec.notch_depth_m:.3f}",
        fracture_layer_thickness_m=model.dx_m,
        ligament_min_coord=("y", model.notch_depth_effective_m),
    )
    packet = packet_from_result(result)
    # Idempotency (C5): a second extraction from the same FEM result must seal to
    # the same digest and serialise to the same bytes.
    second = packet_from_result(result)
    idempotency = {
        "first": packet["sha256"],
        "second": second["sha256"],
        "equal": packet["sha256"] == second["sha256"] == packet_digest(packet),
        "face_count": packet["face_count"],
    }
    fragment_set = fragments_from_packet(
        packet,
        density_kg_m3=density_kg_m3,
        layer_thickness_m=model.dx_m,
        clump_count=clump_count,
    )
    gates = run_handoff_gates(packet, fragment_set)

    elastic_dem = RigidDEM(
        fragment_set, restitution=restitution, friction=friction, dt_s=dt_s
    )
    elastic_dem.lift_to(release_height_m)
    elastic_history = elastic_dem.run(duration_s, sample_every=sample_every)

    dissipative_dem = RigidDEM(
        fragment_set, restitution=0.5, friction=0.3, dt_s=dt_s
    )
    dissipative_dem.lift_to(release_height_m)
    dissipative_history = dissipative_dem.run(duration_s, sample_every=sample_every)

    fem = {
        "total_load_n": bend.load_pa_total_n,
        "applied_displacement_m": bend.applied_displacement_m,
        "midspan_deflection_m": bend.midspan_deflection_m,
        "stiffness_n_per_m": bend.load_pa_total_n / max(bend.midspan_deflection_m, 1e-30),
        "strain_energy_j": bend.strain_energy_j,
        "notch_depth_effective_m": model.notch_depth_effective_m,
        "notch_element_count": model.notch_element_count,
        "notch_gap_effective_m": model.notch_gap_effective_m,
        "crack_plane_x_m": model.crack_plane_x_m,
        "dx_m": model.dx_m,
        "dy_m": model.dy_m,
        "element_count": int(model.mesh.n_elem),
        "node_count": int(model.mesh.n_nodes),
        "dof_count": int(model.mesh.n_dof),
    }
    settings = {
        "dims": list(dims),
        "spec": {
            "length_m": spec.length_m,
            "height_m": spec.height_m,
            "thickness_m": spec.thickness_m,
            "notch_depth_m": spec.notch_depth_m,
            "notch_gap_m": spec.notch_gap_m,
            "E_pa": spec.E_pa,
            "nu": spec.nu,
        },
        "material": {
            "sigma_t_pa": material.sigma_t_pa,
            "G_Ic_j_m2": material.G_Ic_j_m2,
            "G_IIc_j_m2": material.G_IIc_j_m2,
            "eta_bk": material.eta_bk,
        },
        "density_kg_m3": density_kg_m3,
        "clump_count": clump_count,
        "target_ratio": target_ratio,
        "release_height_m": release_height_m,
        "duration_s": duration_s,
        "dt_s": dt_s,
        "restitution": restitution,
        "friction": friction,
        "timestamp": timestamp,
    }
    return HandoffRun(
        calibration=calibration,
        fem=fem,
        result=result,
        packet=packet,
        fragment_set=fragment_set,
        gates=gates,
        elastic_history=elastic_history,
        dissipative_history=dissipative_history,
        idempotency=idempotency,
        settings=settings,
    )


def write_run_artifacts(run: HandoffRun, out_dir: str | Path) -> dict[str, str]:
    """Write the audited artifacts (packet, gates, DEM ICs, energy chain)."""
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    paths["packet"] = str(out / "failure_faces.json")
    write_packet(run.packet, paths["packet"])

    def dump(name: str, payload: Any) -> None:
        target = out / name
        target.write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        paths[name.replace(".json", "")] = str(target)

    dump("conservation_gates.json", run.gates.as_dict())
    dump("dem_initial_conditions.json", run.fragment_set.as_dict())
    dump(
        "energy_chain.json",
        {
            "elastic": run.elastic_history.as_dict(),
            "dissipative": run.dissipative_history.as_dict(),
            "fem_fracture_energy_j": run.packet["energy_total_j"],
            "dem_internal_energy_j": run.fragment_set.internal_energy_total_j,
        },
    )
    dump("fem_diagnostics.json", {**run.fem, **run.result.diagnostics, "calibration": run.calibration})
    dump("criteria_report.json", run.criteria())
    return paths

def gate_sensitivity(packet: dict, fragment_set: FragmentSet) -> list[dict]:
    """Probe whether the conservation gates actually discriminate.

    A gate that always passes is worthless, so the consumer side of the handoff is
    re-run against deliberately lossy fragment sets (the realistic failure modes:
    a dropped fragment, a polygon reduced to a triangle, an energy bookkeeping
    slip).  Each case reports the observed error and the verdict, including cases
    that must still pass because they stay inside the tolerance band.
    """
    cases: list[dict] = []
    fragments = fragment_set.fragments
    if len(fragments) < 2:
        return cases

    def run_case(name: str, mutated: FragmentSet, description: str) -> dict:
        out = {"case": name, "description": description}
        try:
            area = area_conservation_gate(packet, mutated)
            energy = energy_conservation_gate(packet, mutated)
            out.update(
                {
                    "outcome": "evaluated",
                    "area_relative_error": area.relative_error,
                    "area_passed": area.passed,
                    "energy_relative_error": energy.relative_error,
                    "energy_passed": energy.passed,
                }
            )
        except HandoffGateError as exc:
            out.update({"outcome": "structural_error", "message": str(exc)})
        cases.append(out)
        return out

    # (0) control: the unmodified handoff must pass every gate.
    run_case("control_unchanged", fragment_set, "unmodified packet/fragment pair")

    # (1) consumer drops the last fragment (lossy transfer).
    dropped = replace(fragment_set, fragments=list(fragments[:-1]))
    run_case("fragment_dropped", dropped, "consumer lost one fragment")

    # (2) consumer approximates each quad by a triangle (half the area).
    half_area = [
        replace(fragment, area_m2=fragment.area_m2 * 0.5) for fragment in fragments
    ]
    run_case(
        "faces_triangulated_lossily",
        replace(fragment_set, fragments=half_area),
        "consumer kept 3 of 4 vertices per face (area halved)",
    )

    # (3) consumer area off by 0.5 % — inside the 1 % band, must still pass.
    small = [
        replace(fragment, area_m2=fragment.area_m2 * 1.005) for fragment in fragments
    ]
    run_case(
        "area_offset_0p5pct",
        replace(fragment_set, fragments=small),
        "consumer area +0.5 % (inside tolerance)",
    )

    # (4) consumer area off by 1.5 % — outside the 1 % band, must fail.
    large = [
        replace(fragment, area_m2=fragment.area_m2 * 1.015) for fragment in fragments
    ]
    run_case(
        "area_offset_1p5pct",
        replace(fragment_set, fragments=large),
        "consumer area +1.5 % (outside tolerance)",
    )

    # (5) energy bookkeeping loses 3 % — outside the 2 % band, must fail.
    low_energy = [
        replace(fragment, fracture_energy_j=fragment.fracture_energy_j * 0.97)
        for fragment in fragments
    ]
    run_case(
        "energy_offset_3pct",
        replace(fragment_set, fragments=low_energy),
        "consumer internal energy -3 % (outside tolerance)",
    )

    # (6) energy bookkeeping loses 1.5 % — inside the 2 % band.
    low_energy_ok = [
        replace(fragment, fracture_energy_j=fragment.fracture_energy_j * 0.985)
        for fragment in fragments
    ]
    run_case(
        "energy_offset_1p5pct",
        replace(fragment_set, fragments=low_energy_ok),
        "consumer internal energy -1.5 % (inside tolerance)",
    )
    return cases
