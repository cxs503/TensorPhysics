"""HTTP API for pod-ice interaction analysis.

Exposes ``POST /api/v1/calculations/pod_ice_interaction`` returning the thrust
attainable domain, the ice/housing contact stress (Hertz + FEM refinement) and
the propeller-ice interaction time history with its spectral phase check.

Run standalone with ``uvicorn scenarios.pod_ice_interaction.api:app``.
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field

from .contact import ice_pod_contact_case
from .envelope import IceLoadCase, PodSpec, rated_operating_point, thrust_domain, verify_domain
from .interaction import InteractionConfig, compare_coupling, simulate_interaction, verify_phase
from .propeller import BladeGeometry
from .structure import StrutSection


class PodIceRequest(BaseModel):
    """Input of the pod-ice interaction calculation."""

    power_rated: float = Field(15.0e6, gt=0.0, description="rated motor power [W]")
    n_rated_rpm: float = Field(125.0, gt=0.0, description="rated shaft speed [rpm]")
    diameter: float = Field(6.0, gt=0.0, description="propeller diameter [m]")
    hub_ratio: float = Field(0.35, gt=0.0, lt=1.0)
    n_blades: int = Field(4, ge=2, le=8)
    pitch_ratio: float = Field(0.95, gt=0.0, description="P/D")
    strut_width: float = Field(1.40, gt=0.0, description="strut chord along thrust [m]")
    strut_thickness: float = Field(0.60, gt=0.0, description="strut thickness [m]")
    strut_wall: float = Field(0.08, gt=0.0)
    strut_height: float = Field(5.00, gt=0.0, description="hull root to pod centre [m]")
    yield_stress: float = Field(355.0e6, gt=0.0, description="strut steel yield stress [Pa]")
    allowable_factor: float = Field(0.60, gt=0.0, le=1.0)
    v_ship: float = Field(8.0, gt=0.0, description="ship speed [m/s]")
    ice_thickness: float = Field(1.5, gt=0.0, description="level ice thickness [m]")
    ice_crushing_pressure: float = Field(0.8e6, gt=0.0, description="effective ice pressure [Pa]")
    ice_contact_width: float = Field(1.0, gt=0.0, description="ice/housing contact width [m]")
    ice_radius: float = Field(1.0, gt=0.0, description="local ice edge radius [m]")
    housing_radius: float = Field(1.20, gt=0.0, description="pod housing radius [m]")
    azimuths_deg: list[float] = Field(default_factory=lambda: [float(a) for a in range(0, 181, 15)])
    ice_torque_mean: float = Field(0.30e6, ge=0.0, description="mean ice torque [N m]")
    ice_torque_duty: float = Field(0.18, gt=0.0, le=0.25)
    interaction_rpm: float | None = Field(None, gt=0.0, description="commanded rpm for the interaction run")
    interaction_duration: float = Field(4.8, gt=0.0)
    interaction_dt: float = Field(2.0e-4, gt=0.0)
    contact_h_ratio: float = Field(0.20, gt=0.0)
    include_contact: bool = True
    include_interaction: bool = True


def compute(req: PodIceRequest) -> dict:
    """Run the complete pod-ice analysis for one request."""
    geom = BladeGeometry(
        diameter=req.diameter,
        hub_ratio=req.hub_ratio,
        n_blades=req.n_blades,
        pitch_ratio=req.pitch_ratio,
    )
    strut = StrutSection(
        width=req.strut_width,
        thickness=req.strut_thickness,
        wall=req.strut_wall,
        height=req.strut_height,
        yield_stress=req.yield_stress,
        allowable_factor=req.allowable_factor,
    )
    pod = PodSpec(
        power_rated=req.power_rated,
        n_rated_rpm=req.n_rated_rpm,
        geometry=geom,
        strut=strut,
    )
    ice = IceLoadCase(
        thickness=req.ice_thickness,
        crushing_pressure=req.ice_crushing_pressure,
        contact_width=req.ice_contact_width,
    )
    curve = pod.curve()
    op = rated_operating_point(pod, curve, req.v_ship)
    domain = thrust_domain(
        pod, ice, v_ship=req.v_ship, azimuths_deg=tuple(req.azimuths_deg), curve=curve
    )
    check = verify_domain(pod, ice, domain)

    out: dict = {
        "status": "ok",
        "operating_point": op,
        "motor_torque_limit": pod.motor_torque_limit,
        "ice_characteristic_force": ice.characteristic_force,
        "thrust_domain": {
            "stations": domain["stations"],
            "motor_limit": domain["motor_limit"]["thrust"],
            "bearing_limit": domain["bearing_limit"],
        },
        "structural_check": {
            "allowable_stress": strut.allowable_stress,
            "max_utilization_inside": check["max_utilization_inside"],
            "overstressed_points": len(check["overstressed_points"]),
            "boundary_error_max": check["boundary_error_max"],
            "maximality_violations": check["maximality_violations"],
            "n_structure_governed": check["n_structure_governed"],
            "passed": check["passed"],
        },
        "verdicts": {
            "domain_matches_strength_constraint": bool(check["passed"]),
        },
    }

    if req.include_contact:
        contact = ice_pod_contact_case(
            ice.characteristic_force,
            ice_radius=req.ice_radius,
            housing_radius=req.housing_radius,
            h_ratio=req.contact_h_ratio,
        )
        out["contact"] = {
            "contact_radius": contact["contact_radius"],
            "peak_contact_pressure": contact["peak_contact_pressure"],
            "mean_contact_pressure": contact["mean_contact_pressure"],
            "approach": contact["approach"],
            "fem_max_von_mises": contact["fem"]["fem_max_von_mises"],
            "fem_max_over_p0": contact["fem"]["fem_max_over_p0"],
            "fem_max_depth_over_a": contact["fem"]["fem_max_depth_over_a"],
            "hertz_theory_max_von_mises": contact["hertz_max_von_mises"],
            "von_mises_deviation_rel": contact["von_mises_deviation_rel"],
            "displacement_deviation_rel": contact["displacement_deviation_rel"],
            "max_von_mises_location": {
                "radius_over_a": 0.0,
                "depth_over_a": contact["fem"]["fem_max_depth_over_a"],
            },
            "pressure_profile": {
                "radius": contact["profile_radius"].tolist(),
                "pressure": contact["profile_pressure"].tolist(),
            },
        }
        out["verdicts"]["contact_within_10pct_of_hertz"] = bool(contact["contact_pressure_ok"])

    if req.include_interaction:
        cfg = InteractionConfig(
            ice_torque_mean=req.ice_torque_mean,
            blade_duty=req.ice_torque_duty,
            duration=req.interaction_duration,
            dt=req.interaction_dt,
        )
        rpm = req.interaction_rpm or pod.n_rated_rpm
        one_way = simulate_interaction(
            curve, pod, ice, cfg, v_ship=req.v_ship, two_way=False, n_cmd_rpm=rpm
        )
        two_way = simulate_interaction(
            curve, pod, ice, cfg, v_ship=req.v_ship, two_way=True, n_cmd_rpm=rpm
        )
        phase = verify_phase(one_way, curve, pod, cfg)
        coupling = compare_coupling(one_way, two_way)
        step = max(1, one_way["t"].size // 400)
        out["interaction"] = {
            "commanded_rpm": rpm,
            "mean_rpm": float(np.mean(one_way["n_rpm"])),
            "min_rpm": float(np.min(one_way["n_rpm"])),
            "mean_thrust": float(np.mean(one_way["thrust"])),
            "mean_ice_force": float(np.mean(one_way["ice_force"])),
            "blade_frequency": phase["blade_frequency"],
            "phase_diff_analytic_deg": phase["phase_diff_analytic_deg"],
            "phase_diff_measured_deg": phase["phase_diff_measured_deg"],
            "phase_error_deg": phase["phase_error_deg"],
            "amplitude_ratio_analytic": phase["amp_ratio_analytic"],
            "amplitude_ratio_measured": phase["amp_ratio_measured"],
            "amplitude_error_rel": phase["amp_error_rel"],
            "motor_saturated": phase["motor_saturated"],
            "peak_torque_clipping": phase["peak_torque_clipping"],
            "coupling": coupling,
            "history": {
                "t": one_way["t"][::step].tolist(),
                "thrust": one_way["thrust"][::step].tolist(),
                "torque_prop": one_way["torque_prop"][::step].tolist(),
                "torque_ice": one_way["torque_ice"][::step].tolist(),
                "ice_force": one_way["ice_force"][::step].tolist(),
                "n_rpm": one_way["n_rpm"][::step].tolist(),
            },
        }
        out["verdicts"]["phase_relation_correct"] = bool(phase["passed"])

    out["verdicts"]["all_checks_passed"] = all(out["verdicts"].values())
    return out


router = APIRouter(prefix="/api/v1/calculations", tags=["pod-ice"])


@router.post("/pod_ice_interaction")
def pod_ice_interaction(req: PodIceRequest) -> dict:
    """Podded propulsor thrust vectoring and ice-hull load interaction."""
    return compute(req)


def create_app() -> FastAPI:
    app = FastAPI(
        title="TensorPhysics pod-ice interaction service",
        version="1.0.0",
        description="Podded propulsor thrust attainable domain, ice-housing contact "
                    "stress and propeller-ice interaction coupling.",
    )
    app.include_router(router)
    return app


app = create_app()