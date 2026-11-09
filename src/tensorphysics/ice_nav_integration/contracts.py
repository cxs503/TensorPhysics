"""Cross-domain interface contracts for the ice-navigation end-to-end gate (IN66-N08).

Three handoffs are contract-bearing in the DEM -> RANS -> FSI -> resistance ->
energy-ledger loop:

``ice_field_volume_fraction.v1``  (DEM -> RANS)
    The Lagrangian floe population is handed to the flow solver as an Eulerian
    volume-fraction field ``alpha_i(x, y, t)`` on a declared Cartesian grid.  A
    field is only usable downstream when it carries its grid descriptor
    (origin, spacing, frame), its ice thickness and density, and the *conserved*
    ice volume; otherwise the flow solver cannot reconstruct the porous-drag
    length scale or the ice mass, and any resistance number is off by an
    arbitrary area factor.

``hull_traction_handoff.v1``      (RANS -> FSI)
    The flow solver returns per-panel pressure, wall shear and the streamwise
    momentum budget (per-panel ``dp/dx``, body force, water area, mass flux).
    Carrying the budget lets the consumer *verify* the discretised momentum
    equation instead of trusting the numbers.

``structural_feedback_handoff.v1`` (FSI -> DEM)
    The structural solver returns the panel normal deflection, the resulting
    hull-ice clearance field and the packing (crush) pressure, plus the
    fixed-point residual.  The DEM stage consumes the clearance field; a
    feedback without a residual cannot be certified as converged.

The validators are pure functions over plain JSON payloads: they never invent a
number, they only reject what is dimensionally inconsistent, grid-mismatched,
out of sync in time, or contract-incomplete.  Every check carries a stable
``check_id`` so gate reports and regression diffs can be diffed mechanically.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCHEMA_ICE_FIELD = "ice_field_volume_fraction.v1"
SCHEMA_HULL_TRACTION = "hull_traction_handoff.v1"
SCHEMA_STRUCT_FEEDBACK = "structural_feedback_handoff.v1"

#: C2 acceptance: interface dimension / grid / time-step mismatch < 1 %.
TOL_INTERFACE = 1.0e-2
#: internal conservation bookkeeping is required to be far tighter than the
#: acceptance bound, so that a real interface defect cannot hide inside the
#: discretisation noise of the acceptance metric.
TOL_CONSERVATION = 1.0e-6
#: consumer/source step ratio and clock-drift bound for the subcycled loop.
TOL_TIME_SYNC = 1.0e-2

#: base dimensions needed by this loop (kg, m, s) -- enough to prove that
#: pressure, wall shear, volume, density and power agree among the modules.
UNIT_DIM: Dict[str, Dict[str, int]] = {
    "1": {},
    "s": {"s": 1},
    "m": {"m": 1},
    "kg": {"kg": 1},
    "K": {"K": 1},
    "m/s": {"m": 1, "s": -1},
    "m/s^2": {"m": 1, "s": -2},
    "m^2": {"m": 2},
    "m^3": {"m": 3},
    "kg/m^3": {"kg": 1, "m": -3},
    "kg/m^2": {"kg": 1, "m": -2},
    "N": {"kg": 1, "m": 1, "s": -2},
    "N/m": {"kg": 1, "s": -2},
    "N/m^3": {"kg": 1, "m": -2, "s": -2},
    "Pa": {"kg": 1, "m": -1, "s": -2},
    "Pa/m": {"kg": 1, "m": -2, "s": -2},
    "J": {"kg": 1, "m": 2, "s": -2},
    "W": {"kg": 1, "m": 2, "s": -3},
    "1/s": {"s": -1},
    "rad/s": {"s": -1},
    "Hz": {"s": -1},
    "m^2/s": {"m": 2, "s": -1},
    "kg/s": {"kg": 1, "s": -1},
}


class ContractError(ValueError):
    """Raised when a payload violates a contract in a non-recoverable way."""


# ---------------------------------------------------------------------------
# dimensional algebra
# ---------------------------------------------------------------------------


def unit_dimension(unit: str) -> Dict[str, int]:
    """Exponent vector (kg, m, s, ...) of a declared unit string."""
    try:
        return dict(UNIT_DIM[unit])
    except KeyError:  # pragma: no cover - defensive
        raise ContractError("unknown unit %r" % (unit,)) from None


def same_dimension(unit_a: str, unit_b: str) -> bool:
    return unit_dimension(unit_a) == unit_dimension(unit_b)


def product_dimension(*terms: Sequence[Any]) -> Dict[str, int]:
    """Dimension of ``prod(unit ** exponent)`` given ``(unit, exponent)`` terms."""
    total: Dict[str, int] = {}
    for unit, exponent in terms:
        for base, power in unit_dimension(unit).items():
            total[base] = total.get(base, 0) + power * exponent
    return {k: v for k, v in total.items() if v != 0}


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One contract check with a stable id."""

    check_id: str
    passed: bool
    value: Any
    tolerance: Optional[float]
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": "PASS" if self.passed else "FAIL",
            "value": self.value,
            "tolerance": self.tolerance,
            "detail": self.detail,
        }


def _rel(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def all_passed(checks: Sequence[Check]) -> bool:
    return all(c.passed for c in checks)


def summarize(checks: Sequence[Check]) -> Dict[str, Any]:
    violations = [
        float(c.value) / c.tolerance
        for c in checks
        if (not c.passed) and isinstance(c.value, (int, float)) and c.tolerance
    ]
    return {
        "n_checks": len(checks),
        "n_pass": sum(1 for c in checks if c.passed),
        "n_fail": sum(1 for c in checks if not c.passed),
        "worst_violation_ratio": max(violations, default=0.0),
        "checks": [c.as_dict() for c in checks],
    }


# ---------------------------------------------------------------------------
# canonical hashing (provenance)
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    raise TypeError("not JSON serialisable: %r" % (type(obj),))


def canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def grid_hash(descriptor: Mapping[str, Any]) -> str:
    """Stable identity of a grid descriptor (spacing + origin + shape + frame)."""
    return canonical_hash(dict(descriptor))[:16]


# ---------------------------------------------------------------------------
# payload builders
# ---------------------------------------------------------------------------

#: unit declarations attached to each handoff payload (checked by the validators)
_UNITS_ICE_FIELD: Dict[str, str] = {
    "time_s": "s",
    "dt_s": "s",
    "thickness_m": "m",
    "ice_density_kg_m3": "kg/m^3",
    "volume_fraction": "1",
    "total_ice_volume_m3": "m^3",
    "total_ice_mass_kg": "kg",
    "mean_volume_fraction": "1",
    "nominal_coverage": "1",
    "upstream_velocity_m_s": "m/s",
}

_UNITS_HULL_TRACTION: Dict[str, str] = {
    "time_s": "s",
    "dt_s": "s",
    "source_dt_s": "s",
    "mass_flux_kg_s": "kg/s",
    "p_pa": "Pa",
    "tau_wall_pa": "Pa",
    "u_m_s": "m/s",
    "dpdx_pa_per_m": "Pa/m",
    "du_dx_1_per_s": "1/s",
    "source_N_per_m": "N/m",
    "water_area_m2": "m^2",
    "wetted_area_m2": "m^2",
    "projected_area_m2": "m^2",
}

_UNITS_STRUCT_FEEDBACK: Dict[str, str] = {
    "time_s": "s",
    "dt_s": "s",
    "source_dt_s": "s",
    "gap0_m": "m",
    "deflection_m": "m",
    "clearance_m": "m",
    "contact_pressure_pa": "Pa",
    "residual": "1",
}


def _attach_units(payload: Dict[str, Any], units: Mapping[str, str]) -> Dict[str, Any]:
    payload["units"] = dict(units)
    return payload



def build_ice_field_payload(
    *,
    grid: Mapping[str, Any],
    time_s: float,
    dt_s: float,
    thickness_m: float,
    ice_density_kg_m3: float,
    nominal_coverage: float,
    volume_fraction: Sequence[Sequence[float]],
    upstream_velocity_m_s: float,
) -> Dict[str, Any]:
    """DEM -> RANS handoff payload (mass bookkeeping computed, never typed in)."""
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    dx = float(grid["dx_m"])
    dy = float(grid["dy_m"])
    cell_volume = dx * dy * thickness_m
    total = 0.0
    weighted = 0.0
    for row in volume_fraction:
        for a in row:
            total += float(a) * cell_volume
            weighted += float(a)
    mean_alpha = weighted / float(nx * ny)
    return _attach_units({
        "schema": SCHEMA_ICE_FIELD,
        "producer": "dem_ice_field",
        "consumer": "rans_streamwise",
        "time_s": float(time_s),
        "dt_s": float(dt_s),
        "frame": str(grid.get("frame", "")),
        "grid": dict(grid),
        "grid_hash": grid_hash(grid),
        "thickness_m": float(thickness_m),
        "ice_density_kg_m3": float(ice_density_kg_m3),
        "nominal_coverage": float(nominal_coverage),
        "mean_volume_fraction": float(mean_alpha),
        "total_ice_volume_m3": float(total),
        "total_ice_mass_kg": float(total * ice_density_kg_m3),
        "upstream_velocity_m_s": float(upstream_velocity_m_s),
        "volume_fraction": [[float(a) for a in row] for row in volume_fraction],
    }, _UNITS_ICE_FIELD)


def build_traction_payload(
    *,
    grid: Mapping[str, Any],
    time_s: float,
    dt_s: float,
    source_dt_s: float,
    panels: Sequence[Mapping[str, float]],
    mass_flux_kg_s: float,
    totals: Mapping[str, float],
) -> Dict[str, Any]:
    """RANS -> FSI handoff payload incl. the verified momentum budget."""
    return _attach_units({
        "schema": SCHEMA_HULL_TRACTION,
        "producer": "rans_streamwise",
        "consumer": "fsi_panel",
        "time_s": float(time_s),
        "dt_s": float(dt_s),
        "source_dt_s": float(source_dt_s),
        "grid": dict(grid),
        "grid_hash": grid_hash(grid),
        "mass_flux_kg_s": float(mass_flux_kg_s),
        "n_panels": len(panels),
        "panels": [
            {
                "x_m": float(p["x_m"]),
                "u_m_s": float(p["u_m_s"]),
                "p_pa": float(p["p_pa"]),
                "tau_wall_pa": float(p["tau_wall_pa"]),
                "dpdx_pa_per_m": float(p["dpdx_pa_per_m"]),
                "du_dx_1_per_s": float(p["du_dx_1_per_s"]),
                "source_N_per_m": float(p["source_N_per_m"]),
                "water_area_m2": float(p["water_area_m2"]),
                "wetted_area_m2": float(p["wetted_area_m2"]),
                "projected_area_m2": float(p["projected_area_m2"]),
            }
            for p in panels
        ],
        "totals": {k: float(v) for k, v in totals.items()},
    }, _UNITS_HULL_TRACTION)


def build_struct_feedback_payload(
    *,
    grid: Mapping[str, Any],
    time_s: float,
    dt_s: float,
    source_dt_s: float,
    deflection_m: Sequence[float],
    clearance_m: Sequence[float],
    contact_pressure_pa: Sequence[float],
    gap0_m: float,
    residual: float,
    iterations: int,
    converged: bool,
    spectral_radius: float,
    added_mass_ratio: float,
) -> Dict[str, Any]:
    """FSI -> DEM handoff payload (deflection / clearance / packing pressure)."""
    return _attach_units({
        "schema": SCHEMA_STRUCT_FEEDBACK,
        "producer": "fsi_panel",
        "consumer": "dem_ice_field",
        "time_s": float(time_s),
        "dt_s": float(dt_s),
        "source_dt_s": float(source_dt_s),
        "grid": dict(grid),
        "grid_hash": grid_hash(grid),
        "gap0_m": float(gap0_m),
        "deflection_m": [float(v) for v in deflection_m],
        "clearance_m": [float(v) for v in clearance_m],
        "contact_pressure_pa": [float(v) for v in contact_pressure_pa],
        "fixed_point": {
            "residual": float(residual),
            "iterations": int(iterations),
            "converged": bool(converged),
            "spectral_radius": float(spectral_radius),
            "added_mass_ratio": float(added_mass_ratio),
        },
    }, _UNITS_STRUCT_FEEDBACK)


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------

_REQUIRED_UNITS_ICE = {
    "time_s": "s",
    "dt_s": "s",
    "thickness_m": "m",
    "ice_density_kg_m3": "kg/m^3",
    "volume_fraction": "1",
    "total_ice_volume_m3": "m^3",
    "upstream_velocity_m_s": "m/s",
}

_REQUIRED_UNITS_TRACTION = {
    "time_s": "s",
    "dt_s": "s",
    "mass_flux_kg_s": "kg/s",
    "p_pa": "Pa",
    "tau_wall_pa": "Pa",
    "u_m_s": "m/s",
    "dpdx_pa_per_m": "Pa/m",
    "source_N_per_m": "N/m",
    "water_area_m2": "m^2",
    "wetted_area_m2": "m^2",
    "projected_area_m2": "m^2",
}

_REQUIRED_UNITS_FEEDBACK = {
    "time_s": "s",
    "dt_s": "s",
    "gap0_m": "m",
    "deflection_m": "m",
    "clearance_m": "m",
    "contact_pressure_pa": "Pa",
}


def _check_units(prefix: str, declared: Mapping[str, str], required: Mapping[str, str]) -> Check:
    missing = [k for k in required if k not in declared]
    wrong: List[str] = []
    for key, expected in required.items():
        unit = declared.get(key)
        if unit is None:
            continue
        try:
            if not same_dimension(unit, expected):
                wrong.append("%s:%s!=%s" % (key, unit, expected))
        except ContractError as exc:
            wrong.append("%s:%s" % (key, exc))
    ok = not missing and not wrong
    return Check(
        "%s_units_declared" % prefix,
        ok,
        ",".join(missing + wrong) or "complete",
        None,
        "required units present and dimensionally equal to the contract",
    )


def validate_ice_field(payload: Mapping[str, Any]) -> List[Check]:
    """Validate the DEM -> RANS volume-fraction handoff."""
    checks: List[Check] = [
        Check(
            "D1_schema",
            payload.get("schema") == SCHEMA_ICE_FIELD,
            payload.get("schema"),
            None,
            "schema must be %s" % SCHEMA_ICE_FIELD,
        ),
        _check_units("D2", payload.get("units", {}), _REQUIRED_UNITS_ICE),
    ]

    grid = payload.get("grid", {})
    nx, ny = int(grid.get("nx", 0)), int(grid.get("ny", 0))
    dx, dy = float(grid.get("dx_m", 0.0)), float(grid.get("dy_m", 0.0))
    field = payload.get("volume_fraction", [])
    shape_ok = bool(nx > 0 and ny > 0 and len(field) == nx and all(len(row) == ny for row in field))
    checks.append(
        Check(
            "D3_grid_shape",
            shape_ok,
            "%dx%d field vs %dx%d grid" % (len(field), len(field[0]) if field else 0, nx, ny),
            None,
            "field shape must equal the declared grid",
        )
    )
    checks.append(
        Check(
            "D4_grid_hash",
            payload.get("grid_hash") == grid_hash(grid),
            payload.get("grid_hash"),
            None,
            "declared grid_hash must equal sha256(grid descriptor)[:16]",
        )
    )

    if not shape_ok:  # remaining numeric checks are not meaningful
        return checks

    flat = [float(a) for row in field for a in row]
    a_min, a_max = min(flat), max(flat)
    checks.append(
        Check(
            "D5_fraction_bounds",
            a_min >= 0.0 and a_max <= 1.0,
            round(max(abs(a_min), abs(a_max - 1.0) if a_max > 1.0 else 0.0), 12),
            0.0,
            "volume fraction must satisfy 0 <= alpha <= 1 (measured min=%.6g max=%.6g)"
            % (a_min, a_max),
        )
    )

    thickness = float(payload.get("thickness_m", 0.0))
    cell_volume = dx * dy * thickness
    volume = sum(flat) * cell_volume
    declared = float(payload.get("total_ice_volume_m3", 0.0))
    err = _rel(volume, declared)
    checks.append(
        Check(
            "D6_ice_volume_closure",
            err <= TOL_CONSERVATION,
            err,
            TOL_CONSERVATION,
            "sum(alpha)*cell_volume must reproduce the declared ice volume",
        )
    )

    nominal = float(payload.get("nominal_coverage", 0.0))
    mean_alpha = payload.get("mean_volume_fraction")
    if mean_alpha is None:
        mean_alpha = sum(flat) / float(len(flat))
    cov_err = abs(float(mean_alpha) - nominal) / max(abs(nominal), 1e-12)
    checks.append(
        Check(
            "D7_coverage_consistency",
            cov_err <= TOL_INTERFACE,
            cov_err,
            TOL_INTERFACE,
            "area-weighted mean alpha must reproduce the nominal coverage C_i",
        )
    )

    density = float(payload.get("ice_density_kg_m3", 0.0))
    mass = float(payload.get("total_ice_mass_kg", 0.0))
    expected_mass = volume * density
    mass_err = abs(mass - expected_mass) / max(expected_mass, 1.0)
    checks.append(
        Check(
            "D8_mass_dimension",
            product_dimension(("kg/m^3", 1), ("m^3", 1)) == unit_dimension("kg")
            and mass_err <= TOL_CONSERVATION,
            mass_err,
            TOL_CONSERVATION,
            "mass must equal density x volume (kg = kg/m^3 * m^3)",
        )
    )

    dt = float(payload.get("dt_s", 0.0))
    checks.append(
        Check(
            "D9_time_axis",
            dt > 0.0 and float(payload.get("time_s", -1.0)) >= 0.0,
            dt,
            0.0,
            "DEM step must be positive and the clock non-negative",
        )
    )
    return checks


def _time_sync(prefix: str, payload: Mapping[str, Any], source: Mapping[str, Any]) -> Check:
    dt = float(payload.get("dt_s", 0.0))
    src_dt = float(payload.get("source_dt_s", source.get("dt_s", 0.0)))
    ratio = dt / src_dt if src_dt > 0 else float("inf")
    int_ratio = abs(ratio - round(ratio))
    t_err = abs(float(payload.get("time_s", 0.0)) - float(source.get("time_s", 0.0)))
    tol_t = 0.5 * src_dt
    ok = (
        src_dt > 0.0
        and dt > 0.0
        and (int_ratio <= TOL_TIME_SYNC or _rel(dt, src_dt) <= TOL_TIME_SYNC)
        and t_err <= tol_t
    )
    return Check(
        "%s_time_sync" % prefix,
        ok,
        t_err / tol_t if tol_t > 0 else float("inf"),
        1.0,
        "consumer step must be the source step or an integer subcycle of it "
        "(ratio=%.6f) and the clocks must agree within half a source step (|dt|=%.3e s)"
        % (ratio, t_err),
    )


def validate_hull_traction(
    payload: Mapping[str, Any], source: Mapping[str, Any]
) -> List[Check]:
    """Validate the RANS -> FSI traction handoff against its source field."""
    checks: List[Check] = [
        Check(
            "R1_schema",
            payload.get("schema") == SCHEMA_HULL_TRACTION,
            payload.get("schema"),
            None,
            "schema must be %s" % SCHEMA_HULL_TRACTION,
        ),
        _check_units("R2", payload.get("units", {}), _REQUIRED_UNITS_TRACTION),
    ]

    grid = payload.get("grid", {})
    dx = float(grid.get("dx_m", 0.0))
    panels = payload.get("panels", [])
    checks.append(
        Check(
            "R3_grid_identity",
            payload.get("grid_hash") == source.get("grid_hash"),
            payload.get("grid_hash"),
            None,
            "traction panels must live on the grid of the volume-fraction field",
        )
    )

    cells_x = [float(grid.get("x0_m", 0.0)) + (i + 0.5) * dx for i in range(int(grid.get("nx", 0)))]
    panel_x = [float(p["x_m"]) for p in panels]
    if len(panel_x) == len(cells_x) and dx > 0.0:
        align = max(abs(a - b) for a, b in zip(panel_x, cells_x))
        align_err = align / dx
    else:
        align_err = float("inf")
    checks.append(
        Check(
            "R4_panel_grid_alignment",
            align_err <= TOL_INTERFACE,
            align_err,
            TOL_INTERFACE,
            "panel centres must coincide with the field cell centres (error/dx)",
        )
    )

    checks.append(_time_sync("R5", payload, source))

    # discretised streamwise momentum balance per panel:
    #   dp/dx = -(mdot * du/dx + sum(source)) / A_water
    mdot = float(payload.get("mass_flux_kg_s", 0.0))
    worst = 0.0
    n_ok = 0
    for p in panels:
        area = float(p["water_area_m2"])
        if area <= 0.0:
            worst = float("inf")
            continue
        expected = -(mdot * float(p["du_dx_1_per_s"]) + float(p["source_N_per_m"])) / area
        err = _rel(float(p["dpdx_pa_per_m"]), expected)
        worst = max(worst, err)
        n_ok += 1
    checks.append(
        Check(
            "R6_momentum_balance_closure",
            worst <= TOL_INTERFACE,
            worst,
            TOL_INTERFACE,
            "per-panel dp/dx must satisfy the discretised Reynolds-averaged momentum "
            "balance (worst relative error over %d panels)" % n_ok,
        )
    )

    tau_min = min(float(p["tau_wall_pa"]) for p in panels) if panels else float("inf")
    checks.append(
        Check(
            "R7_shear_sign",
            tau_min >= -1e-12,
            tau_min,
            0.0,
            "wall shear magnitude must be non-negative",
        )
    )

    # panel force closure: sum(p * projected_area) + sum(tau * wetted_area) == totals
    f_p = sum(float(p["p_pa"]) * float(p["projected_area_m2"]) for p in panels)
    f_t = sum(float(p["tau_wall_pa"]) * float(p["wetted_area_m2"]) for p in panels)
    totals = payload.get("totals", {})
    r_p = float(totals.get("R_pressure_N", f_p))
    r_t = float(totals.get("R_friction_N", f_t))
    err_f = max(_rel(f_p, r_p), _rel(f_t, r_t))
    checks.append(
        Check(
            "R8_panel_force_closure",
            err_f <= TOL_INTERFACE,
            err_f,
            TOL_INTERFACE,
            "integrated panel loads must reproduce the declared force totals",
        )
    )

    # dynamic-head dimension: 0.5*rho*u^2 is a pressure
    checks.append(
        Check(
            "R9_dynamic_head_dimension",
            product_dimension(("kg/m^3", 1), ("m/s", 2)) == unit_dimension("Pa"),
            "dim(0.5 rho u^2) vs Pa",
            None,
            "dynamic head must be dimensionally a pressure so that p and tau are commensurate",
        )
    )
    return checks


def validate_struct_feedback(
    payload: Mapping[str, Any], source: Mapping[str, Any]
) -> List[Check]:
    """Validate the FSI -> DEM feedback handoff against its traction source."""
    checks: List[Check] = [
        Check(
            "S1_schema",
            payload.get("schema") == SCHEMA_STRUCT_FEEDBACK,
            payload.get("schema"),
            None,
            "schema must be %s" % SCHEMA_STRUCT_FEEDBACK,
        ),
        _check_units("S2", payload.get("units", {}), _REQUIRED_UNITS_FEEDBACK),
        Check(
            "S3_grid_identity",
            payload.get("grid_hash") == source.get("grid_hash"),
            payload.get("grid_hash"),
            None,
            "feedback must be returned on the grid the load was applied to",
        ),
        _time_sync("S4", payload, source),
    ]

    gap0 = float(payload.get("gap0_m", 0.0))
    w = [float(v) for v in payload.get("deflection_m", [])]
    c = [float(v) for v in payload.get("clearance_m", [])]
    checks.append(
        Check(
            "S5_length_consistency",
            len(w) == len(c) == len(payload.get("contact_pressure_pa", [])) and len(w) > 0,
            len(w),
            None,
            "deflection / clearance / pressure arrays must share one length",
        )
    )
    if len(w) == len(c) and w:
        gap_err = max(abs((gap0 - wi) - ci) for wi, ci in zip(w, c)) / max(abs(gap0), 1e-12)
        w_max = max(abs(v) for v in w)
        bounds_ok = w_max <= 0.5 * gap0
        checks.append(
            Check(
                "S6_gap_closure",
                gap_err <= TOL_CONSERVATION,
                gap_err,
                TOL_CONSERVATION,
                "clearance must equal gap0 - deflection on every panel",
            )
        )
        checks.append(
            Check(
                "S7_displacement_bounds",
                bounds_ok,
                w_max / (0.5 * gap0),
                1.0,
                "panel deflection must stay below half the nominal clearance (%.3e m)" % w_max,
            )
        )

    fp = payload.get("fixed_point", {})
    residual = float(fp.get("residual", float("inf")))
    checks.append(
        Check(
            "S8_fixed_point_residual",
            residual <= 1.0e-6 and bool(fp.get("converged", False)),
            residual,
            1.0e-6,
            "the two-way coupling must be certified converged before the feedback is consumed",
        )
    )
    return checks


def contract_manifest() -> Dict[str, Any]:
    """Static description of the three handoffs (used by the report / data-flow graph)."""
    return {
        "dem_to_rans": {
            "schema": SCHEMA_ICE_FIELD,
            "fields": sorted(_REQUIRED_UNITS_ICE),
            "checks": ["D1_schema", "D2_units_declared", "D3_grid_shape", "D4_grid_hash",
                       "D5_fraction_bounds", "D6_ice_volume_closure", "D7_coverage_consistency",
                       "D8_mass_dimension", "D9_time_axis"],
        },
        "rans_to_fsi": {
            "schema": SCHEMA_HULL_TRACTION,
            "fields": sorted(_REQUIRED_UNITS_TRACTION),
            "checks": ["R1_schema", "R2_units_declared", "R3_grid_identity",
                       "R4_panel_grid_alignment", "R5_time_sync", "R6_momentum_balance_closure",
                       "R7_shear_sign", "R8_panel_force_closure", "R9_dynamic_head_dimension"],
        },
        "fsi_to_dem": {
            "schema": SCHEMA_STRUCT_FEEDBACK,
            "fields": sorted(_REQUIRED_UNITS_FEEDBACK),
            "checks": ["S1_schema", "S2_units_declared", "S3_grid_identity", "S4_time_sync",
                       "S5_length_consistency", "S6_gap_closure", "S7_displacement_bounds",
                       "S8_fixed_point_residual"],
        },
    }