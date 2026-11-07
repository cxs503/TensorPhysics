"""Cross-repo interface contracts for P20 campaign integration.

Three couplings are contract-bearing in P20:

1. **FSI coupling** (TensorLBM surface forces → TensorFEM boundary conditions)
   LBM computes surface traction on the fluid-solid interface; FEM consumes
   it as Neumann boundary conditions.  The contract requires:
   - Matching mesh node IDs or a valid interpolation map
   - Force components in a shared coordinate frame
   - Time-synchronized exchange timestamps

2. **Hydrostatic coupling** (TensorCAD geometry → ship-platform hydrostatics)
   CAD delivers hull geometry (offsets/sections); ship platform computes
   displacement, Cb, L/B, etc.  The contract requires:
   - Principal dimensions (Lpp, B, T) with Cb*L*B*T ≈ ∇ closure
   - Geometry ID matching the file reference
   - Closed surface validation (discrete divergence theorem)

3. **Seakeeping coupling** (TensorBEM potential flow → ship-platform slamming)
   BEM delivers wave excitation forces / RAOs; ship platform uses them for
   slamming / motion analysis.  The contract requires:
   - Frequency array with consistent units (rad/s)
   - Force/moment components in agreed frame
   - Encounter frequency convention declared

All validators are pure functions over JSON payloads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

FSI_SCHEMA = "fsi_surface_force_handoff.v1"
HYDROSTATIC_SCHEMA = "cad_hydrostatic_handoff.v1"
SEAKEEPING_SCHEMA = "bem_seakeeping_handoff.v1"

REL_TOL = 1e-9
CB_CLOSURE_RTOL = 1e-12


@dataclass
class ContractCheck:
    check_id: str
    ok: bool
    detail: str
    value: float | None = None
    limit: float | None = None

    def as_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "ok": self.ok,
            "detail": self.detail,
            "value": self.value,
            "limit": self.limit,
            "status": "PASS" if self.ok else "FAIL",
        }


@dataclass
class ContractResult:
    contract_id: str
    title: str
    producer: str
    consumer: str
    checks: list[ContractCheck] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def failed_checks(self) -> list[ContractCheck]:
        return [c for c in self.checks if not c.ok]

    def as_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "title": self.title,
            "producer": self.producer,
            "consumer": self.consumer,
            "verdict": "PASS" if self.passed else "FAIL",
            "n_checks": len(self.checks),
            "n_failed": len(self.failed_checks),
            "checks": [c.as_dict() for c in self.checks],
            "evidence": self.evidence,
            "notes": self.notes,
        }


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    v = float(x)
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------
# 1. FSI coupling: LBM surface forces → FEM boundary conditions
# ---------------------------------------------------------------------------

FSI_REQUIRED_KEYS = {
    "schema", "frame", "timestamp_s", "interface_nodes", "force_components",
}

FSI_FRAME = {"x": "streamwise", "y": "transverse", "z": "vertical_up"}


def validate_fsi_handoff(payload: dict) -> ContractResult:
    """Validate TensorLBM → TensorFEM FSI surface force handoff.

    ``payload`` example::

        {
          "schema": "fsi_surface_force_handoff.v1",
          "frame": {"x": "streamwise", "y": "transverse", "z": "vertical_up"},
          "timestamp_s": 0.05,
          "interface_nodes": [
            {"node_id": 1, "x": 0.0, "y": 0.1, "z": -0.05,
             "Fx_N": 12.3, "Fy_N": 0.5, "Fz_N": -1.2},
            ...
          ],
          "force_components": ["Fx_N", "Fy_N", "Fz_N"],
          "total_force_N": [1230.0, 45.0, -89.0],
          "mesh_id": "hull_surface_v2"
        }
    """
    res = ContractResult(
        contract_id="fsi_handoff_v1",
        title="TensorLBM surface forces → TensorFEM Neumann BC",
        producer="TensorLBM (P20 FSI node)",
        consumer="TensorFEM boundary condition loader",
    )
    res.evidence["payload_schema"] = payload.get("schema") if isinstance(payload, dict) else None

    # C1 schema tag
    res.checks.append(ContractCheck(
        "F1_schema", isinstance(payload, dict) and payload.get("schema") == FSI_SCHEMA,
        f"schema must be {FSI_SCHEMA!r}, got {res.evidence['payload_schema']!r}"))
    if not res.passed:
        return res

    # C2 frame convention
    frame = payload.get("frame") or {}
    frame_bad = {k: (v, frame.get(k)) for k, v in FSI_FRAME.items() if frame.get(k) != v}
    res.checks.append(ContractCheck(
        "F2_frame", not frame_bad,
        "frame convention matches agreed contract" if not frame_bad
        else f"frame mismatch: {frame_bad}"))

    # C3 timestamp present and non-negative
    ts = _num(payload.get("timestamp_s"))
    res.checks.append(ContractCheck(
        "F3_timestamp", ts is not None and ts >= 0.0,
        f"timestamp_s = {ts}" if ts is not None and ts >= 0.0
        else f"timestamp_s missing or non-finite/negative: {payload.get('timestamp_s')!r}",
        value=ts))

    # C4 interface nodes present and non-empty
    nodes = payload.get("interface_nodes")
    if not isinstance(nodes, list) or len(nodes) == 0:
        res.checks.append(ContractCheck(
            "F4_nodes", False,
            f"need >=1 interface nodes, got {0 if not isinstance(nodes, list) else 0}"))
        return res

    # C5 each node has coordinates + force components, all finite
    bad_nodes = []
    total_fx = total_fy = total_fz = 0.0
    for i, nd in enumerate(nodes):
        if not isinstance(nd, dict):
            bad_nodes.append((i, "not an object"))
            continue
        for key in ("node_id", "x", "y", "z", "Fx_N", "Fy_N", "Fz_N"):
            if key not in nd:
                bad_nodes.append((i, f"missing {key}"))
                break
        else:
            for key in ("x", "y", "z", "Fx_N", "Fy_N", "Fz_N"):
                v = _num(nd.get(key))
                if v is None:
                    bad_nodes.append((i, f"non-finite {key}={nd.get(key)!r}"))
                    break
            else:
                total_fx += float(nd["Fx_N"])
                total_fy += float(nd["Fy_N"])
                total_fz += float(nd["Fz_N"])

    res.checks.append(ContractCheck(
        "F5_node_data", not bad_nodes,
        "all nodes have finite coordinates and force components" if not bad_nodes
        else f"malformed nodes: {bad_nodes[:5]}"))

    # C6 force_components declares the vector components
    fc = payload.get("force_components")
    res.checks.append(ContractCheck(
        "F6_force_components",
        isinstance(fc, list) and set(fc) == {"Fx_N", "Fy_N", "Fz_N"},
        f"force_components = {fc}" if isinstance(fc, list) and set(fc) == {"Fx_N", "Fy_N", "Fz_N"}
        else f"force_components must be ['Fx_N','Fy_N','Fz_N'], got {fc!r}"))

    # C7 total force consistency (sum of nodal forces vs declared total)
    declared = payload.get("total_force_N")
    if isinstance(declared, list) and len(declared) == 3:
        dx = abs(total_fx - declared[0])
        dy = abs(total_fy - declared[1])
        dz = abs(total_fz - declared[2])
        scale = max(abs(total_fx), abs(total_fy), abs(total_fz), 1.0)
        rel_err = max(dx, dy, dz) / scale
        res.checks.append(ContractCheck(
            "F7_total_force_consistency", rel_err < 1e-6,
            f"nodal sum vs declared total rel_err={rel_err:.2e}",
            value=rel_err, limit=1e-6))
        res.evidence["total_force_nodal"] = [total_fx, total_fy, total_fz]
        res.evidence["total_force_declared"] = declared
    else:
        res.checks.append(ContractCheck(
            "F7_total_force_consistency", False,
            f"total_force_N missing or malformed: {declared!r}"))

    res.evidence["n_interface_nodes"] = len(nodes) if isinstance(nodes, list) else 0
    return res


# ---------------------------------------------------------------------------
# 2. Hydrostatic coupling: CAD geometry → ship platform
# ---------------------------------------------------------------------------

HYDRO_REQUIRED_KEYS = {"schema", "geometry_id", "principal_dimensions", "displacement_m3"}


def validate_hydrostatic_handoff(payload: dict) -> ContractResult:
    """Validate TensorCAD → ship-platform hydrostatic geometry handoff.

    ``payload`` example::

        {
          "schema": "cad_hydrostatic_handoff.v1",
          "geometry_id": "kcs_hull_v3",
          "principal_dimensions": {
            "Lpp_m": 230.0, "B_m": 32.2, "T_m": 10.0,
            "Cb": 0.65, "Cm": 0.98, "Cp": 0.66
          },
          "displacement_m3": 51234.0,
          "wetted_area_m2": 8900.0,
          "sections": [{"x_m": 0.0, "area_m2": 0.0}, ...]
        }
    """
    res = ContractResult(
        contract_id="hydrostatic_handoff_v1",
        title="TensorCAD hull geometry → ship-platform hydrostatics",
        producer="TensorCAD (P20 geometry node)",
        consumer="ship-platform hydrostatic calculator",
    )
    res.evidence["payload_schema"] = payload.get("schema") if isinstance(payload, dict) else None

    # C1 schema tag
    res.checks.append(ContractCheck(
        "H1_schema", isinstance(payload, dict) and payload.get("schema") == HYDROSTATIC_SCHEMA,
        f"schema must be {HYDROSTATIC_SCHEMA!r}, got {res.evidence['payload_schema']!r}"))
    if not res.passed:
        return res

    # C2 geometry_id present and non-empty
    gid = payload.get("geometry_id")
    res.checks.append(ContractCheck(
        "H2_geometry_id", isinstance(gid, str) and gid.strip(),
        f"geometry_id = {gid!r}" if isinstance(gid, str) and gid.strip()
        else f"geometry_id missing or empty: {gid!r}"))

    # C3 principal dimensions present with required keys
    pd = payload.get("principal_dimensions") or {}
    required_pd = {"Lpp_m", "B_m", "T_m", "Cb"}
    missing_pd = required_pd - set(pd.keys())
    res.checks.append(ContractCheck(
        "H3_principal_dimensions", not missing_pd,
        "all principal dimensions present" if not missing_pd
        else f"missing principal dimensions: {missing_pd}"))

    # C4 dimensional sanity: Lpp > B > T > 0
    lpp = _num(pd.get("Lpp_m"))
    b = _num(pd.get("B_m"))
    t = _num(pd.get("T_m"))
    cb = _num(pd.get("Cb"))
    dim_ok = (lpp is not None and b is not None and t is not None
              and lpp > b > t > 0)
    res.checks.append(ContractCheck(
        "H4_dimensional_order", dim_ok,
        f"Lpp={lpp} > B={b} > T={t} > 0" if dim_ok
        else f"dimensional order violated: Lpp={lpp}, B={b}, T={t}"))

    # C5 Cb in physical range (0.3–0.9 for conventional ships)
    cb_ok = cb is not None and 0.3 <= cb <= 0.9
    res.checks.append(ContractCheck(
        "H5_Cb_range", cb_ok,
        f"Cb={cb} in [0.3, 0.9]" if cb_ok
        else f"Cb={cb} outside physical range [0.3, 0.9]"))

    # C6 Cb * Lpp * B * T ≈ displacement closure
    disp = _num(payload.get("displacement_m3"))
    if all(v is not None for v in (cb, lpp, b, t, disp)) and disp > 0:
        computed = cb * lpp * b * t
        rel_err = abs(computed - disp) / disp
        res.checks.append(ContractCheck(
            "H6_displacement_closure", rel_err <= CB_CLOSURE_RTOL,
            f"Cb*L*B*T={computed:.2f} vs ∇={disp:.2f}, rel_err={rel_err:.2e}",
            value=rel_err, limit=CB_CLOSURE_RTOL))
    else:
        res.checks.append(ContractCheck(
            "H6_displacement_closure", False,
            "cannot check closure: missing or non-positive values"))

    res.evidence["principal_dimensions"] = pd
    res.evidence["displacement_m3"] = disp
    return res


# ---------------------------------------------------------------------------
# 3. Seakeeping coupling: BEM potential flow → ship platform slamming
# ---------------------------------------------------------------------------

SEAKEEPING_REQUIRED_KEYS = {"schema", "frequency_rad_s", "force_amplitude", "frame"}
SEAKEEPING_FRAME = {"x": "surge", "y": "sway", "z": "heave",
                    "rx": "roll", "ry": "pitch", "rz": "yaw"}


def validate_seakeeping_handoff(payload: dict) -> ContractResult:
    """Validate TensorBEM → ship-platform seakeeping / slamming handoff.

    ``payload`` example::

        {
          "schema": "bem_seakeeping_handoff.v1",
          "frame": {"x": "surge", "y": "sway", "z": "heave",
                    "rx": "roll", "ry": "pitch", "rz": "yaw"},
          "frequency_rad_s": [0.5, 1.0, 1.5, 2.0],
          "wave_heading_deg": 180.0,
          "force_amplitude": [
            {"freq_rad_s": 0.5, "Fx_N": 1e5, "Fy_N": 0, "Fz_N": 2e5,
             "Mx_Nm": 0, "My_Nm": 5e6, "Mz_Nm": 0},
            ...
          ],
          "rao": [...],
          "encounter_convention": "positive_head_seas"
        }
    """
    res = ContractResult(
        contract_id="seakeeping_handoff_v1",
        title="TensorBEM potential flow → ship-platform slamming / motion",
        producer="TensorBEM (P20 seakeeping node)",
        consumer="ship-platform slamming analyzer",
    )
    res.evidence["payload_schema"] = payload.get("schema") if isinstance(payload, dict) else None

    # C1 schema tag
    res.checks.append(ContractCheck(
        "S1_schema", isinstance(payload, dict) and payload.get("schema") == SEAKEEPING_SCHEMA,
        f"schema must be {SEAKEEPING_SCHEMA!r}, got {res.evidence['payload_schema']!r}"))
    if not res.passed:
        return res

    # C2 frame convention
    frame = payload.get("frame") or {}
    frame_bad = {k: (v, frame.get(k)) for k, v in SEAKEEPING_FRAME.items()
                 if frame.get(k) != v}
    res.checks.append(ContractCheck(
        "S2_frame", not frame_bad,
        "frame convention matches agreed contract" if not frame_bad
        else f"frame mismatch: {frame_bad}"))

    # C3 frequency array present, sorted, positive
    freqs = payload.get("frequency_rad_s")
    if not isinstance(freqs, list) or len(freqs) < 2:
        res.checks.append(ContractCheck(
            "S3_frequency", False,
            f"need >=2 frequencies, got {0 if not isinstance(freqs, list) else len(freqs)}"))
        return res
    freq_vals = [_num(f) for f in freqs]
    freq_ok = all(v is not None and v > 0 for v in freq_vals)
    if freq_ok:
        sorted_ok = all(freq_vals[i] < freq_vals[i + 1] for i in range(len(freq_vals) - 1))
        freq_ok = freq_ok and sorted_ok
    res.checks.append(ContractCheck(
        "S3_frequency", freq_ok,
        f"{len(freqs)} frequencies, sorted and positive" if freq_ok
        else f"frequency array invalid: {freqs[:5]}"))

    # C4 force amplitude entries match frequency count
    fa = payload.get("force_amplitude")
    if not isinstance(fa, list) or len(fa) != len(freqs):
        res.checks.append(ContractCheck(
            "S4_force_amplitude_count", False,
            f"force_amplitude entries ({0 if not isinstance(fa, list) else len(fa)}) "
            f"!= frequency count ({len(freqs)})"))
    else:
        bad_fa = []
        for i, entry in enumerate(fa):
            if not isinstance(entry, dict):
                bad_fa.append((i, "not an object"))
                continue
            freq_match = abs(_num(entry.get("freq_rad_s")) - freq_vals[i]) < REL_TOL * max(abs(freq_vals[i]), 1.0)
            if not freq_match:
                bad_fa.append((i, f"freq mismatch: {entry.get('freq_rad_s')} vs {freq_vals[i]}"))
                continue
            for key in ("Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm"):
                v = _num(entry.get(key))
                if v is None:
                    bad_fa.append((i, f"non-finite {key}"))
                    break
        res.checks.append(ContractCheck(
            "S4_force_amplitude", not bad_fa,
            "all force amplitude entries valid" if not bad_fa
            else f"malformed entries: {bad_fa[:5]}"))

    # C5 encounter convention declared
    enc = payload.get("encounter_convention")
    res.checks.append(ContractCheck(
        "S5_encounter_convention", isinstance(enc, str) and enc.strip(),
        f"encounter_convention = {enc!r}" if isinstance(enc, str) and enc.strip()
        else f"encounter_convention missing: {enc!r}"))

    res.evidence["n_frequencies"] = len(freqs) if isinstance(freqs, list) else 0
    res.evidence["wave_heading_deg"] = payload.get("wave_heading_deg")
    return res
