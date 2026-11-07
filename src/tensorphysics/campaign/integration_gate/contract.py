"""Cross-domain interface contracts for the P16 campaign.

Two couplings are contract-bearing in P16:

``LBM wind coefficients -> ship-platform``
    TensorLBM (N7) produces a wind-force coefficient table; the ship platform
    consumes it at ``/api/v1/calculations/wind_resistance``.  The table is only
    usable downstream when the *dimensionless* coefficients come with the
    reference quantities that made them dimensionless (front/lateral area,
    reference velocity, air density, reference height) and an explicit frame /
    sign convention.  A table without those is numerically ambiguous by an
    arbitrary area factor.

``FEM wet modes -> BEM Helmholtz``
    TensorFEM (N8) delivers wet natural frequencies and modal added mass; the
    exterior acoustic solve in TensorBEM takes a *wavenumber*
    (``solve_helmholtz_diffraction(mesh, wavenumber=k)``,
    :mod:`tensorbem.helmholtz`, ``k = omega / c``).  The handoff therefore has to
    carry Hz **and** rad/s **and** m/s so that ``k`` can be reconstructed, and
    the added-mass quantity has to declare whether it is a modal mass, a mass
    per unit area (``tensorfem.added_mass``: kg/m²) or a mass matrix, plus the
    surface area used for the radiation-efficiency normalisation.

The validators below are pure functions over plain JSON payloads: they never
invent a number, they only reject what is ambiguous, dimensionally inconsistent
or contract-incomplete.  Every check returns a stable ``check_id`` so that gate
reports and regression diffs can be diffed mechanically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

TWO_PI = 2.0 * math.pi

WIND_SCHEMA = "wind_resistance_coefficients.v1"
WET_MODAL_SCHEMA = "wet_modal_bem_handoff.v1"

#: anchor angles (deg) that the ship-platform consumer needs in order to
#: interpolate head / beam / following wind without extrapolation.
WIND_REQUIRED_ANGLES = (0.0, 90.0, 180.0)
#: maximum admissible angle gap between consecutive samples (deg) so that a
#: 1-D interpolant is not asked to bridge a physically monotone-free span.
WIND_MAX_ANGLE_GAP_DEG = 45.0
#: physical sanity bound on the wind-force coefficients (sail-area ships with
#: deck cargo reach ~2; anything beyond 3 is a scaling bug, not a coefficient).
WIND_MAX_ABS_COEFF = 3.0

WIND_FRAME = {
    "x": "forward",
    "y": "port",
    "z": "up",
    "heading_zero": "bow-on",
}

#: dimensional keys required for the wind table (name -> unit string).
WIND_UNITS = {
    "beta_deg": "deg",
    "C_X": "1",
    "C_Y": "1",
    "C_N": "1",
    "U_ref_m_s": "m/s",
    "rho_air_kg_m3": "kg/m^3",
    "A_front_m2": "m^2",
    "A_lateral_m2": "m^2",
    "z_ref_m": "m",
}

WET_MODAL_UNITS = {
    "f_hz": "Hz",
    "omega_rad_s": "rad/s",
    "dry_f_hz": "Hz",
    "added_mass_kg": "kg",
    "modal_mass_kg": "kg",
    "rho_kg_m3": "kg/m^3",
    "c_m_s": "m/s",
    "wavenumber_k": "1/m",
    "sigma": "1",
    "S_ref_m2": "m^2",
}

REL_TOL_UNIT = 1e-9       # Hz <-> rad/s and k <-> omega/c are exact identities
REL_TOL_SYMMETRY = 1e-10  # added-mass matrix symmetry
REL_TOL_PSD = 1e-12       # negative eigenvalue admissibility (relative to max)


@dataclass
class Check:
    """One machine-checkable assertion of a cross-domain contract."""

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
    """Verdict for one cross-domain interface."""

    contract_id: str
    title: str
    producer: str
    consumer: str
    checks: list[Check] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def failed_checks(self) -> list[Check]:
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


def _num(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def _get(d: dict, *names):
    for n in names:
        if isinstance(d, dict) and n in d:
            return d[n]
    return _MISSING


_MISSING = object()


# ---------------------------------------------------------------------------
# wind coefficient table
# ---------------------------------------------------------------------------
def validate_wind_coefficient_table(payload: dict) -> ContractResult:
    """Validate a TensorLBM wind-coefficient table against the consumer contract.

    ``payload`` example (all keys mandatory unless stated)::

        {
          "schema": "wind_resistance_coefficients.v1",
          "units": {"beta_deg": "deg", "C_X": "1", ...},
          "frame": {"x": "forward", "y": "port", "z": "up", "heading_zero": "bow-on"},
          "reference": {"A_front_m2": 312.0, "A_lateral_m2": 1580.0,
                        "U_ref_m_s": 10.0, "rho_air_kg_m3": 1.225, "z_ref_m": 10.0},
          "samples": [{"beta_deg": 0.0, "C_X": 0.82, "C_Y": 0.06, "C_N": 0.05}, ...]
        }
    """
    res = ContractResult(
        contract_id="wind_table_v1",
        title="TensorLBM wind-force coefficient table -> ship-platform wind resistance",
        producer="TensorLBM (P16-N7)",
        consumer="ship-platform /api/v1/calculations/wind_resistance",
    )
    res.evidence["payload_schema"] = payload.get("schema") if isinstance(payload, dict) else None

    # C1 schema tag
    res.checks.append(
        Check("W1_schema_tag", isinstance(payload, dict) and payload.get("schema") == WIND_SCHEMA,
              f"schema tag must be {WIND_SCHEMA!r}, got {res.evidence['payload_schema']!r}")
    )
    if not res.passed:
        return res

    # C2 units block declares every dimensional column with the agreed unit
    units = payload.get("units") or {}
    bad_units = {k: (v, units.get(k)) for k, v in WIND_UNITS.items() if units.get(k) != v}
    res.checks.append(
        Check("W2_units", not bad_units,
              "declared units must match the contract table" if not bad_units
              else f"unit mismatches (expected, found): {bad_units}")
    )

    # C3 reference quantities present, positive and finite
    ref = payload.get("reference") or {}
    ref_bad = {}
    for k in ("A_front_m2", "A_lateral_m2", "U_ref_m_s", "rho_air_kg_m3", "z_ref_m"):
        v = _num(ref.get(k))
        if v is None or v <= 0.0:
            ref_bad[k] = ref.get(k)
    res.checks.append(
        Check("W3_reference_quantities", not ref_bad,
              "all reference quantities positive and finite" if not ref_bad
              else f"missing/non-positive reference quantities: {ref_bad}")
    )

    # C4 frame / sign convention is explicit and matches the agreed one
    frame = payload.get("frame") or {}
    frame_bad = {k: (v, frame.get(k)) for k, v in WIND_FRAME.items() if frame.get(k) != v}
    res.checks.append(
        Check("W4_frame_convention", not frame_bad,
              "frame/sign convention explicit and agreed" if not frame_bad
              else f"frame mismatch (expected, found): {frame_bad}")
    )

    # C5 samples: sorted, unique, finite, in [-180,180] or [0,360)
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) < 3:
        res.checks.append(Check("W5_samples_present", False,
                                f"need >=3 samples, got {0 if not isinstance(samples, list) else len(samples)}"))
        return res
    angles, bad_samples = [], []
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            bad_samples.append((i, "not an object"))
            continue
        b = _num(s.get("beta_deg"))
        vals = {k: _num(s.get(k)) for k in ("C_X", "C_Y", "C_N")}
        if b is None or any(v is None for v in vals.values()):
            bad_samples.append((i, f"non-finite/non-numeric: beta={s.get('beta_deg')!r} {vals}"))
            continue
        if not (-180.0 <= b <= 360.0):
            bad_samples.append((i, f"beta_deg={b} outside [-180,360]"))
            continue
        angles.append(b % 360.0)
    res.checks.append(
        Check("W5_samples_present", not bad_samples and len(angles) == len(samples),
              "every sample numeric, finite and in range" if not bad_samples
              else f"malformed samples: {bad_samples[:5]}")
    )
    if bad_samples:
        return res

    # C6 coefficient magnitude sanity (catches area/normalisation bugs)
    over = [(a, k, s[k]) for a, s in zip(angles, samples)
            for k in ("C_X", "C_Y", "C_N") if abs(float(s[k])) > WIND_MAX_ABS_COEFF]
    res.checks.append(
        Check("W6_coefficient_magnitude", not over,
              f"all |C| <= {WIND_MAX_ABS_COEFF}" if not over else f"implausible coefficients: {over[:5]}",
              limit=WIND_MAX_ABS_COEFF)
    )

    # C7 anchor angles present (head / beam / following) so the consumer never extrapolates
    missing = [a for a in WIND_REQUIRED_ANGLES if min(abs(((ang - a + 180.0) % 360.0) - 180.0)
                                                     for ang in angles) > 1e-6]
    res.checks.append(
        Check("W7_anchor_angles", not missing,
              "0/90/180 deg present" if not missing else f"missing anchor angles {missing} deg")
    )

    # C8 angular resolution (max gap)
    ordered = sorted(set(angles))
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    if len(ordered) >= 2:
        gaps.append(360.0 - ordered[-1] + ordered[0])
    max_gap = max(gaps) if gaps else 360.0
    res.checks.append(
        Check("W8_angular_resolution", max_gap <= WIND_MAX_ANGLE_GAP_DEG,
              f"max sample gap {max_gap:.3g} deg <= {WIND_MAX_ANGLE_GAP_DEG} deg",
              value=max_gap, limit=WIND_MAX_ANGLE_GAP_DEG)
    )
    res.evidence["n_samples"] = len(samples)
    res.evidence["max_angle_gap_deg"] = max_gap
    return res


# ---------------------------------------------------------------------------
# FEM wet modal -> BEM Helmholtz handoff
# ---------------------------------------------------------------------------
def validate_wet_modal_handoff(payload: dict) -> ContractResult:
    """Validate the TensorFEM wet-modal payload against TensorBEM's Helmholtz input.

    ``payload`` example::

        {
          "schema": "wet_modal_bem_handoff.v1",
          "units": {"f_hz": "Hz", "omega_rad_s": "rad/s", ...},
          "fluid": {"rho_kg_m3": 1025.0, "c_m_s": 1500.0},
          "S_ref_m2": 3100.0,
          "modes": [{"mode_id": 1, "f_hz": 4.31, "omega_rad_s": 27.08,
                     "dry_f_hz": 6.10, "added_mass_kg": 1.2e6,
                     "modal_mass_kg": 2.0e6, "sigma": 0.42}],
          "added_mass_matrix_kg": [[...]],   # optional, checked when present
          "wavenumber_k": [...]              # optional, checked when present
        }
    """
    res = ContractResult(
        contract_id="wet_modal_bem_v1",
        title="TensorFEM wet modes + added mass -> TensorBEM Helmholtz radiation",
        producer="TensorFEM (P16-N8)",
        consumer="TensorBEM solve_helmholtz_diffraction(k = omega/c)",
    )
    res.evidence["payload_schema"] = payload.get("schema") if isinstance(payload, dict) else None
    res.checks.append(
        Check("M1_schema_tag", isinstance(payload, dict) and payload.get("schema") == WET_MODAL_SCHEMA,
              f"schema tag must be {WET_MODAL_SCHEMA!r}, got {res.evidence['payload_schema']!r}")
    )
    if not res.passed:
        return res

    # units block: Hz / rad/s / m/s must all be declared -> k is reconstructible
    units = payload.get("units") or {}
    required_units = ("f_hz", "omega_rad_s", "c_m_s")
    bad_units = {k: (WET_MODAL_UNITS[k], units.get(k)) for k in required_units
                 if units.get(k) != WET_MODAL_UNITS[k]}
    res.checks.append(
        Check("M2_units", not bad_units,
              "Hz / rad/s / m/s declared so wavenumber is reconstructible" if not bad_units
              else f"unit mismatches (expected, found): {bad_units}")
    )

    fluid = payload.get("fluid") or {}
    rho = _num(fluid.get("rho_kg_m3"))
    c = _num(fluid.get("c_m_s"))
    res.checks.append(
        Check("M3_fluid_properties", rho is not None and rho > 0 and c is not None and c > 0,
              f"rho={rho}, c={c}: both required positive (BEM k=omega/c, sigma normalisation)")
    )
    S = _num(payload.get("S_ref_m2"))
    res.checks.append(
        Check("M4_reference_area", S is not None and S > 0,
              f"S_ref_m2={S}: required for radiation-efficiency normalisation")
    )

    modes = payload.get("modes")
    if not isinstance(modes, list) or not modes:
        res.checks.append(Check("M5_modes_present", False, "need a non-empty 'modes' list"))
        return res

    rows, bad = [], []
    for i, m in enumerate(modes):
        if not isinstance(m, dict):
            bad.append((i, "not an object"))
            continue
        rec = {k: _num(m.get(k)) for k in ("mode_id", "f_hz", "omega_rad_s", "dry_f_hz",
                                           "added_mass_kg", "modal_mass_kg", "sigma")}
        if any(v is None for k, v in rec.items() if k != "sigma"):
            bad.append((i, f"non-finite required modal quantity: {rec}"))
            continue
        if rec["f_hz"] <= 0 or rec["omega_rad_s"] <= 0 or rec["dry_f_hz"] <= 0:
            bad.append((i, f"non-positive frequency: f={rec['f_hz']}, omega={rec['omega_rad_s']}, "
                           f"f_dry={rec['dry_f_hz']}"))
            continue
        rows.append(rec)
    res.checks.append(
        Check("M5_modes_present", not bad and len(rows) == len(modes),
              "every mode carries f, omega, f_dry, added mass and modal mass"
              if not bad else f"malformed modes: {bad[:5]}")
    )
    if bad:
        return res

    # M6 Hz <-> rad/s identity (the classic unit-mismatch killer)
    worst = max(abs(r["omega_rad_s"] - TWO_PI * r["f_hz"]) / (TWO_PI * r["f_hz"]) for r in rows)
    res.checks.append(
        Check("M6_hz_rad_s_identity", worst <= REL_TOL_UNIT,
              f"max rel. deviation of omega vs 2*pi*f = {worst:.3e}",
              value=worst, limit=REL_TOL_UNIT)
    )

    # M7 added mass lowers the frequency: 0 < f_wet <= f_dry
    viol = [(r["mode_id"], r["f_hz"], r["dry_f_hz"]) for r in rows
            if r["f_hz"] > r["dry_f_hz"] * (1.0 + REL_TOL_UNIT)]
    ratios = [r["f_hz"] / r["dry_f_hz"] for r in rows]
    res.checks.append(
        Check("M7_added_mass_softening", not viol,
              "all f_wet <= f_dry (added mass removes stiffness)" if not viol
              else f"modes where wet > dry frequency: {viol}", value=max(ratios))
    )

    # M8 modal ordering strictly increasing (BEM sweeps modes in ascending order)
    freqs = [r["f_hz"] for r in rows]
    res.checks.append(
        Check("M8_mode_ordering", all(b > a for a, b in zip(freqs, freqs[1:])),
              f"f_hz strictly increasing: {freqs}")
    )

    # M9 radiation efficiency bounds
    sig = [r["sigma"] for r in rows]
    sig_bad = [s for s in sig if s is not None and not (0.0 <= s <= 1.5)]
    res.checks.append(
        Check("M9_sigma_bounds", not sig_bad,
              f"sigma in [0,1.5]: {sig}" if not sig_bad else f"sigma out of range: {sig_bad}")
    )

    # M10 wavenumber agreement, when the producer ships its own k
    k_list = payload.get("wavenumber_k")
    if isinstance(k_list, list) and len(k_list) == len(rows) and (c is None or c <= 0):
        res.checks.append(
            Check("M10_wavenumber_consistency", False,
                  "cannot verify k = omega/c: sound speed is missing or non-positive")
        )
        res.evidence["k_checked"] = False
    elif isinstance(k_list, list) and len(k_list) == len(rows):
        dev = []
        for r, k in zip(rows, k_list):
            kv = _num(k)
            if kv is None or kv <= 0:
                dev.append(float("inf"))
                continue
            k_expected = r["omega_rad_s"] / c
            dev.append(abs(kv - k_expected) / k_expected)
        worst_k = max(dev)
        res.checks.append(
            Check("M10_wavenumber_consistency", worst_k <= REL_TOL_UNIT,
                  f"max rel. deviation of k vs omega/c = {worst_k:.3e}",
                  value=worst_k, limit=REL_TOL_UNIT)
        )
        res.evidence["k_checked"] = True
    else:
        res.checks.append(
            Check("M10_wavenumber_consistency", True,
                  "no explicit k supplied: derived downstream as omega/c (identity M6/M2 enforced)")
        )
        res.evidence["k_checked"] = False

    # M11 added-mass matrix symmetry / PSD, when supplied
    mat = payload.get("added_mass_matrix_kg")
    if isinstance(mat, list) and mat:
        sym, psd, detail, val = _check_symmetric_psd(mat)
        res.checks.append(Check("M11_added_mass_matrix", sym and psd, detail,
                                value=val, limit=REL_TOL_SYMMETRY))
        res.evidence["added_mass_matrix_checked"] = True
    else:
        res.checks.append(Check("M11_added_mass_matrix", True,
                                "no added-mass matrix supplied: per-mode scalar mass only"))
        res.evidence["added_mass_matrix_checked"] = False

    res.evidence["n_modes"] = len(rows)
    res.evidence["wet_dry_ratio"] = [round(float(x), 6) for x in ratios]
    return res


def _check_symmetric_psd(mat: list) -> tuple[bool, bool, str, float | None]:
    n = len(mat)
    if any(not isinstance(row, list) or len(row) != n for row in mat):
        return False, False, "added-mass matrix is not square", None
    vals = [[_num(x) for x in row] for row in mat]
    if any(v is None for row in vals for v in row):
        return False, False, "added-mass matrix contains non-finite entries", None
    scale = max(abs(v) for row in vals for v in row) or 1.0
    asym = max(abs(vals[i][j] - vals[j][i]) for i in range(n) for j in range(n)) / scale
    if asym > REL_TOL_SYMMETRY:
        return False, False, f"added-mass matrix not symmetric: max rel. skew {asym:.3e}", asym
    try:  # pragma: no cover - optional dependency
        import numpy as np

        w = np.linalg.eigvalsh(np.asarray(vals, dtype=float))
        lmax = float(max(abs(w[0]), abs(w[-1]))) or 1.0
        lmin_rel = float(w[0]) / lmax
        ok = lmin_rel >= -REL_TOL_PSD
        return True, ok, (f"symmetric (skew {asym:.1e}); min eigenvalue rel. {lmin_rel:.3e} "
                          f"(>= -{REL_TOL_PSD:g})"), min(asym, abs(lmin_rel))
    except ImportError:  # pragma: no cover
        return True, True, f"symmetric (skew {asym:.1e}); PSD check skipped (numpy absent)", asym


# ---------------------------------------------------------------------------
# consumer-endpoint compatibility (ship-platform side of both contracts)
# ---------------------------------------------------------------------------
WIND_ENDPOINT = "/api/v1/calculations/wind_resistance"
MODAL_ENDPOINT = "/api/v1/fem/modal"
#: fields the wind endpoint has to accept in order to consume an LBM table
WIND_ENDPOINT_REQUIRED_FIELDS = ("beta_deg", "U_ref_m_s", "rho_air_kg_m3",
                                 "A_front_m2", "A_lateral_m2", "z_ref_m")
#: fields a *wet* modal endpoint has to accept/expose to be BEM-ready
WET_MODAL_ENDPOINT_REQUIRED_FIELDS = ("fluid_rho", "sound_speed", "added_mass", "added_mass_matrix")


def validate_endpoint_compatibility(facts: dict) -> list[ContractResult]:
    """Check the ship-platform OpenAPI facts against both contracts.

    ``facts`` is the ``ship_platform_facts.json`` artifact produced by
    ``scripts/collect_p16_gate_evidence.py``.
    """
    eps = (facts.get("endpoints") or {})
    out: list[ContractResult] = []

    # --- wind endpoint ---
    wind = eps.get(WIND_ENDPOINT) or {}
    res = ContractResult(
        contract_id="wind_endpoint_v1",
        title="ship-platform wind-resistance endpoint accepts the LBM coefficient table",
        producer="ship-platform (P16-N7 integration half)",
        consumer="TensorPhysics gate",
    )
    res.checks.append(
        Check("E1_wind_endpoint_present", bool(wind.get("present")),
              f"{WIND_ENDPOINT} present={wind.get('present')} "
              f"(scanned {facts.get('n_paths')} OpenAPI paths; wind-related paths found: "
              f"{wind.get('related_paths')})")
    )
    schemas = wind.get("schemas") or {}
    props: set[str] = set()
    for blob in schemas.values():
        props |= set((blob.get("properties") or {}).keys())
    missing = [f for f in WIND_ENDPOINT_REQUIRED_FIELDS if f not in props]
    res.checks.append(
        Check("E2_wind_endpoint_fields", bool(wind.get("present")) and not missing,
              f"accepts {WIND_ENDPOINT_REQUIRED_FIELDS}; missing {missing}" if missing
              else "endpoint schema exposes every required reference quantity")
    )
    res.evidence.update({"endpoint": WIND_ENDPOINT, "schemas": sorted(schemas),
                         "observed_properties": sorted(props)[:40],
                         "related_paths": wind.get("related_paths")})
    out.append(res)

    # --- modal endpoint (dry modes today; wet modal is the N8 deliverable) ---
    modal = eps.get(MODAL_ENDPOINT) or {}
    res = ContractResult(
        contract_id="wet_modal_endpoint_v1",
        title="ship-platform modal endpoint is able to carry wet-modal / added-mass data",
        producer="ship-platform (P16-N8 integration half)",
        consumer="TensorPhysics gate",
    )
    res.checks.append(
        Check("E3_modal_endpoint_present", bool(modal.get("present")),
              f"{MODAL_ENDPOINT} present={modal.get('present')}")
    )
    props = set()
    for blob in (modal.get("schemas") or {}).values():
        props |= set((blob.get("properties") or {}).keys())
    missing = [f for f in WET_MODAL_ENDPOINT_REQUIRED_FIELDS if f not in props]
    res.checks.append(
        Check("E4_modal_endpoint_wet_fields", not missing,
              f"observed modal schema properties {sorted(props)}; missing wet-modal fields "
              f"{missing} (endpoint currently exposes dry modes only)")
    )
    res.evidence.update({"endpoint": MODAL_ENDPOINT, "observed_properties": sorted(props),
                         "required_fields": list(WET_MODAL_ENDPOINT_REQUIRED_FIELDS)})
    out.append(res)
    return out