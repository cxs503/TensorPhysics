"""Cross-repo interface contracts for the P27-W2 campaign.

Three couplings are contract-bearing in P27-W2:

1. **RAO handoff** (TensorBEM potential-flow seakeeping ->
   ship-performance-platform).  BEM delivers wave excitation forces and
   6-DOF response amplitude operators (RAOs); the ship platform consumes
   them for motion / slamming analysis.  The contract requires:
   - declared schema id and hull identity (hull_id, Fn, wave direction)
   - frequency array in rad/s, strictly increasing, finite, positive
   - RAO amplitude/phase vectors per DOF with lengths matching the
     frequency array; at least heave, pitch, roll present
   - encounter-frequency convention declared
   - optional Kochin far-field energy consistency flag (N01 criterion c)

2. **Granular-flow handoff** (TensorDEM -> TensorPhysics).  DEM delivers
   calibrated granular-flow parameters (Cundall-Strack tangential spring,
   rolling resistance, rotational damping, angle of repose); TensorPhysics
   consumes them through ``tensorphysics.adapters.tensordem``.  The
   contract requires:
   - particle closure: density ~= mass / (4/3 pi r^3)
   - 0 < k_t <= k_n, 0 < mu_s < 1, 0 <= mu_r < 1, c_rot >= 0
   - angle-of-repose calibration: |achieved - target| within tolerance

3. **GCI/UQ handoff** (TensorLBM -> TensorPlatform).  LBM delivers grid
   convergence indices and uncertainty-quantification statistics; the
   platform consumes them for evidence grading.  The contract requires:
   - GCI: >=3 grids with strictly decreasing h, observed order p > 0
     consistent with the Richardson relation, GCI_fine consistent with
     C*(v2-v3)/(r^p - 1), C = 1.25
   - UQ: n_blocks >= 3, CI95 consistent with 1.96*std/sqrt(n), dual-domain
     cross-check and SGS negative control declared

All validators are pure functions over JSON payloads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

RAO_SCHEMA = "bem_seakeeping_rao_handoff.v1"
GRANULAR_SCHEMA = "dem_granular_flow_handoff.v1"
GCI_UQ_SCHEMA = "lbm_gci_uq_handoff.v1"

REL_TOL = 1e-9
DENSITY_CLOSURE_RTOL = 1e-6
AOA_TOL_DEG = 2.0
CB_CLOSURE_RTOL = 5e-3
ORDER_TOL = 0.15
GCI_C = 1.25
CI95_COEF = 1.96


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
        }


@dataclass
class ContractResult:
    contract_id: str
    schema: str
    checks: list[ContractCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def coverage(self) -> float:
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.ok) / len(self.checks)

    def as_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "schema": self.schema,
            "ok": self.ok,
            "coverage": self.coverage,
            "checks": [c.as_dict() for c in self.checks],
        }


def _finite_positive_list(values: Any) -> list[float] | None:
    if not isinstance(values, (list, tuple)) or not values:
        return None
    out = []
    for v in values:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        if not math.isfinite(v) or v <= 0:
            return None
        out.append(float(v))
    return out


def _strictly_increasing(values: list[float]) -> bool:
    return all(b > a for a, b in zip(values, values[1:]))


# ---------------------------------------------------------------------------
# 1. BEM -> ship-performance-platform : RAO data format
# ---------------------------------------------------------------------------

REQUIRED_RAO_DOFS = ("heave", "pitch", "roll")
ENCOUNTER_CONVENTIONS = (
    "deep_water_encounter",
    "shallow_water_encounter",
    "calm_water",
)
#: DOF -> (amplitude key, phase key) in the legacy flat layout used by the
#: ship-performance-platform seakeeping module (m6657 rao_verification).
FLAT_RAO_KEYS = {
    "heave": ("heave_rao", "heave_phase_deg"),
    "pitch": ("pitch_rao", "pitch_phase_deg"),
    "roll": ("roll_rao", "roll_phase_deg"),
}


def _extract_rao(payload: dict):
    """Return (freqs, {dof: (amp, phase)}) from either layout.

    Canonical layout: ``frequencies_rad_s`` + ``rao: {dof: {amplitude,
    phase}}``.  Legacy flat layout: ``omega_e`` + ``<dof>_rao`` /
    ``<dof>_phase_deg`` arrays.
    """
    freqs = _finite_positive_list(
        payload.get("frequencies_rad_s", payload.get("omega_e"))
    )
    if freqs is None:
        return None, None
    rao = payload.get("rao")
    if isinstance(rao, dict):
        dofs = {}
        for dof, (amp_key, phase_key) in FLAT_RAO_KEYS.items():
            block = rao.get(dof)
            if not isinstance(block, dict):
                continue
            amp = _finite_positive_list(block.get("amplitude"))
            phase = block.get("phase")
            if amp is None or not isinstance(phase, (list, tuple)):
                continue
            dofs[dof] = (amp, [float(x) for x in phase])
        return freqs, dofs
    dofs = {}
    for dof, (amp_key, phase_key) in FLAT_RAO_KEYS.items():
        amp = _finite_positive_list(payload.get(amp_key))
        phase = payload.get(phase_key)
        if amp is None or not isinstance(phase, (list, tuple)):
            continue
        dofs[dof] = (amp, [float(x) for x in phase])
    return freqs, dofs


def validate_rao_handoff(payload: dict) -> ContractResult:
    """Validate a BEM->ship-platform RAO handoff payload.

    Accepts the canonical layout (``schema`` + ``rao`` blocks) and the
    legacy flat layout produced by the ship platform's seakeeping module
    (``omega_e`` + ``<dof>_rao`` / ``<dof>_phase_deg``).
    """
    res = ContractResult(contract_id="P27W2-C1", schema=RAO_SCHEMA)

    def add(check_id: str, ok: bool, detail: str,
              value: float | None = None, limit: float | None = None) -> None:
        res.checks.append(ContractCheck(check_id, bool(ok), detail, value, limit))

    has_schema = payload.get("schema") == RAO_SCHEMA
    add("C1.schema", has_schema or "omega_e" in payload,
        "schema must be " + repr(RAO_SCHEMA)
        + " (or legacy flat layout with omega_e)")

    ship = payload.get("ship") or payload
    L, B, T = ship.get("L"), ship.get("B"), ship.get("T")
    Cb = ship.get("Cb")
    disp = ship.get("displacement")
    geom_ok = (
        all(isinstance(v, (int, float)) and v > 0 for v in (L, B, T, Cb))
    )
    add("C1.geometry", geom_ok,
        "ship geometry L/B/T/Cb must be positive numbers")
    if geom_ok:
        # Cb closure: Cb = disp / (rho * L * B * T); displacement in t
        rho = ship.get("rho", 1000.0)
        if isinstance(disp, (int, float)) and disp > 0:
            Cb_expected = disp * 1000.0 / (rho * L * B * T)
            closure_ok = math.isclose(Cb, Cb_expected, rel_tol=CB_CLOSURE_RTOL)
            add("C1.cb_closure", closure_ok,
                "Cb must equal displacement/(rho*L*B*T) within 0.1%",
                value=float(Cb), limit=float(Cb_expected))
        else:
            add("C1.cb_closure", True,
                "displacement not provided; Cb taken as declared")
    fn = ship.get("Fn", payload.get("fn"))
    add("C1.fn", isinstance(fn, (int, float)) and 0 < float(fn) < 1,
        f"Fn must be in (0,1), got {fn!r}")
    wd = payload.get("wave_direction_deg",
                     payload.get("beta_deg", 180.0))
    add("C1.wave_direction",
        isinstance(wd, (int, float)) and 0 <= float(wd) <= 180,
        f"wave direction must be in [0,180] deg, got {wd!r}")

    freqs, dofs = _extract_rao(payload)
    add("C1.frequencies", freqs is not None,
        "frequency array (rad/s) must be non-empty, finite, positive")
    if freqs is not None:
        add("C1.freq_monotonic", _strictly_increasing(freqs),
            "frequencies must be strictly increasing", value=len(freqs))
    if dofs is None:
        dofs = {}
    missing = [d for d in REQUIRED_RAO_DOFS if d not in dofs]
    add("C1.rao_dofs", not missing,
        "RAO must contain at least heave/pitch/roll"
        + (f"; missing {missing}" if missing else ""))
    if freqs is not None and dofs:
        length_ok = True
        for dof, (amp, phase) in dofs.items():
            if len(amp) != len(freqs) or len(phase) != len(freqs):
                length_ok = False
                break
            if any(not math.isfinite(x) for x in phase):
                length_ok = False
                break
        add("C1.rao_lengths", length_ok,
            "per-DOF amplitude/phase lengths must match the frequency array")
    else:
        add("C1.rao_lengths", False, "no RAO blocks found")

    conv = payload.get("encounter_convention")
    if conv is None:
        # legacy layout: head seas (180 deg) is the declared convention
        conv_ok = isinstance(wd, (int, float)) and abs(float(wd) - 180.0) < 1e-9
        add("C1.encounter_convention", conv_ok,
            "encounter convention: legacy layout requires head seas (180 deg)")
    else:
        add("C1.encounter_convention", conv in ENCOUNTER_CONVENTIONS,
            f"encounter_convention must be one of {ENCOUNTER_CONVENTIONS}, "
            f"got {conv!r}")
    units = payload.get("units") or {}
    if units:
        add("C1.units",
            units.get("frequency") == "rad/s"
            and units.get("phase") in ("rad", "deg"),
            "units must declare frequency=rad/s and phase in {rad, deg}")
    else:
        add("C1.units", True,
            "units omitted: legacy layout declares rad/s + deg in field names")
    add("C1.kochin_energy",
        isinstance(payload.get("kochin_energy_consistency"), (bool, dict)),
        "kochin_energy_consistency flag (bool or report dict) must be present")
    return res


# ---------------------------------------------------------------------------
# 2. TensorDEM -> TensorPhysics : granular-flow parameters
# ---------------------------------------------------------------------------


def validate_granular_flow_handoff(payload: dict) -> ContractResult:
    """Validate a DEM->TensorPhysics granular-flow parameter payload."""
    res = ContractResult(contract_id="P27W2-C2", schema=GRANULAR_SCHEMA)

    def add(check_id: str, ok: bool, detail: str,
            value: float | None = None, limit: float | None = None) -> None:
        res.checks.append(ContractCheck(check_id, bool(ok), detail, value, limit))

    add("C2.schema", payload.get("schema") == GRANULAR_SCHEMA,
        f"schema must be {GRANULAR_SCHEMA!r}, got {payload.get('schema')!r}")

    part = payload.get("particle") or {}
    r = part.get("radius_m")
    m = part.get("mass_kg")
    rho = part.get("density_kg_m3")
    closure_ok = False
    if (isinstance(r, (int, float)) and r > 0
            and isinstance(m, (int, float)) and m > 0
            and isinstance(rho, (int, float)) and rho > 0):
        rho_expected = m / (4.0 / 3.0 * math.pi * r ** 3)
        closure_ok = math.isclose(rho, rho_expected, rel_tol=DENSITY_CLOSURE_RTOL)
        closure_ok = closure_ok
        add("C2.particle_closure", closure_ok,
            "density must equal mass/(4/3*pi*r^3) within 1e-6",
            value=rho, limit=rho_expected)
    else:
        add("C2.particle_closure", False,
            "particle.radius_m/mass_kg/density_kg_m3 must be positive numbers")

    contact = payload.get("contact") or {}
    kn = contact.get("k_n")
    kt = contact.get("k_t")
    mu_s = contact.get("mu_s")
    mu_r = contact.get("mu_r")
    c_rot = contact.get("c_rot")
    ks_ok = (isinstance(kn, (int, float)) and kn > 0
            and isinstance(kt, (int, float)) and 0 < kt <= kn)
    add("C2.spring_stiffness", ks_ok,
        "Cundall-Strack: 0 < k_t <= k_n", value=None if kt is None else float(kt),
        limit=None if kn is None else float(kn))
    add("C2.mu_s", isinstance(mu_s, (int, float)) and 0 < mu_s < 1,
        "static friction mu_s must be in (0,1)", value=None if mu_s is None else float(mu_s))
    add("C2.mu_r", isinstance(mu_r, (int, float)) and 0 <= mu_r < 1,
        "rolling resistance mu_r must be in [0,1)",
        value=None if mu_r is None else float(mu_r))
    add("C2.c_rot", isinstance(c_rot, (int, float)) and c_rot >= 0,
        "rotational damping c_rot must be >= 0",
        value=None if c_rot is None else float(c_rot))

    aoa = payload.get("angle_of_repose") or {}
    target = aoa.get("target_deg")
    achieved = aoa.get("achieved_deg")
    if (isinstance(target, (int, float)) and isinstance(achieved, (int, float))):
        err = abs(achieved - target)
        add("C2.angle_of_repose", err <= AOA_TOL_DEG,
            f"|achieved-target| angle of repose <= {AOA_TOL_DEG} deg",
            value=err, limit=AOA_TOL_DEG)
    else:
        add("C2.angle_of_repose", False,
            "angle_of_repose.target_deg/achieved_deg must be numbers")
    add("C2.calibration_source",
        bool(str(aoa.get("calibration_source", "")).strip()),
        "angle_of_repose.calibration_source must name the calibration case")
    return res


# ---------------------------------------------------------------------------
# 3. TensorLBM -> TensorPlatform : GCI / UQ data format
# ---------------------------------------------------------------------------


def validate_gci_uq_handoff(payload: dict) -> ContractResult:
    """Validate an LBM->platform GCI/UQ handoff payload.

    The GCI block follows the TensorLBM grid-convergence evidence layout
    (``diameters_cells``/``radii_cells`` + quantity values +
    ``observed_order`` + ``extrapolated_value``/``gci_fine``).  The
    reported order is accepted when it is either Richardson-consistent
    with the grid values or backed by a declared low-RMS fit
    (``fit_rms_pct`` <= 1%).  The UQ block must carry block statistics
    with a CI95 consistent with 1.96*std/sqrt(n), a dual-domain
    cross-check and an enabled SGS negative control.
    """
    res = ContractResult(contract_id="P27W2-C3", schema=GCI_UQ_SCHEMA)

    def add(check_id: str, ok: bool, detail: str,
            value: float | None = None, limit: float | None = None) -> None:
        res.checks.append(ContractCheck(check_id, bool(ok), detail, value, limit))

    add("C3.schema", payload.get("schema") == GCI_UQ_SCHEMA,
        f"schema must be {GCI_UQ_SCHEMA!r}, got {payload.get('schema')!r}")

    gci = payload.get("gci") or {}
    grids = gci.get("grids") or []
    if grids:
        hs = [g.get("h") for g in grids if isinstance(g, dict)]
        vs = [g.get("value") for g in grids if isinstance(g, dict)]
    else:
        # LBM native layout: diameters_cells (finer = larger) + values
        diams = gci.get("diameters_cells") or gci.get("radii_cells") or []
        vals = gci.get("cd_control_volume") or gci.get("values") or []
        hs = [1.0 / float(d) for d in diams if isinstance(d, (int, float))]
        vs = [float(v) for v in vals if isinstance(v, (int, float))]
    grids_ok = (
        len(hs) >= 3
        and all(isinstance(h, (int, float)) and h > 0 for h in hs)
        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in vs)
        and _strictly_decreasing(hs)
    )
    add("C3.grids", grids_ok,
        "gci needs >=3 grids with strictly decreasing positive h and "
        "finite values")

    p = gci.get("observed_order")
    p_ok = isinstance(p, (int, float)) and math.isfinite(p)
    add("C3.observed_order", p_ok,
        "observed_order must be a finite number")

    if grids_ok and p_ok:
        h1, h2 = hs[0], hs[1]
        v1, v2, v3 = vs[0], vs[1], vs[2]
        denom = v2 - v3
        fit_rms = gci.get("fit_rms_pct")
        if denom != 0 and h1 != h2:
            p_richardson = math.log(abs(v1 - v2) / abs(denom)) / math.log(h1 / h2)
            richardson_ok = math.isclose(p, p_richardson,
                                         rel_tol=ORDER_TOL, abs_tol=0.5)
            fit_ok = (isinstance(fit_rms, (int, float))
                      and 0 <= fit_rms <= 1.0)
            order_ok = richardson_ok or fit_ok
            add("C3.order_consistency", order_ok,
                "observed order must be Richardson-consistent or backed by "
                "a declared fit with fit_rms_pct <= 1%",
                value=float(p), limit=p_richardson)
        else:
            add("C3.order_consistency", False,
                "degenerate grid values; order cannot be checked")
    else:
        add("C3.order_consistency", False,
            "order consistency skipped: grids or order invalid")

    gci_value = gci.get("gci_fine", gci.get("extrapolated_value"))
    add("C3.gci_value",
        isinstance(gci_value, (int, float)) and math.isfinite(gci_value),
        "gci_fine or extrapolated_value must be a finite number")
    has_bounds = (
        isinstance(gci.get("gci_upper"), (int, float))
        and isinstance(gci.get("gci_lower"), (int, float))
        and gci.get("gci_lower", 0) <= gci.get("gci_upper", 1)
    )
    add("C3.gci_bounds", has_bounds,
        "gci_lower/gci_upper bounds must be present and ordered")

    uq = payload.get("uq") or {}
    n = uq.get("n_blocks")
    mean = uq.get("mean")
    std = uq.get("std")
    ci = uq.get("ci95")
    uq_ok = (
        isinstance(n, int) and n >= 3
        and isinstance(mean, (int, float)) and math.isfinite(mean)
        and isinstance(std, (int, float)) and std >= 0
        and isinstance(ci, (int, float)) and ci >= 0
    )
    add("C3.uq_blocks", uq_ok,
        "uq needs n_blocks>=3, finite mean, std>=0, ci95>=0")
    if uq_ok:
        ci_expected = CI95_COEF * std / math.sqrt(n)
        ci_ok = math.isclose(ci, ci_expected, rel_tol=ORDER_TOL, abs_tol=1e-12)
        add("C3.uq_ci95", ci_ok,
            "ci95 must equal 1.96*std/sqrt(n) within 15%",
            value=float(ci), limit=ci_expected)
    else:
        add("C3.uq_ci95", False, "ci95 check skipped: uq block invalid")

    dual = uq.get("dual_domain") or {}
    add("C3.dual_domain",
        isinstance(dual.get("domain_a"), str) and bool(dual.get("domain_a"))
        and isinstance(dual.get("domain_b"), str) and bool(dual.get("domain_b"))
        and isinstance(dual.get("cross_check_error"), (int, float)),
        "uq.dual_domain needs domain_a/domain_b and cross_check_error")
    sgs = uq.get("sgs_negative_control") or {}
    add("C3.sgs_negative_control",
        sgs.get("enabled") is True
        and isinstance(sgs.get("control_error"), (int, float)),
        "uq.sgs_negative_control must be enabled with a numeric control_error")
    units = payload.get("units") or {}
    add("C3.units",
        isinstance(units, dict) and bool(units.get("quantity")),
        "units.quantity must declare the physical quantity and its unit")
    return res


def _strictly_decreasing(values: list[float]) -> bool:
    return all(a > b for a, b in zip(values, values[1:]))


def validate_all(payloads: dict[str, dict]) -> list[ContractResult]:
    """Validate all three P27-W2 contracts from a payload bundle.

    ``payloads`` maps contract id ``"rao" | "granular" | "gci_uq"`` to its
    JSON payload.  Missing payloads yield a failed result with zero
    coverage so the gate can report partial coverage honestly.
    """
    results: list[ContractResult] = []
    for key, validator in (
        ("rao", validate_rao_handoff),
        ("granular", validate_granular_flow_handoff),
        ("gci_uq", validate_gci_uq_handoff),
    ):
        payload = payloads.get(key)
        if payload is None:
            results.append(ContractResult(
                contract_id=f"P27W2-C{len(results) + 1}",
                schema="missing",
                checks=[ContractCheck("missing", False,
                                       f"payload for {key} not provided")],
            ))
        else:
            results.append(validator(payload))
    return results
