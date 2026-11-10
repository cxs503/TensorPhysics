"""Maturity driver of the ice-vibration transfer chain with a QZS isolation
layer (campaign node ``p227w2-N08``).

What this driver adds to the verified chain (``chain.py``, node p228w1-N08)
-------------------------------------------------------------------------
The verified chain reports, as disclosure D1, that a conventional resilient
mount *cannot* attenuate the hull-girder band that carries the ice energy:
with ``f_n = 4 Hz`` the RMS ratio ``RMS(equipment base)/RMS(hull)`` is 1.21,
and the node's ``|H(f)| < 1`` criterion could therefore only be posed above
``sqrt(2) f_n``.  This driver closes that gap with the quasi-zero-stiffness
mount of :mod:`.mitigation`:

1. **Design step** (prerequisite, frequency domain).  The ice-energy band is
   taken from the hull acceleration PSD itself (0.3-3 Hz), a QZS design map is
   built over ``(f_n,dyn, zeta)`` under two constraints - static deflection not
   above :data:`MitigationConfig.static_deflection_limit_m` and residual
   stiffness not below :data:`MitigationConfig.residual_stiffness_budget` -
   and the design point is selected by a rule fixed before the numbers are
   seen (:func:`mitigation.select_design`).  The QZS mount keeps the 15.5 mm
   static deflection of the 4 Hz mount while moving the dynamic frequency into
   the hull band.
2. **Chain rerun** (time domain).  For every ice severity the hull
   acceleration PSD is pushed through (a) the conventional mount and (b) the
   QZS mount, then through the pipe stress transfer, and finally through the
   real random-phase stress synthesis + ASTM E1049-85 rainflow + DNV-RP-C203
   D-curve + Miner accumulation of :mod:`.pipe`.  The baseline and the QZS
   lives are reported side by side, so the maturity claim has a pre/post
   comparison instead of an isolated number.
3. **Robustness** (the price of the QZS design).  Stiffness-cancellation
   tolerance (elastomer ageing), mount stroke under the declared wave /
   whipping envelope, and the amplitude threshold of the Duffing
   linearisation are all computed and disclosed, because a 98 % stiffness
   cancellation is only useful if it survives service.

Criteria (numbers stored with each row)
---------------------------------------
``M1`` ``f_wet < f_dry`` for every elastic mode (the node's first criterion).
``M2`` ``|H(f)| < 1`` above ``sqrt(2) f_n`` for both mounts (pointwise sense).
``M3`` ``RMS(base) < RMS(hull)`` over the ice band with the QZS mount; the
       conventional mount is reported on the same band as the pre/post pair.
``M4`` pipe stress positive for the three ice severities.
``M5`` fatigue life finite and positive for the three ice severities.
``M6`` fatigue life decreasing when the ice load increases (monotonic).
``M7`` QZS hardware feasibility: static deflection, residual stiffness budget,
       stroke under the declared wave/whipping envelope.
``M8`` model verification: RK4 time integration reproduces the analytic
       transmissibility of the design mount.
``M9`` fatigue-life improvement factor of the QZS mount over the conventional
       mount at the design severity.

Disclosures ``D1``-``D8`` carry the limits (passive impossibility at DC, the
QZS amplification band, the stiffness tolerance window, stroke versus
wave/whipping motion, the Duffing linearity amplitude, the absolute life
levels, the declared inputs and the ice-band energy split).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..fatigue.sn_curve import DNCurve
from ..hull_hydroelastic.presets import HullCase, hull_cases
from .excitation import IceTorqueExcitation, milling_torque_excitation
from .hull_modal import HullModalBasis, geometry_gate, wet_dry_modal_basis
from .mitigation import (
    DEFAULT_WAVE_WHIPPING_LOAD,
    QZSMount,
    band_attenuation,
    design_map,
    relative_displacement_psd,
    select_design,
    service_stroke,
    simulate_base_excitation,
    tolerance_window,
)
from .pipe import PipeSpan, fatigue_life
from .response import acceleration_psd
from .transfer import MountIsolation

__all__ = ["MitigationConfig", "MitigationResult", "run_mitigation"]

DEFAULT_SEVERITIES: tuple[str, ...] = ("light", "design", "heavy")
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0

DEFAULT_F_N_GRID: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.50)
DEFAULT_ZETA_GRID: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy containers to plain Python for JSON output."""
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _hull_case(name: str) -> HullCase:
    for case in hull_cases():
        if case.name.upper() == name.upper():
            return case
    raise ValueError(f"unknown hull case {name!r}; available: {[c.name for c in hull_cases()]}")


def _band_rms(f: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f <= hi)
    if np.count_nonzero(m) < 2:
        return 0.0
    return float(np.sqrt(max(np.trapezoid(psd[m], f[m]), 0.0)))


@dataclass
class MitigationConfig:
    """Configuration of one maturity run (all inputs declared, none fitted)."""

    hull_case_name: str = "KCS"
    n_modes: int = 12
    n_elements: int = 100
    zeta_structural: float = 0.02
    n_rps: float = 2.5
    duration_s: float = 60.0
    dt_s: float = 1.0e-3
    eta_hull: float = 0.30
    seed: int = 20261110
    #: conventional mount of the verified chain (baseline of the comparison)
    baseline_mount_f_n_hz: float = 4.0
    baseline_mount_zeta: float = 0.15
    #: deck / foundation bracket (stiff local mode, path A of the verified chain)
    foundation_f_n_hz: float = 25.0
    foundation_zeta: float = 0.02
    equipment_mass_kg: float = 2200.0
    #: ice-energy band (0.3-3 Hz: 2-node wet mode to above the 4-node mode)
    ice_band_hz: tuple[float, float] = (0.3, 3.0)
    full_band_hz: tuple[float, float] = (0.2, 500.0)
    #: QZS design constraints
    static_deflection_limit_m: float = 0.05
    residual_stiffness_budget: float = 0.010
    qzs_zeta: float = 0.15
    qzs_zeta_min: float = 0.15
    qzs_ns_range_m: float = 0.020
    qzs_stroke_m: float = 0.050
    stop_force_limit_fraction: float = 0.25
    #: pipe / fatigue settings (identical to the verified chain)
    pipe: PipeSpan = field(default_factory=PipeSpan)
    operating_fraction: float = 0.25
    fatigue_duration_s: float = 60.0
    fatigue_dt_s: float = 2.0e-3
    design_life_years: float = 25.0
    sn_curve_class: str = "D"
    sn_environment: str = "seawater_cp"
    #: robustness of the design statement to the ice-band definition
    sensitivity_bands: tuple[tuple[float, float], ...] = ((0.3, 3.0), (0.8, 4.0), (0.2, 5.0), (0.3, 5.0))

    def mount_baseline(self) -> MountIsolation:
        return MountIsolation(
            f_n_hz=float(self.baseline_mount_f_n_hz),
            zeta=float(self.baseline_mount_zeta),
            equipment_mass_kg=float(self.equipment_mass_kg),
            label="conventional resilient mount (baseline)",
        )

    def bracket(self) -> MountIsolation:
        return MountIsolation(
            f_n_hz=float(self.foundation_f_n_hz),
            zeta=float(self.foundation_zeta),
            label="deck / foundation bracket",
        )

    def sn_curve(self) -> DNCurve:
        return DNCurve(self.sn_curve_class, self.sn_environment)


@dataclass
class MitigationResult:
    """Payload of one maturity run."""

    config: MitigationConfig
    basis: HullModalBasis
    geometry: Any
    selection: dict[str, Any]
    design: QZSMount
    baseline: MountIsolation
    severities: dict[str, dict[str, Any]]
    design_map_rows: list[dict[str, Any]]
    tolerance: dict[str, Any]
    stroke: dict[str, Any]
    verification: dict[str, Any]
    criteria: list[dict[str, Any]]
    disclosures: list[dict[str, Any]]
    curves: dict[str, Any]

    def sn_curve_class(self) -> str:
        return f"{self.config.sn_curve_class}/{self.config.sn_environment}"

    band_sensitivity: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return all(c["passed"] for c in self.criteria)

    def summary(self) -> str:
        c = {row["id"]: row for row in self.criteria}
        life = {k: v["qzs"]["fatigue"]["life_years"] for k, v in self.severities.items()}
        ratio = c["M3"]["value"]["qzs_ratio"]
        base_ratio = c["M3"]["value"]["baseline_ratio"]
        return (
            f"max wet/dry ratio {np.max(self.basis.frequency_ratio):.3f} "
            f"(M1 {'PASS' if c['M1']['passed'] else 'FAIL'}); band RMS base/hull "
            f"conventional {base_ratio:.3f} -> QZS {ratio:.3f} (M3 {'PASS' if c['M3']['passed'] else 'FAIL'}); "
            f"QZS life light/design/heavy = {life['light']:.3e}/{life['design']:.3e}/"
            f"{life['heavy']:.3e} yr (M5/M6 {'PASS' if c['M5']['passed'] and c['M6']['passed'] else 'FAIL'}); "
            f"life gain at design severity x{c['M9']['value']:.1f}"
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "node": "p227w2-N08",
            "maturity_of": "p228w1-N08",
            "chain": (
                "ice milling torque -> hull wet modes -> foundation/bracket -> QZS isolation -> "
                "equipment base acceleration -> pipe stress spectrum -> rainflow/S-N/Miner life"
            ),
            "config": {
                "hull_case": self.config.hull_case_name,
                "n_modes": int(self.config.n_modes),
                "n_elements": int(self.config.n_elements),
                "zeta_structural": float(self.config.zeta_structural),
                "n_rps": float(self.config.n_rps),
                "duration_s": float(self.config.duration_s),
                "dt_s": float(self.config.dt_s),
                "eta_hull": float(self.config.eta_hull),
                "seed": int(self.config.seed),
                "ice_band_hz": [float(v) for v in self.config.ice_band_hz],
                "full_band_hz": [float(v) for v in self.config.full_band_hz],
                "static_deflection_limit_m": float(self.config.static_deflection_limit_m),
                "residual_stiffness_budget": float(self.config.residual_stiffness_budget),
                "operating_fraction": float(self.config.operating_fraction),
                "design_life_years": float(self.config.design_life_years),
                "fatigue_duration_s": float(self.config.fatigue_duration_s),
                "fatigue_dt_s": float(self.config.fatigue_dt_s),
            },
            "geometry_gate": self.geometry.to_json_dict(),
            "hull_modal": self.basis.to_json_dict(),
            "sn_curve": {
                "class": self.sn_curve_class(),
                "m1": float(self.config.sn_curve().params.m1),
                "m2": float(self.config.sn_curve().params.m2),
                "log_K1": float(self.config.sn_curve().params.log_K1),
                "log_K2": float(self.config.sn_curve().params.log_K2),
                "n_transition": float(self.config.sn_curve().params.n_transition),
                "stress_at_1e7_cycles_MPa": float(self.config.sn_curve().stress_for_life(1.0e7)),
                "standard": "DNV-RP-C203 D curve, seawater with cathodic protection",
            },
            "baseline_mount": self.baseline.as_dict(),
            "qzs_mount": self.design.as_dict(),
            "design_selection": self.selection,
            "design_map": self.design_map_rows,
            "band_sensitivity": self.band_sensitivity,
            "tolerance": self.tolerance,
            "stroke": self.stroke,
            "verification": self.verification,
            "severities": self.severities,
            "criteria": self.criteria,
            "disclosures": self.disclosures,
            "curves": _jsonable(self.curves),
            "all_criteria_pass": self.all_pass,
        }


def _criterion(cid: str, name: str, value: Any, threshold: Any, passed: bool, detail: str = "", units: str = "") -> dict[str, Any]:
    return {
        "id": cid,
        "name": name,
        "value": value,
        "threshold": threshold,
        "units": units,
        "passed": bool(passed),
        "detail": detail,
    }


def run_mitigation(config: MitigationConfig | None = None) -> MitigationResult:
    """Run the maturity chain: design the QZS mount, rerun the chain, judge it."""
    cfg = config or MitigationConfig()
    case = _hull_case(cfg.hull_case_name)
    geometry = geometry_gate(case)
    basis = wet_dry_modal_basis(
        case, n_modes=cfg.n_modes, n_elements=cfg.n_elements, zeta_structural=cfg.zeta_structural
    )
    baseline = cfg.mount_baseline()
    bracket = cfg.bracket()
    sn_curve = cfg.sn_curve()

    # ---------------------------------------------------------------- design
    design_exc = milling_torque_excitation(
        severity="design",
        n_rps=cfg.n_rps,
        duration_s=cfg.duration_s,
        dt_s=cfg.dt_s,
        eta_hull=cfg.eta_hull,
        scale=1.0,
        seed=cfg.seed,
    )
    design_resp = acceleration_psd(basis, design_exc.freqs_hz, design_exc.force_spectrum_scaled())
    f = design_resp.freqs_hz
    s_hull = design_resp.s_acceleration_m2_s4_per_hz
    s_support = s_hull * bracket.magnitude(f) ** 2

    rows = design_map(
        f,
        s_support,
        cfg.ice_band_hz,
        f_rigid_hz=cfg.baseline_mount_f_n_hz,
        equipment_mass_kg=cfg.equipment_mass_kg,
        f_n_grid=DEFAULT_F_N_GRID,
        zeta_grid=DEFAULT_ZETA_GRID,
        static_deflection_limit_m=cfg.static_deflection_limit_m,
        residual_stiffness_budget=cfg.residual_stiffness_budget,
        ns_range_m=cfg.qzs_ns_range_m,
        stroke_m=cfg.qzs_stroke_m,
        zeta_min=cfg.qzs_zeta_min,
        service_loads=DEFAULT_WAVE_WHIPPING_LOAD,
        stop_force_limit_fraction=cfg.stop_force_limit_fraction,
    )
    selection = select_design(
        rows,
        static_deflection_limit_m=cfg.static_deflection_limit_m,
        residual_stiffness_budget=cfg.residual_stiffness_budget,
    )
    best = selection["selected"]
    design = QZSMount(
        equipment_mass_kg=cfg.equipment_mass_kg,
        f_rigid_hz=cfg.baseline_mount_f_n_hz,
        residual_stiffness_fraction=float(best["residual_stiffness_fraction"]),
        zeta=float(best["zeta"]),
        stroke_m=cfg.qzs_stroke_m,
        ns_range_m=cfg.qzs_ns_range_m,
    )

    tolerance = tolerance_window(f, s_support, cfg.ice_band_hz, design)

    # -------------------------------------------------------------- the chain
    severities: dict[str, dict[str, Any]] = {}
    curves: dict[str, Any] = {}
    for severity in DEFAULT_SEVERITIES:
        exc: IceTorqueExcitation = milling_torque_excitation(
            severity=severity,
            n_rps=cfg.n_rps,
            duration_s=cfg.duration_s,
            dt_s=cfg.dt_s,
            eta_hull=cfg.eta_hull,
            scale=1.0,
            seed=cfg.seed,
        )
        resp = acceleration_psd(basis, exc.freqs_hz, exc.force_spectrum_scaled())
        fr = resp.freqs_hz
        s_h = resp.s_acceleration_m2_s4_per_hz
        s_sup = s_h * bracket.magnitude(fr) ** 2

        scf_factor = float(
            cfg.pipe.scf
            * (cfg.pipe.wall_thickness_m / cfg.pipe.reference_thickness_m) ** cfg.pipe.thickness_exponent
        )
        t_sigma = cfg.pipe.stress_transfer(fr)

        blocks: dict[str, Any] = {}
        for label, mount in (("conventional", baseline), ("qzs", design)):
            h2 = mount.magnitude(fr) ** 2
            s_base = s_sup * h2
            s_stress = np.abs(t_sigma) ** 2 * s_base * scf_factor**2
            stress_rms = float(np.sqrt(max(np.trapezoid(s_stress, fr), 0.0)))
            att = band_attenuation(fr, s_sup, mount, cfg.ice_band_hz)
            full = band_attenuation(fr, s_sup, mount, cfg.full_band_hz)
            fat = fatigue_life(
                s_stress,
                fr,
                cfg.pipe,
                sn_curve,
                duration_s=cfg.fatigue_duration_s,
                dt_s=cfg.fatigue_dt_s,
                operating_fraction=cfg.operating_fraction,
                seed=cfg.seed,
            )
            blocks[label] = {
                "mount": mount.as_dict() if hasattr(mount, "as_dict") else {},
                "rms_base_m_s2": float(np.sqrt(max(np.trapezoid(s_base, fr), 0.0))),
                "rms_base_ice_band_m_s2": att["rms_base_m_s2"],
                "rms_ratio_base_over_hull_ice_band": att["rms_ratio_base_over_hull"],
                "rms_ratio_base_over_hull_full_band": full["rms_ratio_base_over_hull"],
                "max_magnitude_above_isolation": att["max_magnitude_above_isolation"],
                "stress_rms_MPa": stress_rms / 1.0e6,
                "stress_rms_scf_MPa": fat["stress_rms_MPa"],
                "fatigue": {
                    "life_years": fat["life_years"],
                    "damage_per_year": fat["damage_per_year"],
                    "max_stress_range_MPa": fat["max_stress_range_MPa"],
                    "equivalent_stress_range_1e7_MPa": fat["equivalent_stress_range_1e7_MPa"],
                    "rainflow_cycles": fat["rainflow_cycles"],
                    "rainflow_to_narrowband_ratio": fat["narrowband_crosscheck"]["rainflow_to_narrowband_ratio"],
                },
                "stress_psd": s_stress,
                "base_psd": s_base,
            }

        severities[severity] = {
            "excitation": exc.as_dict(),
            "hull_rms_m_s2": resp.rms_m_s2,
            "hull_peak_frequency_hz": resp.peak_frequency_hz,
            "hull_ice_band_rms_m_s2": _band_rms(fr, s_h, *cfg.ice_band_hz),
            "conventional": {k: v for k, v in blocks["conventional"].items() if not k.endswith("_psd")},
            "qzs": {k: v for k, v in blocks["qzs"].items() if not k.endswith("_psd")},
            "life_gain_qzs_over_conventional": float(
                blocks["qzs"]["fatigue"]["life_years"] / blocks["conventional"]["fatigue"]["life_years"]
            )
            if blocks["conventional"]["fatigue"]["life_years"] > 0
            else float("inf"),
        }
        curves[severity] = {
            "freqs_hz": fr,
            "hull_psd": s_h,
            "support_psd": s_sup,
            "base_psd_conventional": blocks["conventional"]["base_psd"],
            "base_psd_qzs": blocks["qzs"]["base_psd"],
            "h_magnitude_conventional": baseline.magnitude(fr),
            "h_magnitude_qzs": design.magnitude(fr),
            "stress_psd_conventional": blocks["conventional"]["stress_psd"],
            "stress_psd_qzs": blocks["qzs"]["stress_psd"],
            "life_years_qzs": blocks["qzs"]["fatigue"]["life_years"],
            "life_years_conventional": blocks["conventional"]["fatigue"]["life_years"],
            "stress_rms_MPa_qzs": blocks["qzs"]["stress_rms_MPa"],
            "stress_rms_MPa_conventional": blocks["conventional"]["stress_rms_MPa"],
        }

    # ------------------------------------------------------- band sensitivity
    f_des = curves["design"]["freqs_hz"]
    s_sup_des = curves["design"]["support_psd"]
    band_sensitivity = []
    for band in cfg.sensitivity_bands:
        band_sensitivity.append(
            {
                "band_hz": [float(v) for v in band],
                "conventional_ratio": float(
                    band_attenuation(f_des, s_sup_des, baseline, band)["rms_ratio_base_over_hull"]
                ),
                "qzs_ratio": float(
                    band_attenuation(f_des, s_sup_des, design, band)["rms_ratio_base_over_hull"]
                ),
            }
        )

    # ---------------------------------------------------------------- stroke
    service = service_stroke(design, DEFAULT_WAVE_WHIPPING_LOAD)
    ice_rel_psd = relative_displacement_psd(f, s_support, design)
    ice_rel_rms = float(np.sqrt(max(np.trapezoid(ice_rel_psd, f), 0.0)))
    stroke_rows = [{**row, "source": "declared"} for row in service["rows"]]
    stroke_rows.append(
        {
            "label": "ice-induced (this chain)",
            "freq_hz": None,
            "accel_rms_m_s2": float(np.sqrt(max(np.trapezoid(s_support, f), 0.0))),
            "relative_displacement_rms_m": ice_rel_rms,
            "source": "computed",
        }
    )
    stroke = {
        "rows": stroke_rows,
        "stroke_m": float(design.stroke_m),
        "ns_range_m": float(design.ns_range_m),
        "ice_relative_displacement_rms_m": ice_rel_rms,
        "linearity_amplitude_m": design.linearity_amplitude_m(),
        "worst_relative_displacement_rms_m": float(service["worst_relative_displacement_rms_m"]),
        "worst_case": str(service["worst_case"]),
        "peak_factor": float(service["peak_factor"]),
        "ice_case_inside_linear_range": bool(ice_rel_rms <= design.linearity_amplitude_m()),
        "worst_case_inside_stroke": bool(
            2.0 * float(service["worst_relative_displacement_rms_m"]) <= design.stroke_m
        ),
        "peak_relative_displacement_m": float(service["peak_relative_displacement_m"]),
        "bump_stop_engaged": bool(service["stop_engaged"]),
        "bump_stop_force_n": float(service["bump_stop_force_n"]),
        "bump_stop_force_fraction_of_weight": float(service["bump_stop_force_fraction_of_weight"]),
        "bump_stop_stiffness_n_m": float(service["bump_stop_stiffness_n_m"]),
        "stop_force_limit_fraction": float(cfg.stop_force_limit_fraction),
    }

    # ---------------------------------------------------------- verification
    probe = float(np.sqrt(design.residual_stiffness_fraction) * cfg.baseline_mount_f_n_hz)
    sim = simulate_base_excitation(
        f_n_hz=design.f_n_hz,
        zeta=design.zeta,
        f_probe_hz=probe,
        cycles=120,
        steps_per_cycle=400,
        relative_amplitude_m=1.0e-3,
    )
    analytic = float(design.magnitude(np.array([probe]))[0])
    f_static = np.array([1.0e-6])
    modal_static = float(np.abs(cfg.pipe.stress_transfer(f_static))[0])
    closed_static = float(cfg.pipe.static_stress_per_acceleration)
    verification = {
        "method": "RK4 time integration of the base-excited SDOF vs analytic |H(f)|",
        "pipe_static_limit": {
            "modal_transfer_Pa_per_m_s2": modal_static,
            "closed_form_mL2_over_8Z_Pa_per_m_s2": closed_static,
            "relative_error": float(abs(modal_static - closed_static) / closed_static),
        },
        "probe_frequency_hz": probe,
        "analytic_transmissibility": analytic,
        "time_domain_transmissibility": float(sim["acceleration_transmissibility"]),
        "relative_error": float(abs(sim["acceleration_transmissibility"] - analytic) / analytic),
        "dc_limit_magnitude": float(design.magnitude(np.array([1.0e-6]))[0]),
        "linearity_amplitude_m": design.linearity_amplitude_m(),
        "backbone_peak_hz_at_linearity_amplitude": design.backbone_peak_hz(design.linearity_amplitude_m()),
    }

    # ------------------------------------------------------------- criteria
    design_ratio = severities["design"]["qzs"]["rms_ratio_base_over_hull_ice_band"]
    base_ratio = severities["design"]["conventional"]["rms_ratio_base_over_hull_ice_band"]
    base_full_ratio = severities["design"]["conventional"]["rms_ratio_base_over_hull_full_band"]
    qzs_full_ratio = severities["design"]["qzs"]["rms_ratio_base_over_hull_full_band"]

    lives_qzs = {k: v["qzs"]["fatigue"]["life_years"] for k, v in severities.items()}
    stresses = {k: v["qzs"]["stress_rms_MPa"] for k, v in severities.items()}
    life_gain = severities["design"]["life_gain_qzs_over_conventional"]
    finite_positive = all(np.isfinite(v) and v > 0.0 for v in lives_qzs.values())
    monotonic = lives_qzs["light"] > lives_qzs["design"] > lives_qzs["heavy"]

    criteria = [
        _criterion(
            "M1",
            "wet natural frequencies below dry (all elastic modes)",
            [float(v) for v in basis.frequency_ratio],
            1.0,
            bool(basis.all_wet_below_dry),
            f"max ratio {float(np.max(basis.frequency_ratio)):.4f}",
        ),
        _criterion(
            "M2",
            "|H(f)| < 1 pointwise above sqrt(2) f_n (both mounts)",
            {
                "conventional_max_above_isolation": severities["design"]["conventional"]["max_magnitude_above_isolation"],
                "qzs_max_above_isolation": severities["design"]["qzs"]["max_magnitude_above_isolation"],
            },
            1.0,
            severities["design"]["conventional"]["max_magnitude_above_isolation"] < 1.0
            and severities["design"]["qzs"]["max_magnitude_above_isolation"] < 1.0,
            f"conventional f_n {baseline.f_n_hz:.2f} Hz (iso {baseline.isolation_frequency_hz:.2f}), "
            f"QZS f_n {design.f_n_hz:.2f} Hz (iso {design.isolation_frequency_hz:.2f})",
        ),
        _criterion(
            "M3",
            "RMS(equipment base) < RMS(hull) over the ice band (energy-sense |H|<1)",
            {
                "baseline_ratio": float(base_ratio),
                "qzs_ratio": float(design_ratio),
                "band_hz": [float(v) for v in cfg.ice_band_hz],
                "full_band_baseline_ratio": float(base_full_ratio),
                "full_band_qzs_ratio": float(qzs_full_ratio),
            },
            1.0,
            bool(design_ratio < 1.0),
            "conventional mount amplifies the ice band (ratio > 1); QZS must bring it below unity",
        ),
        _criterion(
            "M4",
            "pipe stress positive for the three ice severities (QZS chain)",
            {k: float(v) for k, v in stresses.items()},
            0.0,
            all(v > 0.0 for v in stresses.values()),
            "stress RMS at the pipe detail (SCF + thickness corrected)",
            "MPa",
        ),
        _criterion(
            "M5",
            "fatigue life finite and positive for the three ice severities (QZS chain)",
            {k: float(v) for k, v in lives_qzs.items()},
            0.0,
            bool(finite_positive),
            "random-phase stress synthesis + rainflow + DNV-RP-C203 D + Miner",
            "years",
        ),
        _criterion(
            "M6",
            "fatigue life decreases when the ice load increases",
            {
                "light": float(lives_qzs["light"]),
                "design": float(lives_qzs["design"]),
                "heavy": float(lives_qzs["heavy"]),
            },
            "life(light) > life(design) > life(heavy)",
            bool(monotonic),
            "monotonicity of the life vs ice-severity curve on the QZS chain",
        ),
        _criterion(
            "M7",
            "QZS hardware feasibility (static deflection, stiffness budget, stroke)",
            {
                "static_deflection_m": float(design.static_deflection_m),
                "static_deflection_limit_m": float(cfg.static_deflection_limit_m),
                "residual_stiffness_fraction": float(design.residual_stiffness_fraction),
                "residual_stiffness_budget": float(cfg.residual_stiffness_budget),
                "zeta": float(design.zeta),
                "zeta_min": float(cfg.qzs_zeta_min),
                "peak_amplification": design.peak_amplification,
                "ice_band_relative_displacement_rms_m": stroke["ice_relative_displacement_rms_m"],
                "linearity_amplitude_m": stroke["linearity_amplitude_m"],
                "bump_stop_force_fraction_of_weight": stroke["bump_stop_force_fraction_of_weight"],
                "stop_force_limit_fraction": float(cfg.stop_force_limit_fraction),
                "stroke_m": float(design.stroke_m),
                "peak_service_relative_displacement_m": stroke["peak_relative_displacement_m"],
            },
            "static deflection, stiffness budget, damping floor, linear ice band and bump-stop force",
            bool(
                design.static_deflection_m <= cfg.static_deflection_limit_m
                and design.residual_stiffness_fraction >= cfg.residual_stiffness_budget
                and design.zeta >= cfg.qzs_zeta_min
                and stroke["ice_case_inside_linear_range"]
                and (
                    stroke["worst_case_inside_stroke"]
                    or stroke["bump_stop_force_fraction_of_weight"] <= cfg.stop_force_limit_fraction
                )
            ),
            "wave/whipping motion (2 x RMS) either fits the stroke or is taken by a bump stop below the force limit",
        ),
        _criterion(
            "M8",
            "RK4 time-domain integration reproduces the analytic transmissibility",
            float(verification["relative_error"]),
            0.01,
            bool(verification["relative_error"] < 0.01),
            "independent check of the frequency-domain QZS model",
        ),
        _criterion(
            "M9",
            "fatigue-life gain of the QZS mount over the conventional mount (design severity)",
            float(life_gain),
            1.0,
            bool(life_gain > 1.0),
            "relative metric; absolute life levels are disclosed (D6)",
        ),
    ]

    # ---------------------------------------------------------- disclosures
    d1_ratio = base_full_ratio
    disclosures = [
        {
            "id": "D1",
            "statement": (
                "Any passive mount has |H(0)| = 1: at zero frequency the masses move rigidly with the "
                "foundation, so 'base acceleration below hull acceleration at every frequency' is "
                "impossible for a passive suspension. The conventional mount even amplifies the ice band; "
                "the criterion is therefore posed pointwise above sqrt(2) f_n (M2) and in the energy sense "
                "over the ice band (M3)."
            ),
            "value": float(d1_ratio),
        },
        {
            "id": "D2",
            "statement": (
                "QZS amplification band: below sqrt(2) f_n the softened mount still amplifies, with a "
                "narrow-band resonance at f_n,dyn. Pointwise |H| maximum inside the ice band is reported "
                "with the design; only the band integral is below unity."
            ),
            "value": {
                "peak_amplification": float(design.peak_amplification),
                "grid_max_in_band": float(
                    np.max(
                        design.magnitude(
                            f[(f >= float(cfg.ice_band_hz[0])) & (f <= float(cfg.ice_band_hz[1]))]
                        )
                    )
                ),
                "note": "grid max is sampled on the Welch frequency grid; peak_amplification is the analytic maximum",
            },
        },
        {
            "id": "D3",
            "statement": (
                "Stiffness-cancellation tolerance: an error delta_alpha in the negative-stiffness fraction "
                "moves the dynamic frequency by -delta_alpha/(2 (1-alpha)). The window in which the "
                "attenuation criterion still holds is reported; outside it the design degrades to the "
                "conventional-mount regime (stiffer, band ratio > 1)."
            ),
            "value": {
                "max_abs_delta_alpha_still_attenuating": tolerance["max_abs_delta_alpha_still_attenuating"],
                "first_failing_delta_alpha": tolerance["first_failing_delta_alpha"],
            },
        },
        {
            "id": "D4",
            "statement": (
                "Stroke: the QZS mechanism must remain linear over the whole service motion. Wave-induced "
                "and whipping relative displacements (declared envelope) are compared with the ice-band "
                "motion and with the stroke; they exceed the ice-band motion by orders of magnitude."
            ),
            "value": {
                "ice_relative_displacement_rms_m": stroke["ice_relative_displacement_rms_m"],
                "linearity_amplitude_m": stroke["linearity_amplitude_m"],
                "worst_case": stroke["worst_case"],
                "worst_relative_displacement_rms_m": stroke["worst_relative_displacement_rms_m"],
                "peak_service_relative_displacement_m": stroke["peak_relative_displacement_m"],
                "stroke_m": stroke["stroke_m"],
                "bump_stop_engaged": stroke["bump_stop_engaged"],
                "bump_stop_force_fraction_of_weight": stroke["bump_stop_force_fraction_of_weight"],
                "bump_stop_stiffness_N_per_m": stroke["bump_stop_stiffness_n_m"],
            },
        },
        {
            "id": "D5",
            "statement": (
                "Duffing linearisation: beyond the NS linear range the QZS force gains a cubic term "
                "(beta = alpha k_v / ns_range^2), the resonance backbone hardens. The relative displacement "
                "below which the cubic correction stays inside 1 % is reported and compared with the "
                "ice-band motion."
            ),
            "value": {
                "cubic_coefficient_N_per_m3": design.cubic_coefficient_n_m3,
                "linearity_amplitude_m": design.linearity_amplitude_m(),
                "ice_relative_displacement_rms_m": stroke["ice_relative_displacement_rms_m"],
                "backbone_peak_hz_at_linearity_amplitude": verification["backbone_peak_hz_at_linearity_amplitude"],
            },
        },
        {
            "id": "D6",
            "statement": (
                "Absolute life levels: the ice-induced pipe detail stresses are far below the DNV-RP-C203 "
                "D-curve endurance level (50.2 MPa at 1e7 cycles), so the computed lives are astronomically "
                "long (1e8 yr and above). Ice-vibration fatigue does not govern this pipe detail; the QZS "
                "gain is a relative mitigation metric, not a design-life statement."
            ),
            "value": {
                "stress_rms_MPa_design": float(stresses["design"]),
                "life_years_design": float(lives_qzs["design"]),
                "stress_at_1e7_cycles_MPa": float(sn_curve.stress_for_life(1.0e7)),
            },
        },
        {
            "id": "D7",
            "statement": (
                "Declared inputs: wave/whipping envelope (used only for the stroke check), QZS damping "
                "ratio, NS linear range, stroke, equipment mass, foundation bracket mode and the "
                "shaft-to-hull transmission efficiency inherited from the verified chain. The QZS "
                "linearisation is used for the chain; the nonlinear check is reported separately."
            ),
            "value": {
                "eta_hull": float(cfg.eta_hull),
                "qzs_zeta": float(design.zeta),
                "ns_range_m": float(design.ns_range_m),
                "stroke_m": float(design.stroke_m),
                "equipment_mass_kg": float(cfg.equipment_mass_kg),
                "foundation_f_n_hz": float(cfg.foundation_f_n_hz),
                "wave_whipping_envelope": [dict(r) for r in DEFAULT_WAVE_WHIPPING_LOAD],
            },
        },
        {
            "id": "D8",
            "statement": (
                "Ice-band definition and energy split: the band is fixed at 0.3-3 Hz before the design is "
                "selected; the share of hull acceleration variance below the QZS isolation frequency is "
                "reported because that share cannot be attenuated by this mount."
            ),
            "value": {
                "ice_band_hz": [float(v) for v in cfg.ice_band_hz],
                "qzs_isolation_frequency_hz": design.isolation_frequency_hz,
                "variance_share_below_isolation": float(
                    np.trapezoid(
                        s_support[f <= design.isolation_frequency_hz], f[f <= design.isolation_frequency_hz]
                    )
                    / max(np.trapezoid(s_support, f), 1e-300)
                ),
            },
        },
    ]

    return MitigationResult(
        config=cfg,
        basis=basis,
        geometry=geometry,
        selection=selection,
        design=design,
        baseline=baseline,
        severities=severities,
        design_map_rows=rows,
        tolerance=tolerance,
        stroke=stroke,
        verification=verification,
        criteria=criteria,
        disclosures=disclosures,
        curves=curves,
        band_sensitivity=band_sensitivity,
    )