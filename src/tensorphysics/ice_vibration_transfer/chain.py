"""End-to-end ice-induced vibration transfer chain (campaign node p228w1-N08).

Two load paths are carried through the same downstream chain, because the
physics says they differ by orders of magnitude and only one of them can govern
the pipe fatigue:

**Path A - global hull girder** (the chain as specified by the node):
::

    N04 DEM milling torque spectrum                   (:mod:`.excitation`)
      -> self-equilibrated hull excitation couple     (:mod:`.excitation`)
      -> wet / dry hull modal basis                   (:mod:`.hull_modal`)
      -> hull-girder acceleration at the station      (:mod:`.response`)
      -> foundation bracket (deck local mode A_fnd)   [base-excited SDOF]
      -> resilient equipment mount H(f)               [base-excited SDOF]
      -> equipment base acceleration                  a_base = A_fnd H a_hull

**Path B - local shaft-line reaction** (the engine-room load path of the same
torque fluctuation):
::

    torque fluctuation -> tangential force at the ice contact radius
      -> engine-room foundation mass (SDOF, m_fnd, f_fnd, zeta)
      -> resilient equipment mount H(f) -> equipment base acceleration

Both paths share the pipe-span stress model, the rainflow counting, the
DNV-RP-C203 S-N curve and the Palmgren-Miner accumulation (:mod:`.pipe`).  The
payload reports both, declares :data:`GOVERNING_PATH` for the fatigue criteria,
and quantifies the gap between them: on the reference hull the global path is
three decades weaker than the local path, i.e. the shaft-torque excitation of the
*hull girder* is not the fatigue driver even though it is the load path the node
asks about.

Criteria (all numeric, stored with their numbers)
------------------------------------------------
``G1`` geometry gate (Cb / displacement / L/B / B/T vs published reference).
``C1`` ``f_wet < f_dry`` for every elastic mode.
``C2a`` ``|H(f)| < 1`` above the isolation frequency ``sqrt(2) f_n``.
``C2b`` RMS(base) < RMS(support) over the whole attenuation band (path A).
``C2c`` RMS(base) < RMS(hull girder) over the attenuation band (path A).
``C3a`` ``life(light) > life(design) > life(heavy)`` on the governing path.
``C3b`` life strictly decreasing along the torque-intensity sweep.
``C4`` rainflow damage vs narrowband Rayleigh closed form.
``C5`` pipe modal stress transfer reproduces the static ``m L^2/(8 Z)`` limit.
``C6`` static-strength sanity of the pipe detail (< 2/3 yield).

Disclosures ``D1``-``D8`` report the model limits (full-band transmissibility,
variance shares, stress amplification still required for the design life, S-N
provenance, declared parameters, critical torque-intensity multiplier and the
path-A vs path-B gap).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from ..fatigue.sn_curve import DNCurve
from ..hull_hydroelastic.presets import HullCase, hull_cases
from .excitation import IceTorqueExcitation, milling_torque_excitation
from .hull_modal import HullModalBasis, wet_dry_modal_basis
from .pipe import PipeSpan, fatigue_life
from .response import ResponseSpectrum, acceleration_psd, band_rms, point_force_acceleration
from .transfer import MountIsolation, transmissibility_envelope

__all__ = [
    "ChainConfig",
    "ChainResult",
    "DEFAULT_SEVERITIES",
    "GOVERNING_PATH",
    "run_chain",
    "replace_scaled",
]

DEFAULT_SEVERITIES: tuple[str, ...] = ("light", "design", "heavy")

#: Fatigue criteria are evaluated on this path (see module docstring).
GOVERNING_PATH = "shaft_reaction"

#: Torque-intensity sweep used for the life-vs-ice-load curve.
DEFAULT_SCALE_SWEEP: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5)

#: Mount tuning study grids (full-band attenuation feasibility).
TUNING_F_N_HZ: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
TUNING_ZETA: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50)


def _hull_case(name: str) -> HullCase:
    for case in hull_cases():
        if case.name.upper() == name.upper():
            return case
    raise ValueError(f"unknown hull case {name!r}; available: {[c.name for c in hull_cases()]}")


@dataclass
class ChainConfig:
    """Configuration of one run of the vibration -> fatigue chain."""

    hull_case_name: str = "KCS"
    n_modes: int = 16
    n_elements: int = 120
    zeta_structural: float = 0.02
    severity: str = "design"
    n_rps: float = 2.5
    duration_s: float = 120.0
    dt_s: float = 1.0e-3
    eta_hull: float = 0.30
    seed: int = 20261110
    #: equipment resilient mount (the mission's ``H(f)``)
    mount_f_n_hz: float = 4.0
    mount_zeta: float = 0.15
    equipment_mass_kg: float = 2200.0
    #: local deck / machinery-foundation mode (path A bracket, path B mass)
    foundation_f_n_hz: float = 25.0
    foundation_zeta: float = 0.02
    local_foundation_mass_kg: float = 8000.0
    pipe: PipeSpan = field(default_factory=PipeSpan)
    operating_fraction: float = 0.25
    fatigue_duration_s: float = 120.0
    fatigue_dt_s: float = 2.0e-3
    design_life_years: float = 25.0
    scale_sweep: Sequence[float] = DEFAULT_SCALE_SWEEP
    sn_curve_class: str = "D"
    sn_environment: str = "seawater_cp"
    yield_stress_mpa: float = 315.0

    def mount(self) -> MountIsolation:
        return MountIsolation(
            f_n_hz=float(self.mount_f_n_hz),
            zeta=float(self.mount_zeta),
            equipment_mass_kg=float(self.equipment_mass_kg),
            label="equipment mount H(f)",
        )

    def foundation_bracket(self) -> MountIsolation:
        return MountIsolation(
            f_n_hz=float(self.foundation_f_n_hz),
            zeta=float(self.foundation_zeta),
            label="deck / foundation bracket A_fnd(f)",
        )

    def sn_curve(self) -> DNCurve:
        return DNCurve(self.sn_curve_class, self.sn_environment)


@dataclass
class ChainResult:
    """Full payload of one chain run (JSON through ``to_json_dict``)."""

    config: ChainConfig
    basis: HullModalBasis
    mount: MountIsolation
    bracket: MountIsolation
    sn_curve: DNCurve
    severities: dict[str, dict[str, Any]]
    scale_sweep: list[dict[str, Any]]
    tuning_map: list[dict[str, Any]]
    disclosures: list[dict[str, Any]]
    criteria: list[dict[str, Any]]
    curves: dict[str, dict[str, Any]] = field(default_factory=dict)
    bode: dict[str, Any] = field(default_factory=dict)

    @property
    def all_pass(self) -> bool:
        return all(c["passed"] for c in self.criteria)

    def _sn_block(self) -> dict[str, Any]:
        return {
            "class": self.sn_curve.curve_class,
            "environment": self.sn_curve.environment,
            "m1": float(self.sn_curve.params.m1),
            "m2": float(self.sn_curve.params.m2),
            "log_K1": float(self.sn_curve.params.log_K1),
            "log_K2": float(self.sn_curve.params.log_K2),
            "n_transition": float(self.sn_curve.params.n_transition),
            "stress_at_1e7_cycles_MPa": float(self.sn_curve.stress_for_life(1.0e7)),
            "standard": "DNV-RP-C203 (D curve, seawater with cathodic protection)",
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "node": "p228w1-N08",
            "chain": (
                "ice milling torque -> hull wet modes -> equipment base acceleration "
                "-> pipe stress spectrum -> rainflow/S-N/Miner fatigue life"
            ),
            "governing_path": GOVERNING_PATH,
            "config": {
                "hull_case": self.config.hull_case_name,
                "n_modes": int(self.config.n_modes),
                "n_elements": int(self.config.n_elements),
                "zeta_structural": float(self.config.zeta_structural),
                "severity": self.config.severity,
                "n_rps": float(self.config.n_rps),
                "duration_s": float(self.config.duration_s),
                "dt_s": float(self.config.dt_s),
                "eta_hull": float(self.config.eta_hull),
                "seed": int(self.config.seed),
                "operating_fraction": float(self.config.operating_fraction),
                "fatigue_duration_s": float(self.config.fatigue_duration_s),
                "fatigue_dt_s": float(self.config.fatigue_dt_s),
                "design_life_years": float(self.config.design_life_years),
                "local_foundation_mass_kg": float(self.config.local_foundation_mass_kg),
                "scale_sweep": [float(s) for s in self.config.scale_sweep],
                "yield_stress_MPa": float(self.config.yield_stress_mpa),
            },
            "hull_modal": self.basis.to_json_dict(),
            "mount": self.mount.as_dict(),
            "foundation_bracket": self.bracket.as_dict(),
            "sn_curve": self._sn_block(),
            "pipe": self.config.pipe.as_dict(),
            "severities": self.severities,
            "scale_sweep": self.scale_sweep,
            "mount_tuning_map": self.tuning_map,
            "curves": self.curves,
            "bode": self.bode,
            "criteria": self.criteria,
            "disclosures": self.disclosures,
            "all_criteria_pass": self.all_pass,
        }

    def summary(self) -> str:
        c = {row["id"]: row for row in self.criteria}
        life = {
            k: v["paths"][GOVERNING_PATH]["pipe"]["fatigue"]["life_years"]
            for k, v in self.severities.items()
        }
        return (
            f"wet/dry max ratio {float(np.max(self.basis.frequency_ratio)):.3f} "
            f"(C1 {'PASS' if c['C1']['passed'] else 'FAIL'}); max |H| above sqrt(2)f_n "
            f"{c['C2a']['value']:.3f} (C2a {'PASS' if c['C2a']['passed'] else 'FAIL'}); base/support RMS "
            f"{c['C2b']['value']:.3f} (C2b {'PASS' if c['C2b']['passed'] else 'FAIL'}); pipe life "
            f"light/design/heavy = {life['light']:.3e}/{life['design']:.3e}/{life['heavy']:.3e} yr "
            f"(C3a {'PASS' if c['C3a']['passed'] else 'FAIL'})"
        )


def _criterion(
    cid: str, name: str, value: Any, threshold: Any, passed: bool, detail: str = "", units: str = ""
) -> dict[str, Any]:
    return {
        "id": cid,
        "name": name,
        "value": value,
        "threshold": threshold,
        "units": units,
        "passed": bool(passed),
        "detail": detail,
    }


def _band_with_variance(psd: np.ndarray, freqs: np.ndarray, fraction: float = 0.90) -> tuple[float, float]:
    total = float(np.trapezoid(psd, freqs))
    if total <= 0.0:
        return (float(freqs[0]), float(freqs[-1]))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (psd[1:] + psd[:-1]) * np.diff(freqs))]) / total
    tail = 0.5 * (1.0 - fraction)
    return (float(np.interp(tail, cum, freqs)), float(np.interp(1.0 - tail, cum, freqs)))


def _variance_share_below(psd: np.ndarray, freqs: np.ndarray, f_split: float) -> float:
    total = float(np.trapezoid(psd, freqs))
    if total <= 0.0:
        return 0.0
    m = freqs <= f_split
    return float(np.trapezoid(psd[m], freqs[m]) / total)


def _stage_block(
    freqs: np.ndarray,
    s_support: np.ndarray,
    s_reference: np.ndarray,
    mount: MountIsolation,
    span: PipeSpan,
    sn_curve: DNCurve,
    config: ChainConfig,
    fatigue_seed: int,
    reference_label: str,
    with_fatigue: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Mount -> base acceleration -> pipe stress -> fatigue for one support PSD."""
    f = freqs
    h_mag = mount.magnitude(f)
    s_base = s_support * h_mag**2
    f_iso = mount.isolation_frequency_hz
    band_att = (float(f_iso), float(f[-1]))
    att = f > f_iso * (1.0 + 1e-9)

    rms_sup_band = band_rms(f, s_support, *band_att)
    rms_base_band = band_rms(f, s_base, *band_att)
    rms_ref_band = band_rms(f, s_reference, *band_att)
    rms_sup_all = band_rms(f, s_support, float(f[0]), float(f[-1]))
    rms_base_all = band_rms(f, s_base, float(f[0]), float(f[-1]))
    rms_ref_all = band_rms(f, s_reference, float(f[0]), float(f[-1]))

    env = transmissibility_envelope(
        f,
        mount,
        band_att,
        checkpoints={
            "mount_natural": mount.f_n_hz,
            "isolation_frequency": mount.isolation_frequency_hz,
            "support_peak": float(f[int(np.argmax(s_support))]),
        },
    )
    scf_factor = float(
        span.scf * (span.wall_thickness_m / span.reference_thickness_m) ** span.thickness_exponent
    )
    t_sigma = span.stress_transfer(f)
    s_stress = np.abs(t_sigma) ** 2 * s_base * scf_factor**2
    fatigue = None
    amp_needed = float("nan")
    if with_fatigue:
        fatigue = fatigue_life(
            s_stress,
            f,
            span,
            sn_curve,
            duration_s=config.fatigue_duration_s,
            dt_s=config.fatigue_dt_s,
            operating_fraction=config.operating_fraction,
            seed=fatigue_seed,
        )
        d_year = float(fatigue["damage_per_year"])
        if d_year > 0.0:
            amp_needed = float(
                (1.0 / config.design_life_years / d_year) ** (1.0 / float(sn_curve.params.m2))
            )
    block = {
        "reference": reference_label,
        "support_acceleration": {
            "rms_all_m_s2": float(rms_sup_all),
            "rms_attenuation_band_m_s2": float(rms_sup_band),
            "peak_frequency_hz": float(f[int(np.argmax(s_support))]),
            "variance_band_hz": [float(v) for v in _band_with_variance(s_support, f, 0.90)],
        },
        "base_acceleration": {
            "rms_all_m_s2": float(rms_base_all),
            "rms_attenuation_band_m_s2": float(rms_base_band),
            "ratio_over_support_all": float(rms_base_all / rms_sup_all) if rms_sup_all > 0 else float("nan"),
            "ratio_over_support_band": float(rms_base_band / rms_sup_band) if rms_sup_band > 0 else float("nan"),
            "ratio_over_reference_all": float(rms_base_all / rms_ref_all) if rms_ref_all > 0 else float("nan"),
            "ratio_over_reference_band": float(rms_base_band / rms_ref_band) if rms_ref_band > 0 else float("nan"),
            "variance_share_below_isolation_frequency": _variance_share_below(s_base, f, f_iso),
        },
        "transmissibility": env,
        "max_transmissibility_above_isolation_hz": float(np.max(h_mag[att])) if np.any(att) else float("nan"),
        "pipe": {
            "stress_psd_peak_frequency_hz": float(f[int(np.argmax(s_stress))]),
            "stress_rms_MPa": float(np.sqrt(max(np.trapezoid(s_stress, f), 0.0)) / 1.0e6),
            "stress_transfer_static_Pa_per_m_s2": float(np.real(t_sigma[0])),
            "static_closed_form_Pa_per_m_s2": float(span.static_stress_per_acceleration),
            "static_relative_error": float(
                abs(abs(np.real(t_sigma[0])) - span.static_stress_per_acceleration)
                / span.static_stress_per_acceleration
            ),
            "first_span_frequency_hz": float(span.first_frequency_hz),
            "fatigue": fatigue,
            "stress_amplification_to_reach_design_life": amp_needed,
        },
    }
    arrays = {
        "s_support": s_support,
        "s_base": s_base,
        "s_stress": s_stress,
        "h_magnitude": h_mag,
    }
    return block, arrays


def _case_payload(
    basis: HullModalBasis,
    exc: IceTorqueExcitation,
    resp: ResponseSpectrum,
    mount: MountIsolation,
    bracket: MountIsolation,
    span: PipeSpan,
    sn_curve: DNCurve,
    config: ChainConfig,
    fatigue_seed: int,
    detailed: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Both load paths for one excitation case.

    ``detailed=False`` skips the (arithmetically irrelevant, ``~1e22`` yr) fatigue
    integration of the global path and the rigid-support reference, which are
    reported for the design ice class only.
    """
    f = resp.freqs_hz
    s_hull = resp.s_acceleration_m2_s4_per_hz

    # --- path A: hull girder -> foundation bracket -> mount -> base --------
    s_support_a = s_hull * bracket.magnitude(f) ** 2
    block_a, arrays_a = _stage_block(
        f,
        s_support_a,
        s_hull,
        mount,
        span,
        sn_curve,
        config,
        fatigue_seed,
        "hull girder acceleration",
        with_fatigue=detailed,
    )
    block_a["hull_girder_acceleration"] = {
        **resp.to_json_dict(),
        "variance_band_hz": [float(v) for v in _band_with_variance(s_hull, f, 0.90)],
        "excitation_energy_band_hz": [float(v) for v in exc.energy_band(0.90)],
    }
    # peak (time-domain) hull acceleration for the design verification: the
    # largest torque impact mapped through the largest modal gain
    block_a["hull_girder_acceleration"]["variance_share_below_isolation_frequency"] = _variance_share_below(
        s_hull, f, mount.isolation_frequency_hz
    )

    # --- path B: local shaft-line reaction -> foundation mass -> mount -----
    a_force = point_force_acceleration(
        f, config.local_foundation_mass_kg, config.foundation_f_n_hz, config.foundation_zeta
    )
    s_support_b = np.abs(a_force) ** 2 * exc.force_spectrum_scaled()
    block_b, arrays_b = _stage_block(
        f,
        s_support_b,
        s_support_b,
        mount,
        span,
        sn_curve,
        config,
        fatigue_seed,
        "local foundation acceleration",
    )
    # rigid-support reference of path B: same support motion, no resilient mount.
    # This quantifies what the mount buys (protection factor of the pipe detail).
    t_sigma = span.stress_transfer(f)
    scf_factor = float(
        span.scf * (span.wall_thickness_m / span.reference_thickness_m) ** span.thickness_exponent
    )
    s_stress_rigid = np.abs(t_sigma) ** 2 * s_support_b * scf_factor**2
    block_b["pipe"]["stress_rms_rigid_reference_MPa"] = float(
        np.sqrt(max(np.trapezoid(s_stress_rigid, f), 0.0)) / 1.0e6
    )
    if detailed:
        block_b["pipe"]["fatigue_rigid_reference"] = fatigue_life(
            s_stress_rigid,
            f,
            span,
            sn_curve,
            duration_s=config.fatigue_duration_s,
            dt_s=config.fatigue_dt_s,
            operating_fraction=config.operating_fraction,
            seed=fatigue_seed,
        )
    block_b["local_foundation"] = {
        "modal_mass_kg": float(config.local_foundation_mass_kg),
        "f_n_hz": float(config.foundation_f_n_hz),
        "zeta": float(config.foundation_zeta),
        "rigid_mass_limit_m_s2_per_N": float(1.0 / config.local_foundation_mass_kg),
        "force_rms_N": float(exc.force_rms_n),
        "force_peak_N": float(np.max(np.abs(exc.force_n))),
    }

    payload = {
        "excitation": exc.as_dict(),
        "paths": {"hull_girder": block_a, "shaft_reaction": block_b},
    }
    arrays = {
        "hull_girder": {**arrays_a, "s_hull": s_hull},
        "shaft_reaction": arrays_b,
    }
    return payload, arrays


def _tuning_map(
    freqs: np.ndarray,
    a_hull: np.ndarray,
    bracket: MountIsolation,
    f_grid: Sequence[float] = TUNING_F_N_HZ,
    zeta_grid: Sequence[float] = TUNING_ZETA,
) -> list[dict[str, Any]]:
    """Full-band attenuation feasibility of the mount (frequency domain only)."""
    a_fnd = bracket.magnitude(freqs)
    s_support = a_hull * a_fnd**2
    rms_sup = float(np.sqrt(max(np.trapezoid(s_support, freqs), 0.0)))
    rms_hull = float(np.sqrt(max(np.trapezoid(a_hull, freqs), 0.0)))
    rows = []
    for f_n in f_grid:
        for zeta in zeta_grid:
            m = MountIsolation(f_n_hz=float(f_n), zeta=float(zeta))
            h = m.magnitude(freqs)
            rms_base = float(np.sqrt(max(np.trapezoid(s_support * h**2, freqs), 0.0)))
            rows.append(
                {
                    "f_n_hz": float(f_n),
                    "zeta": float(zeta),
                    "rms_ratio_base_over_hull": float(rms_base / rms_hull) if rms_hull > 0 else float("nan"),
                    "rms_ratio_base_over_support": float(rms_base / rms_sup) if rms_sup > 0 else float("nan"),
                    "peak_amplification": m.peak_amplification,
                    "static_deflection_m": m.static_deflection_m,
                    "isolation_frequency_hz": m.isolation_frequency_hz,
                }
            )
    return rows


def replace_scaled(exc: IceTorqueExcitation, scale: float) -> IceTorqueExcitation:
    """Return the same excitation with a new load-intensity factor ``scale``."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    ratio = float(scale) / float(exc.scale)
    return replace(
        exc, scale=float(scale), force_n=exc.force_n * ratio, force_rms_n=exc.force_rms_n * ratio
    )


def _critical_lambda(sweep: Sequence[dict[str, Any]], design_life_years: float, sn_curve: DNCurve) -> float:
    """Torque-intensity multiplier that brings the governing damage to ``1/design_life``."""
    ref = [row for row in sweep if abs(float(row["scale"]) - 1.0) < 1e-12]
    if not ref or float(ref[0]["damage_per_year"]) <= 0.0:
        return float("inf")
    target = 1.0 / float(design_life_years)
    return float((target / float(ref[0]["damage_per_year"])) ** (1.0 / float(sn_curve.params.m2)))


def run_chain(config: ChainConfig | None = None) -> ChainResult:
    """Run the full chain for the three ice severities plus the intensity sweep."""
    cfg = config or ChainConfig()
    case = _hull_case(cfg.hull_case_name)
    basis = wet_dry_modal_basis(
        case, n_modes=cfg.n_modes, n_elements=cfg.n_elements, zeta_structural=cfg.zeta_structural
    )
    mount = cfg.mount()
    bracket = cfg.foundation_bracket()
    sn_curve = cfg.sn_curve()

    severities: dict[str, dict[str, Any]] = {}
    excitations: dict[str, IceTorqueExcitation] = {}
    curve_arrays: dict[str, dict[str, np.ndarray]] = {}
    for severity in DEFAULT_SEVERITIES:
        exc = milling_torque_excitation(
            severity=severity,
            n_rps=cfg.n_rps,
            duration_s=cfg.duration_s,
            dt_s=cfg.dt_s,
            eta_hull=cfg.eta_hull,
            scale=1.0,
            seed=cfg.seed,
        )
        resp = acceleration_psd(basis, exc.freqs_hz, exc.force_spectrum_scaled())
        payload, arrays = _case_payload(
            basis,
            exc,
            resp,
            mount,
            bracket,
            cfg.pipe,
            sn_curve,
            cfg,
            cfg.seed + 11,
            detailed=(severity == cfg.severity),
        )
        severities[severity] = payload
        excitations[severity] = exc
        curve_arrays[severity] = arrays

    # ---- torque-intensity sweep on the design ice class --------------------
    base_exc = excitations[cfg.severity]
    sweep: list[dict[str, Any]] = []
    for lam in cfg.scale_sweep:
        exc = replace_scaled(base_exc, float(lam))
        resp = acceleration_psd(basis, exc.freqs_hz, exc.force_spectrum_scaled())
        # identical seed at every lambda: the excitation is a scaled copy of one
        # torque train, so every rainflow range scales exactly with lambda
        payload, _arrays = _case_payload(
            basis, exc, resp, mount, bracket, cfg.pipe, sn_curve, cfg, cfg.seed + 11
        )
        gov = payload["paths"][GOVERNING_PATH]
        hull = payload["paths"]["hull_girder"]
        sweep.append(
            {
                "scale": float(lam),
                "force_rms_N": float(exc.force_rms_n),
                "hull_girder_rms_m_s2": float(hull["support_acceleration"]["rms_all_m_s2"]),
                "base_rms_m_s2": float(gov["base_acceleration"]["rms_all_m_s2"]),
                "pipe_stress_rms_MPa": float(gov["pipe"]["stress_rms_MPa"]),
                "damage_per_year": float(gov["pipe"]["fatigue"]["damage_per_year"]),
                "life_years": float(gov["pipe"]["fatigue"]["life_years"]),
                "equivalent_stress_range_1e7_MPa": gov["pipe"]["fatigue"]["equivalent_stress_range_1e7_MPa"],
                "hull_girder_pipe_life_years": float(
                    hull["pipe"]["fatigue"]["life_years"]
                ),
            }
        )

    base = severities[cfg.severity]
    path_a = base["paths"]["hull_girder"]
    path_b = base["paths"][GOVERNING_PATH]
    ratios = basis.frequency_ratio
    life = {k: float(v["paths"][GOVERNING_PATH]["pipe"]["fatigue"]["life_years"]) for k, v in severities.items()}
    life_a = {
        k: float(v["paths"]["hull_girder"]["pipe"]["fatigue"]["life_years"])
        for k, v in severities.items()
        if v["paths"]["hull_girder"]["pipe"].get("fatigue")
    }
    criteria = [
        _criterion(
            "G1",
            "geometry gate (Cb / L / B / displacement vs published reference)",
            {
                "block_coefficient": float(basis.geometry.block_coefficient),
                "displacement_t": float(basis.geometry.displacement_t),
                "displacement_reference_t": basis.geometry.displacement_reference_t,
                "displacement_rel_error": basis.geometry.displacement_rel_error,
                "L_over_B": float(basis.geometry.l_over_b),
                "B_over_T": float(basis.geometry.b_over_t),
            },
            "Cb within 3 %, displacement within 9 %, plausible L/B and B/T",
            basis.geometry.passed,
            detail=(
                f"Cb={basis.geometry.block_coefficient:.4f}; displacement {basis.geometry.displacement_t:.0f} t "
                f"vs reference {basis.geometry.displacement_reference_t or float('nan'):.0f} t; rel.err "
                f"{(basis.geometry.displacement_rel_error or 0.0):.4%}; L/B={basis.geometry.l_over_b:.2f}"
            ),
        ),
        _criterion(
            "C1",
            "wet natural frequencies below dry (all elastic modes)",
            [float(v) for v in ratios],
            "max(f_wet/f_dry) < 1",
            bool(np.all(ratios < 1.0)),
            units="-",
            detail=(
                f"max ratio {float(np.max(ratios)):.4f} (mode {int(np.argmax(ratios)) + 1}); mode-1 added-mass "
                f"ratio {float(basis.added_mass_ratio[0]):.3f}; f1 dry/wet = "
                f"{float(basis.dry_frequencies_hz[0]):.4f}/{float(basis.wet_frequencies_hz[0]):.4f} Hz"
            ),
        ),
        _criterion(
            "C2a",
            "|H(f)| < 1 on the whole mount attenuation band (f > sqrt(2) f_n)",
            float(path_a["max_transmissibility_above_isolation_hz"]),
            "< 1.0",
            bool(path_a["max_transmissibility_above_isolation_hz"] < 1.0),
            units="-",
            detail=(
                f"band ({mount.isolation_frequency_hz:.3f}, {path_a['transmissibility']['band_hz'][1]:.1f}] Hz; "
                f"|H| = 1 at sqrt(2) f_n by construction, amplification below it = "
                f"{path_a['transmissibility']['amplification_band_max']:.3f} "
                f"(exact peak {mount.peak_amplification:.3f} at f/f_n = "
                f"{np.sqrt((np.sqrt(1.0 + 8.0 * mount.zeta**2) - 1.0) / (4.0 * mount.zeta**2)):.3f})"
            ),
        ),
        _criterion(
            "C2b",
            "RMS(base) < RMS(support) over the attenuation band (path A)",
            float(path_a["base_acceleration"]["ratio_over_support_band"]),
            "< 1.0",
            bool(path_a["base_acceleration"]["ratio_over_support_band"] < 1.0),
            units="-",
            detail=(
                f"support {path_a['support_acceleration']['rms_attenuation_band_m_s2']:.4e} m/s^2 -> base "
                f"{path_a['base_acceleration']['rms_attenuation_band_m_s2']:.4e} m/s^2, with the foundation "
                f"bracket amplification {bracket.peak_amplification:.2f} already included"
            ),
        ),
        _criterion(
            "C2c",
            "RMS(base) < RMS(hull girder) over the attenuation band (path A)",
            float(path_a["base_acceleration"]["ratio_over_reference_band"]),
            "< 1.0",
            bool(path_a["base_acceleration"]["ratio_over_reference_band"] < 1.0),
            units="-",
            detail=(
                f"hull girder {path_a['hull_girder_acceleration']['rms_m_s2']:.4e} m/s^2 (full band) -> base "
                f"{path_a['base_acceleration']['rms_attenuation_band_m_s2']:.4e} m/s^2 on the attenuation band"
            ),
        ),
        _criterion(
            "C3a",
            "fatigue life decreases with ice severity (governing path B)",
            life,
            "life(light) > life(design) > life(heavy)",
            bool(life["light"] > life["design"] > life["heavy"]),
            units="years",
            detail=(
                "damage/yr "
                + ", ".join(
                    f"{k}={v['paths'][GOVERNING_PATH]['pipe']['fatigue']['damage_per_year']:.3e}"
                    for k, v in severities.items()
                )
                + "; global hull-girder path (A) lives: "
                + ", ".join(f"{k}={life_a[k]:.3e}" for k in life_a)
            ),
        ),
        _criterion(
            "C3b",
            "fatigue life strictly decreasing along the torque-intensity sweep",
            [float(row["life_years"]) for row in sweep],
            "strictly decreasing in lambda",
            bool(all(sweep[i + 1]["life_years"] < sweep[i]["life_years"] for i in range(len(sweep) - 1))),
            units="years",
            detail=f"lambda = {[float(r['scale']) for r in sweep]}",
        ),
        _criterion(
            "C4",
            "rainflow damage vs narrowband Rayleigh closed form",
            float(path_b["pipe"]["fatigue"]["narrowband_crosscheck"]["rainflow_to_narrowband_ratio"]),
            "0.2 <= ratio <= 1.1 (rainflow <= narrowband for a broadband process)",
            bool(0.2 <= path_b["pipe"]["fatigue"]["narrowband_crosscheck"]["rainflow_to_narrowband_ratio"] <= 1.1),
            units="-",
            detail=(
                f"rainflow {path_b['pipe']['fatigue']['damage_per_year']:.3e} /yr vs narrowband "
                f"{path_b['pipe']['fatigue']['narrowband_crosscheck']['damage_per_year']:.3e} /yr"
            ),
        ),
        _criterion(
            "C5",
            "pipe modal stress transfer reproduces the static m*L^2/(8Z) limit",
            float(path_b["pipe"]["static_relative_error"]),
            "< 1 %",
            bool(path_b["pipe"]["static_relative_error"] < 0.01),
            units="-",
            detail=(
                f"modal {abs(path_b['pipe']['stress_transfer_static_Pa_per_m_s2']):.4e} Pa/(m/s^2) vs closed form "
                f"{path_b['pipe']['static_closed_form_Pa_per_m_s2']:.4e} Pa/(m/s^2)"
            ),
        ),
        _criterion(
            "C6",
            "static strength sanity of the pipe detail (vibration fatigue governs)",
            float(path_b["pipe"]["fatigue"]["max_stress_range_MPa"]),
            f"< {2.0 / 3.0 * cfg.yield_stress_mpa:.1f} MPa (2/3 of yield)",
            bool(path_b["pipe"]["fatigue"]["max_stress_range_MPa"] < 2.0 / 3.0 * cfg.yield_stress_mpa),
            units="MPa",
            detail="max rainflow stress range of the design ice case",
        ),
    ]

    # tuning map on the design-case hull PSD (frequency domain only)
    resp_base = acceleration_psd(basis, base_exc.freqs_hz, base_exc.force_spectrum_scaled())
    tuning = _tuning_map(resp_base.freqs_hz, resp_base.s_acceleration_m2_s4_per_hz, bracket)

    disclosures = [
        {
            "id": "D1",
            "statement": (
                "Full-band (0.2-500 Hz) RMS(base)/RMS(hull girder) is NOT below one for path A: a base-excited "
                "mount has H(0) = 1 by construction and the hull-girder ice response is concentrated below the "
                "isolation frequency. Reported rather than hidden behind a narrow band."
            ),
            "value": float(path_a["base_acceleration"]["ratio_over_reference_all"]),
            "units": "-",
        },
        {
            "id": "D2",
            "statement": "share of the base acceleration variance below sqrt(2) f_n (path A)",
            "value": float(path_a["base_acceleration"]["variance_share_below_isolation_frequency"]),
            "units": "-",
        },
        {
            "id": "D3",
            "statement": "stress amplification still required for the pipe detail to reach the design life (path B)",
            "value": float(path_b["pipe"]["stress_amplification_to_reach_design_life"]),
            "units": "-",
        },
        {
            "id": "D4",
            "statement": (
                "S-N provenance: repository DNCurve('D','seawater_cp') log10 K1 = "
                f"{sn_curve.params.log_K1:.3f}, m = {sn_curve.params.m1:.0f} up to N = "
                f"{sn_curve.params.n_transition:.0e} then m = {sn_curve.params.m2:.0f}, i.e. "
                f"{sn_curve.stress_for_life(1.0e7):.1f} MPa at 1e7 cycles; the DNV-RP-C203 seawater-with-CP D "
                "curve differs below 1e6 cycles (repo curve ~2x less conservative). SCF "
                f"{cfg.pipe.scf:.2f} and thickness correction (t/t_ref)^{cfg.pipe.thickness_exponent:.2f} are "
                "applied on top."
            ),
            "value": float(sn_curve.stress_for_life(1.0e7)),
            "units": "MPa at 1e7 cycles",
        },
        {
            "id": "D5",
            "statement": (
                "declared model parameters: shaft-line -> hull transmission efficiency eta_hull = "
                f"{cfg.eta_hull:.2f}; torque fluctuation only (no ice-hull impact pressure path, no thrust "
                "fluctuation); single vertical bending axis; radiation damping not solved (structural damping "
                "2 %); foundation bracket = one base-excited SDOF at "
                f"{bracket.f_n_hz:.0f} Hz (zeta {bracket.zeta:.2f}); local foundation mass "
                f"{cfg.local_foundation_mass_kg:.0f} kg at {cfg.foundation_f_n_hz:.0f} Hz (path B)."
            ),
            "value": float(cfg.eta_hull),
            "units": "-",
        },
        {
            "id": "D6",
            "statement": (
                "torque-intensity multiplier lambda that would bring the governing pipe damage to the design "
                "life (exact power law D ~ lambda^m2 because the S-N ranges stay in the high-cycle branch)"
            ),
            "value": _critical_lambda(sweep, cfg.design_life_years, sn_curve),
            "units": "-",
        },
        {
            "id": "D8",
            "statement": (
                "what the resilient mount buys: fatigue life of the same pipe detail on a rigid support "
                "(no mount, path B support motion) and the life ratio mount/rigid."
            ),
            "value": float(
                path_b["pipe"]["fatigue_rigid_reference"]["life_years"]
                / path_b["pipe"]["fatigue"]["life_years"]
            ),
            "units": "-",
        },
        {
            "id": "D7",
            "statement": (
                "path comparison: the global hull-girder load path (A) is far weaker than the local shaft-line "
                "reaction path (B). Ratio of the pipe fatigue lives (A/B) at the design ice class, and the "
                "hull-girder acceleration level that the torque reaction produces."
            ),
            "value": float(
                severities[cfg.severity]["paths"]["hull_girder"]["pipe"]["fatigue"]["life_years"]
                / severities[cfg.severity]["paths"][GOVERNING_PATH]["pipe"]["fatigue"]["life_years"]
            ),
            "units": "-",
        },
    ]

    # ---- response curves (log-spaced, JSON friendly) -----------------------
    curves: dict[str, dict[str, list[float]]] = {}
    for severity, arrays in curve_arrays.items():
        f = excitations[severity].freqs_hz
        mask = (f >= 0.2) & (f <= 250.0)
        idx = np.unique(np.geomspace(1, np.count_nonzero(mask), 320).astype(int) - 1)
        ff = f[mask][idx]
        hull = arrays["hull_girder"]
        shaft = arrays["shaft_reaction"]
        curves[severity] = {
            "freqs_hz": [float(v) for v in ff],
            "s_hull_acc": [float(hull["s_hull"][mask][idx][i]) for i in range(ff.size)],
            "s_base_acc_hull_path": [float(hull["s_base"][mask][idx][i]) for i in range(ff.size)],
            "s_base_acc_shaft_path": [float(shaft["s_base"][mask][idx][i]) for i in range(ff.size)],
            "s_pipe_stress_hull_path": [float(hull["s_stress"][mask][idx][i]) for i in range(ff.size)],
            "s_pipe_stress_shaft_path": [float(shaft["s_stress"][mask][idx][i]) for i in range(ff.size)],
            "h_magnitude": [float(shaft["h_magnitude"][mask][idx][i]) for i in range(ff.size)],
        }

    bode = mount.bode(np.logspace(np.log10(0.2), np.log10(250.0), 240))
    return ChainResult(
        config=cfg,
        basis=basis,
        mount=mount,
        bracket=bracket,
        sn_curve=sn_curve,
        severities=severities,
        scale_sweep=sweep,
        tuning_map=tuning,
        disclosures=disclosures,
        criteria=criteria,
        curves=curves,
        bode={
            "freqs_hz": [float(v) for v in bode["freqs_hz"]],
            "magnitude": [float(v) for v in bode["magnitude"]],
            "magnitude_db": [float(v) for v in bode["magnitude_db"]],
            "phase_deg": [float(v) for v in bode["phase_deg"]],
            "f_n_hz": float(mount.f_n_hz),
            "zeta": float(mount.zeta),
            "isolation_frequency_hz": mount.isolation_frequency_hz,
            "peak_amplification": mount.peak_amplification,
        },
    )