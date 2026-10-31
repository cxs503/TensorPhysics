"""End-to-end case: cavitation inception, ice effect and radiated noise spectrum.

The case is the reduced model of one propeller blade passage:

* geometry -- quasi-1-D converging-diverging passage of area ratio 2
  (throat/inlet 0.5), inlet velocity ``u_in``, 120 finite-volume stations;
* ice -- dilute particle phase at volumetric concentration 0/5/10/15 %;
* cavitation -- Schnerr-Sauer mass transfer with vapour-fraction transport;
* acoustics -- compact monopole (Lighthill/FW-H) fed by the Rayleigh-Plesset
  response of the nucleus population to the blade-passage pressure history.

Everything numeric that ends up in a report is produced here and written to
JSON; nothing is transcribed by hand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .acoustics import (
    P_REF,
    amplitude_spectrum,
    blade_pass_frequency,
    compact_source_far_field,
    peak_frequency,
    rms_of_band,
    sound_pressure_level,
)
from .cavitation import (
    C_L,
    CavitationParams,
    P_V,
    RHO_L,
    bubble_radius_forced,
    radius_from_alpha,
    rayleigh_collapse_time,
)
from .flow1d import Quasi1DNozzle, analytic_inception_sigma
from .particles import IcePhaseParams, drag_force_per_particle, pressure_fluctuation_rms
from .steady1d import SteadyNozzleFlow


@dataclass
class CaseConfig:
    """Configuration of the W3-N06 validation case."""

    u_in: float = 2.0            # passage inlet velocity [m/s]
    n_cells: int = 120
    area_ratio: float = 2.0
    ice_concentrations: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15)
    rpm: float = 1200.0          # model propeller shaft speed [rpm]
    n_blades: int = 4
    distance: float = 1.0        # acoustic observer distance [m]
    sigma_test: float = 3.0      # operating point for the volume trend
    inception_margin: float = 0.03   # noise operating point below sigma_i
    pressure_amplitude_fraction: float = 0.10   # of the throat dynamic head
    kappa: float = 1.0           # pressure-fluctuation coefficient
    bubble_dt: float = 2.0e-6
    analysis_band_hz: float = 2000.0
    n_cycles: int = 8


@dataclass
class CaseResult:
    config: dict
    geometry: dict
    baseline: dict
    ice: list[dict]
    noise: list[dict]
    acoustic_verification: dict
    criteria: dict
    figures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _nozzle(config: CaseConfig) -> Quasi1DNozzle:
    area_in = 1.0e-2
    return Quasi1DNozzle(
        n_cells=config.n_cells,
        area_in=area_in,
        area_throat=area_in / config.area_ratio,
    )


def geometry_report(nozzle: Quasi1DNozzle) -> dict:
    """Geometry pre-check: area profile must be monotone in/out and consistent."""
    x = np.linspace(0.0, nozzle.length, 2001)
    area = nozzle.area(x)
    i_throat = int(np.argmin(area))
    contraction = area[: i_throat + 1]
    diffuser = area[i_throat:]
    return {
        "length_m": nozzle.length,
        "area_in_m2": nozzle.area_in,
        "area_throat_m2": nozzle.area_throat,
        "area_ratio": float(nozzle.area_in / np.min(area)),
        "area_ratio_nominal": float(nozzle.area_ratio),
        "throat_position_fraction": float(x[i_throat] / nozzle.length),
        "contraction_monotone": bool(np.all(np.diff(contraction) <= 1e-15)),
        "diffuser_monotone": bool(np.all(np.diff(diffuser) >= -1e-15)),
        "min_area_equals_throat": bool(abs(np.min(area) - nozzle.area_throat) < 1e-12),
        "volume_m3": float(np.trapezoid(area, x)),
    }


def _reference_susceptible_volume(
    config: CaseConfig,
    nozzle: Quasi1DNozzle,
    ice: IcePhaseParams,
    cav: CavitationParams,
    base_result,
    concentration: float,
) -> float:
    """Independent frozen-field estimate of the cavitation-susceptible volume.

    Uses only the *clear-water* fields produced by a separate run: the ice
    particle velocity is obtained by marching the particle momentum equation
    ``u_p du_p/dx = (u - u_p)/tau(slip)`` through the clear-water velocity
    field (an independent ODE, no coupling to the liquid march), the pressure
    deficit caused by the particle drag is accumulated as
    ``dp(x) = int_0^x n_p F_drag dx'``, and the susceptible volume is where
    ``p_0(x) - dp(x) <= p_v``.
    """
    x = base_result.x
    area = base_result.area
    u = base_result.u
    p = base_result.p
    d_p = ice.d_p
    n_p = concentration / (np.pi * d_p ** 3 / 6.0)

    if concentration <= 0.0:
        below = p <= cav.p_v
        return float(np.trapezoid(np.where(below, area, 0.0), x))

    def response_time(slip: float) -> float:
        stokes = d_p ** 2 * (ice.rho_ice + 0.5 * ice.rho_l) / (18.0 * ice.mu_l)
        if abs(slip) < 1.0e-9:
            return stokes
        force = abs(drag_force_per_particle(slip, d_p, ice.rho_l, ice.mu_l))
        mass = ice.rho_ice * np.pi * d_p ** 3 / 6.0
        return max(mass * abs(slip) / max(force, 1e-30), 1.0e-9)

    u_p = np.empty_like(x)
    u_p[0] = u[0]
    for i in range(x.size - 1):
        dx = x[i + 1] - x[i]
        local = u_p[i]
        slip = u[i] - local
        f1 = slip / (max(abs(local), 1.0e-6) * response_time(slip))
        mid = local + 0.5 * dx * f1
        slip2 = u[i] - mid
        f2 = slip2 / (max(abs(mid), 1.0e-6) * response_time(slip2))
        u_p[i + 1] = max(local + dx * f2, 1.0e-6)

    slip = u - u_p
    force = np.array([drag_force_per_particle(s_, d_p, ice.rho_l, ice.mu_l) for s_ in slip])
    acc = n_p * force
    dp_drag = np.concatenate(
        [[0.0], np.cumsum(0.5 * (acc[1:] + acc[:-1]) * np.diff(x))]
    )
    below = (p - dp_drag) <= cav.p_v
    return float(np.trapezoid(np.where(below, area, 0.0), x))


def _pressure_drive(
    t: np.ndarray,
    p_mean: float,
    amplitude: float,
    f_blade: float,
    ice_pressure_rms: float,
    seed: int = 0,
) -> np.ndarray:
    """Blade-passage pressure history seen by a nucleus.

    Tonal part: the passage pressure oscillation (``amplitude``, at the
    blade-passing frequency).  Broadband part: the ice-induced pressure
    fluctuation, represented as a sum of narrow-band components around the
    particle wake-shedding frequency with a reproducible random phase.
    """
    tonal = amplitude * np.sin(2.0 * np.pi * f_blade * t)
    if ice_pressure_rms <= 0.0:
        return p_mean + tonal
    rng = np.random.default_rng(seed)
    f_shed = 200.0  # Hz, St * |w| / d_p for the modelled slip and particle size
    signal = np.zeros_like(t)
    n_modes = 12
    for _ in range(n_modes):
        fi = f_shed * rng.uniform(0.6, 1.6)
        signal += rng.uniform(0.5, 1.5) * np.sin(2.0 * np.pi * fi * t + rng.uniform(0, 2 * np.pi))
    signal /= max(np.std(signal), 1.0e-30)
    return p_mean + tonal + ice_pressure_rms * signal


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run_case(config: CaseConfig | None = None):
    """Run the full case; returns ``(CaseResult, spectra, series)``."""
    cfg = config or CaseConfig()
    cav = CavitationParams()
    ice = IcePhaseParams(kappa=cfg.kappa)
    nozzle = _nozzle(cfg)
    f_blade = blade_pass_frequency(cfg.rpm, cfg.n_blades)

    geo = geometry_report(nozzle)
    sigma_analytic = analytic_inception_sigma(geo["area_ratio"])

    # --- clear-water baseline -------------------------------------------
    base_flow = SteadyNozzleFlow(
        nozzle=nozzle, u_in=cfg.u_in, ice_concentration=0.0, cav=cav, ice=ice
    )
    bracket = (0.90 * sigma_analytic, 1.10 * sigma_analytic)
    base_scan = base_flow.inception_scan(bracket, n_scan=6)
    base_field = base_flow.march(sigma_of_inlet=cfg.sigma_test, freeze_vapour=True)
    dynamic_head = 0.5 * cav.rho_l * cfg.u_in ** 2
    # Inception criterion used for the acceptance gate: the single-phase pressure
    # minimum reaches the vapour pressure, i.e. sigma_i = (p_in - p_min)/q.
    sigma_i_pressure = (base_field.p_in - base_field.p_min) / dynamic_head
    baseline = {
        "sigma_inception_analytic": sigma_analytic,
        "sigma_inception_pressure_criterion": float(sigma_i_pressure),
        "sigma_inception_relative_error": float(abs(sigma_i_pressure / sigma_analytic - 1.0)),
        "sigma_inception_rate_model": float(base_scan["sigma"]),
        "sigma_inception_rate_model_records": base_scan["records"],
        "p_min_at_sigma_test": float(base_field.p_min),
        "p_throat_at_sigma_test": float(base_field.p_throat),
        "p_in_at_sigma_test": float(base_field.p_in),
        "dynamic_head_pa": float(dynamic_head),
        "u_throat": float(np.max(base_field.u)),
    }

    # --- ice concentrations ---------------------------------------------
    ice_rows: list[dict] = []
    series: dict[float, dict] = {}
    for concentration in cfg.ice_concentrations:
        flow = SteadyNozzleFlow(
            nozzle=nozzle, u_in=cfg.u_in, ice_concentration=concentration, cav=cav, ice=ice
        )
        field = flow.march(sigma_of_inlet=cfg.sigma_test, freeze_vapour=True)
        susceptible = flow.suscept_volume(cfg.sigma_test)
        reference_volume = _reference_susceptible_volume(
            cfg, nozzle, ice, cav, base_field, concentration
        )
        deficit = float(field.p_in - field.p_min)
        p_rms_max = float(field.pressure_fluctuation_rms_max)
        sigma_pressure = deficit / dynamic_head
        sigma_fluctuation = (deficit + cfg.kappa * p_rms_max) / dynamic_head
        rate_model = flow.inception_scan(bracket, n_scan=6)

        row = {
            "ice_concentration": concentration,
            "sigma_inception_pressure": float(sigma_pressure),
            "sigma_inception_fluctuation": float(sigma_fluctuation),
            "sigma_inception_rate_model": float(rate_model["sigma"]),
            "sigma_inception_rate_model_records": rate_model["records"],
            "p_in_at_sigma_test": float(field.p_in),
            "p_min_at_sigma_test": float(field.p_min),
            "p_throat_at_sigma_test": float(field.p_throat),
            "pressure_deficit_pa": deficit,
            "susceptible_volume_model": float(susceptible["volume"]),
            "susceptible_volume_reference": float(reference_volume),
            "susceptible_length": float(susceptible["length"]),
            "max_ice_slip": float(np.max(np.abs(field.u - field.u_particle))),
            "max_ice_concentration_local": float(np.max(field.concentration)),
            "max_pressure_fluctuation_rms": p_rms_max,
        }
        ice_rows.append(row)
        series[concentration] = {"field": field, "row": row, "inception": rate_model}

    # --- radiated noise per concentration -------------------------------
    noise_rows: list[dict] = []
    spectra: dict[float, dict] = {}
    R0 = float(radius_from_alpha(cav.alpha_nuc, cav.n0))
    for concentration in cfg.ice_concentrations:
        entry = series[concentration]
        row = entry["row"]
        field = entry["field"]
        sigma_op = row["sigma_inception_fluctuation"] * (1.0 - cfg.inception_margin)
        op = SteadyNozzleFlow(
            nozzle=nozzle, u_in=cfg.u_in, ice_concentration=concentration, cav=cav, ice=ice
        ).march(sigma_of_inlet=sigma_op, freeze_vapour=True)

        u_throat = float(np.max(op.u))
        amplitude = cfg.pressure_amplitude_fraction * 0.5 * cav.rho_l * u_throat ** 2
        p_ice_rms = float(row["max_pressure_fluctuation_rms"]) if concentration > 0 else 0.0
        duration = cfg.n_cycles / f_blade
        t = np.arange(0.0, duration, cfg.bubble_dt)
        p_drive = _pressure_drive(
            t, float(op.p_throat), amplitude, f_blade, p_ice_rms, seed=int(1000 * concentration)
        )

        radius = bubble_radius_forced(t, p_drive, R0)
        zone_volume = float(series[concentration]["field"].area.sum() * (nozzle.length / cfg.n_cells))
        bubble_count = cav.n0 * zone_volume
        vapour_volume = bubble_count * 4.0 / 3.0 * np.pi * radius ** 3

        far = compact_source_far_field(
            t,
            vapour_volume,
            distance=cfg.distance,
            rho=cav.rho_l,
            f_max=cfg.analysis_band_hz,
        )
        freq, amp = amplitude_spectrum(t, far["p"])
        peak = peak_frequency(t, far["p"], f_min=10.0, f_max=1000.0)
        band_lo, band_hi = f_blade * 1.5, f_blade * 6.0
        broadband = rms_of_band(freq, amp, band_lo, band_hi)
        tone = rms_of_band(freq, amp, 0.9 * f_blade, 1.1 * f_blade)

        noise_rows.append(
            {
                "ice_concentration": concentration,
                "sigma_operating": float(op.sigma),
                "p_throat_mean": float(op.p_throat),
                "drive_amplitude": float(amplitude),
                "ice_pressure_rms": p_ice_rms,
                "radius_min": float(radius.min()),
                "radius_max": float(radius.max()),
                "radius_ratio": float(radius.max() / radius.min()),
                "far_field_p_rms": float(far["p_rms"]),
                "far_field_spl_db": sound_pressure_level(far["p_rms"]),
                "peak_frequency": float(peak["f_peak"]),
                "blade_pass_frequency": float(f_blade),
                "tone_relative_error": float(abs(peak["f_peak"] / f_blade - 1.0)),
                "tone_rms": float(tone),
                "broadband_rms": float(broadband),
                "broadband_db": sound_pressure_level(broadband),
                "spectral_df": float(peak["df"]),
            }
        )
        spectra[concentration] = {
            "freq": freq,
            "amp": amp,
            "t": t,
            "p": far["p"],
            "vapour_volume": vapour_volume,
            "drive": p_drive,
        }

    # --- acoustic-solver verification against the analytic monopole ------
    acoustic = _verify_acoustic_solver(cfg)

    criteria = evaluate_criteria(
        config=cfg,
        baseline=baseline,
        ice_rows=ice_rows,
        noise_rows=noise_rows,
        acoustic=acoustic,
        f_blade=f_blade,
    )

    return CaseResult(
        config=asdict(cfg),
        geometry=geo,
        baseline=baseline,
        ice=ice_rows,
        noise=noise_rows,
        acoustic_verification=acoustic,
        criteria=criteria,
    ), spectra, series


def _verify_acoustic_solver(config: CaseConfig) -> dict:
    """Validate the compact-monopole pipeline against the analytic solution.

    A pulsating sphere of radius ``a`` with surface velocity
    ``u_s = U sin(2 pi f t)`` radiates (``k a << 1``)

        p(r, t) = rho a^2 / r * du_s/dt (t - r/c)

    which is exact linear acoustics and independent of this code.
    """
    a = 2.0e-3
    freq = 200.0
    u_amp = 0.05
    t = np.arange(0.0, 40.0 / freq, 2.0e-6)
    r_ref = 1.0
    volume = 4.0 / 3.0 * np.pi * (a + u_amp / (2 * np.pi * freq) * np.sin(2 * np.pi * freq * t)) ** 3
    far = compact_source_far_field(t, volume, distance=r_ref)
    analytic_amp = RHO_L * a ** 2 * u_amp * 2.0 * np.pi * freq / r_ref
    freq_out, amp_out = amplitude_spectrum(t, far["p"])
    peak = peak_frequency(t, far["p"], f_min=10.0, f_max=2000.0)
    # geometric spreading: 2 r and 4 r observers
    far2 = compact_source_far_field(t, volume, distance=2.0 * r_ref)
    far4 = compact_source_far_field(t, volume, distance=4.0 * r_ref)
    return {
        "surface_radius": a,
        "frequency": freq,
        "analytic_amplitude": float(analytic_amp),
        "simulated_amplitude": float(peak["amplitude"]),
        "amplitude_relative_error": float(abs(peak["amplitude"] / analytic_amp - 1.0)),
        "peak_frequency": float(peak["f_peak"]),
        "frequency_relative_error": float(abs(peak["f_peak"] / freq - 1.0)),
        "p_rms_1r": float(far["p_rms"]),
        "p_rms_2r": float(far2["p_rms"]),
        "p_rms_4r": float(far4["p_rms"]),
        "inverse_square_ratio_2r": float(far["p_rms"] / far2["p_rms"]),
        "inverse_square_ratio_4r": float(far["p_rms"] / far4["p_rms"]),
    }


def evaluate_criteria(
    *,
    config: CaseConfig,
    baseline: dict,
    ice_rows: list[dict],
    noise_rows: list[dict],
    acoustic: dict,
    f_blade: float,
) -> dict:
    """Apply the machine-checkable acceptance criteria of the campaign node."""
    from scipy.stats import spearmanr

    # (a) inception number versus the clear-water (Bernoulli) baseline
    err_a = baseline["sigma_inception_relative_error"]

    # (b) susceptible-volume trend with ice concentration versus the independent
    #     frozen-field reference estimate
    v_model = [row["susceptible_volume_model"] for row in ice_rows]
    v_ref = [row["susceptible_volume_reference"] for row in ice_rows]
    n_distinct = len(set(np.round(v_model, 12)))
    if n_distinct > 2:
        rho_b = float(spearmanr(v_model, v_ref).statistic)
    else:
        rho_b = float("nan")

    # (c) far-field tonal frequency versus the blade-passing frequency anchor
    tone_errors = [row["tone_relative_error"] for row in noise_rows]
    err_c = float(max(tone_errors)) if tone_errors else float("nan")

    return {
        "a_inception_vs_clear_water": {
            "value": float(err_a),
            "limit": 0.05,
            "passed": bool(err_a < 0.05),
            "sigma_inception_analytic": baseline["sigma_inception_analytic"],
            "sigma_inception_model": baseline["sigma_inception_pressure_criterion"],
            "sigma_inception_rate_model": baseline.get("sigma_inception_rate_model"),
        },
        "b_volume_trend_spearman": {
            "value": rho_b,
            "limit": 0.9,
            "passed": bool(np.isfinite(rho_b) and rho_b > 0.9),
            "volumes_model": v_model,
            "volumes_reference": v_ref,
            "distinct_model_values": int(n_distinct),
            "ice_concentrations": [row["ice_concentration"] for row in ice_rows],
        },
        "c_tone_frequency": {
            "value": err_c,
            "limit": 0.10,
            "passed": bool(err_c < 0.10),
            "blade_pass_frequency": float(f_blade),
            "peak_frequencies": [row["peak_frequency"] for row in noise_rows],
        },
        "acoustic_solver_verification": {
            "amplitude_error": acoustic["amplitude_relative_error"],
            "frequency_error": acoustic["frequency_relative_error"],
            "inverse_square_2r": acoustic["inverse_square_ratio_2r"],
            "inverse_square_4r": acoustic["inverse_square_ratio_4r"],
        },
    }


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------
def write_artifacts(result, spectra, series, outdir: str | Path) -> dict:
    """Write JSON artifacts and figures; return the produced paths."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    produced: list[str] = []

    summary = {
        "config": result.config,
        "geometry": result.geometry,
        "baseline": result.baseline,
        "ice": result.ice,
        "noise": result.noise,
        "acoustic_verification": result.acoustic_verification,
        "criteria": result.criteria,
    }
    summary_path = out / "case_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    produced.append(str(summary_path))

    profiles = {
        "x": {},
        "rows": {},
    }
    for concentration, entry in series.items():
        field = entry["field"]
        profiles["rows"][str(concentration)] = {
            "x": field.x.tolist(),
            "area": field.area.tolist(),
            "p": field.p.tolist(),
            "u": field.u.tolist(),
            "u_particle": field.u_particle.tolist(),
            "concentration_local": field.concentration.tolist(),
        }
    prof_path = out / "profiles.json"
    prof_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    produced.append(str(prof_path))

    noise_path = out / "noise_spectra.json"
    noise_payload = {
        str(c): {
            "freq": spec["freq"][::8].tolist(),
            "amplitude": spec["amp"][::8].tolist(),
            "vapour_volume": spec["vapour_volume"][::200].tolist(),
            "drive": spec["drive"][::200].tolist(),
            "decimation_freq": 8,
            "decimation_time": 200,
            "time_step": float(np.mean(np.diff(spec["t"]))),
        }
        for c, spec in spectra.items()
    }
    noise_path.write_text(json.dumps(noise_payload), encoding="utf-8")
    produced.append(str(noise_path))

    produced.extend(_make_figures(result, spectra, series, out))
    return {"files": produced}


def _make_figures(result, spectra, series, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    produced: list[str] = []
    concentrations = sorted(spectra)

    # 1. pressure and ice-slip profiles
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c in concentrations:
        field = series[c]["field"]
        axes[0].plot(field.x, field.p / 1.0e3, label=f"c_ice={c:.0%}")
        axes[1].plot(
            field.x, field.u - field.u_particle, label=f"c_ice={c:.0%}"
        )
    axes[0].axhline(2.339, color="k", ls="--", lw=1, label=r"$p_v$")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("p [kPa]")
    axes[0].set_title("Single-phase pressure field (sigma = %.2f)" % result.config["sigma_test"])
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("liquid - ice slip [m/s]")
    axes[1].set_title("Ice slip velocity")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = out / "fig_profiles.png"; fig.savefig(path, dpi=140); plt.close(fig)
    produced.append(str(path))

    # 2. inception number vs ice concentration
    fig, ax = plt.subplots(figsize=(6, 4))
    cs = [row["ice_concentration"] for row in result.ice]
    ax.plot(cs, [row["sigma_inception_pressure"] for row in result.ice], "o-",
            label=r"$p_{min}=p_v$ criterion (single-phase field)")
    ax.plot(cs, [row["sigma_inception_fluctuation"] for row in result.ice], "s--",
            label=r"$p_{min}-p'_{rms}=p_v$ (ice fluctuation)")
    ax.axhline(result.baseline["sigma_inception_analytic"], color="k", ls=":",
               label="Bernoulli baseline (A_in/A_t)^2-1")
    ax.set_xlabel("ice volumetric concentration [-]"); ax.set_ylabel(r"$\sigma_i$")
    ax.set_title("Cavitation inception number vs ice concentration")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "fig_inception.png"; fig.savefig(path, dpi=140); plt.close(fig)
    produced.append(str(path))

    # 3. susceptible volume trend
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cs, [row["susceptible_volume_model"] for row in result.ice], "o-",
            label="two-fluid march")
    ax.plot(cs, [row["susceptible_volume_reference"] for row in result.ice], "s--",
            label="frozen-field estimate")
    ax.set_xlabel("ice volumetric concentration [-]")
    ax.set_ylabel(r"susceptible volume [m$^3$]")
    ax.set_title("Cavitation-susceptible volume (sigma = %.2f)" % result.config["sigma_test"])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "fig_susceptible_volume.png"; fig.savefig(path, dpi=140); plt.close(fig)
    produced.append(str(path))

    # 4. far-field spectra
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in concentrations:
        freq, amp = spectra[c]["freq"], spectra[c]["amp"]
        mask = (freq > 10) & (freq < 1000)
        ax.semilogy(freq[mask], amp[mask], lw=1.0, label=f"c_ice={c:.0%}")
    f_blade = result.noise[0]["blade_pass_frequency"]
    ax.axvline(f_blade, color="k", ls=":", lw=1, label=f"BPF={f_blade:.0f} Hz")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"far-field pressure amplitude [Pa] @ 1 m")
    ax.set_title("Radiated noise spectra")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out / "fig_noise_spectra.png"; fig.savefig(path, dpi=140); plt.close(fig)
    produced.append(str(path))

    # 5. bubble radius history and volume acceleration at one concentration
    c_show = concentrations[-1]
    spec = spectra[c_show]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n = min(4000, spec["t"].size)
    axes[0].plot(spec["t"][:n] * 1e3, spec["drive"][:n] / 1e3)
    axes[0].set_xlabel("t [ms]"); axes[0].set_ylabel("drive p [kPa]")
    axes[0].set_title(f"Nucleus driving pressure (c_ice={c_show:.0%})")
    axes[0].grid(alpha=0.3)
    axes[1].plot(spec["t"][:n] * 1e3, spec["vapour_volume"][:n] * 1e9)
    axes[1].set_xlabel("t [ms]")
    axes[1].set_ylabel(r"vapour volume [mm$^3$]")
    axes[1].set_title("Cavity volume history")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = out / "fig_bubble_history.png"; fig.savefig(path, dpi=140); plt.close(fig)
    produced.append(str(path))
    return produced