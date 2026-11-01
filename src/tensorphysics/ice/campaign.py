"""End-to-end driver: acceptance-criteria bundle for ice-cover bow impact.

Produces, for the mission acceptance criteria (a)-(f):

(a) characteristic wavelength / frequency of the ice-coupled flexural waves
    measured in the time-domain solver, checked against the analytic
    plate-fluid dispersion (target <= 3 %);
(b) steady deflection under a load patch against the Hetenyi / Hankel
    closed-form solution (target <= 5 %), with a periodic-domain truncation
    study;
(c) bow-impact load peak: analytic closed-form ramming anchor, impulse-momentum
    check and comparison of the peak contact pressure with the empirical
    pressure-area envelope used by the crushing law (consistency);
(d) flexural fracture criterion: crack initiation location/time, radial (star)
    versus ring (circumferential) crack pattern and ring radius / l_c;
(e) energy ledger: ship kinetic energy = ice strain + buoyancy + ice kinetic +
    fluid kinetic + crushing + damping + fracture energy (residual <= 5 %);
(f) pytest suite + this driver's figures and animation.

Everything is written to ``artifacts/m1138`` (JSON + PNG + GIF).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .contact import CrushingLaw, WedgeBow, ram_rigid_closed_form
from .coupled import CoupledPlateFluid, IceWaterParameters
from .dispersion import FlexuralDispersion, GRAVITY, RHO_WATER
from .fracture import (
    bottom_stress_field,
    crack_initiation,
    fracture_energy,
    max_tensile_stress,
    ring_crack_radius,
    top_stress_field,
)
from .static_plate import (
    hankel_gauss_deflection,
    hetenyi_center_deflection,
    hetenyi_point_load,
    solve_biharmonic_fft,
)

__all__ = [
    "ImpactConfig",
    "run_dispersion_case",
    "run_static_case",
    "measure_static_wavelength",
    "run_impact_case",
    "run_campaign",
]


# --------------------------------------------------------------------------- (a)
def run_dispersion_case(
    E: float = 9.0e9,
    h: float = 1.0,
    nu: float = 0.3,
    depth: float | None = 50.0,
    lx: float = 4096.0,
    nx: int = 1024,
    ly: float = 16.0,
    ny: int = 8,
    amplitude: float = 0.05,
    packet_width: float = 350.0,
    plane_duration: float = 60.0,
    packet_duration: float = 12.0,
    dt: float = 0.01,
) -> dict:
    """Measure the flexural-gravity dispersion of the *numerical* system.

    Two independent runs on the same grid:

    * a monochromatic plane wave (an exact grid mode) -- gives the angular
      frequency from the zero-crossing train and the wavelength from the spatial
      spectrum; both are compared with the analytic plate-fluid dispersion;
    * a wave packet -- gives the group velocity from the drift of the energy
      envelope centroid, compared with the analytic ``d omega / d k``.

    Period elongation of the Newmark scheme is ``O((omega dt)^2)``, i.e. ~1e-5
    here, so the measured errors are expected to be dominated by the signal
    processing (zero crossings, spectral peak), not by the integrator.
    """
    disp = FlexuralDispersion.from_geometry(E, h, nu, depth=depth)
    params = IceWaterParameters(E=E, h=h, nu=nu, depth=depth, damping=0.0)
    lc = disp.characteristic_length()
    sim = CoupledPlateFluid(params, nx=nx, ny=ny, lx=lx, ly=ly)
    x, _ = sim.grid()
    m_mode = max(int(round((1.0 / lc) * lx / (2.0 * np.pi))), 3)
    k0 = 2.0 * np.pi * m_mode / lx
    omega_exact = float(disp.omega(k0))

    # ---- (i) plane wave: frequency from zero crossings -----------------------
    sim.set_deflection(amplitude * np.cos(k0 * x))
    e0 = sim.energies()["total_J"]
    n_steps = int(round(plane_duration / dt))
    probe, times = [], []
    for _ in range(n_steps):
        sim.step(dt, np.zeros_like(sim.W))
        times.append(sim.t)
        probe.append(sim.value_at(0.5 * lx, 0.5 * ly, "w"))
    probe = np.asarray(probe)
    times = np.asarray(times)
    crossings = np.where(np.diff(np.signbit(probe)))[0]
    half_periods = times[crossings[1:]] - times[crossings[:-1]]
    half_periods = half_periods[half_periods > 2 * dt]
    omega_meas = float(np.pi / np.mean(half_periods))
    e1 = sim.energies()["total_J"]

    # wavelength from the spatial spectrum of the plane wave
    snap = sim.field("w")[ny // 2]
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=sim.dx)
    kspec = np.abs(np.fft.rfft(snap * np.hanning(nx)))
    k_meas = float(kx[int(np.argmax(kspec[1:]) + 1)])

    # ---- (ii) wave packet: group velocity ------------------------------------
    sim2 = CoupledPlateFluid(params, nx=nx, ny=ny, lx=lx, ly=ly)
    x0 = 0.25 * lx
    shape = amplitude * np.exp(-0.5 * ((x - x0) / packet_width) ** 2) * np.cos(k0 * x)
    # exact right-going initial condition of the *discrete* system: the velocity
    # spectrum must satisfy v_hat = -i sgn(kx) omega(k) w_hat, otherwise the packet splits
    # into counter-propagating halves and the energy centroid does not translate
    sim2.set_deflection(shape)
    x_axis = x[ny // 2]
    omega_k = np.sqrt(np.where(np.isfinite(sim2.K / sim2.mass), sim2.K / sim2.mass, 0.0))
    sim2.V = -1j * np.sign(sim2.KX) * omega_k * sim2.W
    sim2.V[0, 0] = 0.0
    sim2.A = (-sim2.K * sim2.W - sim2.c * sim2.V) / sim2.mass
    sim2.A[0, 0] = 0.0
    n_steps2 = int(round(packet_duration / dt))
    cent, tt = [], []
    for _ in range(n_steps2):
        sim2.step(dt, np.zeros_like(sim2.W))
        w = sim2.field("w")[ny // 2]
        energy = w**2
        cent.append(float(np.sum(x_axis * energy) / max(np.sum(energy), 1e-30)))
        tt.append(sim2.t)
    cent = np.asarray(cent)
    tt = np.asarray(tt)
    cg_meas = float(np.polyfit(tt, cent, 1)[0])
    cg_exact = float(disp.group_speed(k0))
    # a dispersive packet's energy centroid travels at the energy-weighted mean
    # group velocity of its spectrum, not at c_g(k0)
    Wp = np.abs(np.fft.fft2(shape)) ** 2
    Wp[0, 0] = 0.0
    cg_k = np.where(np.isfinite(disp.group_speed(sim2.k_mag)), disp.group_speed(sim2.k_mag), 0.0)
    cg_packet = float(np.sum(Wp * cg_k) / max(np.sum(Wp), 1e-30))

    k_from_omega = float(disp.k_from_omega(omega_meas))
    lambda_meas = 2.0 * np.pi / k_meas
    lambda_disp = 2.0 * np.pi / k_from_omega
    statics = measure_static_wavelength()
    return {
        "rigidity_Nm": disp.rigidity,
        "characteristic_length_m": lc,
        "static_wavelength_m": 2.0 * np.pi * lc,
        "k0_1_m": k0,
        "omega_exact_rad_s": omega_exact,
        "omega_measured_rad_s": omega_meas,
        "omega_error_pct": abs(omega_meas - omega_exact) / omega_exact * 100.0,
        "k_measured_1_m": k_meas,
        "k_from_measured_omega_1_m": k_from_omega,
        "wavelength_measured_m": lambda_meas,
        "wavelength_analytic_m": float(2.0 * np.pi / k0),
        "wavelength_from_dispersion_m": lambda_disp,
        "wavelength_error_pct": abs(lambda_disp - lambda_meas) / lambda_meas * 100.0,
        "phase_speed_measured_m_s": omega_meas / k_meas,
        "phase_speed_analytic_m_s": float(disp.phase_speed(k0)),
        "group_speed_measured_m_s": cg_meas,
        "group_speed_analytic_m_s": cg_exact,
        "group_speed_of_packet_analytic_m_s": cg_packet,
        "group_speed_error_pct": abs(cg_meas - cg_packet) / cg_packet * 100.0,
        "group_speed_error_vs_k0_pct": abs(cg_meas - cg_exact) / cg_exact * 100.0,
        "energy_drift_rel": float(abs(e1 - e0) / max(abs(e0), 1e-30)),
        "grid": {"nx": nx, "ny": ny, "lx": lx, "dx": sim.dx, "dt": dt},
        "probe_times_s": times.tolist(),
        "probe_deflection_m": probe.tolist(),
        "minimum_phase_speed_m_s": disp.minimum_phase_speed()[0],
        "static_pattern": statics,
    }


# --------------------------------------------------------------------------- (b)
def run_static_case(
    E: float = 9.0e9,
    h: float = 1.5,
    nu: float = 0.3,
    total_force: float = 4.36e6,
    patch_sigma: float = 4.0,
    domains=(256.0, 384.0, 512.0, 768.0, 1024.0),
    dx: float = 2.0,
) -> dict:
    """Hetenyi / Hankel verification of the steady plate deflection."""
    disp = FlexuralDispersion.from_geometry(E, h, nu, depth=50.0)
    lc = disp.characteristic_length()
    D = disp.rigidity
    radii = np.array([0.0, 0.25 * lc, 0.5 * lc, lc, 1.5 * lc, 2.0 * lc, 3.0 * lc])
    ref = np.atleast_1d(hankel_gauss_deflection(radii, total_force, patch_sigma, D))
    truncation = []
    for L in domains:
        nx = int(round(L / dx))
        x = np.arange(nx) * dx - 0.5 * L
        X, Y = np.meshgrid(x, x, indexing="xy")
        load = total_force / (2.0 * np.pi * patch_sigma**2) * np.exp(
            -(X**2 + Y**2) / (2.0 * patch_sigma**2))
        w = solve_biharmonic_fft(load, dx, D, RHO_WATER, GRAVITY)
        iy, ix = np.unravel_index(np.argmin(np.hypot(X, Y)), X.shape)
        row = w[iy, ix:]
        r_axis = x[ix:]
        samp = np.interp(radii, r_axis, row)
        err = float(np.max(np.abs(samp - ref)) / abs(ref[0]) * 100.0)
        truncation.append({
            "domain_m": float(L),
            "domain_over_lc": float(L / lc),
            "nx": nx,
            "w_centre_m": float(w[iy, ix]),
            "w_centre_error_pct": float(abs(w[iy, ix] - ref[0]) / abs(ref[0]) * 100.0),
            "max_profile_error_pct": err,
        })
    best = min(truncation, key=lambda t: t["max_profile_error_pct"])
    return {
        "rigidity_Nm": D,
        "characteristic_length_m": lc,
        "total_force_N": total_force,
        "patch_sigma_m": patch_sigma,
        "w_centre_analytic_m": float(ref[0]),
        "w_centre_hetenyi_pointload_m": float(hetenyi_point_load(total_force, D, 0.0)),
        "w_centre_deep_foundation_m": float(hetenyi_center_deflection(total_force, D)),
        "radii_m": radii.tolist(),
        "w_reference_m": ref.tolist(),
        "truncation_study": truncation,
        "best_domain_m": best["domain_m"],
        "w_centre_error_pct": best["w_centre_error_pct"],
        "max_profile_error_pct": best["max_profile_error_pct"],
    }


def measure_static_wavelength(
    total_force: float = 4.36e6,
    E: float = 9.0e9,
    h: float = 1.5,
    nu: float = 0.3,
    patch_sigma: float = 4.0,
    L: float = 768.0,
    dx: float = 1.0,
) -> dict:
    """Characteristic static wavelength from the zeros of the deflection profile."""
    disp = FlexuralDispersion.from_geometry(E, h, nu, depth=50.0)
    D = disp.rigidity
    nx = int(round(L / dx))
    x = np.arange(nx) * dx - 0.5 * L
    X, Y = np.meshgrid(x, x, indexing="xy")
    load = total_force / (2.0 * np.pi * patch_sigma**2) * np.exp(
        -(X**2 + Y**2) / (2.0 * patch_sigma**2))
    w = solve_biharmonic_fft(load, dx, D, RHO_WATER, GRAVITY)
    iy, ix = np.unravel_index(np.argmin(np.hypot(X, Y)), X.shape)
    r = x[ix:]
    row = w[iy, ix:]
    keep = r > 5.0 * patch_sigma
    r, row = r[keep], row[keep]
    sign_change = np.where(np.diff(np.signbit(row)))[0]
    zeros = r[sign_change]
    lam_an = float(2.0 * np.pi * disp.characteristic_length())
    from scipy.optimize import brentq
    from scipy.special import kei

    lc = disp.characteristic_length()
    r_kei_zero = float(brentq(lambda u: float(kei(u)), 0.5, 6.0, xtol=1e-12))
    out = {
        "first_zero_measured_m": float(zeros[0]) if len(zeros) else float("nan"),
        "first_zero_hetenyi_m": r_kei_zero * lc,
        "first_zero_error_pct": float(abs(zeros[0] - r_kei_zero * lc) / (r_kei_zero * lc) * 100.0)
        if len(zeros) else float("nan"),
        "wavelength_analytic_m": lam_an,
        "wavelength_measured_m": float(2.0 * np.mean(np.diff(zeros))) if len(zeros) > 2 else float("nan"),
    }
    out["wavelength_error_pct"] = (
        abs(out["wavelength_measured_m"] - lam_an) / lam_an * 100.0
        if out["wavelength_measured_m"] == out["wavelength_measured_m"] else float("nan"))
    return out


# --------------------------------------------------------------------------- (c)-(e)
@dataclass
class ImpactConfig:
    """Configuration of a bow-impact case."""

    params: IceWaterParameters = field(
        default_factory=lambda: IceWaterParameters(
            E=9.0e9, h=1.5, nu=0.3, depth=50.0, damping=2.0e3)
    )
    ship_mass: float = 8.0e6        # [kg]
    impact_speed: float = 1.5       # [m/s]
    half_angle_deg: float = 22.0
    max_contact_width: float = 12.0
    ice_thickness_contact: float = 1.5
    crushing_coefficient: float = 2.6e6
    flexural_strength: float | None = 0.7e6
    nx: int = 512
    ny: int = 512
    lx: float = 384.0
    ly: float = 384.0
    dt: float = 0.01
    max_time: float = 12.0
    stop_speed_ratio: float = 0.02
    patch_cells: float = 1.5
    stress_every: int = 20
    speed_override: float | None = None
    moving_load: bool = False
    label: str = "impact"
    keep_frames: bool = True

    def __post_init__(self) -> None:
        if self.impact_speed <= 0.0 or self.ship_mass <= 0.0:
            raise ValueError("ship_mass and impact_speed must be positive")


def run_impact_case(cfg: ImpactConfig) -> dict:
    """Coupled impact simulation with an energy ledger and a fracture check."""
    p = cfg.params
    sim = CoupledPlateFluid(p, nx=cfg.nx, ny=cfg.ny, lx=cfg.lx, ly=cfg.ly)
    x, y = sim.grid()
    disp = FlexuralDispersion.from_geometry(p.E, p.h, p.nu, depth=p.depth)
    lc = disp.characteristic_length()
    bow = WedgeBow(cfg.half_angle_deg, cfg.max_contact_width)
    law = CrushingLaw(coefficient=cfg.crushing_coefficient)

    xc, yc = 0.5 * cfg.lx, 0.5 * cfg.ly
    iy0 = int(round(yc / sim.dy)) % cfg.ny
    ix0 = int(round(xc / sim.dx)) % cfg.nx

    v0 = cfg.speed_override if cfg.speed_override else cfg.impact_speed
    v_ship = v0
    xi = 0.0
    delta = 1e-6
    plate_energy0 = sim.energies()["total_J"]

    work_plate = diss = crush_energy = 0.0
    ke0 = 0.5 * cfg.ship_mass * v0**2
    keys = ("time_s", "speed_m_s", "penetration_m", "force_N", "area_m2",
            "pressure_Pa", "patch_sigma_m", "plate_energy_J", "work_on_plate_J",
            "dissipation_J", "crushing_energy_J", "ship_ke_J", "w_contact_m",
            "sigma1_max_Pa", "fluid_ke_J", "bending_J")
    hist: dict[str, list[float]] = {k: [] for k in keys}
    frames: list[dict] = []
    crack = None
    v_plate_prev = 0.0
    power_prev = 0.0
    force_prev = 0.0
    delta_prev = 0.0
    step = 0
    t = 0.0
    der = None
    while t < cfg.max_time and v_ship > cfg.stop_speed_ratio * v0:
        area = float(bow.contact_area(delta, cfg.ice_thickness_contact))
        force = float(law.force(area))
        sigma_c = max(math.sqrt(area / (2.0 * np.pi)), cfg.patch_cells * sim.dx)
        x_contact = xc + (xi if cfg.moving_load else 0.0)
        sim.step(cfg.dt, sim.patch_load_hat(force, x_contact, yc, sigma_c))

        icx = int(round(x_contact / sim.dx)) % cfg.nx
        w_c = sim.value_at(x_contact, yc, "w")
        v_c = sim.value_at(x_contact, yc, "v")

        work_plate += 0.5 * (force_prev + force) * 0.5 * (v_plate_prev + v_c) * cfg.dt
        power = sim.dissipation_power()
        diss += 0.5 * (power_prev + power) * cfg.dt
        crush_energy += 0.5 * (force_prev + force) * max(delta - delta_prev, 0.0)

        v_ship = max(v_ship - force / cfg.ship_mass * cfg.dt, 0.0)
        xi += v_ship * cfg.dt
        delta_prev = delta
        delta = max(xi - w_c, 1e-9)
        t += cfg.dt
        step += 1

        sigma1 = None
        if step % cfg.stress_every == 0:
            der = sim.derivatives("w")
            sigma1 = max_tensile_stress(bottom_stress_field(der, p.E, p.h, p.nu))
            si = crack_initiation(sigma1, x, y, cfg.flexural_strength or 1e30,
                                  contact_xy=(x_contact, yc),
                                  characteristic_length=lc, time=t)
            sigma1_max = si["peak_tensile_stress_Pa"]
        else:
            si = None
            sigma1_max = hist["sigma1_max_Pa"][-1] if hist["sigma1_max_Pa"] else 0.0

        en = sim.energies()
        values = (t, v_ship, delta, force, area, float(law.pressure(area)), sigma_c,
                  en["total_J"] - plate_energy0, work_plate, diss, crush_energy,
                  0.5 * cfg.ship_mass * v_ship**2, w_c, sigma1_max,
                  en["fluid_kinetic_J"], en["bending_J"])
        for k, val in zip(keys, values):
            hist[k].append(float(val))

        if cfg.keep_frames and sigma1 is not None and len(frames) < 40:
            frames.append({"t": t, "w": sim.field("w").copy(), "sigma1": sigma1.copy()})

        if si is not None and si["triggered"] and crack is None and cfg.flexural_strength:
            crack = dict(si)
            crack["force_at_crack_N"] = force
            crack["penetration_at_crack_m"] = delta
            top = top_stress_field(der, p.E, p.h, p.nu)
            s1_top = max_tensile_stress(top)
            ring = ring_crack_radius(s1_top, x, y, contact_xy=(x_contact, yc),
                                     inner_radius=3.0 * sigma_c,
                                     characteristic_length=lc)
            crack["ring"] = ring
            crack["top_stress_max_Pa"] = float(np.max(s1_top))
            crack["fields"] = {"w": sim.field("w"), "sigma1_bottom": sigma1.copy(),
                               "sigma1_top": s1_top.copy()}
            if cfg.flexural_strength is None:
                crack["triggered"] = False
            break

        force_prev, v_plate_prev, power_prev = force, v_c, power
        if force == 0.0:
            break

    hist_arr = {k: np.asarray(v, dtype=float) for k, v in hist.items()}
    ke_end = float(hist_arr["ship_ke_J"][-1]) if hist_arr["ship_ke_J"].size else ke0
    plate_end = float(hist_arr["plate_energy_J"][-1]) if hist_arr["plate_energy_J"].size else 0.0
    diss_end = float(hist_arr["dissipation_J"][-1]) if hist_arr["dissipation_J"].size else 0.0
    crush_end = float(hist_arr["crushing_energy_J"][-1]) if hist_arr["crushing_energy_J"].size else 0.0
    work_end = float(hist_arr["work_on_plate_J"][-1]) if hist_arr["work_on_plate_J"].size else 0.0
    impulse = float(np.trapezoid(hist_arr["force_N"], hist_arr["time_s"])) \
        if hist_arr["time_s"].size > 1 else 0.0
    momentum_change = cfg.ship_mass * (v0 - float(hist_arr["speed_m_s"][-1])) \
        if hist_arr["speed_m_s"].size else 0.0

    frac_energy = 0.0
    if crack is not None and crack.get("triggered"):
        frac_energy = fracture_energy(
            crack_length=max(crack["ring"]["ring_radius_m"], 1.0),
            ice_thickness=p.h, young_modulus=p.E)["energy_J"]

    ledger = {
        "ship_kinetic_initial_J": float(ke0),
        "ship_kinetic_final_J": ke_end,
        "ship_kinetic_released_J": float(ke0 - ke_end),
        "plate_energy_final_J": plate_end,
        "plate_bending_J": float(hist_arr["bending_J"][-1]) if hist_arr["bending_J"].size else 0.0,
        "plate_fluid_kinetic_J": float(hist_arr["fluid_ke_J"][-1]) if hist_arr["fluid_ke_J"].size else 0.0,
        "dissipation_J": diss_end,
        "crushing_energy_J": crush_end,
        "fracture_energy_J": float(frac_energy),
        "plate_received_work_J": work_end,
        "impulse_Ns": impulse,
        "momentum_change_Ns": float(momentum_change),
    }
    ledger["residual_plate_side_pct"] = float(
        abs(work_end - plate_end - diss_end) / max(abs(work_end), 1e-12) * 100.0)
    ledger["residual_total_pct"] = float(
        abs((ke0 - ke_end) - plate_end - diss_end - crush_end - frac_energy)
        / max(abs(ke0 - ke_end), 1e-12) * 100.0)
    ledger["impulse_error_pct"] = float(
        abs(impulse - momentum_change) / max(abs(momentum_change), 1e-12) * 100.0)

    closed = ram_rigid_closed_form(cfg.ship_mass, v0, bow,
                                   cfg.ice_thickness_contact, law)
    peak_force = float(np.max(hist_arr["force_N"])) if hist_arr["force_N"].size else 0.0
    peak_area = float(np.max(hist_arr["area_m2"])) if hist_arr["area_m2"].size else 1e-9
    result = {
        "label": cfg.label,
        "config": {
            "params": p.as_dict(),
            "ship_mass_kg": cfg.ship_mass,
            "impact_speed_m_s": v0,
            "half_angle_deg": cfg.half_angle_deg,
            "ice_thickness_contact_m": cfg.ice_thickness_contact,
            "crushing_coefficient_Pa_m0.5": cfg.crushing_coefficient,
            "flexural_strength_Pa": cfg.flexural_strength,
            "grid": {"nx": cfg.nx, "ny": cfg.ny, "lx": cfg.lx, "dt": cfg.dt},
            "moving_load": cfg.moving_load,
        },
        "characteristic_length_m": lc,
        "history": {k: v.tolist() for k, v in hist_arr.items()},
        "ledger": ledger,
        "peak_force_N": peak_force,
        "peak_area_m2": peak_area,
        "peak_pressure_Pa": float(law.pressure(peak_area)),
        "peak_contact_pressure_Pa": float(
            hist_arr["force_N"][int(np.argmax(hist_arr["force_N"]))]
            / (2.0 * np.pi * hist_arr["patch_sigma_m"][int(np.argmax(hist_arr["force_N"]))] ** 2))
        if hist_arr["force_N"].size else 0.0,
        "rigid_closed_form": closed,
        "peak_force_vs_closed_form_pct": float(
            abs(peak_force - closed["peak_force_N"]) / closed["peak_force_N"] * 100.0),
        "fracture": {k: v for k, v in crack.items() if k != "fields"} if crack else None,
        "duration_s": float(hist_arr["time_s"][-1]) if hist_arr["time_s"].size else 0.0,
        "steps": int(step),
    }
    final = {
        "x": x, "y": y,
        "w": sim.field("w"),
        "fluid_pressure": sim.fluid_pressure(),
        "sigma1": None if crack is None else crack["fields"]["sigma1_bottom"],
        "sigma1_top": None if crack is None else crack["fields"]["sigma1_top"],
    }
    return {"result": result, "frames": frames, "final": final}


# --------------------------------------------------------------------------- (f)
def run_moving_load_case(
    speed: float = 40.0,
    force: float = 2.0e6,
    E: float = 9.0e9,
    h: float = 1.0,
    nu: float = 0.3,
    depth: float | None = 50.0,
    damping: float = 2.0e3,
    nx: int = 384,
    lx: float = 768.0,
    dt: float = 0.01,
    duration: float = 6.0,
    patch_sigma: float = 3.0,
) -> dict:
    """Constant-force load moving across an ice cover (wave-radiation pattern).

    With ``speed`` above the minimum phase speed of the ice-coupled wave system
    the load is *supersonic*: a stationary flexural wave train builds up behind
    the contact plus a radiating Mach-like front.  This case is used for the
    deflection/underwater-pressure field figures; it also gives the ratio
    ``V / c_min`` quoted in the report.
    """
    params = IceWaterParameters(E=E, h=h, nu=nu, depth=depth, damping=damping)
    sim = CoupledPlateFluid(params, nx=nx, ny=nx, lx=lx, ly=lx)
    disp = FlexuralDispersion.from_geometry(E, h, nu, depth=depth)
    c_min, k_min = disp.minimum_phase_speed()
    x0, y0 = 0.5 * lx, 0.5 * lx
    n_steps = int(round(duration / dt))
    for n in range(n_steps):
        xc = x0 + speed * sim.t
        sim.step(dt, sim.patch_load_hat(force, xc, y0, patch_sigma))
        if xc > lx - 4.0 * patch_sigma:
            break
    return {
        "speed_m_s": speed,
        "force_N": force,
        "minimum_phase_speed_m_s": c_min,
        "wavenumber_at_minimum_phase_speed_1_m": k_min,
        "supersonic_ratio": speed / c_min,
        "duration_s": sim.t,
        "characteristic_length_m": disp.characteristic_length(),
        "energies_J": sim.energies(),
        "fields": {"w": sim.field("w"), "fluid_pressure": sim.fluid_pressure(),
                   "x": sim.grid()[0], "y": sim.grid()[1]},
    }


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def make_figures(bundle: dict, outdir: Path) -> list[dict]:
    """Render the report figures; returns metadata (name, caption, path)."""
    plt = _mpl()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figs: list[dict] = []

    # ---- fig 1: dispersion ------------------------------------------------
    disp = bundle["dispersion"]
    d = FlexuralDispersion.from_geometry(9.0e9, 1.0, 0.3, depth=50.0)
    k = np.logspace(-3.0, 0.6, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    ax.loglog(k, d.omega(k), "k-", label="analytic $\\omega(k)$ (plate+fluid)")
    ax.loglog(k, d.phase_speed(k), "b--", lw=1, label="$c_p=\\omega/k$")
    ax.plot(disp["k_measured_1_m"], disp["omega_measured_rad_s"], "ro", ms=6,
            label=f"numerical, err {disp['omega_error_pct']:.3f}%")
    ax.set_xlabel("wavenumber $k$ [1/m]")
    ax.set_ylabel("$\\omega$ [rad/s], $c_p$ [m/s]")
    ax.set_title("(a) flexural-gravity dispersion")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    tt = np.asarray(disp["probe_times_s"])
    pr = np.asarray(disp["probe_deflection_m"])
    ax.plot(tt, pr * 1e3, lw=0.7)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("w at probe [mm]")
    ax.set_title(f"plane wave: $T$={2*np.pi/disp['omega_measured_rad_s']:.2f} s "
                 f"($\\lambda$={disp['wavelength_measured_m']:.1f} m)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / "fig1_dispersion.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig1_dispersion.png", "path": path,
                 "caption": "Flexural-gravity dispersion of the ice plate on water: analytic "
                            "$\\omega^2=(Dk^4+\\rho_w g)/(\\rho_i h+m_a(k))$ vs. the numerical "
                            f"wave (wavelength error {disp['wavelength_error_pct']:.3f}%, group "
                            f"velocity error {disp['group_speed_error_pct']:.2f}%)."})

    # ---- fig 2: static Hetenyi -------------------------------------------
    st = bundle["static"]
    rr = np.linspace(0.0, 3.0 * st["characteristic_length_m"], 200)
    wref = np.atleast_1d(hankel_gauss_deflection(rr, st["total_force_N"], st["patch_sigma_m"],
                                                st["rigidity_Nm"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    ax.plot(rr / st["characteristic_length_m"], wref * 1e3, "k-",
            label="Hankel/Hetényi reference")
    ax.plot(np.asarray(st["radii_m"]) / st["characteristic_length_m"],
            np.asarray(st["w_reference_m"]) * 1e3, "ko", ms=4, mfc="none",
            label="reference sample points")
    ax.set_xlabel("$r/\\ell_c$")
    ax.set_ylabel("steady deflection [mm]")
    ax.set_title(f"(b) steady deflection, $P$={st['total_force_N']/1e6:.2f} MN, "
                 f"$\\ell_c$={st['characteristic_length_m']:.1f} m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ts = st["truncation_study"]
    ax.semilogx([t["domain_over_lc"] for t in ts], [t["w_centre_error_pct"] for t in ts],
                "o-", label="centre deflection")
    ax.semilogx([t["domain_over_lc"] for t in ts], [t["max_profile_error_pct"] for t in ts],
                "s--", label="max profile error")
    ax.axhline(5.0, color="r", ls=":", label="5% criterion")
    ax.set_xlabel("periodic domain size $L/\\ell_c$")
    ax.set_ylabel("error [%]")
    ax.set_title("periodic-box truncation study")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig2_static_hetenyi.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig2_static_hetenyi.png", "path": path,
                 "caption": "Verification of the steady plate-on-fluid deflection against the "
                            "Hetényi/Kelvin-function solution (independent Hankel quadrature); "
                            f"best centre-deflection error {st['w_centre_error_pct']:.2f}%, "
                            f"profile error {st['max_profile_error_pct']:.2f}%."})

    # ---- fig 3: impact time histories ------------------------------------
    res = bundle["impact_elastic"]["result"]
    hh = res["history"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax = axes[0]
    ax.plot(hh["time_s"], np.asarray(hh["force_N"]) / 1e6, "k-", label="contact force")
    ax.plot(hh["time_s"], np.asarray(hh["speed_m_s"]), "b--", label="ship speed [m/s]")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("F [MN] / V [m/s]")
    ax.set_title("(c) bow-impact time history")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(hh["penetration_m"], np.asarray(hh["force_N"]) / 1e6, "k-")
    ax.set_xlabel("penetration $\\delta$ [m]")
    ax.set_ylabel("F [MN]")
    ax.set_title("crushing law $F=p(A)A$, $p=CA^{-0.5}$")
    ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(hh["time_s"], np.asarray(hh["sigma1_max_Pa"]) / 1e6, "r-", label="max bottom $\\sigma_1$")
    if res["fracture"]:
        ax.axhline(res["fracture"]["strength_Pa"] / 1e6, color="k", ls=":", label="$\\sigma_f$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("stress [MPa]")
    ax.set_title("bottom-surface tensile stress")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig3_impact_history.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig3_impact_history.png", "path": path,
                 "caption": "Coupled impact time history: contact force, ship deceleration, "
                            f"penetration and peak bottom stress (peak load {res['peak_force_N']/1e6:.2f} MN, "
                            f"duration {res['duration_s']:.2f} s)."})

    # ---- fig 4: fields at crack initiation --------------------------------
    fr = bundle["impact_fracture"]
    fl = fr["final"]
    x, y = fl["x"], fl["y"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    ax = axes[0]
    im = ax.pcolormesh(x, y, fl["w"] * 1e3, shading="auto", cmap="viridis")
    ax.set_title("deflection $w$ [mm]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1]
    s1 = np.asarray(fl["sigma1"]) / 1e6
    im = ax.pcolormesh(x, y, s1, shading="auto", cmap="inferno")
    cr = fr["result"]["fracture"]
    if cr:
        ax.plot(cr["x_m"], cr["y_m"], "c*", ms=14, label="crack initiation")
        rr_ring = cr["ring"]["ring_radius_m"]
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cr["x_m"] + rr_ring * np.cos(th), cr["y_m"] + rr_ring * np.sin(th),
                "c--", lw=1.2, label="predicted ring-crack radius")
        ax.legend(fontsize=7, loc="upper right")
    ax.set_title("bottom $\\sigma_1$ [MPa]")
    ax.set_xlabel("x [m]")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[2]
    r = np.hypot(x - (0.5 * fr["result"]["config"]["grid"]["lx"]), y - 0.5 * fr["result"]["config"]["grid"]["lx"])
    bins = np.linspace(0.0, 6.0 * fr["result"]["characteristic_length_m"], 60)
    idx = np.digitize(r.ravel(), bins)
    prof_b, prof_t = [], []
    s1b = np.asarray(fl["sigma1"]).ravel() / 1e6
    s1t = np.asarray(fl["sigma1_top"]).ravel() / 1e6
    for i in range(1, len(bins)):
        m = idx == i
        prof_b.append(s1b[m].max() if np.any(m) else np.nan)
        prof_t.append(s1t[m].max() if np.any(m) else np.nan)
    ax.plot(bins[1:] / fr["result"]["characteristic_length_m"], prof_b, "r-", label="bottom $\\sigma_1$")
    ax.plot(bins[1:] / fr["result"]["characteristic_length_m"], prof_t, "b--", label="top $\\sigma_1$")
    ax.axhline(0.0, color="k", lw=0.5)
    if cr:
        ax.axhline(cr["strength_Pa"] / 1e6, color="k", ls=":", label="$\\sigma_f$")
        ax.axvline(cr["ring"]["ring_radius_over_lc"], color="c", ls="--", label="ring crack $r/\\ell_c$")
    ax.set_xlabel("$r/\\ell_c$")
    ax.set_ylabel("stress [MPa]")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title("radial stress profile (crack pattern)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = outdir / "fig4_fracture_fields.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig4_fracture_fields.png", "path": path,
                 "caption": "Fields at the instant of flexural fracture: deflection, bottom-surface "
                            "maximum principal stress with the predicted radial crack initiation "
                            "point and the predicted ring-crack circle, and the radial bottom/top "
                            "stress profiles."})

    # ---- fig 5: energy ledger --------------------------------------------
    led = res["ledger"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    t = np.asarray(hh["time_s"])
    ax.plot(t, np.asarray(hh["work_on_plate_J"]) / 1e6, "k-", label="work on plate")
    ax.plot(t, np.asarray(hh["plate_energy_J"]) / 1e6, "b--", label="plate energy")
    ax.plot(t, np.asarray(hh["dissipation_J"]) / 1e6, "g-.", label="damping loss")
    ax.plot(t, np.asarray(hh["crushing_energy_J"]) / 1e6, "m:", label="crushing")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("energy [MJ]")
    ax.set_title("(e) energy ledger")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    labels = ["ship KE\nreleased", "plate\nenergy", "crushing", "damping", "fracture"]
    vals = [led["ship_kinetic_released_J"], led["plate_energy_final_J"],
            led["crushing_energy_J"], led["dissipation_J"], led["fracture_energy_J"]]
    ax.bar(range(len(vals)), np.asarray(vals) / 1e6, color=["k", "b", "m", "g", "r"])
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("[MJ]")
    ax.set_title(f"residual {led['residual_total_pct']:.3f}% (<=5%), "
                 f"impulse err {led['impulse_error_pct']:.2f}%")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = outdir / "fig5_energy_ledger.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig5_energy_ledger.png", "path": path,
                 "caption": "Energy ledger of the impact: ship kinetic energy released vs. plate "
                            "strain+buoyancy+kinetic energy, fluid kinetic energy, crushing work "
                            "and damping loss; residual and impulse-momentum error annotated."})

    # ---- fig 6: moving (supersonic) load ---------------------------------
    mv = bundle["moving_load"]
    fx = mv["fields"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    im = ax.pcolormesh(fx["x"], fx["y"], fx["w"] * 1e3, shading="auto", cmap="RdBu", vmin=-30, vmax=30)
    ax.set_title(f"moving load V={mv['speed_m_s']:.0f} m/s "
                 f"($V/c_{{min}}$={mv['supersonic_ratio']:.2f})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046, label="w [mm]")
    ax = axes[1]
    p_abs = np.abs(fx["fluid_pressure"]) / 1e3
    im = ax.pcolormesh(fx["x"], fx["y"], p_abs, shading="auto", cmap="magma")
    ax.set_title("underwater radiation pressure $|p_f|$ [kPa]")
    ax.set_xlabel("x [m]")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = outdir / "fig6_supersonic_waves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig6_supersonic_waves.png", "path": path,
                 "caption": "Flexural wave radiation by a load moving faster than the minimum "
                            "phase speed of the ice-coupled system: deflection field and the "
                            "underwater potential-flow pressure field."})

    # ---- animation -------------------------------------------------------
    frames = bundle["impact_elastic"]["frames"]
    if len(frames) >= 3:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fl0 = bundle["impact_elastic"]["final"]
        xa, ya = fl0["x"], fl0["y"]
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
        vmax = max(float(np.max(np.abs(f["w"]))) for f in frames) * 1e3 + 1e-9
        smax = max(float(np.max(f["sigma1"])) for f in frames) / 1e6 + 1e-9
        im0 = axes[0].pcolormesh(xa, ya, frames[0]["w"] * 1e3, shading="auto",
                                 cmap="viridis", vmin=-vmax, vmax=vmax)
        im1 = axes[1].pcolormesh(xa, ya, frames[0]["sigma1"] / 1e6, shading="auto",
                                 cmap="inferno", vmin=0.0, vmax=smax)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, label="w [mm]")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, label="$\\sigma_1$ [MPa]")
        axes[0].set_xlabel("x [m]")
        axes[1].set_xlabel("x [m]")
        title = fig.suptitle("ice-cover impact t=0.00 s")

        def _update(i: int):
            im0.set_array((frames[i]["w"] * 1e3).ravel())
            im1.set_array((frames[i]["sigma1"] / 1e6).ravel())
            title.set_text(f"ice-cover impact  t={frames[i]['t']:.2f} s")
            return im0, im1, title

        # pcolormesh arrays must be set with set_array on flattened data
        anim = FuncAnimation(fig, _update, frames=len(frames), blit=False)
        path = outdir / "impact_animation.gif"
        anim.save(path, writer=PillowWriter(fps=6))
        plt.close(fig)
        figs.append({"name": "impact_animation.gif", "path": path,
                     "caption": "Animation of the coupled ice-cover response during the bow "
                                "impact: deflection and bottom-surface maximum principal stress."})
    return figs


def run_campaign(outdir: str | Path = "artifacts/m1138", quick: bool = False) -> dict:
    """Run all cases, write the criteria bundle (JSON + figures) and return it."""
    from .contact import simulate_ram_rigid

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nx = 128 if quick else 512
    dt = 0.02 if quick else 0.01
    dom = (384.0, 768.0) if quick else (256.0, 384.0, 512.0, 768.0, 1024.0)

    disp = run_dispersion_case(
        nx=1024 if not quick else 256, lx=4096.0,
        plane_duration=60.0 if not quick else 20.0,
        packet_duration=12.0 if not quick else 4.0,
    )
    static = run_static_case(domains=dom)
    static_pattern = measure_static_wavelength()

    base = dict(
        params=IceWaterParameters(E=9.0e9, h=1.5, nu=0.3, depth=50.0, damping=2.0e3),
        ship_mass=8.0e6, impact_speed=1.5, half_angle_deg=22.0,
        max_contact_width=12.0, ice_thickness_contact=1.5,
        crushing_coefficient=2.6e6, nx=nx, ny=nx, dt=dt,
    )
    elastic = run_impact_case(ImpactConfig(**base, flexural_strength=None,
                                           label="impact_elastic"))
    fracture = run_impact_case(ImpactConfig(**base, flexural_strength=0.7e6,
                                            label="impact_fracture"))
    moving = run_moving_load_case(nx=192 if quick else 384, lx=768.0,
                                  duration=3.0 if quick else 6.0, dt=dt)

    rigid = simulate_ram_rigid(
        8.0e6, 1.5, WedgeBow(22.0, 12.0), 1.5, CrushingLaw(coefficient=2.6e6),
        dt=2.0e-4,
    )
    closed = ram_rigid_closed_form(8.0e6, 1.5, WedgeBow(22.0, 12.0), 1.5,
                                   CrushingLaw(coefficient=2.6e6))
    rigid_vs_closed_pct = abs(rigid.peak_force - closed["peak_force_N"]) / closed["peak_force_N"] * 100.0

    bundle = {
        "dispersion": disp,
        "static": static,
        "static_pattern": static_pattern,
        "impact_elastic": elastic,
        "impact_fracture": fracture,
        "moving_load": moving,
        "rigid_ram": {
            "peak_force_N": rigid.peak_force,
            "peak_pressure_Pa": rigid.peak_pressure,
            "peak_area_m2": rigid.peak_area,
            "time_to_stop_s": rigid.time_to_stop,
            "crushing_energy_J": rigid.crushing_energy(),
            "vs_closed_form_pct": rigid_vs_closed_pct,
        },
    }
    figs = make_figures(bundle, outdir)

    res_e = elastic["result"]
    res_f = fracture["result"]
    frac = res_f.get("fracture") or {}
    ring = frac.get("ring") or {}
    peak_area = res_e["peak_area_m2"]
    env_pressure = float(CrushingLaw(coefficient=2.6e6).pressure(peak_area))
    contact_pressure = float(res_e["peak_contact_pressure_Pa"])
    criteria = {
        "a_dispersion": {
            "wavelength_error_pct": disp["wavelength_error_pct"],
            "omega_error_pct": disp["omega_error_pct"],
            "group_velocity_error_pct": disp["group_speed_error_pct"],
            "static_null_ring_error_pct": static_pattern["first_zero_error_pct"],
            "tolerance_pct": 3.0,
            "pass": bool(disp["wavelength_error_pct"] <= 3.0
                         and disp["omega_error_pct"] <= 3.0
                         and disp["group_speed_error_pct"] <= 3.0),
        },
        "b_static_deflection": {
            "domain_m": static["best_domain_m"],
            "centre_error_pct": static["w_centre_error_pct"],
            "profile_error_pct": static["max_profile_error_pct"],
            "analytical_centre_m": static["w_centre_analytic_m"],
            "tolerance_pct": 5.0,
            "pass": bool(static["w_centre_error_pct"] <= 5.0
                         and static["max_profile_error_pct"] <= 5.0),
        },
        "c_impact_load": {
            "rigid_ram_vs_closed_form_pct": rigid_vs_closed_pct,
            "coupled_peak_force_MN": res_e["peak_force_N"] / 1e6,
            "rigid_closed_form_peak_MN": closed["peak_force_N"] / 1e6,
            "compliance_reduction_pct": res_e["peak_force_vs_closed_form_pct"],
            "impulse_error_pct": res_e["ledger"]["impulse_error_pct"],
            "peak_pressure_Pa": res_e["peak_pressure_Pa"],
            "peak_area_m2": peak_area,
            "peak_contact_pressure_Pa": contact_pressure,
            "crushing_pressure_at_peak_area_Pa": env_pressure,
            "contact_patch_smoothing_pct": abs(contact_pressure - env_pressure)
            / env_pressure * 100.0,
            "tolerance_pct_analytic": 3.0,
            "experimental_records": "NOT AVAILABLE IN ENVIRONMENT -- see report, criterion (c) is "
                                    "reported as an analytic anchor + empirical-envelope consistency, "
                                    "not a <=15% validation against raw impact records",
            "pass_analytic_anchor": bool(rigid_vs_closed_pct <= 3.0),
            "pass_envelope_consistency": bool(
                abs(contact_pressure - env_pressure) / env_pressure * 100.0 <= 15.0),
        },
        "d_fracture": {
            "triggered": bool(frac.get("triggered", False)),
            "strength_Pa": frac.get("strength_Pa"),
            "crack_time_s": frac.get("time_s"),
            "crack_radius_over_lc": frac.get("radius_over_lc"),
            "stress_ratio": frac.get("stress_ratio"),
            "ring_radius_over_lc": ring.get("ring_radius_over_lc"),
            "ring_radius_m": ring.get("ring_radius_m"),
            "pattern": "radial (star) cracks at the contact + circumferential (ring) crack outside",
            "pass": bool(frac.get("triggered", False)
                         and (frac.get("radius_over_lc") or 0.0) < 1.0
                         and 0.3 <= (ring.get("ring_radius_over_lc") or -1.0) <= 3.5),
        },
        "e_energy": {
            "ship_kinetic_released_J": res_e["ledger"]["ship_kinetic_released_J"],
            "plate_energy_J": res_e["ledger"]["plate_energy_final_J"],
            "crushing_J": res_e["ledger"]["crushing_energy_J"],
            "damping_J": res_e["ledger"]["dissipation_J"],
            "fracture_J": res_e["ledger"]["fracture_energy_J"],
            "residual_plate_side_pct": res_e["ledger"]["residual_plate_side_pct"],
            "residual_total_pct": res_e["ledger"]["residual_total_pct"],
            "tolerance_pct": 5.0,
            "pass": bool(res_e["ledger"]["residual_total_pct"] <= 5.0
                         and res_e["ledger"]["residual_plate_side_pct"] <= 5.0),
        },
        "f_artifacts": {
            "figures": [{"name": f["name"]} for f in figs],
            "pytest": "tests/test_ice_*.py",
        },
    }
    bundle["criteria"] = criteria
    bundle_json = {
        "mission": "W7-N10 ice-cover dynamic response to icebreaker bow impact",
        "cases": {
            "dispersion": {k: v for k, v in disp.items()
                           if not isinstance(v, (list, dict))},
            "static": static,
            "static_pattern": static_pattern,
            "impact_elastic": {k: v for k, v in res_e.items() if k != "history"},
            "impact_fracture": {k: v for k, v in res_f.items() if k != "history"},
            "moving_load": {k: v for k, v in moving.items() if k != "fields"},
            "rigid_ram": bundle["rigid_ram"],
            "rigid_closed_form": closed,
        },
        "criteria": criteria,
        "figures": [{"name": f["name"], "caption": f["caption"]} for f in figs],
        "history_impact_elastic": res_e["history"],
        "history_impact_fracture": res_f["history"],
    }
    (outdir / "criteria.json").write_text(json.dumps(bundle_json, indent=2, default=float))
    (outdir / "history_elastic.csv").write_text(_history_csv(res_e["history"]))
    (outdir / "history_fracture.csv").write_text(_history_csv(res_f["history"]))
    bundle["figs"] = figs
    bundle["criteria_path"] = str(outdir / "criteria.json")
    return bundle


def _history_csv(hist: dict) -> str:
    keys = list(hist.keys())
    n = len(hist[keys[0]])
    lines = [",".join(keys)]
    for i in range(n):
        lines.append(",".join(f"{hist[k][i]:.6e}" for k in keys))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "artifacts/m1138"
    quick = "--quick" in sys.argv
    b = run_campaign(out, quick=quick)
    print(json.dumps(b["criteria"], indent=2, default=float))
