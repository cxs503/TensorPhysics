"""Ice load spectrum at blade level.

Model chain (all terms are closed-form physics, no tabulated result values):

1. **Ice milling torque** - the blade edge crushes ice over the contact area
   ``c(r) * h`` where ``h`` is the depth of cut per blade pass,
   ``h = v_advance / (n Z)``.  Integrating the crushing stress over the blade
   gives ``Q_mill = sigma_ice * h * int c(r) r dr``.  This is the classical
   milling (crushing) model, identical in structure to the ice-torque
   formulations used by ice-class rules; it reproduces ``Q_mill ~ O(Q_design)``.
2. **Class envelope** - the maximum ice torque is expressed as a multiple
   ``k_ice`` of the design open-water torque (ice-class rule structure).  The
   *spectrum* of impact magnitudes is log-normal between the milling level
   (median) and the envelope maximum (99th percentile).
3. **Blade-level decomposition** - the shaft ice torque is carried by the single
   blade in contact: ``F_t = Q / r_impact`` with an axial companion
   ``F_a = k_ax F_t``.  Impacts are **phase locked** to blade passage: the struck
   blade is the one currently inside the ice-contact sector, so every impulse on
   a given blade occurs inside the same narrow shaft-angle window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .blade_stress import BladeGeometry


@dataclass
class IceEnvelope:
    """Operating condition / ice class definition."""

    key: str
    label: str
    sigma_ice: float          # ice crushing strength [Pa]
    k_torque_max: float       # Q_ice,max / Q_design [-]
    impacts_per_hour: float   # blade impacts (all blades) per hour of ice ops
    advance_speed: float      # ice milling advance speed [m/s]
    r_impact_hat: float = 0.75   # radius of ice contact, r/R
    k_axial: float = 0.60        # axial / tangential force ratio [-]
    impact_duration: float = 0.015  # pulse width [s] (half-sine)
    sector_half_width: float = 0.35  # ice contact sector half-angle [rad]


#: Ice conditions, ordered by severity.  ``design`` is the design ice class.
ICE_CASES: Dict[str, IceEnvelope] = {
    "light": IceEnvelope("light", "Light brash ice (Ice Class 1C equivalent)",
                         sigma_ice=1.2e6, k_torque_max=1.35, impacts_per_hour=120.0,
                         advance_speed=1.20),
    "design": IceEnvelope("design", "Design ice class 1A (dense channel)",
                          sigma_ice=2.0e6, k_torque_max=2.20, impacts_per_hour=420.0,
                          advance_speed=0.70),
    "heavy": IceEnvelope("heavy", "Heavy ridged ice (1A + ridges, survival)",
                         sigma_ice=2.6e6, k_torque_max=3.00, impacts_per_hour=900.0,
                         advance_speed=0.45),
}


def ice_milling_torque(geom: BladeGeometry, env: IceEnvelope, n_rps: float,
                       sigma_ice: Optional[float] = None,
                       advance_speed: Optional[float] = None) -> Dict[str, float]:
    """Mean ice milling torque from the crushing-area model [N m]."""
    sigma = env.sigma_ice if sigma_ice is None else sigma_ice
    v = env.advance_speed if advance_speed is None else advance_speed
    h_cut = v / (n_rps * geom.Z)                     # depth of cut per blade pass

    r = np.linspace(geom.root_radius(), 0.98 * geom.R, 120)
    integrand = geom.chord(r) * r
    geom_integral = float(np.trapezoid(integrand, r))  # int c r dr  [m^3]
    Q = sigma * h_cut * geom_integral
    # per-blade tangential force at the contact radius
    r_imp = env.r_impact_hat * geom.R
    return {
        "depth_of_cut_m": float(h_cut),
        "geom_integral_c_r_m3": geom_integral,
        "Q_mill_Nm": float(Q),
        "r_impact_m": float(r_imp),
        "F_tangential_mean_N": float(Q / r_imp),
    }


def impact_spectrum(env: IceEnvelope, Q_mill: float, Q_design: float,
                    n_samples: int = 4096, seed: int = 0) -> Dict[str, object]:
    """Log-normal impact magnitude spectrum between milling level and envelope.

    Median = ``Q_mill``, 99th percentile = ``k_torque_max * Q_design``.
    Returns sampled torque peaks together with the distribution parameters.
    """
    Q_max = env.k_torque_max * Q_design
    Q_max = max(Q_max, 1.05 * Q_mill)
    sigma_ln = np.log(Q_max / Q_mill) / 2.3263          # 99 % -> z = 2.3263
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_samples)
    Q = Q_mill * np.exp(sigma_ln * z)
    Q = np.minimum(Q, 1.05 * Q_max)
    return {
        "Q_max_envelope_Nm": float(Q_max),
        "Q_median_Nm": float(Q_mill),
        "sigma_ln": float(sigma_ln),
        "k_ice_envelope": float(Q_max / Q_design),
        "samples_Nm": Q,
        "p50": float(np.percentile(Q, 50)),
        "p99": float(np.percentile(Q, 99)),
        "mean": float(Q.mean()),
    }


def blade_impact_train(geom: BladeGeometry, env: IceEnvelope, n_rps: float,
                       duration: float, Q_spectrum: Dict[str, object],
                       dt: float = 5.0e-4, seed: int = 1) -> Dict[str, object]:
    """Synthesise the blade-level ice load train over ``duration`` seconds.

    Each impact is placed at a random time; the blade that receives it is the
    one inside the ice-contact sector at that instant, which makes every impulse
    phase-locked with respect to that blade's own passage through the sector.

    Returns the shaft torque, the per-blade tangential/axial force histories and
    the impact bookkeeping (times, blade index, magnitude, phase).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(duration / dt))
    t = np.arange(n_steps) * dt
    n_impacts = rng.poisson(env.impacts_per_hour * duration / 3600.0)
    n_impacts = max(int(n_impacts), 1)

    Q_samples = Q_spectrum["samples_Nm"]
    Q_peaks = rng.choice(Q_samples, size=n_impacts, replace=True)
    t_on = np.sort(rng.uniform(0.0, duration, size=n_impacts))

    omega = 2 * np.pi * n_rps
    r_imp = env.r_impact_hat * geom.R
    sector = env.sector_half_width

    torque = np.zeros(n_steps)
    blade_index = np.zeros(n_impacts, dtype=int)
    blade_phase = np.zeros(n_impacts)
    force_t = np.zeros((geom.Z, n_steps))
    force_a = np.zeros((geom.Z, n_steps))
    tau = env.impact_duration

    for i, (t0, Qpk) in enumerate(zip(t_on, Q_peaks)):
        # which blade is inside the ice contact sector (sector centred at theta=0)
        theta0 = (omega * t0) % (2 * np.pi)
        b_contact = int(np.floor((2 * np.pi - theta0) / (2 * np.pi / geom.Z))) % geom.Z
        # blade b angular position: theta_b = omega t + 2 pi b / Z  (mod 2pi)
        theta_b = (omega * t0 + 2 * np.pi * b_contact / geom.Z) % (2 * np.pi)
        phase = (theta_b - 0.0)
        blade_index[i] = b_contact
        blade_phase[i] = phase
        F_t = Qpk / r_imp
        F_a = env.k_axial * F_t
        i0 = int(t0 / dt)
        i1 = min(n_steps, i0 + int(np.ceil(3.0 * tau / dt)))
        tt = t[i0:i1] - t0
        shape = np.where(tt <= tau, np.sin(np.pi * np.clip(tt, 0, tau) / tau), 0.0)
        torque[i0:i1] += Qpk * shape
        force_t[b_contact, i0:i1] += F_t * shape
        force_a[b_contact, i0:i1] += F_a * shape

    return {
        "t": t,
        "torque": torque,
        "force_tangential": force_t,
        "force_axial": force_a,
        "blade_index": blade_index,
        "blade_phase": blade_phase,
        "t_on": t_on,
        "Q_peaks": Q_peaks,
        "n_impacts": n_impacts,
        "dt": dt,
        "r_impact_m": float(r_imp),
        "sector_half_width": sector,
    }