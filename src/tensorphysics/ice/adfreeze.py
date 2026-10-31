"""Ice-structure adfreeze (freeze-bond) mechanics.

Physical background
===================
A vessel (or a fixed structure) held in an ice cover loses mobility when the
water film that fills the gap between the ice and the steel surface freezes:
a *chilled joint* forms whose shear strength ``tau_adfreeze`` controls the
pull-out ("breakout") force needed to break the vessel free.

The platform model factorises the joint strength as

    tau_adfreeze = tau_ref(dT) * eta_freeze(t, dT) * phi_rough(Ra) * phi_salt(T, S)

with

``dT``
    Effective undercooling of the joint, ``dT = T_f(S) - T_i`` [K].  For
    freshwater ice ``T_f = 0``; for saline ice the liquidus temperature of
    the brine (freezing point depression) is used, so warm saline joints
    are much weaker than cold freshwater joints at the same ice
    temperature.

``tau_ref(dT)``
    Reference shear strength of a *freshwater* joint with saturated contact
    on smooth steel.  Power law in the undercooling,
    ``tau_ref = tau_sat * dT / (dT + d_half)``, where ``tau_sat``, ``d_half``
    are least-squares calibrated on the digitised Frederking (1984) /
    Frederking & Barker (1979) experiments (four temperatures, -2 .. -20 degC,
    see :mod:`tensorphysics.ice.references`) and the result is capped at the
    shear strength of ice itself (``TAU_ICE_CAP_MPA``), because a bond cannot
    be stronger than the ice it is made of.  The saturating form reproduces
    the observed levelling-off of adfreeze strength at low temperature (no
    free water left to freeze).

``eta_freeze(t, dT)``
    Fraction of the joint area that has already frozen solid after a
    contact time ``t``.  The water film freezes by conduction of the latent
    heat into the ice (Stefan problem), hence the frozen thickness is
    ``h(t) = sqrt(2*k*dT*t / (rho_i*L_f))``.  The gap between ice and steel
    is not uniform (as-received steel roughness, hull waviness, ice edge
    waviness), so it is modelled as exponentially distributed with mean
    ``h_gap``; the frozen area fraction is then ``1 - exp(-h/h_gap)``.
    This is the contact-time kinetics of the bond.

``phi_rough(Ra)``
    Mechanical interlocking amplification, ``1 + k*(1 - exp(-Ra/Ra_ref))``.
    Polished steel is the ``Ra -> 0`` limit (``phi -> 1``).

``phi_salt(T, S)``
    Saline-ice penalty.  The brine volume fraction ``v_b`` (from the
    brine mass balance, Frankenstein & Garner 1967) is unfrozen water that
    carries no shear stress, so ``phi_salt = (1 - v_b)**n``.

Every closure constant carries a provenance label in
:data:`PARAMETER_PROVENANCE` (``physics_derived`` /
``literature_calibrated`` / ``engineering_estimate``) so that the report can
separate first-principles content from calibrated content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (fresh water / ice)
# ---------------------------------------------------------------------------
RHO_ICE = 917.0  # kg/m^3, pure bubble-free ice
RHO_PURE_ICE = 917.0
RHO_BRINE = 1024.0  # kg/m^3, representative brine density (-5..-20 degC)
LATENT_HEAT_FUSION = 3.34e5  # J/kg
K_ICE = 2.2  # W/(m*K), thermal conductivity of ice (-10 degC)
GRAVITY = 9.81

#: Physical upper bound on the joint strength: the adfreeze shear strength
#: cannot exceed the shear strength of the ice itself (~2 MPa).
TAU_ICE_CAP_MPA = 2.0

#: Frederking-calibrated hyperbolic (saturating) closure, see
#: ``calibrate_reference_strength``.  tau_ref = tau_sat*dT/(dT+d_half),
#: capped at TAU_ICE_CAP_MPA.  Calibration range -2 .. -20 degC.
TAU_SAT_MPA = 5.8301
D_HALF_K = 76.1562

#: Roughness interlocking constants (engineering closure, +-30%).
ROUGH_K = 0.80
ROUGH_RA_REF_UM = 20.0

#: Brine-penalty exponent, calibrated so that saline ice (S=32, T=-10 degC)
#: retains ~0.5 of the freshwater adfreeze strength.
SALINITY_EXPONENT = 3.0

#: Default mean gap thickness of the ice/steel joint (hull waviness scale).
DEFAULT_GAP_SCALE_M = 5.0e-3

#: Default as-received (grit-blasted, painted) hull roughness [um].
DEFAULT_ROUGHNESS_UM = 2.5

PARAMETER_PROVENANCE = {
    "TAU_SAT_MPA, D_HALF_K": "literature_calibrated (Frederking 1984, 4 temperatures, -2..-20 degC)",
    "TAU_ICE_CAP_MPA": "physics_derived (adfreeze strength bounded by ice shear strength)",
    "ROUGH_K, ROUGH_RA_REF_UM": "engineering_estimate (interlocking, +-30%)",
    "SALINITY_EXPONENT": "literature_calibrated (single comparative datum, +-40%)",
    "DEFAULT_GAP_SCALE_M": "engineering_estimate (hull waviness / joint pooling)",
    "K_ICE, RHO_ICE, LATENT_HEAT_FUSION": "physics_derived (Stefan problem)",
}


# ---------------------------------------------------------------------------
# Seawater liquidus (freezing point depression) and brine volume fraction
# ---------------------------------------------------------------------------
#: Tabulated seawater liquidus anchors (brine salinity [g/kg] vs temperature
#: [degC]); Assur (1958) / Cox & Weeks (1983) phase diagram, tabulated to
#: +-5%.  Used only for the salinity correction.
_LIQUIDUS_T_C = np.array([-23.0, -20.0, -15.0, -10.0, -5.0, -2.0, -1.0, -0.1])
_LIQUIDUS_SB_GKG = np.array([255.0, 240.0, 197.0, 143.0, 78.0, 35.0, 17.5, 2.0])


def seawater_freezing_point(salinity_gkg: float) -> float:
    """Liquidus (freezing point) temperature [degC] of seawater of given salinity.

    Parameters
    ----------
    salinity_gkg : float
        Bulk ice/water salinity in g/kg (psu). 0 -> 0 degC.

    Notes
    -----
    Inverse of the tabulated Assur/Cox-Weeks liquidus, monotone
    interpolated.  For seawater salinity (S ~ 32 g/kg) this returns
    about -1.8 degC, matching the accepted value.
    """
    s = float(salinity_gkg)
    if s < 0.0:
        raise ValueError("salinity must be non-negative")
    if s == 0.0:
        return 0.0
    # _LIQUIDUS_* is monotone in S_b, so interpolate S_b -> T
    # _LIQUIDUS_SB_GKG increases monotonically with temperature
    order = np.argsort(_LIQUIDUS_SB_GKG)
    t = np.interp(s, _LIQUIDUS_SB_GKG[order], _LIQUIDUS_T_C[order])
    if s > _LIQUIDUS_SB_GKG.max():
        # extrapolate linearly with the last (warmest) slope
        sb_s, t_s = _LIQUIDUS_SB_GKG[order], _LIQUIDUS_T_C[order]
        slope = (t_s[-1] - t_s[-2]) / (sb_s[-1] - sb_s[-2])
        t = t_s[-1] + slope * (s - sb_s[-1])
    return float(t)


def brine_salinity(temperature_c: float) -> float:
    """Brine salinity [g/kg] in equilibrium with ice at ``temperature_c`` [degC]."""
    t = float(temperature_c)
    if t >= 0.0:
        raise ValueError("brine salinity is undefined for T >= 0 degC")
    t = min(max(t, float(_LIQUIDUS_T_C.min())), float(_LIQUIDUS_T_C.max()))
    order = np.argsort(_LIQUIDUS_T_C)
    sb = np.interp(t, _LIQUIDUS_T_C[order], _LIQUIDUS_SB_GKG[order])
    return float(sb)


def brine_volume_fraction(temperature_c: float, salinity_gkg: float) -> float:
    """Brine volume fraction ``v_b`` of saline ice.

    Brine mass balance: all salt is in the brine, so

        v_b = rho * S / (rho_brine * S_b(T)),   rho = rho_i*(1-v_b) + rho_b*v_b

    solved by a few fixed-point iterations.  ``S`` is the bulk salinity
    (kg/kg), ``S_b`` the brine salinity at the ice temperature.
    Clipped to ``[0, 0.6]``; above that the ice is a slush and the model is
    out of its validity range.
    """
    t = float(temperature_c)
    s = float(salinity_gkg) / 1000.0  # kg/kg
    if s < 0.0:
        raise ValueError("salinity must be non-negative")
    if s == 0.0:
        return 0.0
    if t >= seawater_freezing_point(salinity_gkg):
        # ice cannot exist above its liquidus: fully liquid joint
        return 0.6
    sb = brine_salinity(t) / 1000.0  # kg/kg
    rho = RHO_PURE_ICE
    vb = 0.0
    for _ in range(8):
        vb = rho * s / (RHO_BRINE * sb)
        rho = RHO_PURE_ICE * (1.0 - vb) + RHO_BRINE * vb
    return float(min(max(vb, 0.0), 0.6))


# ---------------------------------------------------------------------------
# Reference (freshwater) joint strength: calibrated power law
# ---------------------------------------------------------------------------
def reference_strength_mpa(undercooling_k: float) -> float:
    """Freshwater adfreeze shear strength of a saturated, smooth joint [MPa].

    Saturating (hyperbolic) closure calibrated on the four Frederking (1984)
    temperatures over ``-2 .. -20 degC``::

        tau_ref = tau_sat * dT / (dT + d_half)

    capped at the physical bound ``TAU_ICE_CAP_MPA`` (a bond cannot be
    stronger than the ice it is made of).  The cap is inactive inside the
    calibration range and only prevents runaway extrapolation below -40 degC.
    """
    dT = np.asarray(undercooling_k, dtype=float)
    if np.any(dT < -1e-12):
        raise ValueError("undercooling must be non-negative")
    dT = np.clip(dT, 0.0, None)
    tau = TAU_SAT_MPA * dT / (dT + D_HALF_K)
    tau = np.minimum(tau, TAU_ICE_CAP_MPA)
    return float(tau) if np.ndim(undercooling_k) == 0 else tau


def _relative_residual(tau_sat: float, d_half: float, dT: np.ndarray, tau: np.ndarray) -> float:
    model = np.minimum(tau_sat * dT / (dT + d_half), TAU_ICE_CAP_MPA)
    return float(np.sum(((model - tau) / tau) ** 2))


def calibrate_reference_strength(
    temperatures_c: Iterable[float], strengths_mpa: Iterable[float], refine: int = 4
):
    """Least-squares fit of ``(tau_sat, d_half)`` to adfreeze data (relative error).

    Deterministic coarse grid + successive bisection refinement, numpy only (no
    scipy dependency).  Returns ``(tau_sat, d_half, rms_rel_error)`` so the
    calibration is reproducible from :mod:`tensorphysics.ice.references`
    instead of being a hand-entered constant.
    """
    t = np.asarray(list(temperatures_c), dtype=float)
    tau = np.asarray(list(strengths_mpa), dtype=float)
    if t.shape != tau.shape or t.size < 2:
        raise ValueError("need matching arrays with at least two points")
    if np.any(t >= 0.0) or np.any(tau <= 0.0):
        raise ValueError("temperatures must be negative and strengths positive")
    dT = -t
    # coarse search on a log grid around physically sensible values
    sats = np.geomspace(1.0, 20.0, 60)
    halves = np.geomspace(5.0, 400.0, 60)
    best = (np.inf, 1.0, 1.0)
    for ts in sats:
        for dh in halves:
            r = _relative_residual(ts, dh, dT, tau)
            if r < best[0]:
                best = (r, float(ts), float(dh))
    _, ts, dh = best
    # successive refinement (shrinking box)
    span_s, span_d = ts * 0.5, dh * 0.5
    for _ in range(refine):
        for cand_s in np.linspace(max(ts - span_s, 0.2), ts + span_s, 21):
            for cand_d in np.linspace(max(dh - span_d, 0.5), dh + span_d, 21):
                r = _relative_residual(cand_s, cand_d, dT, tau)
                if r < best[0]:
                    best = (r, float(cand_s), float(cand_d))
        _, ts, dh = best
        span_s *= 0.2
        span_d *= 0.2
    rms = float(np.sqrt(best[0] / tau.size))
    return float(ts), float(dh), rms


# ---------------------------------------------------------------------------
# Freezing kinetics (Stefan problem + distributed gap)
# ---------------------------------------------------------------------------
def stefan_frozen_thickness(undercooling_k: float, time_s: float) -> float:
    """Frozen thickness [m] of a water film cooled from one side (Stefan).

    ``h(t) = sqrt(2*k*dT*t / (rho*L))``.  This is the classical Neumann
    similarity solution in the quasi-steady (Stefan) limit and is used here
    as the analytic benchmark for the kinetics sub-model.
    """
    dT = float(undercooling_k)
    t = float(time_s)
    if dT < 0.0 or t < 0.0:
        raise ValueError("undercooling and time must be non-negative")
    return float(np.sqrt(2.0 * K_ICE * dT * t / (RHO_ICE * LATENT_HEAT_FUSION)))


def frozen_area_fraction(
    undercooling_k: float,
    time_s: float,
    gap_scale_m: float = DEFAULT_GAP_SCALE_M,
) -> float:
    """Fraction of the joint area frozen solid after contact time ``time_s``.

    Gap widths between ice and steel are modelled as exponentially
    distributed with mean ``gap_scale_m``; a gap of width ``g`` is frozen
    once ``h(t) > g``, hence ``eta = 1 - exp(-h(t)/gap_scale)``.  This gives
    the slow, strongly temperature-dependent build-up of adfreeze strength
    observed in contact-time experiments (weak after seconds, saturated
    after tens of minutes to hours).
    """
    if gap_scale_m <= 0.0:
        raise ValueError("gap_scale_m must be positive")
    h = stefan_frozen_thickness(undercooling_k, time_s)
    return float(1.0 - np.exp(-h / gap_scale_m))


# ---------------------------------------------------------------------------
# Roughness and salinity factors
# ---------------------------------------------------------------------------
def roughness_factor(roughness_um: float) -> float:
    """Interlocking amplification of the joint strength (``Ra -> 0`` gives 1)."""
    ra = float(roughness_um)
    if ra < 0.0:
        raise ValueError("roughness must be non-negative")
    return float(1.0 + ROUGH_K * (1.0 - np.exp(-ra / ROUGH_RA_REF_UM)))


def salinity_factor(temperature_c: float, salinity_gkg: float) -> float:
    """Brine penalty ``(1 - v_b)**n``; zero when the joint cannot freeze."""
    t = float(temperature_c)
    if salinity_gkg < 0.0:
        raise ValueError("salinity must be non-negative")
    if t >= seawater_freezing_point(salinity_gkg):
        return 0.0
    vb = brine_volume_fraction(t, salinity_gkg)
    return float((1.0 - vb) ** SALINITY_EXPONENT)


# ---------------------------------------------------------------------------
# Combined adfreeze strength
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdfreezeState:
    """Decomposition of the adfreeze strength at one joint state."""

    tau_adfreeze_mpa: float
    tau_reference_mpa: float
    frozen_fraction: float
    roughness_factor: float
    salinity_factor: float
    undercooling_k: float
    freezing_point_c: float

    def as_dict(self) -> dict:
        return {
            "tau_adfreeze_MPa": self.tau_adfreeze_mpa,
            "tau_reference_MPa": self.tau_reference_mpa,
            "frozen_area_fraction": self.frozen_fraction,
            "roughness_factor": self.roughness_factor,
            "salinity_factor": self.salinity_factor,
            "undercooling_K": self.undercooling_k,
            "freezing_point_C": self.freezing_point_c,
        }


def adfreeze_strength_state(
    ice_temperature_c: float,
    contact_time_s: float = 1.0e4,
    salinity_gkg: float = 0.0,
    roughness_um: float = DEFAULT_ROUGHNESS_UM,
    gap_scale_m: float = DEFAULT_GAP_SCALE_M,
) -> AdfreezeState:
    """Full adfreeze joint state (strength + factor decomposition).

    Parameters
    ----------
    ice_temperature_c : float
        Temperature of the ice / joint interface [degC].
    contact_time_s : float
        Time the ice and the structure have been in contact [s].
    salinity_gkg : float
        Bulk salinity of the ice at the joint [g/kg]; 0 for freshwater ice.
    roughness_um : float
        Arithmetic mean roughness ``Ra`` of the steel surface [um].
    gap_scale_m : float
        Mean width of the water-filled gap between ice and steel [m].

    Returns
    -------
    AdfreezeState
    """
    if contact_time_s < 0.0:
        raise ValueError("contact time must be non-negative")
    tf = seawater_freezing_point(salinity_gkg)
    dT = tf - float(ice_temperature_c)
    if dT < 0.0:
        # Joint is too warm to freeze: no bond at all.
        return AdfreezeState(0.0, 0.0, 0.0, roughness_factor(roughness_um), 0.0, 0.0, tf)
    tau_ref = reference_strength_mpa(dT)
    eta = frozen_area_fraction(dT, contact_time_s, gap_scale_m)
    phi_r = roughness_factor(roughness_um)
    phi_s = salinity_factor(ice_temperature_c, salinity_gkg)
    tau = tau_ref * eta * phi_r * phi_s
    return AdfreezeState(tau, tau_ref, eta, phi_r, phi_s, dT, tf)


def adfreeze_strength(
    ice_temperature_c: float,
    contact_time_s: float = 1.0e4,
    salinity_gkg: float = 0.0,
    roughness_um: float = DEFAULT_ROUGHNESS_UM,
    gap_scale_m: float = DEFAULT_GAP_SCALE_M,
) -> float:
    """Adfreeze shear strength of the joint [MPa]."""
    return adfreeze_strength_state(
        ice_temperature_c, contact_time_s, salinity_gkg, roughness_um, gap_scale_m
    ).tau_adfreeze_mpa


def adfreeze_strength_curve(
    ice_temperatures_c: Iterable[float],
    contact_time_s: float = 1.0e4,
    salinity_gkg: float = 0.0,
    roughness_um: float = DEFAULT_ROUGHNESS_UM,
    gap_scale_m: float = DEFAULT_GAP_SCALE_M,
) -> np.ndarray:
    """Vectorised :func:`adfreeze_strength` over an array of temperatures."""
    return np.array(
        [
            adfreeze_strength(t, contact_time_s, salinity_gkg, roughness_um, gap_scale_m)
            for t in ice_temperatures_c
        ],
        dtype=float,
    )


def contact_time_to_saturation(
    ice_temperature_c: float, salinity_gkg: float = 0.0, fraction: float = 0.95
) -> float:
    """Contact time [s] needed to reach ``fraction`` of the saturated bond.

    Inverts ``eta = 1 - exp(-h/h_gap)`` through the Stefan solution, i.e.
    ``t = (rho*L*h_gap**2 / (2*k*dT)) * ln(1/(1-fraction))**2``.
    Useful for operation planning: how long a hull may sit in ice before the
    bond becomes fully developed.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie in (0, 1)")
    tf = seawater_freezing_point(salinity_gkg)
    dT = tf - float(ice_temperature_c)
    if dT <= 0.0:
        return float("inf")
    h_gap = DEFAULT_GAP_SCALE_M
    t_scale = RHO_ICE * LATENT_HEAT_FUSION * h_gap**2 / (2.0 * K_ICE * dT)
    return float(t_scale * np.log(1.0 / (1.0 - fraction)) ** 2)