"""Coupled ice-floe / water / panel transient impact model.

Reduced-order three-body model
------------------------------
* **floe** -- rigid disk, mass ``m_ice``, brittle (fractures when the contact
  stress reaches ``sigma_ice``);
* **panel** -- 2-D Kirchhoff plate; its effective center stiffness ``k_panel``
  and the 2-D peak field come from :mod:`.plate_fem`;
* **water** -- Wagner-type slamming pressure ``p_w = C_p 0.5 rho V_ice^2``
  (constant during the impact, set by the impact speed) plus hydrodynamic
  added mass / damping.

The floe dynamics is a single-DOF ODE integrated with a fixed step
``dt < 1e-4 s`` (the task's time-step gate).  The contact is a single spring
of stiffness ``k_eff = (1/k_ice + 1/k_panel)^{-1}`` that drops to zero when
the ice fractures (the load path then shifts to the water).

Load paths on the panel
-----------------------
* **ice contact** ``k_eff delta`` (zero after fracture);
* **water slamming** ``p_w * A_water`` with ``A_water = C_area * A_c``
  (``C_area > 1``: the wave wets more of the flare than the floe footprint).
  This is the ``V^2`` load path that sets the literature-consistent scaling.

``P_max = max_t [F_ice(t) + p_w A_water] / A_c``.

Energy balance (acceptance criterion c)
---------------------------------------
    0.5 m_eff V_n^2 = 0.5 m_eff v^2 + 0.5 k_eff delta^2
                      + int c v^2 dt + int p_w A_c v dt + E_fracture
i.e. initial kinetic energy = kinetic + structural strain + fluid dissipation
+ ice fracture energy.  The harness checks the closure error < 5 %.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .contact import ContactLaw, IceFractureState
from .hydro import SlammingHydro
from .plate_fem import PlateFEM
from .properties import (IceFloeProperties, SlammingStructureProperties,
                         SlammingWaterProperties)

__all__ = ["SlammingCase", "SlammingResult", "run_impact"]


@dataclass(frozen=True)
class SlammingCase:
    """One impact case (floe, water, structure, impact state)."""

    v_ice: float                 # m/s, floe speed relative to the ship
    theta_deg: float             # deg, impact angle (floe velocity vs panel surface)
    h_s: float                   # m, wave height
    floe: IceFloeProperties
    water: SlammingWaterProperties
    structure: SlammingStructureProperties
    n_plate: int = 16            # FEM mesh (N x N)
    dt: float = 1.0e-5           # s, time step (< 1e-4 s gate)
    t_end: float = 0.05           # s, simulation window
    contact_radius: float | None = None   # m, default = floe radius
    water_area_factor: float = 3.0       # A_water / A_c (> 1)

    def __post_init__(self) -> None:
        if self.dt >= 1.0e-4:
            raise ValueError("dt must be < 1e-4 s (task time-step gate)")
        if self.v_ice < 0:
            raise ValueError("v_ice must be >= 0")


@dataclass
class SlammingResult:
    """Time histories + peak quantities for one case."""

    t: np.ndarray
    delta: np.ndarray            # penetration [m]
    v: np.ndarray                # velocity [m/s]
    f_ice: np.ndarray            # ice contact force [N]
    f_water_panel: np.ndarray    # water load on the panel [N]
    f_panel: np.ndarray           # total panel load [N]
    p_panel: np.ndarray           # panel pressure [Pa]
    p_max: float                 # Pa
    t_peak: float                # s
    a_contact: float             # m^2
    a_contact_hist: np.ndarray   # contact area history [m^2]
    fracture: IceFractureState
    # energy bookkeeping [J]
    ke0: float
    e_kin: np.ndarray
    e_strain: np.ndarray
    e_damp: np.ndarray
    e_water: np.ndarray
    e_fracture: float
    energy_closure: float        # relative closure error at t_end
    # structural field at the peak (from the FEM)
    peak_field: object | None = None
    # derived
    v_n: float = 0.0
    p_w: float = 0.0
    c_p: float = 0.0
    k_panel: float = 0.0
    k_eff: float = 0.0
    m_eff: float = 0.0


def _build_law(case: SlammingCase, k_panel: float) -> ContactLaw:
    a = case.contact_radius if case.contact_radius is not None else case.floe.radius_m
    sigma_y = case.floe.yield_fraction * case.floe.sigma_ice
    return ContactLaw(
        a=a, k_floe=case.floe.bending_stiffness, k_panel=k_panel,
        sigma_y=sigma_y, sigma_ice=case.floe.sigma_ice, G_f=case.floe.G_f,
        nu=case.floe.nu_ice,
    )


def run_impact(case: SlammingCase) -> SlammingResult:
    """Integrate the impact and return the time histories + peaks."""
    if case.v_ice == 0.0:
        n = int(math.ceil(case.t_end / case.dt)) + 1
        t = np.linspace(0.0, case.t_end, n)
        zero = np.zeros_like(t)
        fr = IceFractureState()
        return SlammingResult(
            t=t, delta=zero.copy(), v=zero.copy(), f_ice=zero.copy(),
            f_water_panel=zero.copy(), f_panel=zero.copy(), p_panel=zero.copy(),
            p_max=0.0, t_peak=0.0, a_contact=0.0, a_contact_hist=zero.copy(),
            fracture=fr, ke0=0.0, e_kin=zero.copy(), e_strain=zero.copy(),
            e_damp=zero.copy(), e_water=zero.copy(), e_fracture=0.0,
            energy_closure=0.0, v_n=0.0, p_w=0.0, c_p=0.0, k_panel=0.0,
            k_eff=0.0, m_eff=0.0,
        )

    # -- panel FEM (effective stiffness) --------------------------------
    fem = PlateFEM(case.structure.side_m, case.structure.thickness_m,
                   case.structure.E_steel, case.structure.nu_steel,
                   case.structure.rho_steel, n=case.n_plate)
    a = case.contact_radius if case.contact_radius is not None else case.floe.radius_m
    res0 = fem.static_circular(a, 1.0e4)
    k_panel = res0.k_eff

    law = _build_law(case, k_panel)
    A_c = law.contact_area
    A_water = case.water_area_factor * A_c
    k_ice = law.k_ice
    k_eff = 1.0 / (1.0 / k_ice + 1.0 / k_panel)

    # -- water side ------------------------------------------------------
    hydro = SlammingHydro(case.water.rho_water, case.water.wave_period_s,
                          case.water.added_mass_coeff, case.water.damping_coeff)
    c_p = hydro.c_p(case.theta_deg)
    p_w = hydro.pressure(case.v_ice, case.theta_deg)   # Pa, constant
    m_add = hydro.added_mass(A_c)
    c_hydro = case.water.damping_coeff * A_c

    # -- effective mass / damping ---------------------------------------
    m_plate_eff = 0.5 * case.structure.total_mass
    m_eff = case.floe.mass + m_plate_eff + m_add
    c_struct = 2.0 * case.structure.damping_ratio * math.sqrt(k_panel * m_plate_eff)
    c = c_hydro + c_struct

    # -- integrate (semi-implicit Euler) --------------------------------
    n = int(round(case.t_end / case.dt)) + 1
    t = np.linspace(0.0, case.t_end, n)
    delta = np.zeros(n)
    v = np.zeros(n)
    f_ice = np.zeros(n)
    f_wp = np.full(n, p_w * A_water)
    state = IceFractureState()

    v_n = case.v_ice * math.sin(math.radians(case.theta_deg))
    v[0] = v_n

    for i in range(n - 1):
        d = delta[i]
        vi = v[i]
        k_t = 0.0 if state.fractured else k_eff
        f_i = k_t * d
        f_ice[i] = f_i
        a_acc = (-f_i - c * vi - p_w * A_c) / m_eff
        v_new = vi + a_acc * case.dt
        d_new = d + v_new * case.dt
        if d_new < 0.0:
            d_new = 0.0
            v_new = 0.0
        delta[i + 1] = d_new
        v[i + 1] = v_new
        # update fracture state from the contact stress
        if not state.fractured and d_new > 0.0:
            sigma = (k_eff * d_new) / A_c
            if sigma >= case.floe.sigma_ice:
                se = 0.5 * k_eff * d_new**2
                a_f = min(A_c, se / case.floe.G_f)
                state.fractured = True
                state.fracture_area = a_f
                state.fracture_energy = case.floe.G_f * a_f
                state.fracture_time = t[i + 1]
            elif sigma >= case.floe.yield_fraction * case.floe.sigma_ice:
                state.yield_onset = True
                if state.yield_time is None:
                    state.yield_time = t[i + 1]
        f_ice[i + 1] = (0.0 if state.fractured else k_eff) * d_new

    f_panel = f_ice + f_wp
    p_panel = f_panel / A_c
    ipk = int(np.argmax(p_panel))
    p_max = float(p_panel[ipk])
    t_peak = float(t[ipk])

    in_contact = delta > 0.0
    a_hist = np.where(in_contact, A_c, 0.0)
    if state.fractured and state.fracture_time is not None:
        after = t >= state.fracture_time
        a_hist = np.where(after, A_c - state.fracture_area, A_c)

    # -- energy bookkeeping ---------------------------------------------
    ke0 = 0.5 * m_eff * v_n**2
    e_kin = 0.5 * m_eff * v**2
    e_strain = 0.5 * k_eff * delta**2
    e_damp = np.cumsum(c * v**2 * case.dt)
    e_water = np.cumsum(p_w * A_c * np.maximum(0.0, v) * case.dt)
    e_fracture = state.fracture_energy
    total = e_kin + e_strain + e_damp + e_water + e_fracture
    closure = np.abs(ke0 - total) / ke0 if ke0 > 0 else 0.0
    energy_closure = float(closure[-1])

    # -- peak structural field (FEM) ------------------------------------
    peak_field = None
    if p_max > 0:
        p_ice_peak = float(f_ice[ipk]) / A_c
        fem_peak = PlateFEM(case.structure.side_m, case.structure.thickness_m,
                           case.structure.E_steel, case.structure.nu_steel,
                           case.structure.rho_steel, n=case.n_plate)
        peak_field = fem_peak.static_circular(a, max(0.0, p_ice_peak))

    return SlammingResult(
        t=t, delta=delta, v=v, f_ice=f_ice, f_water_panel=f_wp,
        f_panel=f_panel, p_panel=p_panel, p_max=p_max, t_peak=t_peak,
        a_contact=A_c, a_contact_hist=a_hist, fracture=state,
        ke0=ke0, e_kin=e_kin, e_strain=e_strain, e_damp=e_damp,
        e_water=e_water, e_fracture=e_fracture,
        energy_closure=energy_closure, peak_field=peak_field,
        v_n=v_n, p_w=p_w, c_p=c_p, k_panel=k_panel, k_eff=k_eff,
        m_eff=m_eff,
    )
