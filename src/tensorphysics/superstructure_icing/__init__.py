"""Superstructure icing prediction and anti-icing optimisation (TensorPhysics).

Physics chain (all kernels are real solvers, no hard-coded results):

``air``
    moist-air properties (Sutherland, Buck, latent heats).
``flowfield``
    exact potential-flow sections: circular cylinder and Joukowski ellipse
    (pipes, masts, crane booms, blunt panel edges).
``droplets``
    Lagrangian supercooled-droplet tracking (Langmuir-Blodgett trajectory
    method) with Schiller-Naumann drag, exponential (stiff-stable) integrator,
    local collection efficiency ``beta(s)`` and total efficiency ``E``.
``messinger``
    Messinger (1953) control-volume heat balance (rime/glaze regimes, runback,
    evaporation/sublimation, radiation) and transient 1-D ice growth.
``geometry``
    superstructure component model (masts, derricks, piping, bulkheads, deck
    cargo) and per-component ice thickness from the coupled chain.
``anti_icing``
    electro-thermal design: skin-conduction spreading matrix, constrained power
    minimisation (SLSQP) and SIMP-style 0/1 heater-layout optimisation.
``stability``
    KG rise, GM loss, roll inertia and roll-period change from the icing mass
    distribution, with the ISO-15138 style design-load sweep.
``sensitivity``
    the 27-case LWC x MVD x air-temperature matrix and time-step convergence
    study.
``case_runner``
    end-to-end campaign case producing the machine-checkable acceptance record.

Report: ``docs/reports/p380w2_n07/REPORT.md``.
"""

from .air import (  # noqa: F401
    C_W,
    L_F,
    L_S,
    L_V,
    RHO_ICE_GLAZE,
    RHO_ICE_RIME,
    cp_air,
    k_air,
    l_vap,
    le_air,
    mu_air,
    nu_air,
    p_sat_ice,
    p_sat_water,
    rho_air,
    rho_v_sat,
)
from .flowfield import BluffBody, CylinderFlow, EllipseFlow, section_flow  # noqa: F401
from .droplets import (  # noqa: F401
    ImpingementResult,
    beta_ballistic,
    droplet_reynolds,
    impingement,
    particle_relaxation_time,
    stokes_number,
    track_droplets,
)
from .messinger import (  # noqa: F401
    IceGrowthResult,
    MessingerState,
    heat_transfer_coefficient,
    ice_growth_1d,
    messinger_balance,
    messinger_stagnation_reference,
    nusselt_cylinder_churchill_bernstein,
    nusselt_cylinder_hilpert,
    nusselt_flat_plate,
    rime_thickness,
    stefan_rime_thickness,
)
from .geometry import (  # noqa: F401
    Component,
    Superstructure,
    component_ice_thickness,
    design_ice_thickness,
    reference_superstructure,
)
from .anti_icing import (  # noqa: F401
    ElectrothermalAntiIcing,
    HeaterDesign,
    power_saving,
)
from .stability import (  # noqa: F401
    ShipParticulars,
    StabilityResult,
    icing_load_sweep,
    icing_stability,
    reference_hull,
    roll_period,
    threshold_load,
    uniform_load_thickness,
)

__all__ = [n for n in dir() if not n.startswith("_")]