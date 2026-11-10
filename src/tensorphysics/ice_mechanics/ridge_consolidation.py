"""Ice-ridge consolidation thermodynamics and its coupling to ship resistance.

An ice ridge is a pile of broken ice blocks: a sail above the waterline and a
keel below it.  The keel is a *porous rubble body* -- typically 30-50 % of its
gross volume is void, filled with seawater.  In winter the pore water refreezes
inwards from the cold surfaces and the rubble turns into a bonded
"consolidated layer" whose strength grows with time; the consolidated layer is
what a ship actually has to break, while the loose rubble underneath merely has
to be pushed aside.  This module closes that chain:

``(a) keel structure`` -> ``(b) refreezing`` -> ``(c) strength`` -> ``(d) resistance``

(a) Keel macro-pore structure
    Porosity ``phi_keel`` in ``[0.3, 0.5]`` of gross keel volume, pore space
    filled with seawater of salinity ``S_w``.  Effective properties of the
    two-phase composite (ice skeleton + brine-filled pores) are volumetric
    averages of density and heat capacity plus the geometric-mean conductivity

        k_eff = k_ice**(1-phi) * k_brine**phi

    which is bracketed by the Voigt (parallel, upper) and Reuss (series, lower)
    bounds; :func:`keel_properties` returns all three (plus Maxwell-Eucken) so
    the mixing assumption is auditable instead of hidden.

(b) Refreezing -- modified Stefan law
    Only the *pore water* freezes, so the volumetric latent heat available per
    unit gross keel volume is ``phi * rho_ice * L_f`` (not ``rho_ice * L_f``).
    With the consolidated composite as conducting solid phase and the pore water
    held at its liquidus ``T_f``, the one-phase Stefan problem gives

        h_c(t) = 2 * lambda * sqrt(kappa_eff * t),
        lambda * exp(lambda**2) * erf(lambda) = Ste / sqrt(pi),
        Ste = C_vol * (T_f - T_s) / (phi * rho_ice * L_f),
        kappa_eff = k_eff / C_vol,

    i.e. exactly the "modified Stefan" form ``h_c = k_c sqrt(kappa t)`` with the
    *dimensionless* consolidation constant ``k_c = 2 lambda(Ste)``.  The
    quasi-steady limit ``k_c ~ sqrt(2 Ste)`` is returned as well; the difference
    is the heat stored in the growing consolidated layer (~6.5 % at
    ``Ste = 0.45``), reported by the acceptance harness instead of hidden.

    Two fronts are supported (:func:`consolidation_state`): one growing inwards
    from the cold upper surface (air, optionally through a snow cover) and one
    from the lower keel boundary.  In the real ocean the lower boundary sits at
    the *same* liquidus as the pore water, so that front is stagnant; the general
    two-front formulation is kept because it is (i) what the model needs once the
    keel floats in supercooled water or rests on a cold bottom, and (ii) the only
    way to show quantitatively that the lower front is negligible.

(c) Strength evolution
    ``sigma_c(t) = sigma_ice(T_mean) * (1 - phi_keel) * f(C)`` with the
    consolidation degree ``C = h_consolidated / h_keel`` and ``f(C) = C**q``
    (``q = 1``: bonded load-bearing area proportional to the refrozen volume;
    other ``q`` available as a sensitivity knob).  ``(1 - phi_keel)`` is the
    *geometric* solid-area fraction of the keel cross-section (unfrozen voids
    carry no tensile/shear load); the temperature dependence of the solid ice
    strength comes from :mod:`tensorphysics.ice_mechanics.constitutive`
    (Schulson 1999) evaluated with the *brine* volume fraction of the ice matrix,
    so brine softening is counted once, not twice.

(d) Consolidation-resistance coupling
    ``R_ridge = R_sail + R_keel`` with

        R_keel = R_plate(h_c, sigma_c)                       # Lindqvist (1989)
               + C1 (1-phi) rho_i g B (h_k**2 - h_c**2)      # rubble buoyancy
               + C2 (1-phi) rho_i B V**2 (h_k - h_c)         # rubble momentum

    where ``R_plate`` reuses :func:`tensorphysics.ice.resistance.level_ice_resistance`
    with the consolidated layer thickness ``h_c`` *in place of* the level-ice
    thickness (the ISO 19906 view of a ridge keel as a solid sheet of thickness
    ``h_c``).  The two rubble terms are the Keinonen-anchored coefficients of
    :func:`tensorphysics.ice.resistance.ridge_resistance` restricted to the
    *unconsolidated* depth range ``[h_c, h_k]``; the restriction is chosen so
    that ``C -> 0`` reproduces the existing ridge law exactly (verified to
    machine precision by the acceptance harness).

Validity / evidence grading
---------------------------
* implementation : REAL -- closed-form thermodynamics plus reuse of the verified
  resistance laws; every number is reproducible from the module.
* ``k_eff`` mixing law : SEMI-EMPIRICAL (geometric mean, bracketed by bounds).
* ``f(C) = C**q`` and the ``R_keel`` layer decomposition : SEMI-EMPIRICAL
  engineering closures; sensitivity to ``q`` and to the decomposition is reported
  in ``runs/in13_w3n05/``.
* absolute accuracy of ``R_ridge`` against full-scale trials : WITHHELD -- no
  ridge-ramming full-scale record exists in this repository; nothing here claims
  better than the source laws' own (semi-empirical) accuracy.
* validity domain: the Lindqvist plate term is a level-ice law, meaningful for
  ``h_c <= min(0.03 L, 0.10 B)``; the Keinonen rubble law is flagged outside
  ``h_k <= min(0.15 B, 4 m)``.  Requests outside those ranges are *flagged*,
  never silently extrapolated (see ``flags`` in the outputs).

References
----------
* Stefan, J. (1891) / Neumann, F.: similarity solution of the one-phase Stefan
  problem (implemented reference: ``ice_contact/stefan.py``).
* Lindqvist, G. (1989) "A straightforward method for calculation of ice
  resistance of ships", POAC'89, Vol. 2, 722-735.
* Keinonen, A. (1983) "Icebreaking model testing -- a review of recent
  developments", POAC'83, 429-444.
* Timco, G.W. & Weeks, W.F. (2010) "A review of the engineering properties of
  sea ice", Cold Reg. Sci. Technol. 60, 107-129.
* Lepparanta, M. & Hakala, R. (1992) "The structure and strength of first-year
  ice ridges in the Baltic Sea", Cold Reg. Sci. Technol. 20, 295-311.
* ISO 19906:2019, Annex A -- ridge keel loads via the consolidated layer.
* Schulson, E.M. (1999) "The structure and mechanical behavior of ice",
  J. Glaciol. 45, 406-416.

Units: SI throughout (m, s, W, J, N, Pa); temperatures in degC unless a name
carries ``_K``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..ice.resistance import (
    G,
    MU_HULL_ICE,
    RHO_ICE,
    SIGMA_F_REF,
    DomainLimits,
    ShipGeometry,
    level_ice_resistance,
)
from ..ice.resistance import PARAMETER_PROVENANCE as RESISTANCE_PROVENANCE
from ..ice_contact.stefan import neumann_lambda
from .constitutive import compressive_strength
from .thermodynamics import brine_volume_fraction, melting_temperature

__all__ = [
    "OCEAN_SUPERCOOLING",
    "PARAMETER_PROVENANCE",
    "ConsolidationState",
    "KeelProperties",
    "consolidated_layer_temperature",
    "consolidated_strength",
    "consolidation_history",
    "consolidation_state",
    "consolidation_time_to_degree",
    "consolidation_time_to_thickness",
    "keel_properties",
    "keel_resistance_coupled",
    "maxwell_eucken_conductivity",
    "quasi_steady_constant",
    "ridge_total_resistance",
    "sail_resistance_coupled",
    "solid_ice_strength",
    "stefan_constant",
    "stefan_number",
]

# ── physical constants ─────────────────────────────────────────────────
K_ICE = 2.1              # W/(m K) pure ice Ih
K_BRINE = 0.55           # W/(m K) seawater / brine near 0 degC
RHO_BRINE = 1025.0       # kg/m^3
CP_ICE = 2090.0          # J/(kg K)
CP_BRINE = 3985.0        # J/(kg K)
L_FUSION = 334_000.0     # J/kg
S_SEAWATER = 32.0        # g/kg, pore-water salinity of a ridge keel
S_ICE_MATRIX = 8.0       # g/kg, salinity of the ice blocks of a first-year keel
PHI_KEEL_MIN = 0.30      # task / measured keel porosity range
PHI_KEEL_MAX = 0.50

#: Supercooling of the water column below the surrounding sheets.  Observed
#: values are of order 10-50 mK in frazil-producing leads; the *base* case of
#: this module is 0 K, i.e. the pore water and the ocean share one liquidus and
#: the lower consolidation front is stagnant.
OCEAN_SUPERCOOLING = 0.0

PARAMETER_PROVENANCE: Dict[str, Dict[str, Any]] = {
    "phi_keel_range": {
        "value": [PHI_KEEL_MIN, PHI_KEEL_MAX],
        "unit": "-",
        "meaning": "冰脊龙骨总孔隙率范围（孔隙内充满海水）",
        "source": "Timco & Weeks (2010) 实测龙骨孔隙率 0.25-0.40；任务给定工程范围 0.3-0.5",
        "grade": "LITERATURE-INTERVAL",
    },
    "k_ice": {
        "value": K_ICE,
        "unit": "W/(m K)",
        "meaning": "纯冰导热系数（柱状冰，-10 degC 附近）",
        "source": "Timco & Weeks (2010) 1.6-2.3 W/(m K)，取 2.1",
        "grade": "LITERATURE-INTERVAL",
    },
    "k_brine": {
        "value": K_BRINE,
        "unit": "W/(m K)",
        "meaning": "孔隙海水/卤水导热系数",
        "source": "海水 0 degC 约 0.56 W/(m K)；卤水略低，工程取 0.55",
        "grade": "LITERATURE",
    },
    "cp_brine": {
        "value": CP_BRINE,
        "unit": "J/(kg K)",
        "meaning": "孔隙海水比热",
        "source": "S=32 g/kg、0 degC 海水 cp 约 3.99 kJ/(kg K)",
        "grade": "LITERATURE",
    },
    "s_seawater": {
        "value": S_SEAWATER,
        "unit": "g/kg",
        "meaning": "龙骨孔隙水盐度（≈ 海水盐度）",
        "source": "北冰洋/波罗的海工程取值 30-35 g/kg",
        "grade": "LITERATURE-INTERVAL",
    },
    "s_ice_matrix": {
        "value": S_ICE_MATRIX,
        "unit": "g/kg",
        "meaning": "构成龙骨的冰块本体盐度（一年冰）",
        "source": "Timco & Weeks (2010) 一年冰 bulk salinity 4-12 g/kg，取 8",
        "grade": "LITERATURE-INTERVAL",
    },
    "k_snow": {
        "value": 0.25,
        "unit": "W/(m K)",
        "meaning": "积雪导热系数（冰脊顶面可选绝热层）",
        "source": "Sturm et al. (1997) 0.08-0.5 W/(m K)，密度 300 kg/m^3 取 0.25",
        "grade": "LITERATURE-INTERVAL",
    },
    "strength_exponent_q": {
        "value": 1.0,
        "unit": "-",
        "meaning": "固结度-强度闭合 f(C)=C^q；q=1 表示承压面积随再冻结体积线性增长",
        "source": "工程闭合（端点 f(0)=0、f(1)=1 由物理约束固定）；q∈{0.5,1,2} 敏感性在报告中给出",
        "grade": "SEMI-EMPIRICAL",
    },
    "phi_sail": {
        "value": 0.50,
        "unit": "-",
        "meaning": "水上冰帆堆积孔隙率（较龙骨疏松，无浮力压密）",
        "source": "冰帆堆积实测孔隙率高于龙骨（Timco & Weeks 2010 综述）",
        "grade": "SEMI-EMPIRICAL",
    },
}

_RIDGE_C1 = float(RESISTANCE_PROVENANCE["ridge_C1"]["value"])
_RIDGE_C2 = float(RESISTANCE_PROVENANCE["ridge_C2"]["value"])


# ── (a) keel macro-pore structure: effective composite properties ───────


def _check_porosity(porosity: float, *, allow_outside: bool = False) -> float:
    if not math.isfinite(porosity):
        raise ValueError(f"porosity must be finite, got {porosity!r}")
    if not 0.0 <= porosity < 1.0:
        raise ValueError(f"porosity must lie in [0, 1), got {porosity!r}")
    if not allow_outside and not (PHI_KEEL_MIN - 1e-12 <= porosity <= PHI_KEEL_MAX + 1e-12):
        raise ValueError(
            f"keel porosity {porosity!r} outside the measured range "
            f"[{PHI_KEEL_MIN}, {PHI_KEEL_MAX}] (Timco & Weeks 2010 / task spec); "
            "pass allow_outside=True for sensitivity studies"
        )
    return float(porosity)


def maxwell_eucken_conductivity(
    porosity: float, k_ice: float = K_ICE, k_brine: float = K_BRINE
) -> float:
    """Maxwell-Eucken bound for an ice matrix with spherical brine inclusions."""
    _check_porosity(porosity, allow_outside=True)
    num = 2.0 * k_ice + k_brine - 2.0 * porosity * (k_ice - k_brine)
    den = 2.0 * k_ice + k_brine + porosity * (k_ice - k_brine)
    return k_ice * num / den


@dataclass(frozen=True)
class KeelProperties:
    """Effective thermophysical properties of the brine-saturated rubble keel."""

    porosity: float
    k_eff: float            # W/(m K), geometric mean (model)
    k_voigt: float          # W/(m K), parallel bound (upper)
    k_reuss: float          # W/(m K), series bound (lower)
    k_maxwell: float        # W/(m K), Maxwell-Eucken bound
    rho_eff: float          # kg/m^3, gross (bulk) density
    cvol: float             # J/(m^3 K), volumetric heat capacity of the composite
    kappa_eff: float        # m^2/s, k_eff / cvol
    latent_vol: float       # J/m^3, phi * rho_ice * L_f (pore water only)
    freezing_point: float   # degC, liquidus of the pore water

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def keel_properties(
    porosity: float,
    *,
    salinity: float = S_SEAWATER,
    k_ice: float = K_ICE,
    k_brine: float = K_BRINE,
    rho_ice: float = RHO_ICE,
    rho_brine: float = RHO_BRINE,
    cp_ice: float = CP_ICE,
    cp_brine: float = CP_BRINE,
    latent_heat: float = L_FUSION,
) -> KeelProperties:
    """Effective properties of the rubble keel (ice skeleton + brine pores)."""
    phi = _check_porosity(porosity)
    solid = 1.0 - phi
    k_eff = k_ice**solid * k_brine**phi
    k_voigt = solid * k_ice + phi * k_brine
    k_reuss = 1.0 / (solid / k_ice + phi / k_brine)
    rho_eff = solid * rho_ice + phi * rho_brine
    cvol = solid * rho_ice * cp_ice + phi * rho_brine * cp_brine
    return KeelProperties(
        porosity=phi,
        k_eff=k_eff,
        k_voigt=k_voigt,
        k_reuss=k_reuss,
        k_maxwell=maxwell_eucken_conductivity(phi, k_ice, k_brine),
        rho_eff=rho_eff,
        cvol=cvol,
        kappa_eff=k_eff / cvol,
        latent_vol=phi * rho_ice * latent_heat,
        freezing_point=float(melting_temperature(salinity)),
    )


# ── (b) refreezing: modified Stefan law ────────────────────────────────


def stefan_number(delta_t: float, props: KeelProperties) -> float:
    """Stefan number of the porous problem, ``Ste = C_vol dT / (phi rho_i L)``."""
    if not math.isfinite(delta_t):
        raise ValueError(f"delta_t must be finite, got {delta_t!r}")
    if delta_t < 0.0:
        raise ValueError("freezing requires a positive temperature difference")
    if props.latent_vol <= 0.0:
        raise ValueError("latent_vol must be positive")
    return props.cvol * delta_t / props.latent_vol


def stefan_constant(ste: float) -> float:
    """Dimensionless consolidation constant ``k_c = 2 lambda(Ste)`` (exact)."""
    if not math.isfinite(ste) or ste <= 0.0:
        raise ValueError(f"Stefan number must be positive and finite, got {ste!r}")
    return 2.0 * neumann_lambda(ste)


def quasi_steady_constant(ste: float) -> float:
    """Quasi-steady (linear profile) constant ``sqrt(2 Ste)``.

    Leading order in ``Ste``; used as an asymptotic cross-check of
    :func:`stefan_constant`.  It *over*-predicts the front position because the
    heat stored in the growing consolidated layer is neglected.
    """
    if not math.isfinite(ste) or ste <= 0.0:
        raise ValueError(f"Stefan number must be positive and finite, got {ste!r}")
    return math.sqrt(2.0 * ste)


def _front_position(
    t: float,
    delta_t: float,
    props: KeelProperties,
    *,
    snow_resistance: float = 0.0,
) -> float:
    """Consolidated thickness grown from one cold boundary at time ``t`` [s].

    Snow-free: exact similarity solution ``h = 2 lambda sqrt(kappa t)``.

    With a series resistance ``R = h_snow / k_snow`` the quasi-steady balance
    ``k dT / (h + k R) = (phi rho_i L) dh/dt`` integrates to

        h = -kR + sqrt((kR)**2 + 2 k dT t / (phi rho_i L)),

    scaled by the similarity correction ``2 lambda / sqrt(2 Ste)`` so the
    snow-free limit stays exact and the neglected interface heat capacity is
    accounted for at the same order as in the exact solution.
    """
    if t < 0.0:
        raise ValueError(f"time must be >= 0, got {t!r}")
    if t == 0.0 or delta_t <= 0.0:
        return 0.0
    k = props.k_eff
    R = snow_resistance
    if R <= 0.0:
        return stefan_constant(stefan_number(delta_t, props)) * math.sqrt(props.kappa_eff * t)
    ste = stefan_number(delta_t, props)
    correction = stefan_constant(ste) / quasi_steady_constant(ste)
    k_r = k * R
    h_qs = -k_r + math.sqrt(k_r * k_r + 2.0 * k * delta_t * t / props.latent_vol)
    return correction * max(h_qs, 0.0)


@dataclass(frozen=True)
class ConsolidationState:
    """Thermodynamic state of one ridge keel at time ``t``."""

    t: float
    keel_depth: float
    porosity: float
    air_temperature: float
    lower_boundary_temperature: float
    h_upper: float            # consolidated thickness grown from the upper face
    h_lower: float            # consolidated thickness grown from the keel bottom
    h_consolidated: float     # h_upper + h_lower, capped at keel_depth
    degree: float             # h_consolidated / keel_depth, in [0, 1]
    mean_temperature: float   # mean temperature of the consolidated layer [degC]
    ste_upper: float
    ste_lower: float
    kc_upper: float
    kc_lower: float
    props: KeelProperties
    flags: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["props"] = self.props.as_dict()
        return out


def _front_pair(
    t: float,
    keel_depth: float,
    props: KeelProperties,
    d_upper: float,
    d_lower: float,
    snow_resistance: float,
) -> "tuple[float, float, List[str]]":
    """Two-front growth with closure handling (fronts capped at ``keel_depth``)."""
    flags: List[str] = []
    h_up = _front_position(t, d_upper, props, snow_resistance=snow_resistance)
    h_lo = _front_position(t, d_lower, props)
    total = h_up + h_lo
    if total > keel_depth:
        scale = keel_depth / total
        h_up *= scale
        h_lo *= scale
        flags.append("CONSOLIDATION_CLOSED")
    return h_up, h_lo, flags


def consolidation_state(
    t: float,
    *,
    keel_depth: float,
    air_temperature: float,
    porosity: float = 0.40,
    lower_boundary_temperature: Optional[float] = None,
    snow_thickness: float = 0.0,
    snow_conductivity: float = 0.25,
    salinity: float = S_SEAWATER,
    props: Optional[KeelProperties] = None,
) -> ConsolidationState:
    """Consolidation state of a ridge keel at time ``t`` [s].

    Parameters
    ----------
    keel_depth : gross keel depth ``h_k`` [m] (5-20 m in the sweep).
    air_temperature : driving air temperature [degC] for the upper boundary; a
        snow cover of ``snow_thickness`` is added in series.
    lower_boundary_temperature : temperature of the lower keel boundary [degC].
        Defaults to the pore-water liquidus plus :data:`OCEAN_SUPERCOOLING`, i.e.
        *no* driving temperature difference -> the lower front is stagnant.  That
        is the physically relevant sea case: the ocean below the keel is at the
        same liquidus as the pore water, so the latent heat can only be removed
        upwards through the rubble.
    """
    if not math.isfinite(t) or t < 0.0:
        raise ValueError(f"t must be finite and >= 0, got {t!r}")
    if keel_depth <= 0.0:
        raise ValueError(f"keel_depth must be > 0, got {keel_depth!r}")
    if air_temperature >= 0.0:
        raise ValueError(f"air temperature must be below 0 degC, got {air_temperature!r}")
    if snow_thickness < 0.0:
        raise ValueError("snow_thickness must be >= 0")
    props = props or keel_properties(porosity, salinity=salinity)
    t_f = props.freezing_point
    if lower_boundary_temperature is None:
        t_lower = t_f + OCEAN_SUPERCOOLING
    else:
        t_lower = float(lower_boundary_temperature)
    snow_resistance = snow_thickness / snow_conductivity if snow_thickness > 0.0 else 0.0

    d_upper = t_f - air_temperature
    d_lower = t_f - t_lower
    h_up, h_lo, flags = _front_pair(t, keel_depth, props, d_upper, d_lower, snow_resistance)
    h_cons = min(h_up + h_lo, keel_depth)
    degree = h_cons / keel_depth
    if h_cons > 0.0:
        mean_t = (
            h_up * 0.5 * (air_temperature + t_f) + h_lo * 0.5 * (t_lower + t_f)
        ) / h_cons
    else:
        mean_t = air_temperature
    if d_lower <= 0.0:
        flags.append("LOWER_FRONT_STAGNANT")
    if d_upper <= 0.0:
        flags.append("UPPER_FRONT_STAGNANT")
    ste_up = stefan_number(d_upper, props) if d_upper > 0.0 else 0.0
    ste_lo = stefan_number(d_lower, props) if d_lower > 0.0 else 0.0
    return ConsolidationState(
        t=t,
        keel_depth=keel_depth,
        porosity=props.porosity,
        air_temperature=air_temperature,
        lower_boundary_temperature=t_lower,
        h_upper=h_up,
        h_lower=h_lo,
        h_consolidated=h_cons,
        degree=degree,
        mean_temperature=mean_t,
        ste_upper=ste_up,
        ste_lower=ste_lo,
        kc_upper=stefan_constant(ste_up) if ste_up > 0.0 else 0.0,
        kc_lower=stefan_constant(ste_lo) if ste_lo > 0.0 else 0.0,
        props=props,
        flags=flags,
    )


def consolidation_time_to_thickness(
    h_target: float,
    *,
    air_temperature: float,
    porosity: float = 0.40,
    snow_thickness: float = 0.0,
    snow_conductivity: float = 0.25,
    salinity: float = S_SEAWATER,
    props: Optional[KeelProperties] = None,
) -> float:
    """Time [s] for the *upper* front to reach ``h_target`` [m].

    Exact inverse of the similarity law (snow-free) or of the snow-resistance
    closed form; round-trip checked in the test suite.
    """
    if h_target < 0.0:
        raise ValueError("h_target must be >= 0")
    if h_target == 0.0:
        return 0.0
    props = props or keel_properties(porosity, salinity=salinity)
    d_upper = props.freezing_point - air_temperature
    if d_upper <= 0.0:
        return math.inf
    if snow_thickness <= 0.0:
        kc = stefan_constant(stefan_number(d_upper, props))
        return (h_target / kc) ** 2 / props.kappa_eff
    ste = stefan_number(d_upper, props)
    correction = stefan_constant(ste) / quasi_steady_constant(ste)
    k = props.k_eff
    k_r = k * snow_thickness / snow_conductivity
    h_qs = h_target / correction
    return ((h_qs + k_r) ** 2 - k_r * k_r) * props.latent_vol / (2.0 * k * d_upper)


def consolidation_time_to_degree(
    degree: float,
    *,
    keel_depth: float,
    air_temperature: float,
    porosity: float = 0.40,
    **kwargs: Any,
) -> float:
    """Time [s] to reach consolidation degree ``degree`` of a keel ``keel_depth``."""
    if not 0.0 <= degree <= 1.0:
        raise ValueError("degree must lie in [0, 1]")
    if keel_depth <= 0.0:
        raise ValueError("keel_depth must be > 0")
    return consolidation_time_to_thickness(
        degree * keel_depth,
        air_temperature=air_temperature,
        porosity=porosity,
        **kwargs,
    )


def consolidation_history(
    times: Sequence[float],
    *,
    keel_depth: float,
    air_temperature: float,
    porosity: float = 0.40,
    **kwargs: Any,
) -> List[ConsolidationState]:
    """Consolidation states on a time grid (time-scale plots)."""
    return [
        consolidation_state(
            t,
            keel_depth=keel_depth,
            air_temperature=air_temperature,
            porosity=porosity,
            **kwargs,
        )
        for t in times
    ]


# ── (c) strength evolution ─────────────────────────────────────────────


def consolidated_layer_temperature(
    air_temperature: float,
    porosity: float = 0.40,
    *,
    salinity: float = S_SEAWATER,
) -> float:
    """Mean temperature [degC] of a fully developed consolidated layer."""
    props = keel_properties(porosity, salinity=salinity)
    return 0.5 * (air_temperature + props.freezing_point)


def solid_ice_strength(temperature: float, salinity_ice: float = S_ICE_MATRIX) -> float:
    """Compressive strength [Pa] of the ice matrix at ``temperature`` [degC].

    Schulson (1999) temperature dependence evaluated with the *brine* volume
    fraction of the ice matrix, so brine softening is counted once.
    """
    if temperature > 0.0:
        raise ValueError("ice temperature must be <= 0 degC")
    t_eval = min(temperature, -1e-6)
    vb = float(brine_volume_fraction(salinity_ice, t_eval))
    return float(compressive_strength(t_eval, vb))


def consolidated_strength(
    degree: float,
    porosity: float,
    mean_temperature: float,
    *,
    sigma_ice: Optional[float] = None,
    exponent: float = 1.0,
    salinity_ice: float = S_ICE_MATRIX,
) -> float:
    """Consolidated keel strength [Pa]: ``sigma_c = sigma_ice (1-phi) C**q``.

    Endpoints are exact: ``C = 0`` -> no strength; ``C = 1`` -> the
    solid-area-reduced keel strength ``sigma_ice (1-phi)``.
    """
    if not 0.0 <= degree <= 1.0:
        raise ValueError(f"degree must lie in [0, 1], got {degree!r}")
    if not 0.0 < porosity < 1.0:
        raise ValueError(f"porosity must lie in (0, 1), got {porosity!r}")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive")
    if sigma_ice is None:
        sigma_ice = solid_ice_strength(mean_temperature, salinity_ice)
    if sigma_ice <= 0.0:
        raise ValueError("sigma_ice must be positive")
    return float(sigma_ice * (1.0 - porosity) * degree**exponent)


# ── (d) consolidation -> resistance coupling ───────────────────────────


def keel_resistance_coupled(
    keel_depth: float,
    speed: float,
    ship: ShipGeometry,
    *,
    degree: float,
    sigma_c: float,
    porosity: float = 0.40,
    rho_ice: float = RHO_ICE,
    sigma_f: float = SIGMA_F_REF,
    mu: float = MU_HULL_ICE,
    limits: Optional[DomainLimits] = None,
) -> Dict[str, Any]:
    """Keel resistance [N] of a ridge with consolidation degree ``degree``.

    ``R_keel = R_plate(h_c, sigma_c) + R_rubble([h_c, h_k])``: ``R_plate`` is
    Lindqvist (1989) at the consolidated thickness ``h_c = C h_k``, the rubble
    terms are the Keinonen coefficients restricted to the unconsolidated depth
    range.  ``C -> 0`` reproduces
    :func:`tensorphysics.ice.resistance.ridge_resistance` exactly.
    """
    if not 0.0 <= degree <= 1.0:
        raise ValueError(f"degree must lie in [0, 1], got {degree!r}")
    if keel_depth <= 0.0:
        raise ValueError("keel_depth must be > 0")
    if speed < 0.0:
        raise ValueError("speed must be >= 0")
    limits = limits or DomainLimits()
    flags: List[str] = []

    h_c = degree * keel_depth
    solid = 1.0 - porosity

    plate = None
    r_plate = 0.0
    if h_c > 0.0:
        plate = level_ice_resistance(
            h_c, speed, ship, sigma_c=sigma_c, sigma_f=sigma_f, mu=mu, limits=limits
        )
        r_plate = float(plate["R_total_N"])
        h_lim = limits.h_limit(ship)
        if h_c > h_lim:
            flags.append(f"PLATE_BEYOND_LEVEL_ICE_LIMIT:{h_c:.3f}>{h_lim:.3f}")
        if h_c < limits.h_min_m:
            flags.append(f"PLATE_BELOW_FLOOR:{h_c:.4f}")

    r_buoy = _RIDGE_C1 * solid * rho_ice * G * ship.B * (keel_depth**2 - h_c**2)
    r_mom = _RIDGE_C2 * solid * rho_ice * ship.B * speed**2 * (keel_depth - h_c)
    if keel_depth > limits.ridge_limit(ship):
        flags.append(
            f"KEEL_BEYOND_RIDGE_LAW_LIMIT:{keel_depth:.3f}>{limits.ridge_limit(ship):.3f}"
        )

    r_keel = r_plate + r_buoy + r_mom
    return {
        "state": "ridge_keel_consolidated",
        "degree": degree,
        "keel_depth_m": keel_depth,
        "h_consolidated_m": h_c,
        "porosity": porosity,
        "sigma_c_Pa": sigma_c,
        "R_plate_N": r_plate,
        "R_rubble_buoyancy_N": r_buoy,
        "R_rubble_momentum_N": r_mom,
        "R_keel_N": r_keel,
        "R_keel_kN": r_keel / 1e3,
        "plate_components": None
        if plate is None
        else {
            k: plate[k]
            for k in ("R_crush_N", "R_break_N", "R_submersion_N", "C_crush", "C_break")
        },
        "plate_flags": [] if plate is None else list(plate["flags"]),
        "flags": flags,
    }


def sail_resistance_coupled(
    sail_height: float,
    speed: float,
    ship: ShipGeometry,
    *,
    degree: float,
    sigma_f: float = SIGMA_F_REF,
    porosity_sail: float = 0.50,
    rho_ice: float = RHO_ICE,
    mu: float = MU_HULL_ICE,
    k_bend: float = 2.0,
) -> Dict[str, Any]:
    """Sail (above-water rubble) resistance [N].

    The sail is *not* buoyantly supported, so the Keinonen buoyancy term has no
    counterpart; only the rubble-momentum term survives, plus -- once the sail is
    consolidated -- a bending term for breaking the bonded sail cap (Lindqvist
    ``R_break`` form).  Both are blended linearly in the consolidation degree,
    the same closure used for the keel.
    """
    if sail_height < 0.0:
        raise ValueError("sail_height must be >= 0")
    if not 0.0 <= degree <= 1.0:
        raise ValueError("degree must lie in [0, 1]")
    r_mom = _RIDGE_C2 * (1.0 - porosity_sail) * rho_ice * ship.B * speed**2 * sail_height
    wedge = math.sin(0.5 * ship.bow_angle_rad) + mu * math.cos(0.5 * ship.bow_angle_rad)
    r_bend = 0.5 * sigma_f * sail_height * ship.B * wedge / k_bend
    r_sail = (1.0 - degree) * r_mom + degree * r_bend
    return {
        "sail_height_m": sail_height,
        "degree": degree,
        "R_sail_rubble_N": (1.0 - degree) * r_mom,
        "R_sail_bending_N": degree * r_bend,
        "R_sail_N": r_sail,
        "R_sail_kN": r_sail / 1e3,
    }


def ridge_total_resistance(
    keel_depth: float,
    speed: float,
    ship: ShipGeometry,
    *,
    degree: float,
    sail_height: Optional[float] = None,
    porosity: float = 0.40,
    air_temperature: Optional[float] = None,
    mean_temperature: Optional[float] = None,
    sigma_c: Optional[float] = None,
    exponent: float = 1.0,
    limits: Optional[DomainLimits] = None,
) -> Dict[str, Any]:
    """``R_ridge = R_sail + R_keel`` [N] for a ridge of consolidation degree ``degree``.

    ``sigma_c`` is closed from :func:`consolidated_strength` unless supplied; the
    mean consolidated-layer temperature is taken from ``mean_temperature`` or from
    ``air_temperature`` (:func:`consolidated_layer_temperature`).  ``sail_height``
    defaults to ``0.2 h_keel`` (Timco & Weeks 2010 report sail/keel ratios
    0.15-0.3) and is echoed in the output.
    """
    if not 0.0 <= degree <= 1.0:
        raise ValueError(f"degree must lie in [0, 1], got {degree!r}")
    if sail_height is None:
        sail_height = 0.2 * keel_depth
    if sigma_c is None:
        if mean_temperature is None:
            if air_temperature is None:
                raise ValueError("supply sigma_c, mean_temperature or air_temperature")
            mean_temperature = consolidated_layer_temperature(air_temperature, porosity)
        sigma_c = consolidated_strength(degree, porosity, mean_temperature, exponent=exponent)

    sail = sail_resistance_coupled(sail_height, speed, ship, degree=degree)
    keel = keel_resistance_coupled(
        keel_depth,
        speed,
        ship,
        degree=degree,
        sigma_c=sigma_c,
        porosity=porosity,
        limits=limits,
    )
    total = sail["R_sail_N"] + keel["R_keel_N"]
    return {
        "state": "ridge_consolidated",
        "degree": degree,
        "keel_depth_m": keel_depth,
        "sail_height_m": sail_height,
        "porosity": porosity,
        "sigma_c_Pa": sigma_c,
        "sigma_c_MPa": sigma_c / 1e6,
        "R_sail_N": sail["R_sail_N"],
        "R_sail_kN": sail["R_sail_N"] / 1e3,
        "R_keel_N": keel["R_keel_N"],
        "R_keel_kN": keel["R_keel_N"] / 1e3,
        "R_ridge_N": total,
        "R_ridge_kN": total / 1e3,
        "sail": sail,
        "keel": keel,
        "flags": list(keel["flags"]),
    }