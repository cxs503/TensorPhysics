"""Stribeck friction curve extended with the melt-film parameter.

The classical Stribeck curve uses the Hersey/Sommerfeld number to separate
boundary, mixed and hydrodynamic lubrication.  For an ice/structure contact the
natural coordinate is instead the *film parameter*

    Lambda = h_film / sigma_rms

i.e. the meltwater film thickness normalised by the counterface roughness.  This
module extends the Stribeck picture in two ways:

* a load-sharing mixture model in which the fraction of load carried by
  asperities decays exponentially with the film parameter
  (``A_r = exp(-decay * Lambda)``, the Greenwood-Williamson exponential form),

      mu(h, p, V, T) = A_r * mu_boundary + (1 - A_r) * mu_viscous(h, p, V, T)

  with the viscous (hydrodynamic) branch ``mu_viscous = eta(T) V / (p h)``; the
  melt film therefore switches friction off as it thickens, which is the physical
  content of the ice friction "melting regime";

* a residual lubricated friction floor ``mu_floor`` (ploughing, capillary and
  viscoelastic losses survive even when the film is fully separated), which also
  keeps the closure from predicting an unphysically frictionless interface at
  asperity-scale pressures;

* a closed-form extended Stribeck curve with a rising hydrodynamic branch,
  ``mu(H) = mu_min + (mu_b - mu_min) exp(-k H) + c H``, whose minimum is used in
  the report to bracket the operating point.

Regime boundaries follow the standard lubrication convention:
``Lambda < 1`` boundary, ``1 <= Lambda <= 3`` mixed, ``Lambda > 3`` hydrodynamic.

Limitations
-----------
``mu_boundary`` is a dry ice/steel boundary value; asperity-scale pressure
concentration, ploughing and ice fracture contributions are not resolved.  The
blend is a load-sharing mixture, not a solution of the Reynolds equation, so it
is a tribological closure, not a flow solver.
"""

from dataclasses import dataclass, field

from .properties import IceWaterProperties

__all__ = [
    "StribeckParams",
    "film_parameter",
    "lubrication_regime",
    "asperity_load_fraction",
    "viscous_friction_coefficient",
    "friction_coefficient",
    "stribeck_curve",
    "stribeck_curve_minimum",
]

_DEFAULT_PROPS = IceWaterProperties()


@dataclass(frozen=True)
class StribeckParams:
    """Closure parameters of the extended Stribeck model."""

    mu_boundary: float = 0.10          # dry ice/steel boundary friction coefficient
    roughness: float = 1.0e-6          # rms counterface roughness [m]
    asperity_decay: float = 1.0        # A_r = exp(-asperity_decay * Lambda)
    boundary_limit: float = 1.0        # Lambda < boundary_limit  -> boundary regime
    hydrodynamic_limit: float = 3.0    # Lambda > hydrodynamic_limit -> hydrodynamic
    viscous_cap: float = 10.0          # numerical cap on the viscous branch
    mu_floor: float = 5.0e-3           # residual lubricated/ploughing friction

    def __post_init__(self) -> None:
        for name in ("mu_boundary", "roughness", "asperity_decay", "viscous_cap", "mu_floor"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0.0:
                raise ValueError(f"{name} must be a positive real number")
        if not 0.0 < self.boundary_limit < self.hydrodynamic_limit:
            raise ValueError("require 0 < boundary_limit < hydrodynamic_limit")


def film_parameter(film_thickness: float, roughness: float) -> float:
    """Non-dimensional film parameter ``Lambda = h / sigma_rms``."""
    if film_thickness < 0.0:
        raise ValueError("film_thickness must be non-negative")
    if roughness <= 0.0:
        raise ValueError("roughness must be positive")
    return film_thickness / roughness


def lubrication_regime(film_thickness: float, params: StribeckParams | None = None) -> str:
    """Classical lubrication regime label for a given film thickness."""
    p = params or StribeckParams()
    lam = film_parameter(film_thickness, p.roughness)
    if lam < p.boundary_limit:
        return "boundary"
    if lam <= p.hydrodynamic_limit:
        return "mixed"
    return "hydrodynamic"


def asperity_load_fraction(film_thickness: float, params: StribeckParams | None = None) -> float:
    """Fraction of the load carried by asperities, ``exp(-decay * Lambda)``."""
    p = params or StribeckParams()
    lam = film_parameter(film_thickness, p.roughness)
    from math import exp

    return exp(-p.asperity_decay * lam)


def viscous_friction_coefficient(
    film_thickness: float,
    pressure: float,
    velocity: float,
    temperature: float,
    props: IceWaterProperties | None = None,
    params: StribeckParams | None = None,
) -> float:
    """Hydrodynamic (viscous shear) friction coefficient ``eta V / (p h)``."""
    p = params or StribeckParams()
    pr = props or _DEFAULT_PROPS
    if film_thickness <= 0.0:
        return p.viscous_cap
    if pressure <= 0.0:
        raise ValueError("pressure must be positive")
    if velocity < 0.0:
        raise ValueError("velocity must be non-negative")
    mu = pr.viscosity(temperature) * velocity / (pressure * film_thickness)
    return min(mu, p.viscous_cap)


def friction_coefficient(
    film_thickness: float,
    pressure: float,
    velocity: float,
    temperature: float,
    params: StribeckParams | None = None,
    props: IceWaterProperties | None = None,
    adhesion_model=None,
) -> float:
    """Load-sharing friction coefficient of the extended Stribeck model.

    ``mu = A_r mu_b + (1 - A_r) mu_viscous`` with the boundary value optionally
    capped by the interfacial adhesion strength (``sigma_adh / p``), which is what
    makes the friction coefficient temperature dependent through the adhesion
    model as well as through the meltwater viscosity.
    """
    p = params or StribeckParams()
    pr = props or _DEFAULT_PROPS
    if film_thickness < 0.0:
        raise ValueError("film_thickness must be non-negative")
    if pressure <= 0.0:
        raise ValueError("pressure must be positive")
    mu_b = p.mu_boundary
    if adhesion_model is not None:
        mu_b = min(mu_b, adhesion_model.boundary_friction_cap(pressure, temperature))
    shear_fraction = asperity_load_fraction(film_thickness, p)
    mu_v = viscous_friction_coefficient(film_thickness, pressure, velocity, temperature, pr, p)
    mu = shear_fraction * mu_b + (1.0 - shear_fraction) * mu_v
    return max(mu, p.mu_floor)


def stribeck_curve(hersey: float, *, mu_boundary: float = 0.10, mu_minimum: float = 0.01,
                   decay: float = 8.0, hydro_slope: float = 0.02) -> float:
    """Closed-form extended Stribeck curve with a rising hydrodynamic branch.

    ``mu(H) = mu_minimum + (mu_boundary - mu_minimum) exp(-decay H) + hydro_slope H``

    ``H`` is the Hersey number (dimensionless in this normalisation).  The curve
    decreases through the boundary/mixed regimes, passes a minimum and rises
    again in the hydrodynamic regime.
    """
    if hersey < 0.0:
        raise ValueError("hersey must be non-negative")
    if mu_boundary <= 0.0 or mu_minimum <= 0.0 or decay <= 0.0 or hydro_slope <= 0.0:
        raise ValueError("curve parameters must be positive")
    from math import exp

    return mu_minimum + (mu_boundary - mu_minimum) * exp(-decay * hersey) + hydro_slope * hersey


def stribeck_curve_minimum(*, mu_boundary: float = 0.10, mu_minimum: float = 0.01,
                           decay: float = 8.0, hydro_slope: float = 0.02) -> tuple[float, float]:
    """Location and value of the extended Stribeck minimum ``(H*, mu*)``.

    Obtained from ``d mu/dH = 0``: ``exp(-decay H*) = hydro_slope / (decay (mu_b - mu_min))``.
    """
    from math import exp, log

    if mu_boundary <= mu_minimum:
        raise ValueError("mu_boundary must exceed mu_minimum")
    ratio = hydro_slope / (decay * (mu_boundary - mu_minimum))
    if ratio >= 1.0:
        raise ValueError("no interior minimum for this parameter set")
    h_star = -log(ratio) / decay
    return h_star, stribeck_curve(
        h_star,
        mu_boundary=mu_boundary,
        mu_minimum=mu_minimum,
        decay=decay,
        hydro_slope=hydro_slope,
    )