"""Pressure melting of ice Ih via the Clausius-Clapeyron relation.

Background
----------
The ice Ih melting line is anomalous: melting contracts the specific volume
(``v_water - v_ice < 0``), so the Clapeyron relation

    dT_melt/dp = T_melt * (v_water - v_ice) / L_f

has a *negative* slope, i.e. pressure depresses the melting temperature.  For
fresh water ice at the triple point the slope is about -0.074 K/MPa, the
textbook value (1 MPa ~ 9.9 atm gives ~0.073 K).  Consequence: ice that is
cold in the absolute sense can still be superheated with respect to the local
melting point once the contact pressure is high enough -- this is the pressure
melting branch of the ice/structure contact problem, and it becomes active only
at asperity-scale pressures (order 100 MPa).

Model content
-------------
* ``clapeyron_slope`` -- local slope [K/Pa].
* ``melt_temperature_linear`` -- first-order (tangent) approximation.
* ``melt_temperature`` -- exact integral of the Clapeyron ODE for constant
  ``Delta v`` and ``L_f``: ``T(p) = T_ref exp(Delta v (p - p_ref) / L_f)``.
  This is the analytic reference used by the acceptance criterion (error < 1 %
  against a numerically integrated Clapeyron ODE).
* ``integrate_melt_temperature`` -- classical RK4 march of the ODE
  dT/dp = T Delta v / L_f, entirely independent of the closed form above.
* ``pressure_melting_threshold`` -- contact pressure at which ice at a given
  temperature starts to melt with no frictional heating at all.

Limitations
-----------
Constant ``Delta v`` and ``L_f`` are assumed.  Real ice Ih melting line
curvature (IAPWS) becomes visible above ~50 MPa, where the linear/exponential
Clapeyron forms over-predict the depression by O(10-20 %).  The pressure range
used by the contact model (<= 200 MPa) is reported accordingly.
"""

from math import exp, log

from .properties import IceWaterProperties

__all__ = [
    "clapeyron_slope",
    "melt_temperature",
    "melt_temperature_linear",
    "melt_point_depression",
    "integrate_melt_temperature",
    "pressure_melting_threshold",
    "superheat",
]

_DEFAULT_PROPS = IceWaterProperties()


def clapeyron_slope(
    props: IceWaterProperties | None = None,
    pressure_reference: float = 0.0,
) -> float:
    """Clapeyron slope ``dT_melt/dp`` [K/Pa] at the reference pressure.

    Negative for water ice: melting temperature decreases with pressure.
    """
    p = props or _DEFAULT_PROPS
    if pressure_reference < 0.0:
        raise ValueError("pressure_reference must be non-negative")
    t_ref = p.melting_temperature if pressure_reference == 0.0 else melt_temperature(
        pressure_reference, p
    )
    return t_ref * p.specific_volume_change / p.latent_heat_fusion


def melt_temperature(
    pressure: float,
    props: IceWaterProperties | None = None,
    pressure_reference: float = 0.0,
) -> float:
    """Melting temperature [K] at ``pressure`` [Pa].

    Exact integral of the Clausius-Clapeyron ODE with constant specific volume
    change and latent heat.
    """
    p = props or _DEFAULT_PROPS
    pressure = float(pressure)
    if pressure < 0.0:
        raise ValueError("pressure must be non-negative")
    if pressure_reference < 0.0:
        raise ValueError("pressure_reference must be non-negative")
    t_ref = p.melting_temperature if pressure_reference == 0.0 else melt_temperature(
        pressure_reference, p
    )
    exponent = p.specific_volume_change * (pressure - pressure_reference) / p.latent_heat_fusion
    return t_ref * exp(exponent)


def melt_temperature_linear(
    pressure: float,
    props: IceWaterProperties | None = None,
) -> float:
    """Tangent-line (first order) melting temperature [K] at ``pressure``."""
    p = props or _DEFAULT_PROPS
    if pressure < 0.0:
        raise ValueError("pressure must be non-negative")
    return p.melting_temperature + clapeyron_slope(p) * pressure


def melt_point_depression(
    pressure: float,
    props: IceWaterProperties | None = None,
) -> float:
    """Melting point depression ``T_melt(0) - T_melt(p)`` [K], positive."""
    p = props or _DEFAULT_PROPS
    return p.melting_temperature - melt_temperature(pressure, p)


def integrate_melt_temperature(
    pressure_end: float,
    n_steps: int = 200,
    props: IceWaterProperties | None = None,
) -> tuple[list[float], list[float]]:
    """Numerically integrate the Clapeyron ODE with classical RK4.

    Returns ``(pressures, temperatures)`` with ``n_steps + 1`` samples from 0 to
    ``pressure_end``.  Independent of :func:`melt_temperature`, so the two can be
    cross-validated (verification criterion (a)).
    """
    p = props or _DEFAULT_PROPS
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if pressure_end < 0.0:
        raise ValueError("pressure_end must be non-negative")

    slope_factor = p.specific_volume_change / p.latent_heat_fusion  # 1/Pa, negative
    dp = pressure_end / n_steps
    t = p.melting_temperature
    pressures = [0.0]
    temperatures = [t]
    for i in range(n_steps):
        k1 = slope_factor * t
        k2 = slope_factor * (t + 0.5 * dp * k1)
        k3 = slope_factor * (t + 0.5 * dp * k2)
        k4 = slope_factor * (t + dp * k3)
        t += (dp / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        pressures.append(dp * (i + 1))
        temperatures.append(t)
    return pressures, temperatures


def pressure_melting_threshold(
    ice_temperature: float,
    props: IceWaterProperties | None = None,
) -> float:
    """Contact pressure [Pa] at which ice at ``ice_temperature`` begins to melt.

    Inverse of :func:`melt_temperature`: the pressure that depresses the melting
    point down to the ice temperature.  For ice at -10 degC this is ~135 MPa,
    i.e. far above nominal hull contact pressures but inside the asperity-scale
    range.
    """
    p = props or _DEFAULT_PROPS
    if ice_temperature <= 0.0:
        raise ValueError("ice_temperature must be a positive absolute temperature")
    if ice_temperature >= p.melting_temperature:
        return 0.0
    return (p.latent_heat_fusion / p.specific_volume_change) * log(
        ice_temperature / p.melting_temperature
    )


def superheat(
    pressure: float,
    ice_temperature: float,
    props: IceWaterProperties | None = None,
) -> float:
    """Thermodynamic driving temperature ``T_ice - T_melt(p)`` [K].

    Positive values mean the ice is superheated with respect to the local
    melting point, i.e. pressure melting proceeds without frictional heating.
    """
    return float(ice_temperature) - melt_temperature(pressure, props)