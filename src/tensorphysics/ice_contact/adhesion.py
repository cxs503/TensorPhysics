"""Temperature dependence of ice adhesion strength on a structural counterface.

Physical basis
--------------
Ice does not bond to a substrate through a sharp solid-solid interface: an
interfacial quasi-liquid layer (QLL) of premelted water always separates the ice
from the structure.  Its thickness grows as the melting point is approached,
h_QLL ~ A (T_melt - T)^(-1/3) (interfacial premelting, Wettlaufer & Worster
2006).  Two consequences are observed experimentally and are reproduced here:

* within a fraction of a kelvin of the melting point the van der Waals bridging
  stress ``A_H / (6 pi h_QLL^3)`` collapses, so a warm interface is essentially
  non-adhesive -- this is the sharp ``~1 K'' collapse near 0 degC;
* away from the melting point the interfacial shear strength increases roughly
  linearly with supercooling up to a maximum near 10 K of supercooling
  (T ~ -10 degC for fresh water ice on steel) and then decays mildly as the ice
  becomes brittle (Raraty & Tabor 1958; Petrenko 1994 review; measured values on
  steel span roughly 0.2-1.5 MPa).

The implemented strength law is therefore

    sigma(T) = sigma_peak * (dT / dT_peak)**m                 dT <= dT_peak
    sigma(T) = max(sigma_floor,
                   sigma_peak * exp(-(dT - dT_peak)/dT_cold))  dT >  dT_peak

with ``dT = T_melt - T``, exponent ``m = 1`` (the premelting/vdW scaling,
available from :meth:`IceAdhesionModel.vdw_bridging_stress`), peak depression
``dT_peak = 10 K`` and a brittle-range decay length ``dT_cold = 25 K``.  The
``vdw_bridging_stress`` accessor is reported alongside so the near-melting
collapse width is not hidden inside a fitted constant.

Limitations
-----------
``sigma_peak``, ``dT_cold`` and ``dT_peak`` are substrate- and rate-dependent
empirical parameters; the defaults are an engineering calibration for
ice-on-carbon-steel shear tests at low sliding speed, not a first-principles
prediction.  Strain-rate dependence of adhesion is not modelled.
"""

from dataclasses import dataclass, field
from math import exp

from .properties import IceWaterProperties

__all__ = ["IceAdhesionModel", "qll_thickness", "vdw_bridging_stress"]

_DEFAULT_PROPS = IceWaterProperties()


def qll_thickness(
    temperature: float,
    props: IceWaterProperties | None = None,
    coefficient: float = 4.0e-10,
) -> float:
    """Interfacial quasi-liquid layer thickness [m].

    ``h = coefficient * (T_melt - T)^(-1/3)``.  The coefficient is expressed in
    ``m K**(1/3)``; the default gives ~0.19 nm at 10 K of supercooling and
    ~0.86 nm at 0.1 K, the observed order of magnitude.
    """
    p = props or _DEFAULT_PROPS
    if coefficient <= 0.0:
        raise ValueError("coefficient must be positive")
    depression = p.melting_temperature - float(temperature)
    if depression <= 0.0:
        raise ValueError("qll_thickness requires temperature below the melting point")
    return coefficient * depression ** (-1.0 / 3.0)


def vdw_bridging_stress(
    temperature: float,
    props: IceWaterProperties | None = None,
    hamaker: float = 5.0e-21,
    coefficient: float = 4.0e-10,
    cohesive_strength: float = 1.5e6,
) -> float:
    """van der Waals bridging stress across the QLL [Pa], capped by cohesion.

    ``A_H / (6 pi h^3)`` -- grows as ``(T_melt - T)`` because ``h^-3`` carries the
    ``(T_melt-T)`` dependence of the premelting law.  The cap is the cohesive
    strength of polycrystalline ice, beyond which the interface cannot be the
    weak link.
    """
    if hamaker <= 0.0:
        raise ValueError("hamaker must be positive")
    if cohesive_strength <= 0.0:
        raise ValueError("cohesive_strength must be positive")
    h = qll_thickness(temperature, props, coefficient)
    stress = hamaker / (6.0 * 3.141592653589793 * h ** 3)
    return min(stress, cohesive_strength)


@dataclass(frozen=True)
class IceAdhesionModel:
    """Empirical + premelting-informed ice adhesion strength model."""

    props: IceWaterProperties = field(default_factory=IceWaterProperties)
    peak_strength: float = 0.9e6        # Pa, adhesion maximum (ice on steel)
    peak_depression: float = 10.0       # K  below T_melt where the maximum occurs
    cold_decay_length: float = 25.0     # K, brittle-range decay length
    floor_fraction: float = 0.05        # dimensionless, residual adhesion fraction
    exponent: float = 1.0               # warm-branch exponent (premelting scaling)
    qll_coefficient: float = 4.0e-10    # m K**(1/3)

    def __post_init__(self) -> None:
        for name in ("peak_strength", "peak_depression", "cold_decay_length"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0.0:
                raise ValueError(f"{name} must be a positive real number")
        if not 0.0 <= self.floor_fraction < 1.0:
            raise ValueError("floor_fraction must lie in [0, 1)")
        if self.exponent <= 0.0:
            raise ValueError("exponent must be positive")

    # -- premelting geometry ------------------------------------------------
    def qll_thickness(self, temperature: float) -> float:
        """Interfacial quasi-liquid layer thickness [m] at ``temperature``."""
        return qll_thickness(temperature, self.props, self.qll_coefficient)

    def vdw_bridging_stress(self, temperature: float) -> float:
        """van der Waals bridging stress across the QLL [Pa]."""
        return vdw_bridging_stress(
            temperature, self.props, coefficient=self.qll_coefficient
        )

    # -- strength law -------------------------------------------------------
    def adhesion_strength(self, temperature: float) -> float:
        """Interfacial shear strength [Pa] at absolute ``temperature`` [K]."""
        depression = self.props.melting_temperature - float(temperature)
        if depression <= 0.0:
            return 0.0
        if depression <= self.peak_depression:
            ratio = depression / self.peak_depression
            return self.peak_strength * ratio ** self.exponent
        decay = exp(-(depression - self.peak_depression) / self.cold_decay_length)
        return max(self.floor_fraction * self.peak_strength, self.peak_strength * decay)

    def peak_temperature(self) -> float:
        """Absolute temperature [K] at which adhesion is maximal."""
        return self.props.melting_temperature - self.peak_depression

    def near_melting_collapse_width(self, fraction: float = 0.5) -> float:
        """Supercooling [K] over which the vdW bridging stress drops to ``fraction``.

        Diagnostic for how sharp the near-melting adhesion loss is; from
        ``A_H/(6 pi h^3)`` with ``h ~ (T_m - T)^(-1/3)`` the bridging stress is
        linear in the supercooling, so the width is
        ``fraction * sigma_cohesive * 6 pi A^3 / A_H``.
        """
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must lie in (0, 1)")
        props = self.props
        cohesive = 1.5e6
        hamaker = 5.0e-21
        stress = fraction * cohesive
        del props
        return stress * 6.0 * 3.141592653589793 * self.qll_coefficient ** 3 / hamaker

    def boundary_friction_cap(self, pressure: float, temperature: float) -> float:
        """Boundary friction coefficient limited by interfacial shear strength.

        ``min(mu_b, sigma_adh / p)``: at low nominal pressure the adhesion
        strength caps the shear stress the interface can transmit.
        """
        if pressure <= 0.0:
            raise ValueError("pressure must be positive")
        adhesion = self.adhesion_strength(temperature)
        return adhesion / pressure