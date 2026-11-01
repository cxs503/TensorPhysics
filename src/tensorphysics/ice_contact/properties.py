"""Thermophysical property basis for the ice / meltwater / structure contact triad.

The basis is deliberately small and auditable: every number below is a
documented reference value at the triple point (0 degC, 1 atm) for fresh-water
ice Ih in contact with structural steel.  Where a property is temperature
dependent the accessor says so; nothing else is silently extrapolated.

References for the default set
------------------------------
* IAPWS / CRC Handbook of Chemistry and Physics: fusion enthalpy 333.55 kJ/kg,
  densities of ice Ih (916.8 kg/m^3) and water at 0 degC (999.8 kg/m^3),
  specific heats (ice 2050, water 4186 J/kg/K), conductivities
  (ice 2.22, water 0.561 W/m/K).
* Water viscosity correlation (Vogel form, standard engineering fit):
  mu = 2.414e-5 * 10**(247.8/(T - 140)) Pa/s, accurate to ~1 % over
  273-373 K.
* Structural steel (low-carbon): rho 7850 kg/m^3, c 475 J/kg/K, k 45 W/m/K.
"""

from dataclasses import dataclass
from math import sqrt

__all__ = [
    "IceWaterProperties",
    "StructureProperties",
    "water_viscosity",
]


def water_viscosity(temperature: float) -> float:
    """Dynamic viscosity of liquid water [Pa s] (Vogel correlation).

    ``temperature`` is in kelvin.  Valid over the liquid range; the ice-contact
    model never evaluates it outside ~250-290 K.
    """
    if temperature <= 140.0:
        raise ValueError("water_viscosity requires temperature > 140 K")
    return 2.414e-5 * 10.0 ** (247.8 / (temperature - 140.0))


@dataclass(frozen=True)
class IceWaterProperties:
    """Thermophysical constants for ice Ih / meltwater."""

    melting_temperature: float = 273.15          # K, at reference pressure
    latent_heat_fusion: float = 3.3355e5          # J/kg
    rho_ice: float = 916.8                        # kg/m^3
    rho_water: float = 999.8                      # kg/m^3
    c_ice: float = 2050.0                         # J/kg/K
    c_water: float = 4186.0                       # J/kg/K
    k_ice: float = 2.22                           # W/m/K
    k_water: float = 0.561                        # W/m/K
    surface_tension_water: float = 0.0757         # N/m (against vapour)
    viscosity_reference_temperature: float = 273.15  # K, for the Vogel fit

    def __post_init__(self) -> None:
        fields = {
            "melting_temperature": self.melting_temperature,
            "latent_heat_fusion": self.latent_heat_fusion,
            "rho_ice": self.rho_ice,
            "rho_water": self.rho_water,
            "c_ice": self.c_ice,
            "c_water": self.c_water,
            "k_ice": self.k_ice,
            "k_water": self.k_water,
            "surface_tension_water": self.surface_tension_water,
            "viscosity_reference_temperature": self.viscosity_reference_temperature,
        }
        for name, value in fields.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a real number")
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

    @property
    def alpha_ice(self) -> float:
        """Thermal diffusivity of ice [m^2/s]."""
        return self.k_ice / (self.rho_ice * self.c_ice)

    @property
    def alpha_water(self) -> float:
        """Thermal diffusivity of meltwater [m^2/s]."""
        return self.k_water / (self.rho_water * self.c_water)

    @property
    def specific_volume_change(self) -> float:
        """Specific volume change on fusion, ``v_water - v_ice`` [m^3/kg].

        Negative: melting contracts.  This sign is what makes
        dT_melt/dp negative in the Clausius-Clapeyron relation.
        """
        return 1.0 / self.rho_water - 1.0 / self.rho_ice

    @property
    def latent_heat_per_volume_ice(self) -> float:
        """Latent heat per unit volume of ice [J/m^3]."""
        return self.rho_ice * self.latent_heat_fusion

    def viscosity(self, temperature: float) -> float:
        """Meltwater dynamic viscosity [Pa s] at ``temperature`` (kelvin)."""
        return water_viscosity(temperature)

    def thermal_effusivity_ice(self) -> float:
        """Thermal effusivity of ice, ``sqrt(k rho c)`` [J/(m^2 K s^0.5)]."""
        return sqrt(self.k_ice * self.rho_ice * self.c_ice)


@dataclass(frozen=True)
class StructureProperties:
    """Thermophysical constants for the structural counterface (steel)."""

    rho: float = 7850.0        # kg/m^3
    c: float = 475.0           # J/kg/K
    k: float = 45.0            # W/m/K
    name: str = "carbon_steel"

    def __post_init__(self) -> None:
        for name in ("rho", "c", "k"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a real number")
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

    @property
    def alpha(self) -> float:
        """Thermal diffusivity of the structure [m^2/s]."""
        return self.k / (self.rho * self.c)

    def thermal_effusivity(self) -> float:
        """Thermal effusivity of the structure [J/(m^2 K s^0.5)]."""
        return sqrt(self.k * self.rho * self.c)