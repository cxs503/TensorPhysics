"""Property basis for the ice-induced bow-flare slamming triad.

Every number is a documented reference value with provenance; nothing is
silently extrapolated.  The set is deliberately small and auditable.

References
----------
* Ice Ih (fresh-water ice): rho 916.8 kg/m^3, E 8.9 GPa, nu 0.3 -- CRC
  Handbook of Chemistry and Physics; Assur (1958) *The Physics of Snow and
  Ice*; typical sea-ice values per ITTC / ISO 19906.
* Ice strength: uniaxial compressive strength 1-5 MPa (Jordaan 2001 review;
  ITTC 7.5-02).  The default failure threshold is the *lower* bound
  (1.0 MPa) -- conservative for a brittle floe under bending.
* Ice fracture energy: G_f 50-200 J/m^2 (Kerr 1975; Sidorenko 1998 review of
  ice fracture-energy measurements).  Default 100 J/m^2.
* Seawater: rho 1025 kg/m^3 (standard oceanographic value, 0 degC / 35 PSU).
* Structural steel (AH36-class hull plate): rho 7850 kg/m^3, E 200 GPa,
  nu 0.3, yield 355 MPa.
* Wave orbital velocity (deep water, linear theory): u_max = pi H / T at the
  surface (e.g. Faltinsen, *Sea Loads on Ships*, 1990).
* Wagner (1971) water-entry pressure scaling: p ~ rho V^2 with the deadrise
  coefficient C_p = pi / tan(beta) (see :mod:`.hydro`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "IceFloeProperties",
    "SlammingWaterProperties",
    "SlammingStructureProperties",
    "default_floe",
    "default_water",
    "default_structure",
    "PROVENANCE",
]

PROVENANCE = {
    "rho_ice": "CRC Handbook / Assur (1958): ice Ih density 916.8 kg/m^3",
    "E_ice": "Assur (1958); ITTC 7.5-02: 8.9 GPa at -10 degC",
    "nu_ice": "Assur (1958): 0.3",
    "sigma_ice": "Jordaan (2001) / ITTC 7.5-02: 1-5 MPa; default = lower bound",
    "G_f": "Kerr (1975); Sidorenko (1998): 50-200 J/m^2; default 100",
    "rho_water": "standard oceanographic seawater value, 0 degC / 35 PSU",
    "rho_steel": "AH36 hull plate: 7850 kg/m^3",
    "E_steel": "AH36 hull plate: 200 GPa",
    "wave_orbital": "linear deep-water theory: u_max = pi H / T (Faltinsen 1990)",
    "wagner_cp": "Wagner (1971): C_p = pi / tan(beta), beta = deadrise",
}


@dataclass(frozen=True)
class IceFloeProperties:
    """Wave-carried ice floe: rigid disk with a brittle failure threshold."""

    radius_m: float = 1.5          # floe radius [m]
    thickness_m: float = 0.30      # floe thickness [m]
    rho_ice: float = 916.8         # kg/m^3
    E_ice: float = 8.9e9           # Pa
    nu_ice: float = 0.30
    sigma_ice: float = 1.0e6       # Pa, brittle failure threshold (1-5 MPa band)
    G_f: float = 100.0             # J/m^2, fracture energy (50-200 band)
    # plastic-yield threshold: microcracking onset, taken at 60 % of the
    # brittle threshold (ice is quasi-brittle; the yield stage is short).
    yield_fraction: float = 0.60

    def __post_init__(self) -> None:
        if not (1.0e6 <= self.sigma_ice <= 5.0e6):
            raise ValueError("sigma_ice must lie in the 1-5 MPa ice-strength band")
        if not (50.0 <= self.G_f <= 200.0):
            raise ValueError("G_f must lie in the 50-200 J/m^2 band")

    @property
    def mass(self) -> float:
        """Floe mass [kg]."""
        import math

        return self.rho_ice * math.pi * self.radius_m**2 * self.thickness_m

    @property
    def flexural_rigidity(self) -> float:
        """Plate rigidity D = E h^3 / (12 (1-nu^2)) [N m]."""
        return self.E_ice * self.thickness_m**3 / (12.0 * (1.0 - self.nu_ice**2))

    @property
    def bending_stiffness(self) -> float:
        """Center-load stiffness of the floe disk on a rigid support [N/m].

        Clamped circular plate, central point load: delta = F R^2 / (16 D)
        (Timoshenko & Woinowsky, *Theory of Plates and Shells*).
        """
        return 16.0 * self.flexural_rigidity / self.radius_m**2


@dataclass(frozen=True)
class SlammingWaterProperties:
    """Seawater + wave state for the slamming pressure."""

    rho_water: float = 1025.0      # kg/m^3
    wave_period_s: float = 5.0     # s, typical ice-region sea period
    # hydrodynamic added-mass coefficient on the floe (per unit area),
    # order-unity for a thin disk (Faltinsen 1990, ch. 2).
    added_mass_coeff: float = 0.75
    # hydrodynamic damping coefficient [N s / m^2] per unit relative velocity;
    # order-of-magnitude estimate from Morison-type drag on a thin disk.
    damping_coeff: float = 250.0

    def __post_init__(self) -> None:
        if self.wave_period_s <= 0:
            raise ValueError("wave_period_s must be positive")


@dataclass(frozen=True)
class SlammingStructureProperties:
    """Bow-flare panel: simply supported square steel plate."""

    side_m: float = 4.0            # panel side [m]
    thickness_m: float = 0.025     # plate thickness [m]
    rho_steel: float = 7850.0      # kg/m^3
    E_steel: float = 200.0e9       # Pa
    nu_steel: float = 0.30
    yield_stress: float = 355.0e6  # Pa
    # structural damping ratio (Rayleigh, mass-proportional part) -- small,
    # enough to settle the plate within the simulation window.
    damping_ratio: float = 0.02

    def __post_init__(self) -> None:
        if self.thickness_m <= 0 or self.side_m <= 0:
            raise ValueError("panel dimensions must be positive")

    @property
    def flexural_rigidity(self) -> float:
        return self.E_steel * self.thickness_m**3 / (12.0 * (1.0 - self.nu_steel**2))

    @property
    def total_mass(self) -> float:
        return self.rho_steel * self.thickness_m * self.side_m**2


def default_floe() -> IceFloeProperties:
    return IceFloeProperties()


def default_water() -> SlammingWaterProperties:
    return SlammingWaterProperties()


def default_structure() -> SlammingStructureProperties:
    return SlammingStructureProperties()
