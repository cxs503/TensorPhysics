"""
TensorDEM Adfreeze Bond Mapping

Maps the ice-steel CZM to TensorDEM's adfreeze bond model for
discrete element simulations of ice-structure interaction.

TensorDEM adfreeze bond model:
    - Bonds form when ice particle contacts steel surface
    - Bond strength evolves with time (creep sintering)
    - Bond breaks when traction exceeds strength
    - Broken bonds cannot reform (irreversible)

The mapping:
    CZM σ_max → Bond normal strength F_n_max = σ_max * A_bond
    CZM τ_max → Bond shear strength F_s_max = τ_max * A_bond
    CZM G_c → Bond fracture energy G_bond = G_c * A_bond
    
where A_bond is the bond cross-sectional area (related to particle size).

Cross-library contract:
    TensorPhysics (CZM) → TensorDEM (adfreeze bond)
    
    The CZM provides:
    - Constitutive law (traction-separation)
    - Material parameters (strength, toughness)
    - Mixed-mode criterion (B-K)
    
    TensorDEM uses:
    - Force-displacement law (discrete bonds)
    - Bond strength (force-based)
    - Contact detection (geometric)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .calibration import IceSteelCalibration, calibrate_ice_steel_czm
from .bk_mixed_mode import MixedModeCZM


@dataclass
class AdfreezeBondState:
    """State of an adfreeze bond in TensorDEM.
    
    Attributes:
        bond_id: Bond identifier
        active: Whether bond is active (not broken)
        age: Bond age [s] (time since formation)
        delta_n: Normal displacement [m]
        delta_s: Shear displacement [m]
        F_n: Normal force [N]
        F_s: Shear force [N]
        damage: Damage variable D ∈ [0, 1]
        strength_reduction: Strength reduction factor (creep, temperature)
    """
    bond_id: int = 0
    active: bool = True
    age: float = 0.0
    delta_n: float = 0.0
    delta_s: float = 0.0
    F_n: float = 0.0
    F_s: float = 0.0
    damage: float = 0.0
    strength_reduction: float = 1.0


@dataclass
class AdfreezeBondMapper:
    """Mapper from ice-steel CZM to TensorDEM adfreeze bond model.
    
    Converts continuum CZM parameters to discrete bond parameters
    for use in DEM simulations.
    
    The mapping ensures:
    - Energy consistency: bond fracture energy = CZM G_c * A_bond
    - Strength consistency: bond strength = CZM σ_max * A_bond
    - Mixed-mode consistency: B-K criterion applies to both
    
    Attributes:
        calibration: Ice-steel CZM calibration
        bond_area: Bond cross-sectional area [m²]
        characteristic_length: Characteristic length for scaling [m]
    """
    calibration: IceSteelCalibration
    bond_area: float
    characteristic_length: float = 1e-3  # 1 mm default
    
    def __post_init__(self):
        """Validate parameters."""
        if self.bond_area <= 0:
            raise ValueError(f"bond_area must be positive, got {self.bond_area}")
        if self.characteristic_length <= 0:
            raise ValueError(f"characteristic_length must be positive, got {self.characteristic_length}")
    
    @classmethod
    def from_particle_size(
        cls,
        particle_radius: float,
        calibration: Optional[IceSteelCalibration] = None,
        temperature: float = -10.0,
    ) -> "AdfreezeBondMapper":
        """Create mapper from particle size.
        
        Args:
            particle_radius: Ice particle radius [m]
            calibration: CZM calibration (default: standard ice-steel)
            temperature: Temperature [°C]
        
        Returns:
            AdfreezeBondMapper configured for given particle size
        """
        if calibration is None:
            calibration = calibrate_ice_steel_czm(temperature=temperature)
        
        # Bond area: circular cross-section with radius = particle_radius / 2
        bond_radius = particle_radius / 2.0
        bond_area = np.pi * bond_radius**2
        
        return cls(
            calibration=calibration,
            bond_area=bond_area,
            characteristic_length=particle_radius,
        )
    
    def get_bond_strengths(self) -> dict[str, float]:
        """Get bond force strengths.
        
        Returns:
            Dictionary with:
                F_n_max: Maximum normal force [N]
                F_s_max: Maximum shear force [N]
                G_I_bond: Mode I bond fracture energy [J]
                G_II_bond: Mode II bond fracture energy [J]
        """
        cal = self.calibration
        
        return {
            "F_n_max": cal.sigma_max * self.bond_area,
            "F_s_max": cal.tau_max * self.bond_area,
            "G_I_bond": cal.G_Ic * self.bond_area,
            "G_II_bond": cal.G_IIc * self.bond_area,
        }
    
    def get_bond_stiffness(self) -> dict[str, float]:
        """Get bond stiffness parameters.
        
        Returns:
            Dictionary with:
                K_n_bond: Normal bond stiffness [N/m]
                K_s_bond: Shear bond stiffness [N/m]
        """
        cal = self.calibration
        
        # Scale stiffness by bond area
        K_n_bond = cal.K_n * self.bond_area
        K_s_bond = cal.K_s * self.bond_area
        
        return {
            "K_n_bond": K_n_bond,
            "K_s_bond": K_s_bond,
        }
    
    def get_dem_parameters(self) -> dict:
        """Get complete set of DEM bond parameters.
        
        Returns:
            Dictionary with all parameters needed by TensorDEM
        """
        strengths = self.get_bond_strengths()
        stiffness = self.get_bond_stiffness()
        
        return {
            **strengths,
            **stiffness,
            "bond_area": self.bond_area,
            "characteristic_length": self.characteristic_length,
            "eta_bk": 1.45,  # B-K exponent
            "temperature": self.calibration.temperature,
            "source": self.calibration.source,
        }
    
    def create_czm_for_bond(self) -> MixedModeCZM:
        """Create a CZM instance configured for this bond.
        
        Returns:
            MixedModeCZM with bond-scaled parameters
        """
        cal = self.calibration
        
        return MixedModeCZM(
            K_n=cal.K_n,
            K_s=cal.K_s,
            sigma_max=cal.sigma_max,
            tau_max=cal.tau_max,
            G_Ic=cal.G_Ic,
            G_IIc=cal.G_IIc,
            eta=1.45,
        )
    
    def compute_bond_force(
        self,
        delta_n: float,
        delta_s: float,
        bond_age: float = 0.0,
        temperature_factor: float = 1.0,
    ) -> dict:
        """Compute bond force for given displacements.
        
        This method bridges the CZM (continuum) to DEM (discrete) by
        computing forces from separations using the calibrated law.
        
        Args:
            delta_n: Normal displacement [m]
            delta_s: Shear displacement [m]
            bond_age: Bond age [s] (for creep effects)
            temperature_factor: Temperature reduction factor (0-1)
        
        Returns:
            Dictionary with:
                F_n: Normal force [N]
                F_s: Shear force [N]
                damage: Damage variable
                active: Whether bond is still active
        """
        # Create CZM and compute traction
        czm = self.create_czm_for_bond()
        result = czm.compute_traction(delta_n, delta_s)
        
        # Convert traction to force
        F_n = result["sigma_n"] * self.bond_area
        F_s = result["tau_s"] * self.bond_area
        
        # Apply temperature and age effects (simplified)
        # In reality, these would affect strength evolution
        strength_factor = temperature_factor
        
        return {
            "F_n": F_n * strength_factor,
            "F_s": F_s * strength_factor,
            "damage": result["damage"],
            "active": not result["failed"],
            "G_I": result["G_I"] * self.bond_area,
            "G_II": result["G_II"] * self.bond_area,
        }


class AdfreezeBondNetwork:
    """Network of adfreeze bonds for ice-structure interface.
    
    Manages multiple bonds and their evolution during DEM simulation.
    """
    
    def __init__(self, mapper: AdfreezeBondMapper):
        """Initialize bond network.
        
        Args:
            mapper: Adfreeze bond mapper
        """
        self.mapper = mapper
        self.bonds: dict[int, AdfreezeBondState] = {}
        self._next_bond_id = 0
    
    def create_bond(
        self,
        initial_strength_factor: float = 1.0,
    ) -> AdfreezeBondState:
        """Create a new adfreeze bond.
        
        Args:
            initial_strength_factor: Initial strength reduction (e.g., for imperfect contact)
        
        Returns:
            New AdfreezeBondState
        """
        bond_id = self._next_bond_id
        self._next_bond_id += 1
        
        bond = AdfreezeBondState(
            bond_id=bond_id,
            active=True,
            age=0.0,
            strength_reduction=initial_strength_factor,
        )
        
        self.bonds[bond_id] = bond
        return bond
    
    def update_bond(
        self,
        bond_id: int,
        delta_n: float,
        delta_s: float,
        dt: float,
    ) -> AdfreezeBondState:
        """Update bond state for new displacements.
        
        Args:
            bond_id: Bond identifier
            delta_n: Normal displacement [m]
            delta_s: Shear displacement [m]
            dt: Time step [s]
        
        Returns:
            Updated AdfreezeBondState
        """
        if bond_id not in self.bonds:
            raise ValueError(f"Bond {bond_id} not found")
        
        bond = self.bonds[bond_id]
        
        if not bond.active:
            return bond  # Bond already broken
        
        # Update age
        bond.age += dt
        
        # Update displacements
        bond.delta_n = delta_n
        bond.delta_s = delta_s
        
        # Compute force
        result = self.mapper.compute_bond_force(
            delta_n, delta_s,
            bond_age=bond.age,
            temperature_factor=bond.strength_reduction,
        )
        
        # Update state
        bond.F_n = result["F_n"]
        bond.F_s = result["F_s"]
        bond.damage = result["damage"]
        bond.active = result["active"]
        
        return bond
    
    def get_active_bonds(self) -> list[AdfreezeBondState]:
        """Get list of active (unbroken) bonds.
        
        Returns:
            List of active bonds
        """
        return [b for b in self.bonds.values() if b.active]
    
    def get_broken_bonds(self) -> list[AdfreezeBondState]:
        """Get list of broken bonds.
        
        Returns:
            List of broken bonds
        """
        return [b for b in self.bonds.values() if not b.active]
    
    def total_force(self) -> tuple[float, float]:
        """Compute total force from all active bonds.
        
        Returns:
            Tuple of (F_n_total, F_s_total) [N]
        """
        F_n_total = sum(b.F_n for b in self.bonds.values() if b.active)
        F_s_total = sum(b.F_s for b in self.bonds.values() if b.active)
        return F_n_total, F_s_total
    
    def remove_bond(self, bond_id: int):
        """Remove a bond from the network.
        
        Args:
            bond_id: Bond identifier
        """
        if bond_id in self.bonds:
            del self.bonds[bond_id]
