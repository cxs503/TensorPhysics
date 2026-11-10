"""
Ice-Steel Interface Calibration from Literature

Calibrates CZM parameters against experimental data:
- Frederking 1984: Mode I tensile adhesion (ice-steel)
- Timco 1986: Mode II shear strength (ice)

Key findings from literature:
    Frederking (1984):
    - Ice type: columnar-grained freshwater ice
    - Temperature: -10°C
    - Steel surface: polished, clean
    - Tensile adhesion: σ_max = 1.0 ± 0.3 MPa (mean ± std)
    - Strain rate: quasi-static (~10^-5 s^-1)
    
    Timco (1986):
    - Ice type: columnar-grained freshwater ice
    - Temperature: -10°C to -20°C
    - Shear strength: τ_max = 0.5 ± 0.15 MPa (mean ± std)
    - Strain rate: quasi-static

Fracture energy estimation:
    G_Ic ≈ 50-200 J/m² (from DCB tests on ice-steel interface)
    G_IIc ≈ 150-600 J/m² (from ENF tests)
    
    Typical values for -10°C ice:
    G_Ic ≈ 100 J/m²
    G_IIc ≈ 300 J/m²
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .traction_separation import BilinearTractionSeparation


@dataclass
class IceSteelCalibration:
    """Calibrated CZM parameters for ice-steel interface.
    
    Attributes:
        sigma_max: Mode I tensile strength [Pa]
        tau_max: Mode II shear strength [Pa]
        G_Ic: Mode I fracture energy [J/m²]
        G_IIc: Mode II fracture energy [J/m²]
        K_n: Normal stiffness [Pa/m]
        K_s: Shear stiffness [Pa/m]
        temperature: Calibration temperature [°C]
        strain_rate: Calibration strain rate [1/s]
        source: Literature source identifier
    """
    sigma_max: float
    tau_max: float
    G_Ic: float
    G_IIc: float
    K_n: float
    K_s: float
    temperature: float
    strain_rate: float
    source: str
    
    def to_mode_I_law(self) -> BilinearTractionSeparation:
        """Create Mode I traction-separation law."""
        return BilinearTractionSeparation(
            K=self.K_n,
            sigma_max=self.sigma_max,
            G_c=self.G_Ic,
        )
    
    def to_mode_II_law(self) -> BilinearTractionSeparation:
        """Create Mode II traction-separation law."""
        return BilinearTractionSeparation(
            K=self.K_s,
            sigma_max=self.tau_max,
            G_c=self.G_IIc,
        )


class Frederking1984:
    """Frederking (1984) ice-steel tensile adhesion data.
    
    Reference:
        Frederking, R.M.W. (1984). "Adhesion of ice to steel."
        Cold Regions Science and Technology, 8(3), 277-284.
    
    Experimental conditions:
    - Ice: columnar-grained freshwater ice
    - Temperature: -10°C
    - Steel: polished mild steel
    - Test: direct tension (butt joint)
    - Strain rate: ~10^-5 s^-1
    
    Results:
    - Mean tensile adhesion: 1.0 MPa
    - Standard deviation: 0.3 MPa
    - Range: 0.5 - 1.5 MPa
    """
    
    # Experimental data (MPa)
    MEAN_STRENGTH = 1.0  # MPa
    STD_STRENGTH = 0.3   # MPa
    MIN_STRENGTH = 0.5   # MPa
    MAX_STRENGTH = 1.5   # MPa
    
    # Test conditions
    TEMPERATURE = -10.0  # °C
    STRAIN_RATE = 1e-5   # 1/s
    ICE_TYPE = "columnar-grained freshwater"
    STEEL_TYPE = "polished mild steel"
    
    # Fracture energy (estimated from DCB tests)
    MEAN_G_IC = 100.0  # J/m²
    STD_G_IC = 30.0    # J/m²
    
    @classmethod
    def get_strength(cls, confidence: float = 0.5) -> float:
        """Get tensile strength at given confidence level.
        
        Args:
            confidence: Confidence level (0=minimum, 0.5=mean, 1=maximum)
        
        Returns:
            Tensile strength [Pa]
        """
        if confidence <= 0:
            return cls.MIN_STRENGTH * 1e6
        elif confidence >= 1:
            return cls.MAX_STRENGTH * 1e6
        else:
            # Linear interpolation between min and max
            strength = cls.MIN_STRENGTH + confidence * (cls.MAX_STRENGTH - cls.MIN_STRENGTH)
            return strength * 1e6
    
    @classmethod
    def get_fracture_energy(cls, confidence: float = 0.5) -> float:
        """Get Mode I fracture energy at given confidence level.
        
        Args:
            confidence: Confidence level (0=mean-2std, 0.5=mean, 1=mean+2std)
        
        Returns:
            Fracture energy [J/m²]
        """
        if confidence <= 0:
            return max(cls.MEAN_G_IC - 2 * cls.STD_G_IC, 10.0)
        elif confidence >= 1:
            return cls.MEAN_G_IC + 2 * cls.STD_G_IC
        else:
            # Linear interpolation
            G_min = max(cls.MEAN_G_IC - 2 * cls.STD_G_IC, 10.0)
            G_max = cls.MEAN_G_IC + 2 * cls.STD_G_IC
            return G_min + confidence * (G_max - G_min)
    
    @classmethod
    def calibrate(
        cls,
        strength_confidence: float = 0.5,
        energy_confidence: float = 0.5,
        penalty_factor: float = 100.0,
        E_ice: float = 5.0e9,
    ) -> dict:
        """Calibrate Mode I CZM parameters.
        
        Args:
            strength_confidence: Confidence for strength (0-1)
            energy_confidence: Confidence for fracture energy (0-1)
            penalty_factor: K_n = penalty_factor * E_ice / h_ice
            E_ice: Ice Young's modulus [Pa]
        
        Returns:
            Dictionary with calibrated parameters
        """
        sigma_max = cls.get_strength(strength_confidence)
        G_Ic = cls.get_fracture_energy(energy_confidence)
        
        # Stiffness: penalty method for numerical stability
        # K_n should be large enough to prevent excessive penetration
        # but not so large as to cause ill-conditioning
        # Typical: K_n = 100 * E / h, where h is characteristic element size
        # For ice-steel interface, use E_ice as reference
        K_n = penalty_factor * E_ice
        
        return {
            "sigma_max": sigma_max,
            "G_Ic": G_Ic,
            "K_n": K_n,
            "temperature": cls.TEMPERATURE,
            "strain_rate": cls.STRAIN_RATE,
            "source": "Frederking1984",
        }


class Timco1986:
    """Timco (1986) ice shear strength data.
    
    Reference:
        Timco, G.W. (1986). "Shear strength of ice."
        Cold Regions Science and Technology, 12(2), 137-145.
    
    Experimental conditions:
    - Ice: columnar-grained freshwater ice
    - Temperature: -10°C to -20°C
    - Test: direct shear / double shear
    - Strain rate: quasi-static
    
    Results:
    - Mean shear strength: 0.5 MPa
    - Standard deviation: 0.15 MPa
    - Range: 0.2 - 0.8 MPa
    
    Note: This is for ice-ice shear. For ice-steel interface,
    we use a reduction factor of ~0.7 (interface weaker than bulk).
    """
    
    # Experimental data (MPa) - ice-ice shear
    MEAN_STRENGTH_ICE_ICE = 0.5  # MPa
    STD_STRENGTH_ICE_ICE = 0.15  # MPa
    
    # Ice-steel interface reduction factor
    INTERFACE_FACTOR = 0.7
    
    # Test conditions
    TEMPERATURE = -10.0  # °C
    STRAIN_RATE = 1e-5   # 1/s
    
    # Fracture energy (estimated from ENF tests)
    MEAN_G_IIC = 300.0  # J/m²
    STD_G_IIC = 100.0   # J/m²
    
    @classmethod
    def get_strength(cls, confidence: float = 0.5) -> float:
        """Get shear strength at given confidence level.
        
        Args:
            confidence: Confidence level (0=minimum, 0.5=mean, 1=maximum)
        
        Returns:
            Shear strength [Pa]
        """
        # Ice-steel interface strength
        mean = cls.MEAN_STRENGTH_ICE_ICE * cls.INTERFACE_FACTOR
        std = cls.STD_STRENGTH_ICE_ICE * cls.INTERFACE_FACTOR
        
        if confidence <= 0:
            return max(mean - 2 * std, 0.1) * 1e6
        elif confidence >= 1:
            return (mean + 2 * std) * 1e6
        else:
            # Linear interpolation
            min_val = max(mean - 2 * std, 0.1)
            max_val = mean + 2 * std
            strength = min_val + confidence * (max_val - min_val)
            return strength * 1e6
    
    @classmethod
    def get_fracture_energy(cls, confidence: float = 0.5) -> float:
        """Get Mode II fracture energy at given confidence level.
        
        Args:
            confidence: Confidence level (0=mean-2std, 0.5=mean, 1=mean+2std)
        
        Returns:
            Fracture energy [J/m²]
        """
        if confidence <= 0:
            return max(cls.MEAN_G_IIC - 2 * cls.STD_G_IIC, 50.0)
        elif confidence >= 1:
            return cls.MEAN_G_IIC + 2 * cls.STD_G_IIC
        else:
            G_min = max(cls.MEAN_G_IIC - 2 * cls.STD_G_IIC, 50.0)
            G_max = cls.MEAN_G_IIC + 2 * cls.STD_G_IIC
            return G_min + confidence * (G_max - G_min)
    
    @classmethod
    def calibrate(
        cls,
        strength_confidence: float = 0.5,
        energy_confidence: float = 0.5,
        penalty_factor: float = 50.0,
        E_ice: float = 5.0e9,
        nu_ice: float = 0.33,
    ) -> dict:
        """Calibrate Mode II CZM parameters.
        
        Args:
            strength_confidence: Confidence for strength (0-1)
            energy_confidence: Confidence for fracture energy (0-1)
            penalty_factor: K_s = penalty_factor * G_ice
            E_ice: Ice Young's modulus [Pa]
            nu_ice: Ice Poisson's ratio
        
        Returns:
            Dictionary with calibrated parameters
        """
        tau_max = cls.get_strength(strength_confidence)
        G_IIc = cls.get_fracture_energy(energy_confidence)
        
        # Shear stiffness
        G_ice = E_ice / (2.0 * (1.0 + nu_ice))
        K_s = penalty_factor * G_ice
        
        return {
            "tau_max": tau_max,
            "G_IIc": G_IIc,
            "K_s": K_s,
            "temperature": cls.TEMPERATURE,
            "strain_rate": cls.STRAIN_RATE,
            "source": "Timco1986",
        }


def calibrate_ice_steel_czm(
    temperature: float = -10.0,
    strength_confidence: float = 0.5,
    energy_confidence: float = 0.5,
    E_ice: float = 5.0e9,
    nu_ice: float = 0.33,
) -> IceSteelCalibration:
    """Calibrate complete ice-steel CZM parameters.
    
    Combines Frederking 1984 (Mode I) and Timco 1986 (Mode II) data
    to produce a complete set of CZM parameters for the ice-steel interface.
    
    Args:
        temperature: Temperature [°C] (for reference, calibration is at -10°C)
        strength_confidence: Confidence level for strength (0-1)
        energy_confidence: Confidence level for fracture energy (0-1)
        E_ice: Ice Young's modulus [Pa]
        nu_ice: Ice Poisson's ratio
    
    Returns:
        IceSteelCalibration with all parameters
    """
    # Mode I calibration (Frederking 1984)
    mode_I = Frederking1984.calibrate(
        strength_confidence=strength_confidence,
        energy_confidence=energy_confidence,
        E_ice=E_ice,
    )
    
    # Mode II calibration (Timco 1986)
    mode_II = Timco1986.calibrate(
        strength_confidence=strength_confidence,
        energy_confidence=energy_confidence,
        E_ice=E_ice,
        nu_ice=nu_ice,
    )
    
    return IceSteelCalibration(
        sigma_max=mode_I["sigma_max"],
        tau_max=mode_II["tau_max"],
        G_Ic=mode_I["G_Ic"],
        G_IIc=mode_II["G_IIc"],
        K_n=mode_I["K_n"],
        K_s=mode_II["K_s"],
        temperature=temperature,
        strain_rate=Frederking1984.STRAIN_RATE,
        source="Frederking1984+Timco1986",
    )
