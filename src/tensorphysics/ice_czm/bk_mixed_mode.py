"""
Benzeggagh-Kenane (B-K) Mixed Mode Criterion

The B-K criterion provides an empirical law for mixed-mode fracture:
    G_c(ψ) = G_Ic + (G_IIc - G_Ic) * ψ^η

where:
    G_c(ψ) = effective fracture toughness at mode mixity ψ
    G_Ic = Mode I fracture toughness [J/m²]
    G_IIc = Mode II fracture toughness [J/m²]
    ψ = G_II / (G_I + G_II) = mode mixity ratio
    η = material exponent (typically 1.5-2.0 for composites, 1.0-1.5 for ice)

For ice-steel interface:
    η ≈ 1.2-1.5 (based on limited experimental data)
    Default: η = 1.45 (conservative estimate)

The criterion satisfies:
    ψ = 0 → G_c = G_Ic (pure Mode I)
    ψ = 1 → G_c = G_IIc (pure Mode II)
    0 < ψ < 1 → G_Ic < G_c < G_IIc (mixed mode)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .traction_separation import BilinearTractionSeparation, TractionSeparationState


@dataclass
class BenzeggaghKenaneCriterion:
    """Benzeggagh-Kenane mixed mode fracture criterion.
    
    Attributes:
        G_Ic: Mode I fracture toughness [J/m²]
        G_IIc: Mode II fracture toughness [J/m²]
        eta: B-K exponent (dimensionless)
    """
    G_Ic: float
    G_IIc: float
    eta: float = 1.45
    
    def __post_init__(self):
        """Validate parameters."""
        if self.G_Ic <= 0:
            raise ValueError(f"G_Ic must be positive, got {self.G_Ic}")
        if self.G_IIc <= 0:
            raise ValueError(f"G_IIc must be positive, got {self.G_IIc}")
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}")
    
    def effective_toughness(self, psi: float) -> float:
        """Compute effective fracture toughness at given mode mixity.
        
        Args:
            psi: Mode mixity ratio ψ = G_II / (G_I + G_II) ∈ [0, 1]
        
        Returns:
            Effective fracture toughness G_c(ψ) [J/m²]
        """
        psi = np.clip(psi, 0.0, 1.0)
        G_c = self.G_Ic + (self.G_IIc - self.G_Ic) * psi**self.eta
        return float(G_c)
    
    def critical_energy_release_rate(
        self,
        G_I: float,
        G_II: float,
    ) -> Tuple[float, float, bool]:
        """Check if fracture criterion is met.
        
        Args:
            G_I: Mode I energy release rate [J/m²]
            G_II: Mode II energy release rate [J/m²]
        
        Returns:
            Tuple of (G_c, G_total, is_critical):
                G_c: Critical toughness at current mixity [J/m²]
                G_total: Total energy release rate G_I + G_II [J/m²]
                is_critical: True if G_total ≥ G_c (fracture propagates)
        """
        G_total = G_I + G_II
        
        if G_total < 1e-30:
            return self.G_Ic, 0.0, False
        
        psi = G_II / G_total
        G_c = self.effective_toughness(psi)
        is_critical = G_total >= G_c
        
        return G_c, G_total, is_critical
    
    def mode_mixity(self, G_I: float, G_II: float) -> float:
        """Compute mode mixity ratio.
        
        Args:
            G_I: Mode I energy release rate [J/m²]
            G_II: Mode II energy release rate [J/m²]
        
        Returns:
            Mode mixity ψ = G_II / (G_I + G_II) ∈ [0, 1]
        """
        G_total = G_I + G_II
        if G_total < 1e-30:
            return 0.0
        return float(np.clip(G_II / G_total, 0.0, 1.0))
    
    def __repr__(self) -> str:
        return (
            f"BenzeggaghKenaneCriterion(G_Ic={self.G_Ic:.3e}, "
            f"G_IIc={self.G_IIc:.3e}, eta={self.eta})"
        )


class MixedModeCZM:
    """Mixed-mode cohesive zone model with B-K criterion.
    
    Combines:
    - Mode I traction-separation law (normal opening)
    - Mode II traction-separation law (shear sliding)
    - B-K criterion for mixed-mode damage evolution
    
    The model uses an effective displacement measure to couple the modes:
        δ_eff = sqrt(<δ_n>² + δ_s²)
    
    where <x> = max(x, 0) is the Macaulay bracket (no compression damage).
    
    Damage initiation: quadratic stress criterion
        (σ_n / σ_max)² + (τ_s / τ_max)² = 1
    
    Damage evolution: B-K criterion with effective displacement
    """
    
    def __init__(
        self,
        K_n: float,
        K_s: float,
        sigma_max: float,
        tau_max: float,
        G_Ic: float,
        G_IIc: float,
        eta: float = 1.45,
    ):
        """Initialize mixed-mode CZM.
        
        Args:
            K_n: Normal stiffness [Pa/m]
            K_s: Shear stiffness [Pa/m]
            sigma_max: Mode I strength [Pa]
            tau_max: Mode II strength [Pa]
            G_Ic: Mode I fracture energy [J/m²]
            G_IIc: Mode II fracture energy [J/m²]
            eta: B-K exponent
        """
        self.K_n = K_n
        self.K_s = K_s
        self.sigma_max = sigma_max
        self.tau_max = tau_max
        self.G_Ic = G_Ic
        self.G_IIc = G_IIc
        
        # Mode I and II laws
        self.mode_I = BilinearTractionSeparation(K_n, sigma_max, G_Ic)
        self.mode_II = BilinearTractionSeparation(K_s, tau_max, G_IIc)
        
        # B-K criterion
        self.bk = BenzeggaghKenaneCriterion(G_Ic, G_IIc, eta)
        
        # Critical separations
        self.delta_n0 = sigma_max / K_n  # Mode I damage onset
        self.delta_s0 = tau_max / K_s    # Mode II damage onset
        self.delta_nf = 2.0 * G_Ic / sigma_max  # Mode I complete failure
        self.delta_sf = 2.0 * G_IIc / tau_max   # Mode II complete failure
    
    def compute_traction(
        self,
        delta_n: float,
        delta_s: float,
        delta_max_prev: float = 0.0,
        damage_prev: float = 0.0,
    ) -> dict:
        """Compute mixed-mode traction for given separations.
        
        Args:
            delta_n: Normal separation [m] (positive = opening)
            delta_s: Shear separation [m]
            delta_max_prev: Previous maximum effective separation [m]
            damage_prev: Previous damage variable
        
        Returns:
            Dictionary with state variables:
                sigma_n: Normal traction [Pa]
                tau_s: Shear traction [Pa]
                damage: Damage variable D ∈ [0, 1]
                delta_eff: Effective separation [m]
                G_I: Mode I energy release rate [J/m²]
                G_II: Mode II energy release rate [J/m²]
                G_total: Total energy release rate [J/m²]
                G_c: Critical toughness at current mixity [J/m²]
                failed: Whether fully failed
        """
        # Effective separation (Macaulay bracket for normal)
        delta_n_eff = max(delta_n, 0.0)
        delta_eff = np.sqrt(delta_n_eff**2 + delta_s**2)
        
        # Mode mixity (based on displacement)
        if delta_eff > 1e-30:
            beta = delta_n_eff / delta_eff  # Mode I weight
        else:
            beta = 0.0
        
        # Effective stiffness and strength (quadratic interaction)
        K_eff = np.sqrt((beta * self.K_n)**2 + ((1 - beta) * self.K_s)**2)
        sigma_eff_max = np.sqrt((beta * self.sigma_max)**2 + ((1 - beta) * self.tau_max)**2)
        
        if K_eff < 1e-10:
            K_eff = self.K_n
        if sigma_eff_max < 1e-10:
            sigma_eff_max = self.sigma_max
        
        # Critical separations in effective space
        delta_eff_0 = sigma_eff_max / K_eff
        
        # B-K toughness for current mixity
        # Use displacement-based mixity
        if delta_eff > 1e-30:
            # Potential energy in each mode
            G_I_pot = 0.5 * self.K_n * delta_n_eff**2
            G_II_pot = 0.5 * self.K_s * delta_s**2
            G_total_pot = G_I_pot + G_II_pot
            
            if G_total_pot > 1e-30:
                psi = G_II_pot / G_total_pot
            else:
                psi = 0.0
        else:
            psi = 0.0
        
        G_c = self.bk.effective_toughness(psi)
        
        # Final separation in effective space
        delta_eff_f = 2.0 * G_c / sigma_eff_max
        
        # Irreversibility
        delta_max = max(delta_eff, delta_max_prev)
        
        # Damage evolution
        if delta_max <= delta_eff_0:
            damage = 0.0
        elif delta_max >= delta_eff_f:
            damage = 1.0
        else:
            damage = (delta_max - delta_eff_0) / (delta_eff_f - delta_eff_0)
            damage = max(damage, damage_prev)  # No healing
        
        # Compute tractions
        failed = damage >= 1.0 - 1e-10
        
        if delta_n_eff > 0:
            # Tension: damaged stiffness
            sigma_n = (1.0 - damage) * self.K_n * delta_n_eff
        else:
            # Compression: no damage, full stiffness
            sigma_n = self.K_n * delta_n  # Negative value
        
        tau_s = (1.0 - damage) * self.K_s * delta_s
        
        # Energy release rates (from damage dissipation)
        # For bilinear law: G = 0.5 * σ_max * δ_f * damage
        # But we need to decompose into modes
        if delta_eff > 1e-30:
            # Weight by displacement components
            w_I = delta_n_eff**2 / (delta_n_eff**2 + delta_s**2)
            w_II = delta_s**2 / (delta_n_eff**2 + delta_s**2)
        else:
            w_I = 1.0
            w_II = 0.0
        
        # Total dissipated energy
        G_total_diss = damage * G_c
        
        # Decompose
        G_I = w_I * G_total_diss
        G_II = w_II * G_total_diss
        G_total = G_I + G_II
        
        return {
            "sigma_n": sigma_n,
            "tau_s": tau_s,
            "damage": damage,
            "delta_eff": delta_eff,
            "delta_max": delta_max,
            "G_I": G_I,
            "G_II": G_II,
            "G_total": G_total,
            "G_c": G_c,
            "psi": psi,
            "failed": failed,
        }
    
    def compute_path(
        self,
        delta_n_path: np.ndarray,
        delta_s_path: np.ndarray,
    ) -> list[dict]:
        """Compute mixed-mode response along a loading path.
        
        Args:
            delta_n_path: Array of normal separations [m]
            delta_s_path: Array of shear separations [m]
        
        Returns:
            List of state dictionaries
        """
        states = []
        delta_max = 0.0
        damage = 0.0
        
        for dn, ds in zip(delta_n_path, delta_s_path):
            state = self.compute_traction(dn, ds, delta_max, damage)
            states.append(state)
            delta_max = state["delta_max"]
            damage = state["damage"]
        
        return states
    
    def __repr__(self) -> str:
        return (
            f"MixedModeCZM(σ_max={self.sigma_max:.3e}, τ_max={self.tau_max:.3e}, "
            f"G_Ic={self.G_Ic:.3e}, G_IIc={self.G_IIc:.3e}, η={self.bk.eta})"
        )
