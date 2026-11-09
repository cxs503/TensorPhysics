"""
Material Models for Collision and Grounding Analysis

Implements:
- Elastic-plastic material with isotropic hardening
- Von Mises yield criterion
- Cowper-Symonds strain rate model
"""

import numpy as np
from typing import Tuple, Optional


def von_mises_yield(stress_tensor: np.ndarray) -> float:
    """
    Calculate von Mises equivalent stress from stress tensor.
    
    σ_vm = sqrt(3/2 * s:s) where s is deviatoric stress
    
    Parameters
    ----------
    stress_tensor : np.ndarray
        3x3 stress tensor or 6-component Voigt notation [σ11, σ22, σ33, σ12, σ23, σ13]
    
    Returns
    -------
    float
        Von Mises equivalent stress
    """
    if stress_tensor.shape == (6,):
        # Voigt notation: [σ11, σ22, σ33, σ12, σ23, σ13]
        s11, s22, s33, s12, s23, s13 = stress_tensor
        sigma_vm = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2 + 
                                   6 * (s12**2 + s23**2 + s13**2)))
    elif stress_tensor.shape == (3, 3):
        # Full tensor
        mean_stress = np.trace(stress_tensor) / 3.0
        deviatoric = stress_tensor - mean_stress * np.eye(3)
        sigma_vm = np.sqrt(1.5 * np.sum(deviatoric * deviatoric))
    else:
        raise ValueError("Stress tensor must be 3x3 or 6-component Voigt notation")
    
    return sigma_vm


def isotropic_hardening(equivalent_plastic_strain: float, 
                       sigma_y: float, 
                       hardening_modulus: float) -> float:
    """
    Linear isotropic hardening law.
    
    σ_y(ε_p) = σ_y0 + H * ε_p
    
    Parameters
    ----------
    equivalent_plastic_strain : float
        Equivalent plastic strain
    sigma_y : float
        Initial yield stress
    hardening_modulus : float
        Hardening modulus H
    
    Returns
    -------
    float
        Current yield stress
    """
    return sigma_y + hardening_modulus * equivalent_plastic_strain


class ElasticPlasticMaterial:
    """
    Elastic-plastic material with isotropic hardening.
    
    Implements return mapping algorithm for von Mises yield criterion
    with linear isotropic hardening.
    
    Parameters
    ----------
    youngs_modulus : float
        Young's modulus E [Pa]
    poissons_ratio : float
        Poisson's ratio ν
    yield_stress : float
        Initial yield stress σ_y [Pa]
    hardening_modulus : float
        Hardening modulus H [Pa]
    """
    
    def __init__(self, youngs_modulus: float, poissons_ratio: float,
                 yield_stress: float, hardening_modulus: float):
        self.E = youngs_modulus
        self.nu = poissons_ratio
        self.sigma_y0 = yield_stress
        self.H = hardening_modulus
        
        # Elastic constants
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        
        # State variables
        self.plastic_strain = 0.0
        self.current_yield_stress = self.sigma_y0
    
    def elastic_stiffness_tensor(self) -> np.ndarray:
        """
        Compute 6x6 elastic stiffness tensor (Voigt notation).
        
        Returns
        -------
        np.ndarray
            6x6 stiffness matrix C
        """
        C = np.zeros((6, 6))
        
        # Normal components
        C[0, 0] = C[1, 1] = C[2, 2] = self.lam + 2 * self.G
        C[0, 1] = C[0, 2] = C[1, 2] = self.lam
        C[1, 0] = C[2, 0] = C[2, 1] = self.lam
        
        # Shear components
        C[3, 3] = C[4, 4] = C[5, 5] = self.G
        
        return C
    
    def compute_stress(self, strain_increment: np.ndarray, 
                      total_strain: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute stress using return mapping algorithm.
        
        Parameters
        ----------
        strain_increment : np.ndarray
            Strain increment (6-component Voigt)
        total_strain : np.ndarray
            Total strain (6-component Voigt)
        
        Returns
        -------
        stress : np.ndarray
            Updated stress (6-component Voigt)
        plastic_strain_increment : np.ndarray
            Plastic strain increment
        delta_gamma : float
            Plastic multiplier
        """
        # Trial stress (elastic predictor)
        C = self.elastic_stiffness_tensor()
        trial_stress = C @ total_strain
        
        # Compute trial von Mises stress
        trial_vm = von_mises_yield(trial_stress)
        
        # Check yield condition
        f_trial = trial_vm - self.current_yield_stress
        
        if f_trial <= 0:
            # Elastic step
            return trial_stress, np.zeros(6), 0.0
        
        # Plastic correction (return mapping)
        # Deviatoric trial stress
        mean_trial = (trial_stress[0] + trial_stress[1] + trial_stress[2]) / 3.0
        s_trial = trial_stress.copy()
        s_trial[0:3] -= mean_trial
        
        # Direction of plastic flow
        norm_s = np.linalg.norm(s_trial)
        if norm_s < 1e-15:
            return trial_stress, np.zeros(6), 0.0
        n_flow = s_trial / norm_s
        
        # Plastic multiplier
        delta_gamma = f_trial / (3.0 * self.G + self.H)
        
        # Update stress
        stress = trial_stress - 2.0 * self.G * delta_gamma * n_flow
        
        # Plastic strain increment
        plastic_strain_inc = delta_gamma * n_flow
        
        # Update state
        self.plastic_strain += delta_gamma
        self.current_yield_stress = isotropic_hardening(
            self.plastic_strain, self.sigma_y0, self.H
        )
        
        return stress, plastic_strain_inc, delta_gamma
    
    def uniaxial_stress(self, strain_11: float) -> float:
        """
        Compute uniaxial stress (σ11) for given axial strain ε11.
        
        This is the proper 1D constitutive response for uniaxial stress
        conditions (σ22 = σ33 = 0).
        
        Parameters
        ----------
        strain_11 : float
            Axial strain
        
        Returns
        -------
        float
            Axial stress σ11
        """
        # Elastic predictor
        sigma_trial = self.E * strain_11
        
        # Check yield
        if abs(sigma_trial) <= self.current_yield_stress:
            return sigma_trial
        
        # Plastic correction (uniaxial case: σ_vm = |σ11|)
        sign = 1.0 if sigma_trial > 0 else -1.0
        f_trial = abs(sigma_trial) - self.current_yield_stress
        
        # Plastic multiplier (uniaxial: consistent tangent)
        # sigma = sigma_y + H * eps_p
        # eps_total = sigma/E + eps_p
        # sigma = (sigma_y + H * eps_total) / (1 + H/E)
        sigma = (self.current_yield_stress + self.H * strain_11) / (1.0 + self.H / self.E)
        
        # Update state
        eps_p = strain_11 - sigma / self.E
        self.plastic_strain = abs(eps_p)
        self.current_yield_stress = isotropic_hardening(
            self.plastic_strain, self.sigma_y0, self.H
        )
        
        return sigma
    
    def get_stress_strain_curve(self, max_strain: float, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate uniaxial stress-strain curve (monotonic loading).
        
        Parameters
        ----------
        max_strain : float
            Maximum strain
        n_points : int
            Number of points
        
        Returns
        -------
        strains : np.ndarray
            Strain values
        stresses : np.ndarray
            Stress values
        """
        strains = np.linspace(0, max_strain, n_points)
        stresses = np.zeros(n_points)
        
        # Analytical solution for monotonic loading
        eps_y = self.sigma_y0 / self.E
        
        for i in range(n_points):
            eps = strains[i]
            if eps <= eps_y:
                # Elastic
                stresses[i] = self.E * eps
            else:
                # Plastic with hardening
                # sigma = (sigma_y + H * eps) / (1 + H/E)
                stresses[i] = (self.sigma_y0 + self.H * eps) / (1.0 + self.H / self.E)
        
        return strains, stresses


class CowperSymondsModel:
    """
    Cowper-Symonds strain rate dependent material model.
    
    σ_d = σ_y * [1 + (ε̇/C)^(1/p)]
    
    Parameters
    ----------
    quasi_static_model : ElasticPlasticMaterial
        Base quasi-static material model
    C : float
        Cowper-Symonds parameter C [s⁻¹], default 40 s⁻¹
    p : float
        Cowper-Symonds parameter p, default 5
    """
    
    def __init__(self, quasi_static_model: ElasticPlasticMaterial,
                 C: float = 40.0, p: float = 5.0):
        self.base_model = quasi_static_model
        self.C = C
        self.p = p
    
    def dynamic_yield_stress(self, strain_rate: float) -> float:
        """
        Compute dynamic yield stress using Cowper-Symonds model.
        
        Parameters
        ----------
        strain_rate : float
            Strain rate [s⁻¹]
        
        Returns
        -------
        float
            Dynamic yield stress
        """
        if strain_rate <= 0:
            return self.base_model.current_yield_stress
        
        DIF = 1.0 + (strain_rate / self.C) ** (1.0 / self.p)
        return self.base_model.current_yield_stress * DIF
    
    def compute_stress_rate_dependent(self, strain_increment: np.ndarray,
                                     total_strain: np.ndarray,
                                     strain_rate: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute stress with strain rate effects.
        
        Parameters
        ----------
        strain_increment : np.ndarray
            Strain increment
        total_strain : np.ndarray
            Total strain
        strain_rate : float
            Strain rate [s⁻¹]
        
        Returns
        -------
        stress : np.ndarray
            Updated stress
        plastic_strain_increment : np.ndarray
            Plastic strain increment
        delta_gamma : float
            Plastic multiplier
        """
        # Temporarily modify yield stress
        original_yield = self.base_model.current_yield_stress
        self.base_model.current_yield_stress = self.dynamic_yield_stress(strain_rate)
        
        # Compute stress
        stress, plastic_strain_inc, delta_gamma = self.base_model.compute_stress(
            strain_increment, total_strain
        )
        
        # Restore original yield stress
        self.base_model.current_yield_stress = original_yield
        
        return stress, plastic_strain_inc, delta_gamma
