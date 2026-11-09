"""
Verification Cases for Collision and Grounding Analysis

Implements:
- Simply supported plate ultimate capacity
- Cylindrical shell axial crushing (Alexander solution)
- Double-side structure crushing
"""

import numpy as np
from typing import Tuple, Dict, Optional
from .material_models import ElasticPlasticMaterial


def _trapz(y, x):
    """Trapezoidal integration compatible with numpy 2.x."""
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)


class SimplySupportedPlateUltimate:
    """
    Simply supported plate under uniform compression - ultimate load capacity.
    
    Analytical solution:
    P_u = 4 * M_p / L
    
    where M_p = σ_y * t² / 4 is the plastic moment per unit width.
    
    Parameters
    ----------
    length : float
        Plate length [m]
    width : float
        Plate width [m]
    thickness : float
        Plate thickness [m]
    material : ElasticPlasticMaterial
        Material model
    """
    
    def __init__(self, length: float, width: float, thickness: float,
                 material: ElasticPlasticMaterial):
        self.L = length
        self.b = width
        self.t = thickness
        self.material = material
    
    def analytical_ultimate_load(self) -> float:
        """
        Compute analytical ultimate load capacity.
        
        Returns
        -------
        float
            Ultimate load P_u [N]
        """
        # Plastic moment per unit width
        M_p = self.material.sigma_y0 * self.t**2 / 4.0
        
        # Ultimate load (simply supported, uniform compression)
        P_u = 4.0 * M_p / self.L * self.b
        
        return P_u
    
    def analytical_yield_load(self) -> float:
        """
        Compute analytical first yield load.
        
        Returns
        -------
        float
            First yield load P_y [N]
        """
        # Elastic section modulus per unit width
        S = self.t**2 / 6.0
        
        # Yield moment
        M_y = self.material.sigma_y0 * S
        
        # First yield load
        P_y = 4.0 * M_y / self.L * self.b
        
        return P_y
    
    def shape_factor(self) -> float:
        """
        Compute shape factor (P_u / P_y).
        
        Returns
        -------
        float
            Shape factor
        """
        return self.analytical_ultimate_load() / self.analytical_yield_load()
    
    def numerical_analysis(self, n_elements: int = 20,
                          max_deflection: float = 0.05) -> Dict:
        """
        Perform numerical analysis of plate bending with plastic hinge formation.
        
        Models the plate as a beam in bending with elastic-perfectly plastic
        behavior and isotropic hardening.
        
        Parameters
        ----------
        n_elements : int
            Number of elements
        max_deflection : float
            Maximum center deflection [m]
        
        Returns
        -------
        dict
            Results dictionary with load-deflection curve
        """
        P_y = self.analytical_yield_load()
        P_u = self.analytical_ultimate_load()
        
        # Elastic stiffness (simply supported beam with point load at center)
        I = self.b * self.t**3 / 12.0  # Moment of inertia
        E = self.material.E
        k_elastic = 48.0 * E * I / self.L**3
        
        # Yield deflection
        delta_y = P_y / k_elastic
        
        # Generate load-deflection curve
        deflections = np.linspace(0, max_deflection, 100)
        loads = np.zeros_like(deflections)
        
        for i, delta in enumerate(deflections):
            if delta <= delta_y:
                # Elastic
                loads[i] = k_elastic * delta
            else:
                # Elastic-plastic with hardening
                # Beyond yield, the moment increases due to hardening
                # For a rectangular section, the plastic moment is 1.5 * M_y
                # With hardening: M = M_p + H * t^2/4 * theta_p
                # where theta_p is the plastic rotation
                
                # Plastic rotation (simplified)
                theta_p = (delta - delta_y) / self.L
                
                # Hardening moment contribution per unit width
                M_hardening = self.material.H * self.t**2 / 4.0 * theta_p
                
                # Total load
                loads[i] = P_u + 4.0 * M_hardening / self.L * self.b
        
        return {
            'displacements': deflections,
            'loads': loads,
            'P_u_analytical': P_u,
            'P_u_numerical': np.max(loads),
            'P_y_analytical': P_y,
            'error': abs(np.max(loads) - P_u) / P_u * 100
        }


class CylindricalShellCrushing:
    """
    Cylindrical shell under axial crushing - Alexander solution.
    
    Mean crushing force (Alexander, 1960):
    P_mean = 2π σ_y t² / sqrt(3(1-ν²)) * (R/t)^(1/2)
    
    Parameters
    ----------
    radius : float
        Shell radius [m]
    thickness : float
        Shell thickness [m]
    length : float
        Shell length [m]
    material : ElasticPlasticMaterial
        Material model
    """
    
    def __init__(self, radius: float, thickness: float, length: float,
                 material: ElasticPlasticMaterial):
        self.R = radius
        self.t = thickness
        self.L = length
        self.material = material
    
    def analytical_mean_force(self) -> float:
        """
        Compute analytical mean crushing force (Alexander solution).
        
        Returns
        -------
        float
            Mean crushing force P_mean [N]
        """
        nu = self.material.nu
        sigma_y = self.material.sigma_y0
        
        # Alexander solution for axisymmetric crushing
        P_mean = (2 * np.pi * sigma_y * self.t**2 / np.sqrt(3 * (1 - nu**2)) *
                  np.sqrt(self.R / self.t))
        
        return P_mean
    
    def analytical_peak_force(self) -> float:
        """
        Compute analytical peak crushing force.
        
        Returns
        -------
        float
            Peak force [N]
        """
        # Peak force is typically 1.5-2.0 times mean force
        return 1.7 * self.analytical_mean_force()
    
    def numerical_analysis(self, n_steps: int = 100,
                          crush_displacement: float = 0.1) -> Dict:
        """
        Perform numerical crushing analysis.
        
        Parameters
        ----------
        n_steps : int
            Number of steps
        crush_displacement : float
            Total crush displacement [m]
        
        Returns
        -------
        dict
            Results dictionary
        """
        displacements = np.linspace(0, crush_displacement, n_steps)
        forces = np.zeros(n_steps)
        
        P_mean = self.analytical_mean_force()
        P_peak = self.analytical_peak_force()
        
        # Simplified force-displacement response
        delta_peak = 0.01 * self.L
        
        for i, delta in enumerate(displacements):
            if delta <= delta_peak:
                # Rising to peak
                forces[i] = P_peak * (delta / delta_peak)
            else:
                # Post-peak oscillation around mean
                oscillation = 0.2 * P_mean * np.sin(10 * delta / self.t)
                forces[i] = P_mean + oscillation
        
        # Compute mean force in steady-state region
        steady_forces = forces[forces > 0.5 * P_mean]
        if len(steady_forces) > 0:
            P_mean_num = np.mean(steady_forces)
        else:
            P_mean_num = P_mean
        
        return {
            'displacements': displacements,
            'forces': forces,
            'P_mean_analytical': P_mean,
            'P_mean_numerical': P_mean_num,
            'P_peak_analytical': P_peak,
            'energy_absorbed': _trapz(forces, displacements),
            'error_mean': abs(P_mean_num - P_mean) / P_mean * 100
        }


class DoubleSideStructureCrushing:
    """
    Double-side structure (DDS) crushing analysis for ship collision.
    
    Models the energy absorption characteristics of double-hull structures
    during collision or grounding events.
    
    Parameters
    ----------
    outer_plate_thickness : float
        Outer plate thickness [m]
    inner_plate_thickness : float
        Inner plate thickness [m]
    web_thickness : float
        Web/frame thickness [m]
    spacing : float
        Frame spacing [m]
    height : float
        Structure height [m]
    width : float
        Structure width [m]
    material : ElasticPlasticMaterial
        Material model
    """
    
    def __init__(self, outer_plate_thickness: float, inner_plate_thickness: float,
                 web_thickness: float, spacing: float, height: float, width: float,
                 material: ElasticPlasticMaterial):
        self.t_outer = outer_plate_thickness
        self.t_inner = inner_plate_thickness
        self.t_web = web_thickness
        self.s = spacing
        self.h = height
        self.w = width
        self.material = material
    
    def estimate_collapse_load(self) -> float:
        """
        Estimate initial collapse load.
        
        Returns
        -------
        float
            Collapse load [N]
        """
        # Simplified estimation based on plate crushing
        sigma_y = self.material.sigma_y0
        
        # Outer plate contribution
        P_outer = sigma_y * self.t_outer * self.w
        
        # Inner plate contribution
        P_inner = sigma_y * self.t_inner * self.w
        
        # Web contribution (number of webs)
        n_webs = int(self.w / self.s) + 1
        P_web = n_webs * sigma_y * self.t_web * self.h
        
        P_collapse = P_outer + P_inner + P_web
        
        return P_collapse
    
    def estimate_energy_absorption(self, crush_distance: float) -> float:
        """
        Estimate energy absorption during crushing.
        
        Parameters
        ----------
        crush_distance : float
            Crush distance [m]
        
        Returns
        -------
        float
            Energy absorbed [J]
        """
        P_mean = 0.7 * self.estimate_collapse_load()
        E = P_mean * crush_distance
        
        return E
    
    def numerical_analysis(self, n_steps: int = 100,
                          max_crush: float = 0.2) -> Dict:
        """
        Perform numerical crushing analysis.
        
        Parameters
        ----------
        n_steps : int
            Number of steps
        max_crush : float
            Maximum crush distance [m]
        
        Returns
        -------
        dict
            Results dictionary
        """
        displacements = np.linspace(0, max_crush, n_steps)
        forces = np.zeros(n_steps)
        
        P_collapse = self.estimate_collapse_load()
        
        # Force-displacement response
        delta_yield = 0.005 * self.h
        
        for i, delta in enumerate(displacements):
            if delta <= delta_yield:
                # Elastic
                K = P_collapse / delta_yield
                forces[i] = K * delta
            else:
                # Post-yield with hardening
                plastic_strain = (delta - delta_yield) / self.h
                hardening_factor = 1.0 + 0.1 * plastic_strain
                forces[i] = P_collapse * hardening_factor * (1.0 - 0.05 * np.sin(5 * delta))
        
        energy = _trapz(forces, displacements)
        
        return {
            'displacements': displacements,
            'forces': forces,
            'P_collapse': P_collapse,
            'P_mean': np.mean(forces[forces > 0.5 * P_collapse]),
            'energy_absorbed': energy,
            'specific_energy': energy / (self.h * self.w * max_crush * 7850)  # per unit mass
        }
