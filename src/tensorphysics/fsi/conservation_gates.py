"""Conservation gates for FSI interface verification.

These gates verify that momentum and energy are conserved at the fluid-structure
interface, which is a fundamental requirement for physically correct FSI coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConservationCheck:
    """Result of a conservation check.
    
    Attributes
    ----------
    name : str
        Name of the check.
    passed : bool
        Whether the check passed.
    value : float
        Computed value.
    tolerance : float
        Tolerance used for the check.
    message : str
        Human-readable description.
    """
    name: str
    passed: bool
    value: float
    tolerance: float
    message: str


@dataclass
class MomentumGate:
    """Momentum conservation gate for FSI interface.
    
    Verifies that the momentum exchanged at the fluid-structure interface
    is conserved: the force exerted by the fluid on the structure equals
    the reaction force on the fluid.
    
    Parameters
    ----------
    tolerance : float
        Relative tolerance for momentum conservation check.
    """
    tolerance: float = 1e-4
    
    def check(
        self,
        fluid_forces: np.ndarray,
        structural_forces: np.ndarray,
    ) -> ConservationCheck:
        """Check momentum conservation at interface.
        
        Parameters
        ----------
        fluid_forces : np.ndarray, shape (n_steps, 2)
            Forces exerted by fluid on structure (Fx, Fy) at each time step.
        structural_forces : np.ndarray, shape (n_steps, 2)
            Reaction forces from structure (should be equal and opposite).
        
        Returns
        -------
        ConservationCheck
            Result of the momentum conservation check.
        """
        # Momentum conservation: F_fluid + F_structural = 0
        # (Newton's third law)
        residual = fluid_forces + structural_forces
        
        # Compute relative error
        force_magnitude = np.max(np.abs(fluid_forces))
        if force_magnitude > 1e-10:
            rel_error = np.max(np.abs(residual)) / force_magnitude
        else:
            rel_error = np.max(np.abs(residual))
        
        passed = rel_error < self.tolerance
        
        return ConservationCheck(
            name="momentum_conservation",
            passed=passed,
            value=rel_error,
            tolerance=self.tolerance,
            message=f"Momentum residual: {rel_error:.2e} (tol: {self.tolerance:.2e})",
        )
    
    def check_from_coupling(
        self,
        lift_history: np.ndarray,
        moment_history: np.ndarray,
        plunge_velocity: np.ndarray,
        pitch_rate: np.ndarray,
        dt: float,
    ) -> ConservationCheck:
        """Check momentum conservation from coupling result.
        
        Uses impulse-momentum theorem: integral of force = change in momentum.
        
        Parameters
        ----------
        lift_history : np.ndarray
            Lift force history [N/m].
        moment_history : np.ndarray
            Moment history [N·m/m].
        plunge_velocity : np.ndarray
            Plunge velocity history [m/s].
        pitch_rate : np.ndarray
            Pitch rate history [rad/s].
        dt : float
            Time step [s].
        
        Returns
        -------
        ConservationCheck
            Result of the check.
        """
        # For a 2-DOF system, momentum conservation means:
        # ∫F dt = Δ(m·ḣ) and ∫M dt = Δ(I·α̇)
        
        # Compute impulse
        impulse_h = np.trapezoid(lift_history, dx=dt)
        impulse_a = np.trapezoid(moment_history, dx=dt)
        
        # Compute momentum change (assuming unit mass for simplicity)
        delta_momentum_h = plunge_velocity[-1] - plunge_velocity[0]
        delta_momentum_a = pitch_rate[-1] - pitch_rate[0]
        
        # Relative error
        max_impulse = max(abs(impulse_h), abs(impulse_a), 1e-10)
        max_delta = max(abs(delta_momentum_h), abs(delta_momentum_a), 1e-10)
        
        # For the reduced-order model, we check that the impulse is consistent
        # with the velocity change (within numerical tolerance)
        rel_error = abs(impulse_h - delta_momentum_h) / max(abs(impulse_h), 1e-10)
        
        passed = rel_error < self.tolerance
        
        return ConservationCheck(
            name="momentum_conservation_coupling",
            passed=passed,
            value=rel_error,
            tolerance=self.tolerance,
            message=f"Momentum balance error: {rel_error:.2e} (tol: {self.tolerance:.2e})",
        )


@dataclass
class EnergyGate:
    """Energy conservation gate for FSI interface.
    
    Verifies that energy is conserved at the fluid-structure interface:
    the work done by the fluid equals the change in structural energy
    (strain + kinetic) plus dissipated energy.
    
    Parameters
    ----------
    tolerance : float
        Relative tolerance for energy conservation check.
    """
    tolerance: float = 0.03  # 3% relative error
    
    def check(
        self,
        work_done: float,
        strain_energy: float,
        kinetic_energy: float,
        dissipated_energy: float,
    ) -> ConservationCheck:
        """Check energy conservation.
        
        Energy balance: Work = Strain Energy + Kinetic Energy + Dissipated Energy
        
        Parameters
        ----------
        work_done : float
            Total work done by fluid on structure [J/m].
        strain_energy : float
            Final structural strain energy [J/m].
        kinetic_energy : float
            Final structural kinetic energy [J/m].
        dissipated_energy : float
            Total energy dissipated by damping [J/m].
        
        Returns
        -------
        ConservationCheck
            Result of the energy conservation check.
        """
        # Energy balance
        total_structural = strain_energy + kinetic_energy + dissipated_energy
        
        # Relative error
        if abs(work_done) > 1e-10:
            rel_error = abs(work_done - total_structural) / abs(work_done)
        else:
            rel_error = abs(work_done - total_structural)
        
        passed = rel_error < self.tolerance
        
        return ConservationCheck(
            name="energy_conservation",
            passed=passed,
            value=rel_error,
            tolerance=self.tolerance,
            message=f"Energy balance error: {rel_error:.2e} (tol: {self.tolerance:.2e})",
        )
    
    def check_from_history(
        self,
        lift: np.ndarray,
        moment: np.ndarray,
        plunge_velocity: np.ndarray,
        pitch_rate: np.ndarray,
        plunge: np.ndarray,
        pitch: np.ndarray,
        kh: float,
        kα: float,
        m: float,
        Iα: float,
        ch: float,
        cα: float,
        dt: float,
    ) -> ConservationCheck:
        """Check energy conservation from time history.
        
        Parameters
        ----------
        lift : np.ndarray
            Lift force history [N/m].
        moment : np.ndarray
            Moment history [N·m/m].
        plunge_velocity : np.ndarray
            Plunge velocity history [m/s].
        pitch_rate : np.ndarray
            Pitch rate history [rad/s].
        plunge : np.ndarray
            Plunge displacement history [m].
        pitch : np.ndarray
            Pitch angle history [rad].
        kh : float
            Plunge stiffness [N/m].
        kα : float
            Pitch stiffness [N·m/rad].
        m : float
            Mass per unit span [kg/m].
        Iα : float
            Moment of inertia [kg·m].
        ch : float
            Plunge damping coefficient [N·s/m].
        cα : float
            Pitch damping coefficient [N·m·s/rad].
        dt : float
            Time step [s].
        
        Returns
        -------
        ConservationCheck
            Result of the check.
        """
        # Compute work done by fluid
        power = lift * plunge_velocity + moment * pitch_rate
        work_done = np.trapezoid(power, dx=dt)
        
        # Final energies
        strain_energy = 0.5 * kh * plunge[-1]**2 + 0.5 * kα * pitch[-1]**2
        kinetic_energy = 0.5 * m * plunge_velocity[-1]**2 + 0.5 * Iα * pitch_rate[-1]**2
        
        # Dissipated energy (cumulative)
        dissipation_rate = ch * plunge_velocity**2 + cα * pitch_rate**2
        dissipated_energy = np.trapezoid(dissipation_rate, dx=dt)
        
        return self.check(work_done, strain_energy, kinetic_energy, dissipated_energy)
