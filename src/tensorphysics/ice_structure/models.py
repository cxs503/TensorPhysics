"""Ice-structure interaction toy models.

Provides simplified mechanical models for studying partitioned coupling stability:
- 1-DOF: Single mass-spring-damper with added mass from ice/fluid
- 2-DOF: Ice block + structure with coupling spring

These models capture the essential physics of added-mass instability in
partitioned ice-structure coupling schemes.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class OneDOFModel:
    """1-DOF mass-spring-damper with added mass effect.
    
    Equation of motion:
        (m_s + m_a) * x'' + c * x' + k * x = F_ext(t)
    
    where m_a is the added mass from the ice/fluid domain.
    
    Parameters
    ----------
    m_s : float
        Structural mass [kg]
    k : float
        Spring stiffness [N/m]
    c : float
        Damping coefficient [N·s/m]
    m_a : float
        Added mass from ice/fluid [kg]
    """
    m_s: float
    k: float
    c: float
    m_a: float = 0.0
    
    @property
    def m_total(self) -> float:
        """Total mass including added mass."""
        return self.m_s + self.m_a
    
    @property
    def mu(self) -> float:
        """Added mass ratio mu = m_a / m_s."""
        return self.m_a / self.m_s if self.m_s > 0 else 0.0
    
    @property
    def omega_n(self) -> float:
        """Natural frequency [rad/s]."""
        return np.sqrt(self.k / self.m_total)
    
    @property
    def zeta(self) -> float:
        """Damping ratio."""
        return self.c / (2.0 * np.sqrt(self.k * self.m_total))
    
    def ice_force(self, x: float, v: float, t: float) -> float:
        """Compute ice interaction force.
        
        For the 1-DOF model, the ice force is modeled as:
        - Added mass contribution: -m_a * a (implicit in effective mass)
        - Damping from ice: -c_ice * v
        - Stiffness from ice: -k_ice * x
        
        This simplified model captures the reactive nature of ice loading.
        
        Parameters
        ----------
        x : float
            Displacement [m]
        v : float
            Velocity [m/s]
        t : float
            Time [s]
            
        Returns
        -------
        float
            Ice force [N]
        """
        # Simplified ice-structure interaction:
        # The added mass effect is captured through m_a in the effective mass
        # Additional ice forces can be added here for more complex models
        return 0.0  # Added mass already in effective mass
    
    def external_force(self, t: float) -> float:
        """External forcing function.
        
        Default: impulse at t=0
        
        Parameters
        ----------
        t : float
            Time [s]
            
        Returns
        -------
        float
            External force [N]
        """
        # Impulse at t=0
        if abs(t) < 1e-10:
            return 1.0
        return 0.0
    
    def state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return state-space matrices A, B for the system.
        
        State vector: [x, v]^T
        dx/dt = A @ x + B @ u
        
        Returns
        -------
        A : ndarray
            System matrix (2x2)
        B : ndarray
            Input matrix (2x1)
        """
        m = self.m_total
        A = np.array([
            [0.0, 1.0],
            [-self.k / m, -self.c / m]
        ])
        B = np.array([[0.0], [1.0 / m]])
        return A, B


@dataclass
class TwoDOFModel:
    """2-DOF ice block + structure model.
    
    Equations of motion:
        m_i * x_i'' + c_i * x_i' + k_i * x_i + k_c * (x_i - x_s) = F_ice(t)
        m_s * x_s'' + c_s * x_s' + k_s * x_s + k_c * (x_s - x_i) = 0
    
    where:
    - Subscript i: ice block
    - Subscript s: structure
    - k_c: coupling spring between ice and structure
    
    Parameters
    ----------
    m_i : float
        Ice block mass [kg]
    k_i : float
        Ice stiffness [N/m]
    c_i : float
        Ice damping [N·s/m]
    m_s : float
        Structural mass [kg]
    k_s : float
        Structural stiffness [N/m]
    c_s : float
        Structural damping [N·s/m]
    k_c : float
        Coupling stiffness [N/m]
    """
    m_i: float
    k_i: float
    c_i: float
    m_s: float
    k_s: float
    c_s: float
    k_c: float
    
    @property
    def mu(self) -> float:
        """Effective added mass ratio.
        
        For the 2-DOF model, the added mass effect is captured through
        the coupling stiffness and ice mass.
        """
        # Effective added mass from coupling
        return self.m_i / self.m_s if self.m_s > 0 else 0.0
    
    def state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return state-space matrices for the coupled system.
        
        State vector: [x_i, x_s, v_i, v_s]^T
        
        Returns
        -------
        A : ndarray
            System matrix (4x4)
        B : ndarray
            Input matrix (4x1)
        """
        # Mass matrix
        M = np.diag([self.m_i, self.m_s, 1.0, 1.0])
        
        # Stiffness matrix
        K = np.array([
            [self.k_i + self.k_c, -self.k_c, 0.0, 0.0],
            [-self.k_c, self.k_s + self.k_c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Damping matrix
        C = np.array([
            [self.c_i, 0.0, 0.0, 0.0],
            [0.0, self.c_s, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Convert to first-order form: [x; v]' = [0 I; -M^-1 K -M^-1 C] [x; v]
        M_inv = np.diag([1.0/self.m_i, 1.0/self.m_s, 1.0, 1.0])
        
        A = np.zeros((4, 4))
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 0:2] = -M_inv[0:2, 0:2][:, np.newaxis] * K[0:2, 0:2]
        A[2:4, 2:4] = -M_inv[0:2, 0:2][:, np.newaxis] * C[0:2, 0:2]
        
        # Fix the matrix multiplication
        A[2, 0:2] = -(K[0, :] / self.m_i)
        A[3, 0:2] = -(K[1, :] / self.m_s)
        A[2, 2:4] = -(C[0, :] / self.m_i)
        A[3, 2:4] = -(C[1, :] / self.m_s)
        
        B = np.array([[0.0], [0.0], [1.0/self.m_i], [0.0]])
        
        return A, B
    
    def ice_force(self, x_i: float, v_i: float, t: float) -> float:
        """Ice excitation force."""
        # Periodic ice breaking force
        return 100.0 * np.sin(2.0 * np.pi * 0.5 * t)
    
    def solve_coupled(self, t_end: float, dt: float) -> dict:
        """Solve the fully coupled system (reference solution).
        
        Parameters
        ----------
        t_end : float
            End time [s]
        dt : float
            Time step [s]
            
        Returns
        -------
        dict
            Solution with keys 't', 'x_i', 'x_s', 'v_i', 'v_s'
        """
        n_steps = int(t_end / dt)
        t = np.linspace(0, t_end, n_steps + 1)
        
        # State: [x_i, x_s, v_i, v_s]
        state = np.zeros(4)
        history = np.zeros((n_steps + 1, 4))
        history[0] = state.copy()
        
        # Initial conditions: impulse on ice
        state[2] = 1.0 / self.m_i  # Initial velocity
        
        for k in range(n_steps):
            # RK4 integration
            def f(s):
                x_i, x_s, v_i, v_s = s
                F_ice = self.ice_force(x_i, v_i, t[k])
                F_couple = self.k_c * (x_s - x_i)
                a_i = (F_ice - self.c_i * v_i - self.k_i * x_i + F_couple) / self.m_i
                a_s = (-self.c_s * v_s - self.k_s * x_s - F_couple) / self.m_s
                return np.array([v_i, v_s, a_i, a_s])
            
            k1 = f(state)
            k2 = f(state + 0.5 * dt * k1)
            k3 = f(state + 0.5 * dt * k2)
            k4 = f(state + dt * k3)
            
            state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            history[k+1] = state.copy()
        
        return {
            't': t,
            'x_i': history[:, 0],
            'x_s': history[:, 1],
            'v_i': history[:, 2],
            'v_s': history[:, 3],
        }


@dataclass
class IceStructureSystem:
    """High-level ice-structure interaction system.
    
    Combines the toy models with coupling scheme parameters.
    
    Parameters
    ----------
    model : OneDOFModel or TwoDOFModel
        The mechanical model
    dt : float
        Time step [s]
    t_end : float
        Simulation end time [s]
    """
    model: OneDOFModel | TwoDOFModel
    dt: float = 0.001
    t_end: float = 1.0
    
    def solve_reference(self) -> dict:
        """Solve the fully coupled (monolithic) system as reference."""
        if isinstance(self.model, TwoDOFModel):
            return self.model.solve_coupled(self.t_end, self.dt)
        else:
            return self._solve_1dof_reference()
    
    def _solve_1dof_reference(self) -> dict:
        """Reference solution for 1-DOF model."""
        n_steps = int(self.t_end / self.dt)
        t = np.linspace(0, self.t_end, n_steps + 1)
        
        x = np.zeros(n_steps + 1)
        v = np.zeros(n_steps + 1)
        
        # Initial conditions
        x[0] = 0.0
        v[0] = 1.0 / self.model.m_total  # Impulse
        
        m = self.model.m_total
        k = self.model.k
        c = self.model.c
        
        for n in range(n_steps):
            # RK4
            def f(xi, vi, ti):
                F_ext = self.model.external_force(ti)
                F_ice = self.model.ice_force(xi, vi, ti)
                a = (F_ext + F_ice - c * vi - k * xi) / m
                return vi, a
            
            k1x, k1v = f(x[n], v[n], t[n])
            k2x, k2v = f(x[n] + 0.5*self.dt*k1x, v[n] + 0.5*self.dt*k1v, t[n] + 0.5*self.dt)
            k3x, k3v = f(x[n] + 0.5*self.dt*k2x, v[n] + 0.5*self.dt*k2v, t[n] + 0.5*self.dt)
            k4x, k4v = f(x[n] + self.dt*k3x, v[n] + self.dt*k3v, t[n] + self.dt)
            
            x[n+1] = x[n] + (self.dt/6) * (k1x + 2*k2x + 2*k3x + k4x)
            v[n+1] = v[n] + (self.dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
        
        return {'t': t, 'x': x, 'v': v}
