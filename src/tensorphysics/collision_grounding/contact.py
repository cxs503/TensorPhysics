"""
Contact Algorithms for Collision Analysis

Implements:
- Master-slave contact algorithm
- Penalty method for contact constraints
- Hertz contact pressure distribution
"""

import numpy as np
from typing import Tuple, List, Optional


def hertz_contact_pressure(force: float, radius1: float, radius2: float,
                          E1: float, nu1: float, E2: float, nu2: float,
                          n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Hertz contact pressure distribution for two cylinders in contact.
    
    p(x) = p0 * sqrt(1 - (x/a)²)
    
    Parameters
    ----------
    force : float
        Contact force per unit length [N/m]
    radius1 : float
        Radius of first cylinder [m]
    radius2 : float
        Radius of second cylinder [m]
    E1 : float
        Young's modulus of first body [Pa]
    nu1 : float
        Poisson's ratio of first body
    E2 : float
        Young's modulus of second body [Pa]
    nu2 : float
        Poisson's ratio of second body
    n_points : int
        Number of points in contact zone
    
    Returns
    -------
    x : np.ndarray
        Position along contact zone [m]
    pressure : np.ndarray
        Contact pressure [Pa]
    a : float
        Contact half-width [m]
    """
    # Equivalent radius (for two cylinders in contact)
    R_eq = 1.0 / (1.0/radius1 + 1.0/radius2)
    
    # Equivalent modulus (plane strain)
    E1_star = E1 / (1.0 - nu1**2)
    E2_star = E2 / (1.0 - nu2**2)
    E_eq = 1.0 / (1.0/E1_star + 1.0/E2_star)
    
    # Contact half-width (Hertz theory for line contact)
    a = np.sqrt(4.0 * force * R_eq / (np.pi * E_eq))
    
    # Maximum pressure
    p0 = 2.0 * force / (np.pi * a)
    
    # Contact zone
    x = np.linspace(-a, a, n_points)
    
    # Pressure distribution
    pressure = p0 * np.sqrt(np.maximum(0, 1.0 - (x/a)**2))
    
    return x, pressure, a


class PenaltyContactMethod:
    """
    Penalty method for contact constraint enforcement.
    
    Parameters
    ----------
    penalty_parameter : float
        Penalty parameter α [N/m]
    friction_coefficient : float
        Friction coefficient μ, default 0.3
    """
    
    def __init__(self, penalty_parameter: float, friction_coefficient: float = 0.3):
        self.alpha = penalty_parameter
        self.mu = friction_coefficient
    
    def compute_contact_force(self, gap: float, 
                             relative_velocity: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """
        Compute contact force using penalty method.
        
        Parameters
        ----------
        gap : float
            Gap function (negative = penetration)
        relative_velocity : np.ndarray, optional
            Relative tangential velocity [m/s]
        
        Returns
        -------
        normal_force : float
            Normal contact force [N]
        tangential_force : np.ndarray
            Tangential (friction) force [N]
        """
        # Normal force (only when penetrating)
        if gap < 0:
            normal_force = -self.alpha * gap
        else:
            normal_force = 0.0
        
        # Tangential force (Coulomb friction)
        if relative_velocity is not None and normal_force > 0:
            v_tang = np.linalg.norm(relative_velocity)
            if v_tang > 1e-12:
                # Friction force opposes motion
                friction_magnitude = self.mu * normal_force
                tangential_force = -friction_magnitude * relative_velocity / v_tang
            else:
                tangential_force = np.zeros_like(relative_velocity)
        else:
            tangential_force = np.zeros(3)
        
        return normal_force, tangential_force
    
    def compute_contact_stiffness(self, gap: float) -> float:
        """
        Compute contact stiffness contribution.
        
        Parameters
        ----------
        gap : float
            Gap function
        
        Returns
        -------
        float
            Contact stiffness [N/m]
        """
        if gap < 0:
            return self.alpha
        return 0.0


class MasterSlaveContact:
    """
    Master-slave contact algorithm.
    
    Parameters
    ----------
    penalty_method : PenaltyContactMethod
        Penalty method for contact enforcement
    """
    
    def __init__(self, penalty_method: PenaltyContactMethod):
        self.penalty = penalty_method
        self.contact_pairs = []
        self.contact_forces = []
    
    def add_contact_pair(self, master_nodes: np.ndarray, slave_nodes: np.ndarray,
                        master_normals: Optional[np.ndarray] = None):
        """
        Add a master-slave contact pair.
        
        Parameters
        ----------
        master_nodes : np.ndarray
            Master surface nodes (N, 3)
        slave_nodes : np.ndarray
            Slave surface nodes (M, 3)
        master_normals : np.ndarray, optional
            Master surface normals (N, 3)
        """
        if master_normals is None:
            # Default normals pointing outward
            master_normals = np.zeros_like(master_nodes)
            master_normals[:, 2] = 1.0
        
        self.contact_pairs.append({
            'master': master_nodes,
            'slave': slave_nodes,
            'normals': master_normals
        })
    
    def detect_contact(self, tolerance: float = 1e-6) -> List[dict]:
        """
        Detect contact pairs with penetration.
        
        Parameters
        ----------
        tolerance : float
            Contact detection tolerance
        
        Returns
        -------
        list
            List of contact dictionaries with gap, slave_node, master_segment
        """
        contacts = []
        
        for pair in self.contact_pairs:
            master = pair['master']
            slave = pair['slave']
            normals = pair['normals']
            
            # Simple node-to-segment contact detection
            for i, slave_node in enumerate(slave):
                # Find closest master segment
                min_dist = np.inf
                closest_master = None
                closest_normal = None
                
                for j in range(len(master) - 1):
                    # Segment from master[j] to master[j+1]
                    seg_start = master[j]
                    seg_end = master[j+1]
                    seg_vec = seg_end - seg_start
                    seg_len = np.linalg.norm(seg_vec)
                    
                    if seg_len < 1e-12:
                        continue
                    
                    # Project slave node onto segment
                    t = np.dot(slave_node - seg_start, seg_vec) / (seg_len**2)
                    t = np.clip(t, 0, 1)
                    
                    closest_point = seg_start + t * seg_vec
                    dist_vec = slave_node - closest_point
                    dist = np.linalg.norm(dist_vec)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_master = (j, t)
                        closest_normal = normals[j]
                
                if closest_master is not None:
                    # Compute gap (signed distance)
                    gap = np.dot(slave_node - master[closest_master[0]], closest_normal)
                    
                    if gap < tolerance:
                        contacts.append({
                            'slave_node_idx': i,
                            'master_segment': closest_master,
                            'gap': gap,
                            'normal': closest_normal
                        })
        
        return contacts
    
    def compute_contact_forces(self, contacts: List[dict],
                              relative_velocities: Optional[List[np.ndarray]] = None) -> List[dict]:
        """
        Compute contact forces for detected contacts.
        
        Parameters
        ----------
        contacts : list
            List of contact dictionaries
        relative_velocities : list, optional
            Relative velocities at contact points
        
        Returns
        -------
        list
            List of contact force dictionaries
        """
        forces = []
        
        for i, contact in enumerate(contacts):
            rel_vel = None
            if relative_velocities is not None and i < len(relative_velocities):
                rel_vel = relative_velocities[i]
            
            normal_force, tangential_force = self.penalty.compute_contact_force(
                contact['gap'], rel_vel
            )
            
            forces.append({
                'contact': contact,
                'normal_force': normal_force,
                'tangential_force': tangential_force,
                'total_force': normal_force * contact['normal'] + tangential_force
            })
        
        self.contact_forces = forces
        return forces
