"""RunManifest: orchestrator for three-solver clock synchronization and field exchange.

The RunManifest manages the coupling loop between LBM (flow), DEM (ice), and FEM (structure):
- Time synchronization with sub-cycling
- Field exchange (pressure → FEM load, displacement → LBM boundary, ice-hull force → DEM/FEM)
- Energy ledger for conservation tracking
- Convergence monitoring and divergence detection
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .lbm3d import LBM3D, LBM3DConfig
from .dem3d import DEM3D, DEM3DConfig
from .fem3d import FEM3DHull, FEM3DConfig


class CouplingStatus(str, Enum):
    RUNNING = "running"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    MAX_STEPS_REACHED = "max_steps_reached"
    ENERGY_VIOLATION = "energy_violation"


@dataclass
class EnergyLedger:
    """Track energy conservation across the three solvers."""
    fluid_kinetic: float = 0.0
    ice_kinetic: float = 0.0
    ice_potential: float = 0.0
    structure_strain: float = 0.0
    dissipated: float = 0.0
    work_external: float = 0.0

    @property
    def total(self) -> float:
        return (self.fluid_kinetic + self.ice_kinetic + self.ice_potential +
                self.structure_strain + self.dissipated)

    def snapshot(self) -> dict:
        return {
            'fluid_kinetic': self.fluid_kinetic,
            'ice_kinetic': self.ice_kinetic,
            'ice_potential': self.ice_potential,
            'structure_strain': self.structure_strain,
            'dissipated': self.dissipated,
            'work_external': self.work_external,
            'total': self.total,
        }


@dataclass
class FieldExchange:
    step: int
    time: float
    source: str
    target: str
    field_name: str
    magnitude: float
    checksum: str


@dataclass
class RunManifestConfig:
    dt_coupling: float = 1.0  # coupling time step (lattice units)
    n_steps: int = 100
    lbm_substeps: int = 5
    dem_substeps: int = 2
    exchange_interval: int = 1
    energy_tolerance: float = 0.05
    divergence_threshold: float = 1e10
    sample_interval: int = 1
    seed: int = 42


@dataclass
class CouplingDiagnostics:
    step: int
    time: float
    hydrodynamic_force: np.ndarray
    ice_contact_force: np.ndarray
    total_hull_force: np.ndarray
    energy_ledger: dict
    energy_error: float
    lbm_max_velocity: float
    lbm_kinetic_energy: float
    dem_n_contacts: int
    dem_kinetic_energy: float
    fem_max_displacement: float
    fem_max_stress: float
    fem_strain_energy: float
    status: str


class RunManifest:
    """Orchestrator for the 3D flow-solid-ice coupled simulation."""

    def __init__(self, config: Optional[RunManifestConfig] = None):
        self.cfg = config or RunManifestConfig()
        self.status = CouplingStatus.RUNNING
        self.current_step = 0
        self.current_time = 0.0

        self.lbm = LBM3D()
        self.dem = DEM3D(DEM3DConfig(gravity=False, buoyancy=False, drag_coefficient=0.001, water_density=1.0))
        self.fem = FEM3DHull()

        self.ledger = EnergyLedger()
        self.initial_energy = 0.0
        self._energy_samples: List[float] = []

        self.history: List[CouplingDiagnostics] = []
        self.exchanges: List[FieldExchange] = []
        self.force_history: List[Tuple[float, np.ndarray]] = []

    def setup_wedge_breaking(self,
                              velocity: float = 0.05,
                              half_angle_deg: float = 30.0,
                              n_floes: int = 20,
                              floe_radius: float = 0.05) -> None:
        """Set up the 3D wedge icebreaking benchmark in lattice units."""
        self.lbm.cfg.u_inlet = velocity
        self.lbm = LBM3D(self.lbm.cfg)
        self.lbm.set_wedge_geometry(half_angle_deg=half_angle_deg)

        self.dem = DEM3D(DEM3DConfig(gravity=False, buoyancy=False, drag_coefficient=0.001, water_density=1.0))
        np.random.seed(self.cfg.seed)
        for i in range(n_floes):
            x = 12 + np.random.uniform(0, 3)  # ahead of wedge tip (in lattice units)
            y = np.random.uniform(-3, 3)
            z = np.random.uniform(-0.5, 0.5)
            self.dem.add_floe(
                position=(x, y, z),
                radius=floe_radius,
                velocity=(-velocity * 0.1, 0, 0),
            )

        self.wedge_params = {
            'apex': np.array([0.0, 0.0, 0.0]),
            'half_angle': math.radians(half_angle_deg),
            'length': 0.4 * self.lbm.cfg.nx,
        }

    def _exchange_lbm_to_fem(self) -> FieldExchange:
        """Transfer pressure from LBM to FEM."""
        pressure = self.lbm.get_pressure_field()
        n_fem_nodes = self.fem.n_nodes
        nx_w = int(0.4 * self.lbm.nx)
        pressure_on_hull = np.zeros(n_fem_nodes)

        for node_idx in range(n_fem_nodes):
            j_fem = node_idx // self.fem.nx_nodes
            i_fem = node_idx % self.fem.nx_nodes
            i_lbm = min(int(i_fem / max(self.fem.nx_nodes - 1, 1) * nx_w), nx_w - 1)
            j_lbm = min(int(j_fem / max(self.fem.ny_nodes - 1, 1) * self.lbm.ny), self.lbm.ny - 1)
            k_lbm = self.lbm.nz // 2
            if i_lbm < self.lbm.nx and j_lbm < self.lbm.ny and k_lbm < self.lbm.nz:
                pressure_on_hull[node_idx] = pressure[i_lbm, j_lbm, k_lbm]

        self.fem.apply_pressure(pressure_on_hull)
        mag = float(np.max(np.abs(pressure_on_hull)))
        checksum = hashlib.md5(pressure_on_hull.tobytes()).hexdigest()[:8]

        return FieldExchange(self.current_step, self.current_time, "lbm", "fem",
                            "pressure", mag, checksum)

    def _exchange_fem_to_lbm(self) -> FieldExchange:
        """Transfer displacement from FEM to LBM boundary."""
        displacements = self.fem.update_boundary_for_lbm()
        mag = float(np.max(np.abs(displacements)))
        checksum = hashlib.md5(displacements.tobytes()).hexdigest()[:8]
        return FieldExchange(self.current_step, self.current_time, "fem", "lbm",
                            "displacement", mag, checksum)

    def _exchange_dem_to_fem(self) -> FieldExchange:
        """Transfer ice-hull contact force from DEM to FEM."""
        hull_force = self.dem.get_hull_contact_force()
        n_nodes = self.fem.n_nodes
        force_per_node = np.zeros(n_nodes)
        force_mag = float(np.linalg.norm(hull_force))
        loaded_nodes = min(4, n_nodes)
        for i in range(loaded_nodes):
            force_per_node[n_nodes // 2 + i] = force_mag / max(loaded_nodes, 1)

        current_pressure = self.fem.pressure_load.copy()
        self.fem.apply_pressure(current_pressure + force_per_node)
        checksum = hashlib.md5(hull_force.tobytes()).hexdigest()[:8]
        return FieldExchange(self.current_step, self.current_time, "dem", "fem",
                            "contact_force", force_mag, checksum)

    def _exchange_lbm_to_dem(self) -> FieldExchange:
        """Transfer fluid velocity to DEM for drag computation."""
        positions = self.dem.get_positions()
        n_floes = len(self.dem.floes)
        fluid_vel_at_floes = np.zeros((n_floes, 3))

        for idx, pos in enumerate(positions):
            if not np.all(np.isfinite(pos)):
                fluid_vel_at_floes[idx] = 0.0
                continue
            # Map to LBM grid with wrapping
            i = int(pos[0]) % self.lbm.nx
            j = int(pos[1] + self.lbm.ny / 2) % self.lbm.ny
            k = int(pos[2] + self.lbm.nz / 2) % self.lbm.nz
            i = max(0, min(i, self.lbm.nx - 1))
            j = max(0, min(j, self.lbm.ny - 1))
            k = max(0, min(k, self.lbm.nz - 1))
            fluid_vel_at_floes[idx] = [self.lbm.ux[i, j, k],
                                        self.lbm.uy[i, j, k],
                                        self.lbm.uz[i, j, k]]

        mag = float(np.max(np.linalg.norm(fluid_vel_at_floes, axis=1))) if n_floes > 0 else 0.0
        checksum = hashlib.md5(fluid_vel_at_floes.tobytes()).hexdigest()[:8]
        self._fluid_vel_at_floes = fluid_vel_at_floes

        return FieldExchange(self.current_step, self.current_time, "lbm", "dem",
                            "fluid_velocity", mag, checksum)

    def _update_energy_ledger(self) -> None:
        self.ledger.fluid_kinetic = self.lbm.kinetic_energy()
        self.ledger.ice_kinetic = self.dem.total_kinetic_energy()
        self.ledger.ice_potential = self.dem.total_potential_energy()
        self.ledger.structure_strain = self.fem.strain_energy_value()
        self.ledger.dissipated = self.dem.dissipated_energy

    def _check_divergence(self) -> bool:
        if np.any(np.isnan(self.lbm.rho)) or np.any(np.isinf(self.lbm.rho)):
            return True
        if self.lbm.ux.max() > self.cfg.divergence_threshold:
            return True
        # Check DEM for NaN
        for floe in self.dem.floes:
            if not np.all(np.isfinite(floe.position)) or not np.all(np.isfinite(floe.velocity)):
                return True
        return False

    def _compute_energy_error(self) -> float:
        """Compute relative energy conservation error.

        Uses the ratio of energy change to initial kinetic energy.
        """
        current_total = abs(self.ledger.total)
        if self.initial_energy < 1e-20:
            return 0.0
        # Track the max energy seen as reference
        self._energy_samples.append(current_total)
        if len(self._energy_samples) < 2:
            return 0.0
        # Error = max deviation from initial / initial
        max_deviation = max(abs(e - self._energy_samples[0]) for e in self._energy_samples)
        return max_deviation / max(abs(self._energy_samples[0]), 1e-20)

    def step(self) -> CouplingDiagnostics:
        """Advance one coupling step."""
        dt = self.cfg.dt_coupling

        # 1. LBM sub-steps
        lbm_diag = None
        for _ in range(self.cfg.lbm_substeps):
            lbm_diag = self.lbm.step()

        # 2. Exchange: LBM pressure → FEM
        ex1 = self._exchange_lbm_to_fem()
        self.exchanges.append(ex1)

        # 3. FEM solve
        fem_result = self.fem.solve()

        # 4. Exchange: FEM displacement → LBM
        ex2 = self._exchange_fem_to_lbm()
        self.exchanges.append(ex2)

        # 5. Exchange: LBM velocity → DEM
        ex3 = self._exchange_lbm_to_dem()
        self.exchanges.append(ex3)

        # 6. DEM sub-steps
        dem_diag = self.dem.step(
            dt=dt / self.cfg.lbm_substeps,  # DEM dt matches total LBM time
            water_surface_z=0.0,
            fluid_velocity_field=getattr(self, '_fluid_vel_at_floes', None),
            wedge_params=self.wedge_params,
        )

        # 7. Exchange: DEM contact force → FEM
        ex4 = self._exchange_dem_to_fem()
        self.exchanges.append(ex4)

        # 8. Re-solve FEM with combined loads
        fem_result = self.fem.solve()

        # 9. Update energy ledger
        self._update_energy_ledger()

        # 10. Compute total hull force
        hydro_force = self.lbm.get_forces_on_solid()
        ice_force = self.dem.get_hull_contact_force()
        total_force = hydro_force + ice_force

        self.force_history.append((self.current_time, total_force.copy()))

        # 11. Compute energy error
        energy_error = self._compute_energy_error()

        # 12. Build diagnostics
        diag = CouplingDiagnostics(
            step=self.current_step,
            time=self.current_time,
            hydrodynamic_force=hydro_force,
            ice_contact_force=ice_force,
            total_hull_force=total_force,
            energy_ledger=self.ledger.snapshot(),
            energy_error=energy_error,
            lbm_max_velocity=lbm_diag.max_velocity if lbm_diag else 0.0,
            lbm_kinetic_energy=lbm_diag.kinetic_energy if lbm_diag else 0.0,
            dem_n_contacts=dem_diag['n_contacts'],
            dem_kinetic_energy=dem_diag['kinetic_energy'],
            fem_max_displacement=fem_result.max_displacement,
            fem_max_stress=fem_result.max_stress,
            fem_strain_energy=fem_result.strain_energy,
            status=self.status.value,
        )

        if self.current_step % self.cfg.sample_interval == 0:
            self.history.append(diag)

        # 13. Check divergence
        if self._check_divergence():
            self.status = CouplingStatus.DIVERGED

        self.current_step += 1
        self.current_time += dt

        return diag

    def run(self) -> dict:
        """Run the full coupling simulation."""
        self._update_energy_ledger()
        self.initial_energy = self.ledger.total
        self._energy_samples = [abs(self.initial_energy)]

        while self.current_step < self.cfg.n_steps:
            if self.status != CouplingStatus.RUNNING:
                break
            self.step()

        final_energy_error = self._compute_energy_error()

        if self.status == CouplingStatus.RUNNING:
            if self.current_step >= self.cfg.n_steps:
                self.status = CouplingStatus.MAX_STEPS_REACHED

        force_array = np.array([f[1] for f in self.force_history]) if self.force_history else np.zeros((0, 3))
        time_array = np.array([f[0] for f in self.force_history]) if self.force_history else np.zeros(0)

        return {
            'status': self.status.value,
            'n_steps': self.current_step,
            'total_time': self.current_time,
            'force_history': {
                'time': time_array.tolist(),
                'Fx': force_array[:, 0].tolist() if len(force_array) > 0 else [],
                'Fy': force_array[:, 1].tolist() if len(force_array) > 0 else [],
                'Fz': force_array[:, 2].tolist() if len(force_array) > 0 else [],
            },
            'energy_history': [h.energy_ledger for h in self.history],
            'final_energy_error': final_energy_error,
            'max_hull_force': float(np.max(np.linalg.norm(force_array, axis=1))) if len(force_array) > 0 else 0.0,
            'mean_hull_force': float(np.mean(np.linalg.norm(force_array, axis=1))) if len(force_array) > 0 else 0.0,
            'n_exchanges': len(self.exchanges),
            'config': {
                'dt_coupling': self.cfg.dt_coupling,
                'n_steps': self.cfg.n_steps,
                'lbm_substeps': self.cfg.lbm_substeps,
                'dem_substeps': self.cfg.dem_substeps,
                'energy_tolerance': self.cfg.energy_tolerance,
            },
        }

    def get_config_digest(self) -> str:
        config_dict = {
            'dt_coupling': self.cfg.dt_coupling,
            'n_steps': self.cfg.n_steps,
            'lbm_substeps': self.cfg.lbm_substeps,
            'dem_substeps': self.cfg.dem_substeps,
            'lbm_nx': self.lbm.cfg.nx,
            'lbm_ny': self.lbm.cfg.ny,
            'lbm_nz': self.lbm.cfg.nz,
            'seed': self.cfg.seed,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
