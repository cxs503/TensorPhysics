"""Three-field coupling orchestration: LBM wake + DEM ice + FEM hull.

The runner owns what this repository is chartered to own -- the *scientific
workflow semantics* of a multi-solver run:

* one clock (lattice time) with explicit sub-cycling for DEM and FEM,
* conservative, versioned field exchange between the solvers,
* an energy ledger that makes the coupled integration auditable,
* a :class:`tensorphysics.provenance.RunManifest` recording geometry,
  parameters, seeds and revisions for every run.

The default configuration is a *controlled experiment*: the 3-D case and the
2-D baseline share geometry, material parameters, numerical Reynolds number,
step counts and seed.  The only difference is the lateral degree of freedom --
2-D removes y-motion from the fluid (plane flow) and from the ice (plane
strain).  That isolates the 3-D lateral-relief effect.

All quantities are in lattice units; :class:`UnitScaling` declares the mapping
to SI and is reported with every run.
"""

from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..provenance import RunManifest, RunStatus, ValidationStatus
from .contracts import (
    CONTRACT_VERSION,
    FluidFieldPacket,
    ParticlePacket,
    StructurePacket,
    check_energy_handoff,
    check_mass_conservation,
    check_momentum_balance,
)
from .dem import DEMConfig, DEMWorld, IceSheetFracture, WedgeBow
from .fem import ElastostaticHull, HullFEMConfig
from .lattice import FreeSurfaceConfig, LatticeBoltzmannSolver, LatticeSpec


@dataclass
class UnitScaling:
    """Lattice -> SI mapping (declared, not measured)."""

    length_m: float = 0.25
    velocity_m_s: float = 0.5
    rho_ref_kg_m3: float = 1000.0

    def __post_init__(self) -> None:
        if self.length_m <= 0 or self.velocity_m_s <= 0 or self.rho_ref_kg_m3 <= 0:
            raise ValueError("scaling factors must be positive")

    @property
    def time_s(self) -> float:
        return self.length_m / self.velocity_m_s

    @property
    def mass_kg(self) -> float:
        return self.rho_ref_kg_m3 * self.length_m**3

    def force_to_si(self, f: float) -> float:
        return f * self.mass_kg * self.length_m / self.time_s**2

    def length_to_si(self, x: float) -> float:
        return x * self.length_m

    def time_to_si(self, t: float) -> float:
        return t * self.time_s

    def energy_to_si(self, e: float) -> float:
        return e * self.mass_kg * self.length_m**2 / self.time_s**2

    def as_dict(self) -> Dict[str, float]:
        return {
            "length_m_per_lu": self.length_m,
            "velocity_m_s_per_lu_ts": self.velocity_m_s,
            "rho_ref_kg_m3": self.rho_ref_kg_m3,
            "mass_kg_per_mu": self.mass_kg,
            "time_s_per_ts": self.time_s,
        }


@dataclass
class CouplingConfig:
    """Complete, serialisable description of one coupled run."""

    # geometry
    dim: int = 3
    nx: int = 60
    ny: int = 28
    nz: int = 16
    water_level: int = 15
    x_bow_tip: float = 34.0
    bow_half_angle_deg: float = 30.0
    x_bow_transom: float = 52.0
    keel_z: float = 3.0
    ice_width: float = 20.0
    ice_thickness: float = 2.0
    ice_cell_length: float = 2.0
    ice_crack_spacing_factor: float = 1.0
    # fluid
    tau: float = 0.55
    u_ship: float = 0.05
    free_surface: bool = False
    # ice material (lattice units)
    ice_youngs: float = 9.0e4
    ice_poisson: float = 0.30
    ice_flexural_strength: float = 7.0
    ice_density_ratio: float = 0.90
    fracture_energy: float = 4.0e-6
    # dem
    dem_dt: float = 0.10
    dem_youngs: float = 90.0
    dem_restitution: float = 0.25
    dem_friction: float = 0.35
    # fem
    fem_nx: int = 6
    fem_ny: int = 8
    fem_nz: int = 4
    fem_youngs: float = 300.0
    fem_density: float = 0.9
    fem_damping_ratio: float = 0.02
    # coupling
    n_steps: int = 100
    fluid_steps_per_couple: int = 5
    coupling_mode: str = "full"
    seed: int = 20273104
    #: fluid body force (rigid-lid approximation -> no free-surface gravity waves)
    fluid_gravity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: gravity used by the ice (buoyancy + plate-on-elastic-foundation modulus)
    ice_gravity: Tuple[float, float, float] = (0.0, 0.0, -0.049)
    scaling: UnitScaling = field(default_factory=UnitScaling)

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if self.dim == 2 and self.ny != 1:
            raise ValueError("2-D mode requires ny == 1")
        if self.coupling_mode not in ("full", "fluid_only", "dem_only", "fem_only", "frozen"):
            raise ValueError("unknown coupling_mode")
        if self.n_steps < 1 or self.fluid_steps_per_couple < 1:
            raise ValueError("n_steps and fluid_steps_per_couple must be >= 1")

    @property
    def dt_couple(self) -> float:
        return float(self.fluid_steps_per_couple)

    def lattice_spec(self) -> LatticeSpec:
        return LatticeSpec(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            dim=self.dim,
            tau=self.tau,
            gravity=self.fluid_gravity,
            y_boundary="wall",
            inlet_velocity=(self.u_ship, 0.0, 0.0),
            top_boundary="slip",
        )

    def as_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "scaling"}
        d["scaling"] = self.scaling.as_dict()
        d = json.loads(json.dumps(d, default=float))
        return d


@dataclass
class CouplingRunResult:
    """Everything the acceptance gate needs from one coupled run."""

    label: str
    config: CouplingConfig
    times: np.ndarray
    force_history: np.ndarray
    force_components: Dict[str, np.ndarray]
    energy_history: Dict[str, np.ndarray]
    fragment_diameters: np.ndarray
    ledger: Dict[str, Any]
    manifest: RunManifest
    diagnostics: Dict[str, Any]
    wall_time_s: float

    @property
    def mean_force_x(self) -> float:
        return float(np.mean(self.force_history[:, 0]))

    def summary(self) -> Dict[str, Any]:
        f = self.force_history[:, 0]
        half = len(f) // 2
        return {
            "label": self.label,
            "dim": self.config.dim,
            "mean_Fx": float(np.mean(f)),
            "mean_Fx_second_half": float(np.mean(f[half:])) if half else float(np.mean(f)),
            "peak_Fx": float(np.max(f)),
            "std_Fx": float(np.std(f)),
            "mean_fragment_diameter": float(np.mean(self.fragment_diameters)) if len(self.fragment_diameters) else 0.0,
            "n_fragments": int(len(self.fragment_diameters)),
            "energy_balance_error": float(self.ledger["relative_error"]),
            "diverged": bool(self.diagnostics["diverged"]),
            "max_particle_speed": float(self.diagnostics["max_particle_speed"]),
            "max_fem_displacement": float(self.diagnostics["max_fem_displacement"]),
        }


class CouplingRunner:
    """Drives LBM + DEM + FEM through a sub-cycled coupling loop."""

    def __init__(self, config: CouplingConfig, label: Optional[str] = None):
        self.cfg = config
        self.label = label or ("3d" if config.dim == 3 else "2d")
        self.rng = np.random.default_rng(config.seed)
        self.spec = config.lattice_spec()
        self.fluid = LatticeBoltzmannSolver(self.spec)
        self.hull = WedgeBow(
            x_tip=config.x_bow_tip,
            half_angle_deg=config.bow_half_angle_deg,
            z_bottom=config.keel_z,
            z_top=float(config.water_level),
            x_max=config.x_bow_transom,
            y_center=config.ny / 2.0,
        )
        self.fluid.set_solid(self.hull.mask(self.spec.shape))
        self.dem = DEMWorld(
            DEMConfig(
                youngs_modulus=config.dem_youngs,
                poisson=config.ice_poisson,
                restitution=config.dem_restitution,
                friction=config.dem_friction,
                rho_particle=config.ice_density_ratio,
                rho_fluid=1.0,
                gravity=config.ice_gravity,
                fluid_viscosity=self.spec.viscosity,
                dim=config.dim,
            ),
            bounds=[[0.0, 0.0, 0.0], [float(config.nx - 1), float(config.ny - 1), float(config.nz - 1)]],
        )
        self.fracture = IceSheetFracture(
            thickness=config.ice_thickness,
            flexural_strength=config.ice_flexural_strength,
            youngs_modulus=config.ice_youngs,
            poisson=config.ice_poisson,
            rho_fluid=1.0,
            gravity=abs(config.ice_gravity[2]),
            width=config.ice_width,
            cell_length=config.ice_cell_length,
            dim=config.dim,
            lateral_crack_spacing=config.ice_crack_spacing_factor * config.ice_thickness,
        )
        self.fem = ElastostaticHull(
            HullFEMConfig(
                nx=config.fem_nx,
                ny=config.fem_ny,
                nz=config.fem_nz,
                size=(6.0, config.ice_width, float(config.water_level) * 0.5),
                youngs_modulus=config.fem_youngs,
                poisson=0.3,
                density=config.fem_density,
                damping_ratio=config.fem_damping_ratio,
                clamp_face="xmin",
            )
        )
        self.y_center = config.ny / 2.0
        #: x stations of the intact sheet, consumed as the sheet advances
        n_slices = int(np.ceil(config.x_bow_tip / config.ice_cell_length)) + 2
        self.slice_positions = np.array(
            [config.x_bow_tip - s * config.ice_cell_length for s in range(n_slices)]
        )
        self.slice_broken = np.zeros(n_slices, dtype=bool)
        self.time = 0.0
        self._slice_progress = 0.0
        self.fragment_diameters: List[float] = []
        self.surface_energy = 0.0
        self.removed_energy = 0.0
        self.dem_damping_work = 0.0
        self.injected_energy = 0.0
        self.fem_damping_work = 0.0
        self._fem_load_vector = np.zeros(self.fem.ndof)

    # -- fragment generation -------------------------------------------------
    def _cell_geometry(self) -> Tuple[float, int, float]:
        """(cell width, lateral cell count, cell length) of the fracture grid."""
        frac = self.fracture
        if self.cfg.dim == 3:
            return float(frac.cell_width), int(frac.n_lateral), float(self.cfg.ice_cell_length)
        return float(self.cfg.ice_width), 1, float(self.cfg.ice_cell_length)

    def _spawn_fragments(self, n_cells: int, slice_index: int = 0) -> ParticlePacket:
        """Replace one broken ice slice by DEM spheres (n01/n02 hand-off).

        Coarse-grained DEM: one sphere carries the *ice mass* of a broken cell
        but a smaller contact radius, so the initial packing is not overlapping.
        Fragment size reported for C2 is the mass-equivalent diameter, which is
        the physical floe size.
        """
        cfg = self.cfg
        frac = self.fracture
        width, n_lat, length = self._cell_geometry()
        cell_volume = length * width * cfg.ice_thickness
        r_contact = 0.45 * min(length, width, cfg.ice_thickness)
        # the slice keeps the streamwise station it had in the intact sheet,
        # which also guarantees that consecutive slices do not overlap at birth
        x_spawn = max(1.0, cfg.x_bow_tip - (slice_index + 1) * length)
        positions, velocities, masses = [], [], []
        for j in range(n_cells):
            y = self.y_center - cfg.ice_width / 2.0 + (j + 0.5) * width if cfg.dim == 3 else self.y_center
            jitter = self.rng.normal(0.0, 0.01, size=3)
            positions.append(
                [
                    x_spawn + jitter[0],
                    y + jitter[1],
                    cfg.water_level - 0.4 * cfg.ice_thickness + jitter[2] * 0.1,
                ]
            )
            velocities.append([cfg.u_ship, 0.0, 0.0])
            masses.append(cfg.ice_density_ratio * cell_volume)
        pos = np.array(positions)
        sdf, _ = self.hull.signed_distance(pos)
        if np.any(sdf + r_contact > -0.05):
            raise ValueError("fragment spawned inside or touching the hull")
        vel = np.array(velocities)
        m = np.array(masses)
        rad = np.full(len(m), r_contact)
        self.dem.add_particles(pos, rad, vel, mass=m)
        # C2 metric: mass-equivalent (physical) fragment diameter
        d_eq = (6.0 * m / (np.pi * cfg.ice_density_ratio)) ** (1.0 / 3.0)
        self.fragment_diameters.extend(d_eq.tolist())
        g_z = abs(self.cfg.ice_gravity[2])
        v_ice = m / cfg.ice_density_ratio
        pe_birth = (m - 1.0 * v_ice) * g_z * pos[:, 2]
        # buoyancy-corrected potential energy is injected together with the mass
        self.injected_energy += float(np.sum(0.5 * m * np.sum(vel**2, axis=1) + pe_birth))
        self.surface_energy += cfg.fracture_energy * length * width * 2.0 * n_cells
        packet = ParticlePacket(time=self.time, position=pos, velocity=vel, radius=rad, mass=m)
        packet.validate()
        check_mass_conservation(
            {"volume": np.full(len(m), cell_volume), "density": cfg.ice_density_ratio}, packet
        )
        return packet

    # -- forcing helpers -----------------------------------------------------
    def _fluid_velocity_at_particles(self) -> Optional[np.ndarray]:
        if self.dem.count == 0:
            return None
        idx = np.where(self.dem.active)[0]
        return self.fluid.sample_velocity(self.dem.position[idx])

    def _apply_fracture_front(self) -> float:
        """Advance the intact sheet and break the slices the bow has passed.

        The sheet's leading edge starts at the bow tip and advances at the ship
        speed in the ship-fixed frame; slice ``s`` (counting upstream from the
        bow) is consumed when ``u_ship * t >= (s + 1) * cell_length``.
        """
        cfg = self.cfg
        width, n_lat, length = self._cell_geometry()
        consumed = int(np.floor(cfg.u_ship * self.time / length))
        n_new = min(consumed, len(self.slice_broken)) - int(self.slice_broken.sum())
        force = 0.0
        for _ in range(max(0, n_new)):
            idx = int(self.slice_broken.sum())
            self.slice_broken[idx] = True
            self._spawn_fragments(n_lat, slice_index=idx)
            force += self.fracture.q_c * width * n_lat
        return force

    # -- main loop -----------------------------------------------------------
    def run(self) -> CouplingRunResult:
        cfg = self.cfg
        t0 = _time.perf_counter()
        times, forces = [], []
        comp = {"fracture": [], "dem_contact": [], "fluid": []}
        energy = {k: [] for k in ("kinetic", "potential", "strain", "dissipated", "removed", "injected", "total")}
        e0 = None
        diverged = False
        max_speed = 0.0
        max_disp = 0.0
        coupling_enabled = cfg.coupling_mode in ("full", "frozen")
        n_sub = max(1, int(round(cfg.dt_couple / cfg.dem_dt)))
        fem_dt = cfg.dt_couple
        self.fem.initialise_newmark(fem_dt)

        for step in range(cfg.n_steps):
            fluid_force = np.zeros(3)
            if cfg.coupling_mode in ("full", "fluid_only"):
                for _ in range(cfg.fluid_steps_per_couple):
                    fluid_force = self.fluid.step()
            # structural update from the previous step's contact loads
            if cfg.coupling_mode in ("full", "fem_only", "frozen"):
                self.fem.step_newmark(self._fem_load_vector)
                max_disp = max(max_disp, float(np.abs(self.fem.u).max()))
            # ice fracture front
            fracture_force = 0.0
            if cfg.coupling_mode in ("full", "dem_only", "frozen"):
                fracture_force = self._apply_fracture_front()
            # DEM sub-cycled in the coupling interval
            dem_hull_force = np.zeros(3)
            if cfg.coupling_mode in ("full", "dem_only", "frozen"):
                # single-field degradation: with the fluid frozen the drag is
                # not applied, so dem_only reproduces a standalone DEM run
                uf = self._fluid_velocity_at_particles() if cfg.coupling_mode == "full" else None
                water = float(cfg.water_level) if cfg.free_surface else None
                dem_hull_force = np.zeros(3)
                for _ in range(n_sub):
                    diag = self.dem.step(
                        cfg.dem_dt,
                        fluid_velocity=None if uf is None else uf,
                        water_level=water if water is not None else float(cfg.water_level),
                        hull=self.hull,
                    )
                    max_speed = max(max_speed, diag["max_speed"])
                    dem_hull_force = dem_hull_force + diag["hull_reaction"]
                dem_hull_force = dem_hull_force / n_sub
                self._fem_load_vector = self._map_contacts_to_fem()
            else:
                self._fem_load_vector = np.zeros(self.fem.ndof)

            # every term is a resistance on the bow, positive in +x
            fx = dem_hull_force[0] + fluid_force[0] + fracture_force
            forces.append([fx, dem_hull_force[1] + fluid_force[1], dem_hull_force[2] + fluid_force[2]])
            comp["fracture"].append(fracture_force)
            comp["dem_contact"].append(dem_hull_force[0])
            comp["fluid"].append(fluid_force[0])
            self.time += cfg.dt_couple
            times.append(self.time)

            if not np.all(np.isfinite(forces[-1])) or max_speed > 5.0:
                diverged = True

            e_total = self._energy_state()
            if e0 is None:
                e0 = e_total["total"]
            energy["kinetic"].append(e_total["kinetic"])
            energy["potential"].append(e_total["potential"])
            energy["strain"].append(e_total["strain"])
            energy["dissipated"].append(e_total["dissipated"])
            energy["removed"].append(e_total["removed"])
            energy["injected"].append(self.injected_energy)
            energy["total"].append(e_total["total"])
            if diverged:
                break

        wall = _time.perf_counter() - t0
        ledger = self._energy_ledger(e0, energy)
        manifest = self._build_manifest(ledger)
        diagnostics = {
            "diverged": bool(diverged),
            "max_particle_speed": float(max_speed),
            "max_fem_displacement": float(max_disp),
            "n_particles_final": int(self.dem.count),
            "reynolds_lattice": float(cfg.u_ship * cfg.ice_width / self.spec.viscosity),
            "froude_lattice": float(cfg.u_ship / np.sqrt(abs(cfg.ice_gravity[2]) * cfg.ice_thickness)),
            "stable_dt_dem": float(self.dem.stable_dt),
            "dem_substeps": n_sub,
            "fracture": {
                "L_b": float(self.fracture.L_b),
                "q_c_per_unit_width": float(self.fracture.q_c),
                "cell_width": float(self.fracture.cell_width),
                "n_lateral": int(self.fracture.n_lateral),
                "break_force_per_cell": float(self.fracture.break_force),
            },
        }
        return CouplingRunResult(
            label=self.label,
            config=cfg,
            times=np.array(times),
            force_history=np.array(forces),
            force_components={k: np.array(v) for k, v in comp.items()},
            energy_history={k: np.array(v) for k, v in energy.items()},
            fragment_diameters=np.array(self.fragment_diameters),
            ledger=ledger,
            manifest=manifest,
            diagnostics=diagnostics,
            wall_time_s=float(wall),
        )

    def _map_contacts_to_fem(self) -> np.ndarray:
        """DEM contact records -> nodal loads (conservative trilinear mapping)."""
        pts = []
        frc = []
        for rec in self.dem.contact_records:
            if len(rec["positions"]) == 0:
                continue
            pts.append(rec["positions"])
            frc.append(-rec["forces"])  # reaction on the hull
        if not pts:
            return np.zeros(self.fem.ndof)
        points = np.vstack(pts)
        forces = np.vstack(frc)
        # contacts are in lattice x/y/z; the FEM block spans the same box
        loc = points.copy()
        loc[:, 0] = np.clip(loc[:, 0] - (self.cfg.x_bow_tip - 6.0), 0.0, self.fem.cfg.size[0])
        loc[:, 1] = np.clip(loc[:, 1] - (self.y_center - self.cfg.ice_width / 2.0), 0.0, self.fem.cfg.size[1])
        loc[:, 2] = np.clip(loc[:, 2], 0.0, self.fem.cfg.size[2])
        nodal = self.fem.map_contact_loads(loc, forces)
        # conservative hand-off gate: nodal resultant == applied contact resultant
        applied = forces.sum(axis=0)
        scale = max(float(np.linalg.norm(applied)), 1e-30)
        err = float(np.linalg.norm(nodal.reshape(-1, 3).sum(axis=0) - applied)) / scale
        if err > 1e-9:
            raise ValueError(f"load mapping lost momentum: relative error {err:.3e}")
        return nodal

    def _energy_state(self) -> Dict[str, float]:
        kinetic = self.dem.kinetic_energy() + self.fem.kinetic_energy()
        potential = self.dem.potential_energy()
        strain = self.fem.strain_energy()
        dissipated = (
            self.dem.dissipated_contact
            + self.dem.dissipated_drag
            + self.fem_damping_work
        )
        removed = self.removed_energy
        total = kinetic + potential + strain + dissipated + removed + self.surface_energy
        return {
            "kinetic": kinetic,
            "potential": potential,
            "strain": strain,
            "dissipated": dissipated,
            "removed": removed,
            "total": total,
        }

    def _energy_ledger(self, e0: Optional[float], energy: Dict[str, np.ndarray]) -> Dict[str, Any]:
        cfg = self.cfg
        final = float(energy["total"][-1]) if len(energy["total"]) else 0.0
        injected = float(self.injected_energy)
        e_ref = max(abs(injected), abs(final), 1e-30)
        # the one-way fluid coupling transfers work W_drag to the particles;
        # it is an external source for the solid subsystem and must be booked
        residual = (
            final - (e0 or 0.0) - injected - self.dem.drag_work - self.dem.wall_work
        )
        return {
            "E0": float(e0 or 0.0),
            "E_final": final,
            "E_injected": injected,
            "E_lost_to_clearing": float(self.removed_energy),
            "E_dissipated": float(energy["dissipated"][-1]) if len(energy["dissipated"]) else 0.0,
            "E_fluid_work": float(self.dem.drag_work),
            "E_hull_contact_work": float(self.dem.wall_work),
            "E_surface": float(self.surface_energy),
            "residual": float(residual),
            "E_reference": e_ref,
            "relative_error": float(abs(residual) / e_ref),
            "fluid_energy_excluded": True,
            "note": (
                "Ledger covers the solid subsystems (DEM fragments + FEM hull). "
                "The one-way fluid coupling (drag on particles, momentum exchange "
                "on the hull) is a separate open subsystem and is reported but "
                "not closed here."
            ),
        }

    def _build_manifest(self, ledger: Dict[str, Any]) -> RunManifest:
        cfg = self.cfg.as_dict()
        digest = RunManifest.digest_config(cfg)
        manifest = RunManifest(
            run_id=f"fsi-ice-{self.label}-{digest[:12]}",
            status=RunStatus.SUCCEEDED,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            config=cfg,
            config_sha256=digest,
            code_revisions={"TensorPhysics": "worktree"},
            environment={
                "python": "numpy",
                "numpy": np.__version__,
                "contract": CONTRACT_VERSION,
            },
            random_seeds={"dem": self.cfg.seed},
            resources={"n_steps": self.cfg.n_steps, "dem_substeps": self.diagnostics_substeps()},
            validation_status=ValidationStatus.PASSED
            if ledger["relative_error"] < 0.05
            else ValidationStatus.FAILED,
        )
        return manifest

    def diagnostics_substeps(self) -> int:
        return max(1, int(round(self.cfg.dt_couple / self.cfg.dem_dt)))


def run_pair(
    base: Optional[CouplingConfig] = None,
    n_steps: int = 100,
) -> Tuple[CouplingRunResult, CouplingRunResult]:
    """Run the controlled 3-D / 2-D pair sharing every parameter but ``dim``."""
    base = base or CouplingConfig()
    cfg3 = _replace(base, dim=3, n_steps=n_steps, ny=base.ny)
    cfg2 = _replace(base, dim=2, n_steps=n_steps, ny=1)
    r3 = CouplingRunner(cfg3, label="3d").run()
    r2 = CouplingRunner(cfg2, label="2d").run()
    return r3, r2


def _replace(cfg: CouplingConfig, **kwargs: Any) -> CouplingConfig:
    data = {k: v for k, v in cfg.__dict__.items()}
    data.update(kwargs)
    return CouplingConfig(**data)