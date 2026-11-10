"""Soft-sphere DEM for broken ice with fluid and hull coupling.

Contact law is Hertz with viscous (Tsuji) normal damping and a Coulomb-capped
tangential friction impulse.  Rotation is *not* integrated: fragments are
spheres with translational degrees of freedom only.  That is a documented
limitation -- it removes rolling resistance, which matters for packed rubble but
not for the clearing mechanism this benchmark studies.

Environment forces are what make the three-field coupling meaningful:

* buoyancy from the free-surface height supplied by the fluid solver,
* Schiller--Naumann drag from the fluid velocity supplied by the same solver,
* hull contact through an analytic signed distance function of the wedge bow,
  which is the only place the structure talks to the ice.

All quantities are in lattice units; the caller owns the mapping to SI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class DEMConfig:
    """Material and integration parameters (lattice units)."""

    youngs_modulus: float = 400.0
    poisson: float = 0.3
    restitution: float = 0.25
    friction: float = 0.35
    rho_particle: float = 0.90  # sea ice / water density ratio
    rho_fluid: float = 1.0
    gravity: Tuple[float, float, float] = (0.0, 0.0, -1e-3)
    fluid_viscosity: float = 4.5e-3
    drag_enabled: bool = True
    dim: int = 3
    friction_time_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.youngs_modulus <= 0:
            raise ValueError("youngs_modulus must be positive")
        if not (0.0 <= self.restitution < 1.0):
            raise ValueError("restitution must be in [0, 1)")
        if np.size(self.gravity) != 3:
            raise ValueError("gravity must be a 3-vector")
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")


class HertzMindlinContact:
    """Hertz normal / Mindlin tangential stiffness helpers."""

    def __init__(self, youngs_modulus: float, poisson: float):
        self.youngs_modulus = float(youngs_modulus)
        self.poisson = float(poisson)
        self.e_star = self.youngs_modulus / (2.0 * (1.0 - poisson**2))
        self.g_star = self.youngs_modulus / (2.0 * (2.0 - poisson) * (1.0 + poisson))

    def normal_force(self, radius_eff: float, overlap: float):
        """Hertz normal force ``F = 4/3 E* sqrt(R*) delta^(3/2)``."""
        overlap = np.maximum(np.asarray(overlap, dtype=float), 0.0)
        return (4.0 / 3.0) * self.e_star * np.sqrt(radius_eff) * overlap**1.5

    def normal_stiffness(self, radius_eff: float, overlap: float):
        overlap = np.maximum(np.asarray(overlap, dtype=float), 1e-30)
        return 2.0 * self.e_star * np.sqrt(np.asarray(radius_eff) * overlap)

    def tangential_stiffness(self, radius_eff: float, overlap: float):
        overlap = np.maximum(np.asarray(overlap, dtype=float), 1e-30)
        return 8.0 * self.g_star * np.sqrt(np.asarray(radius_eff) * overlap)

    def damping_coefficient(self, mass_eff, radius_eff, overlap, restitution: float):
        """Tsuji damping so the coefficient of restitution is recovered."""
        stiff = self.normal_stiffness(radius_eff, overlap)
        ln_e = np.log(max(restitution, 1e-9))
        beta = ln_e / np.sqrt(ln_e**2 + np.pi**2)
        return -2.0 * beta * np.sqrt((5.0 / 6.0) * stiff * np.asarray(mass_eff))


class WedgeBow:
    """Analytic hull geometry: vertical-stem wedge bow.

    Solid region: ``x > x_tip + |y| / tan(alpha)`` for ``z_bottom <= z <= z_top``
    and ``x <= x_max``.  The signed distance is positive inside the hull.
    In the 2-D baseline the very same hull is used, only the lateral degree of
    freedom of the flow and of the fragments is removed -- that is the
    controlled variable of the whole experiment.
    """

    def __init__(
        self,
        x_tip: float,
        half_angle_deg: float = 30.0,
        z_bottom: float = 0.0,
        z_top: float = 1e9,
        x_max: float = 1e9,
        y_center: float = 0.0,
    ):
        if not (1.0 < half_angle_deg < 89.0):
            raise ValueError("half_angle_deg must be in (1, 89)")
        self.x_tip = float(x_tip)
        self.half_angle = float(half_angle_deg)
        self.cot = 1.0 / np.tan(np.radians(half_angle_deg))
        self.z_bottom = float(z_bottom)
        self.z_top = float(z_top)
        self.x_max = float(x_max)
        self.y_center = float(y_center)
        self.displacement = np.zeros(3)  # structural feedback (mean hull motion)

    def signed_distance(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Signed distance (positive inside) of the convex hull block.

        The body is the intersection of five half-spaces: the two wedge faces,
        the transom at ``x_max``, and the keel/deck planes.  Limiting the body
        in ``x`` is what leaves an open outflow path -- without it the hull
        would seal the channel and the incompressible solver would have nowhere
        to discharge the inflow.
        """
        p = np.atleast_2d(np.asarray(points, dtype=float)) - self.displacement
        x, y, z = p[:, 0], p[:, 1] - self.y_center, p[:, 2]
        s = np.sign(y)
        s[s == 0] = 1.0
        norm = np.sqrt(1.0 + self.cot**2)
        d_wedge = (x - self.x_tip - np.abs(y) * self.cot) / norm
        n_wedge = np.stack(
            [-np.full_like(x, 1.0 / norm), s * self.cot / norm, np.zeros_like(x)], axis=1
        )
        d_xmax = self.x_max - x
        n_xmax = np.stack([-np.ones_like(x), np.zeros_like(x), np.zeros_like(x)], axis=1)
        d_keel = z - self.z_bottom
        n_keel = np.stack([np.zeros_like(x), np.zeros_like(x), np.ones_like(x)], axis=1)
        d_deck = self.z_top - z
        n_deck = np.stack([np.zeros_like(x), np.zeros_like(x), -np.ones_like(x)], axis=1)
        cands = np.stack([d_wedge, d_xmax, d_keel, d_deck], axis=1)
        idx = np.argmin(cands, axis=1)
        d = cands[np.arange(len(x)), idx]
        normals = np.stack([n_wedge, n_xmax, n_keel, n_deck], axis=1)
        return d, normals[np.arange(len(x)), idx]

    def mask(self, shape: Tuple[int, int, int]) -> np.ndarray:
        nz, ny, nx = shape
        zz, yy, xx = np.meshgrid(
            np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
        )
        pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(float)
        d, _ = self.signed_distance(pts)
        return (d > 0.0).reshape(shape)


class DEMWorld:
    """Spherical ice fragments in a bounded box with hull and fluid coupling."""

    def __init__(self, config: DEMConfig, bounds: Sequence[Sequence[float]]):
        self.cfg = config
        b = np.asarray(bounds, dtype=float)
        if b.shape != (2, 3):
            raise ValueError("bounds must be [[xmin,ymin,zmin],[xmax,ymax,zmax]]")
        self.bounds = b
        self.contact = HertzMindlinContact(config.youngs_modulus, config.poisson)
        self.position = np.zeros((0, 3))
        self.velocity = np.zeros((0, 3))
        self.radius = np.zeros((0,))
        self.mass = np.zeros((0,))
        self.active = np.zeros((0,), dtype=bool)
        self.dissipated_contact = 0.0
        self.dissipated_friction = 0.0
        self.dissipated_drag = 0.0
        self.drag_work = 0.0
        self.wall_work = 0.0
        self.removed_mass = 0.0
        self.removed_energy = 0.0
        self.last_hull_force = np.zeros(3)
        self.contact_records: list[dict] = []

    # -- population ---------------------------------------------------------
    def add_particles(self, position: np.ndarray, radius: np.ndarray,
                      velocity: Optional[np.ndarray] = None,
                      mass: Optional[np.ndarray] = None) -> np.ndarray:
        pos = np.atleast_2d(np.asarray(position, dtype=float))
        rad = np.atleast_1d(np.asarray(radius, dtype=float))
        if pos.shape[1] != 3:
            raise ValueError("particle positions must be (n, 3)")
        if rad.shape[0] != pos.shape[0]:
            raise ValueError("radius count must match particle count")
        if np.any(rad <= 0.0):
            raise ValueError("particle radius must be positive")
        if mass is None:
            mass = self.cfg.rho_particle * 4.0 / 3.0 * np.pi * rad**3
        else:
            mass = np.atleast_1d(np.asarray(mass, dtype=float))
            if mass.shape != rad.shape:
                raise ValueError("mass must match the number of particles")
        vel = np.zeros_like(pos) if velocity is None else np.atleast_2d(np.asarray(velocity, dtype=float))
        if vel.shape != pos.shape:
            raise ValueError("velocity shape must match position shape")
        n0 = len(self.radius)
        self.position = np.vstack([self.position, pos])
        self.velocity = np.vstack([self.velocity, vel])
        self.radius = np.concatenate([self.radius, rad])
        self.mass = np.concatenate([self.mass, mass])
        self.active = np.concatenate([self.active, np.ones(len(rad), dtype=bool)])
        return np.arange(n0, n0 + len(rad))

    @property
    def count(self) -> int:
        return int(self.active.sum())

    def total_mass(self) -> float:
        return float(self.mass[self.active].sum())

    def kinetic_energy(self) -> float:
        if self.count == 0:
            return 0.0
        v = self.velocity[self.active]
        m = self.mass[self.active]
        return float(0.5 * np.sum(m[:, None] * v * v))

    def potential_energy(self) -> float:
        """Buoyancy-corrected potential energy."""
        if self.count == 0:
            return 0.0
        g = abs(self.cfg.gravity[2])
        z = self.position[self.active, 2]
        m = self.mass[self.active]
        v = self.mass[self.active] / max(self.cfg.rho_particle, 1e-12)
        buoy = self.cfg.rho_fluid * v
        return float(np.sum((m - buoy) * g * z))

    # -- forces -------------------------------------------------------------
    def _environment_forces(
        self,
        fluid_velocity: Optional[np.ndarray],
        water_level: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        act = self.active
        pos, vel = self.position[act], self.velocity[act]
        rad, mass = self.radius[act], self.mass[act]
        n = len(rad)
        force = np.zeros((n, 3))
        if n == 0:
            return force, np.zeros(n), np.zeros((0, 3))
        g = np.asarray(cfg.gravity, dtype=float)
        force += mass[:, None] * g
        # coarse-grained DEM: the displaced volume is the *ice* volume m/rho_ice,
        # not the collision-sphere volume, so fragments float like sea ice
        vol = mass / max(cfg.rho_particle, 1e-12)
        if water_level is not None:
            z_lo, z_hi = pos[:, 2] - rad, pos[:, 2] + rad
            sub = np.clip((np.minimum(water_level, z_hi) - z_lo) / np.maximum(2.0 * rad, 1e-12), 0.0, 1.0)
            force[:, 2] += cfg.rho_fluid * vol * sub * abs(g[2])
        f_drag_total = np.zeros((n, 3))
        drag_power = np.zeros(n)
        if cfg.drag_enabled and fluid_velocity is not None:
            urel = np.asarray(fluid_velocity, dtype=float) - vel
            speed = np.linalg.norm(urel, axis=1)
            d = 2.0 * rad
            mu = cfg.fluid_viscosity * cfg.rho_fluid
            re = np.maximum(cfg.rho_fluid * speed * d / max(mu, 1e-30), 1e-9)
            cd = np.where(re < 1.0, 24.0 / re, 24.0 / re * (1.0 + 0.15 * re**0.687))
            area = np.pi * rad**2
            coeff = 0.5 * cfg.rho_fluid * cd * area * speed
            f_drag = coeff[:, None] * urel
            force += f_drag
            f_drag_total = f_drag
            drag_power = np.einsum("ij,ij->i", f_drag, urel)  # >0 = dissipated
        return force, drag_power, f_drag_total

    def _pair_forces(self, dt: float) -> Tuple[np.ndarray, float]:
        act = self.active
        idx = np.where(act)[0]
        pos, vel = self.position[idx], self.velocity[idx]
        rad, mass = self.radius[idx], self.mass[idx]
        n = len(idx)
        force = np.zeros((n, 3))
        if n < 2:
            return force, 0.0
        iu, ju = np.triu_indices(n, k=1)
        dx = pos[ju] - pos[iu]
        dist = np.linalg.norm(dx, axis=1)
        overlap = rad[iu] + rad[ju] - dist
        hit = overlap > 0.0
        if not np.any(hit):
            return force, 0.0
        iu, ju, dx, dist, overlap = iu[hit], ju[hit], dx[hit], dist[hit], overlap[hit]
        dist = np.maximum(dist, 1e-12)
        nhat = dx / dist[:, None]
        r_eff = (rad[iu] * rad[ju]) / (rad[iu] + rad[ju])
        m_eff = (mass[iu] * mass[ju]) / (mass[iu] + mass[ju])
        kn = self.contact.normal_force(r_eff, overlap)
        cn = self.contact.damping_coefficient(m_eff, r_eff, overlap, self.cfg.restitution)
        vrel = vel[ju] - vel[iu]
        vn = np.einsum("ij,ij->i", vrel, nhat)
        # damping is capped by the spring force so the total normal force can
        # never become attractive; only the *applied* damping is booked as
        # dissipation, which is what makes the energy ledger close
        approach = np.maximum(-vn, 0.0)
        f_damp = np.minimum(cn * approach, kn)
        fn = kn + f_damp
        vt = vrel - vn[:, None] * nhat
        kt = self.contact.tangential_stiffness(r_eff, overlap)
        ft = kt[:, None] * vt * dt
        ft_mag = np.linalg.norm(ft, axis=1)
        cap = self.cfg.friction * fn
        scale = np.where(ft_mag > cap, cap / np.maximum(ft_mag, 1e-30), 1.0)
        ft = ft * scale[:, None]
        fpair = fn[:, None] * nhat + ft
        # nhat points from i to j, so the repulsive force on i is -fpair
        np.add.at(force, ju, fpair)
        np.add.at(force, iu, -fpair)
        # dissipation is a power: damping force * relative velocity, and the
        # friction impulse * slip velocity; multiplied by dt to give energy
        dissip = float(dt * (np.sum(f_damp * approach)
                             + np.sum(np.abs(np.einsum("ij,ij->i", ft, vt)))))
        return force, dissip

    def _wall_forces(self, hull: Optional[WedgeBow], dt: float) -> Tuple[np.ndarray, np.ndarray]:
        act = self.active
        idx = np.where(act)[0]
        n = len(idx)
        force = np.zeros((n, 3))
        if hull is None or n == 0:
            return force, np.zeros(3)
        pos, vel = self.position[idx], self.velocity[idx]
        rad, mass = self.radius[idx], self.mass[idx]
        sdf, normal = hull.signed_distance(pos)
        # sdf > 0 inside the hull, so a sphere centred at distance |sdf| outside
        # penetrates when rad > |sdf|  ->  overlap = rad + sdf
        pen = rad + sdf
        hit = pen > 0.0
        if not np.any(hit):
            return force, np.zeros(3)
        h = np.where(hit)[0]
        pen_h = pen[h]
        nhat = normal[h]
        kn = self.contact.normal_force(rad[h], pen_h)
        cn = self.contact.damping_coefficient(mass[h], rad[h], pen_h, self.cfg.restitution)
        vn = np.einsum("ij,ij->i", vel[h], nhat)
        approach = np.maximum(-vn, 0.0)
        f_damp = np.minimum(cn * approach, kn)
        fn = kn + f_damp
        self.dissipated_contact += float(dt * np.sum(f_damp * approach))
        vt = vel[h] - vn[:, None] * nhat
        kt = self.contact.tangential_stiffness(rad[h], pen_h)
        ft = kt[:, None] * vt * dt
        ft_mag = np.linalg.norm(ft, axis=1)
        cap = self.cfg.friction * fn
        scale = np.where(ft_mag > cap, cap / np.maximum(ft_mag, 1e-30), 1.0)
        ft = ft * scale[:, None]
        fw = fn[:, None] * nhat + ft
        force[h] = fw
        self.contact_records.append(
            {
                "indices": idx[h],
                "positions": pos[h].copy(),
                "forces": fw.copy(),
                "normal": nhat.copy(),
                "magnitude": fn.copy(),
            }
        )
        return force, -fw.sum(axis=0)

    # -- integration --------------------------------------------------------
    def step(
        self,
        dt: float,
        fluid_velocity: Optional[np.ndarray] = None,
        water_level: Optional[float] = None,
        hull: Optional[WedgeBow] = None,
    ) -> dict:
        """Advance one DEM time step; returns diagnostics."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.contact_records = []
        env, drag_power, f_drag_total = self._environment_forces(fluid_velocity, water_level)
        pair, dissip_c = self._pair_forces(dt)
        wall, hull_reaction = self._wall_forces(hull, dt)
        acc = env + pair + wall
        idx = np.where(self.active)[0]
        m = self.mass[idx]
        self.velocity[idx] = self.velocity[idx] + acc / m[:, None] * dt
        if self.cfg.dim == 2:
            self.velocity[idx, 1] = 0.0
        self.position[idx] = self.position[idx] + self.velocity[idx] * dt
        # box walls: reflect
        lo, hi = self.bounds[0], self.bounds[1]
        for a in range(3):
            if self.cfg.dim == 2 and a == 1:
                continue
            rad = self.radius[idx]
            below = self.position[idx, a] - rad < lo[a]
            above = self.position[idx, a] + rad > hi[a]
            if np.any(below):
                self.position[idx[below], a] = lo[a] + rad[below]
                self.velocity[idx[below], a] = np.abs(self.velocity[idx[below], a]) * self.cfg.restitution
            if np.any(above):
                self.position[idx[above], a] = hi[a] - rad[above]
                self.velocity[idx[above], a] = -np.abs(self.velocity[idx[above], a]) * self.cfg.restitution
        # clearing: fragments that leave the aft boundary are removed
        idx = np.where(self.active)[0]
        out = self.position[idx, 0] > hi[0]
        if np.any(out):
            gone = idx[out]
            g_vec = np.asarray(self.cfg.gravity, dtype=float)
            vol = 4.0 / 3.0 * np.pi * self.radius[gone] ** 3
            ke = 0.5 * self.mass[gone] * np.sum(self.velocity[gone] ** 2, axis=1)
            pe = (self.mass[gone] - self.cfg.rho_fluid * vol) * abs(g_vec[2]) * self.position[gone, 2]
            self.removed_energy += float(np.sum(ke + pe))
            self.removed_mass += float(self.mass[gone].sum())
            self.active[gone] = False
        self.dissipated_contact += dissip_c
        # signed external work on the particle subsystem: fluid drag + hull
        idx_a = np.where(self.active)[0]
        n_a = len(idx_a)
        if n_a:
            self.drag_work += float(np.sum(f_drag_total * self.velocity[idx_a])) * dt
            self.wall_work += float(np.sum(wall * self.velocity[idx_a])) * dt
        self.dissipated_drag += float(drag_power.sum())
        self.last_hull_force = hull_reaction
        idx = np.where(self.active)[0]
        return {
            "hull_reaction": hull_reaction,
            "count": len(idx),
            "kinetic_energy": self.kinetic_energy(),
            "dissipated": dissip_c,
            "max_speed": float(np.linalg.norm(self.velocity[idx], axis=1).max()) if len(idx) else 0.0,
        }

    @property
    def stable_dt(self) -> float:
        """Conservative stable step from the stiffest likely contact."""
        act = self.active
        if act.sum() == 0:
            return 1e-2
        m = self.mass[act].min()
        r = self.radius[act].min()
        k = self.contact.normal_stiffness(r, 0.05 * r)
        if k <= 0:
            return 1e-2
        return 0.2 * 2.0 * np.pi * np.sqrt(m / k)


class IceSheetFracture:
    """Intact-ice front end: flexural capacity and fragment generation.

    The sheet is a plate on a Winkler foundation (water).  Its characteristic
    length is

    .. math:: L_b = \\left(\\frac{D}{k}\\right)^{1/4},\\quad
              D = \\frac{E h^3}{12(1-\\nu^2)},\\quad k = \\rho_w g

    and the first-crack load per unit width under a line load on a beam on an
    elastic foundation is ``q_c = 4 m_u / L_b`` with ``m_u = sigma_f h^2 / 6``.
    The *same* formula is used in the 2-D and the 3-D configuration; what
    differs is the lateral size of a fragment cell: in plane strain a cell spans
    the full channel width (lateral splitting is not representable), while in
    3-D the lateral crack spacing is bounded by the floe-size rule ``pi L_b``.
    """

    def __init__(
        self,
        thickness: float,
        flexural_strength: float,
        youngs_modulus: float,
        poisson: float,
        rho_fluid: float,
        gravity: float,
        width: float,
        cell_length: float,
        dim: int = 3,
        lateral_crack_spacing: Optional[float] = None,
    ):
        for name, value in (
            ("thickness", thickness),
            ("flexural_strength", flexural_strength),
            ("youngs_modulus", youngs_modulus),
            ("width", width),
            ("cell_length", cell_length),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        self.h = thickness
        self.sigma_f = flexural_strength
        self.E = youngs_modulus
        self.nu = poisson
        self.rho_fluid = rho_fluid
        self.g = gravity
        self.width = width
        self.cell_length = cell_length
        self.dim = dim
        self.foundation = rho_fluid * gravity
        self.D = youngs_modulus * thickness**3 / (12.0 * (1.0 - poisson**2))
        self.L_b = (self.D / self.foundation) ** 0.25
        self.m_u = flexural_strength * thickness**2 / 6.0
        self.q_c = 4.0 * self.m_u / self.L_b  # first-crack line load per unit width
        if dim == 2:
            self.cell_width = width
        else:
            # lateral crack spacing: the observed brash size is of order the ice
            # thickness, not the elastic length L_b (which is the *decay* length)
            target = lateral_crack_spacing if lateral_crack_spacing is not None else np.pi * self.L_b
            self.cell_width = float(min(width, target))
        self.n_lateral = max(1, int(round(width / self.cell_width)))
        self.cell_width = width / self.n_lateral
        self.break_force = self.q_c * self.cell_width
        self.fragment_volume = self.cell_length * self.cell_width * thickness

    def fragments_per_cell(self) -> int:
        """Number of DEM spheres a broken cell is replaced by."""
        target_volume = self.cell_length * self.cell_width * self.h
        if self.dim == 2:
            # plane strain: no lateral split is representable -> one slab
            return 1
        floe_area = np.pi * self.L_b**2
        per = target_volume / max(floe_area * self.h, 1e-30)
        return int(max(1, min(4, round(1.0 / max(per, 1e-12)) if per < 1 else 1)))

    def sphere_radius(self, n_fragments: int) -> float:
        volume = self.fragment_volume / max(n_fragments, 1)
        return float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))

    def equivalent_diameter(self) -> float:
        """Volume-equivalent diameter of one fragment (the C2 metric)."""
        return float((6.0 * self.fragment_volume / np.pi) ** (1.0 / 3.0))