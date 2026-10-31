"""Minimal 2-D soft-disc granular DEM shear cell (input for homogenisation).

Purpose
-------
Produce reproducible contact-force networks for brash-ice / rubble homogenisation
studies without dragging in a full DEM solver: a monodisperse-ish 2-D disc
assembly in a periodic square box sheared by the Lees-Edwards (shear-periodic)
boundary condition at constant volume.

Contact law
-----------
Linear spring-dashpot normal contact with restitution-derived damping
(``beta = 2 sqrt(m_eff k_n) * (-ln e) / sqrt(pi^2 + ln^2 e)``), and a *regularised
Coulomb* tangential contact

    f_t = -mu_p f_n tanh(|v_t| / v0) * v_t / |v_t|

which avoids the need for a persistent tangential-history state and remains
vectorisable. This is a standard regularisation (Cundall-Strack with a smooth
viscous cap); it reproduces the ``mu(I)`` growth of the effective friction with
inertial number but does not resolve stick-slip transients. Rigid-body rotation of
the contact frame is not advected in the tangential law (no tangential-history
correction), which is a documented approximation for this reduced model.

Units
-----
Dimensionless granular units: mean diameter ``d = 1``, 2-D grain density
``rho_s = 1 kg/m^2`` (stress is force per unit thickness, N/m), so the inertial
number ``I = gammadot d / sqrt(p / rho_s)`` is dimensionless as required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log, pi, sqrt
from typing import Any

import numpy as np


class DEMError(ValueError):
    """Invalid DEM configuration or state."""


@dataclass
class ShearCell:
    """2-D periodic soft-disc shear cell.

    Parameters
    ----------
    n_particles:
        Number of discs.
    area_fraction:
        Target solid area fraction used to size the periodic box.
    shear_rate:
        Lees-Edwards shear rate ``gammadot`` (1/time).
    kn:
        Normal contact stiffness per unit mass scale.
    mu_particle:
        Particle-scale Coulomb friction coefficient.
    restitution:
        Normal restitution coefficient used to derive the dashpot damping.
    r_mean, polydispersity:
        Mean radius and relative uniform polydispersity of the radii.
    dt:
        Time step; defaults to ``1/30`` of the contact time scale.
    """

    n_particles: int = 300
    area_fraction: float = 0.845
    shear_rate: float = 0.05
    kn: float = 1.0e3
    mu_particle: float = 0.4
    restitution: float = 0.5
    tangential_stiffness_ratio: float = 0.5
    rolling_friction: float = 0.2
    r_mean: float = 0.5
    polydispersity: float = 0.2
    seed: int = 0
    dt: float | None = None
    rho: float = 1.0
    v0: float = 1.0e-3
    state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.n_particles) < 20:
            raise DEMError(f"n_particles must be >= 20, got {self.n_particles!r}")
        self.n_particles = int(self.n_particles)
        if not (0.4 < float(self.area_fraction) < 0.95):
            raise DEMError(f"area_fraction out of range: {self.area_fraction!r}")
        for name in (
            "shear_rate",  # may be zero during preparation

            "kn",
            "mu_particle",
            "restitution",
            "r_mean",
            "rho",
            "v0",
            "tangential_stiffness_ratio",
            "rolling_friction",
        ):
            value = float(getattr(self, name))
            allow_zero = name == "shear_rate"
            if not isfinite(value) or (value < 0.0 if allow_zero else value <= 0.0):
                raise DEMError(f"{name} must be finite and positive, got {value!r}")
            setattr(self, name, value)
        if not (0.0 < self.polydispersity < 1.0):
            raise DEMError(f"polydispersity must be in (0, 1), got {self.polydispersity!r}")
        self._initialise()

    # ------------------------------------------------------------------ setup
    def _initialise(self) -> None:
        rng = np.random.default_rng(self.seed)
        n = self.n_particles
        radii = self.r_mean * rng.uniform(
            1.0 - self.polydispersity, 1.0 + self.polydispersity, size=n
        )
        masses = self.rho * pi * radii**2
        m_eff = 0.5 * masses.mean()
        self.L = float(sqrt(pi * np.sum(radii**2) / self.area_fraction))
        self.radii = radii
        self.masses = masses
        self.m_eff = float(m_eff)

        n_side = int(np.ceil(sqrt(n)))
        spacing = self.L / n_side
        grid = np.array([(i + 0.5, j + 0.5) for j in range(n_side) for i in range(n_side)])[:n]
        pos = grid * spacing + rng.normal(scale=0.08 * spacing, size=(n, 2))
        self.pos = np.mod(pos, self.L)
        self.vel = np.zeros((n, 2))

        ln_e = log(self.restitution)
        self.damping = 2.0 * sqrt(m_eff * self.kn) * (-ln_e) / sqrt(pi**2 + ln_e**2)
        t_contact = pi * sqrt(m_eff / self.kn)
        self.dt = float(self.dt) if self.dt is not None else t_contact / 30.0
        self.strain = 0.0
        self.time = 0.0
        self._pairs = np.triu_indices(n, k=1)
        n_pairs = self._pairs[0].size
        self._ft_state = np.zeros((n_pairs, 2))
        self._ft_seen_mask = np.zeros(n_pairs, dtype=bool)
        self._roll_state = np.zeros(n_pairs)
        self.kt = float(self.tangential_stiffness_ratio) * self.kn
        self.kr = self.kt * float(self.r_mean) ** 2
        self.omega = np.zeros(n)
        self.inertia = 0.5 * masses * radii**2
        self._history: list[dict[str, float]] = []

    # -------------------------------------------------------------- mechanics
    def pair_geometry(self) -> dict[str, np.ndarray]:
        """Upper-triangular contact geometry with minimum-image correction."""
        L = self.L
        iu, ju = self._pairs
        dx = self.pos[iu, 0] - self.pos[ju, 0]
        raw_dy = self.pos[iu, 1] - self.pos[ju, 1]
        n_y = np.round(raw_dy / L)
        dx = dx - np.round(dx / L) * L
        dy = raw_dy - n_y * L
        dist = np.hypot(dx, dy)
        dvx = self.vel[iu, 0] - self.vel[ju, 0] - self.shear_rate * L * n_y
        dvy = self.vel[iu, 1] - self.vel[ju, 1]
        return {
            "i": iu,
            "j": ju,
            "dx": dx,
            "dy": dy,
            "dist": dist,
            "dvx": dvx,
            "dvy": dvy,
        }

    def contact_forces(self) -> dict[str, np.ndarray]:
        """Compute contact forces (Cundall-Strack tangential history) and stress.

        The tangential force is a persistent spring (``kt``) that is clamped to the
        Coulomb cone ``|f_t| <= mu_p f_n``; when clamped the spring is truncated
        (sliding). Contacts that open reset their tangential state. The pair index
        is fixed (all upper-triangle pairs), so the tangential state lives in
        fixed-length arrays and the update stays vectorised.
        """
        geo = self.pair_geometry()
        i, j = geo["i"], geo["j"]
        dist = geo["dist"]
        r_sum = self.radii[i] + self.radii[j]
        overlap = r_sum - dist
        active = overlap > 0.0
        n_active = int(active.sum())

        # Reset the tangential/rolling state of every pair that is not in contact.
        self._ft_state[~active] = 0.0
        self._roll_state[~active] = 0.0

        result: dict[str, np.ndarray] = {
            "i": i[active],
            "j": j[active],
            "fx": np.zeros(n_active),
            "fy": np.zeros(n_active),
            "fn": np.zeros(n_active),
            "ft": np.zeros(n_active),
            "branch_x": geo["dx"][active],
            "branch_y": geo["dy"][active],
            "overlap": overlap[active],
            "rolling_moment": np.zeros(n_active),
        }
        if n_active == 0:
            self._ft_seen_mask[...] = active
            return result
        dx = geo["dx"][active]
        dy = geo["dy"][active]
        d = np.maximum(dist[active], 1e-300)
        nx = dx / d
        ny = dy / d
        dvx = geo["dvx"][active]
        dvy = geo["dvy"][active]
        vn = dvx * nx + dvy * ny
        fn = np.maximum(self.kn * result["overlap"] - self.damping * vn, 0.0)
        vtx = dvx - vn * nx
        vty = dvy - vn * ny

        # Tangential spring increment on the active subset, then Coulomb clamp.
        ft = self._ft_state[active] - self.kt * self.dt * np.column_stack([vtx, vty])
        mag = np.hypot(ft[:, 0], ft[:, 1])
        cap = self.mu_particle * fn
        over = mag > cap
        scale = np.where(over & (mag > 0.0), cap / np.maximum(mag, 1e-300), 1.0)
        ft *= scale[:, None]
        self._ft_state[active] = ft

        # Rolling resistance: a rotational spring clamped by the rolling Coulomb
        # cone. Without it, discs roll like bearings and the assembly carries
        # almost no shear stress (measured mu ~ 0.02 p); interlocking ice rubble
        # is represented here by mu_r = 0.2.
        d_omega = self.omega[i[active]] - self.omega[j[active]]
        moment = self._roll_state[active] - self.kr * self.dt * d_omega
        r_eff = 0.5 * r_sum[active]
        cap_m = self.rolling_friction * fn * r_eff
        moment = np.clip(moment, -cap_m, cap_m)
        self._roll_state[active] = moment
        result["rolling_moment"] = moment

        result["fx"] = fn * nx + ft[:, 0]
        result["fy"] = fn * ny + ft[:, 1]
        result["fn"] = fn
        result["ft"] = np.hypot(ft[:, 0], ft[:, 1])
        result["sliding_fraction"] = float(over.mean()) if n_active else 0.0
        return result

    def stress_tensor(self, contacts: dict[str, np.ndarray]) -> np.ndarray:
        """Volume-averaged (compressive-positive) stress ``sigma_ab = (1/V) sum f_a l_b``.

        Kinetic (streaming) contributions are neglected: the shear cell is dense
        and the granular temperature is small compared with the contact stress.
        """
        V = self.L**2
        sigma = np.zeros((2, 2))
        sigma[0, 0] = np.sum(contacts["fx"] * contacts["branch_x"])
        sigma[0, 1] = np.sum(contacts["fx"] * contacts["branch_y"])
        sigma[1, 0] = np.sum(contacts["fy"] * contacts["branch_x"])
        sigma[1, 1] = np.sum(contacts["fy"] * contacts["branch_y"])
        return sigma / V

    def step(self, *, measure: bool = False) -> dict[str, float] | None:
        """Advance one time step with semi-implicit Euler integration."""
        contacts = self.contact_forces()
        force = np.zeros_like(self.pos)
        np.add.at(force[:, 0], contacts["i"], contacts["fx"])
        np.add.at(force[:, 1], contacts["i"], contacts["fy"])
        np.add.at(force[:, 0], contacts["j"], -contacts["fx"])
        np.add.at(force[:, 1], contacts["j"], -contacts["fy"])
        torque = np.zeros_like(self.omega)
        np.add.at(torque, contacts["i"], -contacts["rolling_moment"])
        np.add.at(torque, contacts["j"], contacts["rolling_moment"])
        self.vel += force / self.masses[:, None] * self.dt
        self.omega += torque / self.inertia * self.dt
        self.pos = np.mod(self.pos + self.vel * self.dt, self.L)
        self.time += self.dt
        self.strain += self.shear_rate * self.dt
        if not measure:
            return None
        sigma = self.stress_tensor(contacts)
        # Love-Weber stress as defined (force-on-i (x) branch-to-i) is
        # compressive-positive: a hydrostatic compression gives positive diagonal.
        p = 0.5 * (sigma[0, 0] + sigma[1, 1])
        tau = 0.5 * (sigma[0, 1] + sigma[1, 0])
        record = {
            "time": self.time,
            "strain": self.strain,
            "p": float(p),
            "tau": float(tau),
            "mu": float(abs(tau) / p) if p > 0.0 else float("nan"),
            "mu_flux": float(abs(0.5 * (sigma[0, 1] + sigma[1, 0])) / p) if p > 0.0 else float("nan"),
            "mu_xy": float(abs(sigma[0, 1]) / p) if p > 0.0 else float("nan"),
            "z": float(2.0 * contacts["i"].size / self.n_particles),
            "sigma_xx": float(sigma[0, 0]),
            "sigma_xy": float(sigma[0, 1]),
            "sigma_yx": float(sigma[1, 0]),
            "sigma_yy": float(sigma[1, 1]),
        }
        self._history.append(record)
        return record

    def _rescale_box(self, factor: float) -> None:
        """Affinely dilate/compress the periodic cell (radii unchanged)."""
        self.L *= factor
        self.pos *= factor

    def servo_pressure(self, pressure_target: float, *, gain: float = 0.2, steps: int = 400,
                       servo_every: int = 20) -> float:
        """Iteratively dilate the cell until the measured pressure matches a target.

        Constant-volume shearing of a freshly jammed packing keeps a large
        preparation pressure (large overlaps), which masks the shear response.
        Servoing the cell volume down to a low pressure restores a near-critical
        packing whose shear response is representative of a sheared rubble layer.
        """
        target = float(pressure_target)
        if not isfinite(target) or target <= 0.0:
            raise DEMError(f"pressure_target must be finite and positive, got {pressure_target!r}")
        for _ in range(int(steps)):
            contacts = self.contact_forces()
            p = 0.5 * float(np.trace(self.stress_tensor(contacts)))
            if p <= 0.0:
                break
            ratio = p / target
            if abs(ratio - 1.0) < 0.02:
                break
            self._rescale_box(1.0 + float(np.clip(gain * (ratio - 1.0), -0.02, 0.02)))
            for _ in range(int(servo_every)):
                self.step()
        contacts = self.contact_forces()
        return 0.5 * float(np.trace(self.stress_tensor(contacts)))

    # ------------------------------------------------------------------ runs
    def run(
        self,
        n_steps: int,
        *,
        measure_every: int = 50,
        relax_steps: int = 0,
        pressure_target: float | None = None,
        servo_every: int = 20,
        servo_gain: float = 0.2,
    ) -> dict[str, Any]:
        """Run the cell and return steady-state averages.

        A static relaxation phase (``relax_steps``) at zero shear rate precedes the
        sheared phase, so that the pressure of the jammed packing is established
        before any rate dependence is measured.
        """
        if int(n_steps) <= 0:
            raise DEMError(f"n_steps must be positive, got {n_steps!r}")
        shear = self.shear_rate
        if relax_steps:
            self.shear_rate = 0.0
            self.vel *= 0.5
            for _ in range(int(relax_steps)):
                self.step()
            self.shear_rate = shear
            self.strain = 0.0
        self._history.clear()
        for k in range(int(n_steps)):
            # Constant-pressure servo: dilate/compact the cell to hold p at target.
            if pressure_target is not None and k % servo_every == 0:
                contacts = self.contact_forces()
                p_now = 0.5 * float(np.trace(self.stress_tensor(contacts)))
                if p_now > 0.0:
                    ratio = p_now / float(pressure_target)
                    if abs(ratio - 1.0) > 0.02:
                        self._rescale_box(
                            1.0 + float(np.clip(servo_gain * (ratio - 1.0), -0.01, 0.01))
                        )
            self.step(measure=(k % measure_every == 0))
        if not self._history:
            raise DEMError("no measurements collected; reduce measure_every")
        hist = {key: np.array([rec[key] for rec in self._history]) for key in self._history[0]}
        half = hist["strain"] > 0.5 * hist["strain"].max()
        window = slice(int(np.argmax(half)) if half.any() else 0, None)
        summary: dict[str, Any] = {
            "n_particles": self.n_particles,
            "final_box_length": self.L,
            "final_area_fraction": float(np.sum(pi * self.radii**2) / self.L**2),
            "box_length": self.L,
            "area_fraction": self.area_fraction,
            "shear_rate": self.shear_rate,
            "strain_total": float(hist["strain"].max()),
            "n_samples": int(hist["strain"].size),
        }
        for key in ("p", "tau", "mu", "mu_flux", "mu_xy", "z"):
            values = hist[key][window]
            values = values[np.isfinite(values)]
            summary[key] = float(values.mean()) if values.size else float("nan")
            summary[f"{key}_std"] = float(values.std()) if values.size else float("nan")
        summary["history"] = hist
        return summary

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> dict[str, np.ndarray]:
        """Return positions, radii, contact forces and branch vectors."""
        contacts = self.contact_forces()
        return {
            "positions": self.pos.copy(),
            "radii": self.radii.copy(),
            "masses": self.masses.copy(),
            "box_length": np.array([self.L]),
            "contacts": {
                "i": contacts["i"],
                "j": contacts["j"],
                "force": np.column_stack([contacts["fx"], contacts["fy"]]),
                "branch": np.column_stack([contacts["branch_x"], contacts["branch_y"]]),
            },
        }