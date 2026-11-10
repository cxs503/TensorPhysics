"""Bonded-particle (lattice discrete element) ice-sheet DEM -- vertical slice.

Physical model
--------------
A level-ice sheet is discretised on a structured lattice of particles (nodes
with mass and rotary inertia) connected by Euler-Bernoulli **beam bonds**
(axial + shear + bending).  This is the bonded-particle / lattice discrete
element model used for ice, and it is the mechanism-qualified face of
Lindqvist's model:

* **elastic response** -- exact 2-D frame element stiffness per unit
  out-of-plane width (``A = d``, ``I = d^3/12``), assembled on the fly
  (explicit dynamics, no global matrix);
* **bending fracture** -- a bond fails when its end surface stress
  ``|N|/A + 6|M|/d^2`` reaches the ice flexural strength ``sigma_f``; failed
  bonds are removed, so the breaking cusp and the sawtooth load spectrum are
  *emergent* (nothing prescribes the spectrum);
* **flotation** -- Winkler buoyancy ``rho_w g`` per unit area plus the ice
  weight ``rho_i g h`` and a near-critical dashpot: free-float equilibrium is
  ``w = -rho_i h / rho_w``;
* **hull contact** -- the bow plating is a rigid line inclined at the flare
  angle ``alpha`` (measured from the horizontal, the TensorPhysics convention)
  advancing at the ship speed ``v``.  The normal force is a penalty law
  **saturated by the ice crushing strength** (``F <= sigma_c d`` per particle,
  per unit width) -- the localised crushing mechanism of Lindqvist's model;
  Coulomb friction ``mu`` acts on the tangential slip; a particle whose
  accumulated crushing work exceeds the specific crushing energy ``e_crush``
  is *eroded* (crushed ice stops carrying load), which lets the crater advance
  into fresh ice.

The model is a 2-D vertical slice (unit out-of-plane width), so every force is
a **line force** [N/m].  This is the face in which bending, crushing and
submersion -- the three Lindqvist mechanisms -- all live.  A plane-view
(lattice *horizontal*) variant was tried first and rejected: it resolves the
contact-patch width but has no out-of-plane bending or submersion, and
under-predicts the resistance by ~2 orders of magnitude (recorded in
``docs/ICE_STRUCTURE_DEM_FEM_P380W2N06.md``).

Reference for the bending mechanism (used as an independent analytic check):
an edge-loaded semi-infinite floating beam on a Winkler foundation breaks at
``F_b = 4 M_p / L_c`` with ``M_p = sigma_f h^2 / 6`` and
``L_c = (4 D / (rho_w g))^{1/4}``, ``D = E h^3 / (12 (1 - nu^2))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "IceSliceConfig",
    "IceSliceResult",
    "simulate_ice_slice",
    "floating_beam_break_force",
]


@dataclass(frozen=True)
class IceSliceConfig:
    """Geometry, ice properties and integration controls of the DEM slice."""

    h: float = 1.0                  # ice thickness [m]
    speed: float = 3.0              # ship speed [m/s]
    flare_deg: float = 25.0         # bow flare from horizontal [deg]
    length_m: float = 6.0           # modelled ice-sheet length ahead of the edge
    rows: int = 5                   # particles through the thickness
    t_end: float = 0.35             # simulated time [s]
    # ice properties
    E: float = 9.0e9
    nu: float = 0.3
    rho_i: float = 900.0
    rho_w: float = 1025.0
    g: float = 9.81
    sigma_f: float = 0.5e6          # flexural strength [Pa]
    sigma_c: float = 5.0e6          # crushing strength [Pa]
    mu: float = 0.15
    e_crush: float = 2.0e5          # specific crushing energy [J/m^3]
    damping: float = 60.0           # water drag on the vertical motion [1/s]
    dt_safety: float = 0.25
    contact_stiffness_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.h <= 0:
            raise ValueError("ice thickness must be positive")
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        if not (5.0 <= self.flare_deg <= 89.0):
            raise ValueError("flare angle out of range")
        if self.rows < 2:
            raise ValueError("need at least two particle rows")
        if self.e_crush <= 0:
            raise ValueError("specific crushing energy must be positive")

    @property
    def d(self) -> float:
        """Particle pitch (element length) [m]."""
        return self.h / (self.rows - 1)

    @property
    def n_cols(self) -> int:
        return max(4, int(round(self.length_m / self.d)) + 1)

    @property
    def flexural_rigidity(self) -> float:
        return self.E * self.h ** 3 / (12.0 * (1.0 - self.nu ** 2))

    @property
    def characteristic_length(self) -> float:
        return (4.0 * self.flexural_rigidity / (self.rho_w * self.g)) ** 0.25


def floating_beam_break_force(cfg: IceSliceConfig) -> float:
    """Edge-load breaking force per unit width of a floating ice beam [N/m].

    Semi-infinite Euler-Bernoulli beam on a Winkler foundation
    ``k = rho_w g``, line load ``P`` at the free edge: ``M_max = P L_c / 4``,
    so the bending strength ``M_p = sigma_f h^2 / 6`` is reached at
    ``P_b = 4 M_p / L_c``.  Independent of the DEM -- used as its analytic
    verification anchor for the bending mechanism.
    """
    L_c = cfg.characteristic_length
    M_p = cfg.sigma_f * cfg.h ** 2 / 6.0
    return 4.0 * M_p / L_c


@dataclass
class IceSliceResult:
    time: np.ndarray
    force_x: np.ndarray
    force_normal: np.ndarray
    dt: float
    peaks: List[Dict[str, float]]
    energy: Dict[str, float]
    config: IceSliceConfig
    n_bonds: int
    n_broken_bonds: int
    n_eroded: int
    contact_nodes: np.ndarray
    penetration: np.ndarray

    # ---- summaries ---------------------------------------------------------
    @property
    def peak_force(self) -> float:
        return float(np.max(self.force_x)) if self.force_x.size else 0.0

    @property
    def mean_force(self) -> float:
        return float(np.mean(self.force_x)) if self.force_x.size else 0.0

    def steady_mean(self, start_fraction: float = 0.35) -> float:
        i0 = int(start_fraction * self.force_x.size)
        return float(np.mean(self.force_x[i0:])) if self.force_x.size else 0.0

    def mean_contact_pressure(self, start_fraction: float = 0.35) -> float:
        i0 = int(start_fraction * self.contact_nodes.size)
        cnt = float(np.mean(self.contact_nodes[i0:]))
        width = max(cnt * self.config.d, 1e-12)
        return self.steady_mean(start_fraction) / (width * math.sin(
            math.radians(self.config.flare_deg)))

    def as_dict(self) -> Dict[str, object]:
        return {
            "h_m": self.config.h,
            "speed_ms": self.config.speed,
            "flare_deg": self.config.flare_deg,
            "particle_pitch_m": self.config.d,
            "n_particles": int(self.n_cols * self.config.rows),
            "n_bonds": int(self.n_bonds),
            "n_broken_bonds": int(self.n_broken_bonds),
            "n_eroded_particles": int(self.n_eroded),
            "dt_s": self.dt,
            "n_time_steps": int(self.time.size),
            "peak_force_N_per_m": self.peak_force,
            "mean_force_N_per_m": self.mean_force,
            "steady_mean_force_N_per_m": self.steady_mean(),
            "mean_contact_pressure_Pa": self.mean_contact_pressure(),
            "n_events": len(self.peaks),
            "event_peaks_N_per_m": [p["peak"] for p in self.peaks],
            "analytic_floating_beam_break_force_N_per_m": floating_beam_break_force(
                self.config),
            "energy": self.energy,
        }


def _beam_stiffness(E: float, A: float, I: float, L: float) -> np.ndarray:
    """Local 2-D Euler-Bernoulli frame stiffness (DOF: u1,w1,th1,u2,w2,th2)."""
    ea, ei = E * A / L, E * I
    k = np.zeros((6, 6))
    k[0, 0] = k[3, 3] = ea
    k[0, 3] = k[3, 0] = -ea
    k[1, 1] = k[4, 4] = 12.0 * ei / L ** 3
    k[1, 4] = k[4, 1] = -12.0 * ei / L ** 3
    k[1, 2] = k[2, 1] = 6.0 * ei / L ** 2
    k[1, 5] = k[5, 1] = 6.0 * ei / L ** 2
    k[4, 2] = k[2, 4] = -6.0 * ei / L ** 2
    k[4, 5] = k[5, 4] = -6.0 * ei / L ** 2
    k[2, 2] = k[5, 5] = 4.0 * ei / L
    k[2, 5] = k[5, 2] = 2.0 * ei / L
    return k


def simulate_ice_slice(config: Optional[IceSliceConfig] = None) -> IceSliceResult:
    """Run the bonded-particle ice slice and return its ice-load spectrum."""
    cfg = config or IceSliceConfig()
    d = cfg.d
    ncol, nrow = cfg.n_cols, cfg.rows

    # ---- lattice -----------------------------------------------------------
    xs = np.arange(ncol) * d
    zs = -0.5 * cfg.h + np.arange(nrow) * d
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    coords = np.stack([X.ravel(), Z.ravel()], axis=1)
    nnode = coords.shape[0]
    idx = np.arange(nnode).reshape(ncol, nrow)

    conns = []
    for j in range(nrow):
        for i in range(ncol - 1):
            conns.append((idx[i, j], idx[i + 1, j]))
    for j in range(nrow - 1):
        for i in range(ncol):
            conns.append((idx[i, j], idx[i, j + 1]))
    conns = np.asarray(conns, dtype=int)
    nelem = conns.shape[0]

    p0, p1 = coords[conns[:, 0]], coords[conns[:, 1]]
    dxy = p1 - p0
    L = np.hypot(dxy[:, 0], dxy[:, 1])
    c, s = dxy[:, 0] / L, dxy[:, 1] / L

    A = d                    # area per unit out-of-plane width
    I = d ** 3 / 12.0
    k_loc = _beam_stiffness(cfg.E, A, I, d)

    T = np.zeros((nelem, 6, 6))
    for e in range(nelem):
        ce, se = c[e], s[e]
        R = np.array([[ce, se, 0.0], [-se, ce, 0.0], [0.0, 0.0, 1.0]])
        T[e, :3, :3] = R
        T[e, 3:, 3:] = R

    edofs = np.stack(
        [
            3 * conns[:, 0], 3 * conns[:, 0] + 1, 3 * conns[:, 0] + 2,
            3 * conns[:, 1], 3 * conns[:, 1] + 1, 3 * conns[:, 1] + 2,
        ],
        axis=1,
    )

    mass = cfg.rho_i * d * d                    # [kg per unit width]
    inertia = mass * d * d / 12.0
    mvec = np.empty((nnode, 3))
    mvec[:, 0] = mvec[:, 1] = mass
    mvec[:, 2] = inertia
    mvec = mvec.ravel()

    weight = -cfg.rho_i * cfg.g * cfg.h * d * d
    k_w = cfg.rho_w * cfg.g * d * d
    c_w = cfg.damping * mass
    f_static = np.zeros(3 * nnode)
    f_static[1::3] = weight

    # ---- explicit stability ------------------------------------------------
    k_ax = cfg.E * A / d
    k_bend = 48.0 * cfg.E * I / d ** 3
    k_rot = 4.0 * cfg.E * I / d
    k_contact = cfg.contact_stiffness_ratio * cfg.E
    w2 = max(
        (k_ax + k_bend) / mass,
        k_rot / inertia,
        k_contact / mass,
    )
    dt = cfg.dt_safety * 2.0 / math.sqrt(w2)
    nsteps = max(1, int(round(cfg.t_end / dt)))
    dt = cfg.t_end / nsteps

    alpha = math.radians(cfg.flare_deg)
    nrm = np.array([math.sin(alpha), -math.cos(alpha)])   # into the ice side
    tdir = np.array([math.cos(alpha), math.sin(alpha)])
    F_cap = cfg.sigma_c * d
    k_c = cfg.contact_stiffness_ratio * cfg.E
    c_c = 0.6 * math.sqrt(k_c * mass)
    x_front0 = 0.0

    # initial free surface: the sheet is already notched by the bow plating,
    # i.e. the bonds inside the plating line at t = 0 do not exist
    pen0 = -((coords[:, 0] - x_front0) * nrm[0]
             + (coords[:, 1] + 0.5 * cfg.h) * nrm[1])
    node_outside = pen0 <= 1e-12
    alive = np.all(node_outside[conns], axis=1)
    operational = node_outside.copy()   # material already removed by the bow
    crush_work = np.zeros(nnode)

    u = np.zeros(3 * nnode)
    v = np.zeros(3 * nnode)

    force_x = np.zeros(nsteps + 1)
    force_n = np.zeros(nsteps + 1)
    n_contact = np.zeros(nsteps + 1)
    penetration = np.zeros(nsteps + 1)

    w_contact = 0.0
    e_fracture = 0.0
    peaks: List[Dict[str, float]] = []
    in_event, ev_peak, ev_t = False, 0.0, 0.0

    def internal_forces(u_vec):
        ue = u_vec[edofs]
        u_loc = np.einsum("eij,ej->ei", T, ue)
        f_loc = np.einsum("ij,ej->ei", k_loc, u_loc)
        f_glob = np.einsum("eji,ej->ei", T, f_loc)
        f_e = np.where(alive[:, None], f_glob, 0.0)
        out = np.zeros(3 * nnode)
        np.add.at(out, edofs.ravel(), f_e.ravel())
        return out, np.where(alive[:, None], f_loc, 0.0)

    def contact(u_vec, v_vec, t_now):
        xf = x_front0 + cfg.speed * t_now
        # penetration of the advancing plating into the ice, measured along
        # the plating normal (-nrm); positive = the plate has swept past the node
        pen = -((coords[:, 0] + u_vec[0::3] - xf) * nrm[0]
                + (coords[:, 1] + u_vec[1::3] + 0.5 * cfg.h) * nrm[1])
        pen = np.where((pen > 0.0) & operational, pen, 0.0)
        if not np.any(pen > 0.0):
            return (np.zeros(3 * nnode), 0.0, 0.0, 0.0)
        fn = np.minimum(F_cap * np.tanh(k_c * pen / F_cap), F_cap)
        vx, vz = v_vec[0::3], v_vec[1::3]
        vn = vx * nrm[0] + vz * nrm[1]
        vt = vx * tdir[0] + vz * tdir[1]
        fn = fn + np.where(pen > 0.0, -c_c * np.minimum(vn, 0.0), 0.0)
        fn = np.where(pen > 0.0, fn, 0.0)
        ft = np.where(pen > 0.0, -cfg.mu * fn * np.tanh(vt / 1e-3), 0.0)
        f = np.zeros(3 * nnode)
        f[0::3] = fn * nrm[0] + ft * tdir[0]
        f[1::3] = fn * nrm[1] + ft * tdir[1]
        act = pen > 0.0
        return (f, float(np.sum(fn[act])), float(np.count_nonzero(act)),
                float(np.max(pen)))

    def accel(u_vec, v_vec, t_now):
        f_int, _ = internal_forces(u_vec)
        f = -f_int + f_static
        fc, fn_sum, ncon, pen_max = contact(u_vec, v_vec, t_now)
        f += fc
        # buoyancy (Winkler) + water drag on the vertical motion
        f[1::3] += -k_w * u_vec[1::3] - c_w * v_vec[1::3]
        a = f / mvec
        return a, fn_sum, ncon, pen_max, fc

    acc, fn_sum, ncon, pen_max, fc = accel(u, v, 0.0)
    force_x[0] = fn_sum * nrm[0]
    force_n[0] = fn_sum
    n_contact[0] = ncon
    penetration[0] = pen_max

    for step in range(nsteps):
        t_now = (step + 1) * dt
        v += 0.5 * dt * acc
        u += dt * v
        acc, fn_sum, ncon, pen_max, fc = accel(u, v, t_now)
        v += 0.5 * dt * acc

        # fracture of the beam bonds
        _, f_loc = internal_forces(u)
        N = np.maximum(np.abs(f_loc[:, 0]), np.abs(f_loc[:, 3]))
        M = np.maximum(np.abs(f_loc[:, 2]), np.abs(f_loc[:, 5]))
        stress = N / A + 6.0 * M / d ** 2
        newly = alive & (stress > cfg.sigma_f)
        if np.any(newly):
            e_fracture += float(np.sum(0.5 * stress[newly] ** 2 / cfg.E * A * d))
            alive = alive & ~newly

        # crushing work -> erosion of the crushed ice
        fn_mag = np.hypot(fc[0::3], fc[1::3])
        vmag = np.hypot(v[0::3], v[1::3])
        crush_work += fn_mag * np.minimum(vmag, cfg.speed) * dt
        w_crit = cfg.e_crush * d * d
        eroded = operational & (crush_work > w_crit)
        if np.any(eroded):
            operational = operational & ~eroded
        w_contact += float(np.sum(fn_mag * np.minimum(vmag, cfg.speed))) * dt

        fx = fn_sum * nrm[0]
        force_x[step + 1] = fx
        force_n[step + 1] = fn_sum
        n_contact[step + 1] = ncon
        penetration[step + 1] = pen_max

        if fx > 0 and not in_event:
            in_event, ev_peak, ev_t = True, fx, t_now
        elif in_event:
            if fx > ev_peak:
                ev_peak, ev_t = fx, t_now
            if fx < 0.6 * ev_peak:
                peaks.append({"t_s": ev_t, "peak": ev_peak})
                in_event = False

    ue = u[edofs]
    u_loc = np.einsum("eij,ej->ei", T, ue)
    f_loc = np.einsum("ij,ej->ei", k_loc, u_loc)
    elastic = 0.5 * float(np.sum(np.where(alive[:, None], u_loc * f_loc, 0.0)))
    kinetic = 0.5 * float(np.sum(mvec * v * v))

    return IceSliceResult(
        time=np.arange(nsteps + 1) * dt,
        force_x=force_x,
        force_normal=force_n,
        dt=dt,
        peaks=peaks,
        energy={
            "hull_contact_work_J_per_m": w_contact,
            "elastic_energy_J_per_m": elastic,
            "kinetic_energy_J_per_m": kinetic,
            "fracture_energy_J_per_m": e_fracture,
        },
        config=cfg,
        n_bonds=int(nelem),
        n_broken_bonds=int(np.count_nonzero(~alive)),
        n_eroded=int(np.count_nonzero(~operational)),
        contact_nodes=n_contact,
        penetration=penetration,
    )