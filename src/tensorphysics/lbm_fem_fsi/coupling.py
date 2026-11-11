"""Partitioned two-way LBM -> FEM coupling with sub-cycling and conservation ledgers.

Campaign node ``p362w2-n09``.  The fluid is advanced *explicitly* on the lattice
time step (``dt_l = 1``); the structure is advanced *implicitly* (Newmark-beta,
unconditionally stable) on the coarse step ``dt_s = n_sub * dt_l``.  The two are
synchronised by sub-cycling: ``n_sub`` lattice steps are accumulated, the
exchange force is condensed on the coarse step, the FE model is advanced, and the
new wall velocity is imposed on the immersed boundary for the next window.

Two exchange policies are provided, and the difference between them is the
subject of the conservation study:

``impulse``  the coarse-step load is the *force accumulated over the window*,
             ``F_s = (1/n) sum_k F_l(k)``.  The impulse handed to the structure,
             ``F_s dt_s``, is then identical to the impulse the lattice actually
             delivered, whatever the window size -- the scheme is
             impulse-conserving by construction (this is the production scheme).
``lagged``   the coarse-step load is the *previous* window's force (explicit
             lagged coupling, the default of most loose-coupling frameworks).
             Its momentum and energy defects are genuinely first order in
             ``dt_s`` and must shrink when the step is refined, which is what
             makes the mesh-independence criterion of this node a real test and
             not a tautology.

Kinematics: the cylinder is rigid and its streamwise position is locked (the
standard 1-DOF transverse VIV arrangement); its transverse position and velocity
are those of the FE attachment patch (mean of the free-end node rows), i.e. the
rigid-link transfer.  The patch nodes receive the force distribution of
:mod:`tensorphysics.lbm_fem_fsi.interface`, which reproduces the fluid resultant
force and moment exactly.

Ledgers (lattice units, ``rho = 1``):

``mass``      interface mass exchange |sum_links (f_opp - f_i)| (exactly zero for
              half-way bounce-back) plus the rasterisation refill/absorption;
``momentum``  ``J_lattice = sum_t F_mea(t)`` against ``J_structure = sum_s F_s dt_s``,
              with the lattice's own double measurement (Ladd exchange vs global
              budget residual) as a third, independent number;
``energy``    ``W_fluid->body = sum_t F(t).v_wall(t)`` against
              ``dKE + dStrain + E_damping`` of the FE model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .interface import audit_transfer, map_to_nodes, surface_load, traction_components
from .lattice import LatticeFSI, Obstacle
from .structure import PlaneStressTether

__all__ = ["FSIConfig", "FSIResult", "run_case", "strouhal_from_history",
           "multirate_study", "build_models"]


@dataclass
class FSIConfig:
    """Benchmark definition (lattice units unless stated)."""

    diameter: int = 16                     # D in lattice nodes
    Re: float = 100.0
    U: float = 0.1                         # inlet velocity
    domain_len: float = 20.0               # in D, streamwise
    domain_ht: float = 16.0                # in D, transverse (blockage = 1/16)
    x_center: float = 5.0                  # in D from the inlet
    mass_ratio: float = 10.0               # m* = m_cyl / (rho D^2)
    damping_ratio: float = 0.01
    reduced_velocity: float = 5.0          # U* = U / (f_n D)
    n_sub: int = 4                         # lattice steps per structural step
    scheme: str = "impulse"                # "impulse" | "lagged"
    n_periods: float = 30.0                # shedding periods after the ramp
    ramp_periods: float = 10.0
    free_motion: bool = True               # False -> fixed cylinder (St validation)
    perturbation: float = 0.02             # initial transverse kick / U
    wall_mode: str = "halfway"             # "halfway" | "interp"
    fem_substeps: int = 8                 # implicit sub-steps inside one window
    transmit_moment: bool = False        # 1-DOF idealisation: resultant force only
    tether_ns: int = 16
    tether_nw: int = 3

    # ---- derived geometry -------------------------------------------------
    @property
    def nx(self) -> int:
        return int(round(self.domain_len * self.diameter))

    @property
    def ny(self) -> int:
        return int(round(self.domain_ht * self.diameter))

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def xc(self) -> float:
        return self.x_center * self.diameter

    @property
    def yc(self) -> float:
        return 0.5 * self.ny

    @property
    def tau(self) -> float:
        return 3.0 * self.U * self.diameter / self.Re + 0.5

    @property
    def shedding_period(self) -> float:
        """Planning estimate of the shedding period, ``D / (St U)`` with St = 0.165."""
        return self.diameter / (0.165 * self.U)

    def with_(self, **kwargs) -> "FSIConfig":
        return replace(self, **kwargs)


@dataclass
class FSIResult:
    config: FSIConfig
    time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v: np.ndarray = field(default_factory=lambda: np.zeros(0))
    force_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    force_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    f_natural: float = 0.0
    strouhal: float = 0.0
    amplitude_ratio: float = 0.0
    cd_mean: float = 0.0
    cl_rms: float = 0.0
    mass_error: float = 0.0
    momentum_error: float = 0.0
    energy_error: float = 0.0
    energy_error_net: float = 0.0
    energy_exchange_error: float = 0.0
    momentum_error_mea_budget: float = 0.0
    mass_closure: float = 0.0
    transfer_audit: dict = field(default_factory=dict)
    ledgers: dict = field(default_factory=dict)
    wall_time: float = 0.0
    extra: dict = field(default_factory=dict)


def build_models(cfg: FSIConfig):
    """Build the lattice and the FE tether, tuned to the requested ``U*``."""
    obs = Obstacle(cfg.xc, cfg.yc, cfg.radius)
    lat = LatticeFSI(cfg.nx, cfg.ny, cfg.tau, cfg.U, obs, wall_mode=cfg.wall_mode)
    lat.perturb(cfg.perturbation)
    lat.mass_initial = lat.mass_total()
    m_cyl = cfg.mass_ratio * cfg.diameter ** 2          # rho = 1
    L = 4.0 * cfg.diameter
    H = 0.5 * cfg.diameter
    tether = PlaneStressTether(length=L, height=H, n_s=cfg.tether_ns, n_w=cfg.tether_nw,
                               E=1.0, nu=0.3, rho=0.05, thickness=1.0)
    tether.add_tip_mass(m_cyl)
    f_target = cfg.U / (cfg.reduced_velocity * cfg.diameter)
    f1 = tether.first_frequency()
    tether.E *= (f_target / f1) ** 2
    tether._assemble()
    tether.add_tip_mass(m_cyl)                          # _assemble rebuilds M
    f1 = tether.set_rayleigh(cfg.damping_ratio)
    return lat, tether, f1


def _attachment(tether: PlaneStressTether):
    """Two rows of nodes at the free end: the rigid attachment patch."""
    s = tether.coords[:, 0]
    h = tether.L / tether.n_s
    sel = np.where(s >= tether.L - 1.5 * h)[0]
    return sel


def strouhal_from_history(t: np.ndarray, cl: np.ndarray, D: float, U: float,
                          band=(0.05, 0.6), smooth: int = 1) -> dict:
    """Strouhal number from a *band-limited* FFT peak plus a zero-crossing check.

    ``band`` is expressed in Strouhal units and is essential: the per-step Ladd
    force carries lattice-scale jitter whose spectral peak sits far above the
    shedding band and would otherwise win the argmax (an 0.9 "Strouhal number"
    is always a measurement bug, never physics).  ``smooth`` applies a moving
    average of that many samples before the transform.
    """
    t = np.asarray(t, dtype=float)
    cl = np.asarray(cl, dtype=float)
    if t.size < 32:
        return {"st": float("nan"), "f_fft": float("nan"), "f_zero": float("nan"),
                "st_zero": float("nan")}
    dt = float(np.median(np.diff(t)))
    sig = cl - cl.mean()
    if smooth > 1 and sig.size > smooth:
        k = np.ones(smooth) / smooth
        sig = np.convolve(sig, k, mode="same")
    win = np.hanning(sig.size)
    spec = np.abs(np.fft.rfft(sig * win))
    freqs = np.fft.rfftfreq(sig.size, dt)
    st_axis = freqs * D / U
    keep = (st_axis >= band[0]) & (st_axis <= band[1])
    if not keep.any():                       # pragma: no cover
        return {"st": float("nan"), "f_fft": float("nan"), "f_zero": float("nan"),
                "st_zero": float("nan")}
    spec = np.where(keep, spec, 0.0)
    f_fft = float(freqs[int(np.argmax(spec))])
    sgn = np.sign(sig)
    idx = np.where(np.diff(sgn) > 0)[0]
    f_zero = float(1.0 / (np.mean(np.diff(idx)) * dt)) if idx.size > 2 else float("nan")
    return {"st": f_fft * D / U, "f_fft": f_fft, "f_zero": f_zero,
            "st_zero": f_zero * D / U}


def run_case(cfg: FSIConfig, lat=None, tether=None, progress: bool = False) -> FSIResult:
    """Run one coupled case (fixed or elastically mounted cylinder)."""
    import time as _time

    t0 = _time.time()
    if lat is None or tether is None:
        lat, tether, f1 = build_models(cfg)
    else:
        f1 = cfg.U / (cfg.reduced_velocity * cfg.diameter)
    attach = _attachment(tether)
    tether.set_attachment(attach)
    attach_pos = tether.coords[attach]              # (n,2) in the beam frame (s, w)
    attach_dofs = 2 * attach + 1                    # transverse DOFs
    ndof = 2 * tether.n_nodes
    centre_beam = np.array([tether.coords[attach, 0].mean(), float(tether.tip_axis)])

    T = cfg.shedding_period
    n_total = int(round((cfg.ramp_periods + cfg.n_periods) * T))
    n_ramp = int(round(cfg.ramp_periods * T))

    y0 = cfg.yc
    u_tip = 0.0
    v_tip = 0.0
    lat.place(y0, 0.0, rebuild=True)

    hist = {k: [] for k in ("t", "y", "v", "fy", "fx")}
    # per-lattice-step force history: the FFT of the lift needs lattice
    # resolution, not the (much coarser) structural-window resolution
    fine_t, fine_fy, fine_fx = [], [], []
    impulse_lat = np.zeros(2)
    impulse_struct = np.zeros(2)
    audited = None
    force_prev = np.zeros(2)
    mass_ref = max(lat.mass_total(), 1e-30)
    mea_total = np.zeros(2)
    budget_total = np.zeros(2)
    n_windows = 0
    work_exchange = 0.0
    abs_work = 0.0
    v_prev = 0.0
    u_prev = 0.0
    boot = None                       # ledger snapshot at the end of the ramp

    dt_s = float(cfg.n_sub)
    t = 0.0
    while t < n_total:
        lat.place(y0 + u_tip, v_tip, rebuild=True)
        F_sum = np.zeros(2)
        samp_pos, samp_frc = [], []
        for _sub in range(cfg.n_sub):
            led = lat.step()
            fine_t.append(lat.steps)
            fine_fy.append(float(led.force_mea[1]))
            fine_fx.append(float(led.force_mea[0]))
            F_sum += led.force_mea
            impulse_lat += led.force_mea
            mea_total += led.force_mea
            budget_total += led.force_budget
            if lat.last_samples is not None:
                p, fr = lat.last_samples
                samp_pos.append(p)
                samp_frc.append(fr)
        n_windows += 1
        t += dt_s
        F_avg = F_sum / cfg.n_sub
        # ---- conservative load transfer onto the attachment patch
        if samp_pos:
            # the geometry is frozen inside a window, so every sub-step produces
            # the same link ordering: average the samples (not concatenate!)
            pos = samp_pos[0]
            frc = np.mean(np.stack(samp_frc, axis=0), axis=0)
            sl = surface_load(pos, frc, np.array([cfg.xc, y0 + u_tip]))
            if cfg.transmit_moment:
                nodal = map_to_nodes(sl, attach_pos)
            else:
                # the 1-DOF VIV idealisation has no rotational degree of freedom:
                # the rigid link transmits the resultant force only, so the load
                # is spread uniformly (moments are still measured and audited)
                nodal = np.full(attach.size, sl.force[1] / attach.size)
            audited = audit_transfer(sl, attach_pos, nodal)
            audited["components"] = traction_components(sl)
        else:                                       # pragma: no cover
            nodal = np.zeros(attach.size)
        load_vec = np.zeros(ndof)
        if cfg.free_motion:
            load_vec[attach_dofs] = nodal
        if cfg.scheme == "lagged":
            applied = np.zeros(ndof)
            applied[attach_dofs] = force_prev[1] / max(attach.size, 1)
            force_prev = F_avg
        else:
            applied = load_vec
        if cfg.free_motion:
            nsub = max(int(cfg.fem_substeps), 1)
            for _ in range(nsub):
                tether.advance(applied, dt_s / nsub)
            u_tip = tether.tip_displacement()
            v_tip = tether.tip_velocity()
            power = F_avg[1] * (u_tip - u_prev)
            work_exchange += power
            abs_work += abs(power)
            u_prev = u_tip
        impulse_struct += np.array([0.0, applied[attach_dofs].sum()]) * dt_s
        if boot is None and t >= n_ramp:
            # the start-up shock (impulsive start, first vortex formation) is not
            # a steady state: the conservation ledgers are evaluated over the
            # physical window after the ramp, and the start-up numbers are kept
            boot = {"work": work_exchange, "abs": abs_work,
                    "external": tether.external_work,
                    "mech": tether.kinetic() + tether.strain(),
                    "damping": tether.damping_dissipation}
        hist["t"].append(t)
        hist["y"].append(u_tip)
        hist["v"].append(v_tip)
        hist["fy"].append(F_avg[1])
        hist["fx"].append(F_avg[0])
        if progress and n_windows % 200 == 0:
            print(f"    t={t:8.0f}/{n_total} y={u_tip:+.3f} vy={v_tip:+.5f} "
                  f"Fy={F_avg[1]:+.5f}", flush=True)

    tarr = np.array(hist["t"])
    yarr = np.array(hist["y"])
    fyarr = np.array(hist["fy"])
    fxarr = np.array(hist["fx"])
    sel = tarr >= n_ramp
    q = 0.5 * cfg.U ** 2 * cfg.diameter
    # response frequency: for the mounted case the *motion* is the response;
    # the lift carries the added-mass component at the same frequency but with
    # a large inertial offset, so both are reported.
    ft = np.array(fine_t, dtype=float)
    ffy = np.array(fine_fy)
    fsel = ft >= n_ramp
    st_lift = strouhal_from_history(ft[fsel], ffy[fsel], cfg.diameter, cfg.U,
                                    smooth=25 if not cfg.free_motion else 5)
    if cfg.free_motion:
        st = strouhal_from_history(tarr[sel], yarr[sel], cfg.diameter, cfg.U)
    else:
        st = st_lift
    amp = 0.5 * (yarr[sel].max() - yarr[sel].min()) / cfg.diameter if cfg.free_motion else 0.0
    res = FSIResult(config=cfg, time=tarr, y=yarr, v=np.array(hist["v"]),
                    force_y=fyarr, force_x=fxarr, f_natural=f1)
    res.strouhal = st["st"]
    res.amplitude_ratio = float(amp)
    res.cd_mean = float(np.mean(fxarr[sel] / q))
    res.cl_rms = float(np.sqrt(np.mean((fyarr[sel] / q) ** 2)))
    # ---------------- conservation ledgers
    mb = lat.mass_balance()
    # the *defect* is the net mass the interface creates: released - absorbed
    res.mass_error = float((abs(mb["body_exchange"]) + abs(mb["released"] - mb["absorbed"]))
                           / mass_ref)
    res.mass_closure = float(abs(mb["closure_error"]) / mass_ref)
    J_lat = float(impulse_lat[1])
    J_fem = float(impulse_struct[1])
    E_mech = tether.kinetic() + tether.strain()
    res.momentum_error = float(abs(J_fem - J_lat) / max(abs(J_lat), 1e-30))
    if boot is None:                      # pragma: no cover - very short run
        boot = {"work": 0.0, "abs": 0.0, "external": 0.0, "mech": 0.0, "damping": 0.0}
    dW = work_exchange - boot["work"]
    dabs = abs_work - boot["abs"]
    dmech = E_mech - boot["mech"]
    ddamp = tether.damping_dissipation - boot["damping"]
    energy_residual = abs(dW - (dmech + ddamp))
    # two normalisations: relative to the *net* work (ill conditioned for a
    # resonant oscillator, the net is a small difference of large numbers) and
    # relative to the exchanged energy scale int|P|dt (the standard measure)
    W_fem_window = tether.external_work - boot["external"]
    res.energy_exchange_error = float(abs(W_fem_window - dW) / max(abs(dW), 1e-30))
    res.energy_error = float(energy_residual / max(dabs, 1e-30))
    res.energy_error_net = float(energy_residual / max(abs(dW), 1e-30))
    res.momentum_error_mea_budget = float(
        np.linalg.norm(mea_total - budget_total) / max(np.linalg.norm(mea_total), 1e-30))
    res.transfer_audit = audited or {}
    res.ledgers = {
        "mass_terms": mb,
        "impulse_lattice": J_lat,
        "impulse_structure": J_fem,
        "work_lattice": float(lat.work_on_body),
        "work_fem": float(tether.external_work),
        "kinetic": float(tether.kinetic()),
        "strain": float(tether.strain()),
        "damping": float(tether.damping_dissipation),
        "n_windows": n_windows,
        "st_zero": st["st_zero"],
        "st_lift": st_lift["st"], "st_lift_zero": st_lift["st_zero"],
        "fine_history": {"t": ft[::4].tolist(), "fy": ffy[::4].tolist(),
                         "fx": np.array(fine_fx)[::4].tolist()},
        "work_exchange": float(work_exchange),
        "abs_work": float(abs_work),
        "window_work": float(dW),
        "window_work_fem": float(tether.external_work - boot["external"]),
        "energy_exchange_error": float(res.energy_exchange_error), "window_abs_work": float(dabs),
        "window_mech_change": float(dmech), "window_damping": float(ddamp),
        "energy_residual": float(energy_residual),
        "work_lattice_power": float(lat.work_on_body),
        "f_fft": st["f_fft"],
    }
    res.extra = {"scheme": cfg.scheme, "n_sub": cfg.n_sub, "tau": cfg.tau,
                 "nx": cfg.nx, "ny": cfg.ny, "blockage": 1.0 / cfg.domain_ht,
                 "n_steps": n_total, "D_eff": lat.effective_diameter(), "n_ramp": n_ramp,
                 "wall_mode": cfg.wall_mode}
    res.wall_time = _time.time() - t0
    return res


def multirate_study(cfg: FSIConfig, subs=(1, 2, 4, 8, 16), periods: float = 6.0,
                    scheme: str = "lagged") -> dict:
    """Conservation defects vs the LBM/FEM step ratio ``n_sub`` (time-sync study)."""
    out = []
    for n in subs:
        c = cfg.with_(n_sub=int(n), scheme=scheme, n_periods=periods, ramp_periods=4.0)
        r = run_case(c)
        out.append({
            "n_sub": int(n),
            "dt_struct": float(n),
            "momentum_error": r.momentum_error,
            "energy_error": r.energy_error,
            "mass_error": r.mass_error,
            "strouhal": r.strouhal,
            "amplitude_ratio": r.amplitude_ratio,
        })
    return {"scheme": scheme, "rows": out}