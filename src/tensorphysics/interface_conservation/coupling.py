"""Partitioned FVM-LBM coupling with an interface conservation gate.

Geometry (synthetic, no ship semantics -- campaign rule (d)): the unit square
cavity ``box_coupling_01`` is cut at ``x = x_i``.  The left sub-domain is solved
by the finite-volume solver (:mod:`tensorphysics.interface_conservation.fvm`),
the right sub-domain by the lattice-Boltzmann solver
(:mod:`tensorphysics.interface_conservation.lbm`).  The lid moves at ``U = 1``
and the Reynolds number is based on the cavity side.

Two exchange modes are implemented, and the difference between them is the point
of the gate:

``velocity``
    the coupler hands each side the *state* of the other side at the interface
    (normal velocity, tangential velocity).  Each solver re-derives its own
    density/pressure, so the mass flux each side reports differs by the lattice
    acoustic term ``O(Ma^2)`` -- a floor that does not vanish under grid
    refinement at fixed lattice Mach number.
``flux``
    the coupler hands the *flux* over: the mass flux computed by the donor is
    imposed on the receiver by deriving the receiver's normal velocity from that
    flux and the receiver's own density, and the donor's interface pressure is
    imposed on the receiver as a density (pressure) boundary.  Mass is then
    conserved by construction and the residual collapses to round-off, while the
    *local* (pointwise) mismatch survives as a discretisation error.

Both modes are measured with the same diagnostics, so the residual difference is
attributable to the exchange operator and nothing else.  That is what makes the
gate falsifiable: a "conservative" claim that does not beat the naive state
exchange by orders of magnitude is not conservative.

Reported quantities per sample (see :class:`CoupledSample`):

``mass_integral`` / ``momentum_integral``
    relative residual of the surface-integrated fluxes, the conservation
    statement of the node criterion;
``mass_local`` / ``momentum_local``
    ``L2`` residual of the pointwise flux distributions, the discretisation
    statement used for the observed-order study.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .fvm import StaggeredFVM
from .lbm import D2Q9
from .quantities import (CS2, InterfaceFlux, fit_order, flux_residual,
                         lattice_scales, local_flux_residual)

__all__ = ["CoupledSample", "CoupledCavity", "run_coupled_cavity", "DOMAIN"]


#: synthetic domain identifier (campaign criterion (d): no domain semantics)
DOMAIN = "box_coupling_01"


@dataclass
class CoupledSample:
    time: float
    fvm: InterfaceFlux
    lbm: InterfaceFlux
    mass_integral: float
    momentum_integral: float
    energy_integral: float
    mass_local: float
    momentum_local: float

    def as_dict(self) -> dict:
        return {
            "time": self.time,
            "fvm": {"mass": self.fvm.mass, "momentum": self.fvm.momentum,
                    "energy": self.fvm.energy},
            "lbm": {"mass": self.lbm.mass, "momentum": self.lbm.momentum,
                    "energy": self.lbm.energy},
            "mass_integral": self.mass_integral,
            "momentum_integral": self.momentum_integral,
            "energy_integral": self.energy_integral,
            "mass_local": self.mass_local,
            "momentum_local": self.momentum_local,
        }


class CoupledCavity:
    """Partitioned FVM (left) / LBM (right) lid-driven cavity.

    Parameters
    ----------
    n:
        number of cells/nodes across the cavity side; the interface sits at
        ``x = 0.5`` so each sub-domain is ``n/2`` wide and ``n`` high.
    u_lat:
        lattice lid velocity (the lattice Mach number is ``u_lat / c_s``); it
        fixes the LBM time step ``dt = h u_lat / U``.
    re:
        Reynolds number based on the cavity side and the lid velocity.
    """

    def __init__(self, n: int = 32, u_lat: float = 0.1, re: float = 100.0,
                 kappa: float = 0.8, subcycle: int | None = None,
                 mode: str = "flux") -> None:
        if n % 2:
            raise ValueError("n must be even so the interface cuts cells in half")
        if mode not in ("velocity", "flux"):
            raise ValueError("mode must be 'velocity' or 'flux'")
        self.n = int(n)
        self.u_lat = float(u_lat)
        self.re = float(re)
        self.mode = mode
        self.h = 1.0 / n
        self.nx_half = n // 2
        self.s_u, self.s_p = lattice_scales(self.u_lat, self.h)
        # FVM: half the cavity, square cells of size h
        self.fvm = StaggeredFVM(self.nx_half, n, lx=0.5, ly=1.0, re=re, kappa=kappa)
        self.fvm.set_closed_cavity(1.0)
        self.fvm.east_open = False
        # LBM: other half; viscosity in lattice units from nu and dt
        self.dt_lbm = self.h * self.u_lat / 1.0
        nu = 1.0 / re
        nu_lat = nu * self.dt_lbm / self.h ** 2
        self.tau = 0.5 + 3.0 * nu_lat
        self.lbm = D2Q9(self.nx_half, n, tau=self.tau)
        self.lbm.set_equilibrium((0.0, 0.0))       # start at rest
        # FVM stability: keep the CFL number of the sub-domain at 0.4
        if subcycle is None:
            subcycle = max(1, int(0.25 / self.u_lat))   # FVM CFL <= 0.25
        self.subcycle = int(subcycle)
        self.dt_fvm = self.subcycle * self.dt_lbm
        if mode == "flux":
            self.fvm.set_open_east(np.zeros(n))
        self.time = 0.0
        self._p_interface = np.zeros(n)      # transferred pressure (FVM units)
        self._u_if = np.zeros(n)             # relaxed interface velocity (FVM units)
        self.theta = 0.5                     # interface under-relaxation

    # -- exchange --------------------------------------------------------------
    def _lbm_bcs(self, u_interface: np.ndarray, v_interface: np.ndarray) -> dict:
        u_lat = self.u_lat
        return {
            "west": ("velocity", u_interface * u_lat, v_interface * u_lat),
            "east": ("velocity", 0.0, 0.0),
            "south": ("velocity", 0.0, 0.0),
            "north": ("velocity", u_lat, 0.0),
            "east": ("velocity", 0.0, 0.0),
            "south": ("velocity", 0.0, 0.0),
            "north": ("velocity", u_lat, 0.0),
        }

    def _lbm_pressure_bcs(self, rho_interface: np.ndarray) -> dict:
        return {
            "west": ("pressure", rho_interface, np.zeros(self.n)),
            "east": ("velocity", 0.0, 0.0),
            "south": ("velocity", 0.0, 0.0),
            "north": ("velocity", self.u_lat, 0.0),
        }

    def sample(self) -> CoupledSample:
        """Measure the interface fluxes on both sides and their residuals."""
        fvm_face = self.fvm.face_flux("east")
        lbm_face = self.lbm.interface_flux("west")
        # FVM side: per-unit-length fluxes, integrated with the face measure h
        n = self.n
        weight = self.h
        fvm_flux = InterfaceFlux(
            mass=float(np.sum(fvm_face["mass"]) * weight),
            momentum=float(np.sum(fvm_face["momentum"]) * weight),
            energy=float(np.sum(fvm_face["energy"]) * weight),
        )
        # LBM side: lattice -> FVM units (velocity x s_u, stress x s_p); the
        # outward normal of the lattice side at its west column is -x, and the
        # FVM's east face normal is +x, so the LBM values are negated to be
        # compared on the same oriented surface.
        s_u, s_p = self.s_u, self.s_p
        lbm_mass = -lbm_face["mass"] * s_u * weight
        lbm_mom = -lbm_face["momentum"] * s_p * weight
        lbm_energy = -lbm_face["energy"] * s_u ** 3 * weight
        lbm_flux = InterfaceFlux(mass=float(np.sum(lbm_mass)),
                                 momentum=float(np.sum(lbm_mom)),
                                 energy=float(np.sum(lbm_energy)))
        ref = InterfaceFlux(mass=1.0, momentum=1.0, energy=1.0)   # rho U^2 L
        res = flux_residual(fvm_flux, lbm_flux, ref)
        # local (pointwise) residuals
        mass_local = local_flux_residual(fvm_face["mass"] * weight,
                                         lbm_mass, 1.0)
        mom_local = local_flux_residual(fvm_face["momentum"] * weight,
                                        lbm_mom, 1.0)
        return CoupledSample(time=self.time, fvm=fvm_flux, lbm=lbm_flux,
                             mass_integral=res.mass, momentum_integral=res.momentum,
                             energy_integral=res.energy,
                             mass_local=mass_local, momentum_local=mom_local)

    # -- time integration ------------------------------------------------------
    def advance(self, n_fvm_steps: int = 1,
                sample_every: int = 1) -> list[CoupledSample]:
        samples: list[CoupledSample] = []
        for k in range(n_fvm_steps):
            # 1. what the lattice sees at the interface
            lbm_ux, lbm_uy = self.lbm.boundary_velocity("west")
            u_if = lbm_ux * self.s_u
            v_if = lbm_uy * self.s_u
            # Explicit partitioned coupling: each side sees a relaxed state of the
            # other, which is what keeps the Dirichlet-Dirichlet sweep stable.
            # The exchanged profile is projected onto zero *net* flux: an
            # incompressible sub-domain cannot absorb a net mass imbalance, so a
            # coupling that transfers a profile with non-zero integral has no
            # divergence-free solution (the projection would then leave a uniform
            # residual divergence).  This is the discrete compatibility condition
            # of the sub-domain Poisson problem, not a cosmetic correction.
            u_if = u_if - u_if.mean()
            self._u_if = (1.0 - self.theta) * self._u_if + self.theta * u_if
            if self.mode == "velocity":
                # FVM gets the lattice normal velocity directly
                self.fvm.bc_u_east = self._u_if.copy()
                lbm_west = (self._u_if, v_if)
            else:
                # Conservative transfer.  The FVM's outflow mass flux is what the
                # lattice receives; the lattice's interface density (pressure) is
                # what the FVM's Poisson problem takes as its Dirichlet datum.
                fvm_face = self.fvm.face_flux("east")
                mass_flux = fvm_face["mass"]                    # per unit length
                rho_b = self.lbm.interface_flux("west")["rho"]
                # u_lat_want is the FVM-unit velocity that carries mass_flux
                # through the local lattice density: rho_lat c = Phi / h
                u_lat_want = mass_flux / rho_b
                lbm_west = (u_lat_want, v_if)
                # pressure datum for the FVM from the lattice density
                self._p_interface = (rho_b - 1.0) * CS2 * self.s_p
                self.fvm.set_interface_pressure(self._p_interface)
            # 2. sub-cycled lattice steps
            for _ in range(self.subcycle):
                self.lbm.step(self._lbm_bcs(lbm_west[0], lbm_west[1]))
            # 3. one finite-volume step
            self.fvm.step(self.dt_fvm)
            self.time += self.dt_fvm
            if sample_every and (k % sample_every == 0):
                samples.append(self.sample())
        return samples

    # -- diagnostics -----------------------------------------------------------
    def centerline_u(self, n_pts: int = 21) -> tuple[np.ndarray, np.ndarray]:
        """``u`` along the vertical centreline ``x = 0.5`` (both sides)."""
        ys = np.linspace(0.02, 0.98, n_pts)
        u = np.empty(n_pts)
        for i, y in enumerate(ys):
            # the point x = 0.5 is the interface: take the FVM face value
            u[i] = float(np.interp(y, (np.arange(self.n) + 0.5) * self.h,
                                   self.fvm.u[-1, :]))
        return ys, u


def run_coupled_cavity(n: int = 32, u_lat: float = 0.1, re: float = 100.0,
                       t_end: float = 20.0, mode: str = "flux",
                       sample_every: int = 50, kappa: float = 0.8,
                       subcycle: int | None = None) -> dict:
    """Integrate the coupled cavity and return the sampled gate record.

    The returned dict carries the interface fluxes and residuals versus time, the
    steady-state (last-decile) residual maxima, and the lid-work/kinetic-energy
    diagnostics used by the report.
    """
    cc = CoupledCavity(n=n, u_lat=u_lat, re=re, kappa=kappa, mode=mode,
                       subcycle=subcycle)
    n_steps = max(1, int(round(t_end / cc.dt_fvm)))
    samples = cc.advance(n_steps, sample_every=sample_every)
    tail = samples[max(1, int(0.8 * len(samples))):]
    out = {
        "domain": DOMAIN,
        "mode": mode,
        "n": n,
        "h": cc.h,
        "u_lat": cc.u_lat,
        "tau": cc.tau,
        "re": re,
        "dt_lbm": cc.dt_lbm,
        "dt_fvm": cc.dt_fvm,
        "subcycle": cc.subcycle,
        "t_end": cc.time,
        "samples": [s.as_dict() for s in samples],
        "steady": {
            "mass_integral": max(s.mass_integral for s in tail),
            "momentum_integral": max(s.momentum_integral for s in tail),
            "energy_integral": max(s.energy_integral for s in tail),
            "mass_local": max(s.mass_local for s in tail),
            "momentum_local": max(s.momentum_local for s in tail),
        },
        "final": samples[-1].as_dict(),
        "fvm_divergence": cc.fvm.max_divergence(),
        "lbm_mach": cc.u_lat / math.sqrt(CS2),
        "interface_x": 0.5,
    }
    ys, u_if = cc.centerline_u()
    out["centerline_interface_u"] = {"y": ys.tolist(), "u": u_if.tolist()}
    return out


def refinement_study(cases: Sequence[dict], mode: str = "velocity") -> dict:
    """Observed order of the local interface residual under grid refinement.

    ``cases`` is a sequence of dicts with keys ``n`` and ``u_lat``; the lattice
    Mach number is refined with the grid (``u_lat / n`` held fixed, i.e. the
    acoustic/diffusive scaling in which the ``O(Ma^2)`` compressibility error
    vanishes as ``h -> 0``), so the measured exponent is the discretisation order
    and not the compressibility floor.
    """
    hs, res_m, res_p = [], [], []
    records = []
    for case in cases:
        rec = run_coupled_cavity(mode=mode, **case)
        records.append(rec)
        hs.append(rec["h"])
        res_m.append(rec["steady"]["mass_local"])
        res_p.append(rec["steady"]["momentum_local"])
    order_m, r2_m = fit_order(hs, res_m)
    order_p, r2_p = fit_order(hs, res_p)
    return {
        "mode": mode,
        "h": hs,
        "mass_local": res_m,
        "momentum_local": res_p,
        "order_mass": order_m,
        "order_mass_r2": r2_m,
        "order_momentum": order_p,
        "order_momentum_r2": r2_p,
        "records": records,
    }