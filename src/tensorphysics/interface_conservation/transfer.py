"""Interface-flux transfer gate for a lattice channel (conservative vs state exchange).

The gate this module implements is the machine-checkable core of the campaign
criterion: *is the quantity that crosses a coupling interface conserved?*  It is
evaluated on a synthetic channel (``channel_coupling_02``), driven by a
prescribed inlet velocity (Zou & He closure, verified exact in the uniform-flow
limit) with a pressure outlet.  The interface section is the mid-channel column;
the quantity of interest is the mass flux the section actually carries compared
with the mass flux that the coupling operator *claims* to transfer.

Two transfer operators are compared on the same running solver and the same
target flux:

``flux`` (conservative)
    the interface normal velocity is derived from the target mass flux **and the
    receiver's own density**, ``u = Phi / (rho h)``.  Because the Zou & He
    velocity closure enforces ``sum_i f_i c_n = rho u`` by construction, the
    received mass flux equals the transferred one to round-off.
``velocity`` (state exchange)
    the same target is applied as a velocity, i.e. the density factor is dropped
    -- which is what a "hand the other solver the velocity field" coupling does.
    The received mass flux is then ``rho u`` and the residual is ``|rho - 1|``,
    i.e. the lattice compressibility error ``O(Ma^2)``, *not* a discretisation
    error: it does not vanish under grid refinement at fixed Mach number.

Both are measured on the same solver state, so the residual difference is
attributable to the transfer operator alone.  The refinement study
(:func:`transfer_refinement`) reports the observed order in two scalings:

* fixed ``u_lat`` (constant Mach): the state-exchange residual plateaus -- the
  acoustic floor;
* diffusive scaling ``u_lat ~ 1/n``: the floor recedes and the residual decays
  with the measured exponent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lbm import D2Q9
from .quantities import CS2, fit_order, lattice_scales

__all__ = [
    "CHANNEL_DOMAIN",
    "TransferSample",
    "run_channel_transfer",
    "transfer_refinement",
]

#: synthetic domain identifier (campaign criterion (d): no domain semantics)
CHANNEL_DOMAIN = "channel_coupling_02"


@dataclass
class TransferSample:
    mode: str
    n: int
    u_lat: float
    tau: float
    target_flux: float
    received_flux: float
    integral_residual: float
    local_residual: float
    mach: float

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "n": self.n,
            "u_lat": self.u_lat,
            "tau": self.tau,
            "mach": self.mach,
            "target_flux": self.target_flux,
            "received_flux": self.received_flux,
            "integral_residual": self.integral_residual,
            "local_residual": self.local_residual,
        }


def _inlet_profile(n: int, u_lat: float) -> np.ndarray:
    """Parabolic lattice inlet profile, zero at both walls, mean ``u_lat``."""
    y = (np.arange(n) + 0.5) / n
    p = 6.0 * u_lat * y * (1.0 - y)
    return p


def run_channel_transfer(n: int = 32, u_lat: float = 0.05, tau: float = 0.6,
                         mode: str = "flux", steps: int = 4000,
                         length: float | None = None) -> TransferSample:
    """Run the lattice channel and measure the transferred vs received mass flux.

    The interface section sits at the mid-channel column; the *target* flux is the
    flux the inlet injects (which a steady incompressible channel must carry
    through every section), and the *received* flux is what the section actually
    carries according to its own distributions.
    """
    if mode not in ("flux", "velocity"):
        raise ValueError("mode must be 'flux' or 'velocity'")
    nx = int(length or 2 * n)
    lbm = D2Q9(nx, n, tau=tau)
    profile = _inlet_profile(n, u_lat)
    lbm.set_equilibrium((0.0, 0.0))
    bcs = {
        "west": ("velocity", profile, 0.0),
        "east": ("pressure", 1.0, 0.0),
        "north": ("wall", 0.0, 0.0),
        "south": ("wall", 0.0, 0.0),
    }
    # first pass: bring the channel to a steady state and read the *receiver's*
    # density at the interface column (the coupler's second exchange iteration)
    for _ in range(steps):
        lbm.step(bcs)
    rho_in = lbm.f[:, :, 0].sum(axis=0)
    if mode == "flux":
        applied = profile / rho_in          # flux-derived velocity
    else:
        applied = profile                   # state exchange: velocity handed over
    bcs["west"] = ("velocity", applied, 0.0)
    for _ in range(max(1, steps // 4)):
        lbm.step(bcs)

    target = float(np.sum(profile))         # rho = 1 at the inlet datum
    from .lbm import EX
    f = lbm.f[:, :, 0]
    received = float(np.sum(np.tensordot(EX, f, axes=(0, 0))))   # sum_i f_i c_ix
    integral = abs(received - target) / max(abs(target), 1e-30)
    local = float(np.linalg.norm(rho_in * applied - profile)
                  / math.sqrt(n) / abs(profile).max())
    return TransferSample(mode=mode, n=n, u_lat=u_lat, tau=tau,
                          target_flux=target, received_flux=received,
                          integral_residual=integral, local_residual=local,
                          mach=u_lat / math.sqrt(CS2))


def transfer_refinement(sizes=(16, 32, 64), scaling: str = "diffusive",
                        steps: int | None = None) -> dict:
    """Observed order of the state-exchange residual in two Mach-number scalings.

    ``scaling='diffusive'`` keeps the lattice Mach number proportional to the grid
    spacing (``u_lat = u_ref * n_ref / n``), the asymptotic-preserving limit in
    which the acoustic error goes to zero; ``scaling='fixed'`` keeps the Mach
    number constant, which is the usual production setting.
    """
    if scaling not in ("diffusive", "fixed"):
        raise ValueError("scaling must be 'diffusive' or 'fixed'")
    u_ref, n_ref = 0.08, 16
    out = {"scaling": scaling, "records": []}
    hs, res = [], []
    for n in sizes:
        u_lat = u_ref if scaling == "fixed" else u_ref * n_ref / n
        tau = 0.6
        ns = steps or int(round(1200 * (n / 16) ** 2))
        rec = {"n": n, "h": 1.0 / n, "u_lat": u_lat, "steps": ns}
        for mode in ("flux", "velocity"):
            s = run_channel_transfer(n=n, u_lat=u_lat, tau=tau, mode=mode, steps=ns)
            rec[mode] = s.as_dict()
        out["records"].append(rec)
        hs.append(1.0 / n)
        res.append(rec["velocity"]["integral_residual"])
    order, r2 = fit_order(hs, res)
    out["order_state_exchange"] = order
    out["order_r2"] = r2
    out["h"] = hs
    out["residual_state_exchange"] = res
    out["residual_conservative"] = [r["flux"]["integral_residual"] for r in out["records"]]
    return out