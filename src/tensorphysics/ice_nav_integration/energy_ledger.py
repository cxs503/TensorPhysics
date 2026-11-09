"""Five-channel energy ledger for the ice-navigation loop.

The ice-resistance decomposition used by the whole loop is the classical
five-channel split (Lindqvist 1989 crushing / breaking / submersion, plus the
brash-ice clearing term and the open-water viscous term):

===================  ====================================================
channel              origin
===================  ====================================================
``crushing``         quasi-static crushing of the ice edge at the bow
                     shoulder (Lindqvist ``R_crush``, coverage-weighted)
``breaking``         bending failure of the ice cusp (Lindqvist ``R_break``)
``submersion``       submergence / lift of the broken ice (Lindqvist
                     ``R_sub``)
``clearing``         momentum needed to accelerate the isolated floes
                     (brash branch ``4 c (1-c) R_floe_impact``)
``viscous_open_water`` ITTC-57 friction over the full wetted surface with
                     the form factor
===================  ====================================================

The ledger's job is *not* to re-derive these laws.  Its job is to check energy
bookkeeping across the module boundary: the thrust power``P_in = R_panel V``
obtained from the panel-integrated flow reaction must equal the sum of the five
channel powers within the accepted 2 %.  The remainder is *attributed*, never
hidden:

* ``momentum_flux`` -- the control volume changes the water momentum flux
  (``mdot (u_L - u_0)``); a genuine, small term of the lane formulation,
* ``quadrature``    -- panel integration / distribution-weight discretisation,
* ``fsi_coupling``  -- the fixed-point residual of the structural sub-iteration.

Units: SI (N, W).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: C3 acceptance bound on the energy residual.
TOL_ENERGY = 2.0e-2

CHANNELS = (
    "crushing",
    "breaking",
    "submersion",
    "clearing",
    "viscous_open_water",
)


def brash_channel_split(brash_result: Mapping[str, Any]) -> Dict[str, float]:
    """Split a brash-ice resistance result into the four ice channels.

    ``brash_ice_resistance`` returns ``R = D(c) R_level + 4c(1-c) R_impact``
    with ``R_level = R_crush + R_break + R_sub``; the coverage decay factor ``D``
    is applied to all three level components, which is exactly how the semi-
    empirical law defines the partial-cover branch.  The identity
    ``sum(split) == result["R_total_N"]`` is asserted by the caller's tests.
    """
    d = float(brash_result["decay_factor"])
    split = {
        "crushing": d * float(brash_result["R_crush_N"]),
        "breaking": d * float(brash_result["R_break_N"]),
        "submersion": d * float(brash_result["R_submersion_N"]),
        "clearing": float(brash_result["R_total_N"])
        - d * float(brash_result["R_level_N"]),
    }
    return split


@dataclass
class LedgerLine:
    channel: str
    force_N: float
    power_W: float
    share: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "force_N": float(self.force_N),
            "power_W": float(self.power_W),
            "share": float(self.share),
        }


def build_ledger(
    *,
    channel_forces: Mapping[str, float],
    speed_m_s: float,
    panel_force_N: float,
    momentum_flux_N: float = 0.0,
    fsi_residual: float = 0.0,
    tol: float = TOL_ENERGY,
) -> Dict[str, Any]:
    """Assemble the ledger and return the residual with its attribution."""
    missing = [c for c in CHANNELS if c not in channel_forces]
    if missing:
        raise ValueError("missing resistance channels: %s" % ", ".join(missing))

    V = float(speed_m_s)
    lines = []
    total_force = 0.0
    for name in CHANNELS:
        f = float(channel_forces[name])
        if f < 0.0:
            raise ValueError("channel %r has negative resistance %r" % (name, f))
        total_force += f
        lines.append(LedgerLine(name, f, f * V, 0.0))

    power_in = float(panel_force_N) * V
    power_sum = total_force * V
    denom = max(abs(power_in), abs(power_sum), 1e-30)
    residual = abs(power_in - power_sum) / denom

    for line in lines:
        line.share = line.power_W / power_sum if power_sum > 0 else 0.0

    momentum_power = float(momentum_flux_N) * V
    quadrature = panel_force_N - total_force - momentum_flux_N
    return {
        "speed_m_s": V,
        "channels": [line.as_dict() for line in lines],
        "total_channel_force_N": total_force,
        "panel_force_N": float(panel_force_N),
        "power_in_W": power_in,
        "power_channels_W": power_sum,
        "residual": float(residual),
        "tolerance": float(tol),
        "passed": bool(residual <= tol),
        "attribution": {
            "momentum_flux_N": float(momentum_flux_N),
            "momentum_flux_power_W": momentum_power,
            "momentum_flux_share": momentum_power / denom,
            "quadrature_N": float(quadrature),
            "quadrature_share": quadrature * V / denom,
            "fsi_coupling_residual": float(fsi_residual),
        },
    }


def accumulate(history: Sequence[Mapping[str, Any]], window_fraction: float = 0.25) -> Dict[str, Any]:
    """Steady-window statistics of a ledger history.

    The last ``window_fraction`` of the time loop is used; the first part is the
    transient during which the ship-generated ice channel is establishing.
    """
    if not history:
        raise ValueError("empty ledger history")
    n = len(history)
    start = max(0, int(round(n * (1.0 - window_fraction))))
    window = history[start:]
    names = CHANNELS
    mean_forces = {
        name: sum(float(row["channel_forces"][name]) for row in window) / len(window)
        for name in names
    }
    mean_residual = sum(float(row["residual"]) for row in window) / len(window)
    max_residual = max(float(row["residual"]) for row in window)
    return {
        "n_samples": n,
        "window_start_index": start,
        "window_samples": len(window),
        "mean_channel_forces_N": mean_forces,
        "mean_total_force_N": sum(mean_forces.values()),
        "mean_panel_force_N": sum(float(r["panel_force_N"]) for r in window) / len(window),
        "mean_residual": mean_residual,
        "max_residual_window": max_residual,
        "max_residual_all": max(float(r["residual"]) for r in history),
        "mean_speed_m_s": sum(float(r["speed_m_s"]) for r in window) / len(window),
    }