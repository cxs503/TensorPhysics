"""Independent finite-difference reference solver for the moving-load beam.

This solver deliberately shares *nothing* with the modal path except the load
definition (:class:`~tensorphysics.moving_load.paths.LoadPath`):

* space: second-order central differences for the fourth derivative on a
  uniform grid of ``N`` elements,
* time: explicit central differences (Leapfrog) with lumped nodal mass
  ``rho A h``,
* boundary conditions: ``u = 0`` and ``u'' = 0`` at both ends, imposed through
  the antisymmetric ghost nodes ``u_{-1} = -u_1`` and ``u_{N+1} = -u_{N-1}``.

Because it neither truncates the mode set nor uses Newmark, agreement with the
modal solver is a genuine cross-method check rather than a tautology.  The
scheme is conditionally stable with

    omega_max = 4 sqrt(EI / rho A) / h^2,   dt_crit = 2 / omega_max,

so ``dt`` is chosen from a CFL factor on that bound and the achieved ratio is
reported with the result.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

from .beam import HullGirderBeam
from .paths import LoadPath

__all__ = ["fd_beam_response", "fd_stability_limit"]


def fd_stability_limit(beam: HullGirderBeam, n_elements: int) -> Tuple[float, float]:
    """Return ``(omega_max, dt_crit)`` of the explicit FD scheme."""
    h = beam.length_m / int(n_elements)
    omega_max = 4.0 * beam.flexural_wave_speed / h ** 2
    return omega_max, 2.0 / omega_max


def fd_beam_response(
    beam: HullGirderBeam,
    paths: Sequence[LoadPath],
    duration_s: float,
    n_elements: int = 200,
    cfl: float = 0.7,
    probe_positions_m: Sequence[float] | None = None,
) -> Dict[str, object]:
    """Integrate the beam ODE in space and time without modal truncation.

    Parameters
    ----------
    duration_s
        Physical end time (s).
    n_elements
        Number of finite-difference cells; ``n_elements + 1`` nodes.
    cfl
        Fraction of the stability limit used for ``dt`` (``< 1``).
    probe_positions_m
        Stations at which the displacement/acceleration history is stored.

    Returns
    -------
    dict
        ``time_s``, ``dt_s``, ``n_elements``, ``dt_crit_s``, ``cfl_used``,
        ``probes`` (``{x: {'displacement_m', 'acceleration_m_s2'}}``),
        ``peak_profile`` (deflection profile at the instant of maximum midspan
        deflection), ``midspan_peak_m``, ``profile_x_m``.
    """
    if not (0.0 < cfl < 1.0):
        raise ValueError("cfl must lie in (0, 1)")
    n = int(n_elements)
    if n < 20:
        raise ValueError("n_elements must be >= 20")
    if duration_s <= 0.0 or not math.isfinite(duration_s):
        raise ValueError("duration_s must be positive and finite")

    h = beam.length_m / n
    omega_max, dt_crit = fd_stability_limit(beam, n)
    dt = cfl * dt_crit
    n_steps = int(math.ceil(duration_s / dt)) + 1
    coeff = beam.ei_nm2 / beam.rho_a_kg_m / h ** 4
    nodal_mass = beam.rho_a_kg_m * h

    n_nodes = n + 1
    u = np.zeros(n_nodes)
    u_old = np.zeros(n_nodes)

    if probe_positions_m is None:
        probe_positions_m = (0.5 * beam.length_m,)
    probes = {float(xp): np.zeros(n_steps) for xp in probe_positions_m}
    probe_acc = {float(xp): np.zeros(n_steps) for xp in probe_positions_m}
    probe_index = {float(xp): int(round(xp / h)) for xp in probe_positions_m}

    time_s = np.arange(n_steps) * dt
    path_mag = (
        np.array([p.magnitude_series(time_s) for p in paths])
        if paths
        else np.zeros((0, n_steps))
    )
    path_pos = (
        np.array([p.position_series(time_s) for p in paths]) if paths else np.zeros((0, n_steps))
    )

    mid_index = n // 2
    mid_peak = 0.0
    peak_profile = np.zeros(n_nodes)
    peak_time = 0.0

    for step in range(n_steps):
        # distributed load -> lumped nodal forces (linear hat distribution)
        force = np.zeros(n_nodes)
        for j in range(path_mag.shape[0]):
            mag = path_mag[j, step]
            if mag == 0.0:
                continue
            x = float(path_pos[j, step])
            k = int(math.floor(x / h))
            k = min(max(k, 0), n - 1)
            xi = (x - k * h) / h
            force[k] += mag * (1.0 - xi)
            force[k + 1] += mag * xi

        # padded array for the fourth difference with antisymmetric ghosts
        pad = np.empty(n_nodes + 2)
        pad[0] = -u[1]
        pad[1:-1] = u
        pad[-1] = -u[-2]
        d4 = pad[4:] - 4.0 * pad[3:-1] + 6.0 * pad[2:-2] - 4.0 * pad[1:-3] + pad[0:-4]

        u_new = np.empty(n_nodes)
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u_new[1:-1] = (
            2.0 * u[1:-1]
            - u_old[1:-1]
            - dt ** 2 * coeff * d4
            + dt ** 2 * force[1:-1] / nodal_mass
        )

        acc = (u_new - 2.0 * u + u_old) / dt ** 2
        for xp, idx in probe_index.items():
            probes[xp][step] = u[idx]
            probe_acc[xp][step] = acc[idx]

        mid_value = u_new[mid_index]
        if abs(mid_value) > abs(mid_peak):
            mid_peak = mid_value
            peak_profile = u_new.copy()
            peak_time = time_s[step]

        u_old, u = u, u_new

    return {
        "time_s": time_s,
        "dt_s": float(dt),
        "n_elements": n,
        "n_nodes": n_nodes,
        "h_m": float(h),
        "omega_max_rad_s": float(omega_max),
        "dt_crit_s": float(dt_crit),
        "cfl_used": float(cfl),
        "probe_positions_m": list(probes),
        "probes": {
            xp: {"displacement_m": probes[xp], "acceleration_m_s2": probe_acc[xp]}
            for xp in probes
        },
        "profile_x_m": np.linspace(0.0, beam.length_m, n_nodes),
        "peak_profile": peak_profile,
        "midspan_peak_m": float(mid_peak),
        "midspan_peak_sign": float(np.sign(mid_peak)),
        "peak_time_s": float(peak_time),
    }