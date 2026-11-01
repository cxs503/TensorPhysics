"""Stefan (moving boundary) reference solutions for the melt film.

Two independent solvers live here:

1. ``neumann_lambda`` / ``neumann_front`` -- the classical similarity solution of
   the one-phase Stefan problem with constant thermophysical properties,

       X(t) = 2 lambda sqrt(alpha t),
       lambda exp(lambda^2) erf(lambda) = Ste / sqrt(pi),
       Ste   = c (T_melt - T_far) / L_f .

   It is used as the *analytic reference* for the moving boundary: for the
   freezing of a melt film on a cold structure (solid grows into liquid held at
   the melting point) and for melting of subcooled ice (melt grows into solid
   held below the melting point).

2. ``solve_stefan_1d`` -- a conservative 1-D enthalpy (apparent heat capacity)
   solver on a uniform grid.  It knows nothing about ``neumann_front``; the
   agreement of the two is therefore an independent verification of both the
   moving-boundary physics and its implementation (criteria (c) and its melting
   counterpart).

The quasi-steady limit ``X = sqrt(2 k DeltaT t / (rho L_f))`` and the
corresponding time-to-freeze are provided for order-of-magnitude checks; they
are first order accurate in the Stefan number and are deliberately *not* used as
acceptance references.

Limitations
-----------
* The similarity solution assumes two phases with *equal* thermophysical
  properties and a semi-infinite domain; the enthalpy solver uses the same
  assumption so that the comparison is meaningful.
* The melt film problem is one-dimensional in the contact-normal direction; the
  actual contact is a mixed-lubrication interface with distributed asperity
  temperatures, which this module does not resolve.
"""

from math import ceil, erf, exp, pi, sqrt

try:  # numpy is an optional dependency of the core package
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without numpy
    _np = None

__all__ = [
    "neumann_lambda",
    "neumann_front",
    "neumann_lambda_small_stefan",
    "quasi_steady_front",
    "solidification_time",
    "friction_heat_flux",
    "solve_stefan_1d",
]


def _require_numpy():
    if _np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for the 1-D Stefan reference solver")
    return _np


def neumann_lambda(stefan_number: float, tolerance: float = 1e-14, max_iter: int = 200) -> float:
    """Similarity constant ``lambda`` of the one-phase Stefan problem.

    Solves ``lambda exp(lambda^2) erf(lambda) = Ste/sqrt(pi)`` by bisection on a
    monotonically increasing function of ``lambda``.
    """
    if not isinstance(stefan_number, (int, float)) or isinstance(stefan_number, bool):
        raise ValueError("stefan_number must be a real number")
    if stefan_number <= 0.0:
        raise ValueError("stefan_number must be positive")
    target = stefan_number / sqrt(pi)

    def residual(lam: float) -> float:
        return lam * exp(lam * lam) * erf(lam) - target

    lo, hi = 0.0, 1.0
    while residual(hi) < 0.0:
        hi *= 2.0
        if hi > 24.0:  # exp(576) already overflows float
            raise ValueError("stefan_number too large for the similarity solution")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def neumann_lambda_small_stefan(stefan_number: float) -> float:
    """Leading-order similarity constant, ``lambda ~ sqrt(Ste/2)``.

    Used only as an asymptotic cross-check of :func:`neumann_lambda`.
    """
    if stefan_number <= 0.0:
        raise ValueError("stefan_number must be positive")
    return sqrt(stefan_number / 2.0)


def neumann_front(time: float, alpha: float, lam: float) -> float:
    """Moving boundary position [m] of the similarity solution."""
    if time < 0.0:
        raise ValueError("time must be non-negative")
    if alpha <= 0.0 or lam <= 0.0:
        raise ValueError("alpha and lam must be positive")
    return 2.0 * lam * sqrt(alpha * time)


def quasi_steady_front(time: float, k: float, rho: float, latent_heat: float, delta_t: float) -> float:
    """Quasi-steady (linear temperature profile) front position [m].

    Valid to first order in the Stefan number; kept for order-of-magnitude
    comparisons only.
    """
    if time < 0.0:
        raise ValueError("time must be non-negative")
    if min(k, rho, latent_heat) <= 0.0:
        raise ValueError("thermophysical inputs must be positive")
    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive")
    return sqrt(2.0 * k * delta_t * time / (rho * latent_heat))


def solidification_time(
    thickness: float, k: float, rho: float, latent_heat: float, delta_t: float
) -> float:
    """Quasi-steady time [s] to freeze a film of ``thickness``."""
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    return rho * latent_heat * thickness * thickness / (2.0 * k * delta_t)


def friction_heat_flux(shear_stress: float, velocity: float) -> float:
    """Frictional power per unit nominal contact area [W/m^2]."""
    if shear_stress < 0.0 or velocity < 0.0:
        raise ValueError("shear_stress and velocity must be non-negative")
    return shear_stress * velocity


def solve_stefan_1d(
    *,
    length: float,
    n_cells: int,
    t_end: float,
    melting_temperature: float,
    wall_temperature: float,
    initial_temperature: float,
    rho: float,
    c: float,
    k: float,
    latent_heat: float,
    mode: str = "freezing",
    far_boundary: str = "adiabatic",
    sample_times: list[float] | None = None,
    dt_safety: float = 0.4,
    front_correction: float = 0.0,
) -> dict:
    """Explicit enthalpy solution of a 1-D two-phase Stefan problem.

    Parameters
    ----------
    length, n_cells : domain extent [m] and number of uniform cells.
    t_end : final time [s].
    melting_temperature : isothermal phase-change temperature [K].
    wall_temperature : Dirichlet temperature at x = 0 [K].  In ``freezing`` mode
        it must be below the melting temperature, in ``melting`` mode at the
        melting temperature.
    initial_temperature : uniform initial temperature [K].  In ``freezing`` mode
        the domain starts fully liquid (typically at the melting temperature);
        in ``melting`` mode it starts fully solid and subcooled.
    mode : ``"freezing"`` (solid grows from x = 0) or ``"melting"`` (melt grows
        from x = 0 into subcooled solid).  Melting requires a *superheated* wall
        (``wall_temperature > melting_temperature``): with the wall held exactly
        at the melting point the lumped enthalpy of the first cell approaches the
        melting enthalpy asymptotically from below and the front never enters the
        neighbouring cell, which is the classical pathology of the enthalpy
        method for a zero-superheat Dirichlet boundary (the physical front is
        thinner than one cell for all times).
    far_boundary : ``"adiabatic"`` (exact for freezing into isothermal liquid)
        or ``"dirichlet"`` at ``initial_temperature`` (required for the
        subcooled-melting case).
    sample_times : ascending times at which the front position is recorded.
    front_correction : optional constant offset [m] subtracted from the reported
        front; only used to document the half-cell convention.

    Returns
    -------
    dict with ``times`` [s], ``front`` [m] (frozen depth in freezing mode, melt
    depth in melting mode), ``dx``, ``dt``, ``n_steps`` and the final
    temperature field.
    """
    np = _require_numpy()
    if mode not in ("freezing", "melting"):
        raise ValueError("mode must be 'freezing' or 'melting'")
    if far_boundary not in ("adiabatic", "dirichlet"):
        raise ValueError("far_boundary must be 'adiabatic' or 'dirichlet'")
    if not isinstance(n_cells, int) or isinstance(n_cells, bool) or n_cells < 8:
        raise ValueError("n_cells must be an integer >= 8")
    if length <= 0.0 or t_end <= 0.0:
        raise ValueError("length and t_end must be positive")
    if min(rho, c, k, latent_heat) <= 0.0:
        raise ValueError("thermophysical inputs must be positive")
    if not 0.0 < dt_safety <= 0.5:
        raise ValueError("dt_safety must lie in (0, 0.5]")

    dx = length / n_cells
    rho_c = rho * c
    alpha = k / rho_c
    dt_stable = dt_safety * dx * dx / alpha

    # enthalpy per unit volume relative to the melting state
    h_melt = rho * latent_heat
    if mode == "freezing":
        if wall_temperature >= melting_temperature:
            raise ValueError("freezing mode requires wall_temperature < melting_temperature")
        h = _np_full(np, n_cells, h_melt + rho_c * (initial_temperature - melting_temperature))
    else:
        if wall_temperature <= melting_temperature:
            raise ValueError(
                "melting mode requires a superheated wall (wall_temperature > "
                "melting_temperature); an isothermal wall has no resolvable melt "
                "layer in the lumped enthalpy scheme"
            )
        if initial_temperature >= melting_temperature:
            raise ValueError("melting mode requires subcooled initial_temperature")
        h = _np_full(np, n_cells, rho_c * (initial_temperature - melting_temperature))

    def temperature_of(field):
        t = np.where(
            field < 0.0,
            melting_temperature + field / rho_c,
            np.where(
                field > h_melt,
                melting_temperature + (field - h_melt) / rho_c,
                melting_temperature,
            ),
        )
        return t

    def liquid_fraction_of(field):
        return np.clip(field / h_melt, 0.0, 1.0)

    def front_of(field):
        frac = liquid_fraction_of(field)
        if mode == "freezing":
            depth = dx * float(np.sum(1.0 - frac))
        else:
            depth = dx * float(np.sum(frac))
        return max(depth - front_correction, 0.0)

    if sample_times is None:
        sample_times = [t_end]
    sample_times = sorted(float(t) for t in sample_times)
    if sample_times[0] < 0.0 or sample_times[-1] > t_end + 1e-18:
        raise ValueError("sample_times must lie inside [0, t_end]")

    times: list[float] = []
    fronts: list[float] = []
    time = 0.0
    steps_done = 0
    for target in sample_times:
        while time < target - 1e-18:
            step = min(dt_stable, target - time)
            temp = temperature_of(h)
            laplacian = np.zeros_like(h)
            interior = k * (temp[2:] - 2.0 * temp[1:-1] + temp[:-2]) / (dx * dx)
            laplacian[1:-1] = interior
            # Dirichlet wall: half-cell control volume
            laplacian[0] = k * (wall_temperature - temp[0]) / (dx * dx * 0.5)
            if far_boundary == "dirichlet":
                laplacian[-1] = k * (initial_temperature - temp[-1]) / (dx * dx * 0.5)
            h = h + step * laplacian
            time += step
            steps_done += 1
            if steps_done > 50_000_000:  # pragma: no cover
                raise RuntimeError("enthalpy solver exceeded its step budget")
        times.append(time)
        fronts.append(front_of(h))

    return {
        "times": times,
        "front": fronts,
        "dx": dx,
        "dt": dt_stable,
        "n_steps": steps_done,
        "temperature": temperature_of(h).tolist(),
        "liquid_fraction": liquid_fraction_of(h).tolist(),
        "mode": mode,
        "far_boundary": far_boundary,
    }


def _np_full(np, n: int, value: float):
    return np.full(n, value, dtype=float)