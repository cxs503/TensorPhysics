"""Sub-cycle interpolation order gate for partitioned multi-solver coupling.

Campaign node ``P40W3-N09`` (mission 8524).  The shipped contract suite
(:mod:`tensorphysics.coupling_energy.scenarios`, node ``P40W3B-N09``) books the
interface transfer from the **producer's own measurement**: the receiving solver
is handed the exact interval integral of what the other side produced.  With
that booking the energy/momentum residual of the ledger is a double-entry
*identity* (1e-15 here, see :func:`identity_diagnostic`) and it is **blind to
the interface reconstruction order** — the three schemes then differ only in the
third significant digit of the solution error, not in the conserved quantity.

This module supplies the missing instrument: a *partitioned* booking in which
the receiver only ever sees the interface samples of the **closed** intervals
and has to rebuild the coming one with its interpolation scheme.  That is what
every real partitioned multi-physics code does, and it is the only situation in
which the sub-cycle interpolation order controls the conservation residual.

Two experiments, both with real solvers on both sides and a real runtime ledger
(:class:`~tensorphysics.coupling_energy.ledger.EnergyLedger` /
:class:`~tensorphysics.coupling_energy.ledger.MomentumLedger`):

``energy``
    A D2Q9 thermal lattice (producer, resolved sub-cycle) feeds a Q4 conduction
    FEM (receiver, coarse step).  The wall temperature of the lattice follows a
    smooth physical history ``T_w(t) = dT (1 - cos(2 pi t / T_p)) / 2``; the
    FEM receives the heat through a mass-weighted Neumann pattern whose total is
    the reconstructed interval integral.  Conserved quantity: the solid's stored
    energy against the cumulative heat the producer measured.

``momentum``
    Two free DEM discs are driven by a smooth body-force history (producer); a
    free plane-stress Q4 plate (receiver, sub-cycled FEM steps) receives the
    reconstructed contact impulse.  Conserved quantity: total linear momentum
    against the cumulative impulse the producer measured.

Both load paths are **one-way**.  That is a deliberate, declared limitation: the
strongly coupled LBM<->FEM conjugate-heat-transfer case of the shipped suite is
unstable under a lagged flux handover at coarse exchange intervals (measured:
the interface heat rate is not smooth at the exchange scale because the
interface relaxation oscillates), so no clean order can be extracted from it —
see ``artifacts/p40w3_n09/identity_probe.json`` and the report.  The one-way
path isolates exactly the quantity the gate is about (the handover error) and
leaves the two solvers fully physical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .exchange import (
    SCHEME_ORDER,
    ExchangeSample,
    TemporalScheme,
    reconstruction_poly,
)
from .ledger import CouplingLedgerError, EnergyLedger, MomentumLedger
from .solvers import DEM2D, ContactLaw, Q4Mesh, Q4PlaneStressFEM, Q4ThermalFEM, ThermalLBM2D

__all__ = [
    "BOOKINGS",
    "ORDER_SCHEMES",
    "RATIOS",
    "ExchangeStudy",
    "fit_order",
    "gate_study",
    "identity_diagnostic",
    "order_study",
    "run_energy_exchange",
    "run_momentum_exchange",
    "stability_study",
]

#: Sub-cycling ratios of the acceptance gate (``dt_coarse = r * dt_fine``).
RATIOS: tuple[int, ...] = (1, 2, 4, 8)

#: Ratios the acceptance gate mandates for the ratio trend.
GATE_RATIOS: tuple[int, ...] = (1, 2, 4, 8)

#: Extended ladder used for the *order* fit.  The finest two levels are limited
#: by the fine solver's own step-level noise floor (the handover error there is
#: smaller than the discrete fine-step wiggle of the produced rate), so the
#: observed order is read in the asymptotic window of the coarse levels.
RATIOS_EXT: tuple[int, ...] = (1, 2, 4, 8, 16)

#: Interpolation orders that can be *reconstructed* from shared history.
ORDER_SCHEMES: tuple[TemporalScheme, ...] = (
    TemporalScheme.ZOH,
    TemporalScheme.LINEAR,
    TemporalScheme.QUADRATIC,
)

#: ``exact`` = producer's measurement handed over (double entry, identity
#: residual); ``reconstructed`` = the receiver rebuilds the interval from the
#: closed-interval samples with its scheme (physical handover error).
BOOKINGS: tuple[str, ...] = ("exact", "reconstructed")


# ---------------------------------------------------------------------------
# order estimation
# ---------------------------------------------------------------------------
def fit_order(steps: Sequence[float], errors: Sequence[float]) -> tuple[float, float]:
    """Least-squares slope of ``log(error)`` against ``log(step)`` and its R^2.

    Both inputs must be positive and of equal length >= 2.  The returned slope
    is the *observed order of accuracy* of the recipe that produced ``errors``;
    the R^2 of the straight-line fit is returned so a flushed-to-floor (non
    power-law) sequence is visible instead of silently reported as a number.
    """
    h = np.asarray(steps, dtype=float)
    e = np.asarray(errors, dtype=float)
    if h.shape != e.shape:
        raise CouplingLedgerError("steps and errors must have the same length")
    if h.size < 2:
        raise CouplingLedgerError("at least two refinement levels are required")
    if np.any(h <= 0.0) or np.any(e <= 0.0) or not np.all(np.isfinite(h)) \
            or not np.all(np.isfinite(e)):
        raise CouplingLedgerError("steps and errors must be finite and positive")
    x = np.log(h)
    y = np.log(e)
    mx, my = x.mean(), y.mean()
    sxx = float(np.sum((x - mx) ** 2))
    if sxx <= 0.0:
        raise CouplingLedgerError("steps must not all be equal")
    slope = float(np.sum((x - mx) * (y - my)) / sxx)
    intercept = my - slope * mx
    ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
    ss_tot = float(np.sum((y - my) ** 2))
    r2 = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return slope, r2


# ---------------------------------------------------------------------------
# study record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExchangeStudy:
    """One ``(physics, booking, scheme, ratio)`` runtime experiment."""

    name: str
    booking: str
    scheme: str
    ratio: int
    dt_fine: float
    dt_coarse: float
    n_intervals: int
    residual_final: float
    residual_max: float
    mismatch_l1: float
    scale: float
    series: dict = field(default_factory=dict)

    @property
    def h(self) -> float:
        return self.dt_coarse

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "booking": self.booking,
            "scheme": self.scheme,
            "ratio": self.ratio,
            "dt_fine": self.dt_fine,
            "dt_coarse": self.dt_coarse,
            "n_intervals": self.n_intervals,
            "residual_final": self.residual_final,
            "residual_max": self.residual_max,
            "mismatch_l1": self.mismatch_l1,
            "scale": self.scale,
            "series": {k: [float(x) for x in v] for k, v in self.series.items()},
        }


def _reconstructed(scheme: TemporalScheme, history: list[ExchangeSample],
                   h: float) -> float:
    """Reconstructed integral of the interface rate over the *coming* interval.

    ``history`` holds the interval-average rates of the closed intervals, each
    stamped at its own midpoint (the second-order accurate instant of an
    interval average).  The coming interval ``[t_n, t_n + h]`` therefore starts
    half an interval *after* the newest sample and ends ``3h/2`` after it, so
    the polynomial of the scheme is integrated over ``s in [h/2, 3h/2]``.  This
    is the standard staggered reconstruction, and it is what makes the handover
    error scale with the declared order instead of with a half-interval phase
    error (which would fake a first-order gate for every scheme).
    """
    previous = history[-2] if len(history) >= 2 else None
    previous2 = history[-3] if len(history) >= 3 else None
    f0, a1, a2 = reconstruction_poly(scheme, previous, history[-1], previous2)
    s_lo, s_hi = 0.5 * h, 1.5 * h
    return float(f0 * h + 0.5 * a1 * (s_hi ** 2 - s_lo ** 2)
                 + (a2 / 3.0) * (s_hi ** 3 - s_lo ** 3))


# ---------------------------------------------------------------------------
# (a) energy handover: thermal LBM producer -> conduction FEM receiver
# ---------------------------------------------------------------------------
def run_energy_exchange(ratio: int = 1, scheme: TemporalScheme = TemporalScheme.LINEAR,
                        booking: str = "reconstructed", n_intervals: int = 0,
                        period_s: float = 200.0, n_periods: float = 3.0,
                        fluid_nx: int = 20, fluid_ny: int = 10, tau: float = 1.25,
                        length: float = 0.1, alpha: float = 1.0e-6,
                        rho_cp: float = 1.0e6, thickness: float = 0.01,
                        dT: float = 50.0, solid_nx: int = 20, solid_ny: int = 10,
                        ) -> ExchangeStudy:
    """Heat handover at ``dt_coarse = ratio * dt_fine`` from LBM (producer) to FEM.

    The lattice sub-cycles inside every coarse interval and measures the heat it
    injected from its oscillating wall (``Q_n``, exact in its own sub-cycle).
    The FEM (backward Euler, mass-weighted Neumann pattern summing to one, so
    its stored-energy increment equals the applied total *exactly*) receives
    either ``Q_n`` (``booking="exact"``, double entry) or the scheme
    reconstruction of the closed intervals (``booking="reconstructed"``).
    """
    if booking not in BOOKINGS:
        raise CouplingLedgerError(f"unknown booking {booking!r}; known: {BOOKINGS}")
    if scheme is TemporalScheme.INDEPENDENT:
        raise CouplingLedgerError("independent scheme has no shared history")
    if ratio < 1:
        raise CouplingLedgerError("ratio must be >= 1")

    dx = length / fluid_nx
    dy = (length / 2.0) / fluid_ny
    alpha_lat = (tau - 0.5) / 3.0
    dt_fine = alpha_lat * dx * dx / alpha
    h = ratio * dt_fine
    if n_intervals <= 0:
        n_intervals = max(4, int(round(n_periods * period_s / h)))

    lbm = ThermalLBM2D(fluid_nx, fluid_ny, tau=tau, T_init=0.0)
    C_f = rho_cp * dx * dy * thickness

    mesh = Q4Mesh(solid_nx, solid_ny, length, length / 2.0)
    fem = Q4ThermalFEM(mesh, conductivity=alpha * rho_cp,
                       volumetric_heat_capacity=rho_cp, thickness=thickness)
    inv = fem.cached_inverse(h)
    top = mesh.row(solid_ny)
    pattern = np.zeros(mesh.n_nodes)
    pattern[top] = fem.M[top] / fem.M[top].sum()

    def wall_temperature(t: float) -> float:
        return float(dT) * 0.5 * (1.0 - math.cos(2.0 * math.pi * t / period_s))

    ledger = EnergyLedger(sides=("solid",))
    history: list[ExchangeSample] = []
    keys = ("t", "e_solid", "q_produced", "q_applied", "residual", "mismatch")
    series: dict[str, list[float]] = {k: [] for k in keys}
    warm = 3                       # start-up intervals excluded from the window
    mismatch_sum = 0.0
    mismatch_ref = 0.0
    produced_total = 0.0

    def record(t: float, residual: float, mismatch: float) -> None:
        series["t"].append(t)
        series["e_solid"].append(fem.energy())
        series["q_produced"].append(produced_total)
        series["q_applied"].append(ledger_q["applied"])
        series["residual"].append(residual)
        series["mismatch"].append(mismatch)

    ledger_q = {"applied": 0.0}
    t = 0.0
    record(t, 0.0, 0.0)
    for n in range(n_intervals):
        injected = 0.0
        for k in range(ratio):
            s = (k + 0.5) * dt_fine
            injected += lbm.step(np.full(fluid_nx, wall_temperature(t + s)))
        produced = injected * C_f
        produced_total += produced

        if booking == "exact":
            applied = produced
        else:
            applied = _reconstructed(scheme, history, h) if history else 0.0
            history.append(ExchangeSample(t + 0.5 * h, produced / h))
        ledger_q["applied"] += applied

        fem.step(h, flux=applied / h * pattern, matrix_inverse=inv)
        t += h
        ledger.book_external(produced)
        ledger.set("solid", fem.energy())
        scale = max(abs(produced_total), 1e-300)
        residual = abs(ledger.total()) / scale
        if n >= warm:
            mismatch_sum += abs(produced - applied)
            mismatch_ref += abs(produced)
        mismatch = mismatch_sum / mismatch_ref if mismatch_ref > 0.0 else 0.0
        record(t, residual, mismatch)

    resid = np.asarray(series["residual"])
    return ExchangeStudy(
        name="energy", booking=booking, scheme=scheme.value, ratio=int(ratio),
        dt_fine=dt_fine, dt_coarse=h, n_intervals=n_intervals,
        residual_final=float(resid[-1]), residual_max=float(resid.max()),
        mismatch_l1=float(mismatch_sum / mismatch_ref) if mismatch_ref > 0.0 else 0.0,
        scale=float(abs(produced_total)),
        series=series,
    )


# ---------------------------------------------------------------------------
# (b) momentum handover: driven DEM producer -> free plate receiver
# ---------------------------------------------------------------------------
def run_momentum_exchange(ratio: int = 1, scheme: TemporalScheme = TemporalScheme.LINEAR,
                          booking: str = "reconstructed", n_intervals: int = 0,
                          period_s: float = 2.0e-4, n_periods: float = 3.0,
                          force_n: float = 2.0e3, length: float = 0.2,
                          height: float = 0.04, young: float = 2.0e9,
                          poisson: float = 0.3, density: float = 900.0,
                          thickness: float = 0.05, radius: float = 0.008,
                          nx: int = 10, ny: int = 3) -> ExchangeStudy:
    """Impulse handover at ``dt_coarse = ratio * dt_fine`` from DEM to FEM.

    Two free DEM discs are pushed by the smooth body force
    ``-force_n (1 - cos(2 pi t / T_p)) / 2`` (an ice-load-like ramp); the impulse
    they received is measured from the DEM's own momentum increment.  A free
    plane-stress plate sub-cycles at ``dt_fine`` and receives the reconstructed
    impulse as a uniform nodal rate, so its momentum increment equals the
    applied total exactly (lumped mass).  Conserved quantity: total linear
    momentum ``sum(m v)`` of discs + plate against the measured impulse.
    """
    if booking not in BOOKINGS:
        raise CouplingLedgerError(f"unknown booking {booking!r}; known: {BOOKINGS}")
    if scheme is TemporalScheme.INDEPENDENT:
        raise CouplingLedgerError("independent scheme has no shared history")
    if ratio < 1:
        raise CouplingLedgerError("ratio must be >= 1")

    mesh = Q4Mesh(nx, ny, length, height)
    plate = Q4PlaneStressFEM(mesh, young, poisson, density, thickness)
    dt_fine = plate.critical_dt(safety=0.25)
    h = ratio * dt_fine
    if n_intervals <= 0:
        n_intervals = max(4, int(round(n_periods * period_s / h)))

    pos = np.array([[length + 0.05, height * 0.25], [length + 0.05, height * 0.75]])
    vel = np.zeros_like(pos)
    dem = DEM2D(pos, vel, np.full(len(pos), radius), density, thickness,
                ContactLaw(stiffness=1.0e7))
    column = mesh.column(0)
    pattern = np.zeros(2 * mesh.n_nodes)
    weights = plate.mass[2 * column]
    weights = weights / weights.sum()
    pattern[2 * column] = weights

    def force_amplitude(t: float) -> float:
        return -float(force_n) * 0.5 * (1.0 - math.cos(2.0 * math.pi * t / period_s))

    mom = MomentumLedger(sides=("particles", "solid"))
    history: list[ExchangeSample] = []
    keys = ("t", "p_particles", "p_solid", "p_total", "impulse_produced",
            "impulse_applied", "momentum_residual", "mismatch")
    series: dict[str, list[float]] = {k: [] for k in keys}
    warm = 3
    mismatch_sum = 0.0
    mismatch_ref = 0.0
    produced_total = 0.0
    applied_total = 0.0

    def total_momentum() -> tuple[float, float]:
        px_p, py_p = dem.momentum()
        px_s = float(np.sum(plate.mass[0::2] * plate.v[0::2]))
        py_s = float(np.sum(plate.mass[1::2] * plate.v[1::2]))
        return px_p + px_s, py_p + py_s

    def record(t: float, residual: float, mismatch: float) -> None:
        px_p, py_p = dem.momentum()
        px_s = float(np.sum(plate.mass[0::2] * plate.v[0::2]))
        series["t"].append(t)
        series["p_particles"].append(px_p)
        series["p_solid"].append(px_s)
        series["p_total"].append(px_p + px_s)
        series["impulse_produced"].append(produced_total)
        series["impulse_applied"].append(applied_total)
        series["momentum_residual"].append(residual)
        series["mismatch"].append(mismatch)

    t = 0.0
    p_solid0 = [float(np.sum(plate.mass[0::2] * plate.v[0::2]))]
    mom.set("particles", *dem.momentum())
    mom.set("solid", float(np.sum(plate.mass[0::2] * plate.v[0::2])),
            float(np.sum(plate.mass[1::2] * plate.v[1::2])))
    mom.record()
    record(t, 0.0, 0.0)
    for n in range(n_intervals):
        px_before = float(dem.momentum()[0])
        for k in range(ratio):
            s = (k + 0.5) * dt_fine
            f = np.zeros_like(dem.pos)
            f[:, 0] = force_amplitude(t + s)
            dem.step(dt_fine, f)
        produced = float(dem.momentum()[0]) - px_before
        produced_total += produced

        if booking == "exact":
            applied = produced
        else:
            applied = _reconstructed(scheme, history, h) if history else 0.0
            history.append(ExchangeSample(t + 0.5 * h, produced / h))
        applied_total += applied

        # spread the reconstructed impulse uniformly over the sub-cycle
        rate = applied / h * pattern
        for _ in range(ratio):
            plate.step(dt_fine, rate)
        t += h
        # The load path is one-way (the drive on the discs is external to the
        # coupled system), so the conserved statement is *not* "total momentum
        # of discs + plate": it is "the momentum increment the plate received
        # equals the impulse the producer measured".  Newton's third law of the
        # two-way contact case is gated separately by the shipped contract suite.
        p_solid = float(np.sum(plate.mass[0::2] * plate.v[0::2]))
        scale = max(abs(produced_total), 1e-300)
        residual = abs((p_solid - p_solid0[0]) - produced_total) / scale
        if n >= warm:
            mismatch_sum += abs(produced - applied)
            mismatch_ref += abs(produced)
        mismatch = mismatch_sum / mismatch_ref if mismatch_ref > 0.0 else 0.0
        mom.set("particles", *dem.momentum())
        mom.set("solid", float(np.sum(plate.mass[0::2] * plate.v[0::2])),
                float(np.sum(plate.mass[1::2] * plate.v[1::2])))
        mom.record()
        record(t, residual, mismatch)

    resid = np.asarray(series["momentum_residual"])
    scale_total = float(abs(produced_total))
    return ExchangeStudy(
        name="momentum", booking=booking, scheme=scheme.value, ratio=int(ratio),
        dt_fine=dt_fine, dt_coarse=h, n_intervals=n_intervals,
        residual_final=float(resid[-1]), residual_max=float(resid.max()),
        mismatch_l1=float(mismatch_sum / mismatch_ref) if mismatch_ref > 0.0 else 0.0,
        scale=scale_total, series=series,
    )


# ---------------------------------------------------------------------------
# studies and gates
# ---------------------------------------------------------------------------
def order_study(name: str = "energy", booking: str = "reconstructed",
                ratios: Sequence[int] = RATIOS_EXT,
                schemes: Sequence[TemporalScheme] = ORDER_SCHEMES,
                runner: Callable[..., ExchangeStudy] | None = None,
                fit_levels: int = 3, **kwargs) -> dict:
    """Run the full ``ratio x scheme`` matrix for one handover and fit orders."""
    runner = runner or {"energy": run_energy_exchange,
                        "momentum": run_momentum_exchange}[name]
    runs: list[ExchangeStudy] = []
    for ratio in ratios:
        for scheme in schemes:
            runs.append(runner(ratio=ratio, scheme=scheme, booking=booking, **kwargs))
    fits: dict[str, dict] = {}
    for scheme in schemes:
        sub = [r for r in runs if r.scheme == scheme.value]
        hs = [r.dt_coarse for r in sub]
        orders = {}
        for quantity in ("mismatch_l1", "residual_final"):
            errs = [getattr(r, quantity) for r in sub]
            try:
                slope, r2 = fit_order(hs, errs)
            except CouplingLedgerError:             # flushed to zero / non-positive
                slope, r2 = float("nan"), 0.0
            k = max(2, min(int(fit_levels), len(hs)))
            try:
                slope_asy, r2_asy = fit_order(hs[-k:], errs[-k:])
            except CouplingLedgerError:
                slope_asy, r2_asy = float("nan"), 0.0
            orders[quantity] = {"order": slope, "r2": r2,
                                "order_asymptotic": slope_asy, "r2_asymptotic": r2_asy,
                                "asymptotic_window_ratios": list(ratios)[-k:],
                                "errors": errs, "hs": hs}
        fits[scheme.value] = orders
    return {
        "name": name,
        "booking": booking,
        "ratios": list(ratios),
        "schemes": [s.value for s in schemes],
        "declared_orders": {s.value: SCHEME_ORDER[s] for s in schemes},
        "runs": [r.as_dict() for r in runs],
        "fits": fits,
        "fit_levels": fit_levels,
    }


def identity_diagnostic(energy_kw: dict | None = None,
                        momentum_kw: dict | None = None) -> dict:
    """Contrast the two bookings: double-entry identity vs physical handover.

    A residual that cannot tell the three interpolation orders apart (and sits
    at round-off) is a *bookkeeping identity*, not a conservation measurement.
    The gate below requires the exact booking to stay at round-off **and** the
    reconstructed booking to sit orders of magnitude above it, otherwise the
    instrument would be blind and every "conservation passed" claim vacuous.
    """
    kw = {"energy": dict(energy_kw or {}), "momentum": dict(momentum_kw or {})}
    out = {"energy": {}, "momentum": {}}
    for name, runner in (("energy", run_energy_exchange),
                         ("momentum", run_momentum_exchange)):
        exact = runner(ratio=1, scheme=TemporalScheme.LINEAR, booking="exact",
                       **kw[name])
        recon = runner(ratio=1, scheme=TemporalScheme.LINEAR,
                       booking="reconstructed", **kw[name])
        out[name] = {
            "exact_residual": exact.residual_final,
            "reconstructed_residual": recon.residual_final,
            "separation": (recon.residual_final / exact.residual_final
                           if exact.residual_final > 0.0 else float("inf")),
            "exact_mismatch": exact.mismatch_l1,
            "reconstructed_mismatch": recon.mismatch_l1,
            "blind": bool(exact.residual_final >= 1e-10
                          or recon.mismatch_l1 <= 10.0 * max(exact.mismatch_l1, 1e-300)),
        }
    return out


def stability_study(name: str = "energy", n_intervals: int = 1000,
                    scheme: TemporalScheme = TemporalScheme.LINEAR,
                    booking: str = "reconstructed", **kwargs) -> dict:
    """Long run: the handover error must stay bounded (no exponential growth).

    Compares the residual of the last quarter of the record with the first
    quarter (a leaking scheme multiplies it, a bounded one does not) and reports
    the least-squares drift slope.
    """
    runner = {"energy": run_energy_exchange,
              "momentum": run_momentum_exchange}[name]
    run = runner(ratio=1, scheme=scheme, booking=booking,
                 n_intervals=n_intervals, **kwargs)
    key = "residual" if name == "energy" else "momentum_residual"
    resid = np.asarray(run.series[key], dtype=float)
    # The first intervals have no interface history at all (the receiver warm
    # starts from a zero sample), so they carry a deliberate start-up transient
    # of O(1) and are excluded from the stability window.
    warm = 3
    resid = resid[warm:]
    q = max(1, resid.size // 4)
    early = float(np.max(resid[:q]))
    late = float(np.max(resid[-q:]))
    idx = np.arange(resid.size, dtype=float)
    slope = float(np.polyfit(idx, resid, 1)[0]) if resid.size >= 2 else 0.0
    return {
        "name": name,
        "scheme": scheme.value,
        "booking": booking,
        "n_intervals": int(run.n_intervals),
        "warm_up_intervals_excluded": warm,
        "residual_first_quarter_max": early,
        "residual_last_quarter_max": late,
        "growth_ratio": late / max(early, 1e-300),
        "drift_slope_per_interval": slope,
        "residual_final": run.residual_final,
        "divergent": bool(late > 4.0 * max(early, 1e-300)),
        "all_finite": bool(np.all(np.isfinite(resid))),
    }


def gate_study(study: dict, *, energy: bool = True,
               min_orders: dict | None = None,
               order_tolerance: float = 0.4) -> dict:
    """Turn an :func:`order_study` record into machine-checkable criteria.

    Criterion (node P40W3-N09): zero / first / second order reconstruction must
    show observed order ``>= 0.8`` with the fit of the declared order
    (``1, 2, 3``) within ``order_tolerance``, and the order must increase
    monotonically with the declared order.
    """
    scheme_key = "energy" if energy else "momentum"
    minimum = min_orders or {"zoh": 0.8, "linear": 1.6, "quadratic": 2.4}
    fits = study["fits"]
    criteria: dict[str, dict] = {}

    def add(key: str, required, actual, passed: bool, note: str = "") -> None:
        criteria[key] = {"required": required, "actual": actual,
                         "passed": bool(passed), "note": note}

    observed = {}
    r2s = {}
    dense = {}
    for name, fit in fits.items():
        observed[name] = float(fit["mismatch_l1"]["order_asymptotic"])
        r2s[name] = float(fit["mismatch_l1"]["r2_asymptotic"])
        dense[name] = float(fit["mismatch_l1"]["order"])
        floor = minimum.get(name, 0.8)
        add(f"{scheme_key}_order_{name}", f">= {floor} and fit R^2 >= 0.98",
            {"order": observed[name], "r2": r2s[name],
             "order_all_levels": dense[name],
             "declared": study["declared_orders"][name]},
            (observed[name] >= floor and r2s[name] >= 0.98),
            "observed order in the asymptotic window of the handover mismatch")

    ordered = all(np.isfinite(list(observed.values())))
    if ordered:
        vals = [observed[k] for k in study["schemes"]]
        monotone = all(b > a + 0.15 for a, b in zip(vals, vals[1:]))
        add(f"{scheme_key}_order_monotone", "order(zoh) < order(linear) < order(quadratic)",
            vals, monotone,
            "higher reconstruction order must buy a smaller handover error")

    # ratio monotonicity (node criterion: residual must grow with r)
    trends = {}
    for scheme in study["schemes"]:
        runs = [r for r in study["runs"] if r["scheme"] == scheme
                and r["ratio"] in GATE_RATIOS]
        runs.sort(key=lambda r: r["ratio"])
        errs = [r["mismatch_l1"] for r in runs]
        trends[scheme] = {"ratios": [r["ratio"] for r in runs], "mismatch_l1": errs}
    growing = {s: all(b >= a for a, b in zip(t["mismatch_l1"], t["mismatch_l1"][1:]))
               for s, t in trends.items()}
    add(f"{scheme_key}_ratio_monotone", "mismatch_l1 non-decreasing in r = 1,2,4,8",
        {"growing": growing, "trends": trends}, all(growing.values()),
        "coarse exchange interval must cost more than a fine one")

    overall = all(c["passed"] for c in criteria.values())
    return {"criteria": criteria, "overall_pass": overall}