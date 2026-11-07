"""Multi-rate two-integrator cross-check for the rational coupling clock.

Purpose
-------
:mod:`tensorphysics.campaign.clock_rational` makes the *clock* exact.  This
module answers the next question M12 actually cares about: does the exchange
cadence produce an acceptably accurate coupled solution, and is the declared
error band honoured *inside* the exchange interval (not only at the exchange
instants, where any scheme is exact by construction)?

Model
-----
Two linear oscillators coupled by a spring, each integrated by velocity Verlet
with its own step size -- a two-degree-of-freedom surrogate of a hull section
(master, mass ``m_master``) coupled to an ice/contact mass (sub, ``m_sub``)::

    m_m x1'' = -k_m x1 + k_c (x2 - x1)
    m_s x2'' = -k_s x2 - k_c (x2 - x1)

The system is linear, so the **exact** solution is available analytically from
the two eigenmodes of ``M^-1 K`` (no reference integration and no
self-comparison tautology).  Every error number below is therefore measured
against a closed-form reference, not against a finer run of the same scheme.
The surrogate is a *numerical* fixture for the clock semantics, not a validated
ice-load model; see the report for that limitation.

Coupling and the causality trade-off
------------------------------------
Within an exchange interval the partner signal must be reconstructed from the
past only:

* ``zoh`` -- hold the exchange-instant value.  Causal; first order in the
  exchange interval ``h`` because the held force is ``O(h)`` stale.
* ``linear_extrapolation`` -- continue the slope measured between the last two
  exchange instants.  Causal, but unstable for oscillatory partners, so it is
  reported as a diagnostic only.
* in-interval **linear interpolation** needs the *next* exchange sample, i.e.
  future data; it is second order but non-causal and is therefore only measured
  (not used) here: :func:`tensorphysics.campaign.clock_rational.assert_causal_hold`
  rejects it.

Declared error bands (dimensionless, ``A`` = signal amplitude, ``omega_max`` =
highest natural frequency of the coupled system)::

    causal ZOH      : max|x - x_exact| / A  <=  BAND_ZOH_COEF    * (omega_max * h)
    linear interp.  : max|x - x_exact| / A  <=  BAND_LINEAR_COEF * (omega_max * h)**2

Both coefficients are *declared constants* in this module and machine-verified
over a scan of exchange intervals; the measured convergence order is reported
alongside, so a band that is merely a curve fit would be visible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

from .clock_rational import (
    CausalityViolation,
    RationalClockError,
    RationalClockSpec,
    RationalCouplingClock,
    assert_causal_hold,
)

#: Declared dimensionless band coefficient for the causal zero-order-hold exchange.
BAND_ZOH_COEF = 1.0
#: Declared dimensionless band coefficient for (non-causal) linear interpolation.
BAND_LINEAR_COEF = 1.0
#: Accepted window for the measured first-order convergence exponent.
ORDER_WINDOW = (0.85, 1.30)
#: Accepted window for the measured second-order (non-causal) exponent.
ORDER_WINDOW_SECOND = (1.85, 2.15)


@dataclass(frozen=True)
class TwoIntegratorModel:
    """Two coupled linear oscillators (hull-section / ice-contact surrogate)."""

    m_master: float = 1.0e6
    m_sub: float = 2.0e5
    k_master: float = 4.0e6
    k_sub: float = 2.0e6
    k_coupling: float = 1.0e6

    def __post_init__(self) -> None:
        for name in ("m_master", "m_sub", "k_master", "k_sub", "k_coupling"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")

    # -- modal analysis -------------------------------------------------------------
    def modes(self) -> tuple[tuple[float, tuple[float, float]], ...]:
        """Mass-normalised eigenmodes ``(omega, (v1, v2))`` sorted by frequency."""
        m1, m2 = self.m_master, self.m_sub
        k1 = self.k_master + self.k_coupling
        k2 = self.k_sub + self.k_coupling
        kc = self.k_coupling
        # det(K - lambda M) = 0
        qa = m1 * m2
        qb = -(k1 * m2 + k2 * m1)
        qc = k1 * k2 - kc * kc
        disc = qb * qb - 4.0 * qa * qc
        if disc <= 0.0:
            raise ValueError("coupled system has no real modes (check stiffnesses)")
        root = math.sqrt(disc)
        lambdas = sorted(((-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa)))
        out: list[tuple[float, tuple[float, float]]] = []
        for lam in lambdas:
            if lam <= 0.0:
                raise ValueError("non-positive eigenvalue: model is not a bounded oscillator")
            omega = math.sqrt(lam)
            v1 = kc
            v2 = k1 - lam * m1
            if abs(v1) < 1e-300 and abs(v2) < 1e-300:
                v1, v2 = 1.0, 0.0
            norm = math.sqrt(v1 * v1 * m1 + v2 * v2 * m2)
            out.append((omega, (v1 / norm, v2 / norm)))
        return tuple(out)

    @property
    def omega_max(self) -> float:
        """Highest natural frequency (non-dimensionalises the exchange error)."""
        return self.modes()[-1][0]

    @property
    def period_min(self) -> float:
        return 2.0 * math.pi / self.omega_max

    def exact(self, t: float, *, x0: Sequence[float], v0: Sequence[float]) -> tuple[float, float]:
        """Closed-form solution of the coupled system at time ``t``."""
        m1, m2 = self.m_master, self.m_sub
        x1 = x2 = 0.0
        for omega, (v1, v2) in self.modes():
            aa = v1 * m1 * x0[0] + v2 * m2 * x0[1]
            bb = v1 * m1 * v0[0] + v2 * m2 * v0[1]
            phase = aa * math.cos(omega * t) + (bb / omega) * math.sin(omega * t)
            x1 += v1 * phase
            x2 += v2 * phase
        return x1, x2

    def energy(self, x: Sequence[float], v: Sequence[float]) -> float:
        """Total mechanical energy of the coupled system."""
        m1, m2 = self.m_master, self.m_sub
        k1, k2, kc = self.k_master, self.k_sub, self.k_coupling
        spring = 0.5 * (k1 * x[0] ** 2 + k2 * x[1] ** 2 + kc * (x[1] - x[0]) ** 2)
        return 0.5 * (m1 * v[0] ** 2 + m2 * v[1] ** 2) + spring


def default_state() -> tuple[tuple[float, float], tuple[float, float]]:
    """Initial condition: 1 mm master displacement, everything else at rest."""
    return (1.0e-3, 0.0), (0.0, 0.0)


def signal_amplitude(
    model: TwoIntegratorModel,
    *,
    x0: Sequence[float],
    v0: Sequence[float],
    duration_s: float,
    samples: int = 400,
) -> float:
    """Amplitude of the exact motion over the horizon (error normaliser)."""
    peak = 0.0
    for k in range(samples + 1):
        x1, x2 = model.exact(duration_s * k / samples, x0=x0, v0=v0)
        peak = max(peak, abs(x1), abs(x2))
    return peak or 1.0


@dataclass
class MultirateResult:
    """Outcome of one coupled multirate run."""

    exchange_interval_s: float
    intervals: int
    master_steps: int
    sub_steps: int
    max_error_master: float
    max_error_sub: float
    amplitude: float
    energy_initial: float
    energy_final: float
    exchange_mode: str

    @property
    def rel_error(self) -> float:
        return max(self.max_error_master, self.max_error_sub) / self.amplitude

    @property
    def rel_energy_drift(self) -> float:
        return abs(self.energy_final - self.energy_initial) / max(abs(self.energy_initial), 1e-300)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_interval_s": self.exchange_interval_s,
            "intervals": self.intervals,
            "master_steps": self.master_steps,
            "sub_steps": self.sub_steps,
            "rel_error": self.rel_error,
            "max_error_master": self.max_error_master,
            "max_error_sub": self.max_error_sub,
            "amplitude": self.amplitude,
            "rel_energy_drift": self.rel_energy_drift,
            "exchange_mode": self.exchange_mode,
        }


def simulate_multirate(
    model: TwoIntegratorModel,
    spec: RationalClockSpec,
    *,
    intervals: int,
    x0: Sequence[float] | None = None,
    v0: Sequence[float] | None = None,
    exchange_mode: str = "zoh",
) -> MultirateResult:
    """Run the coupled system for ``intervals`` exchange intervals.

    Integration proceeds on the spec's atom lattice, so a fractional step ratio
    (e.g. ``3/2``) is handled exactly: master and sub steps fire at atom
    boundaries and the exchange only at common periods.  The partner value used
    inside an interval is reconstructed from past exchange samples only.
    """
    if intervals < 1:
        raise RationalClockError("intervals must be >= 1")
    if exchange_mode not in ("zoh", "linear_extrapolation"):
        raise RationalClockError(
            f"exchange_mode must be 'zoh' or 'linear_extrapolation' (causal), got {exchange_mode!r}"
        )
    if x0 is None or v0 is None:
        x0, v0 = default_state()
    x0 = (float(x0[0]), float(x0[1]))
    v0 = (float(v0[0]), float(v0[1]))

    dt_master = float(spec.master_dt)
    dt_sub = float(spec.sub_dt)
    h_exact = spec.exchange_interval_s
    h = float(h_exact)
    atoms = spec.atoms_per_exchange
    a, b = spec.master_atoms, spec.sub_atoms
    atom_s = float(spec.atom_s)

    x1, x2 = x0
    v1, v2 = v0
    m1, m2 = model.m_master, model.m_sub
    k1 = model.k_master + model.k_coupling
    k2 = model.k_sub + model.k_coupling
    kc = model.k_coupling

    partner1 = x2  # sub displacement held by the master solver
    partner2 = x1  # master displacement held by the sub solver
    prev1: float | None = None
    prev2: float | None = None

    err_master = 0.0
    err_sub = 0.0
    energy0 = model.energy((x1, x2), (v1, v2))
    master_steps = 0
    sub_steps = 0

    for interval in range(intervals):
        # --- exchange at the interval start: consume the *past* partner state --
        slope1 = 0.0 if prev1 is None else (partner1 - prev1) / h
        slope2 = 0.0 if prev2 is None else (partner2 - prev2) / h
        prev1, prev2 = partner1, partner2
        partner1, partner2 = x2, x1

        for atom in range(1, atoms + 1):
            t_local = atom * atom_s
            if exchange_mode == "zoh":
                p1, p2 = partner1, partner2
            else:
                p1, p2 = partner1 + slope1 * t_local, partner2 + slope2 * t_local

            if atom % a == 0:  # master step
                f1 = -k1 * x1 + kc * p1
                v1_half = v1 + 0.5 * dt_master * f1 / m1
                x1_new = x1 + dt_master * v1_half
                f1_new = -k1 * x1_new + kc * p1
                v1 = v1_half + 0.5 * dt_master * f1_new / m1
                x1 = x1_new
                master_steps += 1
            if atom % b == 0:  # sub step
                f2 = -k2 * x2 + kc * p2
                v2_half = v2 + 0.5 * dt_sub * f2 / m2
                x2_new = x2 + dt_sub * v2_half
                f2_new = -k2 * x2_new + kc * p2
                v2 = v2_half + 0.5 * dt_sub * f2_new / m2
                x2 = x2_new
                sub_steps += 1

            # --- error measured at every atom boundary, including interval interior
            t_abs = float(interval * h_exact) + t_local
            ex1, ex2 = model.exact(t_abs, x0=x0, v0=v0)
            err_master = max(err_master, abs(x1 - ex1))
            err_sub = max(err_sub, abs(x2 - ex2))

    duration = float(intervals * h_exact)
    amplitude = signal_amplitude(model, x0=x0, v0=v0, duration_s=duration)
    return MultirateResult(
        exchange_interval_s=h,
        intervals=intervals,
        master_steps=master_steps,
        sub_steps=sub_steps,
        max_error_master=err_master,
        max_error_sub=err_sub,
        amplitude=amplitude,
        energy_initial=energy0,
        energy_final=model.energy((x1, x2), (v1, v2)),
        exchange_mode=exchange_mode,
    )


# ---------------------------------------------------------------------------
# in-interval reconstruction error (the declared bands)
# ---------------------------------------------------------------------------
def interpolation_band(
    model: TwoIntegratorModel,
    *,
    exchange_interval_s: float,
    intervals: int,
    x0: Sequence[float] | None = None,
    v0: Sequence[float] | None = None,
    interior_samples: int = 16,
) -> dict[str, Any]:
    """Measure the in-interval reconstruction error of the exchanged signal.

    The partner signal is the exact master displacement sampled at the exchange
    instants.  Two reconstructions are compared inside every interval:

    * ZOH (causal, uses ``x(t_k)`` only);
    * linear interpolation (uses ``x(t_k)`` **and** ``x(t_{k+1})``, i.e. future
      data at any interior instant -- measured only, rejected by the causal gate).
    """
    if x0 is None or v0 is None:
        x0, v0 = default_state()
    x0 = (float(x0[0]), float(x0[1]))
    v0 = (float(v0[0]), float(v0[1]))
    h = float(exchange_interval_s)
    knots = [k * h for k in range(intervals + 1)]
    knot_values = [model.exact(t, x0=x0, v0=v0)[0] for t in knots]
    amplitude = signal_amplitude(model, x0=x0, v0=v0, duration_s=knots[-1])
    omega = model.omega_max

    err_zoh = 0.0
    err_lin = 0.0
    for k in range(intervals):
        left, right = knot_values[k], knot_values[k + 1]
        for j in range(1, interior_samples + 1):
            frac = j / (interior_samples + 1.0)
            t = knots[k] + frac * h
            exact = model.exact(t, x0=x0, v0=v0)[0]
            err_zoh = max(err_zoh, abs(exact - left))
            err_lin = max(err_lin, abs(exact - (left + frac * (right - left))))

    scaled = omega * h
    ratio_zoh = (err_zoh / amplitude) / scaled
    ratio_lin = (err_lin / amplitude) / scaled ** 2

    # causal gate: an interior consumer cannot use the right-hand sample
    h_frac = exchange_interval_s if isinstance(exchange_interval_s, Fraction) else Fraction(repr(float(h)))
    causal_error: str | None = None
    try:
        assert_causal_hold(h_frac / 2, h_frac, label="interior linear interp")
    except CausalityViolation as exc:  # expected: this is the negative case
        causal_error = str(exc)

    return {
        "exchange_interval_s": h,
        "omega_max": omega,
        "omega_h": scaled,
        "amplitude": amplitude,
        "zoh_error": err_zoh,
        "linear_error": err_lin,
        "zoh_ratio": ratio_zoh,
        "linear_ratio": ratio_lin,
        "band_zoh_coef": BAND_ZOH_COEF,
        "band_linear_coef": BAND_LINEAR_COEF,
        "zoh_within_band": ratio_zoh <= BAND_ZOH_COEF,
        "linear_within_band": ratio_lin <= BAND_LINEAR_COEF,
        "linear_is_causal": causal_error is None,
        "causality_violation": causal_error,
    }


def measure_order(hs: Sequence[float], errors: Sequence[float]) -> dict[str, Any]:
    """Least-squares slope of ``log(err)`` against ``log(h)`` plus pair orders."""
    if len(hs) != len(errors):
        raise RationalClockError("hs and errors must have the same length")
    if len(hs) < 2:
        raise RationalClockError("need at least two refinement levels")
    xs = [math.log(h) for h in hs]
    ys = [math.log(max(e, 1e-300)) for e in errors]
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom if denom else float("nan")
    pairs = [
        math.log(errors[i] / errors[i + 1]) / math.log(hs[i] / hs[i + 1])
        for i in range(len(hs) - 1)
        if errors[i] > 0 and errors[i + 1] > 0
    ]
    return {"order": slope, "pair_orders": pairs}


# ---------------------------------------------------------------------------
# full cross-check (used by tests and the artifact generator)
# ---------------------------------------------------------------------------
def _scale(dt: Any, factor: Fraction) -> Fraction:
    """Exact scaling of a step size (used by the refinement family)."""
    from .clock_rational import as_fraction

    return as_fraction(dt, "dt") * factor


def refinement_scan(
    model: TwoIntegratorModel | None = None,
    *,
    master_dt_s: Any = 0.004,
    sub_dt_s: Any = 0.003,
    levels: int = 4,
    horizon_s: Any = 3.072,
    exchange_mode: str = "zoh",
) -> dict[str, Any]:
    """Refine *both* clocks and the exchange interval together (order measurement).

    The clock pair is halved at every level, so the exchange interval (their LCM)
    also halves.  Refining the clocks with ``h`` isolates the **exchange** error
    from the time-integration error: a constant ``h`` with refined ``dt`` shows an
    error floor, which would otherwise contaminate the measured order.
    """
    if levels < 2:
        raise RationalClockError("levels must be >= 2")
    model = model or TwoIntegratorModel()
    x0, v0 = default_state()
    horizon = horizon_s if isinstance(horizon_s, Fraction) else Fraction(repr(float(horizon_s)))
    rows: list[dict[str, Any]] = []
    for level in range(levels):
        factor = Fraction(1, 2 ** level)
        spec = RationalClockSpec(
            master_name="hull_section",
            sub_name="ice_contact",
            master_dt_s=_scale(master_dt_s, factor),
            sub_dt_s=_scale(sub_dt_s, factor),
            exchange_periods=1,
        )
        intervals_frac = horizon / spec.exchange_interval_s
        if intervals_frac.denominator != 1:
            raise RationalClockError(
                f"horizon {horizon} is not a multiple of the exchange interval "
                f"{spec.exchange_interval_s} at level {level}"
            )
        result = simulate_multirate(
            model, spec, intervals=int(intervals_frac), x0=x0, v0=v0, exchange_mode=exchange_mode
        )
        omega_h = model.omega_max * result.exchange_interval_s
        row = result.to_dict()
        row.update(
            {
                "level": level,
                "master_dt_s": float(spec.master_dt),
                "sub_dt_s": float(spec.sub_dt),
                "ratio_exact": f"{spec.ratio.numerator}/{spec.ratio.denominator}",
                "omega_h": omega_h,
                "band_ratio": result.rel_error / (BAND_ZOH_COEF * omega_h)
                if exchange_mode == "zoh"
                else float("nan"),
            }
        )
        rows.append(row)

    hs = [row["exchange_interval_s"] for row in rows]
    fit = measure_order(hs, [row["rel_error"] for row in rows])
    order = fit["order"]
    return {
        "model": _model_dict(model),
        "exchange_mode": exchange_mode,
        "horizon_s": float(horizon),
        "rows": rows,
        "order": order,
        "pair_orders": fit["pair_orders"],
        "order_window": list(ORDER_WINDOW),
        "order_ok": ORDER_WINDOW[0] <= order <= ORDER_WINDOW[1],
        "worst_band_ratio": max(row["band_ratio"] for row in rows),
        "all_within_band": all(
            row["rel_error"] <= BAND_ZOH_COEF * row["omega_h"] for row in rows
        )
        if exchange_mode == "zoh"
        else None,
    }


def band_scan(
    model: TwoIntegratorModel | None = None,
    *,
    master_dt_s: Any = 0.004,
    sub_dt_s: Any = 0.003,
    periods: Sequence[int] = (1, 2, 4, 8, 16),
    base_intervals: int = 256,
    exchange_mode: str = "zoh",
) -> dict[str, Any]:
    """Scan the exchange interval over one decade, keeping the horizon fixed.

    Used for the *band* claim: the exchange interval is scaled by an integer
    number of the clock's own LCM periods, so the scan never leaves the exact
    lattice, and every level integrates the same physical time span.
    """
    model = model or TwoIntegratorModel()
    x0, v0 = default_state()
    rows: list[dict[str, Any]] = []
    for k in periods:
        if base_intervals % k != 0:
            raise RationalClockError(
                f"base_intervals={base_intervals} must be a multiple of every period multiplier"
            )
        spec = RationalClockSpec(
            master_name="hull_section",
            sub_name="ice_contact",
            master_dt_s=master_dt_s,
            sub_dt_s=sub_dt_s,
            exchange_periods=k,
        )
        result = simulate_multirate(
            model, spec, intervals=base_intervals // k, x0=x0, v0=v0, exchange_mode=exchange_mode
        )
        omega_h = model.omega_max * result.exchange_interval_s
        row = result.to_dict()
        row.update(
            {
                "exchange_periods": k,
                "omega_h": omega_h,
                "band_ratio": result.rel_error / (BAND_ZOH_COEF * omega_h),
                "within_band": result.rel_error <= BAND_ZOH_COEF * omega_h,
            }
        )
        rows.append(row)
    hs = [row["exchange_interval_s"] for row in rows]
    fit = measure_order(hs, [row["rel_error"] for row in rows])
    return {
        "model": _model_dict(model),
        "exchange_mode": exchange_mode,
        "horizon_s": rows[0]["intervals"] * hs[0] if rows else 0.0,
        "rows": rows,
        "order": fit["order"],
        "pair_orders": fit["pair_orders"],
        "band_zoh_coef": BAND_ZOH_COEF,
        "all_within_band": all(row["within_band"] for row in rows),
        "worst_band_ratio": max(row["band_ratio"] for row in rows) if rows else float("inf"),
    }


def _model_dict(model: TwoIntegratorModel) -> dict[str, Any]:
    return {
        "m_master": model.m_master,
        "m_sub": model.m_sub,
        "k_master": model.k_master,
        "k_sub": model.k_sub,
        "k_coupling": model.k_coupling,
        "omega_max": model.omega_max,
        "period_min": model.period_min,
        "modes": [{"omega": w, "mode": list(v)} for w, v in model.modes()],
    }


def interpolation_scan(
    model: TwoIntegratorModel | None = None,
    *,
    exchange_intervals_s: Sequence[float] = (0.012, 0.024, 0.048, 0.096),
    intervals: int = 32,
    interior_samples: int = 16,
) -> dict[str, Any]:
    """Band check for in-interval reconstruction (ZOH causal vs linear non-causal)."""
    model = model or TwoIntegratorModel()
    rows = [
        interpolation_band(
            model,
            exchange_interval_s=h,
            intervals=intervals,
            interior_samples=interior_samples,
        )
        for h in exchange_intervals_s
    ]
    fit_zoh = measure_order(exchange_intervals_s, [row["zoh_error"] for row in rows])
    fit_lin = measure_order(exchange_intervals_s, [row["linear_error"] for row in rows])
    return {
        "model": _model_dict(model),
        "rows": rows,
        "order_zoh": fit_zoh["order"],
        "order_linear": fit_lin["order"],
        "pair_orders_zoh": fit_zoh["pair_orders"],
        "pair_orders_linear": fit_lin["pair_orders"],
        "zoh_order_window": list(ORDER_WINDOW),
        "linear_order_window": list(ORDER_WINDOW_SECOND),
        "zoh_order_ok": ORDER_WINDOW[0] <= fit_zoh["order"] <= ORDER_WINDOW[1],
        "linear_order_ok": ORDER_WINDOW_SECOND[0] <= fit_lin["order"] <= ORDER_WINDOW_SECOND[1],
        "all_zoh_within_band": all(row["zoh_within_band"] for row in rows),
        "all_linear_within_band": all(row["linear_within_band"] for row in rows),
        "linear_rejected_as_non_causal": all(not row["linear_is_causal"] for row in rows),
        "worst_zoh_ratio": max(row["zoh_ratio"] for row in rows),
        "worst_linear_ratio": max(row["linear_ratio"] for row in rows),
    }
