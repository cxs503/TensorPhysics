"""Quasi-zero-stiffness (QZS) isolation layer for the ice-vibration transfer chain
(campaign node p227w2-N08, maturity round of p228w1-N08).

Why this module exists
----------------------
The verified transfer chain (``chain.py``) ends with a disclosure it cannot
escape: a single-stage passive resilient mount has ``|H(0)| = 1`` and
``|H| > 1`` below ``sqrt(2) f_n``, while the ice-induced hull-girder energy
sits at 0.3-3 Hz (2-node wet mode ~0.4-1 Hz).  With a conventionally stiff
mount (``f_n = 4 Hz``, static deflection 15.5 mm) the RMS ratio
``RMS(base)/RMS(hull)`` is 1.21 - the mount *amplifies* the very band that
carries the ice energy.  Lowering ``f_n`` to 0.5 Hz would fix it, but a linear
spring at 0.5 Hz needs a static deflection of 994 mm, which no shipboard mount
can offer.

The only passive route out of that corner is to decouple the two stiffness
roles a mount plays:

* the **positive** spring ``k_v`` carries the equipment weight
  (``delta_st = g / omega_v^2`` - this is what sets the 15.5 mm of a 4 Hz
  mount), while
* a **negative-stiffness mechanism** ``k_n = -alpha k_v`` (pre-buckled struts,
  oblique springs, magnetic repulsion) cancels most of it, so the *dynamic*
  stiffness

      k_dyn = k_v (1 - alpha),      f_n,dyn = f_rigid sqrt(1 - alpha)

  can be tuned down to the hull band while the equipment still sits on a
  15.5 mm static deflection.  This is the quasi-zero-stiffness (QZS) /
  high-static-low-dynamic-stiffness mount of Carrella et al. (2007) and
  Ibrahim (2008).

What the module provides
------------------------
``QZSMount``
    Linearised QZS mount: ``f_n`` from the residual stiffness, unchanged
    static deflection, transmissibility of the same form as the linear mount
    but with ``f_n = f_n,dyn``, plus the mechanism-level bookkeeping that makes
    the design auditable (negative-stiffness fraction ``alpha``, residual
    stiffness fraction ``1-alpha``, stroke, sensitivity
    ``d ln f_n / d alpha = -1/(2 (1-alpha))``).
``band_attenuation``
    Band RMS of base vs hull acceleration - the criterion the verified case
    could not meet.
``design_map`` / ``select_design``
    Feasibility map over ``(f_n,dyn, zeta)`` with the static-deflection
    constraint and the residual-stiffness budget, plus the selection rule
    (lowest band ratio among feasible rows, ties to the larger residual
    stiffness).
``tolerance_window``
    How much error in ``alpha`` the design survives: the band ratio is
    recomputed for perturbed residual stiffness, which is the real engineering
    question because elastomer ageing and temperature move the cancellation.
``simulate_base_excitation`` / ``nonlinear_response_amplitude``
    Independent RK4 time-domain integration of the base-excited SDOF and of the
    QZS Duffing restoring force ``F(u) = k_dyn u + beta u^3``, used to check the
    frequency-domain transmissibility and to bound where the linearisation
    stops being valid.

Honest limits (carried into the payload as disclosures)
-------------------------------------------------------
* ``|H(f)| < 1`` at *every* frequency is impossible for any passive mount
  (``H -> 1`` as ``f -> 0``); the criterion is posed pointwise above
  ``sqrt(2) f_n`` and in the energy sense (band RMS) over the ice band.
* A strongly softened mount has a narrow-band resonance just above
  ``f_n,dyn``; the amplification there is real and is reported.
* The QZS mechanism must hold ``k_n = -alpha k_v`` over the *whole operating
  stroke*.  Wave-induced and whipping relative motions are one to two orders
  larger than the ice-induced motion of this chain, so the stroke check is
  performed with the declared wave/whipping envelope
  :data:`DEFAULT_WAVE_WHIPPING_LOAD` and the result is reported, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "GRAVITY",
    "DEFAULT_WAVE_WHIPPING_LOAD",
    "QZSMount",
    "band_attenuation",
    "design_map",
    "select_design",
    "tolerance_window",
    "relative_displacement_psd",
    "service_stroke",
    "simulate_base_excitation",
    "nonlinear_response_amplitude",
]

GRAVITY = 9.81

#: Declared wave / whipping envelope used only for the QZS stroke check.
#: Two components with amplitudes representative of a 1A-class ship in a design
#: sea state (values are *declared inputs*, not derived here): a quasi-static
#: rigid-body acceleration at the foundation station and a 1 Hz whipping
#: component of the hull girder.
DEFAULT_WAVE_WHIPPING_LOAD: tuple[dict[str, float], ...] = (
    {"label": "wave", "freq_hz": 0.10, "accel_rms_m_s2": 0.60},
    {"label": "whipping", "freq_hz": 1.00, "accel_rms_m_s2": 0.30},
)


def _as_freqs(freqs: Iterable[float]) -> np.ndarray:
    f = np.asarray(freqs, dtype=float)
    if f.ndim != 1 or f.size < 1:
        raise ValueError("frequencies must be a 1-D array")
    if np.any(f < 0.0):
        raise ValueError("frequencies must be non-negative")
    return f


@dataclass
class QZSMount:
    """Linearised quasi-zero-stiffness mount (negative stiffness + load spring).

    Parameters
    ----------
    equipment_mass_kg : float
        Supported mass (equipment package).
    f_rigid_hz : float
        Natural frequency the *positive* spring alone would give, i.e. the
        frequency of the conventional mount with the same static deflection
        (``delta_st = g / (2 pi f_rigid)^2``).
    residual_stiffness_fraction : float
        ``1 - alpha``: fraction of the load-spring stiffness surviving the
        negative-stiffness mechanism.  ``1`` = conventional mount, small values
        = strong cancellation.  Must satisfy ``0 < 1-alpha <= 1``.
    zeta : float
        Damping ratio referred to the *dynamic* (softened) frequency.
    stroke_m : float
        Allowable relative displacement across the mount.
    ns_range_m : float
        Displacement range over which the negative-stiffness mechanism is
        linear; used for the cubic (Duffing) correction
        ``beta = alpha k_v / ns_range^2``.
    """

    equipment_mass_kg: float
    f_rigid_hz: float
    residual_stiffness_fraction: float
    zeta: float = 0.15
    stroke_m: float = 0.05
    ns_range_m: float = 0.02

    def __post_init__(self) -> None:
        if self.equipment_mass_kg <= 0.0:
            raise ValueError("equipment mass must be positive")
        if not np.isfinite(self.f_rigid_hz) or self.f_rigid_hz <= 0.0:
            raise ValueError("rigid natural frequency must be positive and finite")
        s = float(self.residual_stiffness_fraction)
        if not 0.0 < s <= 1.0:
            raise ValueError("residual stiffness fraction must satisfy 0 < s <= 1")
        if not 0.0 < self.zeta < 1.0:
            raise ValueError("damping ratio must satisfy 0 < zeta < 1")
        if self.stroke_m <= 0.0 or self.ns_range_m <= 0.0:
            raise ValueError("stroke and NS range must be positive")

    # ------------------------------------------------------------ mechanism
    @property
    def residual_stiffness(self) -> float:
        return float(self.residual_stiffness_fraction)

    @property
    def negative_stiffness_fraction(self) -> float:
        """``alpha``: share of the load-spring stiffness cancelled by the mechanism."""
        return float(1.0 - self.residual_stiffness_fraction)

    @property
    def positive_spring_stiffness_n_m(self) -> float:
        return float(self.equipment_mass_kg * (2.0 * np.pi * self.f_rigid_hz) ** 2)

    @property
    def dynamic_stiffness_n_m(self) -> float:
        return float(self.positive_spring_stiffness_n_m * self.residual_stiffness_fraction)

    @property
    def f_n_hz(self) -> float:
        """Dynamic natural frequency ``f_rigid sqrt(1-alpha)``."""
        return float(self.f_rigid_hz * np.sqrt(self.residual_stiffness_fraction))

    @property
    def static_deflection_m(self) -> float:
        """Static deflection: set by the load spring only, *not* by ``alpha``."""
        return float(GRAVITY / (2.0 * np.pi * self.f_rigid_hz) ** 2)

    @property
    def equivalent_linear_deflection_m(self) -> float:
        """What a plain linear spring at ``f_n`` would need: ``g/(2 pi f_n)^2``."""
        return float(GRAVITY / (2.0 * np.pi * self.f_n_hz) ** 2)

    @property
    def isolation_frequency_hz(self) -> float:
        return float(np.sqrt(2.0) * self.f_n_hz)

    @property
    def peak_amplification(self) -> float:
        a = 4.0 * self.zeta**2
        u = (np.sqrt(1.0 + 2.0 * a) - 1.0) / a
        return float(np.sqrt((1.0 + a * u) / ((1.0 - u) ** 2 + a * u)))

    @property
    def damping_coefficient_n_s_m(self) -> float:
        omega = 2.0 * np.pi * self.f_n_hz
        return float(2.0 * self.zeta * self.equipment_mass_kg * omega)

    # --------------------------------------------------------- transmissibility
    def H(self, freqs: Iterable[float]) -> np.ndarray:
        """Complex base-acceleration transmissibility ``a_base / a_support``."""
        f = _as_freqs(freqs)
        r = f / self.f_n_hz
        return (1.0 + 2j * self.zeta * r) / ((1.0 - r**2) + 2j * self.zeta * r)

    def magnitude(self, freqs: Iterable[float]) -> np.ndarray:
        return np.abs(self.H(freqs))

    def relative_displacement_magnitude(self, freqs: Iterable[float]) -> np.ndarray:
        """``|x_base - x_support| / X_support`` (mount stroke per unit support motion)."""
        f = _as_freqs(freqs)
        r = f / self.f_n_hz
        return r**2 / np.sqrt((1.0 - r**2) ** 2 + (2.0 * self.zeta * r) ** 2)

    def bode(self, freqs: Iterable[float]) -> dict[str, Any]:
        f = _as_freqs(freqs)
        mag = self.magnitude(f)
        return {
            "freqs_hz": f,
            "magnitude": mag,
            "magnitude_db": 20.0 * np.log10(np.maximum(mag, 1e-300)),
            "phase_deg": np.degrees(np.angle(self.H(f))),
            "f_n_hz": self.f_n_hz,
            "isolation_frequency_hz": self.isolation_frequency_hz,
        }

    # -------------------------------------------------------------- nonlinear
    @property
    def cubic_coefficient_n_m3(self) -> float:
        """``beta`` of the hardening QZS restoring force ``F(u) = k_dyn u + beta u^3``.

        The negative-stiffness mechanism is linear only inside ``ns_range``; the
        geometric stiffening beyond it is lumped into a cubic term sized so that
        the correction equals the cancelled stiffness at ``u = ns_range``.
        """
        return float(
            self.negative_stiffness_fraction * self.positive_spring_stiffness_n_m / self.ns_range_m**2
        )

    def linearity_amplitude_m(self, relative_error: float = 0.01) -> float:
        """Relative displacement below which the cubic term stays under ``relative_error``."""
        if relative_error <= 0.0:
            raise ValueError("relative error must be positive")
        return float(
            np.sqrt(relative_error * self.dynamic_stiffness_n_m / (3.0 * self.cubic_coefficient_n_m3))
        )

    def backbone_peak_hz(self, amplitude_m: float) -> float:
        """Hardening backbone: resonance frequency at relative amplitude ``A``."""
        k = self.dynamic_stiffness_n_m
        beta = self.cubic_coefficient_n_m3
        return float(self.f_n_hz * np.sqrt(1.0 + 3.0 * beta * amplitude_m**2 / (8.0 * k)))

    # ------------------------------------------------------------- sensitivity
    def frequency_sensitivity(self, stiffness_error: float = 0.01) -> dict[str, float]:
        """Frequency shift for an absolute error ``stiffness_error`` in ``alpha``.

        ``f_n = f_rigid sqrt(1-alpha)`` implies
        ``d ln f_n / d alpha = -1/(2 (1-alpha))``, so an error ``delta alpha``
        in the cancellation produces a relative frequency error
        ``-delta alpha / (2 (1-alpha))``.
        """
        s = self.residual_stiffness_fraction
        rel = -float(stiffness_error) / (2.0 * s)
        return {
            "stiffness_error": float(stiffness_error),
            "relative_frequency_error": rel,
            "f_n_perturbed_hz": float(self.f_n_hz * (1.0 + rel)),
        }

    # ------------------------------------------------------------------- views
    def as_dict(self) -> dict[str, Any]:
        return {
            "model": "quasi-zero-stiffness (QZS) mount, linearised about equilibrium",
            "equipment_mass_kg": float(self.equipment_mass_kg),
            "f_rigid_hz": float(self.f_rigid_hz),
            "f_n_hz": self.f_n_hz,
            "negative_stiffness_fraction_alpha": self.negative_stiffness_fraction,
            "residual_stiffness_fraction": self.residual_stiffness_fraction,
            "positive_spring_stiffness_N_per_m": self.positive_spring_stiffness_n_m,
            "dynamic_stiffness_N_per_m": self.dynamic_stiffness_n_m,
            "zeta": float(self.zeta),
            "static_deflection_m": self.static_deflection_m,
            "equivalent_linear_deflection_m": self.equivalent_linear_deflection_m,
            "isolation_frequency_hz": self.isolation_frequency_hz,
            "peak_amplification": self.peak_amplification,
            "cubic_coefficient_N_per_m3": self.cubic_coefficient_n_m3,
            "linearity_amplitude_m": self.linearity_amplitude_m(),
            "stroke_m": float(self.stroke_m),
            "ns_range_m": float(self.ns_range_m),
            "frequency_sensitivity_per_1pct_alpha": self.frequency_sensitivity(0.01),
        }


# --------------------------------------------------------------------- helpers
def relative_displacement_psd(
    freqs: Iterable[float], s_support_accel: Iterable[float], mount: QZSMount
) -> np.ndarray:
    """Relative-displacement PSD [m^2/Hz] from a support acceleration PSD."""
    f = _as_freqs(freqs)
    s = np.asarray(s_support_accel, dtype=float)
    if f.shape != s.shape:
        raise ValueError("freqs and PSD must have the same shape")
    omega = 2.0 * np.pi * np.maximum(f, 1e-12)
    t_rel = mount.relative_displacement_magnitude(f)
    return t_rel**2 * s / omega**4


def band_attenuation(
    freqs: Iterable[float],
    s_hull_accel: Iterable[float],
    mount: QZSMount,
    band_hz: tuple[float, float],
) -> dict[str, Any]:
    """Band RMS of ``base`` vs ``hull`` acceleration (the energy-sense ``|H| < 1``)."""
    f = _as_freqs(freqs)
    s = np.asarray(s_hull_accel, dtype=float)
    if f.shape != s.shape:
        raise ValueError("freqs and PSD must have the same shape")
    lo, hi = float(band_hz[0]), float(band_hz[1])
    m = (f >= lo) & (f <= hi)
    if np.count_nonzero(m) < 2:
        raise ValueError("band does not contain enough frequency samples")
    h2 = mount.magnitude(f) ** 2
    rms_hull = float(np.sqrt(max(np.trapezoid(s[m], f[m]), 0.0)))
    rms_base = float(np.sqrt(max(np.trapezoid((s * h2)[m], f[m]), 0.0)))
    mag = mount.magnitude(f)
    iso = mount.isolation_frequency_hz
    att = f > iso * (1.0 + 1e-12)
    return {
        "band_hz": [lo, hi],
        "rms_hull_m_s2": rms_hull,
        "rms_base_m_s2": rms_base,
        "rms_ratio_base_over_hull": float(rms_base / rms_hull) if rms_hull > 0.0 else float("nan"),
        "attenuated": bool(rms_base < rms_hull),
        "max_magnitude_in_band": float(np.max(mag[m])),
        "max_magnitude_above_isolation": float(np.max(mag[att])) if np.any(att) else float("nan"),
        "band_all_below_unity_above_isolation": bool(np.all(mag[m & att] < 1.0)) if np.any(m & att) else None,
        "peak_frequency_hz": float(f[m][int(np.argmax(mag[m]))]),
    }


def service_stroke(
    mount: QZSMount, service_loads: Sequence[dict[str, float]] = DEFAULT_WAVE_WHIPPING_LOAD
) -> dict[str, Any]:
    """Relative displacement across the mount under the declared service loads.

    Each declared load is treated as a narrowband component with RMS
    acceleration ``a`` at ``f``; the relative displacement RMS is
    ``|x_rel/X_support| * a / omega^2``.  The ``2 x RMS`` peak convention is
    used for the stroke comparison (a Gaussian peak factor of 2 is the usual
    mount-selection margin and is stated in the payload).
    """
    rows = []
    for load in service_loads:
        fl = float(load["freq_hz"])
        acc = float(load["accel_rms_m_s2"])
        omega = 2.0 * np.pi * fl
        rel = float(mount.relative_displacement_magnitude(np.array([fl]))[0])
        rows.append(
            {
                "label": str(load.get("label", f"{fl:g} Hz")),
                "freq_hz": fl,
                "accel_rms_m_s2": acc,
                "relative_displacement_rms_m": rel * acc / omega**2,
            }
        )
    worst = max(rows, key=lambda r: r["relative_displacement_rms_m"]) if rows else None
    worst_rms = 0.0 if worst is None else float(worst["relative_displacement_rms_m"])
    peak = 2.0 * worst_rms
    # Bump-stop demand: the QZS spring alone restores k_dyn * stroke at the stop,
    # the remainder of the inertia load has to be carried by the stop.
    inertia = 0.0 if worst is None else 2.0 * mount.equipment_mass_kg * float(worst["accel_rms_m_s2"])
    stop_force = max(inertia - mount.dynamic_stiffness_n_m * float(mount.stroke_m), 0.0)
    return {
        "rows": rows,
        "worst_case": None if worst is None else worst["label"],
        "worst_relative_displacement_rms_m": worst_rms,
        "peak_relative_displacement_m": peak,
        "peak_factor": 2.0,
        "stop_engaged": bool(peak > float(mount.stroke_m)),
        "bump_stop_force_n": float(stop_force),
        "bump_stop_force_fraction_of_weight": float(
            stop_force / (mount.equipment_mass_kg * GRAVITY)
        ),
        "bump_stop_stiffness_n_m": float(stop_force / (0.25 * float(mount.stroke_m))) if stop_force > 0.0 else 0.0,
    }


def design_map(
    freqs: Iterable[float],
    s_hull_accel: Iterable[float],
    band_hz: tuple[float, float],
    f_rigid_hz: float,
    equipment_mass_kg: float,
    f_n_grid: Sequence[float],
    zeta_grid: Sequence[float],
    static_deflection_limit_m: float,
    residual_stiffness_budget: float,
    ns_range_m: float = 0.02,
    stroke_m: float = 0.05,
    zeta_min: float = 0.15,
    service_loads: Sequence[dict[str, float]] = DEFAULT_WAVE_WHIPPING_LOAD,
    stop_force_limit_fraction: float = 0.25,
) -> list[dict[str, Any]]:
    """Feasibility map of QZS designs over ``(f_n,dyn, zeta)``.

    Each row carries the band RMS ratio, the residual stiffness fraction the
    mechanism would have to realise, the frequency error for a 1 % ``alpha``
    error and two independent feasibility flags:

    ``static_ok``
        QZS static deflection (``g/(2 pi f_rigid)^2``) below the limit;
    ``stiffness_ok``
        residual stiffness fraction at or above the declared budget, i.e. the
        negative-stiffness mechanism does not have to cancel more stiffness
        than the design allows;
    ``zeta_ok``
        damping ratio at or above the declared floor (a QZS mechanism with less
        than ~15 % damping peaks at more than 3.3, which no equipment mount
        accepts);
    ``stroke_ok``
        the *declared* wave / whipping relative motion either fits inside the
        stroke or needs a bump stop whose force stays below the declared
        fraction of the equipment weight.  This is the constraint that makes a
        very soft mount impractical without a stop: at ``f_n = 0.4 Hz`` a
        0.1 Hz, 0.6 m/s^2 wave component alone asks for 101 mm of relative
        motion;
    ``ice_linear_ok``
        the ice-band relative motion - the vibration the mount is meant to
        isolate, of order micrometres - stays inside the linear range of the
        negative-stiffness mechanism, so the linear QZS model is the relevant
        one for the chain even though the wave motion engages the stop.
    """
    f = _as_freqs(freqs)
    s = np.asarray(s_hull_accel, dtype=float)
    rows: list[dict[str, Any]] = []
    for f_n in f_n_grid:
        frac = (float(f_n) / float(f_rigid_hz)) ** 2
        if not 0.0 < frac <= 1.0:
            continue
        for zeta in zeta_grid:
            mount = QZSMount(
                equipment_mass_kg=equipment_mass_kg,
                f_rigid_hz=f_rigid_hz,
                residual_stiffness_fraction=frac,
                zeta=float(zeta),
                stroke_m=stroke_m,
                ns_range_m=ns_range_m,
            )
            att = band_attenuation(f, s, mount, band_hz)
            sens = mount.frequency_sensitivity(0.01)
            stroke_row = service_stroke(mount, service_loads)
            rel_psd = relative_displacement_psd(f, s, mount)
            m_band = (f >= float(band_hz[0])) & (f <= float(band_hz[1]))
            ice_rel_rms = (
                float(np.sqrt(max(np.trapezoid(rel_psd[m_band], f[m_band]), 0.0)))
                if np.count_nonzero(m_band) > 1
                else 0.0
            )
            rows.append(
                {
                    "f_n_hz": float(mount.f_n_hz),
                    "zeta": float(zeta),
                    "residual_stiffness_fraction": float(frac),
                    "negative_stiffness_fraction_alpha": float(1.0 - frac),
                    "rms_ratio_base_over_hull": att["rms_ratio_base_over_hull"],
                    "max_magnitude_in_band": att["max_magnitude_in_band"],
                    "peak_amplification": mount.peak_amplification,
                    "static_deflection_m": mount.static_deflection_m,
                    "equivalent_linear_deflection_m": mount.equivalent_linear_deflection_m,
                    "frequency_error_per_1pct_alpha": sens["relative_frequency_error"],
                    "linearity_amplitude_m": mount.linearity_amplitude_m(),
                    "service_relative_displacement_rms_m": stroke_row["worst_relative_displacement_rms_m"],
                    "bump_stop_force_fraction_of_weight": stroke_row["bump_stop_force_fraction_of_weight"],
                    "ice_band_relative_displacement_rms_m": ice_rel_rms,
                    "static_ok": bool(mount.static_deflection_m <= float(static_deflection_limit_m)),
                    "stiffness_ok": bool(frac >= float(residual_stiffness_budget)),
                    "zeta_ok": bool(float(zeta) >= float(zeta_min)),
                    "stroke_ok": bool(
                        stroke_row["peak_relative_displacement_m"] <= float(stroke_m)
                        or stroke_row["bump_stop_force_fraction_of_weight"] <= float(stop_force_limit_fraction)
                    ),
                    "ice_linear_ok": bool(ice_rel_rms <= mount.linearity_amplitude_m()),
                }
            )
    return rows


def select_design(
    rows: Sequence[dict[str, Any]],
    static_deflection_limit_m: float,
    residual_stiffness_budget: float,
    target_ratio: float = 1.0,
) -> dict[str, Any]:
    """Pick the lowest band ratio among feasible rows, with the margin reported.

    Selection rule (fixed before looking at the numbers): among rows whose
    four feasibility flags (``static_ok``, ``stiffness_ok``, ``zeta_ok``,
    ``stroke_ok``) hold and whose band ratio is below ``target_ratio``, take
    the smallest band ratio; ties are broken by the
    larger residual stiffness fraction (less cancellation = more robust
    mechanism).  If nothing is feasible, the row with the smallest ratio is
    returned with ``feasible = False`` so the caller reports an honest FAIL
    instead of a silently relaxed criterion.
    """
    if not rows:
        raise ValueError("design map is empty")
    feasible = [
        r
        for r in rows
        if r["static_ok"]
        and r["stiffness_ok"]
        and r.get("zeta_ok", True)
        and r.get("stroke_ok", True)
        and r["static_deflection_m"] <= float(static_deflection_limit_m)
        and r["rms_ratio_base_over_hull"] < float(target_ratio)
    ]
    pool = feasible or list(rows)
    best = min(pool, key=lambda r: (r["rms_ratio_base_over_hull"], -r["residual_stiffness_fraction"]))
    return {
        "feasible": bool(feasible),
        "target_ratio": float(target_ratio),
        "selected": dict(best),
        "n_feasible": int(len(feasible)),
        "n_rows": int(len(rows)),
        "best_ratio_achievable": float(min(r["rms_ratio_base_over_hull"] for r in rows)),
    }


def tolerance_window(
    freqs: Iterable[float],
    s_hull_accel: Iterable[float],
    band_hz: tuple[float, float],
    design: QZSMount,
    alpha_errors: Sequence[float] = (-0.05, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05),
) -> dict[str, Any]:
    """Band ratio under stiffness-cancellation errors ``delta alpha``.

    ``delta alpha > 0`` means the mechanism cancels *more* stiffness than
    designed (softer); ``delta alpha < 0`` is the ageing / relaxation direction
    of a real elastomer or strut mechanism (stiffer).  Returns the rows plus the
    symmetric window in which the attenuation criterion still holds.
    """
    f = _as_freqs(freqs)
    s = np.asarray(s_hull_accel, dtype=float)
    rows = []
    for d in alpha_errors:
        frac = design.residual_stiffness_fraction - float(d)
        if not 0.0 < frac <= 1.0:
            continue
        pert = QZSMount(
            equipment_mass_kg=design.equipment_mass_kg,
            f_rigid_hz=design.f_rigid_hz,
            residual_stiffness_fraction=frac,
            zeta=design.zeta,
            stroke_m=design.stroke_m,
            ns_range_m=design.ns_range_m,
        )
        att = band_attenuation(f, s, pert, band_hz)
        rows.append(
            {
                "delta_alpha": float(d),
                "f_n_hz": pert.f_n_hz,
                "rms_ratio_base_over_hull": att["rms_ratio_base_over_hull"],
                "attenuated": att["attenuated"],
            }
        )
    failing = [r["delta_alpha"] for r in rows if not r["attenuated"]]
    ok = [r["delta_alpha"] for r in rows if r["attenuated"]]
    return {
        "rows": rows,
        "max_abs_delta_alpha_still_attenuating": float(max((abs(v) for v in ok), default=0.0)),
        "first_failing_delta_alpha": float(min(failing, key=abs)) if failing else None,
    }


# ------------------------------------------------------------- time integration
def simulate_base_excitation(
    f_n_hz: float,
    zeta: float,
    f_probe_hz: float,
    cycles: int = 200,
    steps_per_cycle: int = 400,
    relative_amplitude_m: float = 1.0e-3,
    cubic_coefficient: float = 0.0,
    mass_kg: float = 1.0,
) -> dict[str, float]:
    """RK4 integration of the base-excited oscillator (independent check).

    Equation of motion in the relative coordinate ``u = x - x_h``:

        m u'' + c u' + k u + beta u^3 = -m x_h'',
        x_h = X_h sin(2 pi f t)

    For ``beta = 0`` the steady-state mass acceleration amplitude must equal
    ``|H| omega^2 X_h`` with ``|H|`` the transmissibility of :class:`QZSMount`
    (checked to 1 % relative in the tests).  With ``beta > 0`` the measured peak
    frequency shifts upwards (hardening), which is what the linearisation
    cannot capture, and the shift is used to bound its validity.
    """
    if f_n_hz <= 0.0 or not 0.0 < zeta < 1.0:
        raise ValueError("invalid oscillator")
    if f_probe_hz <= 0.0:
        raise ValueError("probe frequency must be positive")
    omega_n = 2.0 * np.pi * float(f_n_hz)
    omega = 2.0 * np.pi * float(f_probe_hz)
    k = float(mass_kg) * omega_n**2
    c = 2.0 * float(zeta) * float(mass_kg) * omega_n
    beta = float(cubic_coefficient)

    dt = 1.0 / (float(f_probe_hz) * steps_per_cycle)
    n_total = int(cycles * steps_per_cycle)

    def accel(u: float, v: float, t: float) -> tuple[float, float]:
        base = -omega**2 * relative_amplitude_m * np.sin(omega * t)
        return v, (-c * v - k * u - beta * u**3) / float(mass_kg) - base

    u = 0.0
    v = 0.0
    t = 0.0
    tail_u: list[float] = []
    tail_a: list[float] = []
    settle = int(0.75 * n_total)
    for i in range(n_total):
        k1u, k1v = accel(u, v, t)
        k2u, k2v = accel(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v, t + 0.5 * dt)
        k3u, k3v = accel(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v, t + 0.5 * dt)
        k4u, k4v = accel(u + dt * k3u, v + dt * k3v, t + dt)
        u += dt / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        t += dt
        if i >= settle:
            tail_u.append(u)
            # mass acceleration = base acceleration + u'' ; in steady state
            # u'' = -omega^2 u
            tail_a.append(-omega**2 * relative_amplitude_m * np.sin(omega * t) - omega**2 * u)
    amp_u = float(np.max(np.abs(tail_u)))
    amp_a = float(np.max(np.abs(tail_a)))
    return {
        "f_probe_hz": float(f_probe_hz),
        "relative_amplitude_m": float(relative_amplitude_m),
        "relative_displacement_amplitude_m": amp_u,
        "mass_acceleration_amplitude_m_s2": amp_a,
        "acceleration_transmissibility": amp_a / (omega**2 * relative_amplitude_m),
    }


def nonlinear_response_amplitude(
    f_n_hz: float,
    zeta: float,
    beta_over_k: float,
    mass_kg: float,
    support_accel_amplitude: float,
    freq_grid: Sequence[float],
    cycles: int = 100,
    steps_per_cycle: int = 400,
) -> list[dict[str, float]]:
    """Sweep the base-excited QZS oscillator through resonance (hardening check).

    Returns the measured acceleration transmissibility at every probe frequency.
    The peak of that curve drifts to the right with amplitude for ``beta > 0``
    (hardening), which is the physical behaviour the linear QZS model cannot
    reproduce; the payload reports the amplitude at which the linearisation
    error would reach 1 %.
    """
    omega_n = 2.0 * np.pi * float(f_n_hz)
    k = float(mass_kg) * omega_n**2
    beta = float(beta_over_k) * k
    out: list[dict[str, float]] = []
    for f_probe in freq_grid:
        omega = 2.0 * np.pi * float(f_probe)
        x_h = float(support_accel_amplitude) / omega**2
        res = simulate_base_excitation(
            f_n_hz=f_n_hz,
            zeta=zeta,
            f_probe_hz=float(f_probe),
            cycles=cycles,
            steps_per_cycle=steps_per_cycle,
            relative_amplitude_m=x_h,
            cubic_coefficient=beta,
            mass_kg=mass_kg,
        )
        out.append(
            {
                "f_probe_hz": float(f_probe),
                "relative_amplitude_m": x_h,
                "transmissibility": res["acceleration_transmissibility"],
            }
        )
    return out