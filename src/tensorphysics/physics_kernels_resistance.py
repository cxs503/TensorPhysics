"""Pure resistance reference kernels for the physics contract registry.

All functions are pure (no IO, no global mutable state) and use explicit SI
units.  They exist so acceptance checks can compare an *independent*
implementation against production code instead of re-importing it.  Provenance
and error bands live in :mod:`tensorphysics.physics_contracts`, not in
docstrings alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81  # gravitational acceleration [m/s^2]


def ittc_1957_cf(re: float) -> float:
    """ITTC-1957 model-ship correlation line: Cf = 0.075/(log10(Re)-2)^2."""
    if not math.isfinite(re) or re <= 0.0:
        raise ValueError(f"Reynolds number must be finite and positive, got {re}")
    return 0.075 / (math.log10(re) - 2.0) ** 2


def schoenherr_1932_cf(re: float) -> float:
    """Schoenherr (1932) line 0.242/sqrt(Cf) = log10(Re*Cf), bisected.

    Independent historical cross-check of the ITTC-1957 line (two distinct
    correlations, not a tautology).
    """
    if not math.isfinite(re) or re <= 0.0:
        raise ValueError(f"Reynolds number must be finite and positive, got {re}")

    def resid(cf: float) -> float:
        return 0.242 / math.sqrt(cf) - math.log10(re * cf)

    lo, hi = 1e-6, 2e-2
    if resid(lo) < 0.0 or resid(hi) > 0.0:
        raise ValueError(f"Schoenherr bisection bracket failed at re={re}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def axisymmetric_wetted_surface(length_m: float, beam_m: float, cb: float) -> float:
    """Legacy wetted-surface estimate for axisymmetric bodies.

    Measured bias against closed-form shapes: cylinder (Cb=pi/4) -30.4%,
    prolate spheroid (b/a=0.25) -8.7% (TensorPlatform tools/verif audit).
    Must be declared explicitly by the caller -- production contracts prefer
    a geometry-validated wetted surface (audited failure mode: undeclared use
    of this fit biased KVLCC2 S by +50.6% vs the published value).
    """
    if length_m <= 0 or beam_m <= 0:
        raise ValueError("length and beam must be positive")
    if not 0.0 < cb <= 1.0:
        raise ValueError(f"block coefficient out of (0, 1], got {cb}")
    return math.pi * beam_m * length_m * (0.5 + 0.25 * cb)


def suboff_residual_coefficient(froude: float, cb: float, fn_floor: float = 0.05) -> float:
    """Empirical residual coefficient Cr = 0.002*(1+2*Cb)*Fn^2.5 for Fn>floor.

    Curve fit calibrated on SUBOFF-like axisymmetric bodies, no published
    provenance -- recorded as a contract limitation; not a general
    merchant-ship residuary method.
    """
    if not 0.0 < cb <= 1.0:
        raise ValueError(f"block coefficient out of (0, 1], got {cb}")
    if froude <= fn_floor:
        return 0.0
    return 0.002 * (1.0 + 2.0 * cb) * froude ** 2.5


@dataclass(frozen=True)
class SharedCoreBreakdown:
    s_wet_m2: float
    wetted_surface_source: str
    reynolds: float
    froude: float
    cf: float
    cr: float
    form_factor_1pk: float
    rf_n: float
    rform_n: float
    rr_n: float
    rt_n: float
    ct: float


def shared_core_resistance(
    length_m: float,
    beam_m: float,
    cb: float,
    speed_ms: float,
    rho_kg_m3: float,
    nu_m2_s: float,
    wetted_surface_m2: float,
    form_factor: float = 1.0,
) -> SharedCoreBreakdown:
    """Physics shared bitwise by 6 of 11 audited TensorPlatform kernels.

    Composition: ITTC-1957 friction + SUBOFF-calibrated residual fit +
    optional (1+k) form factor.  ``wetted_surface_m2`` is a *required*
    argument with no silent fallback (the audited platform's largest error
    source).  Call :func:`axisymmetric_wetted_surface` yourself if that
    derivation is what you mean.
    """
    if length_m <= 0 or beam_m <= 0:
        raise ValueError("length and beam must be positive")
    if not 0.3 < cb < 1.0:
        raise ValueError(f"block coefficient out of physical range (0.3, 1.0), got {cb}")
    if speed_ms <= 0:
        raise ValueError("speed must be positive")
    if rho_kg_m3 <= 0 or nu_m2_s <= 0:
        raise ValueError("density and viscosity must be positive")
    if wetted_surface_m2 is None or wetted_surface_m2 <= 0:
        raise ValueError(
            "wetted_surface_m2 is required -- no silent fallback; derive it "
            "explicitly via axisymmetric_wetted_surface() if intended"
        )
    if form_factor < 1.0:
        raise ValueError(f"form factor (1+k) must be >= 1.0, got {form_factor}")

    s = float(wetted_surface_m2)
    re = speed_ms * length_m / nu_m2_s
    fn = speed_ms / math.sqrt(G * length_m)
    cf = ittc_1957_cf(re)
    cr = suboff_residual_coefficient(fn, cb)
    q = 0.5 * rho_kg_m3 * s * speed_ms ** 2
    rf = q * cf
    rform = (form_factor - 1.0) * rf
    rr = q * cr
    rt = rf + rform + rr
    return SharedCoreBreakdown(
        s_wet_m2=s,
        wetted_surface_source="caller_declared",
        reynolds=re,
        froude=fn,
        cf=cf,
        cr=cr,
        form_factor_1pk=form_factor,
        rf_n=rf,
        rform_n=rform,
        rr_n=rr,
        rt_n=rt,
        ct=rt / q,
    )
