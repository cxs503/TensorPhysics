"""Strain-life (epsilon-N) low-cycle fatigue: Coffin-Manson-Morrow, Neuber notch rule and the LCF/HCF transition.

Physics
-------
Total-strain approach.  The strain amplitude sustained by a material is split
into an elastic (Basquin) and a plastic (Coffin-Manson) part::

    de/2 = de_e/2 + de_p/2
         = (sigma'_f / E) (2 N_f)^b  +  eps'_f (2 N_f)^c

with ``b < 0`` the fatigue strength exponent, ``c < 0`` the fatigue ductility
exponent, ``sigma'_f`` the fatigue strength coefficient and ``eps'_f`` the
fatigue ductility coefficient.  The *Morrow* mean-stress correction replaces
``sigma'_f`` by ``sigma'_f - sigma_m`` in the elastic term (the plastic term is
mean-stress insensitive to first order), giving::

    de/2 = ((sigma'_f - sigma_m) / E) (2 N_f)^b + eps'_f (2 N_f)^c

**Transition life.**  The elastic and plastic contributions cross where
``de_e = de_p``::

    2 N_T = [ eps'_f E / (sigma'_f - sigma_m) ] ** (1 / (b - c))

Below ``N_T`` plasticity dominates (LCF, strain-controlled), above ``N_T`` the
elastic term dominates (HCF, stress-controlled).  ``N_T`` is therefore the
regime selector for a load spectrum: cycles with ``N_f < N_T`` must be assessed
in strain terms, cycles with ``N_f > N_T`` in stress / S-N terms.

**Cyclic curve.**  The companion Ramberg-Osgood stabilised cycle,

    de/2 = dsigma/(2E) + (dsigma/(2 K'))^(1/n')

is *not* independent of the four strain-life constants.  Requiring the elastic
and plastic lines to intersect also at ``2N = 1`` (Morrow compatibility) gives::

    n' = b / c,      K' = sigma'_f / (eps'_f ** n')

so a material can be entered with the four epsilon-N constants only; ``K'`` and
``n'`` are derived unless supplied explicitly.

**Neuber notch rule.**  For a notched member loaded by a nominal (remote)
elastic stress range ``dS`` the notch-root ranges ``dsigma``, ``deps`` satisfy
the hyperbola::

    (K_t dS)^2 / E = dsigma * deps

intersected with the cyclic curve.  The notch strain is the root of
``dsigma * deps(dsigma) - (K_t dS)^2 / E = 0``.  The classical K_t -> K_f
correction (fatigue notch factor) is

    K_f = 1 + (K_t - 1) / (1 + sqrt(rho / r))        (Neuber)
    K_f = 1 + (K_t - 1) / (1 + a / r)                (Peterson)

with ``rho`` / ``a`` material constants and ``r`` the notch radius; both tend
to 1 as ``r -> inf`` (vanishing stress concentration) and to ``K_t`` as
``r -> 0``.

Everything here is pure arithmetic on documented material constants, so it is
unit-testable without external data; the ice-pressure load spectrum is supplied
by :mod:`tensorphysics.fatigue.ice_pressure_lowcycle`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    "StrainLifeMaterial",
    "n_prime_from_coefficients",
    "k_prime_from_coefficients",
    "elastic_strain_amplitude",
    "plastic_strain_amplitude",
    "total_strain_amplitude",
    "strain_life_curve",
    "reversals_for_strain_amplitude",
    "cycles_to_failure",
    "transition_reversals",
    "transition_life",
    "cyclic_strain_amplitude",
    "stress_amplitude_from_strain",
    "NeuberNotchResult",
    "neuber_notch_strain",
    "glinka_notch_strain",
    "notch_factor_neuber",
    "notch_factor_peterson",
    "notch_factor",
    "stress_concentration_elliptical_hole",
    "notch_factor_from_geometry",
    "DamageBlock",
    "StrainLifeDamageResult",
    "strain_life_damage",
    "endurance_limit_strain_amplitude",
]

#: log10 search window for the reversal count (0.01 .. 1e14 reversals).
_LOG2N_MIN = -2.0
_LOG2N_MAX = 14.0


# ---------------------------------------------------------------------------
# Material definition
# ---------------------------------------------------------------------------


def n_prime_from_coefficients(b: float, c: float) -> float:
    """Cyclic strain-hardening exponent from the strain-life exponents: ``n' = b/c``."""
    if b >= 0.0 or c >= 0.0:
        raise ValueError("b and c must be negative")
    return b / c


def k_prime_from_coefficients(sigma_f_MPa: float, eps_f: float, n_prime: float) -> float:
    """Cyclic strength coefficient from Morrow compatibility: ``K' = sigma'_f / eps'_f**n'``."""
    if sigma_f_MPa <= 0.0 or eps_f <= 0.0 or n_prime <= 0.0:
        raise ValueError("sigma'_f, eps'_f and n' must be positive")
    return sigma_f_MPa / (eps_f ** n_prime)


@dataclass(frozen=True)
class StrainLifeMaterial:
    """Four-constant strain-life material (with derived cyclic curve).

    Parameters
    ----------
    name : str
        Label used in reports / payloads.
    E_MPa : float
        Young's modulus [MPa].
    sigma_f_MPa, b : float
        Fatigue strength coefficient [MPa] and exponent (``b < 0``).
    eps_f, c : float
        Fatigue ductility coefficient [-] and exponent (``c < 0``).
    K_prime_MPa, n_prime : float, optional
        Cyclic strength coefficient [MPa] / hardening exponent; derived from the
        four basic constants (Morrow compatibility) when omitted.
    notch_constant_mm : float, optional
        Material length constant for the K_t -> K_f corrections
        (Neuber ``rho`` / Peterson ``a``) [mm].
    """

    name: str
    E_MPa: float
    sigma_f_MPa: float
    b: float
    eps_f: float
    c: float
    K_prime_MPa: Optional[float] = None
    n_prime: Optional[float] = None
    notch_constant_mm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.E_MPa <= 0.0:
            raise ValueError("E_MPa must be positive")
        if self.sigma_f_MPa <= 0.0 or self.eps_f <= 0.0:
            raise ValueError("sigma_f_MPa and eps_f must be positive")
        if self.b >= 0.0 or self.c >= 0.0:
            raise ValueError("b and c must be negative (fatigue exponents)")
        if self.b == self.c:
            raise ValueError("b == c makes the transition life degenerate")
        n_p = float(self.n_prime) if self.n_prime is not None else n_prime_from_coefficients(self.b, self.c)
        k_p = (
            float(self.K_prime_MPa)
            if self.K_prime_MPa is not None
            else k_prime_from_coefficients(self.sigma_f_MPa, self.eps_f, n_p)
        )
        if n_p <= 0.0 or k_p <= 0.0:
            raise ValueError("cyclic parameters n', K' must be positive")
        object.__setattr__(self, "n_prime", n_p)
        object.__setattr__(self, "K_prime_MPa", k_p)

    # -- derived quantities -------------------------------------------------
    @property
    def elastic_intercept(self) -> float:
        """``sigma'_f / E`` -- elastic strain amplitude at ``2N = 1``."""
        return self.sigma_f_MPa / self.E_MPa

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "E_MPa": self.E_MPa,
            "sigma_f_MPa": self.sigma_f_MPa,
            "b": self.b,
            "eps_f": self.eps_f,
            "c": self.c,
            "n_prime": self.n_prime,
            "K_prime_MPa": self.K_prime_MPa,
            "notch_constant_mm": self.notch_constant_mm,
        }


# ---------------------------------------------------------------------------
# epsilon-N curve
# ---------------------------------------------------------------------------


def elastic_strain_amplitude(
    material: StrainLifeMaterial, reversals: float, mean_stress_MPa: float = 0.0
) -> float:
    """Elastic strain amplitude ``(sigma'_f - sigma_m)/E * (2N)^b`` (Morrow)."""
    if reversals <= 0.0:
        raise ValueError("reversals must be positive")
    amp = material.elastic_intercept * float(reversals) ** material.b
    if mean_stress_MPa:
        if mean_stress_MPa >= material.sigma_f_MPa:
            return 0.0
        amp *= (material.sigma_f_MPa - mean_stress_MPa) / material.sigma_f_MPa
    return float(amp)


def plastic_strain_amplitude(material: StrainLifeMaterial, reversals: float) -> float:
    """Plastic strain amplitude ``eps'_f * (2N)^c`` (Coffin-Manson)."""
    if reversals <= 0.0:
        raise ValueError("reversals must be positive")
    return float(material.eps_f * float(reversals) ** material.c)


def total_strain_amplitude(
    material: StrainLifeMaterial, reversals: float, mean_stress_MPa: float = 0.0
) -> float:
    """Total strain amplitude ``de/2`` at a given number of reversals."""
    return elastic_strain_amplitude(material, reversals, mean_stress_MPa) + plastic_strain_amplitude(
        material, reversals
    )


def strain_life_curve(
    material: StrainLifeMaterial,
    reversals: Optional[Sequence[float]] = None,
    n_points: int = 200,
    mean_stress_MPa: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Tabulate the epsilon-N curve (elastic / plastic / total branches).

    Returns a dict of arrays ``reversals``, ``cycles``, ``elastic``, ``plastic``,
    ``total`` (all strain amplitudes, dimensionless).
    """
    if reversals is None:
        reversals = np.logspace(0.0, 8.0, int(n_points))
    rev = np.asarray(reversals, dtype=float)
    if np.any(rev <= 0.0):
        raise ValueError("reversals must be positive")
    elastic = np.array([elastic_strain_amplitude(material, r, mean_stress_MPa) for r in rev])
    plastic = np.array([plastic_strain_amplitude(material, r) for r in rev])
    return {
        "reversals": rev,
        "cycles": rev / 2.0,
        "elastic": elastic,
        "plastic": plastic,
        "total": elastic + plastic,
    }


def reversals_for_strain_amplitude(
    material: StrainLifeMaterial, strain_amplitude: float, mean_stress_MPa: float = 0.0
) -> float:
    """Invert ``de/2 = f(2N_f)`` for the reversal count.

    The total strain amplitude is monotonically decreasing in ``2N`` (both
    exponents are negative), so a log-domain bisection is unconditionally
    convergent.  A strain amplitude above the curve value at ``2N = 0.01``
    raises ``ValueError`` (life shorter than one reversal).
    """
    if strain_amplitude <= 0.0:
        raise ValueError("strain_amplitude must be positive")
    f_lo = total_strain_amplitude(material, 10.0 ** _LOG2N_MIN, mean_stress_MPa) - strain_amplitude
    f_hi = total_strain_amplitude(material, 10.0 ** _LOG2N_MAX, mean_stress_MPa) - strain_amplitude
    if f_lo < 0.0:
        raise ValueError(
            "strain amplitude exceeds the epsilon-N curve at 2N = 0.01 (life below one reversal)"
        )
    if f_hi > 0.0:
        # Like the log window: hand back the largest tabulated life (run-out).
        return 10.0 ** _LOG2N_MAX
    lo, hi = _LOG2N_MIN, _LOG2N_MAX
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = total_strain_amplitude(material, 10.0 ** mid, mean_stress_MPa) - strain_amplitude
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return float(10.0 ** (0.5 * (lo + hi)))


def cycles_to_failure(
    material: StrainLifeMaterial, strain_amplitude: float, mean_stress_MPa: float = 0.0
) -> float:
    """Cycles to failure ``N_f`` for a total strain amplitude."""
    return reversals_for_strain_amplitude(material, strain_amplitude, mean_stress_MPa) / 2.0


def transition_reversals(material: StrainLifeMaterial, mean_stress_MPa: float = 0.0) -> float:
    """Reversals at which elastic and plastic strain amplitudes are equal.

    ``2 N_T = [eps'_f E / (sigma'_f - sigma_m)] ** (1/(b - c))``
    """
    sigma = material.sigma_f_MPa - float(mean_stress_MPa)
    if sigma <= 0.0:
        raise ValueError("(sigma'_f - sigma_m) must be positive for a finite transition life")
    base = material.eps_f * material.E_MPa / sigma
    return float(base ** (1.0 / (material.b - material.c)))


def transition_life(material: StrainLifeMaterial, mean_stress_MPa: float = 0.0) -> float:
    """Transition cycles ``N_T`` separating LCF (strain) from HCF (stress) dominance."""
    return transition_reversals(material, mean_stress_MPa) / 2.0


def endurance_limit_strain_amplitude(material: StrainLifeMaterial, reversals: float = 2.0e8) -> float:
    """Strain amplitude at a run-out reversal count (``2e8`` reversals = ``1e8`` cycles).

    Used as the "infinite life" screening level: a strain amplitude below this
    value is treated as a run-out (``N_f -> inf``) by :func:`strain_life_damage`.
    """
    return total_strain_amplitude(material, float(reversals))


# ---------------------------------------------------------------------------
# Cyclic stress-strain (Ramberg-Osgood)
# ---------------------------------------------------------------------------


def cyclic_strain_amplitude(material: StrainLifeMaterial, stress_amplitude_MPa: float) -> float:
    """Ramberg-Osgood strain amplitude for a given stress amplitude."""
    if stress_amplitude_MPa <= 0.0:
        raise ValueError("stress_amplitude_MPa must be positive")
    elastic = stress_amplitude_MPa / material.E_MPa
    plastic = (stress_amplitude_MPa / material.K_prime_MPa) ** (1.0 / material.n_prime)
    return float(elastic + plastic)


def stress_amplitude_from_strain(
    material: StrainLifeMaterial, strain_amplitude: float
) -> float:
    """Invert Ramberg-Osgood: stress **amplitude** from a strain amplitude [MPa].

    Monotone in ``dsigma``; log-domain bisection on ``dsigma``.
    """
    if strain_amplitude <= 0.0:
        raise ValueError("strain_amplitude must be positive")
    lo, hi = 1e-10, 10.0 * material.E_MPa
    if cyclic_strain_amplitude(material, hi) < strain_amplitude:
        raise ValueError("strain amplitude beyond the cyclic curve range")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if cyclic_strain_amplitude(material, mid) < strain_amplitude:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-12:
            break
    return float(math.sqrt(lo * hi))


# ---------------------------------------------------------------------------
# Neuber notch correction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeuberNotchResult:
    """Notch-root state from the Neuber hyperbola / cyclic-curve intersection.

    All stresses are ranges in MPa, all strains are ranges (not amplitudes),
    matching the ``dS`` input convention.
    """

    K_t: float
    nominal_stress_range_MPa: float
    nominal_strain_range: float
    notch_stress_range_MPa: float
    notch_strain_range: float
    strain_amplitude: float
    stress_amplitude_MPa: float
    K_sigma: float
    K_eps: float
    neuber_identity_error: float
    plastic_fraction: float

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "K_t": self.K_t,
            "nominal_stress_range_MPa": self.nominal_stress_range_MPa,
            "nominal_strain_range": self.nominal_strain_range,
            "notch_stress_range_MPa": self.notch_stress_range_MPa,
            "notch_strain_range": self.notch_strain_range,
            "strain_amplitude": self.strain_amplitude,
            "stress_amplitude_MPa": self.stress_amplitude_MPa,
            "K_sigma": self.K_sigma,
            "K_eps": self.K_eps,
            "neuber_identity_error": self.neuber_identity_error,
            "plastic_fraction": self.plastic_fraction,
        }


def neuber_notch_strain(
    material: StrainLifeMaterial, K_t: float, nominal_stress_range_MPa: float
) -> NeuberNotchResult:
    """Solve the Neuber hyperbola against the cyclic curve.

    ``(K_t dS)^2 / E = dsigma * deps(dsigma)`` with
    ``deps = dsigma/E + 2 (dsigma/(2K'))^(1/n')``.

    The left side is constant and the right side increases monotonically in
    ``dsigma``, so the intersection is unique; a log-domain bisection is used.
    """
    if K_t < 1.0:
        raise ValueError("K_t must be >= 1")
    if nominal_stress_range_MPa <= 0.0:
        raise ValueError("nominal_stress_range_MPa must be positive")
    target = (K_t * nominal_stress_range_MPa) ** 2 / material.E_MPa

    def g(dsigma: float) -> float:
        deps = dsigma / material.E_MPa + 2.0 * (dsigma / (2.0 * material.K_prime_MPa)) ** (
            1.0 / material.n_prime
        )
        return dsigma * deps - target

    lo = 1e-9 * nominal_stress_range_MPa
    hi = K_t * nominal_stress_range_MPa * 10.0 + material.sigma_f_MPa
    while g(hi) < 0.0:
        hi *= 4.0
        if hi > 1e6:
            raise RuntimeError("Neuber solve failed to bracket the notch stress")
    if g(lo) > 0.0:  # pragma: no cover - only for absurd inputs
        raise RuntimeError("Neuber solve bracket is degenerate")
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-13:
            break
    dsigma = math.sqrt(lo * hi)
    deps = dsigma / material.E_MPa + 2.0 * (dsigma / (2.0 * material.K_prime_MPa)) ** (
        1.0 / material.n_prime
    )
    elastic_part = dsigma / material.E_MPa
    return NeuberNotchResult(
        K_t=float(K_t),
        nominal_stress_range_MPa=float(nominal_stress_range_MPa),
        nominal_strain_range=float(nominal_stress_range_MPa / material.E_MPa),
        notch_stress_range_MPa=float(dsigma),
        notch_strain_range=float(deps),
        strain_amplitude=float(deps / 2.0),
        stress_amplitude_MPa=float(dsigma / 2.0),
        K_sigma=float(dsigma / nominal_stress_range_MPa),
        K_eps=float(deps / (nominal_stress_range_MPa / material.E_MPa)),
        neuber_identity_error=float(abs(dsigma * deps - target) / target),
        plastic_fraction=float(1.0 - elastic_part / deps),
    )


def notch_factor_neuber(K_t: float, radius_mm: float, rho_mm: float) -> float:
    """Neuber fatigue notch factor ``K_f = 1 + (K_t-1)/(1 + sqrt(rho/r))``."""
    if K_t < 1.0:
        raise ValueError("K_t must be >= 1")
    if radius_mm <= 0.0 or rho_mm <= 0.0:
        raise ValueError("radius and rho must be positive")
    return float(1.0 + (K_t - 1.0) / (1.0 + math.sqrt(rho_mm / radius_mm)))


def notch_factor_peterson(K_t: float, radius_mm: float, a_mm: float) -> float:
    """Peterson fatigue notch factor ``K_f = 1 + (K_t-1)/(1 + a/r)``."""
    if K_t < 1.0:
        raise ValueError("K_t must be >= 1")
    if radius_mm <= 0.0 or a_mm <= 0.0:
        raise ValueError("radius and a must be positive")
    return float(1.0 + (K_t - 1.0) / (1.0 + a_mm / radius_mm))


def notch_factor(
    K_t: float,
    radius_mm: float,
    material_constant_mm: Optional[float] = None,
    method: str = "neuber",
    material: Optional[StrainLifeMaterial] = None,
) -> float:
    """Dispatch to the Neuber or Peterson K_t -> K_f correction.

    ``material_constant_mm`` falls back to ``material.notch_constant_mm``.
    """
    constant = material_constant_mm
    if constant is None and material is not None:
        constant = material.notch_constant_mm
    if constant is None:
        raise ValueError("a material length constant is required (radius-independent K_f)")
    key = method.strip().lower()
    if key == "neuber":
        return notch_factor_neuber(K_t, radius_mm, constant)
    if key == "peterson":
        return notch_factor_peterson(K_t, radius_mm, constant)
    raise ValueError(f"unknown notch-factor method {method!r} (use 'neuber' or 'peterson')")


def glinka_notch_strain(
    material: StrainLifeMaterial, K_t: float, nominal_stress_range_MPa: float
) -> NeuberNotchResult:
    """Independent notch closure: Glinka's strain-energy-density method.

    Instead of conserving the product ``dsigma*deps`` (Neuber), the energy
    method conserves the *strain energy density* at the notch root against the
    amplified nominal elastic energy density::

        int_0^{sigma_a} sigma d(eps)  =  K_t^2 * (dS/2)^2 / (2E)

    which for the Ramberg-Osgood cyclic curve gives

        sigma_a^2/(2E) + K'/(n'+1) * (sigma_a/K')^{((n'+1)/n')}
            = K_t^2 dS^2 / (8E)

    The two closures agree in small-scale yielding (both collapse to the
    elastic amplification) and diverge as the plastic zone spreads -- Glinka
    gives the smaller notch strain.  Comparing them is therefore a genuine
    cross-method check on the notch-root strain, independent of any single
    empirical rule (see Glinka 1985; comparative studies of the two approaches
    e.g. *Strain* 2023, doi:10.1111/str.12470).
    """
    if K_t < 1.0:
        raise ValueError("K_t must be >= 1")
    if nominal_stress_range_MPa <= 0.0:
        raise ValueError("nominal_stress_range_MPa must be positive")
    target = K_t ** 2 * nominal_stress_range_MPa ** 2 / (8.0 * material.E_MPa)
    n_p = material.n_prime
    k_p = material.K_prime_MPa

    def energy(sigma_a: float) -> float:
        elastic = sigma_a ** 2 / (2.0 * material.E_MPa)
        plastic = (1.0 / (n_p + 1.0)) * k_p * (sigma_a / k_p) ** ((n_p + 1.0) / n_p)
        return elastic + plastic

    lo = 1e-12 * nominal_stress_range_MPa
    hi = K_t * nominal_stress_range_MPa + material.sigma_f_MPa
    while energy(hi) < target:
        hi *= 4.0
        if hi > 1e6:
            raise RuntimeError("Glinka solve failed to bracket the notch stress")
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        if energy(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-13:
            break
    dsigma = 2.0 * math.sqrt(lo * hi)
    deps = 2.0 * (
        dsigma / (2.0 * material.E_MPa) + (dsigma / (2.0 * k_p)) ** (1.0 / n_p)
    )
    elastic_part = dsigma / material.E_MPa
    return NeuberNotchResult(
        K_t=float(K_t),
        nominal_stress_range_MPa=float(nominal_stress_range_MPa),
        nominal_strain_range=float(nominal_stress_range_MPa / material.E_MPa),
        notch_stress_range_MPa=float(dsigma),
        notch_strain_range=float(deps),
        strain_amplitude=float(deps / 2.0),
        stress_amplitude_MPa=float(dsigma / 2.0),
        K_sigma=float(dsigma / nominal_stress_range_MPa),
        K_eps=float(deps / (nominal_stress_range_MPa / material.E_MPa)),
        neuber_identity_error=float("nan"),
        plastic_fraction=float(1.0 - elastic_part / deps),
    )


def stress_concentration_elliptical_hole(notch_depth_mm: float, radius_mm: float) -> float:
    """Elastic stress concentration of an elliptical notch: ``K_t = 1 + 2 sqrt(a/r)``.

    ``a`` is the notch depth and ``r`` its root radius (Inglis / Neuber).  The
    concentration therefore tends to 1 as the notch becomes blunt
    (``r -> inf``) and grows without bound as the root sharpens.  This is the
    geometry side of the K_t -> K_f correction: the *notch sensitivity*
    ``q = (K_f - 1)/(K_t - 1)`` itself falls for very sharp roots, so K_f must
    always be quoted together with the K_t it was derived from.
    """
    if notch_depth_mm <= 0.0 or radius_mm <= 0.0:
        raise ValueError("notch depth and radius must be positive")
    return float(1.0 + 2.0 * math.sqrt(notch_depth_mm / radius_mm))


def notch_factor_from_geometry(
    notch_depth_mm: float,
    radius_mm: float,
    material_constant_mm: float,
    method: str = "neuber",
) -> Dict[str, float]:
    """Geometry-driven K_t -> K_f: returns ``K_t``, ``K_f`` and sensitivity ``q``.

    ``q = (K_f - 1) / (K_t - 1)`` is the notch sensitivity; ``K_f -> 1`` follows
    from ``K_t -> 1`` as the root radius grows (blunt notch), which is the
    negative control used by the acceptance criteria.
    """
    k_t = stress_concentration_elliptical_hole(notch_depth_mm, radius_mm)
    if k_t == 1.0:
        return {"K_t": 1.0, "K_f": 1.0, "q": 1.0}
    k_f = notch_factor(k_t, radius_mm, material_constant_mm, method=method)
    return {"K_t": k_t, "K_f": k_f, "q": float((k_f - 1.0) / (k_t - 1.0))}


# ---------------------------------------------------------------------------
# Miner accumulation on the strain-life curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DamageBlock:
    """One constant-amplitude block of the load spectrum."""

    strain_amplitude: float
    cycles: float
    mean_stress_MPa: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.strain_amplitude <= 0.0:
            raise ValueError("strain_amplitude must be positive")
        if self.cycles < 0.0:
            raise ValueError("cycles must be non-negative")


@dataclass
class StrainLifeDamageResult:
    """Per-block strain-life damage plus the LCF/HCF split."""

    blocks: List[Dict[str, Any]]
    damage_total: float
    damage_lcf: float
    damage_hcf: float
    transition_life: float

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "blocks": self.blocks,
            "damage_total": self.damage_total,
            "damage_lcf": self.damage_lcf,
            "damage_hcf": self.damage_hcf,
            "transition_life": self.transition_life,
        }


def strain_life_damage(
    material: StrainLifeMaterial,
    blocks: Iterable[DamageBlock],
    runout_reversals: float = 2.0e8,
) -> StrainLifeDamageResult:
    """Palmgren-Miner accumulation on the epsilon-N curve.

    ``D = sum n_i / N_f,i``.  Each block is additionally labelled LCF when
    ``N_f,i < N_T`` and HCF otherwise, and the damage is split accordingly;
    blocks whose strain amplitude is below the run-out level contribute zero
    damage (infinite life).
    """
    n_t = transition_life(material)
    runout_level = endurance_limit_strain_amplitude(material, runout_reversals)
    rows: List[Dict[str, Any]] = []
    d_total = d_lcf = d_hcf = 0.0
    for block in blocks:
        if block.strain_amplitude <= runout_level:
            n_f = math.inf
            d = 0.0
            regime = "run-out"
        else:
            n_f = cycles_to_failure(material, block.strain_amplitude, block.mean_stress_MPa)
            d = block.cycles / n_f if n_f > 0.0 else math.inf
            regime = "LCF" if n_f < n_t else "HCF"
        rows.append(
            {
                "label": block.label,
                "strain_amplitude": float(block.strain_amplitude),
                "cycles": float(block.cycles),
                "mean_stress_MPa": float(block.mean_stress_MPa),
                "N_f": None if math.isinf(n_f) else float(n_f),
                "damage": float(d),
                "regime": regime,
            }
        )
        d_total += d
        if regime == "LCF":
            d_lcf += d
        elif regime == "HCF":
            d_hcf += d
    return StrainLifeDamageResult(
        blocks=rows,
        damage_total=float(d_total),
        damage_lcf=float(d_lcf),
        damage_hcf=float(d_hcf),
        transition_life=float(n_t),
    )