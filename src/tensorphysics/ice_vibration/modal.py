"""Modal models for ice-induced hull-girder vibration.

This module holds *structure-side* data only: natural frequencies, damping
ratios, generalized (modal) masses and mode-shape quantities needed to map a
generalized coordinate back to a physical response (bow displacement, midship
curvature / bending moment).

Two modal sources are supported and strictly separated:

``fem``
    Modal parameters extracted by a specialist solver (TensorFEM) and handed
    over as a JSON payload.  :func:`modal_data_from_fem_extraction` validates
    the payload; TensorPhysics itself does not re-implement the eigensolver.
``euler_bernoulli_reference``
    A closed-form Euler-Bernoulli uniform free-free beam reference used to
    verify the machinery around the eigensolver (frequency ratios, modal
    masses, curvature).  It is a *reference*, not a ship FEM.

Both paths return the same :class:`ModalData` container so that every
downstream analysis (forced vibration, Duffing harmonic balance, transient
whiplash) is source-agnostic.

Units: SI throughout (m, kg, s, N, N*m, Pa).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

__all__ = [
    "HullParameters",
    "ModalData",
    "FREE_FREE_BETA_L",
    "CANTILEVER_BETA_L",
    "free_free_beam_mode_shape",
    "free_free_beam_frequency",
    "free_free_beam_modal_data",
    "cantilever_beam_frequency",
    "rayleigh_coefficients",
    "modal_data_from_fem_extraction",
]

#: (beta_n * L) roots of cosh(bL)cos(bL) = 1 for a uniform free-free beam.
FREE_FREE_BETA_L: Tuple[float, ...] = (
    4.7300407448627040,
    7.8532046240958376,
    10.9956078380016710,
    14.1371654912574642,
)

#: (beta_n * L) roots of cosh(bL)cos(bL) = -1 for a uniform cantilever beam.
CANTILEVER_BETA_L: Tuple[float, ...] = (
    1.8751040687119611,
    4.6940911329741746,
    7.8547574382376126,
    10.9955407348754660,
)

#: Water density used for displacement bookkeeping (kg/m^3).
RHO_SEA_WATER = 1025.0


# ---------------------------------------------------------------------------
# Hull geometry / mass bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HullParameters:
    """Principal particulars of the hull under study.

    The block coefficient is *recomputed* from length/breadth/draught and the
    stated displacement so that an inconsistent input set cannot silently
    propagate into the vibration model (geometry is validated before any
    dynamics is run).
    """

    length_m: float
    breadth_m: float
    draught_m: float
    block_coefficient: float
    displacement_kg: float
    name: str = "generic-ice-class-hull"

    def __post_init__(self) -> None:
        for field_name in ("length_m", "breadth_m", "draught_m", "block_coefficient", "displacement_kg"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"HullParameters.{field_name} must be a positive finite number, got {value!r}")
        if not 0.3 <= self.block_coefficient <= 0.95:
            raise ValueError(
                f"block coefficient {self.block_coefficient} outside plausible ship range [0.3, 0.95]"
            )

    @property
    def displaced_volume_m3(self) -> float:
        return self.displacement_kg / RHO_SEA_WATER

    @property
    def cb_from_displacement(self) -> float:
        """Block coefficient implied by L, B, T and the stated displacement."""
        return self.displaced_volume_m3 / (self.length_m * self.breadth_m * self.draught_m)

    @property
    def mass_per_length_kg_m(self) -> float:
        """Effective hull-girder mass per unit length (rigid-body mass / L).

        The added mass of the surrounding water is *not* included here; it can
        be supplied explicitly through the modal mass of the FEM extraction.
        """
        return self.displacement_kg / self.length_m

    def verify(self, rel_tol: float = 0.02) -> Dict[str, Any]:
        """Check that L/B/T and Cb are mutually consistent.

        Returns a JSON-friendly audit record; ``consistent`` is False when the
        stated block coefficient and the one implied by the displacement differ
        by more than ``rel_tol`` (relative).
        """
        cb_implied = self.cb_from_displacement
        rel_err = abs(cb_implied - self.block_coefficient) / self.block_coefficient
        return {
            "name": self.name,
            "length_m": self.length_m,
            "breadth_m": self.breadth_m,
            "draught_m": self.draught_m,
            "block_coefficient_stated": self.block_coefficient,
            "block_coefficient_from_displacement": cb_implied,
            "displacement_kg": self.displacement_kg,
            "displaced_volume_m3": self.displaced_volume_m3,
            "mass_per_length_kg_m": self.mass_per_length_kg_m,
            "relative_error": rel_err,
            "tolerance": rel_tol,
            "consistent": bool(rel_err <= rel_tol),
        }

    def to_json_dict(self) -> Dict[str, Any]:
        payload = self.verify()
        payload["class"] = "HullParameters"
        return payload


# ---------------------------------------------------------------------------
# Modal data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModalData:
    """Modal description of the hull girder.

    ``mode_shape_bow`` / ``mode_shape_midship_curvature`` are the mode-shape
    ordinates (dimensionless, shape normalized to unit maximum amplitude) and
    the curvature ordinate (1/m) at midship, i.e. the quantities that convert a
    generalized coordinate ``q`` into a physical bow displacement
    ``w_bow = sum(phi_bow * q)`` and a midship bending moment
    ``M_mid = EI * sum(phi''_mid * q)``.
    """

    frequencies_hz: Tuple[float, ...]
    damping_ratios: Tuple[float, ...]
    modal_masses_kg: Tuple[float, ...]
    mode_shape_bow: Tuple[float, ...] = ()
    mode_shape_midship_curvature: Tuple[float, ...] = ()
    mode_shape_quarter_curvature: Tuple[float, ...] = ()
    source: str = "unspecified"
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        freqs = tuple(float(f) for f in self.frequencies_hz)
        zetas = tuple(float(z) for z in self.damping_ratios)
        masses = tuple(float(m) for m in self.modal_masses_kg)
        if not freqs:
            raise ValueError("at least one mode is required")
        if len(freqs) != len(zetas) or len(freqs) != len(masses):
            raise ValueError(
                f"mode arrays must have equal length, got {len(freqs)} freqs / {len(zetas)} damping / "
                f"{len(masses)} masses"
            )
        for index, freq in enumerate(freqs):
            if not math.isfinite(freq) or freq <= 0.0:
                raise ValueError(f"mode {index + 1}: frequency must be positive and finite, got {freq}")
        for index in range(1, len(freqs)):
            if freqs[index] <= freqs[index - 1]:
                raise ValueError(
                    f"frequencies must be strictly increasing, mode {index + 1} = {freqs[index]} Hz "
                    f"<= mode {index} = {freqs[index - 1]} Hz"
                )
        for index, zeta in enumerate(zetas):
            if not 0.0 <= zeta < 1.0:
                raise ValueError(f"mode {index + 1}: damping ratio must satisfy 0 <= zeta < 1, got {zeta}")
        for index, mass in enumerate(masses):
            if not math.isfinite(mass) or mass <= 0.0:
                raise ValueError(f"mode {index + 1}: modal mass must be positive and finite, got {mass}")
        object.__setattr__(self, "frequencies_hz", freqs)
        object.__setattr__(self, "damping_ratios", zetas)
        object.__setattr__(self, "modal_masses_kg", masses)
        object.__setattr__(self, "mode_shape_bow", tuple(float(v) for v in self.mode_shape_bow))
        object.__setattr__(
            self, "mode_shape_midship_curvature", tuple(float(v) for v in self.mode_shape_midship_curvature)
        )
        object.__setattr__(
            self, "mode_shape_quarter_curvature", tuple(float(v) for v in self.mode_shape_quarter_curvature)
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    # -- derived quantities -------------------------------------------------
    @property
    def n_modes(self) -> int:
        return len(self.frequencies_hz)

    def angular_frequencies(self) -> np.ndarray:
        return 2.0 * np.pi * np.asarray(self.frequencies_hz, dtype=float)

    def generalized_stiffness(self) -> np.ndarray:
        """k_n = m_n * omega_n^2 [N/m]."""
        wn = self.angular_frequencies()
        return np.asarray(self.modal_masses_kg, dtype=float) * wn**2

    def generalized_damping(self) -> np.ndarray:
        """c_n = 2 * zeta_n * m_n * omega_n [N*s/m]."""
        wn = self.angular_frequencies()
        return 2.0 * np.asarray(self.damping_ratios, dtype=float) * np.asarray(self.modal_masses_kg, dtype=float) * wn

    def truncated(self, n_modes: int) -> "ModalData":
        """Return the leading ``n_modes`` modes (for modal-truncation studies)."""
        if n_modes < 1 or n_modes > self.n_modes:
            raise ValueError(f"n_modes must be in [1, {self.n_modes}], got {n_modes}")
        return ModalData(
            frequencies_hz=self.frequencies_hz[:n_modes],
            damping_ratios=self.damping_ratios[:n_modes],
            modal_masses_kg=self.modal_masses_kg[:n_modes],
            mode_shape_bow=self.mode_shape_bow[:n_modes],
            mode_shape_midship_curvature=self.mode_shape_midship_curvature[:n_modes],
            mode_shape_quarter_curvature=self.mode_shape_quarter_curvature[:n_modes],
            source=self.source,
            metadata={**self.metadata, "truncated_to": n_modes},
        )

    def frequency_ratio(self, other: "ModalData") -> Tuple[float, ...]:
        """f_n(self) / f_n(other) per mode (verification helper)."""
        if other.n_modes < self.n_modes:
            raise ValueError("comparison modal set too short")
        return tuple(a / b for a, b in zip(self.frequencies_hz, other.frequencies_hz))

    def total_modal_mass(self) -> float:
        return float(sum(self.modal_masses_kg))

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "ModalData",
            "source": self.source,
            "frequencies_hz": list(self.frequencies_hz),
            "damping_ratios": list(self.damping_ratios),
            "modal_masses_kg": list(self.modal_masses_kg),
            "generalized_stiffness_n_m": list(self.generalized_stiffness()),
            "generalized_damping_n_s_m": list(self.generalized_damping()),
            "mode_shape_bow": list(self.mode_shape_bow),
            "mode_shape_midship_curvature_per_m": list(self.mode_shape_midship_curvature),
            "mode_shape_quarter_curvature_per_m": list(self.mode_shape_quarter_curvature),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Euler-Bernoulli reference beam (verification target, not a ship FE model)
# ---------------------------------------------------------------------------


def free_free_beam_mode_shape(beta_l: float, x: np.ndarray, length_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Unit-max normalized free-free beam mode shape and curvature.

    For a uniform free-free Euler-Bernoulli beam measured from one end,

        phi(x)      = cosh(bx) + cos(bx) - sigma (sinh(bx) + sin(bx))
        phi''(x)    = b^2 [cosh(bx) - cos(bx) - sigma (sinh(bx) - sin(bx))]
        sigma       = (cosh(bL) - cos(bL)) / (sinh(bL) - sin(bL))

    The shape is scaled so that ``max|phi| = 1`` (attained at the two ends).
    Returns ``(phi, phi_double_prime)`` with ``phi''`` in 1/m.
    """
    x = np.asarray(x, dtype=float)
    b = beta_l / length_m
    bl = beta_l
    sigma = (math.cosh(bl) - math.cos(bl)) / (math.sinh(bl) - math.sin(bl))
    bx = b * x
    phi = np.cosh(bx) + np.cos(bx) - sigma * (np.sinh(bx) + np.sin(bx))
    phi_pp = b**2 * (np.cosh(bx) - np.cos(bx) - sigma * (np.sinh(bx) - np.sin(bx)))
    scale = float(np.max(np.abs(phi)))
    if scale <= 0.0:
        raise ValueError("degenerate mode shape")
    return phi / scale, phi_pp / scale


def free_free_beam_frequency(beta_l: float, length_m: float, ei_nm2: float, mass_per_length: float) -> float:
    """Cyclic natural frequency of a uniform free-free Euler-Bernoulli beam [Hz].

    f = (beta L)^2 / (2 pi L^2) * sqrt(EI / m')
    """
    if min(length_m, ei_nm2, mass_per_length) <= 0.0:
        raise ValueError("length, EI and mass per length must all be positive")
    return beta_l**2 / (2.0 * math.pi * length_m**2) * math.sqrt(ei_nm2 / mass_per_length)


def cantilever_beam_frequency(beta_l: float, length_m: float, ei_nm2: float, mass_per_length: float) -> float:
    """Cyclic natural frequency of a uniform cantilever beam [Hz] (cross-check)."""
    return free_free_beam_frequency(beta_l, length_m, ei_nm2, mass_per_length)


def free_free_beam_modal_data(
    hull: HullParameters,
    bending_stiffness_nm2: float,
    n_modes: int = 2,
    damping_ratios: Iterable[float] = (0.02, 0.03),
    n_integration_points: int = 4001,
) -> ModalData:
    """Modal data of the equivalent uniform free-free hull girder.

    The hull girder is idealised as a uniform Euler-Bernoulli free-free beam of
    length ``L`` with mass per unit length ``Delta / L`` and bending stiffness
    ``EI``.  Modal masses follow from ``m_n = int_0^L m' phi_n^2 dx`` evaluated
    with the trapezoidal rule on a fine grid (the shapes are analytic, so the
    integral is quadrature-accurate).

    This is an *engineering reference* idealisation: it is used to verify the
    dynamics machinery and to provide a physically plausible modal set when no
    ship-specific FEM extraction is supplied.  When a real extraction exists,
    use :func:`modal_data_from_fem_extraction` instead.
    """
    if n_modes < 1 or n_modes > len(FREE_FREE_BETA_L):
        raise ValueError(f"n_modes must be in [1, {len(FREE_FREE_BETA_L)}]")
    zetas = tuple(damping_ratios)
    if len(zetas) < n_modes:
        raise ValueError(f"{n_modes} modes requested but only {len(zetas)} damping ratios given")
    ei = float(bending_stiffness_nm2)
    if ei <= 0.0:
        raise ValueError("bending stiffness EI must be positive")
    m_prime = hull.mass_per_length_kg_m
    x = np.linspace(0.0, hull.length_m, n_integration_points)
    x_mid = 0.5 * hull.length_m
    x_quarter = 0.25 * hull.length_m
    freqs, masses, phi_bow, phi_mid_pp, phi_q_pp = [], [], [], [], []
    for index in range(n_modes):
        beta_l = FREE_FREE_BETA_L[index]
        phi, phi_pp = free_free_beam_mode_shape(beta_l, x, hull.length_m)
        freq = free_free_beam_frequency(beta_l, hull.length_m, ei, m_prime)
        modal_mass = float(np.trapezoid(m_prime * phi**2, x))
        pp_mid = float(np.interp(x_mid, x, phi_pp))
        pp_quarter = float(np.interp(x_quarter, x, phi_pp))
        if modal_mass <= 0.0:
            raise ValueError(f"mode {index + 1}: non-positive modal mass {modal_mass}")
        freqs.append(freq)
        masses.append(modal_mass)
        # phi(L) is a mode-shape ordinate of order unity (max|phi| = 1 at the ends).
        phi_bow.append(float(phi[-1]))
        phi_mid_pp.append(pp_mid)
        phi_q_pp.append(pp_quarter)
    return ModalData(
        frequencies_hz=tuple(freqs),
        damping_ratios=zetas[:n_modes],
        modal_masses_kg=tuple(masses),
        mode_shape_bow=tuple(phi_bow),
        mode_shape_midship_curvature=tuple(phi_mid_pp),
        mode_shape_quarter_curvature=tuple(phi_q_pp),
        source="euler_bernoulli_reference",
        metadata={
            "model": "uniform free-free Euler-Bernoulli beam",
            "length_m": hull.length_m,
            "bending_stiffness_nm2": ei,
            "mass_per_length_kg_m": m_prime,
            "beta_l": list(FREE_FREE_BETA_L[:n_modes]),
            "hull_name": hull.name,
            "caveat": "engineering idealisation; not a ship-specific FE extraction",
        },
    )


# ---------------------------------------------------------------------------
# Damping bookkeeping
# ---------------------------------------------------------------------------


def rayleigh_coefficients(
    frequency_1_hz: float,
    damping_1: float,
    frequency_2_hz: float,
    damping_2: float,
) -> Tuple[float, float]:
    """Mass- and stiffness-proportional Rayleigh damping coefficients (alpha, beta).

    Solves the 2x2 system

        alpha/(2 w1) + beta*w1/2 = zeta_1
        alpha/(2 w2) + beta*w2/2 = zeta_2

    so that modes 1 and 2 reproduce the requested damping ratios exactly.
    """
    w1 = 2.0 * math.pi * float(frequency_1_hz)
    w2 = 2.0 * math.pi * float(frequency_2_hz)
    if not (0.0 < w1 < w2):
        raise ValueError("require 0 < frequency_1_hz < frequency_2_hz")
    if not (0.0 <= damping_1 < 1.0 and 0.0 <= damping_2 < 1.0):
        raise ValueError("damping ratios must satisfy 0 <= zeta < 1")
    a11, a12 = 1.0 / (2.0 * w1), w1 / 2.0
    a21, a22 = 1.0 / (2.0 * w2), w2 / 2.0
    alpha = (damping_1 * a22 - damping_2 * a12) / (a11 * a22 - a12 * a21)
    beta = (a11 * damping_2 - a21 * damping_1) / (a11 * a22 - a12 * a21)
    return float(alpha), float(beta)


# ---------------------------------------------------------------------------
# TensorFEM extraction interface
# ---------------------------------------------------------------------------


def modal_data_from_fem_extraction(payload: Mapping[str, Any]) -> ModalData:
    """Build :class:`ModalData` from a TensorFEM modal-extraction payload.

    Expected schema (all lengths SI)::

        {
          "solver": "TensorFEM",
          "modes": [
            {"frequency_hz": 1.21, "damping_ratio": 0.02,
             "modal_mass_kg": 8.1e6,
             "mode_shape_bow": 1.0,
             "midship_curvature_per_m": 1.02e-3},
            ...
          ],
          "meta": {...}            # optional passthrough
        }

    ``modal_mass_kg`` is mandatory: modal masses depend on the mass matrix and
    on the shape normalization used by the eigensolver, so they cannot be
    guessed here.  ``damping_ratio`` falls back to ``1e-2`` when omitted, with
    the fallback flagged in ``metadata``.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("FEM extraction payload must be a mapping")
    modes = payload.get("modes")
    if not isinstance(modes, Sequence) or not modes:
        raise ValueError("FEM extraction payload must contain a non-empty 'modes' list")
    freqs, zetas, masses, phi_bow, phi_pp, phi_q_pp = [], [], [], [], [], []
    defaults_used = 0
    for index, mode in enumerate(modes):
        if not isinstance(mode, Mapping):
            raise TypeError(f"mode {index + 1} must be a mapping")
        if "frequency_hz" not in mode:
            raise ValueError(f"mode {index + 1}: missing 'frequency_hz'")
        if "modal_mass_kg" not in mode:
            raise ValueError(
                f"mode {index + 1}: missing 'modal_mass_kg' (modal mass cannot be inferred from a shape alone)"
            )
        freqs.append(float(mode["frequency_hz"]))
        masses.append(float(mode["modal_mass_kg"]))
        if "damping_ratio" in mode and mode["damping_ratio"] is not None:
            zetas.append(float(mode["damping_ratio"]))
        else:
            zetas.append(1.0e-2)
            defaults_used += 1
        phi_bow.append(float(mode.get("mode_shape_bow", 1.0)))
        phi_pp.append(float(mode.get("midship_curvature_per_m", float("nan"))))
        phi_q_pp.append(float(mode.get("quarter_curvature_per_m", 0.0)))
    data = ModalData(
        frequencies_hz=tuple(freqs),
        damping_ratios=tuple(zetas),
        modal_masses_kg=tuple(masses),
        mode_shape_bow=tuple(phi_bow),
        mode_shape_midship_curvature=tuple(phi_pp),
        mode_shape_quarter_curvature=tuple(phi_q_pp),
        source="fem",
        metadata={
            "solver": payload.get("solver", "unknown"),
            "damping_ratio_defaults_used": defaults_used,
            "upstream_meta": dict(payload.get("meta", {})),
        },
    )
    return data