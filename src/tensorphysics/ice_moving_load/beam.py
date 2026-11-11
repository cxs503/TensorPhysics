"""Euler-Bernoulli hull-girder beam, modal basis and an independent FE eigen check.

The structural side of the moving-load problem is a uniform Euler-Bernoulli
beam of length ``L``, bending stiffness ``EI`` and mass per unit length
``rho*A`` (added mass folded in by :attr:`GirderBeam.added_mass_factor`):

    EI w'''' + rho*A w'' = p(x, t)

Two support idealisations are provided and kept strictly separate, because
different verification references belong to them:

``simply_supported``
    ``w(0) = w(L) = w''(0) = w''(L) = 0``, mode shapes ``sin(n pi x / L)``,
    ``omega_n = (n pi / L)^2 sqrt(EI / rho A)``.  This is the classical
    reference configuration of the moving-load literature (Timoshenko) and the
    only configuration in which the static midspan deflection ``P L^3 / (48 EI)``
    and the closed-form moving-load solution are valid.
``free_free``
    the physically relevant configuration of a floating hull girder (no
    end supports; rigid-body modes are excluded from the elastic response).
    Frequencies follow ``omega_n = (beta_n L)^2 / L^2 * sqrt(EI / rho A)`` with
    the roots of ``cosh(bL) cos(bL) = 1``.  The shapes are taken from the
    already verified :mod:`tensorphysics.ice_vibration.modal` helpers instead of
    being re-derived here.

Both bases return the same :class:`ModalBasis` container (angular
frequencies, generalized masses, generalized stiffnesses and a vectorised
shape evaluator), so every downstream analysis is support-agnostic.

An independent numerical eigen-solver (:func:`beam_fem_frequencies`) using
Hermite cubic beam finite elements with the consistent mass matrix is provided
purely as a cross-check of the analytic frequencies - it is *not* the
production path.

Units: SI throughout (m, kg, s, N, Pa).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np

from ..ice_vibration.modal import (
    FREE_FREE_BETA_L,
    HullParameters,
    free_free_beam_mode_shape,
)

__all__ = [
    "GirderBeam",
    "ModalBasis",
    "SIMPLY_SUPPORTED",
    "FREE_FREE",
    "beam_fem_frequencies",
    "static_midspan_deflection_analytic",
    "static_midspan_deflection_modal",
    "classic_critical_speed",
]

SIMPLY_SUPPORTED = "simply_supported"
FREE_FREE = "free_free"

#: Free-free Euler-Bernoulli roots ``cosh(bL)cos(bL) = 1``.  For ``n >= 5`` the
#: asymptotic value ``(2n+1) pi / 2`` is used (accurate to 1e-4 at n = 5 and
#: better further out).
_FREE_FREE_EXTENDED: tuple = FREE_FREE_BETA_L + tuple(
    (2.0 * n + 1.0) * math.pi / 2.0 for n in range(len(FREE_FREE_BETA_L) + 1, 41)
)


# ---------------------------------------------------------------------------
# Beam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GirderBeam:
    """Uniform Euler-Bernoulli hull-girder beam.

    Parameters
    ----------
    length_m:
        Length between perpendiculars [m].
    bending_stiffness_nm2:
        Vertical bending stiffness ``EI`` about the transverse axis [N m^2].
    mass_per_length_kg_m:
        Structural + outfit mass per unit length [kg/m]; use
        :attr:`HullParameters.mass_per_length_kg_m` for the rigid-body
        equivalent.
    added_mass_factor:
        Multiplier on ``rho*A`` accounting for the entrained water mass.  The
        default of 1.0 keeps the *dry* structural beam; a value of 1.2 - 1.4 is
        the usual hull-girder range in vertical vibration and shifts the
        natural frequencies by ``1/sqrt(factor)``.
    """

    length_m: float
    bending_stiffness_nm2: float
    mass_per_length_kg_m: float
    name: str = "generic-hull-girder"
    added_mass_factor: float = 1.0

    def __post_init__(self) -> None:
        for name in ("length_m", "bending_stiffness_nm2", "mass_per_length_kg_m", "added_mass_factor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"GirderBeam.{name} must be a positive finite number, got {value!r}")

    # -- derived quantities -------------------------------------------------
    @property
    def effective_mass_per_length_kg_m(self) -> float:
        return self.mass_per_length_kg_m * self.added_mass_factor

    @property
    def total_mass_kg(self) -> float:
        return self.effective_mass_per_length_kg_m * self.length_m

    @property
    def bending_wave_speed_m_s(self) -> float:
        """``sqrt(EI / (rho A))``, the scale factor of the dispersion relation."""
        return math.sqrt(self.bending_stiffness_nm2 / self.effective_mass_per_length_kg_m)

    def verify(self) -> Dict[str, Any]:
        """Euler-Bernoulli validity record (JSON friendly).

        The Euler-Bernoulli kinematic assumption requires the bending
        wavelength to be long compared with the girder depth.  The first
        elastic free-free wavelength is ``2 L``; its ratio to a nominal girder
        depth of ``L/15`` is reported as ``slenderness``.  Values below ~10
        flag that shear deformation / rotary inertia matter.
        """
        slenderness = 2.0 * self.length_m / (self.length_m / 15.0)
        return {
            "class": "GirderBeam",
            "name": self.name,
            "length_m": self.length_m,
            "bending_stiffness_nm2": self.bending_stiffness_nm2,
            "mass_per_length_kg_m": self.mass_per_length_kg_m,
            "added_mass_factor": self.added_mass_factor,
            "effective_mass_per_length_kg_m": self.effective_mass_per_length_kg_m,
            "bending_wave_speed_m_s": self.bending_wave_speed_m_s,
            "first_wavelength_slenderness": slenderness,
            "euler_bernoulli_applicable": bool(slenderness > 10.0),
        }

    # -- analytic frequencies ----------------------------------------------
    def free_free_frequencies_hz(self, n_modes: int) -> np.ndarray:
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if n_modes > len(_FREE_FREE_EXTENDED):
            raise ValueError(f"n_modes must be <= {len(_FREE_FREE_EXTENDED)}")
        c = self.bending_wave_speed_m_s
        omega = np.array(
            [(bl / self.length_m) ** 2 * c for bl in _FREE_FREE_EXTENDED[:n_modes]], dtype=float
        )
        return omega / (2.0 * math.pi)

    def simply_supported_frequencies_hz(self, n_modes: int) -> np.ndarray:
        n = np.arange(1, n_modes + 1, dtype=float)
        omega = (n * math.pi / self.length_m) ** 2 * self.bending_wave_speed_m_s
        return omega / (2.0 * math.pi)

    def frequencies_hz(self, n_modes: int, support: str = SIMPLY_SUPPORTED) -> np.ndarray:
        if support == SIMPLY_SUPPORTED:
            return self.simply_supported_frequencies_hz(n_modes)
        if support == FREE_FREE:
            return self.free_free_frequencies_hz(n_modes)
        raise ValueError(f"unknown support condition {support!r}")

    # -- modal basis --------------------------------------------------------
    def modal_basis(self, support: str = SIMPLY_SUPPORTED, n_modes: int = 10) -> "ModalBasis":
        return ModalBasis(beam=self, support=support, n_modes=n_modes)


# ---------------------------------------------------------------------------
# Modal basis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModalBasis:
    """Truncated modal basis of a :class:`GirderBeam`.

    Shapes are normalised to unit maximum absolute ordinate, so a generalized
    coordinate has the units of a physical displacement at the shape maximum.
    """

    beam: GirderBeam
    support: str = SIMPLY_SUPPORTED
    n_modes: int = 10
    _quadrature_points: int = 4001
    _x: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.support not in (SIMPLY_SUPPORTED, FREE_FREE):
            raise ValueError(f"unknown support condition {self.support!r}")
        if self.n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        object.__setattr__(self, "_x", np.linspace(0.0, self.beam.length_m, self._quadrature_points))

    # -- modes -------------------------------------------------------------
    @property
    def mode_numbers(self) -> np.ndarray:
        return np.arange(1, self.n_modes + 1, dtype=float)

    @property
    def omega_rad_s(self) -> np.ndarray:
        return 2.0 * math.pi * self.beam.frequencies_hz(self.n_modes, self.support)

    @property
    def frequencies_hz(self) -> np.ndarray:
        return self.beam.frequencies_hz(self.n_modes, self.support)

    def shapes(self, x: Sequence[float] | np.ndarray | None = None) -> np.ndarray:
        """Mode shapes evaluated at ``x``; returns an ``(n_modes, n_points)`` array."""
        xa = self._x if x is None else np.atleast_1d(np.asarray(x, dtype=float))
        if self.support == SIMPLY_SUPPORTED:
            n = self.mode_numbers[:, None]
            return np.sin(n * math.pi * xa[None, :] / self.beam.length_m)
        out = np.empty((self.n_modes, xa.size))
        for i in range(self.n_modes):
            phi, _ = free_free_beam_mode_shape(_FREE_FREE_EXTENDED[i], xa, self.beam.length_m)
            out[i] = phi
        return out

    def shape(self, mode: int, x: float) -> float:
        """Mode shape ``phi_mode(x)`` for a scalar argument (``mode`` is 1-based)."""
        if not 1 <= mode <= self.n_modes:
            raise ValueError(f"mode must be in [1, {self.n_modes}]")
        return float(self.shapes([x])[mode - 1, 0])

    def curvatures(self, x: Sequence[float] | np.ndarray | None = None) -> np.ndarray:
        """Mode-shape second derivatives ``phi_n''(x)`` [1/m]."""
        xa = self._x if x is None else np.atleast_1d(np.asarray(x, dtype=float))
        if self.support == SIMPLY_SUPPORTED:
            n = self.mode_numbers[:, None]
            scale = -(n * math.pi / self.beam.length_m) ** 2
            return scale * np.sin(n * math.pi * xa[None, :] / self.beam.length_m)
        out = np.empty((self.n_modes, xa.size))
        for i in range(self.n_modes):
            _, phi_pp = free_free_beam_mode_shape(_FREE_FREE_EXTENDED[i], xa, self.beam.length_m)
            out[i] = phi_pp
        return out

    @property
    def modal_mass_kg(self) -> np.ndarray:
        phi = self.shapes()
        m_prime = self.beam.effective_mass_per_length_kg_m
        return np.array([np.trapezoid(m_prime * row**2, self._x) for row in phi], dtype=float)

    @property
    def modal_stiffness_nm(self) -> np.ndarray:
        """``k_n = omega_n^2 m_n`` (exact for the uniform beam, both supports)."""
        return self.modal_mass_kg * self.omega_rad_s**2

    @property
    def modal_damping_ns_m(self) -> np.ndarray:
        """Placeholder: damping is supplied by the caller (see ``moving_load``)."""
        raise AttributeError("modal damping depends on the chosen damping ratios; see moving_load")

    def damping(self, zeta: float | Sequence[float]) -> np.ndarray:
        """Diagonal modal damping matrix ``c_n = 2 zeta_n omega_n m_n`` [N s/m]."""
        zetas = np.full(self.n_modes, float(zeta)) if np.isscalar(zeta) else np.asarray(zeta, dtype=float)
        if zetas.shape != (self.n_modes,):
            raise ValueError(f"need {self.n_modes} damping ratios, got shape {zetas.shape}")
        if np.any(zetas < 0.0):
            raise ValueError("damping ratios must be non-negative")
        return 2.0 * zetas * self.omega_rad_s * self.modal_mass_kg

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "ModalBasis",
            "support": self.support,
            "n_modes": self.n_modes,
            "frequencies_hz": self.frequencies_hz.tolist(),
            "modal_masses_kg": self.modal_mass_kg.tolist(),
            "modal_stiffnesses_nm": self.modal_stiffness_nm.tolist(),
            "beam": self.beam.verify(),
        }


# ---------------------------------------------------------------------------
# Static reference of a simply supported beam
# ---------------------------------------------------------------------------


def static_midspan_deflection_analytic(force_n: float, beam: GirderBeam) -> float:
    """Midspan deflection of a *simply supported* beam under a central point load.

    ``w = P L^3 / (48 EI)`` - the closed-form reference used by acceptance
    criterion A1.
    """
    if force_n < 0.0:
        raise ValueError("force must be non-negative")
    return force_n * beam.length_m**3 / (48.0 * beam.bending_stiffness_nm2)


def static_midspan_deflection_modal(
    force_n: float, beam: GirderBeam, n_modes: int = 10, support: str = SIMPLY_SUPPORTED
) -> float:
    """Midspan deflection from modal superposition of a central point load.

    ``w(L/2) = sum_n Q_n phi_n(L/2) / k_n`` with ``Q_n = P phi_n(L/2)``.  For the
    simply supported beam the series is ``(2 P L^3 / (pi^4 EI)) sum_{n odd} 1/n^4``,
    i.e. the exact ``P L^3 / (48 EI)`` at infinite ``n``; ten modes already give
    0.2 %.
    """
    basis = beam.modal_basis(support=support, n_modes=n_modes)
    x_mid = 0.5 * beam.length_m
    phi_mid = basis.shapes([x_mid])[:, 0]
    k_n = basis.modal_stiffness_nm
    return float(np.sum(force_n * phi_mid**2 / k_n))


def classic_critical_speed(beam: GirderBeam, mode: int = 1, support: str = SIMPLY_SUPPORTED) -> float:
    """Classical single-force moving-load critical speed ``V_cr = omega_n L / (n pi)`` [m/s].

    At this speed a *constant* force traversing a simply supported beam excites
    mode ``n`` in resonance.  For a full-scale hull girder the value is of the
    order of hundreds of m/s (see the report): it is a mechanics reference, not
    an operational ship speed.
    """
    basis = beam.modal_basis(support=support, n_modes=mode)
    return float(basis.omega_rad_s[mode - 1] * beam.length_m / (mode * math.pi))


# ---------------------------------------------------------------------------
# Independent finite-element eigen cross-check
# ---------------------------------------------------------------------------


def beam_fem_frequencies(
    beam: GirderBeam,
    n_elem: int = 60,
    support: str = FREE_FREE,
    n_modes: int = 3,
) -> np.ndarray:
    """Eigenfrequencies [Hz] of a Hermite cubic beam FE model (cross-check only).

    Standard two-node element with the exact Euler-Bernoulli stiffness matrix
    and the consistent mass matrix::

        Ke = EI/le^3 [[12, 6le, -12, 6le], [6le, 4le^2, -6le, 2le^2],
                      [-12, -6le, 12, -6le], [6le, 2le^2, -6le, 4le^2]]
        Me = rho A le/420 [[156, 22le, 54, -13le], [22le, 4le^2, 13le, -3le^2],
                           [54, 13le, 156, -22le], [-13le, -3le^2, -22le, 4le^2]]

    For the free-free case the six rigid-body modes are removed before the
    elastic eigenvalues are returned.  This path shares no code with the
    analytic formulas, which is why it is used as verification.
    """
    if n_elem < 4:
        raise ValueError("n_elem must be >= 4")
    le = beam.length_m / n_elem
    ei = beam.bending_stiffness_nm2
    mpl = beam.effective_mass_per_length_kg_m
    ke = ei / le**3 * np.array(
        [
            [12.0, 6 * le, -12.0, 6 * le],
            [6 * le, 4 * le**2, -6 * le, 2 * le**2],
            [-12.0, -6 * le, 12.0, -6 * le],
            [6 * le, 2 * le**2, -6 * le, 4 * le**2],
        ]
    )
    me = (
        mpl
        * le
        / 420.0
        * np.array(
            [
                [156.0, 22 * le, 54.0, -13 * le],
                [22 * le, 4 * le**2, 13 * le, -3 * le**2],
                [54.0, 13 * le, 156.0, -22 * le],
                [-13 * le, -3 * le**2, -22 * le, 4 * le**2],
            ]
        )
    )
    n_dof = 2 * (n_elem + 1)
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))
    for e in range(n_elem):
        sl = slice(2 * e, 2 * e + 4)
        K[sl, sl] += ke
        M[sl, sl] += me
    if support == SIMPLY_SUPPORTED:
        keep = np.ones(n_dof, dtype=bool)
        keep[0] = False
        keep[n_dof - 2] = False
        K = K[np.ix_(keep, keep)]
        M = M[np.ix_(keep, keep)]
    elif support != FREE_FREE:
        raise ValueError(f"unknown support condition {support!r}")
    # Symmetric generalised eigenproblem K phi = lam M phi with M positive
    # definite.  Whitening by the Cholesky factor of M keeps the problem
    # symmetric and needs nothing beyond NumPy (the LAPACK generalised driver
    # shipped with this image aborts on matrices of this size; the whitened
    # standard problem is also more transparent).
    chol = np.linalg.cholesky(M)
    whitened = np.linalg.solve(chol, np.linalg.solve(chol, K).T)
    lam = np.sort(np.linalg.eigvalsh(0.5 * (whitened + whitened.T)))
    lam = lam[np.abs(lam) > 1e-9 * max(abs(lam[-1]), 1.0)]
    omega = np.sqrt(np.abs(lam))
    return omega[:n_modes] / (2.0 * math.pi)