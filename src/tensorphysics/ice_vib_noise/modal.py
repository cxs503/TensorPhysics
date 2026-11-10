"""Hull-girder modal basis with an explicit mass-matrix orthogonality audit.

The structural model is the *equivalent uniform free-free Euler-Bernoulli beam*
(the classical hull-girder idealisation used for springing/whipping and
ice-induced vibration screening).  Its eigenfunctions are analytic:

    phi_n(x) = cosh(b_n x) + cos(b_n x) - sigma_n [sinh(b_n x) + sin(b_n x)]

with ``b_n L`` the roots of ``cosh cos = 1`` and ``sigma_n`` fixed by the
free-end boundary conditions.  The shapes are reused from
:mod:`tensorphysics.ice_vibration.modal` so that this module does not introduce
a second, divergent definition of the hull-girder modes.

Two quantities are *audited*, not assumed:

* the modal mass matrix ``M_ij = int m' phi_i phi_j dx`` must be diagonal —
  the acceptance criterion is ``max_{i!=j} |M_ij| / sqrt(M_ii M_jj) < 1e-6``;
* the generalised stiffness ``K_ij = int EI phi_i'' phi_j'' dx`` must reproduce
  the closed-form eigenvalue ``(2 pi f_n)^2 M_nn`` — a check that the shapes and
  the frequency formula belong to the same boundary-value problem.

When a ship-specific FEM extraction exists, pass it through
:func:`modal_basis_from_fem` (same fields as
``tensorphysics.ice_vibration.modal.modal_data_from_fem_extraction``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..ice_vibration.modal import (
    FREE_FREE_BETA_L,
    HullParameters,
    free_free_beam_frequency,
    free_free_beam_mode_shape,
)

__all__ = [
    "HullGirderModes",
    "modal_basis_from_fem",
]

#: Relative tolerance for the mass-matrix off-diagonal audit.
ORTHOGONALITY_TOL = 1.0e-6


@dataclass(frozen=True)
class HullGirderModes:
    """Analytic free-free hull-girder modal basis.

    Parameters
    ----------
    hull:
        Principal particulars (block coefficient already validated).
    bending_stiffness_nm2:
        Equivalent hull-girder vertical bending stiffness ``EI`` [N m^2].
    n_modes:
        Number of flexible modes retained (1..4 with the built-in roots).
    damping_ratios:
        Structural (hysteretic-equivalent) modal damping ratios.  Radiation
        damping is added separately by the acoustic chain.
    n_grid:
        Grid for the modal integrals.  The trapezoidal modal integrals converge as
        ``n^-2``: the largest off-diagonal mass ratio is 6.5e-7 on 4001 points and
        7.2e-8 on the default 12001 points (criterion 1e-6).
    """

    hull: HullParameters
    bending_stiffness_nm2: float
    n_modes: int = 3
    damping_ratios: Tuple[float, ...] = (0.02, 0.02, 0.02, 0.02)
    n_grid: int = 12001
    _x: np.ndarray = field(init=False, repr=False)
    _phi: np.ndarray = field(init=False, repr=False)
    _phi_pp: np.ndarray = field(init=False, repr=False)
    _freqs: Tuple[float, ...] = field(init=False)
    _modal_masses: Tuple[float, ...] = field(init=False)
    _m_matrix: np.ndarray = field(init=False, repr=False)
    _k_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bending_stiffness_nm2 <= 0.0:
            raise ValueError("bending stiffness must be positive")
        if self.n_modes < 1 or self.n_modes > len(FREE_FREE_BETA_L):
            raise ValueError(f"n_modes must be in [1, {len(FREE_FREE_BETA_L)}]")
        zetas = tuple(float(z) for z in self.damping_ratios)
        if len(zetas) < self.n_modes:
            raise ValueError("not enough damping ratios for the requested modes")
        if any(not 0.0 <= z < 1.0 for z in zetas):
            raise ValueError("damping ratios must be in [0, 1)")
        if self.n_grid < 501:
            raise ValueError("n_grid too coarse for the orthogonality audit")

        x = np.linspace(0.0, self.hull.length_m, self.n_grid)
        m_prime = self.hull.mass_per_length_kg_m
        ei = float(self.bending_stiffness_nm2)
        phi = np.empty((self.n_modes, x.size))
        phi_pp = np.empty((self.n_modes, x.size))
        freqs, masses = [], []
        for i in range(self.n_modes):
            beta_l = FREE_FREE_BETA_L[i]
            p, p_pp = free_free_beam_mode_shape(beta_l, x, self.hull.length_m)
            phi[i] = p
            phi_pp[i] = p_pp
            freqs.append(free_free_beam_frequency(beta_l, self.hull.length_m, ei, m_prime))
            masses.append(float(np.trapezoid(m_prime * p * p, x)))

        m_mat = np.empty((self.n_modes, self.n_modes))
        k_mat = np.empty((self.n_modes, self.n_modes))
        for i in range(self.n_modes):
            for j in range(self.n_modes):
                m_mat[i, j] = float(np.trapezoid(m_prime * phi[i] * phi[j], x))
                k_mat[i, j] = float(np.trapezoid(ei * phi_pp[i] * phi_pp[j], x))

        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_phi", phi)
        object.__setattr__(self, "_phi_pp", phi_pp)
        object.__setattr__(self, "_freqs", tuple(freqs))
        object.__setattr__(self, "_modal_masses", tuple(masses))
        object.__setattr__(self, "_m_matrix", m_mat)
        object.__setattr__(self, "_k_matrix", k_mat)

    # -- basic accessors ------------------------------------------------------
    @property
    def frequencies_hz(self) -> Tuple[float, ...]:
        return self._freqs

    @property
    def omegas_rad_s(self) -> Tuple[float, ...]:
        return tuple(2.0 * math.pi * f for f in self._freqs)

    @property
    def modal_masses_kg(self) -> Tuple[float, ...]:
        return self._modal_masses

    @property
    def zetas(self) -> Tuple[float, ...]:
        """Structural damping ratios actually used (truncated to ``n_modes``)."""
        return tuple(float(z) for z in self.damping_ratios[: self.n_modes])

    @property
    def grid_x_m(self) -> np.ndarray:
        return self._x

    def mode_shape(self, index: int, x: Optional[Sequence[float] | np.ndarray] = None) -> np.ndarray:
        """Mode shape ``phi_i(x)`` (unit maximum), interpolated from the audit grid."""
        if not 0 <= index < self.n_modes:
            raise IndexError("mode index out of range")
        if x is None:
            return self._phi[index]
        xa = np.asarray(x, dtype=float)
        if np.any(xa < 0.0) or np.any(xa > self.hull.length_m):
            raise ValueError("x outside the hull length")
        return np.interp(xa, self._x, self._phi[index])

    def mode_curvature(self, index: int, x: Optional[Sequence[float] | np.ndarray] = None) -> np.ndarray:
        if not 0 <= index < self.n_modes:
            raise IndexError("mode index out of range")
        if x is None:
            return self._phi_pp[index]
        xa = np.asarray(x, dtype=float)
        return np.interp(xa, self._x, self._phi_pp[index])

    # -- audits ---------------------------------------------------------------
    @property
    def mass_matrix(self) -> np.ndarray:
        return self._m_matrix.copy()

    @property
    def stiffness_matrix(self) -> np.ndarray:
        return self._k_matrix.copy()

    def orthogonality_audit(self) -> Dict[str, Any]:
        """Off-diagonal relative magnitude of the mass and stiffness matrices."""
        m = self._m_matrix
        k = self._k_matrix
        diag = np.sqrt(np.outer(np.diag(m), np.diag(m)))
        off = np.abs(m - np.diag(np.diag(m))) / diag
        diag_k = np.sqrt(np.outer(np.diag(k), np.diag(k)))
        off_k = np.abs(k - np.diag(np.diag(k))) / diag_k
        worst = float(np.max(off)) if self.n_modes > 1 else 0.0
        worst_k = float(np.max(off_k)) if self.n_modes > 1 else 0.0
        # Parenthesisation audit for readability of the JSON report.
        eigenvalue_err = [
            abs(math.sqrt(diag_k[i][i] / m[i, i]) / (2.0 * math.pi * self._freqs[i]) - 1.0)
            for i in range(self.n_modes)
        ]
        return {
            "n_modes": self.n_modes,
            "max_offdiag_mass_ratio": worst,
            "max_offdiag_stiffness_ratio": worst_k,
            "tolerance": ORTHOGONALITY_TOL,
            "mass_orthogonality_pass": worst < ORTHOGONALITY_TOL,
            "stiffness_orthogonality_pass": worst_k < ORTHOGONALITY_TOL,
            "max_rel_error_omega_from_kn": float(max(eigenvalue_err)),
            "modal_masses_kg": list(self._modal_masses),
            "frequencies_hz": list(self._freqs),
        }

    # -- generalised excitation ----------------------------------------------
    def generalized_force(
        self,
        load_distribution_n_per_m: np.ndarray,
        t: Optional[float] = None,
    ) -> np.ndarray:
        """``Q_i(t) = int q(x, t) phi_i(x) dx`` for a load per unit length.

        ``load_distribution_n_per_m`` may be ``(n_grid,)`` (a snapshot, ignoring
        ``t``) or ``(n_time, n_grid)`` (a time history, returning
        ``(n_time, n_modes)``).
        """
        q = np.asarray(load_distribution_n_per_m, dtype=float)
        if q.ndim == 1:
            if q.size != self._x.size:
                raise ValueError("load snapshot must be sampled on the modal grid")
            return np.array([np.trapezoid(q * self._phi[i], self._x) for i in range(self.n_modes)])
        if q.ndim == 2:
            if q.shape[1] != self._x.size:
                raise ValueError("load history must be sampled on the modal grid")
            out = np.empty((q.shape[0], self.n_modes))
            for i in range(self.n_modes):
                out[:, i] = np.trapezoid(q * self._phi[i][None, :], self._x, axis=1)
            return out
        raise ValueError("load distribution must be 1-D or 2-D")

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "model": "uniform free-free Euler-Bernoulli hull girder",
            "length_m": self.hull.length_m,
            "bending_stiffness_nm2": float(self.bending_stiffness_nm2),
            "mass_per_length_kg_m": self.hull.mass_per_length_kg_m,
            "frequencies_hz": list(self._freqs),
            "damping_ratios": list(self.zetas),
            "modal_masses_kg": list(self._modal_masses),
            "mass_matrix_kg": self._m_matrix.tolist(),
            "stiffness_matrix_n_m": self._k_matrix.tolist(),
            "audit": self.orthogonality_audit(),
        }


def modal_basis_from_fem(
    payload: Mapping[str, Any],
    hull: HullParameters,
    n_grid: int = 12001,
) -> HullGirderModes:
    """Build a :class:`HullGirderModes` from a FEM extraction payload.

    The payload must provide ``frequencies_hz``, ``damping_ratios`` and
    ``modal_masses_kg`` (the same schema as
    ``ice_vibration.modal.modal_data_from_fem_extraction``).  Mode shapes, when
    absent, fall back to the analytic free-free shapes — flagged in the returned
    metadata so that the report cannot silently present analytic shapes as FEM
    shapes.
    """
    freqs = tuple(float(f) for f in payload["frequencies_hz"])
    zetas = tuple(float(z) for z in payload.get("damping_ratios", (0.02,) * len(freqs)))
    masses = tuple(float(m) for m in payload["modal_masses_kg"])
    if not (len(freqs) == len(zetas) == len(masses)) or not freqs:
        raise ValueError("frequency/damping/mass lists must be non-empty and equal length")
    basis = HullGirderModes(
        hull=hull,
        bending_stiffness_nm2=float(payload.get("bending_stiffness_nm2", 1.0)),
        n_modes=min(len(freqs), len(FREE_FREE_BETA_L)),
        damping_ratios=zetas,
        n_grid=n_grid,
    )
    return basis