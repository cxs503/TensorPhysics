"""Free-free Euler-Bernoulli hull-girder modal superposition (m5528 port).

Provenance
----------
Faithful port of ``backend/modules/ice_fatigue/beam.py`` (node IN32-N07,
mission m5528): free-free Euler-Bernoulli (EB) mode shapes with the analytic
``beta_L`` roots, inconsistent-but-mass-consistent modal mass
``m_n = m' * int(phi_n^2 dx)``, modal SDOF chain discretised with a
zero-order-hold and evaluated with ``scipy.signal.lfilter``, quasi-static
response from the modal static limit ``q_n = Q_n / (omega_n^2 m_n)``, stress
recovery ``sigma = -E z sum_k q_k phi_k''(x_detail)``, and the m5528 whipping
share definition.

Added on top of the port (used by the ice+wave combination study of m7361)
--------------------------------------------------------------------------
* :meth:`BeamModes.stress_transfer_function` -- complex transfer function
  ``T(f)`` from *distributed* wave load (per unit length, per unit wave
  amplitude) to the detail stress, so the wave-induced stress PSD is obtained
  with the *same* structural model as the ice-induced time history.
* :func:`quasi_static_moment_free_free` is kept for the static moment audit.

Units: SI (m, N, Pa, s).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import signal

__all__ = [
    "FREE_FREE_BETA_L",
    "free_free_beta_roots",
    "BeamModes",
    "modal_stress_history",
    "whipping_share",
    "quasi_static_moment_free_free",
]

#: First ten roots of ``cosh(beta L) cos(beta L) = 1`` (free-free beam).
FREE_FREE_BETA_L = np.array(
    [4.730040744862704, 7.853204624095838, 10.995607838001671, 14.137165491257464,
     17.278759657399481, 20.420352248333657, 23.561944902040446, 26.703537555508192,
     29.845130209103256, 32.986722862692826]
)


def free_free_beta_roots(n: int, length: float) -> np.ndarray:
    """beta_1..beta_n [1/m] of the free-free EB beam (rigid modes excluded)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n <= FREE_FREE_BETA_L.size:
        return FREE_FREE_BETA_L[:n] / float(length)
    extra = (2.0 * np.arange(FREE_FREE_BETA_L.size, n) + 1.0) * math.pi / 2.0
    return np.concatenate([FREE_FREE_BETA_L, extra]) / float(length)


@dataclass
class BeamModes:
    """Modal basis, natural frequencies and modal masses of the hull girder."""

    length: float
    EI: float
    mass_per_length: float
    n_modes: int = 10
    bc: str = "free_free"
    damping_ratio: float = 0.015
    _n_quad: int = field(default=2001, repr=False)

    def __post_init__(self) -> None:
        if self.length <= 0 or self.EI <= 0 or self.mass_per_length <= 0:
            raise ValueError("length, EI and mass_per_length must be positive")
        self.L = float(self.length)
        self.beta = self._betas()
        self.omega = self.beta ** 2 * math.sqrt(self.EI / self.mass_per_length)
        self.freq = self.omega / (2.0 * math.pi)
        xs = np.linspace(0.0, self.L, self._n_quad)
        self.modal_mass = np.array(
            [self.mass_per_length * float(np.trapezoid(self.shape(xs, k) ** 2, xs))
             for k in range(self.n_modes)]
        )
        self.zeta = np.full(self.n_modes, float(self.damping_ratio))
        #: modal stiffness k_n = omega_n^2 m_n
        self.modal_stiffness = self.omega ** 2 * self.modal_mass

    # -- basis --------------------------------------------------------------
    def _betas(self) -> np.ndarray:
        if self.bc == "free_free":
            return free_free_beta_roots(self.n_modes, self.L)
        if self.bc == "simply_supported":
            return np.arange(1, self.n_modes + 1) * math.pi / self.L
        raise ValueError(f"unsupported bc: {self.bc}")

    def _sigma(self, k: int) -> float:
        b = self.beta[k]
        return (math.cosh(b * self.L) - math.cos(b * self.L)) / (
            math.sinh(b * self.L) - math.sin(b * self.L)
        )

    def shape(self, x, k: int):
        x = np.asarray(x, dtype=float)
        b = self.beta[k]
        if self.bc == "simply_supported":
            return np.sin(b * x)
        s = self._sigma(k)
        return np.cosh(b * x) + np.cos(b * x) - s * (np.sinh(b * x) + np.sin(b * x))

    def curvature_shape(self, x, k: int):
        x = np.asarray(x, dtype=float)
        b = self.beta[k]
        if self.bc == "simply_supported":
            return -(b ** 2) * np.sin(b * x)
        s = self._sigma(k)
        return b ** 2 * (np.cosh(b * x) - np.cos(b * x) - s * (np.sinh(b * x) - np.sin(b * x)))

    # -- stress recovery ----------------------------------------------------
    def stress_from_modal(self, q: np.ndarray, x_detail: float, z: float,
                          E: float) -> np.ndarray:
        kap = np.zeros(q.shape[1])
        for k in range(q.shape[0]):
            kap += q[k] * float(self.curvature_shape(x_detail, k))
        return -E * z * kap

    def modal_daf(self, freqs_hz, zeta: Optional[float] = None) -> np.ndarray:
        """Dynamic amplification ``|1 - r^2 + 2 i zeta r|^-1`` per mode."""
        f = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
        zt = self.zeta if zeta is None else np.full(self.n_modes, float(zeta))
        out = np.empty((self.n_modes, f.size), dtype=complex)
        for k in range(self.n_modes):
            r = f / self.freq[k]
            out[k] = 1.0 / (1.0 - r ** 2 + 2j * zt[k] * r)
        return out

    def stress_transfer_function(
        self,
        freqs_hz,
        load_shape,
        x_detail: float,
        z: float,
        E: float,
        x_quad: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Complex transfer function from a *distributed* load to detail stress.

        ``load_shape(x, omega)`` returns the complex load per unit length
        [N/m] for unit wave amplitude at circular frequency ``omega``.  The
        generalized modal force is ``Q_k = int phi_k(x) w(x) dx``; the modal
        response follows from the damped SDOF transfer function, and the stress
        is recovered by the same curvature sum as the time-domain path, so the
        frequency- and time-domain legs share one structural model.

        Returns ``T(f)`` in [Pa per unit wave amplitude].
        """
        f = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
        xs = np.linspace(0.0, self.L, 801) if x_quad is None else np.asarray(x_quad, float)
        phi = np.array([self.shape(xs, k) for k in range(self.n_modes)])       # (K, Nx)
        kap = np.array([float(self.curvature_shape(x_detail, k)) for k in range(self.n_modes)])
        out = np.zeros(f.size, dtype=complex)
        for i, fi in enumerate(f):
            om = 2.0 * math.pi * fi
            w = np.asarray(load_shape(xs, om), dtype=complex)
            Q = np.trapezoid(phi * w[None, :], xs, axis=1)                     # (K,)
            r = fi / self.freq
            daf = 1.0 / (1.0 - r ** 2 + 2j * self.zeta * r)                    # (K,)
            q_modal = Q * daf / self.modal_stiffness      # modal response [m]
            out[i] = -E * z * np.dot(q_modal, kap)          # complex amplitude [Pa/m]
        return out


def modal_stress_history(
    modes: BeamModes,
    force: np.ndarray,
    dt: float,
    x_load: float,
    x_detail: float,
    z: float,
    E: float,
    n_dynamic_modes: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Modal superposition of a point-force history -> detail stress history.

    Returns ``sigma`` (total) and ``sigma_qs`` (modal static limit of the same
    force history), i.e. the two series whose difference defines whipping.
    """
    force = np.asarray(force, dtype=float)
    nt = force.size
    kdyn = int(n_dynamic_modes or modes.n_modes)
    q = np.zeros((kdyn, nt))
    for k in range(kdyn):
        m_n = modes.modal_mass[k]
        w = modes.omega[k]
        zeta = modes.zeta[k]
        num = [1.0]
        den = [m_n, 2.0 * zeta * w * m_n, w * w * m_n]
        dnum, dden, _ = signal.cont2discrete((num, den), dt, method="zoh")
        Qn = force * float(modes.shape(x_load, k))
        q[k] = signal.lfilter(dnum[0], dden, Qn)
    sigma = modes.stress_from_modal(q, x_detail, z, E)

    q_static = np.zeros((kdyn, nt))
    for k in range(kdyn):
        q_static[k] = force * float(modes.shape(x_load, k)) / modes.modal_stiffness[k]
    sigma_qs = modes.stress_from_modal(q_static, x_detail, z, E)
    return {"sigma": sigma, "sigma_qs": sigma_qs, "q": q, "force": force}


def whipping_share(sigma: Sequence[float], sigma_qs: Sequence[float]) -> Dict[str, float]:
    """m5528 whipping share: the dynamic (inertia-amplified) part of the stress.

    ``share_rms  = rms(sigma - sigma_qs) / rms(sigma)``
    ``share_peak = max|sigma - sigma_qs| / max|sigma|``  (the 20-40 % caliber)

    Both series are compared after removing their own mean, as in m5528.
    """
    s = np.asarray(sigma, dtype=float)
    q = np.asarray(sigma_qs, dtype=float)
    if s.shape != q.shape:
        raise ValueError("sigma and sigma_qs must have the same shape")
    s0 = s - s.mean()
    w = s - q
    w0 = w - w.mean()
    rms_s = float(np.sqrt(np.mean(s0 ** 2)))
    rms_w = float(np.sqrt(np.mean(w0 ** 2)))
    peak_s = float(np.max(np.abs(s)))
    peak_w = float(np.max(np.abs(w)))
    return {
        "share_rms": rms_w / rms_s if rms_s > 0 else 0.0,
        "share_peak": peak_w / peak_s if peak_s > 0 else 0.0,
        "rms_total": rms_s,
        "rms_whip": rms_w,
        "peak_total": peak_s,
        "peak_whip": peak_w,
    }


def quasi_static_moment_free_free(x, F: float, a: float, L: float) -> np.ndarray:
    """Static bending moment of a free-free beam loaded by ``F`` at ``x = a``.

    Rigid-body reactions balance both force and moment:
    ``r(s) = F/L + c (s - L/2)`` with ``c = 12 F (a - L/2) / L^3``.
    """
    x = np.asarray(x, dtype=float)
    c = 12.0 * F * (a - 0.5 * L) / L ** 3
    intR = F * x ** 2 / (2.0 * L) + c * (x ** 3 / 6.0 - L * x ** 2 / 4.0)
    return -intR + F * np.where(x > a, x - a, 0.0)