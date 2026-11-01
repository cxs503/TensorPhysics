"""Unsteady hydrodynamic blade loading from the circumferential wake field.

Reference theory used (analytic, thin airfoil, incompressible):

* Theodorsen function ``C(k) = H1^(2)/(H1^(2) + i H0^(2))`` (airfoil motion)
* Sears function ``S(k) = C(k) (J0(k) - i J1(k))`` (sinusoidal transverse gust)
* Kuessner function ``Psi(s) = 1 - 0.5 e^{-0.13 s} - 0.5 e^{-s}`` (sharp-edged
  gust, ``s = 2 V t / c`` in semi-chords) used as the *time-domain* indicial kernel

Two independent routes are implemented:

``blade_element_hydro`` (pipeline)
    Radial blade-element integration of the quasi-steady lift driven by the wake
    harmonics, with the harmonic response corrected station-by-station by
    ``|S(k_r)|`` (engineering unsteady BEMT).
``sears_root_moment`` (reference)
    Linear thin-airfoil closed form: gust-amplitude -> lift via ``2 pi alpha``
    and ``|S(k)|``, integrated radially.

A third route, :func:`kussner_gust_response`, integrates the Kuessner indicial
relation in the time domain and is therefore independent of the frequency-domain
algebra; it is used in the tests to verify ``S(k)``.

No RANS/LES reference is available inside this work-tree, so criterion (b) is
evaluated against this analytic unsteady reference and flagged accordingly in
the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import jv, yv


# --------------------------------------------------------------------------- #
#  Frequency-domain unsteady airfoil theory
# --------------------------------------------------------------------------- #
def _hankel2(nu: int, k: float) -> complex:
    """Hankel function of the second kind ``H_nu^(2)(k) = J_nu - i Y_nu``."""
    return complex(jv(nu, k) - 1j * yv(nu, k))


def theodorsen_function(k: float) -> complex:
    """Theodorsen lift-deficiency function ``C(k)`` (C(0) = 1, C(inf) -> 1/2)."""
    k = float(k)
    if k <= 0.0:
        return complex(1.0, 0.0)
    h1 = _hankel2(1, k)
    h0 = _hankel2(0, k)
    return h1 / (h1 + 1j * h0)


def sears_gust_function(k: float) -> complex:
    """Sears gust response ``S(k)`` (S(0) = 1, ``|S| ~ 1/sqrt(2 pi k)``)."""
    k = float(k)
    if k <= 0.0:
        return complex(1.0, 0.0)
    return theodorsen_function(k) * complex(jv(0, k) - 1j * jv(1, k))


def kussner_function(s):
    """Kuessner indicial function for a sharp-edged transverse gust."""
    s = np.asarray(s, dtype=float)
    return 1.0 - 0.5 * np.exp(-0.13 * s) - 0.5 * np.exp(-s)


def kussner_gust_response(t: np.ndarray, upwash: np.ndarray, V: float, c: float,
                          a: float = 0.13, b: float = 1.0) -> np.ndarray:
    """Time-domain gust response ``C_L(t) / (2 pi)`` via the Kuessner kernel.

    With ``W = upwash / V`` (non-dimensional upwash) and ``s = 2 V t / c`` the
    exponential approximation ``Psi(s) = 1 - 0.5 e^{-a s} - 0.5 e^{-b s}`` gives
    the exact LTI response

        resp(s) = 0.5 a int_0^s W e^{-a (s-sigma)} dsigma
                + 0.5 b int_0^s W e^{-b (s-sigma)} dsigma ,

    which reproduces the step-gust response ``Psi(s)`` exactly and is therefore
    an independent (time-domain) route to :func:`sears_gust_function`.
    """
    t = np.asarray(t, dtype=float)
    up = np.asarray(upwash, dtype=float)
    s = 2.0 * V * t / c
    W = up / V
    ds = s[1] - s[0]
    La = fftconvolve(W, np.exp(-a * s))[: s.size] * ds
    Lb = fftconvolve(W, np.exp(-b * s))[: s.size] * ds
    return 0.5 * a * La + 0.5 * b * Lb


def harmonic_amplitude_angle(signal: np.ndarray, theta: np.ndarray, k: int) -> complex:
    """Complex amplitude of the ``k``-th angular harmonic of ``signal(theta)``.

    ``signal ~ Re(c e^{i k theta})``, so ``|c|`` is the amplitude and the phase is
    measured from ``theta = 0``.  ``k = 1`` is the once-per-revolution content the
    blade sees.
    """
    th = np.asarray(theta, dtype=float)
    sig = np.asarray(signal, dtype=float)
    basis = np.column_stack([np.cos(k * th), np.sin(k * th), np.ones_like(th)])
    coef, *_ = np.linalg.lstsq(basis, sig, rcond=None)
    return complex(coef[0], -coef[1])


def harmonic_amplitude(signal: np.ndarray, t: np.ndarray, f: float) -> complex:
    """Least-squares complex amplitude of ``signal`` at frequency ``f``."""
    w = 2 * np.pi * f
    basis = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(basis, signal, rcond=None)
    return complex(coef[0], -coef[1])


# --------------------------------------------------------------------------- #
#  Wake field
# --------------------------------------------------------------------------- #
@dataclass
class WakeField:
    """Circumferential (once-per-revolution) wake field.

    ``w(theta, r) = w_mean * (1 + sum_k a_k cos(k theta + psi_k))``
    (axial wake fraction, the usual convention for a single-screw hull).
    """

    w_mean: float = 0.28
    harmonics: Tuple[Tuple[int, float, float], ...] = ((1, 0.45, 0.0), (2, 0.20, 0.6), (3, 0.10, 1.1))

    def fraction(self, theta, radius_ratio=0.7) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        val = np.ones_like(theta)
        for k, a, psi in self.harmonics:
            val = val + a * np.cos(k * theta + psi)
        return self.w_mean * val

    def axial_velocity(self, theta, V: float) -> np.ndarray:
        """Axial inflow velocity seen by an element at circumferential position theta."""
        return V * (1.0 - self.fraction(theta))


# --------------------------------------------------------------------------- #
#  Radial blade element
# --------------------------------------------------------------------------- #
@dataclass
class BladeStations:
    """Radial discretisation of the blade (arrays, hub radius excluded)."""

    r: np.ndarray            # radii [m]
    c: np.ndarray            # chord [m]
    beta: np.ndarray         # pitch (blade) angle [rad]
    r_root: float            # hub/root radius used for the bending lever [m]
    z_blades: int = 4
    cl_slope: float = 2 * np.pi
    span_factor: float = 0.9

    @property
    def dr(self) -> np.ndarray:
        return np.gradient(self.r)


def _station_kinematics(st: BladeStations, V: float, n_rps: float, theta) -> Dict[str, np.ndarray]:
    Va = np.asarray(theta, dtype=float)  # placeholder for broadcasting
    Va = np.full_like(Va, V)
    omega = 2 * np.pi * n_rps
    W = np.sqrt(Va[..., None] ** 2 + (omega * st.r) ** 2)  # (..., n_r)
    phi = np.arctan2(Va[..., None], omega * st.r)
    return {"Va": Va, "W": W, "phi": phi, "omega": omega}


def blade_element_hydro(st: BladeStations, V: float, n_rps: float, wake: WakeField,
                        n_theta: int = 1440, unsteady: bool = True) -> Dict[str, object]:
    """Blade-element root bending stress history from the wake harmonics.

    The quasi-steady lift is evaluated station-by-station from the wake-modulated
    axial inflow; when ``unsteady`` the 1P..3P content of every station is scaled
    by ``|S(k)|`` at that station's reduced frequency before the radial
    integration (physically correct, unlike a post-hoc scaling of the moment).

    Returns shaft angle, per-blade root moment histories, mean/amplitude and the
    reduced-frequency statistics.
    """
    omega = 2 * np.pi * n_rps
    theta_shaft = np.linspace(0.0, 4 * np.pi, n_theta, endpoint=False)
    rho = 1025.0
    moments = np.zeros((st.z_blades, n_theta))
    W_stat = np.sqrt(V ** 2 + (omega * st.r) ** 2)
    for b in range(st.z_blades):
        theta_b = theta_shaft + 2 * np.pi * b / st.z_blades
        Va = wake.axial_velocity(theta_b, V)                      # (n_theta,)
        W = np.sqrt(Va[:, None] ** 2 + (omega * st.r) ** 2)        # (n_theta, n_r)
        phi = np.arctan2(Va[:, None], omega * st.r)
        alpha = np.clip(st.beta - phi, -0.35, 0.35)
        CL = st.span_factor * st.cl_slope * alpha
        dL = 0.5 * rho * W ** 2 * st.c * CL                        # [N/m]
        dT = dL * np.cos(phi)                                      # axial component
        if unsteady:
            dT = _apply_unsteady_per_station(dT, theta_shaft, omega, st, W_stat)
        moments[b] = np.trapezoid(dT * (st.r - st.r_root), st.r, axis=1)
    mean_m = moments.mean(axis=1)
    amp_m = 0.5 * (moments.max(axis=1) - moments.min(axis=1))
    worst = int(np.argmax(amp_m))
    return {
        "theta": theta_shaft,
        "moment": moments,
        "worst_blade": worst,
        "moment_mean": mean_m,
        "moment_amplitude": amp_m,
        "reduced_frequency": reduced_frequency(st, n_rps, V),
        "shaft_rate_hz": n_rps,
    }


def reduced_frequency(st: BladeStations, n_rps: float, V: float) -> Dict[str, float]:
    """Station-wise reduced frequency ``k = pi f c / W`` at the shaft rate."""
    omega = 2 * np.pi * n_rps
    W = np.sqrt(V ** 2 + (omega * st.r) ** 2)
    k = np.pi * n_rps * st.c / W
    return {"k_min": float(k.min()), "k_max": float(k.max()), "k_mean": float(k.mean()),
            "k_stations": k.tolist()}


def _apply_unsteady_per_station(dT: np.ndarray, theta: np.ndarray, omega: float,
                                st: BladeStations, W_stat: np.ndarray) -> np.ndarray:
    """Scale the 1P..3P content of the lift at every station by ``|S(k)|``."""
    out = np.empty_like(dT)
    for i in range(dT.shape[1]):
        sig = dT[:, i]
        acc = np.full_like(sig, sig.mean())
        for kk in (1, 2, 3):
            f = kk * omega / (2.0 * np.pi)
            amp = harmonic_amplitude_angle(sig, theta, kk)
            gain = abs(sears_gust_function(np.pi * f * st.c[i] / W_stat[i]))
            acc = acc + amp.real * gain * np.cos(kk * theta) - amp.imag * gain * np.sin(kk * theta)
        out[:, i] = acc
    return out


def sears_root_moment(st: BladeStations, V: float, n_rps: float, wake: WakeField,
                      harmonic: int = 1) -> Dict[str, float]:
    """Analytic thin-airfoil root bending moment for a single wake harmonic.

    First-order linearisation of the 2D thin-airfoil lift
    ``L = rho pi W^2 c alpha`` with respect to the axial inflow ``Va``::

        dL/dVa = rho pi c [ 2 Va alpha  -  Vt ]          (W^2 and incidence terms)
        dT/dVa = cos(phi) dL/dVa
        amplitude = |dT/dVa| dVa |S(k)| ,  k = pi f c / W

    The two terms partially cancel; omitting the ``2 Va alpha`` part is a common
    error and biases the result by ~15 % at high loading.  The radial integral of
    ``amplitude * (r - r_root)`` is the root bending moment amplitude that the
    blade-element route must reproduce.
    """
    k_order, a_k, _ = [h for h in wake.harmonics if h[0] == harmonic][0]
    omega = 2 * np.pi * n_rps
    f = harmonic * n_rps
    Vt = omega * st.r
    W = np.sqrt(V ** 2 + Vt ** 2)
    phi = np.arctan2(V, Vt)
    alpha = st.beta - phi
    cos_phi = Vt / W
    dV = V * wake.w_mean * a_k
    dT_dVa = 1025.0 * np.pi * st.c * cos_phi * (2.0 * V * alpha - Vt)
    k_stat = np.pi * f * st.c / W
    S = np.array([abs(sears_gust_function(kk)) for kk in k_stat])
    amp = np.abs(dT_dVa) * dV * S
    M = np.trapezoid(amp * (st.r - st.r_root), st.r)
    return {"harmonic": harmonic, "f_Hz": f, "moment_amplitude": float(M),
            "k_mean": float(k_stat.mean()), "sears_gain_mean": float(S.mean()),
            "dV_m_s": float(dV)}
