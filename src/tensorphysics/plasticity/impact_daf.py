"""Dynamic amplification factor (DAF) of rate-dependent plate-strip impact.

The dynamic amplification factor compares the *dynamic* peak section stress of a
plate strip hit by a pressure pulse with the *quasi-static* stress under the same
peak pressure,

.. math::

    \mathrm{DAF} = \frac{\max_t \sigma_{\mathrm{dyn}}(t)}
                         {\sigma_{\mathrm{static}}},

folding two physically distinct effects together:

* **inertia** -- the transient response to a finite-duration pulse (the classic
  half-sine shock spectrum: ``DAF -> 1`` for long pulses, maximum
  ``~1.77`` at ``tau/T ~ 0.8`` for an undamped SDOF);
* **material strain-rate sensitivity** -- the Cowper-Symonds overstress factor of
  :class:`~tensorphysics.plasticity.cowper_symonds.CowperSymonds`, which raises
  the flow stress and hence the plastic hinge capacity with the plastic strain
  rate.

Module contents
---------------
:func:`half_sine_pulse`, :func:`pulse_impulse`
    Ice-impact load shape ``P(t) = P0 sin(pi t / tau)`` on ``0 <= t <= tau``.

:func:`sdof_half_sine_response`, :func:`sdof_half_sine_daf`
    **Exact** undamped SDOF response to a half-sine pulse: piecewise closed form
    (Duhamel integral plus post-pulse free vibration) -- the classical shock
    spectrum, used as the analytic reference.

:func:`sdof_half_sine_daf_rk4`
    Independent numerical path (RK4 on the state-space ODE), so that the closed
    form is not self-referential.

:func:`beam_modal_elastic_daf`
    Exact semi-analytic elastic reference for the plate strip: modal
    superposition over the Euler-Bernoulli modes of the end conditions used,
    with the closed-form modal response -- no time marching.

:class:`BeamHingeFEM`
    Finite-element plate strip: Euler-Bernoulli elements (plate strip in
    cylindrical bending, ``E/(1-nu^2)``), lumped mass, Newmark average
    acceleration, and **three rate-dependent plastic hinges** (both ends and
    mid-span) with ``M_cap = W (sigma_0 + H eps_p) A(eps_p_dot)``.

Why three hinges
    For a clamped strip under uniform pressure the end moments ``q L^2/12``
    reach first yield before the mid-span moment ``q L^2/24``, and the collapse
    mechanism needs all three hinges.  Modelling all three keeps the response
    **bounded** between first yield and collapse
    (``12 M_y/L^2 <= P <= 16 M_y/L^2`` in the perfectly plastic limit), which is
    the window in which a DAF including strain-rate effects is physically
    meaningful.  A lone mid-span hinge turns into a mechanism as soon as it
    yields, which produces runaway plastic rotations (see the module notes in
    ``docs/RATE_PLASTICITY_DAF_P351W1N05.md``).

Conventions
-----------
``length`` is the span, ``width`` the strip width (1 m = unit-width plate
strip), ``thickness`` the plate thickness.  Stresses are *nominal extreme-fibre
stresses* ``sigma = M / W`` with ``W = b h^2 / 6``.  The hinge capacity is the
first-yield moment (no section shape factor), so DAF is referred to the same
first-yield stress that the Cowper-Symonds law amplifies.

References
----------
- N. Jones, *Structural Impact*, Cambridge UP, 1989 -- pulse-loaded beams and
  plates; Cowper-Symonds constants ``D = 40.4 s^-1``, ``p = 5`` for mild steel.
- A.K. Chopra, *Dynamics of Structures*, ch. 4 -- response to half-sine impulse.
- C.M. Harris, C.E. Crede, *Shock and Vibration Handbook* -- undamped half-sine
  peak DAF ~ 1.77.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cowper_symonds import CowperSymonds

__all__ = [
    "half_sine_pulse",
    "pulse_impulse",
    "sdof_half_sine_response",
    "sdof_half_sine_daf",
    "sdof_half_sine_daf_rk4",
    "beam_first_mode_period",
    "beam_clamped_static_moments",
    "beam_modal_elastic_daf",
    "BeamHingeFEM",
    "ImpactCase",
]

SUPPORTS = ("clamped", "simply_supported")


# ---------------------------------------------------------------------------
# Pulse shape
# ---------------------------------------------------------------------------
def half_sine_pulse(t, duration: float, amplitude: float = 1.0) -> np.ndarray:
    """Half-sine pressure pulse ``P0 sin(pi t / tau)`` on ``[0, tau]``, zero after."""
    if duration <= 0.0:
        raise ValueError("pulse duration must be > 0")
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    inside = (t >= 0.0) & (t <= duration)
    out[inside] = amplitude * np.sin(np.pi * t[inside] / duration)
    return out


def pulse_impulse(duration: float, amplitude: float = 1.0) -> float:
    """Impulse of the half-sine pulse, ``2 P0 tau / pi`` [Pa s]."""
    if duration <= 0.0:
        raise ValueError("pulse duration must be > 0")
    return 2.0 * amplitude * duration / np.pi


# ---------------------------------------------------------------------------
# SDOF shock spectrum: exact closed form + independent RK4
# ---------------------------------------------------------------------------
def sdof_half_sine_response(t, tau_over_T: float) -> np.ndarray:
    """Exact undamped SDOF response to a half-sine pulse, normalised by ``x_static``.

    With ``r = Omega/omega = T/(2 tau)``,

    .. math::

        x(t)/x_{st} = \frac{\sin(\Omega t) - r\sin(\omega t)}{1 - r^2},
        \qquad 0 \le t \le \tau,

    and free vibration after the pulse with amplitude
    ``2 r |cos(omega tau / 2)| / |1 - r^2|``.  Time is in units of ``T``.
    """
    if tau_over_T <= 0.0:
        raise ValueError("tau_over_T must be > 0")
    t = np.asarray(t, dtype=float)
    tau = float(tau_over_T)  # in units of T = 1
    omega = 2.0 * np.pi
    Omega = np.pi / tau
    r = Omega / omega
    out = np.empty_like(t)

    inside = (t >= 0.0) & (t <= tau)
    if np.any(inside):
        ti = t[inside]
        if abs(1.0 - r * r) < 1e-12:  # resonant limit (l'Hopital)
            out[inside] = 0.5 * (np.sin(Omega * ti) - Omega * ti * np.cos(Omega * ti))
        else:
            out[inside] = (np.sin(Omega * ti) - r * np.sin(omega * ti)) / (1.0 - r * r)

    after = t > tau
    if np.any(after):
        ta = t[after]
        if abs(1.0 - r * r) < 1e-12:
            amp, phase = 0.5 * Omega * tau, 0.0
        else:
            x_tau = -r * np.sin(omega * tau) / (1.0 - r * r)
            v_tau = r * (-1.0 - np.cos(omega * tau)) / (1.0 - r * r)
            amp = float(np.hypot(x_tau, v_tau))
            phase = float(np.arctan2(x_tau, v_tau / omega))
        out[after] = amp * np.abs(np.sin(omega * ta + phase))
    return out


def sdof_half_sine_daf(tau_over_T: float, n_samples: int = 40001) -> float:
    """Maximum ``|x|/x_static`` of the exact half-sine SDOF response."""
    if tau_over_T <= 0.0:
        raise ValueError("tau_over_T must be > 0")
    tau = float(tau_over_T)
    t_pulse = np.linspace(0.0, tau, n_samples)
    daf = float(np.max(np.abs(sdof_half_sine_response(t_pulse, tau))))
    r = (np.pi / tau) / (2.0 * np.pi)
    if abs(1.0 - r * r) < 1e-12:
        daf = max(daf, 0.5 * (np.pi / tau) * tau / max(r, 1e-12) * r)
    else:
        daf = max(daf, float(2.0 * r * abs(np.cos(np.pi * tau)) / abs(1.0 - r * r)))
    return daf


def sdof_half_sine_daf_rk4(tau_over_T: float, n_steps: int = 20000) -> float:
    """Independent numerical shock spectrum (RK4 on the state-space ODE)."""
    if tau_over_T <= 0.0:
        raise ValueError("tau_over_T must be > 0")
    tau = float(tau_over_T)
    omega = 2.0 * np.pi
    dt = (tau + 3.0) / n_steps

    def rhs(state, tt):
        x, v = state
        f = omega * omega * np.sin(np.pi * tt / tau) if tt <= tau else 0.0
        return np.array([v, f - omega * omega * x])

    state = np.zeros(2)
    peak = 0.0
    for i in range(n_steps):
        tt = i * dt
        k1 = rhs(state, tt)
        k2 = rhs(state + 0.5 * dt * k1, tt + 0.5 * dt)
        k3 = rhs(state + 0.5 * dt * k2, tt + 0.5 * dt)
        k4 = rhs(state + dt * k3, tt + dt)
        state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        peak = max(peak, abs(state[0]))
    return float(peak)


# ---------------------------------------------------------------------------
# Elastic plate-strip references (modal superposition, no time marching)
# ---------------------------------------------------------------------------
def beam_rigidity(youngs_modulus, thickness, width, poisson):
    """``EI`` of a plate strip in cylindrical bending."""
    return youngs_modulus / (1.0 - poisson**2) * width * thickness**3 / 12.0


def beam_first_mode_period(
    length: float,
    youngs_modulus: float,
    thickness: float,
    density: float,
    width: float = 1.0,
    poisson: float = 0.3,
    support: str = "clamped",
) -> float:
    """First natural period of the plate strip (Euler-Bernoulli, exact coefficient)."""
    root = {"clamped": 4.730040744862704, "simply_supported": np.pi}[support]
    ei = beam_rigidity(youngs_modulus, thickness, width, poisson)
    rho_a = density * width * thickness
    omega_1 = (root / length) ** 2 * np.sqrt(ei / rho_a)
    return float(2.0 * np.pi / omega_1)


def beam_clamped_static_moments(length: float, uniform_load: float) -> dict:
    """Closed-form elastic static moments of the clamped-clamped strip under UDL."""
    return {
        "end": float(uniform_load * length**2 / 12.0),
        "mid": float(uniform_load * length**2 / 24.0),
    }


def _clamped_roots(n_modes: int) -> np.ndarray:
    """Roots ``beta L`` of ``cos(beta L) cosh(beta L) = 1``."""
    roots = []
    for n in range(1, n_modes + 1):
        lo, hi = (n + 0.25) * np.pi, (n + 0.75) * np.pi
        f = lambda b: np.cos(b) * np.cosh(b) - 1.0
        flo = f(lo)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if flo * fmid <= 0.0:
                hi = mid
            else:
                lo, flo = mid, fmid
        roots.append(0.5 * (lo + hi))
    return np.array(roots)


def _mode_shape(support: str, beta_l: float, x, length: float):
    """Mode shape and second derivative for wavenumber ``beta = beta_l / L``."""
    beta = beta_l / length
    if support == "simply_supported":
        phi = np.sin(beta * x)
        d2phi = -(beta**2) * np.sin(beta * x)
        return phi, d2phi
    sigma = (np.cosh(beta_l) - np.cos(beta_l)) / (np.sinh(beta_l) - np.sin(beta_l))
    phi = (np.cosh(beta * x) - np.cos(beta * x)) - sigma * (np.sinh(beta * x) - np.sin(beta * x))
    d2phi = beta**2 * (
        (np.cosh(beta * x) + np.cos(beta * x)) - sigma * (np.sinh(beta * x) + np.sin(beta * x))
    )
    return phi, d2phi


def beam_modal_elastic_daf(
    length: float,
    thickness: float,
    pressure_amplitude: float,
    pulse_duration: float,
    youngs_modulus: float = 210.0e9,
    density: float = 7850.0,
    width: float = 1.0,
    poisson: float = 0.3,
    support: str = "clamped",
    n_modes: int = 12,
    n_samples: int = 20001,
) -> dict:
    """Exact elastic DAF of the plate strip from modal superposition.

    Each Euler-Bernoulli mode is an independent SDOF with the *closed-form*
    half-sine response :func:`sdof_half_sine_response`; modal mass, stiffness and
    load are obtained by quadrature of the exact mode shapes, and the moment at
    the end and mid-span sections is ``M(x,t) = -EI sum_n q_n(t) phi_n''(x)``.

    Returns a dict with the modal coefficients, the static moments, the moment
    time histories and ``daf`` = ``max(|M_dyn|)/max(|M_static|)`` over the two
    monitored sections.
    """
    if support not in SUPPORTS:
        raise ValueError(f"support must be one of {SUPPORTS}")
    ei = beam_rigidity(youngs_modulus, thickness, width, poisson)
    rho_a = density * width * thickness
    q0 = pressure_amplitude * width

    beta_l = np.pi * np.arange(1, n_modes + 1) if support == "simply_supported" else _clamped_roots(n_modes)
    xq = np.linspace(0.0, length, 2001)

    omega1 = (beta_l[0] / length) ** 2 * np.sqrt(ei / rho_a)
    t1 = 2.0 * np.pi / omega1
    t = np.linspace(0.0, pulse_duration + 3.0 * t1, n_samples)

    x_sections = {"end": 0.0, "mid": length / 2.0}
    hist = {k: np.zeros_like(t) for k in x_sections}
    static = {k: 0.0 for k in x_sections}

    for bl in beta_l:
        phi, d2phi = _mode_shape(support, bl, xq, length)
        mode_mass = rho_a * np.trapezoid(phi * phi, xq)
        mode_stiff = ei * np.trapezoid(d2phi * d2phi, xq)
        mode_load = q0 * np.trapezoid(phi, xq)
        omega_n = np.sqrt(mode_stiff / mode_mass)
        tn = 2.0 * np.pi / omega_n
        amp_static = mode_load / mode_stiff
        q_n = amp_static * sdof_half_sine_response(t / tn, pulse_duration / tn)
        for key, xs in x_sections.items():
            _, d2 = _mode_shape(support, bl, np.array([xs]), length)
            hist[key] += -ei * q_n * float(d2[0])
            static[key] += -ei * amp_static * float(d2[0])

    peaks = {k: float(np.max(np.abs(hist[k]))) for k in x_sections}
    daf_sections = {k: peaks[k] / abs(static[k]) for k in x_sections}
    return {
        "support": support,
        "t": t,
        "moment_end": hist["end"],
        "moment_mid": hist["mid"],
        "static_moment_end": static["end"],
        "static_moment_mid": static["mid"],
        "daf_end": daf_sections["end"],
        "daf_mid": daf_sections["mid"],
        "daf_global": max(peaks.values()) / max(abs(static[k]) for k in x_sections),
        "daf": max(daf_sections.values()),
        "first_period": float(t1),
        "n_modes": n_modes,
    }


# ---------------------------------------------------------------------------
# Impact case definition
# ---------------------------------------------------------------------------
@dataclass
class ImpactCase:
    """One impact case: pulse plus structural properties (all SI units)."""

    pressure_amplitude: float
    pulse_duration: float
    length: float = 1.0
    thickness: float = 0.008
    width: float = 1.0
    youngs_modulus: float = 210.0e9
    poisson: float = 0.3
    density: float = 7850.0
    yield_stress: float = 235.0e6
    hardening: float = 2.0e9
    support: str = "clamped"
    n_elements: int = 24
    cs: CowperSymonds = field(default_factory=CowperSymonds)
    hinge_penalty: float = 1.0e4
    hinge_length_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.n_elements % 2 != 0 or self.n_elements < 4:
            raise ValueError("n_elements must be even and >= 4")
        if self.thickness <= 0.0 or self.length <= 0.0:
            raise ValueError("thickness and length must be > 0")
        if self.pressure_amplitude < 0.0:
            raise ValueError("pressure_amplitude must be >= 0")
        if self.support not in SUPPORTS:
            raise ValueError(f"support must be one of {SUPPORTS}")

    @property
    def section_modulus(self) -> float:
        return self.width * self.thickness**2 / 6.0

    @property
    def yield_moment(self) -> float:
        return self.yield_stress * self.section_modulus

    @property
    def yield_pressure(self) -> float:
        """Pressure at which the *static* max section moment reaches ``M_y``."""
        if self.support == "clamped":
            return 12.0 * self.yield_moment / (self.length**2 * self.width)
        return 8.0 * self.yield_moment / (self.length**2 * self.width)

    @property
    def collapse_pressure(self) -> float:
        """Static plastic collapse pressure (mechanism with all hinges plastic)."""
        if self.support == "clamped":
            return 16.0 * self.yield_moment / (self.length**2 * self.width)
        return 8.0 * self.yield_moment / (self.length**2 * self.width)

    def stress_from_moment(self, moment):
        return np.asarray(moment, dtype=float) / self.section_modulus

    def static_stress(self, section: str = "max") -> float:
        """Quasi-static extreme-fibre stress under the peak pressure.

        Elastic analysis below first yield; above it the section stress is capped
        at the current flow stress ``sigma_0`` (limit-load plateau of the
        perfectly plastic hinge).
        """
        q0 = self.pressure_amplitude * self.width
        if self.support == "clamped":
            moments = beam_clamped_static_moments(self.length, q0)
        else:
            moments = {"end": 0.0, "mid": q0 * self.length**2 / 8.0}
        if section == "max":
            m = max(abs(moments["end"]), abs(moments["mid"]))
        else:
            m = abs(moments[section])
        return float(min(m / self.section_modulus, self.yield_stress))


# ---------------------------------------------------------------------------
# Finite-element plate strip with rate-dependent plastic hinges
# ---------------------------------------------------------------------------
class BeamHingeFEM:
    """Euler-Bernoulli plate-strip FE with rate-dependent plastic hinges.

    Elements
        Standard 2-node Hermite beam elements (2 DOF/node) for a plate strip in
        cylindrical bending (``E_eff = E/(1-nu^2)``), lumped translational mass.

    Hinges
        Rotational springs at both ends and at mid-span.  The mid-span node has
        two independent rotations; the ends have one rotation each, connected to
        the ground (clamped) or free (simply supported).  A hinge carries
        ``M = k (d_theta - theta_p)`` elastically, and plastically

        .. math:: M_{cap} = W (\\sigma_0 + H\\,\\epsilon_p)\\,A(\\dot\\epsilon_p),

        with ``\\epsilon_p = |\\theta_p| h / (2\\,\\ell_h)``,
        ``\\dot\\epsilon_p = |\\Delta\\theta_p| h / (2\\,\\ell_h)/\\Delta t`` and
        ``\\ell_h = hinge_length_ratio * h`` the plastic hinge length.

    Integration
        Newmark average acceleration (``beta = 1/4``, ``gamma = 1/2``) with
        Newton iterations.  The hinge moments enter as internal forces and the
        tangents through the exact Woodbury rank-``n`` update of one
        pre-inverted elastic effective matrix, so a time step costs a few
        matrix-vector products only.

    Notes
        The static elastic-limit pressure is ``yield_pressure`` and the plastic
        collapse pressure ``collapse_pressure``; between them the response is
        bounded and mildly plastic.  Above ``collapse_pressure`` the model has a
        mechanism and the plastic rotations grow with the pulse duration -- that
        regime is outside the validated envelope (a real plate would develop
        membrane action).
    """

    def __init__(self, case: ImpactCase) -> None:
        self.case = case
        n = case.n_elements
        self.n_nodes = n + 1
        self.mid = n // 2
        self.le = case.length / n

        ei = beam_rigidity(case.youngs_modulus, case.thickness, case.width, case.poisson)
        rho_a = case.density * case.width * case.thickness
        self.EI = ei
        self.k_hinge = case.hinge_penalty * ei / case.length
        self.m_y0 = case.yield_moment
        self.hinge_len = case.hinge_length_ratio * case.thickness
        self.sigma_0 = case.yield_stress
        self.hardening = case.hardening
        self.W = case.section_modulus

        # --- DOFs: w(node) | theta_left(0..mid) | theta_right(mid..N) --------
        self.n_w = self.n_nodes
        self.theta_left = self.n_w + np.arange(self.mid + 1)
        self.theta_right = self.n_w + (self.mid + 1) + np.arange(self.n_nodes - self.mid)
        self.n_dof = self.n_w + (self.mid + 1) + (self.n_nodes - self.mid)

        fixed = [0, self.n_nodes - 1]
        if case.support == "clamped":
            self._fixed = fixed  # end rotations are hinge DOFs, kept free
        else:
            self._fixed = fixed
        self._free = np.array([i for i in range(self.n_dof) if i not in self._fixed])

        el = np.array(
            [
                [12.0, 6.0 * self.le, -12.0, 6.0 * self.le],
                [6.0 * self.le, 4.0 * self.le**2, -6.0 * self.le, 2.0 * self.le**2],
                [-12.0, -6.0 * self.le, 12.0, -6.0 * self.le],
                [6.0 * self.le, 2.0 * self.le**2, -6.0 * self.le, 4.0 * self.le**2],
            ]
        ) * ei / self.le**3

        K = np.zeros((self.n_dof, self.n_dof))
        Mmat = np.zeros((self.n_dof, self.n_dof))
        f_unit = np.zeros(self.n_dof)
        q_unit = case.width
        for e in range(n):
            if e + 1 <= self.mid:
                dofs = [e, int(self.theta_left[e]), e + 1, int(self.theta_left[e + 1])]
            else:
                dofs = [
                    e,
                    int(self.theta_right[e - self.mid]),
                    e + 1,
                    int(self.theta_right[e + 1 - self.mid]),
                ]
            K[np.ix_(dofs, dofs)] += el
            m_e = rho_a * self.le / 2.0
            Mmat[dofs[0], dofs[0]] += m_e
            Mmat[dofs[2], dofs[2]] += m_e
            f_unit[dofs] += np.array(
                [
                    q_unit * self.le / 2.0,
                    q_unit * self.le**2 / 12.0,
                    q_unit * self.le / 2.0,
                    -q_unit * self.le**2 / 12.0,
                ]
            )
        self.K = K
        self.M = Mmat
        self._f_unit = f_unit

        # --- hinge topology: list of (dof_a, dof_b_or_None) ------------------
        self.hinges: list[tuple[int, int | None]] = []
        if case.support == "clamped":
            self.hinges.append((int(self.theta_left[0]), None))
        self.hinges.append((int(self.theta_left[self.mid]), int(self.theta_right[0])))
        if case.support == "clamped":
            self.hinges.append((int(self.theta_right[-1]), None))
        # ground hinges use s = -e_a (theta_b = 0)
        self._s = []
        for a, b in self.hinges:
            s = np.zeros(self.n_dof)
            if b is None:
                s[a] = -1.0
            else:
                s[b] = 1.0
                s[a] = -1.0
            self._s.append(s)

    # ------------------------------------------------------------------ hinges
    @staticmethod
    def _cs_amplification(cs: CowperSymonds, rate: float) -> float:
        if rate <= 0.0:
            return 1.0
        r = rate if cs.rate_threshold is None else min(rate, cs.rate_threshold)
        return 1.0 + (r / cs.D) ** (1.0 / cs.p)

    def hinge_capacity(self, eps_p: float, eps_p_dot: float) -> float:
        """``W (sigma_0 + H eps_p) A(eps_p_dot)``."""
        return self.W * (self.sigma_0 + self.hardening * eps_p) * self._cs_amplification(
            self.case.cs, eps_p_dot
        )

    def hinge_state(self, d_theta: float, theta_p: float, eps_p: float, dt: float):
        """Moment, plastic rotation, accumulated plastic strain and strain rate.

        Solves ``k (d_theta - theta_p_new) = M_cap(eps_p(theta_p_new), rate)`` by
        bisection towards the total rotation (monotone residual), and falls back
        to the elastic branch when the trial moment is admissible.
        """
        k = self.k_hinge
        m_trial = k * (d_theta - theta_p)
        cap0 = self.hinge_capacity(eps_p, 0.0)
        if abs(m_trial) <= cap0 or abs(d_theta - theta_p) < 1e-16:
            return m_trial, theta_p, eps_p, 0.0

        step = d_theta - theta_p
        sign = 1.0 if step >= 0.0 else -1.0
        lo, hi = 0.0, abs(step)
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            moment_mag = k * (abs(step) - mid)
            d_eps = mid * self.case.thickness / (2.0 * self.hinge_len)
            rate = d_eps / dt
            cap = self.hinge_capacity(eps_p + d_eps, rate)
            if moment_mag > cap:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-15 * max(1.0, abs(step)):
                break
        d_theta_p = 0.5 * (lo + hi)
        d_eps = d_theta_p * self.case.thickness / (2.0 * self.hinge_len)
        rate = d_eps / dt
        theta_p_new = theta_p + sign * d_theta_p
        moment = sign * self.hinge_capacity(eps_p + d_eps, rate)
        return moment, theta_p_new, eps_p + d_eps, rate

    def _hinge_forces(self, u, states, dt):
        """Return hinge moments, updated states, rates and elastic/plastic flags."""
        moments, rates, plastic = [], [], []
        new_states = []
        for (a, b), st in zip(self.hinges, states):
            theta_b = 0.0 if b is None else u[b]
            d_theta = theta_b - u[a]
            m, theta_p, eps_p, rate = self.hinge_state(d_theta, st[0], st[1], dt)
            new_states.append((theta_p, eps_p))
            moments.append(m)
            rates.append(rate)
            plastic.append(abs(theta_p - st[0]) > 0.0)
        return moments, new_states, rates, plastic

    def _elastic_operator(self, base: np.ndarray, plastic_tuple, s_vecs=None) -> np.ndarray:
        """``base + sum_k k_k s_k s_k^T`` with soft tangents for plastic hinges."""
        op = base.copy()
        s_vecs = self._s if s_vecs is None else s_vecs
        for s, is_plastic in zip(s_vecs, plastic_tuple):
            k = 1e-3 * self.k_hinge if is_plastic else self.k_hinge
            op += k * np.outer(s, s)
        return op

    # ----------------------------------------------------------------- statics
    def _elastic_static_operator(self) -> np.ndarray:
        op = self.K.copy()
        for s in self._s:
            op += self.k_hinge * np.outer(s, s)
        return op

    def static_solution(self) -> dict:
        """Elastic static solution under the peak pressure: moments at all hinges."""
        op = self._elastic_static_operator()
        free = self._free
        f = self._f_unit * self.case.pressure_amplitude
        u = np.zeros(self.n_dof)
        u[free] = np.linalg.solve(op[np.ix_(free, free)], f[free])
        moments = []
        for (a, b), s in zip(self.hinges, self._s):
            theta_b = 0.0 if b is None else u[b]
            moments.append(self.k_hinge * (theta_b - u[a]))
        return {"u": u, "hinge_moments": moments, "w_mid": float(u[self.mid])}

    def static_moment(self) -> float:
        """Largest elastic static hinge moment (the DAF denominator section)."""
        return float(max(abs(m) for m in self.static_solution()["hinge_moments"]))

    # ---------------------------------------------------------------- dynamics
    def dynamic(
        self,
        dt: float | None = None,
        t_end: float | None = None,
        max_newton: int = 15,
        rtol: float = 1e-9,
    ) -> dict:
        """Newmark time integration under the half-sine pulse; returns histories."""
        case = self.case
        t1 = beam_first_mode_period(
            case.length,
            case.youngs_modulus,
            case.thickness,
            case.density,
            case.width,
            case.poisson,
            case.support,
        )
        if dt is None:
            dt = t1 / 400.0
        if t_end is None:
            t_end = case.pulse_duration + 3.0 * t1

        n_steps = int(np.ceil(t_end / dt))
        beta, gamma = 0.25, 0.5
        free = self._free
        k_ff = self.K[np.ix_(free, free)]
        m_ff = self.M[np.ix_(free, free)]
        base = m_ff / (beta * dt * dt) + k_ff

        s_free = [s[free] for s in self._s]
        hinge_rows = []
        for a, b in self.hinges:
            ia = int(np.where(free == a)[0][0]) if a in free else None
            ib = (int(np.where(free == b)[0][0]) if b is not None and b in free else None)
            hinge_rows.append((ia, ib))

        inv_cache: dict[tuple, np.ndarray] = {}

        def operator_inverse(plastic_tuple):
            key = tuple(bool(p) for p in plastic_tuple)
            mat = inv_cache.get(key)
            if mat is None:
                mat = np.linalg.inv(self._elastic_operator(base, key, s_free))
                inv_cache[key] = mat
            return mat

        u = np.zeros(self.n_dof)
        theta = np.zeros(self.n_dof)
        acc = np.zeros(self.n_dof)
        states = [(0.0, 0.0) for _ in self.hinges]

        n_rec = n_steps + 1
        t_hist = np.zeros(n_rec)
        m_hist = np.zeros((len(self.hinges), n_rec))
        w_hist = np.zeros(n_rec)
        rate_hist = np.zeros(n_rec)
        p_hist = np.zeros(n_rec)

        u_free = u[free].copy()
        v_free = theta[free].copy()
        a_free = acc[free].copy()
        residual_scale = max(
            1e-12,
            float(np.max(np.abs(self._f_unit))) * max(case.pressure_amplitude, 1.0),
        )

        for step in range(n_steps):
            t_next = (step + 1) * dt
            p_load = float(half_sine_pulse(t_next, case.pulse_duration, case.pressure_amplitude))
            f = (self._f_unit * p_load)[free]

            u_new = u_free.copy()
            plastic_flags = tuple(False for _ in self.hinges)
            u_full = u.copy()
            for _ in range(max_newton):
                a_trial = (
                    (u_new - u_free) / (beta * dt * dt)
                    - v_free / (beta * dt)
                    - (1.0 / (2.0 * beta) - 1.0) * a_free
                )
                u_full[free] = u_new
                moments, new_states, rates, plastic = self._hinge_forces(u_full, states, dt)
                resid = m_ff @ a_trial + k_ff @ u_new - f
                for m, (ia, ib) in zip(moments, hinge_rows):
                    if ia is not None:
                        resid[ia] += -m
                    if ib is not None:
                        resid[ib] += m
                plastic_flags = tuple(plastic)
                du = -operator_inverse(plastic_flags) @ resid
                # backtracking line search: softening hinges make the
                # near-mechanism tangent unreliable, so the step is shortened
                # whenever it does not reduce the residual norm
                def _resid_norm(uu):
                    u_full[free] = uu
                    mm, _, _, _ = self._hinge_forces(u_full, states, dt)
                    rr = m_ff @ (
                        (uu - u_free) / (beta * dt * dt)
                        - v_free / (beta * dt)
                        - (1.0 / (2.0 * beta) - 1.0) * a_free
                    ) + k_ff @ uu - f
                    for m_, (ia, ib) in zip(mm, hinge_rows):
                        if ia is not None:
                            rr[ia] += -m_
                        if ib is not None:
                            rr[ib] += m_
                    return float(np.linalg.norm(rr))

                nrm0 = float(np.linalg.norm(resid))
                if nrm0 <= 1e-12 * residual_scale:
                    break
                omega = 1.0
                for _ls in range(3):
                    if _resid_norm(u_new + omega * du) <= nrm0:
                        break
                    omega *= 0.5
                u_new = u_new + omega * du
                if np.max(np.abs(omega * du)) < rtol * max(1e-14, np.max(np.abs(u_new))):
                    break

            u_full[free] = u_new
            moments, new_states, rates, _ = self._hinge_forces(u_full, states, dt)
            states = new_states
            a_new = (u_new - u_free) / (beta * dt * dt) - v_free / (beta * dt) - (
                1.0 / (2.0 * beta) - 1.0
            ) * a_free
            v_new = v_free + dt * ((1.0 - gamma) * a_free + gamma * a_new)

            u[free] = u_new
            u_free = u_new.copy()
            v_free = v_new.copy()
            a_free = a_new.copy()

            t_hist[step + 1] = t_next
            for idx, m in enumerate(moments):
                m_hist[idx, step + 1] = m
            w_hist[step + 1] = u[self.mid]
            rate_hist[step + 1] = max(rates) if rates else 0.0
            p_hist[step + 1] = p_load

        stress_hist = m_hist / case.section_modulus
        return {
            "t": t_hist,
            "moment": m_hist,
            "stress": stress_hist,
            "w_mid": w_hist,
            "pressure": p_hist,
            "plastic_strain_rate": rate_hist,
            "plastic_states": states,
            "dt": dt,
            "first_period": float(t1),
            "n_steps": n_steps,
            "hinge_labels": [self._label(i) for i in range(len(self.hinges))],
        }

    def _label(self, idx: int) -> str:
        if len(self.hinges) == 3:
            return ("end_0", "mid", "end_L")[idx]
        return "mid"

    def daf(self, section: str = "max", **kwargs) -> dict:
        """``DAF = max |sigma_dyn| / sigma_static`` (section: hinge label or 'max')."""
        hist = self.dynamic(**kwargs)
        sigma_static_hinges = [
            self.case.stress_from_moment(m) for m in self.static_solution()["hinge_moments"]
        ]
        labels = hist["hinge_labels"]
        peaks = np.max(np.abs(hist["stress"]), axis=1)
        daf_sections = {
            lab: float(peaks[i] / abs(sigma_static_hinges[i])) for i, lab in enumerate(labels)
        }
        if section == "max":
            daf_value = max(daf_sections.values())
            idx = int(np.argmax([daf_sections[lab] for lab in labels]))
        else:
            idx = labels.index(section)
            daf_value = daf_sections[section]
        sigma_static_global = float(max(abs(s) for s in sigma_static_hinges))
        daf_global = float(np.max(peaks) / sigma_static_global)
        return {
            "section": labels[idx],
            "daf_global": daf_global,
            "sigma_static_global": sigma_static_global,
            "sigma_static": float(abs(sigma_static_hinges[idx])),
            "sigma_dynamic_max": float(peaks[idx]),
            "sigma_static_sections": {
                lab: float(abs(sigma_static_hinges[i])) for i, lab in enumerate(labels)
            },
            "sigma_dynamic_max_sections": {lab: float(peaks[i]) for i, lab in enumerate(labels)},
            "daf_sections": daf_sections,
            "daf": float(daf_value),
            "max_plastic_strain_rate": float(np.max(hist["plastic_strain_rate"])),
            "final_plastic_strain": {
                lab: float(hist["plastic_states"][i][1]) for i, lab in enumerate(labels)
            },
            "history": hist,
        }