"""Flutter prediction: time-domain onset search and comparison with theory.

The time-domain route used here is the *coupled* one: for a given free-stream
speed the structural model is marched with the fluid load produced by the
Glauert vortex lattice (:mod:`tensorphysics.aeroelastic.uvlm`), the decay (or
growth) rate of the response is measured from a windowed RMS envelope, and the
critical speed ``V_f`` is bracketed by bisection on that rate.  The reference is
the Theodorsen p-k solution of :mod:`tensorphysics.aeroelastic.theodorsen`;
the two routes share no discretisation, which is what makes the agreement
informative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coupling import run_coupled
from .section import TypicalSection
from .theodorsen import pk_flutter
from .uvlm import GlauertVortexLattice
from .wagner import RationalAero


@dataclass
class Case:
    """A 2-D aeroelastic section case (SI units, per unit span)."""

    name: str = "naca0012-typical-section"
    semi_chord: float = 0.5
    mass: float = 20.0
    x_alpha: float = 0.2
    r_alpha2: float = 0.25
    elastic_axis: float = -0.4
    omega_h: float = 40.0
    omega_alpha: float = 80.0
    rho: float = 1.225
    zeta: float = 0.0

    def section(self):
        return TypicalSection(
            mass=self.mass, semi_chord=self.semi_chord,
            omega_h=self.omega_h, omega_alpha=self.omega_alpha,
            x_alpha=self.x_alpha, r_alpha2=self.r_alpha2,
            elastic_axis=self.elastic_axis,
            zeta_h=self.zeta, zeta_alpha=self.zeta)

    def aero(self, n_terms=20, wake_length_chords=20.0):
        """Fluid model: Wagner/Jones rational (default) or spectral UVLM."""
        return self.aero_model("rational")

    def aero_model(self, model="rational", n_terms=20, wake_length_chords=20.0):
        if model == "uvlm":
            return GlauertVortexLattice(
                semi_chord=self.semi_chord, n_terms=n_terms, rho=self.rho,
                elastic_axis=self.elastic_axis,
                wake_length_chords=wake_length_chords)
        return RationalAero(semi_chord=self.semi_chord, rho=self.rho,
                            elastic_axis=self.elastic_axis)

    def as_dict(self):
        return dict(self.__dict__)


def envelope_growth(t, x, t_start=None, n_windows=8):
    """Growth rate (1/s) from a windowed RMS envelope of the response."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t_start is not None:
        m = t >= t_start
        t, x = t[m], x[m]
    if t.size < 4 * n_windows:
        return 0.0
    edges = np.linspace(t[0], t[-1], n_windows + 1)
    tc, amp = [], []
    for i in range(n_windows):
        m = (t >= edges[i]) & (t <= edges[i + 1])
        if m.sum() < 3:
            continue
        rms = float(np.sqrt(np.mean(x[m] ** 2)))
        if rms <= 0.0:
            continue
        tc.append(0.5 * (edges[i] + edges[i + 1]))
        amp.append(np.log(rms))
    if len(tc) < 3:
        return 0.0
    slope, _ = np.polyfit(np.array(tc), np.array(amp), 1)
    return float(slope)


def time_domain_run(case, U, dt, t_end, scheme="tight", n_terms=20,
                    alpha0=1e-4, relax="newton", model="rational",
                    diverge_factor=1e8, **kwargs):
    aero = case.aero_model(model, n_terms=n_terms)
    section = case.section()
    run = run_coupled(aero, section, U, dt, t_end, scheme=scheme,
                      q0=(0.0, alpha0), relax=relax,
                      diverge_factor=diverge_factor, **kwargs)
    run.growth = envelope_growth(run.t, run.q[:, 1], t_start=0.4 * t_end)
    run.growth_h = envelope_growth(run.t, run.q[:, 0], t_start=0.4 * t_end)
    run.U = U
    run.section = section
    run.aero = aero
    return run


def flutter_speed(case, steps_per_cycle=240, cycles=14, scheme="tight",
                  n_terms=20, U_lo=None, U_hi=None, tol=1e-3,
                  max_bisect=14, relax="newton", verbose=False,
                  model="rational"):
    """Bisection on the growth rate to locate the flutter onset speed."""
    section = case.section()
    omega_ref = 0.5 * (case.omega_h + case.omega_alpha)
    T_ref = 2.0 * np.pi / omega_ref
    dt = T_ref / steps_per_cycle
    t_end = cycles * T_ref

    def growth(U):
        run = time_domain_run(case, U, dt, t_end, scheme=scheme,
                              n_terms=n_terms, relax=relax, model=model)
        return run.growth, run

    if U_lo is None:
        U_lo = 0.2 * case.semi_chord * case.omega_alpha
    if U_hi is None:
        U_hi = 20.0 * case.semi_chord * case.omega_alpha
    g_lo, run_lo = growth(U_lo)
    if g_lo >= 0.0:
        return {"U_f": None, "reason": "unstable at U_lo", "dt": dt,
                "t_end": t_end}
    g_hi, run_hi = growth(U_hi)
    n_probe = 0
    while g_hi < 0.0 and n_probe < 8:
        U_hi *= 1.6
        g_hi, run_hi = growth(U_hi)
        n_probe += 1
    if g_hi < 0.0:
        return {"U_f": None, "reason": "no unstable bound found", "dt": dt,
                "t_end": t_end}
    for _ in range(max_bisect):
        U_mid = 0.5 * (U_lo + U_hi)
        g_mid, run_mid = growth(U_mid)
        if verbose:
            print(f"   U={U_mid:.4f} growth={g_mid:+.4e} "
                  f"iters={run_mid.iterations.max()} conv={run_mid.converged}")
        if g_mid >= 0.0:
            U_hi, g_hi, run_hi = U_mid, g_mid, run_mid
        else:
            U_lo, g_lo, run_lo = U_mid, g_mid, run_mid
        if (U_hi - U_lo) / U_lo < tol:
            break
    U_f = 0.5 * (U_lo + U_hi)
    run = time_domain_run(case, U_f, dt, t_end, scheme=scheme,
                          n_terms=n_terms, relax=relax, model=model)
    omega_f = flutter_frequency(run)
    k_f = omega_f * case.semi_chord / U_f
    return {"U_f": float(U_f), "omega_f": float(omega_f), "k_f": float(k_f),
            "growth_at_Uf": float(run.growth), "dt": float(dt),
            "t_end": float(t_end), "converged": bool(run.converged),
            "max_iterations": int(run.iterations.max()),
            "run": run, "scheme": scheme,
            "U_bracket": (float(U_lo), float(U_hi))}


def flutter_frequency(run):
    """Dominant response frequency (rad/s) from the recorded pitch signal."""
    t = np.asarray(run.t, dtype=float)
    x = np.asarray(run.q[:, 1], dtype=float)
    if t.size < 8:
        return 0.0
    m = t >= 0.4 * t[-1]
    t, x = t[m], x[m]
    x = x - np.mean(x)
    zc = np.where(np.diff(np.sign(x)) != 0)[0]
    if zc.size > 2:
        # linear interpolation of the crossings gives the mean half period
        tt = t[zc] + (t[zc + 1] - t[zc]) * np.abs(x[zc]) / (
            np.abs(x[zc]) + np.abs(x[zc + 1]) + 1e-300)
        per = 2.0 * float(np.mean(np.diff(tt)))
        if per > 0:
            return float(2.0 * np.pi / per)
    dt = float(np.mean(np.diff(t)))
    xr = x - x.mean()
    spec = np.abs(np.fft.rfft(xr * np.hanning(xr.size)))
    freqs = np.fft.rfftfreq(xr.size, d=dt)
    if spec.size < 3:
        return 0.0
    return float(2.0 * np.pi * freqs[int(np.argmax(spec[1:]) + 1)])


def theodorsen_reference(case, u_alpha_sign=None):
    """Critical speed/frequency from the Theodorsen p-k eigenvalue solution."""
    section = case.section()
    if u_alpha_sign is None:
        u_alpha_sign = _CONVENTION["u_alpha_sign"]
    out = pk_flutter(section, case.rho, u_alpha_sign=u_alpha_sign)
    out["u_alpha_sign"] = u_alpha_sign
    return out


#: pinned by comparison with the time-domain solver (see
#: ``benchmarks/aeroelastic/run_fsi_flutter_benchmark.py``); updated at import
#: time by :func:`set_convention`.
_CONVENTION = {"u_alpha_sign": 1.0}


def set_convention(u_alpha_sign):
    _CONVENTION["u_alpha_sign"] = float(u_alpha_sign)
    return _CONVENTION["u_alpha_sign"]