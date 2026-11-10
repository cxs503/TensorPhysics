"""Cowper-Symonds strain-rate dependent plasticity.

The Cowper-Symonds (CS) overstress law relates the *dynamic* flow stress to the
quasi-static one through the accumulated plastic strain rate,

.. math::

    \\sigma_d = \\sigma_0 \\left[1 + \\left(\\frac{\\dot\\epsilon_p}{D}\\right)^{1/p}\\right],

with ``D`` [s^-1] and ``p`` [-] material constants.  For shipbuilding mild steel
the widely used pair is ``D = 40.4 s^-1``, ``p = 5`` (Symonds; tabulated in
Jones, *Structural Impact*, Cambridge UP, 1989, ch. 8, and used throughout the
ice/structure impact literature).

Two deliberate modelling layers are provided:

1. :class:`CowperSymonds` -- the pure overstress factor :math:`A(\\dot\\epsilon_p)`
   together with its analytic inverse.  This is the reference ("closed form")
   side of every verification comparison in the module.

2. :class:`RateDependentJ2` -- a 1-D rate-dependent von Mises return mapping
   (isotropic linear hardening) in which the plastic strain rate driving the CS
   factor is solved *self-consistently* with the plastic strain increment of the
   same time step.  Because the CS factor is evaluated on the plastic (not the
   total) strain rate, the rate-independent limit is exact:
   ``A(0) == 1``, so ``sigma_d -> sigma_0`` as ``\\dot\\epsilon_p -> 0``.

Extension: :class:`CowperSymonds` accepts an optional ``rate_threshold``
``\\dot\\epsilon_th``; below the threshold the amplification is frozen at its
value at the threshold (a standard "rate-insensitive plateau").  This is an
*extension*, not classical CS, and is off by default so that the classical law
can be tested against its own closed form.

References
----------
- N. Jones, *Structural Impact*, Cambridge University Press, 1989 (ch. 8:
  Cowper-Symonds constants for mild steel, ``D=40.4 s^-1``, ``p=5``).
- G.R. Cowper, P.S. Symonds, "Strain-hardening and strain-rate effects in the
  impact loading of cantilever beams", Brown University Report No. 28, 1957.
- J.D. Campbell, R.H. Cooper, "Yield and flow of low-carbon steel at medium
  strain rates", Proc. Inst. Mech. Eng. 180 (1966) -- experimental dynamic /
  static flow-stress ratio of mild steel (~2 at 100 s^-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MILD_STEEL_D",
    "MILD_STEEL_P",
    "CowperSymonds",
    "RateDependentJ2",
    "uniaxial_tension_constant_rate",
    "analytic_uniaxial_flow_stress",
]

#: Cowper-Symonds constant for mild steel [s^-1] (Jones 1989, ch. 8).
MILD_STEEL_D = 40.4
#: Cowper-Symonds exponent for mild steel [-] (Jones 1989, ch. 8).
MILD_STEEL_P = 5.0


@dataclass(frozen=True)
class CowperSymonds:
    """Overstress factor ``A = sigma_d / sigma_0`` of the Cowper-Symonds law.

    Parameters
    ----------
    D:
        Cowper-Symonds constant [s^-1], ``D > 0``.
    p:
        Cowper-Symonds exponent [-], ``p > 0``.
    rate_threshold:
        Optional plastic strain rate [s^-1] below which the amplification is
        clamped (rate-insensitive plateau).  ``None`` reproduces classical CS.

    Notes
    -----
    ``amplification(rate)`` is exactly ``1`` at ``rate == 0``, so the
    rate-independent limit is structural, not a tolerance.
    """

    D: float = MILD_STEEL_D
    p: float = MILD_STEEL_P
    rate_threshold: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.D) or self.D <= 0.0:
            raise ValueError("Cowper-Symonds D must be finite and > 0")
        if not np.isfinite(self.p) or self.p <= 0.0:
            raise ValueError("Cowper-Symonds p must be finite and > 0")
        if self.rate_threshold is not None and self.rate_threshold < 0.0:
            raise ValueError("rate_threshold must be >= 0")

    # -- forward relation ---------------------------------------------------
    def amplification(self, plastic_strain_rate) -> np.ndarray:
        """``A = 1 + (rate/D)^(1/p)``; ``A == 1`` exactly at ``rate == 0``."""
        rate = np.asarray(plastic_strain_rate, dtype=float)
        if np.any(rate < 0.0):
            raise ValueError("plastic strain rate must be >= 0")
        rate = rate.copy()
        if self.rate_threshold is not None:
            rate = np.clip(rate, 0.0, self.rate_threshold)
        return 1.0 + (rate / self.D) ** (1.0 / self.p)

    def dynamic_yield_stress(self, sigma_0, plastic_strain_rate) -> np.ndarray:
        """``sigma_d = sigma_0 * A(rate)``."""
        sigma_0 = np.asarray(sigma_0, dtype=float)
        if np.any(sigma_0 < 0.0):
            raise ValueError("quasi-static stress must be >= 0")
        return sigma_0 * self.amplification(plastic_strain_rate)

    # -- analytic inverse ---------------------------------------------------
    def rate_for_amplification(self, amplification) -> np.ndarray:
        """``rate = D * (A - 1)^p`` (analytic inverse of :meth:`amplification`)."""
        amp = np.asarray(amplification, dtype=float)
        if np.any(amp < 1.0):
            raise ValueError("amplification must be >= 1")
        if self.rate_threshold is not None and np.any(amp > self.amplification(self.rate_threshold)):
            raise ValueError("amplification exceeds the plateau value set by rate_threshold")
        return self.D * (amp - 1.0) ** self.p

    def rate_for_target_amplification(self, target: float) -> float:
        """Rate at which the CS factor reaches ``target`` (``target >= 1``)."""
        return float(self.rate_for_amplification(target))

    # -- diagnostics --------------------------------------------------------
    def amplification_report(self, rates) -> dict:
        """Small JSON-ready table ``{rate: amplification}`` for reporting."""
        rates = np.atleast_1d(np.asarray(rates, dtype=float))
        return {
            "D": self.D,
            "p": self.p,
            "rate_threshold": self.rate_threshold,
            "table": [
                {"rate": float(r), "amplification": float(a)}
                for r, a in zip(rates, self.amplification(rates))
            ],
        }


@dataclass
class UniaxialResponse:
    """Result container for a constant-strain-rate uniaxial tension run."""

    strain: np.ndarray
    stress: np.ndarray
    plastic_strain: np.ndarray
    plastic_strain_rate: np.ndarray
    total_strain_rate: float
    sigma_0: float
    hardening: float
    youngs_modulus: float

    @property
    def final_stress(self) -> float:
        return float(self.stress[-1])

    @property
    def final_plastic_strain_rate(self) -> float:
        return float(self.plastic_strain_rate[-1])

    def steady_state_analytic(self, cs: CowperSymonds) -> float:
        """Closed-form steady-state flow stress of this run.

        For a linear-hardening von Mises material under a constant *total*
        strain rate, the steady state has
        ``eps_p_dot = eps_dot / (1 + H/E)`` and

        .. math:: \\sigma = (\\sigma_0 + H \\epsilon_p)\\, A(\\dot\\epsilon_p).

        With ``H = 0`` and ``E -> inf`` this reduces to the textbook
        ``sigma_0 A(eps_dot)``.
        """
        return float(
            analytic_uniaxial_flow_stress(
                eps_p=self.plastic_strain[-1],
                total_strain_rate=self.total_strain_rate,
                sigma_0=self.sigma_0,
                hardening=self.hardening,
                youngs_modulus=self.youngs_modulus,
                cs=cs,
            )
        )


def analytic_uniaxial_flow_stress(
    eps_p: float,
    total_strain_rate: float,
    sigma_0: float,
    hardening: float,
    youngs_modulus: float,
    cs: CowperSymonds,
) -> float:
    """Closed-form steady-state flow stress (see :meth:`UniaxialResponse.steady_state_analytic`)."""
    eps_p_dot = total_strain_rate / (1.0 + hardening / youngs_modulus)
    return float((sigma_0 + hardening * eps_p) * cs.amplification(eps_p_dot))


class RateDependentJ2:
    """1-D rate-dependent von Mises material with linear isotropic hardening.

    The return mapping solves, per increment, for the plastic strain increment
    ``d_eps_p`` such that

    .. math::
        \\sigma_{tr} - E\\,\\Delta\\epsilon_p
        = \\bigl(\\sigma_0 + H(\\epsilon_p^n + \\Delta\\epsilon_p)\\bigr)\\
          A\\!\\left(\\Delta\\epsilon_p / \\Delta t\\right),

    i.e. the plastic strain rate of the *same step* drives the CS overstress.
    The residual is monotone in ``\\Delta\\epsilon_p``, so a robust bisection is
    used instead of a Newton iteration (no convergence failures).

    Parameters
    ----------
    youngs_modulus:
        Young's modulus ``E`` [Pa].
    sigma_0:
        Initial yield stress [Pa].
    hardening:
        Linear isotropic hardening modulus ``H`` [Pa] (``0`` = perfectly plastic).
    cs:
        Cowper-Symonds law.
    """

    def __init__(
        self,
        youngs_modulus: float = 210.0e9,
        sigma_0: float = 235.0e6,
        hardening: float = 0.0,
        cs: CowperSymonds | None = None,
        bisection_steps: int = 90,
    ) -> None:
        if youngs_modulus <= 0.0:
            raise ValueError("youngs_modulus must be > 0")
        if sigma_0 <= 0.0:
            raise ValueError("sigma_0 must be > 0")
        if hardening < 0.0:
            raise ValueError("hardening must be >= 0")
        self.E = float(youngs_modulus)
        self.sigma_0 = float(sigma_0)
        self.H = float(hardening)
        self.cs = cs if cs is not None else CowperSymonds()
        self.bisection_steps = int(bisection_steps)

    # -- helpers -----------------------------------------------------------
    def yield_stress(self, eps_p, eps_p_dot=0.0) -> np.ndarray:
        """Rate-dependent current yield stress."""
        eps_p = np.asarray(eps_p, dtype=float)
        return (self.sigma_0 + self.H * eps_p) * self.cs.amplification(eps_p_dot)

    def return_map(self, stress_trial, eps_p_n, dt) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Radial-return update of a 1-D stress state.

        Parameters
        ----------
        stress_trial:
            Elastic predictor of the axial stress ``E (eps - eps_p_n)``.
        eps_p_n:
            Plastic strain at the start of the step (signed magnitude, >= 0).
        dt:
            Time step [s].

        Returns
        -------
        (stress, eps_p, eps_p_dot)
        """
        s_tr = np.atleast_1d(np.asarray(stress_trial, dtype=float))
        eps_p_n_arr = np.broadcast_to(np.atleast_1d(np.asarray(eps_p_n, dtype=float)), s_tr.shape)
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        sign = np.sign(s_tr)
        sign[sign == 0.0] = 1.0
        mag = np.abs(s_tr)
        yield_now = self.sigma_0 + self.H * eps_p_n_arr

        # Elastic branch: no overstress even at zero rate.
        elastic = mag <= yield_now
        d_eps_p = np.zeros_like(mag)

        # Bisection on d_eps_p in [0, |s_tr|/E] for the plastic branch.
        lo = np.zeros_like(mag)
        hi = mag / self.E
        for _ in range(self.bisection_steps):
            mid = 0.5 * (lo + hi)
            stress_mid = mag - self.E * mid
            capacity = (self.sigma_0 + self.H * (eps_p_n_arr + mid)) * self.cs.amplification(
                mid / dt
            )
            over = stress_mid > capacity  # still too strong -> more plasticity
            lo = np.where(over, mid, lo)
            hi = np.where(over, hi, mid)
        d_eps_p = np.where(elastic, 0.0, 0.5 * (lo + hi))

        eps_p_new = eps_p_n_arr + d_eps_p
        stress_mag = mag - self.E * d_eps_p
        return sign * stress_mag, eps_p_new, d_eps_p / dt

    # -- scalar fast path (used by the uniaxial driver) ---------------------
    def _amp_scalar(self, rate: float) -> float:
        if rate <= 0.0:
            return 1.0
        r = rate if self.cs.rate_threshold is None else min(rate, self.cs.rate_threshold)
        return 1.0 + (r / self.cs.D) ** (1.0 / self.cs.p)

    def return_map_scalar(self, stress_trial: float, eps_p_n: float, dt: float):
        """Scalar version of :meth:`return_map` (same equations, no array overhead)."""
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        sign = 1.0 if stress_trial >= 0.0 else -1.0
        mag = abs(stress_trial)
        yield_now = (self.sigma_0 + self.H * eps_p_n) * self._amp_scalar(0.0)
        if mag <= yield_now:
            return stress_trial, eps_p_n, 0.0
        lo, hi = 0.0, mag / self.E
        for _ in range(self.bisection_steps):
            mid = 0.5 * (lo + hi)
            stress_mid = mag - self.E * mid
            capacity = (self.sigma_0 + self.H * (eps_p_n + mid)) * self._amp_scalar(mid / dt)
            if stress_mid > capacity:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-16 * max(1.0, mag / self.E):
                break
        d_eps_p = 0.5 * (lo + hi)
        return sign * (mag - self.E * d_eps_p), eps_p_n + d_eps_p, d_eps_p / dt

    # -- drivers -----------------------------------------------------------
    def uniaxial_tension_constant_rate(
        self,
        total_strain_rate: float,
        strain_max: float,
        n_steps: int = 400,
    ) -> UniaxialResponse:
        """Integrate uniaxial tension at a constant *total* strain rate."""
        if total_strain_rate <= 0.0:
            raise ValueError("total_strain_rate must be > 0")
        if strain_max <= 0.0:
            raise ValueError("strain_max must be > 0")
        dt = strain_max / total_strain_rate / n_steps
        strain = np.zeros(n_steps + 1)
        stress = np.zeros(n_steps + 1)
        eps_p = np.zeros(n_steps + 1)
        rate = np.zeros(n_steps + 1)

        eps_n, eps_p_n = 0.0, 0.0
        for i in range(n_steps):
            eps_n = eps_n + total_strain_rate * dt
            s_tr = self.E * (eps_n - eps_p_n)
            _s, _ep, _epd = self.return_map_scalar(s_tr, eps_p_n, dt)
            stress[i + 1] = _s
            eps_p[i + 1] = _ep
            rate[i + 1] = _epd
            eps_p_n = _ep
            strain[i + 1] = eps_n
        return UniaxialResponse(
            strain=strain,
            stress=stress,
            plastic_strain=eps_p,
            plastic_strain_rate=rate,
            total_strain_rate=float(total_strain_rate),
            sigma_0=self.sigma_0,
            hardening=self.H,
            youngs_modulus=self.E,
        )

def uniaxial_tension_constant_rate(
    total_strain_rate: float,
    strain_max: float = 0.02,
    n_steps: int = 600,
    youngs_modulus: float = 210.0e9,
    sigma_0: float = 235.0e6,
    hardening: float = 0.0,
    cs: CowperSymonds | None = None,
) -> UniaxialResponse:
    """Convenience wrapper: rate-dependent uniaxial tension at constant total strain rate."""
    material = RateDependentJ2(
        youngs_modulus=youngs_modulus, sigma_0=sigma_0, hardening=hardening, cs=cs
    )
    return material.uniaxial_tension_constant_rate(total_strain_rate, strain_max, n_steps)
