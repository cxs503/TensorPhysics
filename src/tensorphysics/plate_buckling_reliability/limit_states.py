"""Limit states of the unified chain: deterministic capacity inside ``g(X)``.

The unified framework makes the *deterministic* ultimate strength the core of
the probabilistic limit-state function:

.. math::
    g_{ult}(X) = \\sigma_u\\!\\big(t_{nom} - \\Delta t_c,\\; f_y\\big)
                 - k_{mod}\\big(\\sigma_{global} + \\psi\\, p_{ice}\\big)

with the random vector

``X = (p_ice, f_y, dt_corr, k_mod)``

i.e. annual-maximum local ice pressure (Gumbel), yield strength (lognormal),
corrosion/abrasion margin (normal, deducted from the as-built thickness) and a
model-uncertainty factor (lognormal).  Two further modes of the same panel form
a series (weakest-link) system: first-yield of the net section and an excessive
deflection (serviceability) mode.  A panel fails if any mode is violated --
Ditlevsen bounds are used for the system probability (see
:mod:`.reliability`).

Units of ``X`` follow the W3-N04 variable set (``p_ice`` MPa, ``f_y`` MPa,
``dt_corr`` mm) because those marginals are shared with the structural
reliability core; the limit states convert to SI internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .panel import PanelSpec
from .surrogate import UltimateStrengthSurrogate

__all__ = ["PanelLimitStates", "MODE_ULTIMATE", "MODE_YIELD", "MODE_ELASTIC",
           "MODE_DEFLECTION", "DEFAULT_MODES", "VAR_NAMES"]

MODE_ULTIMATE = "ultimate"
MODE_YIELD = "yield"
MODE_ELASTIC = "elastic_buckling"
MODE_DEFLECTION = "deflection"

#: Modes of the default series (weakest-link) system.  Only the two ultimate
#: limit states compete: the arc-length limit load (post-buckling reserve
#: included) and first yield of the net section.  ``MODE_ELASTIC`` and
#: ``MODE_DEFLECTION`` are computed for diagnosis but are *not* failure modes of
#: this panel -- plate buckling is an onset event with a large post-buckling
#: reserve, and the uniform-pressure deflection mode is inconsistent with the
#: patch-like ice load (both are quantified in the W4-N03 report).
DEFAULT_MODES = (MODE_ULTIMATE, MODE_YIELD)
VAR_NAMES = ("p_ice", "fy", "dt_corr", "k_mod")


@dataclass
class PanelLimitStates:
    """Limit-state functions of one panel at a nominal (as-built) thickness.

    Parameters
    ----------
    panel:
        Panel specification; ``panel.t`` is the as-built thickness, from which
        the corrosion margin is deducted.
    surrogate:
        FE-calibrated ultimate-strength response surface.
    deflection_limit_ratio:
        Serviceability limit on the elastic centre deflection as a fraction of
        the frame span (``b/100``, ISO 19906 / DNV plating practice).
    navier_terms:
        Odd half-wave terms of the Navier series used for the deflection mode
        (the geometric coefficient is pre-summed, so evaluation is O(1)).
    """

    panel: PanelSpec
    surrogate: UltimateStrengthSurrogate
    deflection_limit_ratio: float = 0.01
    navier_terms: int = 9

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        self._b = float(self.panel.b)
        self._a = float(self.panel.a)
        self._E = float(self.panel.E)
        self._nu = float(self.panel.nu)
        self._sigma_global = float(self.panel.sigma_global)
        self._psi = float(self.panel.psi_membrane)
        self._t_nom = float(self.panel.t)
        self._w_limit = self.deflection_limit_ratio * self._b
        #: Navier coefficient: ``w_centre = q * C_geom / D`` for a SSSS plate.
        self._c_geom = self._navier_coefficient(self.navier_terms)

    def _navier_coefficient(self, n_terms: int) -> float:
        a, b = self._a, self._b
        total = 0.0
        for m in range(1, 2 * n_terms, 2):
            for n in range(1, 2 * n_terms, 2):
                if m % 2 == 1 and n % 2 == 1:
                    total += (16.0 / np.pi**6) / (
                        m * n * (m * m / a**2 + n * n / b**2) ** 2
                    ) * np.sin(m * np.pi / 2.0) * np.sin(n * np.pi / 2.0)
        return total

    # ---------------------------------------------------------------- physics
    def net_thickness(self, dt_corr_mm):
        """As-built thickness minus the corrosion margin (m)."""
        return self._t_nom - np.asarray(dt_corr_mm, dtype=float) * 1e-3

    def capacity(self, fy_mpa, dt_corr_mm):
        """Ultimate compressive stress from the FE-calibrated surrogate (Pa)."""
        t = self.net_thickness(dt_corr_mm)
        return self.surrogate.sigma_u_batch(t, np.asarray(fy_mpa, dtype=float) * 1e6,
                                            self._b, self._E)

    def demand(self, p_ice_mpa, k_mod):
        """Applied membrane compression from ice pressure and global stress (Pa)."""
        return np.asarray(k_mod, dtype=float) * (
            self._sigma_global + self._psi * np.asarray(p_ice_mpa, dtype=float) * 1e6)

    def deflection(self, p_ice_mpa, dt_corr_mm, k_mod):
        """Elastic centre deflection under the local pressure (m)."""
        t = self.net_thickness(dt_corr_mm)
        rigidity = self._E * t**3 / (12.0 * (1.0 - self._nu**2))
        q = np.asarray(k_mod, dtype=float) * np.asarray(p_ice_mpa, dtype=float) * 1e6
        return q * self._c_geom / rigidity

    def elastic_critical_stress(self, dt_corr_mm, k_factor: float = 4.0):
        """Elastic plate buckling stress ``C pi^2 E/(12(1-nu^2)) (t/b)^2`` (Pa).

        ``C = 4`` is the DNV-RP-C201 / Timoshenko coefficient of a simply
        supported, longitudinally compressed plate with ``a/b >= 1``.  Kept
        analytic (no eigensolve) because the mode is evaluated inside the FORM
        iterations and the Monte-Carlo batches.
        """
        t = self.net_thickness(dt_corr_mm)
        return k_factor * np.pi**2 * self._E / (12.0 * (1.0 - self._nu**2)) * (t / self._b)**2

    # ------------------------------------------------------------ limit states
    def g_elastic(self, x: Sequence[float]) -> float:
        p_ice, fy, dt_corr, k_mod = x
        return float(self.elastic_critical_stress(dt_corr) - self.demand(p_ice, k_mod))

    def g_ultimate(self, x: Sequence[float]) -> float:
        p_ice, fy, dt_corr, k_mod = x
        return float(self.capacity(fy, dt_corr) - self.demand(p_ice, k_mod))

    def g_yield(self, x: Sequence[float]) -> float:
        p_ice, fy, dt_corr, k_mod = x
        return float(fy * 1e6 - self.demand(p_ice, k_mod))

    def g_deflection(self, x: Sequence[float]) -> float:
        p_ice, fy, dt_corr, k_mod = x
        return float(self._w_limit - self.deflection(p_ice, dt_corr, k_mod))

    # --------------------------------------------------------- batch versions
    def batch_limit_states(self) -> Dict[str, object]:
        """NumPy-vectorised limit states ``f(x)`` with ``x`` of shape (n, 4)."""
        def split(x):
            arr = np.asarray(x, dtype=float)
            return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        def g_ult(x):
            p_ice, fy, dt, k = split(x)
            return self.capacity(fy, dt) - self.demand(p_ice, k)

        def g_y(x):
            p_ice, fy, dt, k = split(x)
            return fy * 1e6 - self.demand(p_ice, k)

        def g_d(x):
            p_ice, fy, dt, k = split(x)
            return self._w_limit - self.deflection(p_ice, dt, k)

        def g_e(x):
            p_ice, fy, dt, k = split(x)
            return self.elastic_critical_stress(dt) - self.demand(p_ice, k)

        return {MODE_ULTIMATE: g_ult, MODE_YIELD: g_y, MODE_ELASTIC: g_e,
                MODE_DEFLECTION: g_d}

    def limit_states(self) -> Dict[str, object]:
        return {
            MODE_ULTIMATE: self.g_ultimate,
            MODE_YIELD: self.g_yield,
            MODE_ELASTIC: self.g_elastic,
            MODE_DEFLECTION: self.g_deflection,
        }

    # ------------------------------------------------------------------ misc
    def design_margin(self, x=None) -> Dict[str, float]:
        """Limit-state values at the *mean* realisation (diagnostic only)."""
        x_mean = (2.5, 313.0, 2.0, 1.0) if x is None else tuple(x)
        return {name: float(fn(x_mean)) for name, fn in self.limit_states().items()}

    def with_thickness(self, t: float) -> "PanelLimitStates":
        """Same problem at another as-built thickness (design iteration)."""
        return PanelLimitStates(self.panel.with_thickness(t), self.surrogate,
                                deflection_limit_ratio=self.deflection_limit_ratio,
                                navier_terms=self.navier_terms)