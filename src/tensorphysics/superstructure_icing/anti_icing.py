"""Electro-thermal anti-/de-icing design optimisation.

The protected surface strip (arc length ``s``, unit span) carries a heater power
density field ``q(s)`` [W/m2].  Two pieces of physics connect the heater power to
the ice that remains after a given exposure:

1. **Skin conduction (thermal spreading).**  The added heat does not stay where
   it is generated: the panel conducts it along the surface.  For a unit heat
   input at cell ``i`` the steady fin equation

   .. math:: -k_s t_s \\frac{d^2T}{ds^2} + h_c T = q(s)

   is solved on the strip and the fraction of that heat convected away from each
   cell is collected into a spreading matrix ``W`` with unit column sums
   (energy conserving: every watt injected is finally convected out).

2. **Messinger icing with heat input.**  The effective heat ``q_eff = W q`` enters
   the Messinger balance; heating raises the surface temperature, lowers the
   freezing fraction and therefore the ice thickness accumulated over
   ``t_end``.

The design problem solved here is

   minimise   ``P = sum_i A_i q_i``
   subject to ``h_ice,i(t_end) <= h_allow``  (default 5 mm, ISO 15138:2017
   constraints for protected surfaces) and ``q_i >= 0``.

The *uniform* baseline sizes every heater to the worst segment
(``q_i = q_uniform`` satisficing the binding constraint), which is what a
non-optimised installation does; the saving is ``1 - P_opt/P_uniform``.

Both a continuous optimum (SLSQP) and a SIMP-style 0/1 material-layout
optimisation (density variable with ``q = q_max rho^p``, ``p = 3``) are provided;
the latter is the shape/topology-optimisation pathway requested for heater
layout (heat-on / heat-off bands rather than a linear power schedule).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import air
from .messinger import heat_transfer_coefficient, ice_growth_1d, ice_growth_steady


@dataclass
class HeaterDesign:
    """A heater power distribution and its icing performance."""

    s: np.ndarray
    q: np.ndarray
    h_ice: np.ndarray
    power_w: float
    h_allow: float
    label: str = ""
    converged: bool = False
    n_iter: int = 0
    history: list = field(default_factory=list)

    @property
    def constraint_violation(self) -> float:
        return float(max(np.max(self.h_ice - self.h_allow), 0.0))

    @property
    def power_w_m2_mean(self) -> float:
        ds = np.gradient(self.s) if self.s.size > 1 else np.array([1.0])
        return float(self.power_w / max(np.sum(ds), 1e-30))


class ElectrothermalAntiIcing:
    """Optimisation driver for a heater strip on an iced section."""

    def __init__(
        self,
        s,
        beta,
        U: float,
        T_air: float,
        LWC: float,
        t_end: float = 600.0,
        dt: float = 10.0,
        h_c: float | None = None,
        L_char: float | None = None,
        k_skin: float = 50.0,
        t_skin: float = 0.006,
        kind: str = "cylinder",
        q_cap: float = 12000.0,
    ):
        self.s = np.atleast_1d(np.asarray(s, dtype=float))
        self.beta = np.broadcast_to(np.atleast_1d(np.asarray(beta, dtype=float)), self.s.shape).copy()
        self.U = U
        self.T_air = T_air
        self.LWC = LWC
        self.t_end = t_end
        self.dt = dt
        self.L_char = float(L_char if L_char is not None else max(2.0 * self.s.max(), 0.05))
        self.h_c = float(h_c if h_c is not None else heat_transfer_coefficient(U, self.L_char, T_air, kind=kind))
        self.k_skin = k_skin
        self.t_skin = t_skin
        self.kind = kind
        # installed heater rating: electro-thermal anti-icing mats are specified
        # at 2-12 kW/m2 (declared hardware constraint, not a fitted parameter)
        self.q_cap = float(q_cap)
        if self.s.size > 1:
            self.ds = np.gradient(self.s)
        else:
            self.ds = np.array([1e-3])
        self.area = self.ds.copy()  # per unit span
        self.W = self._spreading_matrix()

    # -- thermal spreading -------------------------------------------------
    def _spreading_matrix(self) -> np.ndarray:
        n = self.s.size
        if n == 1:
            return np.ones((1, 1))
        g = self.k_skin * self.t_skin / np.maximum(np.diff(self.s), 1e-9)  # W/K per unit span
        diag = np.zeros(n)
        off = np.zeros(n - 1)
        diag[0] = g[0]
        diag[-1] = g[-1]
        for i in range(n - 1):
            diag[i] += g[i]
            diag[i + 1] += g[i]
            off[i] = -g[i]
        diag += self.h_c * self.ds  # convective boundary
        M = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        W = np.zeros((n, n))
        M_inv = np.linalg.inv(M)
        for i in range(n):
            T = M_inv[:, i]  # response to unit power injected at cell i (per unit span)
            loss = self.h_c * self.ds * T
            W[:, i] = loss / np.sum(loss)  # energy-conserving redistribution
        return W

    # -- icing model -------------------------------------------------------
    def ice_thickness(self, q) -> np.ndarray:
        """Ice thickness after ``t_end`` for the power density field ``q``.

        Uses the quasi-steady Messinger solution (time-independent freezing
        fraction), which is algebraically identical to the transient solver for
        constant ambient conditions -- see
        :func:`tensorphysics.superstructure_icing.messinger.ice_growth_steady`.
        """
        q = np.atleast_1d(np.asarray(q, dtype=float))
        q_eff = self.W @ q
        out = ice_growth_steady(
            self.s,
            self.beta,
            self.U,
            self.T_air,
            self.LWC,
            self.t_end,
            h_c=self.h_c,
            L_char=self.L_char,
            q_heat=q_eff,
        )
        return out["h"]

    def power(self, q) -> float:
        return float(np.sum(np.atleast_1d(q) * self.area))

    def max_protective_power_density(self) -> float:
        """Power density that keeps the whole strip ice-free (upper bound)."""
        # physical first guess: convective + evaporative loss at T_s = 0
        lo, hi = 0.0, max(2000.0, 6.0 * self.h_c * max(-self.T_air, 1.0))
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            if np.max(self.ice_thickness(np.full(self.s.size, mid))) > 0.0:
                lo = mid
            else:
                hi = mid
        return hi

    # -- designs -----------------------------------------------------------
    def uniform_design(self, h_allow: float = 5e-3) -> HeaterDesign:
        """Baseline: one power density for every heater, sized to the worst cell."""
        lo, hi = 0.0, self.max_protective_power_density() * 1.2 + 1.0
        best = None
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            h = self.ice_thickness(np.full(self.s.size, mid))
            if np.max(h) > h_allow:
                lo = mid
            else:
                hi = mid
                best = h
        q = np.full(self.s.size, hi)
        h = self.ice_thickness(q)
        return HeaterDesign(self.s, q, h, self.power(q), h_allow, label="uniform", converged=True)

    def optimize(self, h_allow: float = 5e-3, q_scale: float | None = None, maxiter: int = 400) -> HeaterDesign:
        """Continuous constrained optimum (SLSQP) in normalised variables.

        Design variables ``x_i in [0, 1]`` with ``q_i = x_i q_max`` (well scaled,
        which matters: on raw W/m2 variables the default finite-difference step
        of SLSQP is far below the resolution of the icing solver and the
        gradients become noise).  Objective ``sum A_i q_i`` (kW), constraints
        ``h_allow - h_i(q) >= 0`` (mm).
        """
        from scipy.optimize import minimize

        n = self.s.size
        q_max = q_scale if q_scale is not None else self.q_cap
        history: list[float] = []
        margin = 0.98 * h_allow  # design margin: reported design stays feasible

        def q_of(x):
            return np.clip(x, 0.0, 1.0) * q_max

        def obj(x):
            p = self.power(q_of(x)) / 1000.0
            history.append(self.power(q_of(x)))
            return p

        def cons(x):
            return (margin - self.ice_thickness(q_of(x))) * 1e3

        res = minimize(
            obj,
            np.full(n, 0.5),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "ineq", "fun": cons}],
            options={"maxiter": maxiter, "ftol": 1e-10, "eps": 2e-3},
        )
        q = q_of(res.x)
        h = self.ice_thickness(q)
        n_repair = 0
        while np.max(h) > h_allow and n_repair < 80:
            q = q * 1.01
            h = self.ice_thickness(q)
            n_repair += 1
        return HeaterDesign(
            self.s,
            q,
            h,
            self.power(q),
            h_allow,
            label="optimized",
            converged=bool(np.max(h) <= h_allow),
            n_iter=int(res.nit) + n_repair,
            history=history,
        )

    def simp_design(
        self,
        h_allow: float = 5e-3,
        p: float = 3.0,
        q_max: float | None = None,
        maxiter: int = 60,
        rho_threshold: float = 0.5,
    ) -> HeaterDesign:
        """SIMP-style 0/1 heater-layout optimisation.

        Density variable ``rho_i in [0, 1]`` with power density
        ``q_i = q_max rho_i^p`` (``p = 3`` penalisation, as in the classic SIMP
        interpolation).  The optimised density field is then projected to a
        discrete layout (``rho > rho_threshold`` -> heated at full power) which
        is what a real installation would build; the reported power is that of
        the *projected* layout, and its ice thickness is re-checked.
        """
        from scipy.optimize import minimize

        n = self.s.size
        q_max = q_max if q_max is not None else self.q_cap
        history: list[float] = []

        def q_of(rho):
            return q_max * np.clip(rho, 0.0, 1.0) ** p

        def obj(rho):
            p_tot = self.power(q_of(rho))
            history.append(p_tot)
            return p_tot

        def cons(rho):
            return h_allow - self.ice_thickness(q_of(rho))

        rho0 = np.full(n, 0.5)
        res = minimize(
            obj,
            rho0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "ineq", "fun": cons}],
            options={"maxiter": maxiter, "ftol": 1e-6},
        )
        rho = np.clip(res.x, 0.0, 1.0)
        binary = (rho > rho_threshold).astype(float)
        q_bin = q_max * binary
        h_bin = self.ice_thickness(q_bin)
        # if the projected layout violates the constraint, raise the switched-on
        # heaters until it is satisfied (documented in the report)
        scale = 1.0
        while np.max(h_bin) > h_allow and scale < 20.0:
            scale *= 1.05
            q_bin = q_max * binary * scale
            h_bin = self.ice_thickness(q_bin)
        return HeaterDesign(
            self.s,
            q_bin,
            h_bin,
            self.power(q_bin),
            h_allow,
            label="simp_binary",
            converged=bool(res.success),
            n_iter=int(res.nit),
            history=history,
        )


def power_saving(design: HeaterDesign, baseline: HeaterDesign) -> float:
    """Relative power saving of ``design`` against the uniform baseline (%).

    Returns ``0.0`` when the baseline needs no power at all (the constraint is
    satisfied unheated, so there is nothing to optimise -- reported as such, not
    as a spurious 100 %).
    """
    if baseline.power_w <= 1e-9:
        return 0.0
    return float(100.0 * (baseline.power_w - design.power_w) / baseline.power_w)