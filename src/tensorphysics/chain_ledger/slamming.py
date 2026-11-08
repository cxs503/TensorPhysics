"""Idealised Wagner-type slamming pressure field (the source of interface C).

Physics
-------
Water entry of a section with deadrise angle ``beta`` produces a spray root that
moves outwards at ``d c / dt = (pi/2) V / tan(beta)`` (Wagner).  The pressure is
peaked at the spray root and decays towards the keel; the classic singular
distribution is replaced here by the bounded idealisation

    p(x, xi, t) = p_peak(t) * s_x(x; x_s, c_x(t)) * s_xi(xi; c_xi(t)),
    s(u; c) = sqrt(max(0, 1 - (u/c)^2)),
    p_peak(t) = 0.5 rho V_rel(t)^2 * C_p,      C_p = pi / tan(beta).

``x`` is the station along the hull, ``xi`` the coordinate across the section
(``xi = 0`` at the keel); the vertical load per unit hull length is
``f(x, t) = int p d xi`` because the inclined-plate area element and the
``cos(beta)`` projection of the normal cancel exactly.

Evidence level
--------------
APPROXIMATE-IDEALISATION: the deadrise coefficient ``C_p`` and the bounded shape
function are the classical Wagner scalings, not a solution of the exact water
entry problem; there is no gravity/spray-jet/physics-of-air coupling and no
compressibility.  The chain gates only require that the *handed-over* force is
conserved across the interfaces, so the absolute accuracy of ``C_p`` does not
enter any verdict - it is a model input, reported with its provenance.

The two-dimensional shape ``s_x * s_xi`` is what makes interfaces C's two
integration routes (full-field virtual work versus strip line load) genuinely
different numerical paths rather than the same one-dimensional integral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WagnerSlamming", "contact_half_width_history"]


@dataclass(frozen=True)
class WagnerSlamming:
    """Bounded Wagner-type slamming pressure field.

    Parameters
    ----------
    rho : float
        Water density [kg/m^3].
    deadrise_deg : float
        Deadrise angle of the impacting bottom [deg]; controls ``C_p``.
    keel_depth : float
        Depth of the keel below the still-water plane [m] (``z`` positive up).
    decay_length : float
        Along-hull decay length of the impact patch [m].
    max_half_width : float
        Cap on the across-section contact half-width [m].
    max_patch_half_length : float
        Cap on the along-hull contact half-length [m].
    """

    rho: float = 1025.0
    deadrise_deg: float = 10.0
    keel_depth: float = 0.3
    decay_length: float = 0.9
    max_half_width: float = 0.45
    max_patch_half_length: float = 1.5

    @property
    def pressure_coefficient(self) -> float:
        """Wagner deadrise coefficient ``C_p = pi / tan(beta)``."""
        beta = np.deg2rad(self.deadrise_deg)
        if beta <= 0.0:
            raise ValueError("deadrise must be positive for a finite C_p")
        return float(np.pi / np.tan(beta))

    @property
    def width_rate(self) -> float:
        """``d c_xi / dt`` per unit entry velocity (Wagner)."""
        return float(0.5 * np.pi / np.tan(np.deg2rad(self.deadrise_deg)))

    def peak_pressure(self, v_rel) -> np.ndarray:
        """Peak (spray-root) pressure [Pa] for a relative entry velocity."""
        v = np.asarray(v_rel, dtype=float)
        return 0.5 * self.rho * v * v * self.pressure_coefficient

    def contact_half_width(self, t: np.ndarray, v_rel: np.ndarray) -> np.ndarray:
        """Across-section contact half-width ``c_xi(t)`` [m] by integrating c_dot."""
        t = np.asarray(t, dtype=float)
        v = np.asarray(v_rel, dtype=float)
        c = np.zeros_like(t)
        for i in range(1, t.size):
            dt = t[i] - t[i - 1]
            c[i] = c[i - 1] + self.width_rate * max(v[i - 1], 0.0) * dt
        return np.minimum(c, self.max_half_width)

    def patch_half_length(self, x: np.ndarray, x_impact: float) -> np.ndarray:
        """Along-hull envelope ``c_x(x)`` (Gaussian of the given decay length)."""
        return self.max_patch_half_length * np.exp(
            -((np.asarray(x, dtype=float) - float(x_impact)) / self.decay_length) ** 2
        )

    def field(
        self,
        x: np.ndarray,
        xi: np.ndarray,
        t: float,
        v_rel: float,
        c_xi: float,
        x_impact: float,
    ) -> np.ndarray:
        """Pressure [Pa] on the grid ``(x, xi)`` at time ``t``.

        Returns an array of shape ``(x.size, xi.size)``.
        """
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        xi = np.asarray(xi, dtype=float).reshape(1, -1)
        p_peak = float(self.peak_pressure(v_rel))
        if p_peak <= 0.0 or c_xi <= 0.0:
            return np.zeros((x.size, xi.size))
        sx = np.sqrt(np.clip(1.0 - ((x - float(x_impact)) / np.maximum(self.patch_half_length(x, x_impact), 1e-9)) ** 2, 0.0, None))
        sxi = np.sqrt(np.clip(1.0 - (xi / c_xi) ** 2, 0.0, None))
        return p_peak * sx * sxi

    def line_load(self, x: np.ndarray, t: float, v_rel: float, c_xi: float, x_impact: float,
                  n_xi: int = 401) -> np.ndarray:
        """Vertical load per unit hull length ``f(x, t) = int p d xi`` [N/m]."""
        xi = np.linspace(-self.max_half_width, self.max_half_width, int(n_xi))
        p = self.field(x, xi, t, v_rel, c_xi, x_impact)
        return np.trapezoid(p, xi, axis=1)


def contact_half_width_history(t: np.ndarray, v_rel: np.ndarray, model: WagnerSlamming) -> np.ndarray:
    """Convenience wrapper: :meth:`WagnerSlamming.contact_half_width`."""
    return model.contact_half_width(t, v_rel)