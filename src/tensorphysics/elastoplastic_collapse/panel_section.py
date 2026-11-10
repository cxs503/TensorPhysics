"""Thin-walled (Vlasov) open-section properties of a ship stiffener with plating.

Why this module exists
----------------------
The tripping (lateral-torsional) check of P70-N01 used hand-typed section
constants (``I_z = t_w h^3/12 + ...``).  Those numbers were not verifiable and
contained a genuine error: the flange transfer term was taken at ``h_s/2``
instead of at the flange centroid ``h_s + t_f/2``, and the attached plate was
entered as ``t*b^3/12`` (a *bending* inertia) into what is a *polar* quantity.
A hand-typed constant turns a strength check into a paper exercise, so this
module supplies an auditable thin-walled calculator:

* ``StiffenerSection`` - area, centroid, ``I_x`` (strong), ``I_y`` (weak),
  ``J`` (Saint-Venant, thin-wall ``sum L t^3/3`` **and** the exact-rectangle
  series), ``C_w`` (Vlasov warping constant), ``I_p`` (polar second moment
  about the plate-stiffener junction) and ``c`` (distance from that junction to
  the outermost fibre).

Conventions
-----------
Mid-surface (thin-walled) model, right-handed axes, ``y`` measured upward from
the attached-plate mid-surface:

* attached plating: horizontal wall at ``y = 0``, tributary width ``b_p``
* stiffener web: vertical wall at ``x = 0``, from ``y = 0`` to ``y = h_s``
* stiffener flange (optional): horizontal wall at ``y = h_s + t_f/2``

The section is symmetric about ``x = 0``, so the shear centre lies on that axis
and is obtained from the sectorial integral.  For a web that passes through the
plate-flange intersection (flat bar and symmetric T are both of this family) the
mid-surface model gives ``C_w = 0`` exactly - the well known degeneracy of the
one-wall/two-wall idealisation.  TensorFEM's independent wall-graph calculator
reports the same degeneracy (``warping_constant`` ~ 1e-41 for a 200x10 tee), and
``compare_with_tensorfem`` cross-checks the two implementations.

References
----------
- Vlasov, V.Z. (1961). Thin-Walled Elastic Beams. NSF/Israel Program.
- Hughes, O.F. (1988). Ship Structural Design. SNAME, Ch. 8.
- Paik, J.K. & Thayamballi, A.K. (2003). Ultimate Limit State Design of
  Steel-Plated Structures. Wiley, Ch. 2 and 6.
- Timoshenko, S.P. & Gere, J.M. (1961). Theory of Elastic Stability, Ch. 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

__all__ = [
    "StiffenerSection",
    "stiffener_section",
    "rectangle_torsion_constant",
]


# --------------------------------------------------------------------------- #
# Saint-Venant torsion constant of a rectangle (exact series)
# --------------------------------------------------------------------------- #
def rectangle_torsion_constant(width: float, thickness: float) -> float:
    """Saint-Venant torsion constant of a solid rectangle ``width x thickness``.

    ``J = a b^3 (1/3 - 0.21 (b/a) (1 - b^4/(12 a^4)))`` for ``a >= b``.
    """
    a, b = max(abs(width), abs(thickness)), min(abs(width), abs(thickness))
    if b <= 0.0:
        return 0.0
    return a * b**3 * (1.0 / 3.0 - 0.21 * (b / a) * (1.0 - b**4 / (12.0 * a**4)))


def _walls(
    b_p: float, t: float, h_s: float, t_w: float, b_f: float, t_f: float,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Mid-surface wall list (a tree: plate | web | flange)."""
    y_f = h_s + 0.5 * t_f if (b_f > 0.0 and t_f > 0.0) else h_s
    walls: List[Tuple[np.ndarray, np.ndarray, float]] = [
        (np.array([-0.5 * b_p, 0.0]), np.array([0.0, 0.0]), t),
        (np.array([0.0, 0.0]), np.array([0.5 * b_p, 0.0]), t),
        (np.array([0.0, 0.0]), np.array([0.0, y_f]), t_w),
    ]
    if b_f > 0.0 and t_f > 0.0:
        walls.append((np.array([-0.5 * b_f, y_f]), np.array([0.0, y_f]), t_f))
        walls.append((np.array([0.0, y_f]), np.array([0.5 * b_f, y_f]), t_f))
    return walls


@dataclass(frozen=True)
class StiffenerSection:
    """Thin-walled properties of a stiffener + attached plate tributary strip.

    All quantities are per stiffener.  ``plate_width`` is the tributary plate
    width (normally the stiffener spacing ``b`` for the Smith unit, or the
    effective width ``b_e`` when a buckled plate is being reduced).
    """

    plate_width: float
    plate_thickness: float
    web_height: float
    web_thickness: float
    flange_width: float
    flange_thickness: float
    young: float
    poisson: float

    # ---- basic measures -------------------------------------------------- #
    @property
    def area(self) -> float:
        """Gross cross-sectional area [m^2]."""
        a = self.plate_width * self.plate_thickness + self.web_height * self.web_thickness
        return a + self.flange_width * self.flange_thickness

    @property
    def stiffener_area(self) -> float:
        return self.web_height * self.web_thickness + self.flange_width * self.flange_thickness

    @property
    def plate_area(self) -> float:
        return self.plate_width * self.plate_thickness

    @property
    def centroid_y(self) -> float:
        """Centroid height above the plate mid-surface [m]."""
        a_p = self.plate_area
        a_w = self.web_height * self.web_thickness
        a_f = self.flange_width * self.flange_thickness
        s = a_p * 0.0 + a_w * 0.5 * self.web_height + a_f * (self.web_height + 0.5 * self.flange_thickness)
        return s / self.area

    @property
    def i_x(self) -> float:
        """Strong-axis (bending about the horizontal centroid axis) inertia [m^4]."""
        y_bar = self.centroid_y
        i = self.plate_width * self.plate_thickness**3 / 12.0
        i += self.plate_area * (0.0 - y_bar) ** 2
        i += self.web_thickness * self.web_height**3 / 12.0
        i += self.web_height * self.web_thickness * (0.5 * self.web_height - y_bar) ** 2
        if self.flange_width > 0.0:
            y_f = self.web_height + 0.5 * self.flange_thickness
            i += self.flange_width * self.flange_thickness**3 / 12.0
            i += self.flange_width * self.flange_thickness * (y_f - y_bar) ** 2
        return i

    @property
    def i_y(self) -> float:
        """Weak-axis (lateral) inertia [m^4]."""
        i = self.plate_thickness * self.plate_width**3 / 12.0
        i += self.web_height * self.web_thickness**3 / 12.0
        i += self.flange_thickness * self.flange_width**3 / 12.0
        return i

    @property
    def j_thin(self) -> float:
        """Thin-wall Saint-Venant constant ``sum L t^3/3`` [m^4]."""
        j = self.plate_width * self.plate_thickness**3 / 3.0
        j += self.web_height * self.web_thickness**3 / 3.0
        j += self.flange_width * self.flange_thickness**3 / 3.0
        return j

    @property
    def j_rect(self) -> float:
        """Saint-Venant constant from exact rectangle series [m^4]."""
        j = rectangle_torsion_constant(self.plate_width, self.plate_thickness)
        j += rectangle_torsion_constant(self.web_height, self.web_thickness)
        j += rectangle_torsion_constant(self.flange_width, self.flange_thickness)
        return j

    @property
    def i_p_junction(self) -> float:
        """Polar second moment about the plate-stiffener junction [m^4].

        ``I_p = I_x(junction) + I_y`` with ``I_x(junction)`` by the parallel-axis
        theorem applied to the centroidal value.
        """
        return self.i_x + self.area * self.centroid_y**2 + self.i_y

    @property
    def c_max(self) -> float:
        """Distance from the junction to the outermost section fibre [m]."""
        y_top = self.web_height + self.flange_thickness
        x_top = 0.5 * self.flange_width
        return float(np.hypot(x_top, y_top))

    @property
    def c_flange(self) -> float:
        """Distance from the centroid to the flange mid-surface [m]."""
        return self.web_height + 0.5 * self.flange_thickness - self.centroid_y

    @property
    def polar_radius(self) -> float:
        """Radius of gyration about the junction, ``sqrt(I_p/A)`` [m]."""
        return float(np.sqrt(self.i_p_junction / self.area))

    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    # ---- warping (Vlasov) ------------------------------------------------ #
    @property
    def shear_centre_y(self) -> float:
        """Shear-centre height above the plate mid-surface [m].

        Closed-form thin-wall results for the supported family (a vertical web
        with an optional symmetric flange):

        * flat bar - the single wall's own centroid, ``h_s/2``;
        * symmetric tee - the wall intersection, ``h_s + t_f/2`` (a horizontal
          force produces shear flow only in the flange, whose resultant acts at
          the flange mid-line, so the shear centre sits there).

        Documented scope: the module refuses nothing but is *verified* only for
        this family (see ``tests/test_panel_ultimate_maturity.py``).
        """
        if self.flange_width > 0.0 and self.flange_thickness > 0.0:
            return self.web_height + 0.5 * self.flange_thickness
        return 0.5 * self.web_height

    def warping_constant(self) -> float:
        """Warping constant ``C_w`` of the mid-surface wall tree [m^6].

        ``C_w = int w^2 dA`` with the sectorial coordinate ``w`` taken about the
        shear centre and normalised so that ``int w dA = 0``.
        """
        walls = _walls(self.plate_width, self.plate_thickness,
                       self.web_height, self.web_thickness,
                       self.flange_width, self.flange_thickness)
        lengths = np.array([np.linalg.norm(b - a) for a, b, _ in walls])
        thick = np.array([t for _, _, t in walls])
        weights = thick * lengths
        area = float(weights.sum())
        pole = (0.0, float(self.shear_centre_y))
        per_wall = self._node_sectorial(walls, pole)
        mean = 0.0
        for k, (_, _, t) in enumerate(walls):
            o0, o1 = per_wall[k]
            mean += t * lengths[k] * 0.5 * (o0 + o1)
        mean /= area
        c_w = 0.0
        for k, (_, _, t) in enumerate(walls):
            o0, o1 = per_wall[k]
            a0, a1 = o0 - mean, o1 - mean
            c_w += t * lengths[k] * (a0**2 + a0 * a1 + a1**2) / 3.0
        return float(max(c_w, 0.0))

    @staticmethod
    def _sectorial(walls, pole: Tuple[float, float]) -> np.ndarray:
        """Sectorial coordinate at wall mid-points."""
        nodes_omega = StiffenerSection._node_sectorial(walls, pole)
        return np.array([0.5 * (o0 + o1) for o0, o1 in nodes_omega])

    @staticmethod
    def _node_sectorial(walls, pole: Tuple[float, float]):
        """Per-wall node sectorial coordinates (BFS over the wall tree)."""
        xp, yp = pole
        canon = lambda p: (round(float(p[0]), 15), round(float(p[1]), 15))
        adj = {}
        for k, (a, b, _) in enumerate(walls):
            adj.setdefault(canon(a), []).append((k, canon(b), a, b))
            adj.setdefault(canon(b), []).append((k, canon(a), b, a))
        root = canon(walls[0][0])
        omega = {root: 0.0}
        per_wall = [None] * len(walls)
        queue = [root]
        while queue:
            node = queue.pop(0)
            for k, other, a, b in adj[node]:
                if other in omega:
                    continue
                # r_t = (x - xp) dy/ds - (y - yp) dx/ds along the wall, s = arc length
                length = float(np.linalg.norm(b - a))
                dx, dy = (b - a) / length
                if canon(a) == node:
                    r_t = (a[0] - xp) * dy - (a[1] - yp) * dx
                    omega[other] = omega[node] + r_t * length
                else:
                    r_t = (b[0] - xp) * dy - (b[1] - yp) * dx
                    omega[other] = omega[node] + r_t * length
                queue.append(other)
        for k, (a, b, _) in enumerate(walls):
            per_wall[k] = (omega[canon(a)], omega[canon(b)])
        return per_wall


def stiffener_section(
    *,
    plate_width: float,
    plate_thickness: float,
    web_height: float,
    web_thickness: float,
    flange_width: float = 0.0,
    flange_thickness: float = 0.0,
    young: float = 2.06e11,
    poisson: float = 0.3,
) -> StiffenerSection:
    """Convenience constructor with validation."""
    if plate_width <= 0 or plate_thickness <= 0:
        raise ValueError("plate width/thickness must be positive")
    if web_height <= 0 or web_thickness <= 0:
        raise ValueError("web height/thickness must be positive")
    if flange_width < 0 or flange_thickness < 0:
        raise ValueError("flange dimensions must be non-negative")
    return StiffenerSection(
        plate_width=float(plate_width),
        plate_thickness=float(plate_thickness),
        web_height=float(web_height),
        web_thickness=float(web_thickness),
        flange_width=float(flange_width),
        flange_thickness=float(flange_thickness),
        young=float(young),
        poisson=float(poisson),
    )