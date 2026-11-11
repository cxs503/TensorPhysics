"""Stiffened-panel cross-section: exact elastic **and plastic** properties.

The pre-existing :meth:`StiffenedPlateGeometry.section_properties` returns only
the elastic quantities of the *stiffener with attached plate* (``A``, ``z_bar``,
``I``, ``Z_top``, ``Z_bot``).  Two capabilities used by panel-collapse work were
missing:

* the **plastic** section modulus ``Z_p`` (equal-area axis ``y_p``), which sets
  the rigid-plastic bending capacity ``M_p = sigma_y Z_p`` of the panel and
  therefore the axial-bending interaction of the collapse surface;
* the **torsional** quantities (``J``, ``C_w``, ``I_p``) needed by the tripping
  mode, so that tripping can compete with plate and column buckling inside one
  panel-ultimate calculation instead of being a separate side calculation.

Geometry convention
-------------------
``z`` is measured from the plate mid-thickness, positive towards the stiffener:
plate ``[-t/2, +t/2]`` full width ``b``; web ``[t/2, t/2+h_s]`` thickness
``t_w``; flange ``[t/2+h_s, t/2+h_s+t_f]`` width ``b_f``.  A flat bar is
``b_f = t_f = 0``.

Everything is *closed form*: the section is a union of axis-aligned rectangles,
so area, first moment, second moment and plastic modulus follow exactly by
rectangle splitting.  :meth:`StiffenedPanelSection.numeric_properties`
recomputes the same quantities by a completely independent strip quadrature and
is used in the tests to prove the closed forms.

References
----------
- Paik, J.K. & Thayamballi, A.K. (2003). *Ultimate Limit State Design of
  Steel-Plated Structures*. Wiley, Ch. 3 (plastic section modulus) and Ch. 5
  (tripping of flat bars and T-sections).
- Timoshenko, S.P. & Gere, J.M. (1961). *Theory of Elastic Stability*, 2nd ed.,
  Ch. 6 (torsional / lateral-torsional buckling of open sections).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .von_mises_plasticity import VonMisesMaterial

__all__ = [
    "StiffenedPanelSection",
    "SectionGeometryError",
]


class SectionGeometryError(ValueError):
    """Raised for non-physical panel/section dimensions."""


@dataclass(frozen=True)
class _Element:
    """Axis-aligned rectangular wall element of the section."""

    z_lo: float
    z_hi: float
    width: float
    name: str

    @property
    def area(self) -> float:
        return self.width * (self.z_hi - self.z_lo)

    def part(self, lo: float, hi: float) -> Tuple[float, float]:
        """Area and first moment ``int z dA`` of the clipped part [lo, hi]."""
        a = max(self.z_lo, lo)
        b = min(self.z_hi, hi)
        if b <= a:
            return 0.0, 0.0
        w = self.width
        return w * (b - a), 0.5 * w * (b * b - a * a)

    def second_moment_about(self, z0: float) -> float:
        """``int (z - z0)^2 dA`` over the whole element (closed form)."""
        w = self.width
        a = self.z_lo - z0
        b = self.z_hi - z0
        return w * (b ** 3 - a ** 3) / 3.0


@dataclass(frozen=True)
class StiffenedPanelSection:
    """Single stiffener with its attached plate (one stiffener spacing wide).

    Parameters
    ----------
    a, b, t : float
        Frame spacing (column length), stiffener spacing, plate thickness [m].
    h_s, t_w : float
        Stiffener web height and thickness [m].
    b_f, t_f : float
        Flange width and thickness [m]; ``b_f = t_f = 0`` gives a flat bar.
    material : VonMisesMaterial
        Material (``E``, ``nu``, ``sigma_y0``, ``H``).
    """

    a: float
    b: float
    t: float
    h_s: float
    t_w: float
    material: VonMisesMaterial = field(default=None)  # type: ignore[assignment]
    b_f: float = 0.0
    t_f: float = 0.0

    # -- validation ---------------------------------------------------------
    def __post_init__(self) -> None:
        for name in ("a", "b", "t", "h_s", "t_w"):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val <= 0.0:
                raise SectionGeometryError(f"{name} must be positive, got {val!r}")
        for name in ("b_f", "t_f"):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val < 0.0:
                raise SectionGeometryError(f"{name} must be non-negative, got {val!r}")
        if (self.b_f > 0.0) != (self.t_f > 0.0):
            raise SectionGeometryError("b_f and t_f must be given together (or both zero)")
        if self.material is None:
            raise SectionGeometryError("material is required")
        if self.b_f >= 0.5 * self.b:
            raise SectionGeometryError(
                "flange wider than half the stiffener spacing is outside the model domain"
            )

    # -- element decomposition ---------------------------------------------
    @property
    def elements(self) -> Tuple[_Element, ...]:
        """Plate + web (+ flange) as rectangles, plate mid-plane at z = 0."""
        elements = [
            _Element(-0.5 * self.t, 0.5 * self.t, self.b, "plate"),
            _Element(0.5 * self.t, 0.5 * self.t + self.h_s, self.t_w, "web"),
        ]
        if self.b_f > 0.0:
            elements.append(
                _Element(
                    0.5 * self.t + self.h_s,
                    0.5 * self.t + self.h_s + self.t_f,
                    self.b_f,
                    "flange",
                )
            )
        return tuple(elements)

    # -- areas --------------------------------------------------------------
    @property
    def plate_area(self) -> float:
        return self.b * self.t

    @property
    def web_area(self) -> float:
        return self.h_s * self.t_w

    @property
    def flange_area(self) -> float:
        return self.b_f * self.t_f

    @property
    def stiffener_area(self) -> float:
        return self.web_area + self.flange_area

    @property
    def total_area(self) -> float:
        return self.plate_area + self.stiffener_area

    # -- elastic properties -------------------------------------------------
    @property
    def z_bottom(self) -> float:
        return -0.5 * self.t

    @property
    def z_top(self) -> float:
        return 0.5 * self.t + self.h_s + self.t_f

    def first_moment(self, z0: float = 0.0) -> float:
        """``int (z - z0) dA``."""
        total = 0.0
        for el in self.elements:
            area, moment = el.part(el.z_lo, el.z_hi)
            total += moment - z0 * area
        return total

    def second_moment_about(self, z0: float) -> float:
        """``int (z - z0)^2 dA`` about the axis ``z = z0``."""
        return sum(el.second_moment_about(z0) for el in self.elements)

    @property
    def z_bar(self) -> float:
        """Elastic centroid measured from the plate mid-thickness [m]."""
        return self.first_moment(0.0) / self.total_area

    @property
    def second_moment(self) -> float:
        """Second moment of area about the elastic centroid [m^4]."""
        return self.second_moment_about(self.z_bar)

    @property
    def gyration_radius(self) -> float:
        return math.sqrt(self.second_moment / self.total_area)

    @property
    def c_top(self) -> float:
        return self.z_top - self.z_bar

    @property
    def c_bottom(self) -> float:
        return self.z_bar - self.z_bottom

    @property
    def section_modulus_top(self) -> float:
        return self.second_moment / self.c_top

    @property
    def section_modulus_bottom(self) -> float:
        return self.second_moment / self.c_bottom

    # -- plastic properties -------------------------------------------------
    def area_below(self, y: float) -> float:
        return sum(el.part(el.z_lo, y)[0] for el in self.elements)

    @property
    def plastic_neutral_axis(self) -> float:
        """Equal-area (plastic) neutral axis ``y_p`` [m], bisection to 1e-16."""
        target = 0.5 * self.total_area
        lo, hi = self.z_bottom, self.z_top
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.area_below(mid) < target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-16:
                break
        return 0.5 * (lo + hi)

    @property
    def plastic_section_modulus(self) -> float:
        """``Z_p = int |z - y_p| dA`` (exact for the rectangular decomposition)."""
        y_p = self.plastic_neutral_axis
        total = 0.0
        for el in self.elements:
            a_b, s_b = el.part(el.z_lo, y_p)
            a_a, s_a = el.part(y_p, el.z_hi)
            total += (y_p * a_b - s_b) + (s_a - y_p * a_a)
        return total

    @property
    def shape_factor(self) -> float:
        """``Z_p / Z_el,min`` -- the reserve bought by the plastic redistribution."""
        return self.plastic_section_modulus / min(
            self.section_modulus_top, self.section_modulus_bottom
        )

    @property
    def yield_bending_moment(self) -> float:
        """First-yield moment ``M_y = sigma_y0 * min(Z_top, Z_bot)`` [N m]."""
        return self.material.sigma_y0 * min(
            self.section_modulus_top, self.section_modulus_bottom
        )

    @property
    def plastic_bending_moment(self) -> float:
        """Fully plastic moment ``M_p = sigma_y0 * Z_p`` [N m]."""
        return self.material.sigma_y0 * self.plastic_section_modulus

    @property
    def squash_load(self) -> float:
        """Axial squash load ``N_p = sigma_y0 * A`` [N]."""
        return self.material.sigma_y0 * self.total_area

    # -- torsional / tripping quantities ------------------------------------
    @property
    def torsion_constant(self) -> float:
        """Saint-Venant constant of the *stiffener* (web + flange) [m^4]."""
        j = self.h_s * self.t_w ** 3 / 3.0
        if self.b_f > 0.0:
            j += self.b_f * self.t_f ** 3 / 3.0
        return j

    @property
    def torsion_constant_with_plate(self) -> float:
        """Stiffener constant plus half a plate strip (rotational restraint) [m^4]."""
        return self.torsion_constant + 0.5 * self.b * self.t ** 3 / 3.0

    @property
    def warping_constant(self) -> float:
        """Warping constant of a T-section, ``I_flange * d_f^2`` [m^6].

        Mono-symmetric T approximation (Timoshenko & Gere, Ch. 6); zero for a
        flat bar, which is why flat bars trip much earlier than T-sections.
        """
        if self.b_f <= 0.0:
            return 0.0
        i_f = self.t_f * self.b_f ** 3 / 12.0
        return i_f * (self.h_s + 0.5 * self.t_f) ** 2

    @property
    def polar_second_moment(self) -> float:
        """Polar second moment of the stiffener about the plate-level web axis."""
        c_w = 0.5 * self.t + 0.5 * self.h_s
        i_p = self.h_s * self.t_w ** 3 / 12.0 + self.web_area * c_w ** 2
        if self.b_f > 0.0:
            c_f = 0.5 * self.t + self.h_s + 0.5 * self.t_f
            i_p += self.t_f * self.b_f ** 3 / 12.0 + self.flange_area * c_f ** 2
        return i_p

    @property
    def shear_modulus(self) -> float:
        return self.material.E / (2.0 * (1.0 + self.material.nu))

    # -- slenderness and critical stresses ----------------------------------
    @property
    def beta_plate(self) -> float:
        """Plate slenderness ``beta = (b/t) sqrt(sigma_y0/E)``."""
        return (self.b / self.t) * math.sqrt(self.material.sigma_y0 / self.material.E)

    @property
    def lambda_column(self) -> float:
        """Paik column slenderness ``(a/(pi r)) sqrt(sigma_y0/E)``, r with plate."""
        return (self.a / (math.pi * self.gyration_radius)) * math.sqrt(
            self.material.sigma_y0 / self.material.E
        )

    @property
    def lambda_web(self) -> float:
        """Web plate slenderness ``(h_s/t_w) sqrt(sigma_y0/E)``."""
        return (self.h_s / self.t_w) * math.sqrt(
            self.material.sigma_y0 / self.material.E
        )

    @property
    def lambda_flange(self) -> float:
        """Flange outstand slenderness; ``inf`` for a flat bar."""
        if self.b_f <= 0.0:
            return float("inf")
        return (self.b_f / self.t_f) * math.sqrt(
            self.material.sigma_y0 / self.material.E
        )

    def _plate_buckling_k(self) -> float:
        a_over_b = self.a / self.b
        if a_over_b >= 1.0:
            return 4.0
        return (a_over_b + 1.0 / a_over_b) ** 2

    @property
    def sigma_cr_plate(self) -> float:
        """Elastic plate buckling stress (long panel, simply supported) [Pa]."""
        d_plate = self.material.E * self.t ** 3 / (12.0 * (1.0 - self.material.nu ** 2))
        return self._plate_buckling_k() * math.pi ** 2 * d_plate / (self.b ** 2 * self.t)

    @property
    def sigma_cr_web(self) -> float:
        """Web local (plate) buckling stress, both edges supported [Pa]."""
        d_web = (
            self.material.E * self.t_w ** 3 / (12.0 * (1.0 - self.material.nu ** 2))
        )
        return 4.0 * math.pi ** 2 * d_web / (self.h_s ** 2 * self.t_w)

    @property
    def sigma_cr_flange(self) -> float:
        """Flange outstand buckling stress (one edge free, k = 0.425) [Pa]."""
        if self.b_f <= 0.0:
            return float("inf")
        d_f = self.material.E * self.t_f ** 3 / (12.0 * (1.0 - self.material.nu ** 2))
        return 0.425 * math.pi ** 2 * d_f / (self.b_f ** 2 * self.t_f)

    @property
    def sigma_euler_column(self) -> float:
        """Euler column stress of stiffener + attached plate over the span ``a``."""
        return math.pi ** 2 * self.material.E * self.gyration_radius ** 2 / self.a ** 2

    @property
    def sigma_tripping(self) -> float:
        """Elastic torsional (tripping) buckling stress of the stiffener [Pa].

        ``sigma_T = (G J + pi^2 E C_w / a^2) / I_p`` (Timoshenko & Gere).  The
        attached plate is not part of the rotating body; it enters only through
        the global column check.
        """
        return (
            self.shear_modulus * self.torsion_constant
            + math.pi ** 2 * self.material.E * self.warping_constant / self.a ** 2
        ) / self.polar_second_moment

    # -- independent numerical check ---------------------------------------
    def numeric_properties(self, n_strips: int = 400) -> Dict[str, float]:
        """Independent strip quadrature of ``A, z_bar, I, y_p, Z_p``.

        Each element is cut into ``n_strips`` horizontal strips (no closed form
        is reused), so agreement with the analytic values tests the algebra.
        """
        if n_strips < 10:
            raise ValueError("need at least 10 strips per element")
        strips: List[Tuple[float, float, float]] = []  # (z_centre, dz, width)
        for el in self.elements:
            dz = (el.z_hi - el.z_lo) / n_strips
            for i in range(n_strips):
                strips.append((el.z_lo + (i + 0.5) * dz, dz, el.width))
        area = sum(dz * w for _, dz, w in strips)
        z_bar = sum(z * dz * w for z, dz, w in strips) / area
        inertia = sum(
            w * dz ** 3 / 12.0 + w * dz * (z - z_bar) ** 2 for z, dz, w in strips
        )
        lo = min(z - 0.5 * dz for z, dz, _ in strips)
        hi = max(z + 0.5 * dz for z, dz, _ in strips)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            below = sum(dz * w for z, dz, w in strips if z <= mid)
            if below < 0.5 * area:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-16:
                break
        y_p = 0.5 * (lo + hi)
        z_p = sum(dz * w * abs(z - y_p) for z, dz, w in strips)
        return {
            "area": area,
            "z_bar": z_bar,
            "second_moment": inertia,
            "plastic_neutral_axis": y_p,
            "plastic_section_modulus": z_p,
        }

    # -- interoperability ---------------------------------------------------
    def to_scantlings(self):
        """Return the ``axial_collapse_ep`` scantlings dataclass for this panel."""
        from .axial_collapse_ep import StiffenedPanelScantlings

        return StiffenedPanelScantlings(
            a=self.a,
            b=self.b,
            t=self.t,
            h_s=self.h_s,
            t_w=self.t_w,
            b_f=self.b_f if self.b_f > 0.0 else 1e-9,
            t_f=self.t_f if self.t_f > 0.0 else 1e-9,
        )

    def to_isotropic_hardening(self):
        """Return the ``axial_collapse_ep`` material dataclass."""
        from .axial_collapse_ep import IsotropicHardeningMaterial

        return IsotropicHardeningMaterial(
            E=self.material.E,
            nu=self.material.nu,
            sigma_y0=self.material.sigma_y0,
            H=self.material.H,
        )

    def summary(self) -> Dict[str, float]:
        """Flat dict of every derived quantity (report/JSON friendly)."""
        return {
            "a": self.a,
            "b": self.b,
            "t": self.t,
            "h_s": self.h_s,
            "t_w": self.t_w,
            "b_f": self.b_f,
            "t_f": self.t_f,
            "plate_area": self.plate_area,
            "stiffener_area": self.stiffener_area,
            "total_area": self.total_area,
            "z_bar": self.z_bar,
            "second_moment": self.second_moment,
            "gyration_radius": self.gyration_radius,
            "section_modulus_top": self.section_modulus_top,
            "section_modulus_bottom": self.section_modulus_bottom,
            "plastic_neutral_axis": self.plastic_neutral_axis,
            "plastic_section_modulus": self.plastic_section_modulus,
            "shape_factor": self.shape_factor,
            "yield_bending_moment": self.yield_bending_moment,
            "plastic_bending_moment": self.plastic_bending_moment,
            "squash_load": self.squash_load,
            "torsion_constant": self.torsion_constant,
            "warping_constant": self.warping_constant,
            "polar_second_moment": self.polar_second_moment,
            "beta_plate": self.beta_plate,
            "lambda_column": self.lambda_column,
            "lambda_web": self.lambda_web,
            "lambda_flange": self.lambda_flange,
            "sigma_cr_plate": self.sigma_cr_plate,
            "sigma_cr_web": self.sigma_cr_web,
            "sigma_cr_flange": self.sigma_cr_flange,
            "sigma_euler_column": self.sigma_euler_column,
            "sigma_tripping": self.sigma_tripping,
            "sigma_y0": self.material.sigma_y0,
        }