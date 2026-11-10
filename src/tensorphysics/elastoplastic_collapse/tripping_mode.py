"""Stiffener tripping: torsional-buckling mode with an explicit mode switch.

Physics
-------
Tripping is the lateral-torsional failure of a stiffener that rotates about its
attachment line to the plating.  The elastic torsional-buckling stress of the
stiffener (Vlasov / Timoshenko, column torsional buckling) is

    sigma_ET = [ G J_s + pi^2 E C_w,s / L_t^2 ] / I_ps

where
    ``J_s``    Saint-Venant torsion constant of the *stiffener alone*
               (``sum L t^3/3`` for web + flange),
    ``C_w,s``  warping constant of the stiffener alone,
    ``I_ps``   polar second moment of the stiffener alone about its shear
               centre, ``I_x + I_y + A_s (y_sc - y_bar)^2``,
    ``L_t``    unrestrained tripping length (default: transverse frame spacing).
The elastoplastic reduction uses the DNV-RP-C201 column curve with
``lambda_T = sqrt(sigma_y / sigma_ET)`` (``mu0 = 0.34``, curve b), so the
mode switches continuously from yielding (``lambda_T <~ 0.2``) through
elastoplastic tripping to elastic tripping (``lambda_T >~ 1.5``).

Why the polar moment is about the shear centre and *not* about the plate line
---------------------------------------------------------------------------
The baseline module used ``I_z = t_w h^3/12 + t_f b_f^3/12 + b_f t_f (h_s/2)^2``
plus a plate term ``t b^3/12`` and a 1/I-interaction of torsion and weak-axis
bending.  Those are bending inertias, not polar ones, and the flange transfer
was taken at ``h_s/2`` instead of the flange centroid ``h_s + t_f/2``; the
result is an unanchored constant that cannot be checked.  For an isolated
stiffener the torsional-buckling polar radius is taken about the shear centre
(Timoshenko & Gere Ch. 6, Vlasov); using the polar moment about the plate line
*including* the attached plate would raise ``I_p`` by an order of magnitude and
drive every panel into a spurious tripping mode, which is why the attached
plating enters here only through the restraint factor ``lambda_trip``.

Benchmark
---------
``ltb_critical_moment`` reproduces the classical uniform-moment lateral
torsional buckling reference used by ``tensorfem.marine_stiffener_failure``

    Mcr = (pi/L) sqrt(E I_y G J) sqrt(1 + pi^2 E C_w / (L^2 G J))

and ``ltb_stress`` converts it to the axial stress equivalent
``Mcr / (I_ps / c)`` requested by the mission.  ``ltb_numerical`` solves the same
bifurcation with a centred-difference Dirichlet Laplacian so that the closed
form is verified by an independent discretisation (error must vanish as
``segments`` grows).

References
----------
- Timoshenko, S.P. & Gere, J.M. (1961). Theory of Elastic Stability, Ch. 6.
- Vlasov, V.Z. (1961). Thin-Walled Elastic Beams.
- Hughes, O.F. (1988). Ship Structural Design. SNAME, Ch. 8.
- DNV-RP-C201 (2017). Buckling of Ships, Sec. 8-9.
- Paik, J.K. & Thayamballi, A.K. (2003). Ultimate Limit State Design of
  Steel-Plated Structures, Ch. 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .von_mises_plasticity import VonMisesMaterial
from .panel_section import StiffenerSection, stiffener_section, rectangle_torsion_constant

__all__ = [
    "TrippingAnalysis",
    "TrippingResult",
    "tripping_capacity",
    "ltb_critical_moment",
    "ltb_numerical",
]


#: DNV-RP-C201 imperfection factor (curve b) used for the tripping column curve
TRIPPING_MU0 = 0.34


def _column_curve(lam: float, mu0: float = TRIPPING_MU0) -> float:
    """DNV-RP-C201 column curve ``sigma/sigma_y`` (``1/lam^2`` asymptote)."""
    if lam <= 0.2:
        return 1.0
    mu = mu0 * (lam - 0.2)
    a = (1.0 + mu + lam**2) / (2.0 * lam**2)
    return float(max(a - math.sqrt(max(a * a - 1.0 / lam**2, 0.0)), 0.0))


@dataclass(frozen=True)
class TrippingResult:
    """Result of a tripping assessment.

    Attributes
    ----------
    sigma_trip : float
        Elastoplastic tripping stress [Pa] (on the stiffener section).
    lambda_trip : float
        Restraint factor used.
    failure_mode : str
        ``'tripping'``, ``'yielding'``, ``'elastic_tripping'`` or ``'none'``.
    capacity_reduction : float
        ``sigma_trip / sigma_y``.
    """

    sigma_trip: float
    lambda_trip: float
    failure_mode: str
    capacity_reduction: float


@dataclass
class TrippingAnalysis:
    """Stiffener tripping analysis.

    Parameters
    ----------
    a : float
        Stiffener span between transverse frames [m] (= tripping length).
    b : float
        Stiffener spacing [m] (plating tributary width; documentation only).
    t : float
        Plate thickness [m].
    h_s, t_w : float
        Web height / thickness [m].
    b_f, t_f : float
        Flange width / thickness [m] (0 for a flat bar).
    material : VonMisesMaterial
    plate_width : float, optional
        Attached-plate width used when reporting the *panel* sections
        (defaults to ``b``).  Not used in the tripping stress itself.
    """

    a: float
    b: float
    t: float
    h_s: float
    t_w: float
    b_f: float
    t_f: float
    material: VonMisesMaterial
    plate_width: Optional[float] = None

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0 or self.t <= 0:
            raise ValueError("panel dimensions must be positive")
        if self.h_s <= 0 or self.t_w <= 0:
            raise ValueError("stiffener web dimensions must be positive")
        if self.b_f < 0 or self.t_f < 0:
            raise ValueError("flange dimensions must be non-negative")
        if (self.b_f == 0) != (self.t_f == 0):
            raise ValueError("flange width and thickness must be given together")

    # ---- stiffener-alone section ---------------------------------------- #
    @property
    def stiffener_area(self) -> float:
        return self.h_s * self.t_w + self.b_f * self.t_f

    @property
    def stiffener_centroid_y(self) -> float:
        a_w = self.h_s * self.t_w
        a_f = self.b_f * self.t_f
        return (a_w * 0.5 * self.h_s + a_f * (self.h_s + 0.5 * self.t_f)) / self.stiffener_area

    @property
    def stiffener_i_x(self) -> float:
        y_bar = self.stiffener_centroid_y
        i = self.t_w * self.h_s**3 / 12.0 + self.h_s * self.t_w * (0.5 * self.h_s - y_bar) ** 2
        if self.b_f > 0:
            y_f = self.h_s + 0.5 * self.t_f
            i += (self.b_f * self.t_f**3 / 12.0
                  + self.b_f * self.t_f * (y_f - y_bar) ** 2)
        return i

    @property
    def stiffener_i_y(self) -> float:
        return (self.h_s * self.t_w**3 / 12.0 + self.t_f * self.b_f**3 / 12.0)

    @property
    def shear_centre_y(self) -> float:
        """Shear-centre height above the plate [m] (thin-wall, symmetric section)."""
        if self.b_f > 0:
            return self.h_s + 0.5 * self.t_f   # wall intersection (flange mid-line)
        return 0.5 * self.h_s                 # single wall: its own centroid

    @property
    def j_saint_venant(self) -> float:
        """Saint-Venant constant of the stiffener alone, ``sum L t^3/3`` [m^4]."""
        return self.h_s * self.t_w**3 / 3.0 + self.b_f * self.t_f**3 / 3.0

    @property
    def j_saint_venant_rect(self) -> float:
        """Saint-Venant constant of the stiffener alone (exact rectangle series)."""
        return (rectangle_torsion_constant(self.h_s, self.t_w)
                + rectangle_torsion_constant(self.b_f, self.t_f))

    @property
    def warping_constant(self) -> float:
        """Warping constant of the stiffener alone [m^6].

        The flat-bar and symmetric-tee families have every wall passing through
        the wall intersection where the shear centre sits, so the mid-surface
        (Vlasov) model returns ``C_w = 0`` exactly.  The general sectorial
        machinery of :mod:`panel_section` is reused with a vanishing plate strip
        so that the value is *computed*, not asserted.
        """
        sec = stiffener_section(
            plate_width=1e-9, plate_thickness=min(self.t, 1e-3),
            web_height=self.h_s, web_thickness=self.t_w,
            flange_width=self.b_f, flange_thickness=self.t_f,
            young=self.material.E, poisson=self.material.nu,
        )
        return sec.warping_constant()

    @property
    def i_p_shear_centre(self) -> float:
        """Polar second moment of the stiffener alone about its shear centre [m^4]."""
        e = self.shear_centre_y - self.stiffener_centroid_y
        return self.stiffener_i_x + self.stiffener_i_y + self.stiffener_area * e**2

    @property
    def i_p_attachment(self) -> float:
        """Polar second moment about the plate attachment line [m^4] (reported)."""
        return (self.stiffener_i_x + self.stiffener_area * self.stiffener_centroid_y**2
                + self.stiffener_i_y)

    @property
    def c_from_sc(self) -> float:
        """Distance from the shear centre to the outermost fibre [m]."""
        y_top = self.h_s + self.t_f
        x_top = 0.5 * self.b_f
        dy = abs(y_top - self.shear_centre_y)
        return float(np.hypot(x_top, dy))

    def panel_section(self, plate_width: Optional[float] = None) -> StiffenerSection:
        """Stiffener + attached-plate section (reported properties)."""
        bw = self.b if plate_width is None else float(plate_width)
        return stiffener_section(
            plate_width=bw, plate_thickness=self.t,
            web_height=self.h_s, web_thickness=self.t_w,
            flange_width=self.b_f, flange_thickness=self.t_f,
            young=self.material.E, poisson=self.material.nu,
        )

    def section_properties(self) -> Dict[str, float]:
        """Dictionary of the section quantities (backwards compatible keys).

        ``I_z``/``I_w``/``J``/``z_s`` are kept for the P70-N01 API; they now hold
        the weak-axis inertia, the warping constant, the Saint-Venant constant
        and the shear-centre height of the stiffener alone.
        """
        return {
            "I_z": float(self.stiffener_i_y),
            "I_x": float(self.stiffener_i_x),
            "I_w": float(self.warping_constant),
            "J": float(self.j_saint_venant),
            "J_rect": float(self.j_saint_venant_rect),
            "A_s": float(self.stiffener_area),
            "y_bar": float(self.stiffener_centroid_y),
            "z_s": float(self.shear_centre_y),
            "I_p_sc": float(self.i_p_shear_centre),
            "I_p_attachment": float(self.i_p_attachment),
            "c": float(self.c_from_sc),
        }

    # ---- elastic tripping ------------------------------------------------- #
    def elastic_tripping_stress(
        self, lambda_trip: float = 1.0, length: Optional[float] = None,
    ) -> float:
        """Elastic tripping (torsional buckling) stress [Pa].

        ``sigma_ET = [G J_s + pi^2 E C_w,s / L_t^2] / I_ps / lambda_trip``

        ``lambda_trip >= 1`` is the lateral/torsional restraint factor: the
        plating restraint is represented conservatively as a penalty (the
        stiffener-alone value is the unrestrained upper bound).
        """
        if lambda_trip <= 0:
            raise ValueError("lambda_trip must be positive")
        L = float(self.a if length is None else length)
        mat = self.material
        g = mat.shear_modulus
        sigma = (g * self.j_saint_venant
                 + np.pi**2 * mat.E * self.warping_constant / L**2) / self.i_p_shear_centre
        return float(sigma / lambda_trip)

    # ---- lateral-torsional buckling reference ---------------------------- #
    def ltb_critical_moment(self, length: Optional[float] = None) -> float:
        """Classical uniform-moment elastic LTB moment ``Mcr`` [N m].

        ``Mcr = (pi/L) sqrt(E I_y G J) sqrt(1 + pi^2 E C_w/(L^2 G J))``
        (identical to ``tensorfem.marine_stiffener_failure``'s oracle).
        """
        L = float(self.a if length is None else length)
        mat = self.material
        e_iy = mat.E * self.stiffener_i_y
        gj = mat.shear_modulus * self.j_saint_venant
        amp = math.sqrt(max(1.0 + np.pi**2 * mat.E * self.warping_constant / (L**2 * gj), 0.0))
        return float((np.pi / L) * math.sqrt(e_iy * gj) * amp)

    def ltb_stress(self, length: Optional[float] = None) -> float:
        """Axial-stress equivalent of ``Mcr``: ``Mcr / (I_ps / c)`` [Pa]."""
        return float(self.ltb_critical_moment(length) / (self.i_p_shear_centre / self.c_from_sc))

    # ---- elastoplastic tripping capacity --------------------------------- #
    def tripping_capacity(
        self, lambda_trip: float = 1.0, length: Optional[float] = None,
    ) -> TrippingResult:
        """Elastoplastic tripping stress with failure-mode classification."""
        sigma_y = self.material.sigma_y0
        sigma_et = self.elastic_tripping_stress(lambda_trip, length)
        if sigma_et <= 0:
            return TrippingResult(0.0, lambda_trip, "tripping", 0.0)
        lam_t = math.sqrt(sigma_y / sigma_et)
        reduction = _column_curve(lam_t)
        sigma = reduction * sigma_y
        if lam_t <= 0.2:
            mode = "yielding"
        elif lam_t < 1.5:
            mode = "tripping"
        else:
            mode = "elastic_tripping"
        return TrippingResult(
            sigma_trip=float(sigma),
            lambda_trip=float(lambda_trip),
            failure_mode=mode,
            capacity_reduction=float(reduction),
        )


def ltb_critical_moment(
    *, length: float, young: float, shear_modulus: float,
    weak_axis_inertia: float, torsion_constant: float, warping_constant: float,
) -> float:
    """Closed-form uniform-moment elastic LTB moment ``Mcr`` [N m].

    ``Mcr = (pi/L) sqrt(E I_y G J) sqrt(1 + pi^2 E C_w/(L^2 G J))``.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    gj = shear_modulus * torsion_constant
    amp = math.sqrt(max(1.0 + np.pi**2 * young * warping_constant / (length**2 * gj), 0.0))
    return float((math.pi / length) * math.sqrt(young * weak_axis_inertia * gj) * amp)


def ltb_numerical(
    *, segments: int, length: float = 3.0, young: float = 2.06e11,
    shear_modulus: float = 79.2e9, weak_axis_inertia: float = 8.0e-7,
    torsion_constant: float = 1.2e-7, warping_constant: float = 2.5e-8,
) -> Dict[str, float]:
    """LTB moment from a centred-difference Dirichlet Laplacian eigenvalue.

    Mirrors ``tensorfem.marine_stiffener_failure.lateral_torsional_buckling_screen``
    so that the closed form can be verified by an independent discretisation.
    """
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 2:
        raise ValueError("segments must be an integer of at least two")
    if length <= 0 or young <= 0 or shear_modulus <= 0:
        raise ValueError("length/young/shear_modulus must be positive")
    h = length / (segments + 1)
    wave_sq = 4.0 * math.sin(math.pi / (2.0 * (segments + 1))) ** 2 / h**2
    computed = math.sqrt(young * weak_axis_inertia
                         * (shear_modulus * torsion_constant * wave_sq
                            + young * warping_constant * wave_sq**2))
    exact = (math.pi / length) ** 2
    oracle = math.sqrt(young * weak_axis_inertia
                       * (shear_modulus * torsion_constant * exact
                          + young * warping_constant * exact**2))
    return {
        "critical_moment": computed,
        "oracle_moment": oracle,
        "relative_error": abs(computed / oracle - 1.0),
        "discrete_wave_number_squared": wave_sq,
        "oracle_wave_number_squared": exact,
    }


def tripping_capacity(
    a: float,
    b: float,
    t: float,
    h_s: float,
    t_w: float,
    b_f: float,
    t_f: float,
    material: VonMisesMaterial,
    lambda_trip: float = 1.0,
) -> TrippingResult:
    """Convenience wrapper for the tripping capacity."""
    analysis = TrippingAnalysis(
        a=a, b=b, t=t, h_s=h_s, t_w=t_w, b_f=b_f, t_f=t_f, material=material,
    )
    return analysis.tripping_capacity(lambda_trip)
