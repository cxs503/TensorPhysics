"""Full-spectrum ice resistance model family (three ice states).

This module owns the *ice state -> ship resistance -> speed-power* chain of the
ship overall-performance domain: the three physical regimes a Polar-Class ship
meets in a single voyage are modelled by three closed-form branches,

================  ==========================================================
state             model
================  ==========================================================
``level``         intact level ice, Lindqvist (1989) three components
                  ``R = R_crush + R_break + R_submersion``
``brash``         broken ice / partial ice cover, coverage-decay law
                  ``D(c)`` (linear clearing regime + packing amplification)
``ridge``         ice ridge ramming, Keinonen-style buoyancy-displacement
                  term plus floe-momentum empirical term
================  ==========================================================

Every branch is a deterministic closed form (no fitted surrogate, no random
term), exposes its component breakdown, and enforces an explicit validity
domain.  Outside the domain the evaluator either *refuses* (``strict``) or
*truncates and flags* the request -- empirical laws are never silently
extrapolated (``h > 0.03 L`` etc., see :class:`DomainLimits`).

Evidence grading (TensorPhysics vocabulary)
-------------------------------------------
* implementation  : REAL -- closed-form physics, component-resolved.
* coefficients    : LITERATURE / SEMI-EMPIRICAL -- see
  :data:`PARAMETER_PROVENANCE`; transcription uncertainty is propagated and
  reported (``sensitivity`` helpers), never hidden.
* accuracy vs full-scale measurement : WITHHELD -- no digitised full-scale ice
  resistance record is available in this repository, therefore absolute
  accuracy is *not* claimed; validity is established against analytic
  identities, published scaling laws of the cited laws, cross-law consistency
  and a propulsion-power plausibility band (see
  ``benchmarks/ice_resistance/corpus/l2_anchors.json``).

Units: SI throughout (m, s, N, W), except where a name carries ``_kn``,
``_MW``, ``_deg`` or ``_t``.

References
----------
* Lindqvist, G. (1989) "A straightforward method for calculation of ice
  resistance of ships", POAC'89, Lulea, Vol. 2, 722-735.
* Keinonen, A. (1983) "Icebreaking model testing -- a review of recent
  developments", POAC'83, Helsinki, 429-444.
* Enkvist, E. (1972) "On the ice resistance encountered by ships", Acta
  Polytechnica Scandinavica, Publication 24.
* Riska, K. (1997) "On the mechanics of ice breaking with a double acting
  ship", POAC'97, Vol. 2, 303-314.
* ISO 19906:2019 / ITTC (2017) Ice Committee: compressive and flexural
  strength ranges of sea ice.
* Timco, G.W. & Weeks, W.F. (2010) "A review of the engineering properties of
  sea ice", Cold Regions Science and Technology 60, 107-129.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

G = 9.80665  # m/s^2
RHO_ICE = 900.0  # kg/m^3, first-year sea ice (Timco & Weeks 2010)
RHO_WATER = 1025.0  # kg/m^3
MU_HULL_ICE = 0.15  # steel/ice friction (Riska 1997)
SIGMA_C_REF = 5.0e6  # Pa, ITTC/ISO 19906 ice compressive strength range
SIGMA_F_REF = 0.5e6  # Pa, sea-ice flexural strength (slow loading)
E_ICE = 5.0e9  # Pa
NU_ICE = 0.3

#: Provenance of every physical constant / empirical closure coefficient used
#: by this module.  ``grade`` uses the platform evidence vocabulary.
PARAMETER_PROVENANCE: Dict[str, Dict[str, Any]] = {
    "sigma_c_Pa": {
        "value": SIGMA_C_REF,
        "unit": "Pa",
        "meaning": "冰压溃强度（有效平均，非局部峰值）",
        "source": "ITTC(2017) Ice Committee / ISO 19906:2019 范围 2-8 MPa，取 5 MPa",
        "grade": "LITERATURE-INTERVAL",
    },
    "sigma_f_Pa": {
        "value": SIGMA_F_REF,
        "unit": "Pa",
        "meaning": "海冰弯曲强度（慢加载）",
        "source": "Timco & Weeks (2010); ITTC(2017) 0.35-0.7 MPa",
        "grade": "LITERATURE-INTERVAL",
    },
    "mu_hull_ice": {
        "value": MU_HULL_ICE,
        "unit": "-",
        "meaning": "船体-冰摩擦系数",
        "source": "Riska (1997) 0.10-0.20，取 0.15",
        "grade": "LITERATURE-INTERVAL",
    },
    "flare_deg": {
        "value": 25.0,
        "unit": "deg",
        "meaning": "艏部外飘板与水平面夹角 alpha（Lindqvist R_crush 用角）",
        "source": "冰区船艏部水线附近外飘角 20-35°",
        "grade": "SEMI-EMPIRICAL",
    },
    "k_bend": {
        "value": 2.0,
        "unit": "-",
        "meaning": "弯曲破冰闭合常数 R_b = 0.5 sigma_f h B (sin(phi/2)+mu cos(phi/2)) / k_bend",
        "source": "由 Lindqvist 冰尖弯曲概念（冰尖长度 ~ k_bend·h）标定；"
        "取值与两条独立文献总阻力定律（Keinonen 1983 / Enkvist 1972）同量级自洽，"
        "±50% 敏感性在报告 §5 给出（非独立验证，属闭合常数标定）",
        "grade": "SEMI-EMPIRICAL",
    },
    "c_crit": {
        "value": 0.6,
        "unit": "-",
        "meaning": "碎冰覆盖率临界值：以下为可清道区、以上进入堆积区",
        "source": "碎冰通道/冰池模型试验中阻力在覆盖率约 0.6 后陡增",
        "grade": "SEMI-EMPIRICAL",
    },
    "packing_A": {
        "value": 1.0,
        "unit": "-",
        "meaning": "堆积区闭合强度：D = c + (1-c) A w，A=1 表示 c->1 时阻力完全趋近平整冰",
        "source": "工程闭合（端点 D(0)=0、D(1)=1、清除区 D=c 由物理约束固定；A 为区间 [0,1] 内参数）",
        "grade": "SEMI-EMPIRICAL",
    },
    "ridge_C1": {
        "value": 0.35,
        "unit": "-",
        "meaning": "冰脊浮力排开项系数 C1 (1-e) rho_i g h_k^2 B",
        "source": "Keinonen (1983) 冰脊冲撞经验量级；系数转录不确定性 ±30%",
        "grade": "SEMI-EMPIRICAL",
    },
    "ridge_C2": {
        "value": 0.5,
        "unit": "-",
        "meaning": "冰脊碎块动量项系数 C2 (1-e) rho_i h_k B V^2",
        "source": "Keinonen (1983) 冰脊冲撞速度相关项；±30%",
        "grade": "SEMI-EMPIRICAL",
    },
    "eta_D": {
        "value": 0.55,
        "unit": "-",
        "meaning": "总推进效率（准推进效率 x 轴系 x 伴流/推力减额）",
        "source": "冰区船设计典型 0.45-0.65",
        "grade": "LITERATURE-INTERVAL",
    },
}


class IceDomainError(ValueError):
    """Requested ice state / speed lies outside the validity domain of the law."""


# ---------------------------------------------------------------------------
# geometry / state containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShipGeometry:
    """Mono-hull ship geometry used by the ice-resistance laws."""

    name: str = "PC6 reference ship"
    L_pp: float = 150.0  # m
    B: float = 24.0  # m
    T: float = 8.5  # m
    bow_angle_deg: float = 30.0  # 水线面艏半角 phi（平面内）
    flare_deg: float = float(PARAMETER_PROVENANCE["flare_deg"]["value"])
    displacement_t: float = 21000.0
    installed_power_MW: float = 22.0
    eta_D: float = float(PARAMETER_PROVENANCE["eta_D"]["value"])

    # -- derived helpers ---------------------------------------------------
    @property
    def bow_angle_rad(self) -> float:
        return math.radians(self.bow_angle_deg)

    @property
    def flare_rad(self) -> float:
        return math.radians(self.flare_deg)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "L_pp_m": self.L_pp,
            "B_m": self.B,
            "T_m": self.T,
            "bow_angle_deg": self.bow_angle_deg,
            "flare_deg": self.flare_deg,
            "displacement_t": self.displacement_t,
            "installed_power_MW": self.installed_power_MW,
            "eta_D": self.eta_D,
        }


@dataclass(frozen=True)
class DomainLimits:
    """Validity domain of the empirical resistance laws.

    The classic semi-empirical level-ice laws were regressed on ships where the
    ice thickness stays well below ``0.03 L`` and below ``0.10 B``; beyond that
    the ship is in a *compaction / besetting* regime that the laws do not
    represent.  Ridge laws are limited by the hull entrance geometry.
    """

    h_over_L_max: float = 0.03  # h <= 0.03 L
    h_over_B_max: float = 0.10  # h <= 0.10 B
    h_min_m: float = 0.05
    froude_L_max: float = 0.25  # V <= 0.25 sqrt(g L)
    coverage_min: float = 0.0
    coverage_max: float = 1.0
    ridge_keel_over_B_max: float = 0.15  # h_k <= 0.15 B
    ridge_keel_max_m: float = 4.0
    ridge_porosity_min: float = 0.05
    ridge_porosity_max: float = 0.60

    def h_limit(self, ship: ShipGeometry) -> float:
        return min(self.h_over_L_max * ship.L_pp, self.h_over_B_max * ship.B)

    def v_limit(self, ship: ShipGeometry) -> float:
        return self.froude_L_max * math.sqrt(G * ship.L_pp)

    def ridge_limit(self, ship: ShipGeometry) -> float:
        return min(self.ridge_keel_over_B_max * ship.B, self.ridge_keel_max_m)

    def as_dict(self, ship: ShipGeometry) -> Dict[str, Any]:
        return {
            "h_limit_m": self.h_limit(ship),
            "h_over_L_max": self.h_over_L_max,
            "h_over_B_max": self.h_over_B_max,
            "h_min_m": self.h_min_m,
            "v_limit_ms": self.v_limit(ship),
            "froude_L_max": self.froude_L_max,
            "ridge_keel_limit_m": self.ridge_limit(ship),
            "ridge_porosity_range": [self.ridge_porosity_min, self.ridge_porosity_max],
        }


@dataclass(frozen=True)
class IceState:
    """One ice condition (one sample of the ice-condition spectrum)."""

    kind: str  # "level" | "brash" | "ridge"
    thickness_m: float = 1.0  # level-ice / floe thickness
    coverage: float = 1.0  # broken-ice coverage c (brash branch)
    ridge_keel_m: float = 0.0  # ridge keel depth h_k
    ridge_spacing_m: float = 0.0  # mean ridge spacing (0 = single ridge)
    ridge_porosity: float = 0.30
    label: str = ""

    def key(self) -> str:
        k = self.kind
        if k == "level":
            return f"level_h{self.thickness_m:.2f}"
        if k == "brash":
            return f"brash_h{self.thickness_m:.2f}_c{self.coverage:.2f}"
        if k == "ridge":
            return f"ridge_hk{self.ridge_keel_m:.2f}"
        return k


# ---------------------------------------------------------------------------
# branch 1: level ice -- Lindqvist (1989) three components
# ---------------------------------------------------------------------------


def level_ice_resistance(
    h: float,
    V: float,
    ship: ShipGeometry,
    *,
    sigma_c: float = SIGMA_C_REF,
    sigma_f: float = SIGMA_F_REF,
    mu: float = MU_HULL_ICE,
    rho_i: float = RHO_ICE,
    k_bend: float = float(PARAMETER_PROVENANCE["k_bend"]["value"]),
    limits: Optional[DomainLimits] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Lindqvist (1989) level-ice resistance, resolved into its three components.

    ``R_crush``      quasi-static crushing of the ice edge at the bow shoulder::

        R_c = 0.5 * sigma_c * h^2 * (tan(alpha) + mu / cos(alpha))

    with ``alpha`` the *flare* angle of the bow plating measured from the
    horizontal (NOT ``90 deg - phi/2``; see ``notes/audit`` in the benchmark
    report -- the ``90 deg - phi/2`` convention inflates ``tan(alpha)`` by ~6x
    and is the documented defect of ``ice_anchor.anchors.lindqvist_1989``).

    ``R_break``      bending failure of the ice cusp in front of the bow::

        R_b = 0.5 * sigma_f * h * B * (sin(phi/2) + mu cos(phi/2)) / k_bend

    ``R_submersion`` submergence / friction of the broken ice under the hull::

        R_s = C_s * rho_i * V^2 * h * B,   C_s = 0.15 + 0.3 * Fr_h,
        Fr_h = V / sqrt(g h)

    Parameters
    ----------
    h, V : float
        Level-ice thickness [m] and ship speed [m/s].
    strict : bool
        ``True`` -> raise :class:`IceDomainError` outside the validity domain.
        ``False`` -> clip to the domain and report it in ``flags``.
    """

    limits = limits or DomainLimits()
    flags: List[str] = []
    h_raw, V_raw = h, V

    if not math.isfinite(h) or not math.isfinite(V):
        raise IceDomainError(f"non-finite input: h={h!r}, V={V!r}")
    if h <= 0.0:
        raise IceDomainError(f"ice thickness must be > 0, got {h!r}")
    if V < 0.0:
        raise IceDomainError(f"speed must be >= 0, got {V!r}")

    h_lim = limits.h_limit(ship)
    v_lim = limits.v_limit(ship)
    if strict:
        if h > h_lim:
            raise IceDomainError(
                f"h={h:.3f} m exceeds ice-law validity limit {h_lim:.3f} m "
                f"(h <= {limits.h_over_L_max}*L, h <= {limits.h_over_B_max}*B); "
                "refusing to extrapolate the empirical law"
            )
        if h < limits.h_min_m:
            raise IceDomainError(f"h={h:.3f} m below validity floor {limits.h_min_m} m")
        if V > v_lim:
            raise IceDomainError(
                f"V={V:.3f} m/s exceeds Froude limit {v_lim:.3f} m/s "
                f"(Fr_L <= {limits.froude_L_max})"
            )
    else:
        if h > h_lim:
            h = h_lim
            flags.append(f"H_TRUNCATED:{h_raw:.3f}->{h:.3f}")
        if h < limits.h_min_m:
            h = limits.h_min_m
            flags.append(f"H_FLOOR:{h_raw:.3f}->{h:.3f}")
        if V > v_lim:
            V = v_lim
            flags.append(f"V_TRUNCATED:{V_raw:.3f}->{V:.3f}")

    phi = ship.bow_angle_rad
    alpha = ship.flare_rad

    # -- crushing (quasi-static, speed independent) ------------------------
    c_crush = 0.5 * (math.tan(alpha) + mu / math.cos(alpha))
    r_crush = c_crush * sigma_c * h**2

    # -- bending failure of the ice cusp -----------------------------------
    wedge = math.sin(0.5 * phi) + mu * math.cos(0.5 * phi)
    c_break = 0.5 * wedge / k_bend
    r_break = c_break * sigma_f * h * ship.B

    # -- submergence / friction of broken ice ------------------------------
    fr_h = V / math.sqrt(G * h)
    c_sub = 0.15 + 0.3 * fr_h
    r_sub = c_sub * rho_i * V**2 * h * ship.B

    total = r_crush + r_break + r_sub
    return {
        "state": "level",
        "h_m": h,
        "V_ms": V,
        "V_kn": V / 0.514444,
        "R_crush_N": r_crush,
        "R_break_N": r_break,
        "R_submersion_N": r_sub,
        "R_total_N": total,
        "R_total_kN": total / 1e3,
        "C_crush": c_crush,
        "C_break": c_break,
        "C_submersion": c_sub,
        "Fr_h": fr_h,
        "Fr_L": V / math.sqrt(G * ship.L_pp),
        "flags": flags,
        "domain_ok": not flags,
        "requested": {"h_m": h_raw, "V_ms": V_raw},
    }


# ---------------------------------------------------------------------------
# branch 2: brash ice / partial cover -- coverage decay law
# ---------------------------------------------------------------------------


def coverage_decay(
    c: float,
    *,
    c_crit: float = float(PARAMETER_PROVENANCE["c_crit"]["value"]),
    packing_A: float = float(PARAMETER_PROVENANCE["packing_A"]["value"]),
) -> Dict[str, float]:
    """Coverage-decay law ``D(c)`` multiplying the intact-sheet resistance.

    ``c <= c_crit``   free-clearing regime: ``D = c`` -- the number of floe
                      encounters, and hence the resistance, is proportional to
                      the ice-covered fraction.
    ``c  > c_crit``   packing regime: ``D = c + (1 - c) A w`` with
                      ``w = ((c - c_crit)/(1 - c_crit))^2`` and ``A <= 1``.
                      The channel can no longer be cleared, so the resistance
                      is pulled towards the intact-sheet value as ``c -> 1``.

    Properties (all used as anchors): ``D(0) = 0``, ``D(1) = 1``,
    ``D(c) = c`` on the clearing branch, ``0 <= D <= 1`` and monotone
    non-decreasing on ``[0, 1]``, with a slope ``> 1`` above ``c_crit``.
    """

    if not math.isfinite(c):
        raise IceDomainError(f"non-finite coverage: {c!r}")
    if c < 0.0 or c > 1.0:
        raise IceDomainError(f"coverage must be within [0, 1], got {c!r}")

    if c <= c_crit:
        return {"D": c, "regime": "clearing", "w": 0.0}

    w = min(((c - c_crit) / (1.0 - c_crit)) ** 2, 1.0)
    dev = c + (1.0 - c) * packing_A * w
    return {"D": dev, "regime": "packing", "w": w}


def brash_ice_resistance(
    h: float,
    V: float,
    c: float,
    ship: ShipGeometry,
    *,
    C_impact: float = 0.5,
    sigma_c: float = SIGMA_C_REF,
    sigma_f: float = SIGMA_F_REF,
    mu: float = MU_HULL_ICE,
    rho_i: float = RHO_ICE,
    limits: Optional[DomainLimits] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Brash-ice (broken cover) resistance with an explicit decay law.

    ``R(c) = D(c) * R_level(h, V) + 4 c (1 - c) * R_floe_impact``

    where ``R_floe_impact = 0.5 * C_impact * rho_i * h * V^2 * B`` is the
    momentum flux needed to accelerate the isolated floes; the weight
    ``4 c (1 - c)`` (peak 1 at ``c = 1/2``) expresses that floe impacts require
    *both* floes (``c > 0``) and clear water to accelerate them into
    (``c < 1``).  The two endpoint limits are therefore exact:
    ``R(c=1) = R_level`` (intact sheet) and ``R(c=0) = 0`` (open water).
    """

    decay = coverage_decay(c)
    base = level_ice_resistance(
        h,
        V,
        ship,
        sigma_c=sigma_c,
        sigma_f=sigma_f,
        mu=mu,
        rho_i=rho_i,
        limits=limits,
        strict=strict,
    )
    r_level = base["R_total_N"]
    r_impact = 0.5 * C_impact * rho_i * base["h_m"] * base["V_ms"] ** 2 * ship.B
    r_total = decay["D"] * r_level + 4.0 * c * (1.0 - c) * r_impact

    out = dict(base)
    out.update(
        {
            "state": "brash",
            "coverage": c,
            "decay_factor": decay["D"],
            "decay_regime": decay["regime"],
            "R_level_N": r_level,
            "R_floe_impact_N": r_impact,
            "R_total_N": r_total,
            "R_total_kN": r_total / 1e3,
        }
    )
    return out


# ---------------------------------------------------------------------------
# branch 3: ice ridge ramming -- Keinonen-anchored empirical term
# ---------------------------------------------------------------------------


def ridge_resistance(
    h_keel: float,
    V: float,
    ship: ShipGeometry,
    *,
    porosity: float = 0.30,
    C1: float = float(PARAMETER_PROVENANCE["ridge_C1"]["value"]),
    C2: float = float(PARAMETER_PROVENANCE["ridge_C2"]["value"]),
    rho_i: float = RHO_ICE,
    limits: Optional[DomainLimits] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Ridge penetration (ramming) resistance, Keinonen-anchored empirical form.

    Two physical mechanisms, both scaling with the *solid* (non-porous) ice
    volume that has to be displaced per unit advance::

        R_keel   = C1 (1 - e) rho_i g h_k^2 B     buoyancy / displacement work
        R_moment = C2 (1 - e) rho_i h_k B V^2     floe & rubble momentum flux

    The work needed to penetrate a ridge of keel depth ``h_k`` is
    ``W(h_k) = C1 (1-e) rho_i g B h_k^3 / 3 + C2 (1-e) rho_i B V^2 h_k^2 / 2``
    (:func:`ridge_penetration_work`), used for the energy-based ramming
    capacity and for converting a ridge field with mean spacing ``s`` into a
    field-averaged resistance ``R_avg = W/s`` (see :func:`ridge_field_mean_resistance`).
    """

    limits = limits or DomainLimits()
    if not math.isfinite(h_keel) or not math.isfinite(V):
        raise IceDomainError(f"non-finite input: h_keel={h_keel!r}, V={V!r}")
    if h_keel <= 0.0:
        raise IceDomainError(f"ridge keel depth must be > 0, got {h_keel!r}")
    if V < 0.0:
        raise IceDomainError(f"speed must be >= 0, got {V!r}")
    if not (limits.ridge_porosity_min <= porosity <= limits.ridge_porosity_max):
        raise IceDomainError(
            f"ridge porosity {porosity!r} outside "
            f"[{limits.ridge_porosity_min}, {limits.ridge_porosity_max}] "
            "(outside measured ridge porosity range)"
        )

    flags: List[str] = []
    hk_lim = limits.ridge_limit(ship)
    if h_keel > hk_lim:
        if strict:
            raise IceDomainError(
                f"ridge keel {h_keel:.2f} m exceeds validity limit {hk_lim:.2f} m "
                f"(h_k <= {limits.ridge_keel_over_B_max}*B, h_k <= "
                f"{limits.ridge_keel_max_m} m); refusing to extrapolate"
            )
        flags.append(f"HK_TRUNCATED:{h_keel:.3f}->{hk_lim:.3f}")
        h_keel = hk_lim

    solid = 1.0 - porosity
    r_keel = C1 * solid * rho_i * G * h_keel**2 * ship.B
    r_moment = C2 * solid * rho_i * h_keel * ship.B * V**2
    total = r_keel + r_moment
    return {
        "state": "ridge",
        "h_keel_m": h_keel,
        "porosity": porosity,
        "V_ms": V,
        "V_kn": V / 0.514444,
        "R_keel_N": r_keel,
        "R_momentum_N": r_moment,
        "R_total_N": total,
        "R_total_kN": total / 1e3,
        "flags": flags,
        "domain_ok": not flags,
    }


def ridge_penetration_work(
    h_keel: float,
    V: float,
    ship: ShipGeometry,
    *,
    porosity: float = 0.30,
    C1: float = float(PARAMETER_PROVENANCE["ridge_C1"]["value"]),
    C2: float = float(PARAMETER_PROVENANCE["ridge_C2"]["value"]),
    rho_i: float = RHO_ICE,
    n_steps: int = 400,
) -> Dict[str, float]:
    """Work [J] to penetrate a ridge down to ``h_keel`` (numerical + analytic).

    Returns both the quadrature and the closed form; their relative difference
    is an internal consistency check used by the anchor corpus.
    """

    solid = 1.0 - porosity
    kw = C1 * solid * rho_i * G * ship.B
    mw = C2 * solid * rho_i * ship.B * V**2
    analytic = kw * h_keel**3 / 3.0 + mw * h_keel**2 / 2.0

    n = max(8, int(n_steps))
    dh = h_keel / n
    quad = 0.0
    for i in range(n):
        hk = (i + 0.5) * dh
        quad += (kw * hk**2 + mw * hk) * dh
    return {
        "W_analytic_J": analytic,
        "W_quadrature_J": quad,
        "rel_diff": abs(quad - analytic) / max(analytic, 1e-30),
    }


def ridge_field_mean_resistance(
    h_keel: float,
    V: float,
    ship: ShipGeometry,
    spacing_m: float,
    *,
    background_h: float = 0.0,
    porosity: float = 0.30,
    ridge_width_m: Optional[float] = None,
    limits: Optional[DomainLimits] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Field-averaged resistance of a ridge field of mean spacing ``s``.

    ``R_mean = R_level(h_bg) (1 - w_r/s) + R_ridge(h_k, V) (w_r/s)``

    where ``w_r`` is the ridge cross-section width at the waterline (default
    ``3 h_k``, i.e. a 1:1.5 side slope) -- the ship spends the fraction
    ``w_r/s`` of the distance inside a ridge and the rest in the background
    level ice.  The energy of one crossing, ``W(h_k, V)``
    (:func:`ridge_penetration_work`), is reported alongside so that the ramming
    budget can be audited separately from the resistance average.
    """

    if not math.isfinite(spacing_m) or spacing_m <= 0.0:
        raise IceDomainError(f"ridge spacing must be > 0, got {spacing_m!r}")
    ridge = ridge_resistance(
        h_keel, V, ship, porosity=porosity, limits=limits, strict=strict
    )
    work = ridge_penetration_work(h_keel, V, ship, porosity=porosity)
    w_r = 3.0 * ridge["h_keel_m"] if ridge_width_m is None else float(ridge_width_m)
    if w_r <= 0.0 or w_r >= spacing_m:
        raise IceDomainError(
            f"ridge width {w_r:.2f} m must be in (0, spacing {spacing_m:.2f} m)"
        )
    r_level = 0.0
    if background_h > 0.0:
        r_level = level_ice_resistance(
            background_h, V, ship, limits=limits, strict=strict
        )["R_total_N"]
    frac = w_r / spacing_m
    r_mean = r_level * (1.0 - frac) + ridge["R_total_N"] * frac
    return {
        "state": "ridge_field",
        "h_keel_m": ridge["h_keel_m"],
        "spacing_m": spacing_m,
        "ridge_width_m": w_r,
        "ridge_fraction": frac,
        "background_h_m": background_h,
        "R_level_N": r_level,
        "W_per_ridge_J": work["W_analytic_J"],
        "R_ridge_instantaneous_N": ridge["R_total_N"],
        "R_total_N": r_mean,
        "R_total_kN": r_mean / 1e3,
        "flags": list(ridge["flags"]),
        "domain_ok": ridge["domain_ok"],
    }


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def ice_resistance(
    state: IceState,
    V: float,
    ship: ShipGeometry,
    *,
    limits: Optional[DomainLimits] = None,
    strict: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate the resistance of one ice state at one speed (branch dispatch)."""

    if state.kind == "level":
        return level_ice_resistance(
            state.thickness_m, V, ship, limits=limits, strict=strict, **kwargs
        )
    if state.kind == "brash":
        return brash_ice_resistance(
            state.thickness_m, V, state.coverage, ship, limits=limits, strict=strict, **kwargs
        )
    if state.kind == "ridge":
        if state.ridge_spacing_m > 0.0:
            return ridge_field_mean_resistance(
                state.ridge_keel_m,
                V,
                ship,
                state.ridge_spacing_m,
                background_h=state.thickness_m,
                porosity=state.ridge_porosity,
                limits=limits,
                strict=strict,
            )
        return ridge_resistance(
            state.ridge_keel_m,
            V,
            ship,
            porosity=state.ridge_porosity,
            limits=limits,
            strict=strict,
        )
    raise IceDomainError(f"unknown ice state kind {state.kind!r}")


# ---------------------------------------------------------------------------
# speed - power chain
# ---------------------------------------------------------------------------


def delivered_power_MW(R_total_N: float, V: float, eta_D: float) -> float:
    """Delivered (brake) power ``P = R V / eta_D`` in MW."""

    if not (0.0 < eta_D <= 1.0):
        raise IceDomainError(f"propulsive efficiency must be in (0, 1], got {eta_D!r}")
    return R_total_N * V / eta_D / 1e6


def speed_for_power(
    state: IceState,
    ship: ShipGeometry,
    power_MW: float,
    *,
    limits: Optional[DomainLimits] = None,
    v_lo: float = 0.01,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> Dict[str, Any]:
    """Speed achieved with a given installed power (monotone bisection).

    ``P(V) = R(V) V / eta_D`` is strictly increasing in ``V`` on the validity
    domain, so the bracket is safe; the residual is returned for auditing.
    """

    v_hi = (limits or DomainLimits()).v_limit(ship)
    target = power_MW

    def power_at(v: float) -> float:
        r = ice_resistance(state, v, ship, limits=limits, strict=False)
        return delivered_power_MW(r["R_total_N"], v, ship.eta_D)

    p_lo, p_hi = power_at(v_lo), power_at(v_hi)
    if target <= p_lo:
        v_star = 0.0
        residual = p_lo - target
    elif target >= p_hi:
        v_star = v_hi  # power-limited beyond the validity domain: report the cap
        residual = p_hi - target
    else:
        lo, hi = v_lo, v_hi
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if power_at(mid) < target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        v_star = 0.5 * (lo + hi)
        residual = power_at(v_star) - target
    return {
        "V_ms": v_star,
        "V_kn": v_star / 0.514444,
        "P_requested_MW": target,
        "P_achieved_MW": power_at(v_star),
        "power_residual_MW": residual,
        "capped": target >= p_hi,
        "state": state.key(),
    }


def ram_capacity(
    ship: ShipGeometry,
    ram_speed: float,
    *,
    porosity: float = 0.30,
    C1: float = float(PARAMETER_PROVENANCE["ridge_C1"]["value"]),
    C2: float = float(PARAMETER_PROVENANCE["ridge_C2"]["value"]),
    limits: Optional[DomainLimits] = None,
) -> Dict[str, float]:
    """Deepest ridge keel penetrable by a ram at ``ram_speed`` (energy budget).

    The ship's kinetic energy (with an added-mass factor ``1 + C_a``,
    ``C_a = 0.1`` for a slender hull in the horizontal plane) must exceed the
    ridge penetration work; ``h_k`` follows from the monotone inversion of
    ``W(h_k)`` by bisection.
    """

    mass = ship.displacement_t * 1e3 * 1.1  # +10 % hydrodynamic added mass
    energy = 0.5 * mass * ram_speed**2
    lo, hi = 1e-3, (limits or DomainLimits()).ridge_limit(ship)
    if ridge_penetration_work(hi, ram_speed, ship, porosity=porosity)["W_analytic_J"] <= energy:
        h_cap = hi
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            w = ridge_penetration_work(mid, ram_speed, ship, porosity=porosity)["W_analytic_J"]
            if w < energy:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-9:
                break
        h_cap = 0.5 * (lo + hi)
    return {
        "ram_speed_ms": ram_speed,
        "ram_speed_kn": ram_speed / 0.514444,
        "kinetic_energy_MJ": energy / 1e6,
        "h_keel_capacity_m": h_cap,
        "capacity_capped_by_domain": h_cap >= (limits or DomainLimits()).ridge_limit(ship),
    }


# ---------------------------------------------------------------------------
# sensitivity helper (epistemic honesty: report coefficient leverage)
# ---------------------------------------------------------------------------


def coefficient_sensitivity(
    h: float, V: float, ship: ShipGeometry, *, delta: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """Relative change of ``R_level`` for ``+-delta`` on each closure constant."""

    base = level_ice_resistance(h, V, ship)["R_total_N"]
    out: Dict[str, Dict[str, float]] = {}
    cases = {
        "sigma_c": {"sigma_c": SIGMA_C_REF * (1 + delta)},
        "sigma_c_low": {"sigma_c": SIGMA_C_REF * (1 - delta)},
        "sigma_f": {"sigma_f": SIGMA_F_REF * (1 + delta)},
        "sigma_f_low": {"sigma_f": SIGMA_F_REF * (1 - delta)},
        "k_bend": {"k_bend": float(PARAMETER_PROVENANCE["k_bend"]["value"]) * (1 + delta)},
        "k_bend_low": {"k_bend": float(PARAMETER_PROVENANCE["k_bend"]["value"]) * (1 - delta)},
        "mu": {"mu": MU_HULL_ICE * (1 + delta)},
        "mu_low": {"mu": MU_HULL_ICE * (1 - delta)},
    }
    for name, kw in cases.items():
        r = level_ice_resistance(h, V, ship, **kw)["R_total_N"]
        out[name] = {
            "R_total_N": r,
            "rel_change": (r - base) / base,
        }
    return out