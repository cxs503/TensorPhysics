"""Residual (damage) stability: hydrostatics, GZ curve and IMO criteria.

Campaign-node item (f).  The hull is described by station sections (convex
polygons in the ``(y, z)`` plane); for a given heel angle and trim the
submerged part of every station is obtained by clipping the section with the
free-surface plane ``z - y tan(theta) = c - x tan(psi)``.  Station areas,
moments and hence the whole hydrostatics follow from the shoelace formula, so
the model is exact for polygonal sections (the limiting error is the station
spacing, not an approximation of the physics).

Floating equilibrium (2 unknowns ``c``, ``psi``) enforces volume and
longitudinal moment balance, which is the standard free-floating condition.
Flooded water is treated with the *added-weight* method (lost-buoyancy is
equivalent for the final GZ curve; the difference is documented as a
limitation), and the free-surface effect of the flooded water is applied as an
effective rise of the centre of gravity.

Damage-stability criteria follow the campaign node (area to 30 deg > 0.055
m.rad etc.); the thresholds are exposed as data (``IMO_CRITERIA``) so that the
applicable code (SOLAS II-1 / 2008 IS Code) can be selected without touching
the physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["HullForm", "Section", "Hydrostatics", "IMO_CRITERIA",
           "gz_curve", "check_imo_criteria", "RHO_SEA", "G"]

G = 9.80665
RHO_SEA = 1025.0


# ---------------------------------------------------------------------------
# Section geometry and clipping
# ---------------------------------------------------------------------------
@dataclass
class Section:
    """Convex station section as a closed polygon in the (y, z) plane."""

    x: float
    points: np.ndarray      # (m, 2) counter-clockwise

    def area(self) -> float:
        p = self.points
        y, z = p[:, 0], p[:, 1]
        return 0.5 * float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))

    def centroid(self) -> Tuple[float, float]:
        p = self.points
        y, z = p[:, 0], p[:, 1]
        yn, zn = np.roll(y, -1), np.roll(z, -1)
        cross = y * zn - yn * z
        a = 0.5 * cross.sum()
        if abs(a) < 1e-12:
            return 0.0, 0.0
        cy = ((y + yn) * cross).sum() / (6.0 * a)
        cz = ((z + zn) * cross).sum() / (6.0 * a)
        return float(cy), float(cz)


def clip_half_plane(points: np.ndarray, tan_theta: float, limit: float) -> np.ndarray:
    """Keep the part of the polygon with ``z - y tan(theta) <= limit``."""
    n = len(points)
    if n == 0:
        return points
    out: List[Tuple[float, float]] = []
    for i in range(n):
        p = points[i]
        q = points[(i + 1) % n]
        fp = p[1] - p[0] * tan_theta - limit
        fq = q[1] - q[0] * tan_theta - limit
        if fp <= 0.0:
            out.append((p[0], p[1]))
        if (fp < 0.0) != (fq < 0.0):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# Hull form
# ---------------------------------------------------------------------------
@dataclass
class HullForm:
    """Parametric hull form (wall-sided with flat of bottom and bilge knuckle).

    The waterline half-breadth follows ``y_w(x) = (B/2) (1 - |2x/L|^n)^(1/2)``;
    the fullness exponent ``n`` is solved so that the block coefficient at the
    design draft matches ``cb`` (geometry gate: Cb / L / B are verified before
    any hydrostatic result is used).
    """

    length: float = 148.0
    breadth: float = 23.0
    depth: float = 12.5
    draft: float = 8.6
    cb: float = 0.76
    flat_ratio: float = 0.80       # flat-of-bottom width / local breadth
    bilge_height: float = 0.30     # knuckle height above baseline
    n_stations: int = 61
    fullness: Optional[float] = None

    # ------------------------------------------------------------------ form
    def half_breadth(self, x: np.ndarray | float) -> np.ndarray:
        n = self.fullness if self.fullness is not None else 4.0
        x = np.asarray(x, dtype=float)
        if n > 100.0:                      # rectangular (barge) waterline
            return np.full_like(x, 0.5 * self.breadth)
        s = np.clip(1.0 - np.abs(2.0 * x / self.length) ** n, 0.0, None)
        return 0.5 * self.breadth * np.sqrt(s)

    def x_grid(self) -> np.ndarray:
        return np.linspace(-0.5 * self.length, 0.5 * self.length, int(self.n_stations))

    def section(self, x: float) -> Section:
        yw = float(self.half_breadth(x))
        yf = self.flat_ratio * yw
        zb = self.bilge_height
        zt = self.depth
        pts = np.array([
            [-yf, 0.0], [yf, 0.0], [yw, zb], [yw, zt], [-yw, zt], [-yw, zb]
        ])
        if yf <= 1e-9:                      # degenerate section (apex)
            pts = np.array([[0.0, 0.0], [yw, zb], [yw, zt], [-yw, zt], [-yw, zb]])
        return Section(x=x, points=pts)

    def sections(self) -> List[Section]:
        return [self.section(x) for x in self.x_grid()]

    # ------------------------------------------------------------ properties
    def displacement_volume(self, draft: Optional[float] = None) -> float:
        """Moulded displacement volume at even keel (m^3)."""
        draft = self.draft if draft is None else float(draft)
        xs = self.x_grid()
        secs = self.sections()
        areas = np.array([clip_area(s.points, 0.0, draft) for s in secs])
        return float(np.trapezoid(areas, xs))

    def solve_fullness(self) -> float:
        """Fullness exponent reproducing the required block coefficient."""
        target = self.cb
        lo, hi = 1.2, 12.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            self.fullness = mid
            cb = self.displacement_volume() / (self.length * self.breadth * self.draft)
            if cb < target:
                lo = mid
            else:
                hi = mid
        self.fullness = 0.5 * (lo + hi)
        return self.fullness

    def gate(self) -> Dict[str, float]:
        """Geometry gate: L, B, Cb actually realised by the hull form."""
        if self.fullness is None:
            self.solve_fullness()
        v = self.displacement_volume()
        return {"length": self.length, "breadth": self.breadth,
                "draft": self.draft, "volume": v,
                "cb": v / (self.length * self.breadth * self.draft),
                "cb_target": self.cb, "fullness": float(self.fullness)}

    # ------------------------------------------------------- hydrostatics
    def properties(self, heel_deg: float, c: float, trim_deg: float = 0.0
                   ) -> Dict[str, float]:
        """Volume and buoyancy-centre coordinates of the heeled/trimmed hull."""
        tan_t = np.tan(np.deg2rad(heel_deg))
        tan_p = np.tan(np.deg2rad(trim_deg))
        xs = self.x_grid()
        secs = self.sections()
        vol = 0.0
        mx = my = mz = 0.0
        prev = None
        for x, s in zip(xs, secs):
            sub = clip_half_plane(s.points, tan_t, c - x * tan_p)
            a, cy, cz = poly_area_centroid(sub)
            cur = (a, a * cy, a * cz)
            if prev is not None:
                dx = x - prev[0]
                vol += 0.5 * (cur[0] + prev[1][0]) * dx
                my += 0.5 * (cur[1] + prev[1][1]) * dx
                mz += 0.5 * (cur[2] + prev[1][2]) * dx
                mx += 0.5 * (x * cur[0] + prev[0] * prev[1][0]) * dx
            prev = (x, cur)
        if vol <= 0.0:
            return {"volume": 0.0, "yb": 0.0, "zb": 0.0, "xb": 0.0}
        return {"volume": vol, "yb": my / vol, "zb": mz / vol, "xb": mx / vol}

    def equilibrium(self, heel_deg: float, volume: float, lcg: float,
                    c0: Optional[float] = None, max_iter: int = 60,
                    tol: float = 1e-7) -> Dict[str, float]:
        """Free-floating equilibrium: find ``(c, trim)`` for ``V`` and ``LCG``."""
        c = self.draft if c0 is None else float(c0)
        trim = 0.0
        for _ in range(int(max_iter)):
            st = self.properties(heel_deg, c, trim)
            r1 = st["volume"] - volume
            r2 = st["xb"] - lcg
            if abs(r1) < tol * max(volume, 1.0) and abs(r2) < 1e-4:
                break
            h1 = 1e-4 * max(c, 1.0)
            h2 = 1e-4
            dc = self.properties(heel_deg, c + h1, trim)
            dt = self.properties(heel_deg, c, trim + h2)
            J = np.array([
                [(dc["volume"] - st["volume"]) / h1, (dt["volume"] - st["volume"]) / h2],
                [(dc["xb"] - st["xb"]) / h1, (dt["xb"] - st["xb"]) / h2],
            ])
            try:
                step = np.linalg.solve(J, -np.array([r1, r2]))
            except np.linalg.LinAlgError:
                step = np.array([-0.1 * r1 / max(abs(J[0, 0]), 1e-9), 0.0])
            c += float(np.clip(step[0], -1.0, 1.0))
            trim += float(np.clip(step[1], -0.5, 0.5))
        st = self.properties(heel_deg, c, trim)
        st.update({"c": c, "trim": trim})
        return st

    def waterplane_area(self, draft: Optional[float] = None) -> float:
        """Waterplane area at even keel (m^2)."""
        draft = self.draft if draft is None else float(draft)
        xs = self.x_grid()
        ys = self.half_breadth(xs)
        return float(np.trapezoid(2.0 * np.minimum(ys, ys + 0.0) * (ys > 1e-9), xs))

    def waterplane_inertia_transverse(self, draft: Optional[float] = None) -> float:
        """Transverse waterplane second moment ``I_T`` at even keel (m^4)."""
        draft = self.draft if draft is None else float(draft)
        xs = self.x_grid()
        ys = self.half_breadth(xs)
        return float(np.trapezoid(2.0 / 3.0 * ys**3, xs))

    def metacentric_radius(self, draft: Optional[float] = None) -> float:
        draft = self.draft if draft is None else float(draft)
        return self.waterplane_inertia_transverse(draft) / self.displacement_volume(draft)


def clip_area(points: np.ndarray, tan_theta: float, limit: float) -> float:
    sub = clip_half_plane(points, tan_theta, limit)
    if len(sub) < 3:
        return 0.0
    return Section(x=0.0, points=sub).area()


def poly_area_centroid(points: np.ndarray) -> Tuple[float, float, float]:
    if len(points) < 3:
        return 0.0, 0.0, 0.0
    s = Section(x=0.0, points=points)
    a = s.area()
    if abs(a) < 1e-12:
        return 0.0, 0.0, 0.0
    cy, cz = s.centroid()
    return abs(a), cy, cz


# ---------------------------------------------------------------------------
# GZ curve and IMO criteria
# ---------------------------------------------------------------------------
#: Damage-stability criteria (per campaign node; thresholds exposed as data).
IMO_CRITERIA = {
    "area_to_30": 0.055,        # m rad, area under GZ up to 30 deg
    "area_to_40": 0.090,        # m rad, area under GZ up to 40 deg (or downflooding angle)
    "max_gz": 0.10,             # m
    "angle_max_gz": 15.0,       # deg
    "range": 20.0,              # deg
    "gm_min": 0.05,             # m (after flooding)
}


def gz_curve(hull: HullForm, kg: float, volume: float, lcg: float,
             angles: Sequence[float] = tuple(np.arange(0.0, 61.0, 5.0)),
             free_surface_moment: float = 0.0) -> Dict[str, np.ndarray]:
    """GZ curve of the (flooded) ship.

    ``KG_eff = KG + FSM / (rho V)`` accounts for the free surface of the flooded
    water; ``GZ = KN - KG_eff sin(theta)`` with ``KN = y_B cos + z_B sin``.
    """
    kg_eff = kg + free_surface_moment / (RHO_SEA * max(volume, 1e-9))
    out_ang, out_gz, out_kn, out_trim = [], [], [], []
    c0 = hull.draft
    for th in angles:
        st = hull.equilibrium(th, volume, lcg, c0=c0)
        c0 = st.get("c", hull.draft)
        th_r = np.deg2rad(th)
        kn = st["yb"] * np.cos(th_r) + st["zb"] * np.sin(th_r)
        out_ang.append(th)
        out_gz.append(kn - kg_eff * np.sin(th_r))
        out_kn.append(kn)
        out_trim.append(st.get("trim", 0.0))
    return {"angle_deg": np.asarray(out_ang), "gz": np.asarray(out_gz),
            "kn": np.asarray(out_kn), "trim_deg": np.asarray(out_trim),
            "kg_effective": kg_eff}


def check_imo_criteria(angles: Sequence[float], gz: Sequence[float],
                       gm: Optional[float] = None,
                       criteria: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Evaluate the damage-stability criteria on a GZ curve."""
    crit = dict(IMO_CRITERIA)
    if criteria:
        crit.update(criteria)
    a = np.asarray(angles, dtype=float)
    g = np.asarray(gz, dtype=float)
    rad = np.deg2rad(a)

    def area_to(limit: float) -> float:
        m = a <= limit + 1e-9
        if m.sum() < 2:
            return 0.0
        return float(np.trapezoid(g[m], rad[m]))

    area30 = area_to(30.0)
    area40 = area_to(40.0)
    i_max = int(np.argmax(g))
    gz_max = float(g[i_max])
    ang_max = float(a[i_max])
    # range of positive stability
    rng = 0.0
    for th, val in zip(a, g):
        if val > 0.0:
            rng = float(th)
    out = {
        "area_to_30": area30, "area_to_40": area40,
        "gz_max": gz_max, "angle_max_gz": ang_max, "range": rng,
        "area_to_30_ok": float(area30 >= crit["area_to_30"]),
        "area_to_40_ok": float(area40 >= crit["area_to_40"]),
        "gz_max_ok": float(gz_max >= crit["max_gz"]),
        "angle_max_ok": float(ang_max >= crit["angle_max_gz"]),
        "range_ok": float(rng >= crit["range"]),
        "gm_ok": float(gm is not None and gm >= crit["gm_min"]),
        "criteria": crit,
    }
    out["all_ok"] = float(all(out[k] == 1.0 for k in
                              ("area_to_30_ok", "area_to_40_ok", "gz_max_ok",
                               "angle_max_ok", "range_ok")) and
                          (out["gm_ok"] == 1.0 if gm is not None else True))
    return out