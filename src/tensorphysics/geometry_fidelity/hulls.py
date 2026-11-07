"""Analytic CAD truth sources for the geometry fidelity contract.

Three hulls are described by closed-form *parametric lines* -- a half-breadth
law ``b(x, z)`` -- and two independent integrations of the same law are
provided:

``cad_volume_slice`` / ``cad_surface_area``
    deterministic transverse-slice Simpson quadrature (the conventional
    hydrostatic route: section areas integrated along the length);
``cad_volume_monte_carlo`` / ``cad_surface_monte_carlo``
    stratified Monte-Carlo replication with ``N = 1e6`` samples (a route that
    shares no quadrature weights with the slice integration).

The truth source is mesh independent: the discrete solver families
(:mod:`tensorphysics.geometry_fidelity.mesh`) only ever see ``half_breadth``
and the parametric surface map, never the quadrature grids used here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "HullForm",
    "contract_hulls",
    "kcs_hull",
    "kvlcc2_hull",
    "suboff_hull",
]


def _np():
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - numpy is in the test extra
        raise RuntimeError("numpy is required for Monte-Carlo replication") from exc
    return numpy


def _simpson(f, a: float, b: float, n: int) -> float:
    """Composite Simpson rule with ``n`` (even) panels."""
    if n % 2:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4.0 if i % 2 else 2.0) * f(a + i * h)
    return total * h / 3.0


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


@dataclass(frozen=True)
class HullForm:
    """Analytic CAD truth source for one hull.

    ``family`` selects the half-breadth law:

    ``wall_sided_power``
        mono-hull ships: ``b = B/2 * w(x/L) * (z/T)^(1/m)``; the exponent ``m``
        is solved so that ``Cb`` closes exactly against the contract value.
    ``revolution``
        axisymmetric body (SUBOFF): circular sections ``b^2 + (z - R)^2 = r(x)^2``
        with the standard fore / parallel-middle / aft radii distribution.
    """

    geometry_id: str
    family: str
    length_m: float
    breadth_m: float
    draught_m: float
    params: Mapping[str, float]
    cb_contract: float | None = None
    nabla_contract_m3: float | None = None
    s_contract_m2: float | None = None

    # -- elementary geometry -------------------------------------------------
    def waterline_factor(self, xi: float) -> float:
        """Half-breadth factor on the design waterline, ``xi = x / L``."""
        p = self.params
        xi = _clip01(xi)
        mid = p["mid"]
        if xi <= mid:
            u = (mid - xi) / mid
            core = (1.0 - u ** p["aft_a"]) ** p["aft_b"]
            return p["transom"] + (1.0 - p["transom"]) * core
        u = (xi - mid) / (1.0 - mid)
        return (1.0 - u ** p["fore_a"]) ** p["fore_b"]

    def waterline_factor_slope(self, xi: float) -> float:
        h = 1e-5
        lo, hi = max(0.0, xi - h), min(1.0, xi + h)
        if hi - lo <= 0.0:
            return 0.0
        return (self.waterline_factor(hi) - self.waterline_factor(lo)) / (hi - lo)

    def radius(self, x: float) -> float:
        """Body-of-revolution radius (``revolution`` family only)."""
        p = self.params
        lf, lp, la, re = p["lf"], p["lp"], p["la"], p["aft_radius"]
        big_r = self.breadth_m / 2.0
        if x <= lf:
            u = (lf - x) / lf
            return big_r * (1.0 - u * u) ** p["fore_b"]
        if x >= lf + lp:
            u = (x - lf - lp) / la
            return big_r * (re + (1.0 - re) * (1.0 - u * u) ** p["aft_b"])
        return big_r

    def radius_slope(self, x: float) -> float:
        h = 1e-5 * self.length_m
        lo, hi = max(0.0, x - h), min(self.length_m, x + h)
        if hi - lo <= 0.0:
            return 0.0
        return (self.radius(hi) - self.radius(lo)) / (hi - lo)

    def half_breadth(self, x: float, z: float) -> float:
        """Half-breadth of the solid at ``(x, z)``; ``0`` outside the body."""
        if z < 0.0 or z > self.draught_m:
            return 0.0
        if self.family == "revolution":
            big_r = self.breadth_m / 2.0
            inner = self.radius(x) ** 2 - (z - big_r) ** 2
            return math.sqrt(inner) if inner > 0.0 else 0.0
        if self.family == "wall_sided_power":
            xi = x / self.length_m
            if xi < 0.0 or xi > 1.0:
                return 0.0
            m = self.params["m"]
            return (
                0.5
                * self.breadth_m
                * self.waterline_factor(xi)
                * max(z / self.draught_m, 0.0) ** (1.0 / m)
            )
        raise ValueError(f"unknown hull family {self.family!r}")

    # -- parametric surface in (x, s) ---------------------------------------
    def z_of_s(self, s: float) -> float:
        """Depth map ``s in [0, 1] -> z in [0, T]`` (keeps integrands smooth)."""
        if self.family == "revolution":
            return 0.5 * self.breadth_m * (1.0 - math.cos(math.pi * s))
        return self.draught_m * s ** self.params["m"]

    def dz_ds(self, s: float) -> float:
        if self.family == "revolution":
            return 0.5 * self.breadth_m * math.pi * math.sin(math.pi * s)
        m = self.params["m"]
        return self.draught_m * m * max(s, 0.0) ** (m - 1.0)

    def surface_yz(self, x: float, s: float) -> tuple[float, float]:
        """Starboard surface point ``(y, z)`` at streamwise ``x``, section ``s``.

        This is the map the meshes are built on, so mesh nodes and the CAD
        quadrature describe exactly the same surface.
        """
        if self.family == "revolution":
            r = self.radius(x)
            return r * math.sin(math.pi * s), 0.5 * self.breadth_m - r * math.cos(math.pi * s)
        return (
            0.5 * self.breadth_m * self.waterline_factor(x / self.length_m) * s,
            self.draught_m * s ** self.params["m"],
        )

    def surface_terms(self, xi: float, s: float) -> tuple[float, float, float, float, float]:
        """``(b, b_x, b_s, dz/ds, dz/dx)`` of the parametric hull surface."""
        x = xi * self.length_m
        if self.family == "revolution":
            r = self.radius(x)
            dr = self.radius_slope(x)
            theta = math.pi * s
            b = r * math.sin(theta)
            b_s = r * math.pi * math.cos(theta)
            b_x = dr * math.sin(theta)
            dz_s = r * math.pi * math.sin(theta)
            dz_x = -dr * math.cos(theta)
            return b, b_x, b_s, dz_s, dz_x
        m = self.params["m"]
        half_b = 0.5 * self.breadth_m
        w = self.waterline_factor(xi)
        b = half_b * w * s
        b_x = half_b * self.waterline_factor_slope(xi) / self.length_m * s
        b_s = half_b * w
        return b, b_x, b_s, self.draught_m * m * max(s, 0.0) ** (m - 1.0), 0.0

    def _area_element(self, xi: float, s: float) -> float:
        _b, b_x, b_s, dz_s, dz_x = self.surface_terms(xi, s)
        # |r_x x r_s| for the parametrisation (x, b(x, s), z(x, s))
        return math.sqrt((b_x * dz_s - dz_x * b_s) ** 2 + dz_s * dz_s + b_s * b_s)

    # -- CAD-side integral quantities ---------------------------------------
    def _section_area(self, xi: float, n_s: int) -> float:
        return 2.0 * _simpson(
            lambda s: self.surface_terms(xi, s)[0] * self.surface_terms(xi, s)[3],
            0.0,
            1.0,
            n_s,
        )

    def cad_volume_slice(self, n_x: int = 240, n_s: int = 160) -> float:
        """Deterministic transverse-slice integration (Simpson in x and s)."""
        return self.length_m * _simpson(lambda xi: self._section_area(xi, n_s), 0.0, 1.0, n_x)

    def cad_end_area(self, xi: float, n_s: int = 160) -> float:
        """Area of the transverse end plane at ``xi`` (transom / bow / base)."""
        return self._section_area(xi, n_s)

    def cad_surface_area(self, n_x: int = 240, n_s: int = 160) -> float:
        """Deterministic parametric-quadrature wetted area (sides + both ends)."""
        side = 2.0 * self.length_m * _simpson(
            lambda xi: _simpson(lambda s: self._area_element(xi, s), 0.0, 1.0, n_s),
            0.0,
            1.0,
            n_x,
        )
        return side + self.cad_end_area(0.0, n_s) + self.cad_end_area(1.0, n_s)

    def waterplane_z(self) -> float:
        """Height of the plane used for the waterplane coefficient."""
        return 0.5 * self.breadth_m if self.family == "revolution" else self.draught_m

    def cad_waterplane_area(self, n_x: int = 240) -> float:
        z = self.waterplane_z()
        return 2.0 * self.length_m * _simpson(
            lambda xi: self.half_breadth(xi * self.length_m, z), 0.0, 1.0, n_x
        )

    # -- stratified Monte-Carlo replication ---------------------------------
    def _half_breadth_rows(self, x, z):
        """Vectorised half-breadth for the Monte-Carlo samplers."""
        np = _np()
        if self.family == "revolution":
            r = np.vectorize(self.radius)(x)
            inner = np.clip(r * r - (z - 0.5 * self.breadth_m) ** 2, 0.0, None)
            return np.sqrt(inner)
        m = self.params["m"]
        w = np.vectorize(self.waterline_factor)(np.clip(x / self.length_m, 0.0, 1.0))
        return 0.5 * self.breadth_m * w * np.clip(z / self.draught_m, 0.0, 1.0) ** (1.0 / m)

    def cad_volume_monte_carlo(self, n: int = 1_000_000, seed: int = 20261106) -> float:
        """Stratified Monte-Carlo volume with ``n`` jittered samples."""
        np = _np()
        rng = np.random.default_rng(seed)
        per_axis = max(4, int(round(n ** (1.0 / 3.0))))
        cells = np.array_split(np.arange(per_axis), 20)
        length, breadth, draught = self.length_m, self.breadth_m, self.draught_m
        inside = 0
        total = 0
        for ix in cells:
            for iy in cells:
                for iz in cells:
                    gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
                    u = ((gx + rng.random(gx.shape)) / per_axis).ravel()
                    v = ((gy + rng.random(gy.shape)) / per_axis).ravel()
                    w = ((gz + rng.random(gz.shape)) / per_axis).ravel()
                    half = self._half_breadth_rows(u * length, w * draught)
                    inside += int(np.count_nonzero(np.abs((v - 0.5) * breadth) <= half))
                    total += gx.size
        return inside / total * length * breadth * draught

    def cad_surface_monte_carlo(self, n: int = 1_000_000, seed: int = 20261106) -> float:
        """Stratified Monte-Carlo replication of the wetted-area integral."""
        np = _np()
        rng = np.random.default_rng(seed)
        per_axis = max(4, int(round(math.sqrt(n))))
        cells = np.array_split(np.arange(per_axis), 60)
        acc = 0.0
        count = 0
        for cx in cells:
            for cs in cells:
                gx, gs = np.meshgrid(cx, cs, indexing="ij")
                xi = ((gx + rng.random(gx.shape)) / per_axis).ravel()
                sv = ((gs + rng.random(gs.shape)) / per_axis).ravel()
                acc += float(
                    np.fromiter(
                        (self._area_element(float(a), float(b)) for a, b in zip(xi, sv)),
                        dtype=float,
                        count=xi.size,
                    ).sum()
                )
                count += xi.size
        side = 2.0 * self.length_m * acc / count
        return side + self._mc_end_area(n // 10, seed + 11, 0.0) + self._mc_end_area(n // 10, seed + 13, 1.0)

    def _mc_end_area(self, n: int, seed: int, xi_end: float) -> float:
        np = _np()
        rng = np.random.default_rng(seed)
        s = rng.random(max(n, 16))
        vals = np.fromiter(
            (
                2.0 * self.surface_terms(xi_end, float(sv))[0] * self.surface_terms(xi_end, float(sv))[3]
                for sv in s
            ),
            dtype=float,
            count=s.size,
        )
        return float(vals.mean())

    def cad_integrals(self, n_x: int = 240, n_s: int = 160) -> dict[str, float]:
        volume = self.cad_volume_slice(n_x, n_s)
        area = self.cad_surface_area(n_x, n_s)
        return {
            "nabla_m3": volume,
            "s_wetted_m2": area,
            "cb": volume / (self.length_m * self.breadth_m * self.draught_m),
            "cwp": self.cad_waterplane_area(n_x) / (self.length_m * self.breadth_m),
        }


# ---------------------------------------------------------------------------
# Contract hulls (published principal dimensions and coefficients)
# ---------------------------------------------------------------------------


def _solve_section_exponent(base: Mapping[str, float], cb_target: float) -> float:
    probe = HullForm("probe", "wall_sided_power", 1.0, 1.0, 1.0, dict(base))
    cwp = probe.cad_waterplane_area()  # = 2 * integral(w) / (L*B) with unit dims
    t = cb_target / cwp
    if not 0.05 < t < 0.995:
        raise ValueError(f"Cb/Cwp = {t:.4f} outside the realisable band")
    return t / (1.0 - t)


def kcs_hull() -> HullForm:
    """KRISO Container Ship: Lpp 230 m, B 32.2 m, T 10.8 m, Cb 0.651."""
    base = {
        "mid": 0.48,
        "transom": 0.10,
        "fore_a": 4.0,
        "fore_b": 1.10,
        "aft_a": 3.0,
        "aft_b": 1.00,
        "m": 1.0,
    }
    m = _solve_section_exponent(base, 0.651)
    return HullForm("kcs", "wall_sided_power", 230.0, 32.2, 10.8, dict(base, m=m), cb_contract=0.651)


def kvlcc2_hull() -> HullForm:
    """KVLCC2 tanker: Lpp 320 m, B 58 m, T 20.8 m, Cb 0.8098."""
    base = {
        "mid": 0.50,
        "transom": 0.30,
        "fore_a": 6.0,
        "fore_b": 1.20,
        "aft_a": 5.0,
        "aft_b": 1.00,
        "m": 1.0,
    }
    m = _solve_section_exponent(base, 0.8098)
    return HullForm("kvlcc2", "wall_sided_power", 320.0, 58.0, 20.8, dict(base, m=m), cb_contract=0.8098)


SUBOFF_LENGTH_M = 4.356
SUBOFF_DIAMETER_M = 0.508
SUBOFF_FORE_LENGTH_M = 1.016
SUBOFF_PARALLEL_LENGTH_M = 2.229
SUBOFF_NABLA_REF_M3 = 0.699211
SUBOFF_S_REF_M2 = 5.998


def _suboff_form(fore_b: float, aft_b: float, aft_radius: float, lf: float, lp: float) -> HullForm:
    la = SUBOFF_LENGTH_M - lf - lp
    params = {
        "lf": lf,
        "lp": lp,
        "la": la,
        "aft_radius": aft_radius,
        "fore_b": fore_b,
        "aft_b": aft_b,
        "m": 2.0,
    }
    return HullForm("probe", "revolution", SUBOFF_LENGTH_M, SUBOFF_DIAMETER_M, SUBOFF_DIAMETER_M, params)


def _suboff_volume(fore_b: float, aft_b: float, aft_radius: float, lf: float, lp: float) -> float:
    return _suboff_form(fore_b, aft_b, aft_radius, lf, lp).cad_volume_slice(200, 120)


def _solve_fairing(
    aft_radius: float = 0.08,
    lf: float = SUBOFF_FORE_LENGTH_M,
    lp: float = SUBOFF_PARALLEL_LENGTH_M,
    target_volume: float = SUBOFF_NABLA_REF_M3,
    tol: float = 1e-13,
    iters: int = 80,
) -> tuple[float, HullForm]:
    """Solve the common fore/aft fairing exponent so the displacement closes.

    The radius law is ``r = R (1 - u^2)^b`` on both ends (published SUBOFF fore /
    parallel-middle / aft lengths), so one exponent controls the displacement;
    it is solved by bisection to machine precision.  The wetted area is then a
    *prediction* of the closed geometry, not an input.
    """
    lo, hi = 0.05, 6.0
    f_lo = _suboff_volume(lo, lo, aft_radius, lf, lp) - target_volume
    f_hi = _suboff_volume(hi, hi, aft_radius, lf, lp) - target_volume
    if f_lo * f_hi > 0.0:
        raise RuntimeError("SUBOFF displacement cannot be closed by the fairing exponent")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = _suboff_volume(mid, mid, aft_radius, lf, lp) - target_volume
        if abs(f_mid) < tol or (hi - lo) < 1e-12:
            lo = hi = mid
            break
        if f_mid * f_lo > 0.0:
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    b = 0.5 * (lo + hi)
    return b, _suboff_form(b, b, aft_radius, lf, lp)


def suboff_hull() -> HullForm:
    """DARPA SUBOFF bare hull, displacement-closed (L 4.356 m, D 0.508 m).

    The fairing exponent is solved so that ``nabla = 0.699211 m3`` exactly; the
    wetted area then comes out of the integral and is compared with the
    delivered ``S = 5.998 m2`` (see :func:`suboff_area_envelope` and the report:
    the two delivered references cannot both be met by a faithful D = 0.508
    body, so the deviation is reported rather than tuned away).
    """
    b, form = _solve_fairing()
    return HullForm(
        "suboff",
        "revolution",
        form.length_m,
        form.breadth_m,
        form.draught_m,
        form.params,
        nabla_contract_m3=SUBOFF_NABLA_REF_M3,
        s_contract_m2=SUBOFF_S_REF_M2,
    )


def suboff_area_envelope() -> list[dict[str, float]]:
    """Wetted area of displacement-closed fairings over the shape family.

    Every row closes ``nabla`` exactly, so the spread in ``S`` is the genuine
    geometric freedom left by the contract; the minimum is the best the family
    can do against the delivered ``S = 5.998 m2``.
    """
    rows: list[dict[str, float]] = []
    for aft_radius in (0.02, 0.08, 0.15, 0.25):
        for lf_shift in (-0.10, 0.0, 0.10):
            lf = SUBOFF_FORE_LENGTH_M + lf_shift
            lp = SUBOFF_PARALLEL_LENGTH_M
            b, form = _solve_fairing(aft_radius=aft_radius, lf=lf, lp=lp)
            rows.append(
                {
                    "aft_radius_ratio": aft_radius,
                    "fore_length_m": lf,
                    "fairing_exponent": b,
                    "nabla_m3": form.cad_volume_slice(200, 120),
                    "s_wetted_m2": form.cad_surface_area(200, 120),
                }
            )
    return rows


def contract_hulls() -> dict[str, HullForm]:
    """The three contract hulls, keyed by ``geometry_id``."""
    return {"kcs": kcs_hull(), "kvlcc2": kvlcc2_hull(), "suboff": suboff_hull()}
