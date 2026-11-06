"""Ice-belt plate-panel finite element model (reuse of the H3 Hermite element).

The element is *not* re-implemented here: this module drives
:class:`tensorphysics.plate_buckling.PlateModel`, i.e. the flat rectangular
shell element (bilinear Q4 membrane + conforming bicubic Hermite bending
rectangle, 6 dofs per node) built in the W3-N03 / H3 plate campaign, in its
linear elastic path (``elastic_stiffness`` + ``solve_linear``).  Reusing the
element is what makes the ice-belt numbers comparable with the buckling /
ultimate-strength chain of the same repository: one element, one stiffness,
one constraint builder.

What this module adds on top of the reused element:

* the ice-belt panel *specification* (frame spacing, stiffener spacing, plate
  thickness, steel grade) with its geometry gate;
* post-processing of the surface bending stress from the curvature field,
  ``sigma = 6 M / t^2`` with ``M = D_b kappa`` -- the reused model exposes the
  membrane resultants but not the bending ones;
* load application through the IACS patch geometry
  (:class:`tensorphysics.plate_buckling.loads.IcePatchLoad`), clipped to the
  panel so that a patch larger than the panel degenerates to the uniform
  pressure case (the PC6 situation: ``0.6 h_I = 0.9 m`` exceeds a 0.35 m
  stiffener spacing).

Load application
----------------
The reused routine ``PlateModel.pressure_load`` distributes a patch by testing
element *centres* against the patch rectangle.  That is adequate for a pressure
that covers whole elements, but for a patch whose edges fall between element
faces it carries two errors: the loaded area is quantised to whole elements,
and an element centre lying exactly on a patch edge is counted by both
neighbouring rows (the load is then up to a factor 2 too large).  Because the
ice-belt check *is* a patch-load check, this module integrates the patch
consistently instead: the bicubic Hermite rows are evaluated on a 4x4 Gauss
rule clipped to the patch/element intersection, which is exact for the
polynomial basis and reproduces ``consistent_pressure_load`` for a
full-element uniform pressure.  The element-centre rule of the reused model is
kept selectable (``load_rule="element_centre"``) so the discretisation error it
introduces can be quantified rather than asserted away, and
``tests/test_hull_icebelt_fem.py`` pins the local Hermite rows against
``PlateModel.kern["Nw"]`` so the two cannot drift apart.

Units: SI (m, Pa, N).  Mesh, boundary conditions and the through-thickness
integration are the reused model's own; nothing here relaxes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..plate_buckling import PlateGeometry, PlateModel
from ..plate_buckling.loads import PATCH_ASPECT_RATIO, PATCH_HEIGHT_FACTOR

__all__ = ["IceBeltPanel", "PanelSolution", "STEEL_GRADES"]

Box = Tuple[float, float, float, float]

#: Nominal mechanical properties of the common ice-grade hull steels.
STEEL_GRADES: Dict[str, Dict[str, float]] = {
    "AH32": {"E": 2.06e11, "nu": 0.30, "sigma_y": 313.0e6},
    "AH36": {"E": 2.06e11, "nu": 0.30, "sigma_y": 355.0e6},
    "DH36": {"E": 2.06e11, "nu": 0.30, "sigma_y": 355.0e6},
    "EH36": {"E": 2.06e11, "nu": 0.30, "sigma_y": 355.0e6},
}



def _hermite_1d(t):
    """Cubic Hermite basis (value/slope at both ends) on ``t`` in [0, 1]."""
    return (1.0 - 3.0 * t ** 2 + 2.0 * t ** 3, t - 2.0 * t ** 2 + t ** 3,
            3.0 * t ** 2 - 2.0 * t ** 3, -t ** 2 + t ** 3)


def _hermite_to_nodal(ex: float, ey: float) -> np.ndarray:
    """Nodal -> Hermite-generalised dof map, same convention as the reused
    element (node dofs ``[w, theta_x, theta_y, w_xy]``, ``theta_x = dw/dy``,
    ``theta_y = -dw/dx``; generalised dofs ordered ``A[i, j]``)."""
    M = np.zeros((16, 16))
    for c, (t, s) in enumerate(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))):
        it, js = int(round(2 * t)), int(round(2 * s))
        base = 4 * c
        M[it * 4 + js, base + 0] = 1.0
        M[(it + 1) * 4 + js, base + 2] = -ex
        M[it * 4 + (js + 1), base + 1] = ey
        M[(it + 1) * 4 + (js + 1), base + 3] = ex * ey
    return M


def hermite_w_row(ex: float, ey: float, t: float, s: float) -> np.ndarray:
    """Nodal row of the transverse interpolation at local ``(t, s)`` in [0, 1].

    ``w(t, s) = hermite_w_row(ex, ey, t, s) @ u_elem_bending``.  Verified
    element-wise against ``PlateModel.kern["Nw"]`` in the unit tests.
    """
    h = _hermite_1d(t)
    hs = _hermite_1d(s)
    row = np.array([h[i] * hs[j] for i in range(4) for j in range(4)])
    return row @ _hermite_to_nodal(ex, ey)


def consistent_patch_load(model, pressure: float, box: Box) -> np.ndarray:
    """Consistent transverse load vector of a uniform pressure on ``box``.

    Integrates ``q N^T`` over the patch/element intersection with a 4x4 Gauss
    rule per element (exact for the bicubic basis on the clipped rectangle).
    """
    if pressure < 0.0:
        raise ValueError(f"pressure must be non-negative, got {pressure!r}")
    ex, ey = model.ex, model.ey
    x0, x1, y0, y1 = box
    gx, gw = np.polynomial.legendre.leggauss(4)
    tp = 0.5 * (gx + 1.0)
    wp = 0.5 * gw
    M = _hermite_to_nodal(ex, ey)
    F = np.zeros(model.ndof)
    if pressure == 0.0:
        return F
    for el, (xc, yc) in enumerate(model.elem_xy):
        ex0, ey0 = xc - 0.5 * ex, yc - 0.5 * ey
        ax0, ax1 = max(x0, ex0), min(x1, ex0 + ex)
        ay0, ay1 = max(y0, ey0), min(y1, ey0 + ey)
        if ax1 <= ax0 or ay1 <= ay0:
            continue
        ta, tb = (ax0 - ex0) / ex, (ax1 - ex0) / ex
        sa, sb = (ay0 - ey0) / ey, (ay1 - ey0) / ey
        tt = ta + (tb - ta) * tp
        ss = sa + (sb - sa) * tp
        acc = np.zeros(16)
        for i, ti in enumerate(tt):
            for j, sj in enumerate(ss):
                h = _hermite_1d(ti)
                hs = _hermite_1d(sj)
                row = np.array([h[a] * hs[b] for a in range(4) for b in range(4)])
                acc += wp[i] * wp[j] * (row @ M)
        # dA = ex * ey dt ds over the unit element; the Gauss weights on [0,1]
        # already sum to 1, so the remaining factor is the element area.
        F[model.edof_bend[el]] += pressure * ex * ey * (tb - ta) * (sb - sa) * acc
    return F


@dataclass(frozen=True)
class PanelSolution:
    """Linear elastic response of one ice-belt panel to a lateral pressure."""

    pressure: float
    patch_box: Optional[Box]
    w_max: float
    w_centre: float
    sigma_x_max: float
    sigma_y_max: float
    sigma_vm_max: float
    sigma_centre: float
    M_centre_x: float
    M_centre_y: float
    loaded_area_m2: float
    patch_area_m2: float
    n_dof: int
    n_elem: int
    u: np.ndarray

    def as_dict(self) -> Dict[str, object]:
        return {
            "pressure_Pa": float(self.pressure),
            "pressure_MPa": float(self.pressure) / 1e6,
            "patch_box_m": None if self.patch_box is None else
            [float(v) for v in self.patch_box],
            "w_max_m": float(self.w_max),
            "w_max_mm": float(self.w_max) * 1e3,
            "w_centre_mm": float(self.w_centre) * 1e3,
            "sigma_x_max_MPa": float(self.sigma_x_max) / 1e6,
            "sigma_y_max_MPa": float(self.sigma_y_max) / 1e6,
            "sigma_vm_max_MPa": float(self.sigma_vm_max) / 1e6,
            "sigma_centre_MPa": float(self.sigma_centre) / 1e6,
            "M_centre_x_N": float(self.M_centre_x),
            "M_centre_y_N": float(self.M_centre_y),
            "loaded_area_m2": float(self.loaded_area_m2),
            "patch_area_m2": float(self.patch_area_m2),
            "load_area_ratio": float(self.loaded_area_m2 / self.patch_area_m2),
            "n_dof": int(self.n_dof),
            "n_elem": int(self.n_elem),
        }


@dataclass(frozen=True)
class IceBeltPanel:
    """One ice-belt plate panel between longitudinals and web frames.

    Parameters
    ----------
    a, b, t:
        Panel length along the ship (x, web-frame spacing), breadth between
        longitudinal stiffeners (y) and plate thickness, in metres.
    grade:
        Steel grade key of :data:`STEEL_GRADES`.
    nx, ny:
        Mesh of the reused Hermite element.
    bc:
        Boundary condition code of the reused element (``SSSS`` is the
        configuration that the closed-form Navier reference applies to; the
        ice belt panel between web frames is normally taken as ``SSSS`` for
        the plating check).
    gate:
        ``"strict"`` applies :meth:`PlateGeometry.validate` (the W3-N03
        plate-slenderness envelope used by the buckling campaign);
        ``"relaxed"`` records the same numbers without raising, for stocky
        ice-belt plating whose ``beta`` lies below the buckling envelope.
    """

    a: float
    b: float
    t: float
    grade: str = "AH36"
    nx: int = 24
    ny: int = 6
    bc: str = "SSSS"
    gate: str = "relaxed"

    def __post_init__(self) -> None:
        for name, value in (("a", self.a), ("b", self.b), ("t", self.t)):
            v = float(value)
            if not np.isfinite(v) or v <= 0.0:
                raise ValueError(f"panel {name} must be positive and finite, "
                                 f"got {value!r}")
        if self.grade not in STEEL_GRADES:
            raise ValueError(f"unknown steel grade {self.grade!r}; "
                             f"expected one of {sorted(STEEL_GRADES)}")
        if self.gate not in ("strict", "relaxed"):
            raise ValueError(f"gate must be 'strict' or 'relaxed', got "
                             f"{self.gate!r}")
        if self.nx < 1 or self.ny < 1:
            raise ValueError("mesh must have at least one element per direction")

    # ------------------------------------------------------------- geometry
    @property
    def material(self) -> Dict[str, float]:
        return STEEL_GRADES[self.grade]

    def geometry(self) -> PlateGeometry:
        """Reused plate geometry object (one source of truth for the mesh)."""
        m = self.material
        return PlateGeometry(a=self.a, b=self.b, t=self.t, E=m["E"],
                             nu=m["nu"], sigma_y=m["sigma_y"],
                             nx=self.nx, ny=self.ny)

    def checks(self) -> Dict[str, object]:
        """Geometry gate numbers, always reported (pass or fail)."""
        g = self.geometry()
        out: Dict[str, object] = {
            "a_m": float(self.a), "b_m": float(self.b), "t_m": float(self.t),
            "t_mm": float(self.t) * 1e3,
            "a_over_b": float(g.aspect),
            "beta": float(g.beta),
            "element_aspect": float(g.element_aspect),
            "span_over_thickness": float(self.b / self.t),
            "grade": self.grade,
            "sigma_y_MPa": float(g.sigma_y) / 1e6,
            "gate": self.gate,
        }
        if self.gate == "strict":
            out["gate_numbers"] = g.validate()
            out["gate_pass"] = True
        else:
            # The beta lower bound (1.0) of the buckling campaign is not a
            # validity limit for a bending check: stocky ice-belt plating has
            # beta < 1 and is *less* prone to elastic buckling.  We report the
            # excursion explicitly instead of hiding it behind validate().
            out["gate_pass"] = bool(g.aspect <= 8.0 and g.element_aspect <= 3.0)
            out["beta_inside_buckling_envelope"] = bool(1.0 <= g.beta <= 3.0)
        return out

    # ----------------------------------------------------------------- solve
    def solve(self, pressure: float,
              patch_box: Optional[Box] = None,
              model: Optional[PlateModel] = None,
              load_rule: str = "consistent") -> PanelSolution:
        """Linear elastic FE solution under a lateral pressure.

        Parameters
        ----------
        pressure:
            Patch pressure [Pa] (non-negative).
        patch_box:
            ``(x0, x1, y0, y1)`` loaded rectangle [m]; ``None`` loads the whole
            panel uniformly.
        model:
            Optional pre-built :class:`PlateModel` (reused for sweeps, the
            elastic stiffness and the mesh are then factorised once).
        load_rule:
            ``"consistent"`` (default) integrates the patch exactly onto the
            element dofs; ``"element_centre"`` reproduces the reused
            ``PlateModel.pressure_load`` rule so that its discretisation error
            can be quantified.

        Raises
        ------
        ValueError
            For a negative pressure or a degenerate patch box.
        """
        p = float(pressure)
        if not np.isfinite(p) or p < 0.0:
            raise ValueError(f"pressure must be finite and non-negative, "
                             f"got {pressure!r}")
        boxes: Optional[Sequence[Box]] = None
        if patch_box is not None:
            x0, x1, y0, y1 = (float(v) for v in patch_box)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"degenerate patch box {patch_box!r}")
            if x0 < -1e-12 or y0 < -1e-12 or x1 > self.a + 1e-12 or y1 > self.b + 1e-12:
                raise ValueError(f"patch box {patch_box!r} outside the panel "
                                 f"{self.a} x {self.b} m")
            boxes = [(x0, x1, y0, y1)]

        if load_rule not in ("consistent", "element_centre"):
            raise ValueError(f"load_rule must be 'consistent' or "
                             f"'element_centre', got {load_rule!r}")
        mdl = model if model is not None else PlateModel(
            self.geometry(), bc=self.bc, validate=(self.gate == "strict"))
        if boxes is None:
            F = mdl.pressure_load(p)
        elif load_rule == "consistent":
            F = consistent_patch_load(mdl, p, boxes[0])
        else:
            F = mdl.pressure_load(p, boxes=boxes)
        u = mdl.solve_linear(F)

        g = self.geometry()
        nid = np.arange((g.ny + 1) * (g.nx + 1)).reshape(g.ny + 1, g.nx + 1)
        w_nodal = np.zeros_like(nid, dtype=float)
        w_nodal[:, :] = u[6 * nid + 2]
        centre = (g.ny // 2, g.nx // 2)

        # Bending moments from the curvature field of the reused element.
        kern = mdl.kern
        Db = (g.E * g.t ** 3 / (12.0 * (1.0 - g.nu ** 2))) * np.array(
            [[1.0, g.nu, 0.0], [g.nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - g.nu)]])
        u_b = u[mdl.edof_bend]                              # (E, 16)
        kappa = np.einsum("gci,ei->egc", kern["Bb"], u_b)    # (E, G, 3)
        M = kappa @ Db.T                                    # (E, G, 3)
        sig = 6.0 * M / g.t ** 2
        sx, sy, sxy = sig[..., 0], sig[..., 1], sig[..., 2]
        vm = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3.0 * sxy ** 2)

        el_c = (g.ny // 2) * g.nx + (g.nx // 2)
        M_c = M[el_c].mean(axis=0)
        s_c = float(abs(sx[el_c].mean()))

        # Loaded area actually seen by the element-centre load rule of the
        # reused model: the loaded domain is quantised to whole elements whose
        # centre falls inside the patch, so a patch whose edges are not on
        # element faces carries a first-order load-area error.  Both areas are
        # reported so that this discretisation error is never confused with a
        # physics or element error.
        ex, ey = g.a / g.nx, g.b / g.ny
        if boxes is None:
            loaded = float(g.a * g.b)
            ideal = float(g.a * g.b)
        elif load_rule == "consistent":
            loaded = float((boxes[0][1] - boxes[0][0]) * (boxes[0][3] - boxes[0][2]))
            ideal = loaded
        else:
            x0, x1, y0, y1 = boxes[0]
            inside = sum(1 for (xc, yc) in mdl.elem_xy
                         if x0 <= xc <= x1 and y0 <= yc <= y1)
            loaded = float(inside * ex * ey)
            ideal = float((x1 - x0) * (y1 - y0))

        return PanelSolution(
            pressure=p, patch_box=patch_box,
            w_max=float(w_nodal.max()),
            w_centre=float(w_nodal[centre]),
            sigma_x_max=float(np.abs(sx).max()),
            sigma_y_max=float(np.abs(sy).max()),
            sigma_vm_max=float(vm.max()),
            sigma_centre=float(s_c),
            M_centre_x=float(M_c[0]),
            M_centre_y=float(M_c[1]),
            loaded_area_m2=loaded, patch_area_m2=ideal,
            n_dof=int(mdl.ndof), n_elem=int(g.n_elem), u=u)

    # ------------------------------------------------------------ load patch
    def patch_box(self, h_ice: float) -> Box:
        """IACS patch ``(x0, x1, y0, y1)`` of design ice thickness ``h_ice``.

        Uses the reused :class:`tensorphysics.plate_buckling.loads.IcePatchLoad`
        geometry (height ``0.6 h_I``, aspect 4:1) centred on the panel and
        clipped to it.  Raises ``ValueError`` for a non-positive thickness.
        """
        h = float(h_ice)
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError(f"design ice thickness must be positive, got "
                             f"{h_ice!r}")
        geom = self.geometry()
        height = PATCH_HEIGHT_FACTOR * h
        width = PATCH_ASPECT_RATIO * height
        hw = 0.5 * min(width, geom.a)
        hh = 0.5 * min(height, geom.b)
        xc, yc = 0.5 * geom.a, 0.5 * geom.b
        return (xc - hw, xc + hw, yc - hh, yc + hh)

    def utilisation(self, solution: PanelSolution) -> Dict[str, float]:
        """Yield utilisation of the linear elastic surface stress."""
        sy = self.material["sigma_y"]
        return {
            "sigma_vm_max_over_sigma_y": float(solution.sigma_vm_max / sy),
            "sigma_x_max_over_sigma_y": float(solution.sigma_x_max / sy),
            "yield_exceeded": bool(solution.sigma_vm_max > sy),
        }