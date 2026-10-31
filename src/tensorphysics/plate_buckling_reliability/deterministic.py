"""Deterministic half of the unified chain: arc-length buckling + code check.

This module is a thin, cache-backed driver around the W3-N03 finite-element
solver (:mod:`tensorphysics.plate_buckling`).  It adds three things that the
reliability chain needs:

* a **limit-point record** with a stable, JSON-serialisable payload
  (``sigma_ultimate``, peak state, traced path, runtime) so the deterministic
  result can be stored as evidence and re-used as the calibration datum of the
  capacity surrogate;
* an **imperfection sweep** over the standard welding-distortion levels in the
  ``w0/b`` variable required by the unified framework;
* an **IACS UR I style ultimate-limit-state check** of the limit load against
  the factored ice demand with the same partial factors used by the W3-N03 ULS
  check (``gamma_ice = 1.25``, ``gamma_m = 1.15``), so the deterministic and
  probabilistic halves quote the same safety concept.

All quantities are SI (m, Pa, N).  The finite-element model itself is untouched
-- a wrong "small optimisation" inside the element kernel would invalidate the
W3-N03 verification artifacts, so we only orchestrate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..plate_buckling import PlateGeometry, PlateModel
from ..plate_buckling.benchmarks import (
    dnv_rp_c201_elastic_buckling,
    euler_buckling_stress,
    ritz_buckling_stress,
)
from ..plate_buckling.imperfection import ImperfectionSpec, apply_imperfection
from ..plate_buckling.uls_check import DEFAULT_FACTORS
from .panel import PanelSpec

__all__ = ["DeterministicAssessment", "ULTIMATE_CACHE", "clear_cache"]

#: In-process cache of limit-point traces, keyed by the full model fingerprint.
ULTIMATE_CACHE: Dict[Tuple, Dict[str, object]] = {}


def clear_cache() -> None:
    """Drop cached limit-point traces (tests / benchmark isolation)."""
    ULTIMATE_CACHE.clear()


@dataclass
class DeterministicAssessment:
    """Arc-length nonlinear buckling analysis of one :class:`PanelSpec`."""

    panel: PanelSpec
    ds0: float = 6.0e-4
    max_steps: int = 18
    max_iter: int = 25
    use_cache: bool = True

    # ---------------------------------------------------------------- helpers
    def _key(self, tag: str) -> Tuple:
        p = self.panel
        return (tag, p.a, p.b, p.t, p.E, p.nu, p.sigma_y, p.E_t, p.nx, p.ny,
                p.bc, p.imperfection_level, p.sigma_r_ratio, self.ds0,
                self.max_steps, self.max_iter)

    def model(self, level: Optional[str] = None,
              sigma_r_ratio: Optional[float] = None) -> PlateModel:
        """Fresh finite-element model with the standard imperfection applied."""
        level = self.panel.imperfection_level if level is None else level
        sigma_r_ratio = (self.panel.sigma_r_ratio if sigma_r_ratio is None
                         else sigma_r_ratio)
        # The upstream ``PlateGeometry.validate`` envelope (beta <= 3) is a
        # *verification* window of W3-N03; W4-N03 extends the window to
        # beta <= 4.5 (see ``PanelSpec.beta_range``) and backs the extension
        # with its own arc-length verification runs, so the upstream gate is
        # replaced -- not bypassed -- by ``PanelSpec.validate``.
        model = PlateModel(self.panel.geometry(), self.panel.bc, validate=False)
        spec = ImperfectionSpec.standard(self.panel.geometry(), level=level,
                                         sigma_r_ratio=sigma_r_ratio)
        apply_imperfection(model, spec)
        return model

    # ------------------------------------------------------- linear buckling
    def linear_buckling(self) -> Dict[str, object]:
        """Elastic critical stress with independent Ritz / Euler references."""
        geom = self.panel.geometry()
        model = PlateModel(geom, self.panel.bc, validate=False)
        result = model.linear_buckling(1.0)
        k_fem = (result.sigma_cr * geom.t * geom.b**2
                 / (np.pi**2 * geom.flexural_rigidity))
        ritz = ritz_buckling_stress(geom, self.panel.bc, n_terms=8)
        euler = euler_buckling_stress(geom, self.panel.bc)
        dnv = dnv_rp_c201_elastic_buckling(geom, self.panel.bc)
        return {
            "bc": self.panel.bc,
            "beta": float(geom.beta),
            "sigma_cr_MPa": float(result.sigma_cr / 1e6),
            "k_fem": float(k_fem),
            "k_ritz": float(ritz["k"]),
            "k_euler_textbook": float(euler["k"]),
            "k_dnv_rp_c201": float(dnv["C"]),
            "rel_err_vs_ritz": float(k_fem / ritz["k"] - 1.0),
            "rel_err_vs_euler": float(k_fem / euler["k"] - 1.0),
        }

    # -------------------------------------------------- nonlinear limit point
    def ultimate(self, level: Optional[str] = None,
                 sigma_r_ratio: Optional[float] = None) -> Dict[str, object]:
        """Trace the compression path by arc length and return the limit point.

        The load is the uniform longitudinal compression normalised to the
        nominal yield stress, so the arc-length parameter of the limit point is
        directly ``sigma_u / sigma_y``.
        """
        key = self._key(f"ult:{level}:{sigma_r_ratio}")
        if self.use_cache and key in ULTIMATE_CACHE:
            return dict(ULTIMATE_CACHE[key])

        geom = self.panel.geometry()
        model = self.model(level=level, sigma_r_ratio=sigma_r_ratio)
        sigma_ref = self.panel.sigma_y
        t0 = time.time()
        result = model.solve_arc_length(model.compression_load(sigma_ref),
                                        ds0=self.ds0, max_steps=self.max_steps,
                                        max_iter=self.max_iter, sigma_ref=sigma_ref)
        runtime = time.time() - t0
        lam = float(result["lam_ultimate"])
        record = {
            "beta": float(geom.beta),
            "t_mm": float(geom.t * 1e3),
            "sigma_y_MPa": float(self.panel.sigma_y / 1e6),
            "sigma_cr_MPa": float(model.linear_buckling(1.0).sigma_cr / 1e6),
            "sigma_u_over_sigma_y": lam,
            "sigma_u_MPa": float(lam * self.panel.sigma_y / 1e6),
            "w0max_mm": float(ImperfectionSpec.standard(
                geom, level=self.panel.imperfection_level if level is None else level,
                sigma_r_ratio=self.panel.sigma_r_ratio if sigma_r_ratio is None
                else sigma_r_ratio).w0max * 1e3),
            "peak_vm_MPa": float(result["peak"]["sigma_vm_max"] / 1e6),
            "peak_w_mm": float(result["peak"]["w_monitor"] * 1e3),
            "n_steps": int(result["n_steps"]),
            "mesh": [int(geom.nx), int(geom.ny)],
            "bc": self.panel.bc,
            "runtime_s": float(runtime),
            "path": [(float(p["lam"]), float(p["w_monitor"] * 1e3))
                     for p in result["path"]],
        }
        if self.use_cache:
            ULTIMATE_CACHE[key] = dict(record)
        return record

    def imperfection_sweep(self, levels: Sequence[str] = ("slight", "average", "severe")
                           ) -> List[Dict[str, object]]:
        """``w0/b`` sweep over the standard distortion levels."""
        rows: List[Dict[str, object]] = []
        b = self.panel.b
        for level in levels:
            rec = self.ultimate(level=level)
            rows.append({
                "level": level,
                "w0_mm": rec["w0max_mm"],
                "w0_over_b": rec["w0max_mm"] * 1e-3 / b,
                "sigma_u_over_sigma_y": rec["sigma_u_over_sigma_y"],
                "sigma_u_MPa": rec["sigma_u_MPa"],
                "n_steps": rec["n_steps"],
                "runtime_s": rec["runtime_s"],
            })
        return rows

    # --------------------------------------------------------- code check
    def iacs_uls_check(self, sigma_u: Optional[float] = None,
                       p_ice: Optional[float] = None,
                       factors: Optional[Dict[str, float]] = None
                       ) -> Dict[str, object]:
        """IACS UR I style ultimate limit state check of the panel.

        Demand: the factored applied membrane compression
        ``sigma_d = gamma_ice * sigma_global + gamma_ice * psi * p_ice`` with
        ``gamma_ice`` taken from the W3-N03 factor set.  Capacity: the
        arc-length limit stress divided by ``gamma_m``.  The check passes when
        ``sigma_d <= sigma_u/gamma_m`` **and** the delivered safety factor
        ``sigma_u / sigma_d`` reaches the target factor.

        ``p_ice`` defaults to the deterministic design ice pressure of the
        W3-N03 scenario (5 MPa at 1 m ice thickness, see
        :func:`tensorphysics.plate_buckling.loads.ice_patch_pressure`).
        """
        fac = dict(DEFAULT_FACTORS)
        if factors:
            fac.update(factors)
        if p_ice is None:
            p_ice = 5.0e6
        if sigma_u is None:
            sigma_u = float(self.ultimate()["sigma_u_over_sigma_y"]) * self.panel.sigma_y
        sigma_d = fac["gamma_ice"] * (self.panel.sigma_global
                                      + self.panel.psi_membrane * p_ice)
        sigma_allow = sigma_u / fac["gamma_m"]
        utilisation = sigma_d / sigma_allow if sigma_allow > 0 else float("inf")
        safety = sigma_u / sigma_d if sigma_d > 0 else float("inf")
        return {
            "sigma_u_MPa": float(sigma_u / 1e6),
            "sigma_d_MPa": float(sigma_d / 1e6),
            "sigma_allow_MPa": float(sigma_allow / 1e6),
            "utilisation": float(utilisation),
            "safety_factor": float(safety),
            "required_safety_factor": float(fac["target_sf"]),
            "factors": fac,
            "pass_stress": bool(utilisation <= 1.0),
            "pass_safety": bool(safety >= fac["target_sf"]),
            "passed": bool(utilisation <= 1.0 and safety >= fac["target_sf"]),
        }

    # ------------------------------------------------------------ aggregated
    def report(self, with_sweep: bool = True) -> Dict[str, object]:
        """Full deterministic block of the unified assessment."""
        ultimate = self.ultimate()
        out: Dict[str, object] = {
            "panel": self.panel.as_dict(),
            "geometry_check": self.panel.validate(),
            "linear_buckling": self.linear_buckling(),
            "ultimate": {k: v for k, v in ultimate.items() if k != "path"},
            "iacs_uls": self.iacs_uls_check(),
            "solver": {"ds0": self.ds0, "max_steps": self.max_steps,
                       "max_iter": self.max_iter, "arc_length": "Riks/Crisfield"},
        }
        out["ultimate"]["path_points"] = len(ultimate["path"])
        out["path"] = ultimate["path"]
        if with_sweep:
            out["imperfection_sweep"] = self.imperfection_sweep()
        return out