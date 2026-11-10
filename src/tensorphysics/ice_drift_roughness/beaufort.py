"""Beaufort Sea under-ice drag-coefficient anchors, calibration and validation.

Provenance discipline
---------------------
The anchor set below collects *published central values* of the under-ice drag
coefficient referenced to 10 m depth (``C_d10``) for Beaufort Sea ice, together
with the morphological descriptors (ice thickness, mean keel depth, mean keel
spacing) that the parameterisation consumes.  No raw observational archive
(ADCP / Ice-Tethered Profiler time series) ships with this repository, so the
anchors are the compiled ranges reported in the literature; each entry stores the
central value used by the cross-validation *and* the reported range, and the
report states explicitly that criterion C4 is a range-consistency validation
against compiled literature values rather than a re-processing of raw series.

Anchors are split into a ``calibration`` subset (used to fit the single free
parameter of the roughness closure, the wake/form-drag amplification
``lambda_w``) and a disjoint ``validation`` subset (never seen by the fit).

References
----------
* McPhee, M.G. (1980) "An analysis of pack ice drift in summer", in *Sea Ice
  Processes and Models*, Univ. of Washington Press.
* McPhee, M.G. (2008) *Air-Ice-Ocean Interaction: Turbulent Ocean Boundary Layer
  Exchange*, Springer, ch. 2 (compilation of Arctic under-ice C_d10 in
  3.4e-3 ... 5.5e-3, smooth FY ice down to ~2.5e-3).
* Timco, G.W. & Burden, R.P. (1997) Cold Reg. Sci. Technol. 25, 65-77 (ridge
  keel depth / width / spacing statistics).
* Lu, P., Li, Z., Cheng, B. & Lepparanta, M. (2011) J. Geophys. Res. 116,
  C07019 (keel geometry control of the ice-ocean drag coefficient).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .z0_model import (
    KAPPA,
    WAKE_AMPLIFICATION_DEFAULT,
    Z_REF,
    IceUndersideRoughness,
    KeelMorphology,
    ice_underside_roughness,
    log_law_drag_coefficient,
)

CD_TOLERANCE = 0.20  # criterion C4: |error| < 20 % (ice-zone scatter)


@dataclass(frozen=True)
class DragAnchor:
    """One Beaufort Sea under-ice drag-coefficient anchor."""

    anchor_id: str
    site: str
    season: str
    ice_type: str
    h_ice_m: float
    keel_depth_m: float
    keel_spacing_m: float
    cd10_obs: float
    cd10_range: Tuple[float, float]
    split: str          # "calibration" | "validation"
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.split not in ("calibration", "validation"):
            raise ValueError(f"bad split {self.split!r} for anchor {self.anchor_id}")
        lo, hi = self.cd10_range
        if not (lo <= self.cd10_obs <= hi):
            raise ValueError(f"central value outside its own range for {self.anchor_id}")

    def morphology(self, wake_amplification: float = WAKE_AMPLIFICATION_DEFAULT) -> KeelMorphology:
        return KeelMorphology(
            keel_depth_m=self.keel_depth_m,
            keel_spacing_m=self.keel_spacing_m,
            wake_amplification=wake_amplification,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "site": self.site,
            "season": self.season,
            "ice_type": self.ice_type,
            "h_ice_m": self.h_ice_m,
            "keel_depth_m": self.keel_depth_m,
            "keel_spacing_m": self.keel_spacing_m,
            "cd10_obs": self.cd10_obs,
            "cd10_range": list(self.cd10_range),
            "split": self.split,
            "source": self.source,
            "note": self.note,
        }


ANCHORS: Tuple[DragAnchor, ...] = (
    DragAnchor(
        anchor_id="AIDJEX64",
        site="Beaufort Sea, AIDJEX 1975 main camp (Ice Station Alpha area)",
        season="summer 1975",
        ice_type="multiyear, moderately ridged",
        h_ice_m=2.0,
        keel_depth_m=6.0,
        keel_spacing_m=70.0,
        cd10_obs=4.0e-3,
        cd10_range=(3.5e-3, 4.5e-3),
        split="calibration",
        source="McPhee (1980); central value of the reported 3.5-4.5e-3 range",
    ),
    DragAnchor(
        anchor_id="SHEBA98",
        site="Beaufort Sea, SHEBA 1997/98 drifting station",
        season="winter 1997/98",
        ice_type="multiyear, second-year/old ice",
        h_ice_m=1.8,
        keel_depth_m=5.0,
        keel_spacing_m=60.0,
        cd10_obs=3.9e-3,
        cd10_range=(3.4e-3, 4.4e-3),
        split="validation",
        source="McPhee (2008) ch. 2 under-ice C_d10 compilation, Beaufort MYI",
    ),
    DragAnchor(
        anchor_id="FY-NEarshore",
        site="Beaufort Sea nearshore first-year ice (Alaskan coast)",
        season="spring, landfast-to-mobile transition",
        ice_type="first-year, lightly ridged",
        h_ice_m=1.5,
        keel_depth_m=3.0,
        keel_spacing_m=100.0,
        cd10_obs=3.2e-3,
        cd10_range=(2.7e-3, 3.7e-3),
        split="calibration",
        source="Lu et al. (2011) FY-ice parameterisation range; midpoint used",
    ),
    DragAnchor(
        anchor_id="RIDGED-Offshore",
        site="Beaufort Sea offshore pack, heavily ridged winter ice",
        season="winter",
        ice_type="first-year heavily ridged",
        h_ice_m=2.5,
        keel_depth_m=9.0,
        keel_spacing_m=50.0,
        cd10_obs=4.8e-3,
        cd10_range=(4.3e-3, 5.3e-3),
        split="calibration",
        source="McPhee (2008) upper end of the Arctic C_d10 compilation (4.3-5.5e-3)",
    ),
    DragAnchor(
        anchor_id="SMOOTH-Landfast",
        site="Beaufort Sea landfast smooth first-year ice",
        season="winter",
        ice_type="smooth first-year, near-undeformed",
        h_ice_m=1.2,
        keel_depth_m=1.6,
        keel_spacing_m=450.0,
        cd10_obs=2.7e-3,
        cd10_range=(2.4e-3, 3.2e-3),
        split="validation",
        source="McPhee (2008) smooth-FY limit of the under-ice C_d10 compilation",
    ),
)


# ---------------------------------------------------------------------------
# Model response and residuals
# ---------------------------------------------------------------------------
def anchor_roughness(
    anchor: DragAnchor, wake_amplification: float = WAKE_AMPLIFICATION_DEFAULT,
    z_ref_m: float = Z_REF, kappa: float = KAPPA,
) -> IceUndersideRoughness:
    """Ice-underside roughness state implied by the anchor morphology."""
    return ice_underside_roughness(
        anchor.morphology(wake_amplification), anchor.h_ice_m, z_ref_m, kappa
    )


def anchor_cd_model(
    anchor: DragAnchor, wake_amplification: float = WAKE_AMPLIFICATION_DEFAULT,
    z_ref_m: float = Z_REF, kappa: float = KAPPA,
) -> float:
    """Modelled ``C_d10`` (per unit ice area) for the anchor morphology."""
    return anchor_roughness(anchor, wake_amplification, z_ref_m, kappa).drag_coefficient


def _objective(
    anchors: Sequence[DragAnchor], wake_amplification: float,
    z_ref_m: float, kappa: float,
) -> float:
    """Sum of squared log-residuals (symmetric in relative error)."""
    total = 0.0
    for anchor in anchors:
        model = anchor_cd_model(anchor, wake_amplification, z_ref_m, kappa)
        total += (math.log(model) - math.log(anchor.cd10_obs)) ** 2
    return total


def calibrate_wake_amplification(
    anchors: Sequence[DragAnchor] = ANCHORS,
    split: Optional[str] = "calibration",
    lo: float = 0.25,
    hi: float = 8.0,
    n_grid: int = 400,
    z_ref_m: float = Z_REF,
    kappa: float = KAPPA,
) -> Dict[str, Any]:
    """Fit the single free parameter ``lambda_w`` (golden-section refinement).

    Only anchors in ``split`` are used; with ``split=None`` all anchors are used.
    Returns the fitted value, the objective value and the grid used.
    """
    subset = [a for a in anchors if split is None or a.split == split]
    if not subset:
        raise ValueError(f"no anchors for split={split!r}")

    grid = [lo * (hi / lo) ** (i / (n_grid - 1)) for i in range(n_grid)]
    values = [_objective(subset, lam, z_ref_m, kappa) for lam in grid]
    best = int(min(range(n_grid), key=lambda i: values[i]))

    # local golden-section refinement around the best grid point
    a = grid[max(best - 1, 0)]
    b = grid[min(best + 1, n_grid - 1)]
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = _objective(subset, c, z_ref_m, kappa), _objective(subset, d, z_ref_m, kappa)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _objective(subset, c, z_ref_m, kappa)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _objective(subset, d, z_ref_m, kappa)
        if (b - a) < 1e-6:
            break
    lambda_opt = 0.5 * (a + b)
    return {
        "parameter": "wake_amplification",
        "fitted_value": lambda_opt,
        "objective": _objective(subset, lambda_opt, z_ref_m, kappa),
        "n_anchors": len(subset),
        "split": split,
        "grid_bounds": [lo, hi],
        "objective_definition": "sum of squared log(C_d_model / C_d_obs) over the split",
    }


def cross_validate(
    anchors: Sequence[DragAnchor] = ANCHORS,
    wake_amplification: float = WAKE_AMPLIFICATION_DEFAULT,
    z_ref_m: float = Z_REF,
    kappa: float = KAPPA,
    tolerance: float = CD_TOLERANCE,
) -> List[Dict[str, Any]]:
    """Per-anchor model/observation comparison for criterion C4."""
    rows: List[Dict[str, Any]] = []
    for anchor in anchors:
        rough = anchor_roughness(anchor, wake_amplification, z_ref_m, kappa)
        model = rough.drag_coefficient
        rel_err = (model - anchor.cd10_obs) / anchor.cd10_obs
        lo, hi = anchor.cd10_range
        rows.append(
            {
                "anchor_id": anchor.anchor_id,
                "site": anchor.site,
                "ice_type": anchor.ice_type,
                "split": anchor.split,
                "h_ice_m": anchor.h_ice_m,
                "keel_depth_m": anchor.keel_depth_m,
                "keel_spacing_m": anchor.keel_spacing_m,
                "z0_ridge_m": rough.z0_ridge_m,
                "area_fraction": rough.area_fraction,
                "z0_ice_m": rough.z0_m,
                "cd_model": model,
                "cd_obs": anchor.cd10_obs,
                "obs_range": [lo, hi],
                "rel_error": rel_err,
                "abs_rel_error": abs(rel_err),
                "within_tolerance": abs(rel_err) < tolerance,
                "inside_obs_range": lo <= model <= hi,
            }
        )
    return rows


def validation_summary(
    rows: Sequence[Dict[str, Any]], tolerance: float = CD_TOLERANCE
) -> Dict[str, Any]:
    """Aggregate the per-anchor comparison into the criterion-C4 verdict."""
    if not rows:
        raise ValueError("no rows to summarise")

    def _split(name: str) -> List[Dict[str, Any]]:
        return [r for r in rows if r["split"] == name]

    calib = _split("calibration")
    valid = _split("validation")
    max_err = max(r["abs_rel_error"] for r in rows)
    return {
        "tolerance": tolerance,
        "n_anchors": len(rows),
        "max_abs_rel_error": max_err,
        "mean_rel_error": sum(r["rel_error"] for r in rows) / len(rows),
        "rms_rel_error": math.sqrt(sum(r["rel_error"] ** 2 for r in rows) / len(rows)),
        "all_within_tolerance": all(r["within_tolerance"] for r in rows),
        "n_inside_obs_range": sum(1 for r in rows if r["inside_obs_range"]),
        "calibration_max_abs_rel_error": max((r["abs_rel_error"] for r in calib), default=0.0),
        "validation_max_abs_rel_error": max((r["abs_rel_error"] for r in valid), default=0.0),
        "validation_all_within_tolerance": all(r["within_tolerance"] for r in valid),
        "criterion_C4_pass": max_err < tolerance,
    }