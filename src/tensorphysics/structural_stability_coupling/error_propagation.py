"""Error propagation: plate-thickness uncertainty δ_iso → GZ curve.

Physics background
------------------
The plate-panel FEM analysis (N05) delivers an ultimate-strength estimate
with an associated error bound δ_iso that arises from:

1. Thickness measurement uncertainty (UT gauge ±0.05 mm typical).
2. Material property scatter (σ_y, E from coupon tests).
3. Mesh discretisation error (h-convergence study residual).
4. Imperfection amplitude uncertainty (w₀/t ratio).

This error propagates to the GZ curve through the structural weight and
centre-of-gravity calculation:

    δ_weight = ρ · A_panel · δ_t
    δ_KG = δ_weight · (z_panel - KG) / W_total
    δ_GZ(φ) = -δ_KG · sin(φ)

For the transverse centre of gravity (TCG), asymmetric plate thinning
(e.g. corrosion on one side) produces:

    δ_TCG = Σ(δ_w_i · y_i) / W_total

which directly biases the GZ curve at all angles.

References
----------
- IMO IS Code 2008, Intact Stability Code, Part A, Ch. 3.
- DNV-Stable W21, §3 — inclining test and lightweight survey.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DeltaPropagationResult",
    "propagate_delta_iso",
    "PlatePanelContribution",
]


@dataclass
class PlatePanelContribution:
    """A single plate panel's contribution to weight and CG.

    Parameters
    ----------
    name :
        Panel identifier (e.g. 'bottom_shell_AP_1').
    area :
        Panel area (m²).
    z_centroid :
        Vertical centroid above baseline (m).
    y_centroid :
        Transverse centroid from centreline (m, + to port).
    t_nominal :
        Nominal thickness (m).
    delta_t :
        Thickness uncertainty (m, ±1σ).
    rho :
        Material density (kg/m³).
    """
    name: str
    area: float
    z_centroid: float
    y_centroid: float
    t_nominal: float
    delta_t: float
    rho: float = 7850.0

    @property
    def weight_nominal(self) -> float:
        """Nominal weight contribution (N)."""
        return self.area * self.t_nominal * self.rho * 9.81

    @property
    def delta_weight(self) -> float:
        """Weight uncertainty (N, ±1σ)."""
        return self.area * self.delta_t * self.rho * 9.81


@dataclass
class DeltaPropagationResult:
    """Result of δ_iso propagation to GZ curve."""
    delta_KG_m: float
    delta_TCG_m: float
    delta_GZ_max_m: float
    delta_GZ_array_m: np.ndarray
    gz_corrected_m: np.ndarray
    gz_nominal_m: np.ndarray
    angles_deg: np.ndarray
    relative_change_pct: np.ndarray
    panels: List[str]
    summary: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "delta_KG_m": self.delta_KG_m,
            "delta_TCG_m": self.delta_TCG_m,
            "delta_GZ_max_m": self.delta_GZ_max_m,
            "delta_GZ_array_m": self.delta_GZ_array_m.tolist(),
            "gz_corrected_m": self.gz_corrected_m.tolist(),
            "gz_nominal_m": self.gz_nominal_m.tolist(),
            "angles_deg": self.angles_deg.tolist(),
            "relative_change_pct": self.relative_change_pct.tolist(),
            "panels": self.panels,
            "summary": self.summary,
        }


def propagate_delta_iso(
    panels: Sequence[PlatePanelContribution],
    total_displacement_N: float,
    KG_nominal: float,
    TCG_nominal: float,
    gz_nominal_m: np.ndarray,
    angles_deg: np.ndarray,
    confidence_level: float = 1.0,
) -> DeltaPropagationResult:
    """Propagate plate-thickness uncertainty to GZ curve.

    Parameters
    ----------
    panels :
        List of plate panel contributions.
    total_displacement_N :
        Total ship displacement weight (N).
    KG_nominal :
        Nominal vertical centre of gravity (m).
    TCG_nominal :
        Nominal transverse centre of gravity (m).
    gz_nominal_m :
        Nominal GZ curve values (m) at each angle.
    angles_deg :
        Heel angles (degrees).
    confidence_level :
        Multiplier on 1σ (default 1.0 for ±1σ).

    Returns
    -------
    DeltaPropagationResult
        Contains corrected GZ curve and sensitivity metrics.
    """
    if total_displacement_N <= 0:
        raise ValueError("Displacement must be positive")

    # Accumulate weight and CG uncertainties
    delta_W_total = 0.0
    delta_Mz_total = 0.0  # moment about baseline
    delta_My_total = 0.0  # moment about centreline

    for panel in panels:
        dw = panel.delta_weight * confidence_level
        delta_W_total += dw
        delta_Mz_total += dw * panel.z_centroid
        delta_My_total += dw * abs(panel.y_centroid)

    # CG shifts (first-order approximation)
    delta_KG = delta_Mz_total / total_displacement_N
    delta_TCG = delta_My_total / total_displacement_N

    # GZ correction: δGZ(φ) = -δKG · sin(φ) + δTCG · cos(φ)
    phi_rad = np.radians(angles_deg)
    delta_GZ = -delta_KG * np.sin(phi_rad) + delta_TCG * np.cos(phi_rad)

    gz_corrected = gz_nominal_m + delta_GZ

    # Relative change
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_change = np.where(
            np.abs(gz_nominal_m) > 1e-6,
            100.0 * delta_GZ / gz_nominal_m,
            0.0
        )

    # Maximum absolute change in GZ
    delta_GZ_max = float(np.max(np.abs(delta_GZ)))

    summary = {
        "delta_W_total_N": delta_W_total,
        "delta_W_frac": delta_W_total / total_displacement_N,
        "delta_KG_m": delta_KG,
        "delta_TCG_m": delta_TCG,
        "delta_GZ_max_m": delta_GZ_max,
        "max_relative_change_pct": float(np.max(np.abs(rel_change))),
        "n_panels": len(panels),
        "confidence_level": confidence_level,
    }

    return DeltaPropagationResult(
        delta_KG_m=delta_KG,
        delta_TCG_m=delta_TCG,
        delta_GZ_max_m=delta_GZ_max,
        delta_GZ_array_m=delta_GZ,
        gz_corrected_m=gz_corrected,
        gz_nominal_m=gz_nominal_m.copy(),
        angles_deg=angles_deg.copy(),
        relative_change_pct=rel_change,
        panels=[p.name for p in panels],
        summary=summary,
    )
