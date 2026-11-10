"""Stiffness degradation: elastoplastic yield → BM/GM reduction.

Physics background
------------------
When plate panels yield under extreme loading (ice, wave slam, etc.),
their effective contribution to the hull-girder section modulus decreases.
This affects the ship's stability through two mechanisms:

1. **Direct effect on BM**: The waterplane second moment I_T depends on the
   effective deck/bottom area contributing to the hull girder.  Yielded
   panels have reduced effective area (Faulkner effective width), reducing
   I_T and hence BM = I_T / ∇.

2. **Indirect effect on KG**: Progressive collapse shifts the neutral axis,
   changing the structural weight distribution and hence KG.

The degradation model uses the Faulkner effective-width formula for
compressed plate panels::

    b_eff / b = 2/beta - 1/beta²    for beta > 1
    b_eff / b = 1                    for beta <= 1

where beta = (b/t) √(σ_y / E) is the plate slenderness.

The damage parameter D is defined as::

    D = 1 - (I_eff / I_elastic)

where I_eff accounts for the reduced effective widths of yielded panels.

References
----------
- Faulkner, D. (1975). "A review of effective plating for use in the
  analysis of stiffened plating in bending." Trans. SNAME, 83.
- Paik, J.K. & Thayamballi, A.K. (2003). *Ultimate Limit State Design
  of Steel-Plated Structures*, Wiley.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "StiffnessDegradationResult",
    "compute_degradation",
    "degraded_gm",
    "faulkner_effective_width",
    "PlatePanelStructural",
]


def faulkner_effective_width(beta: float) -> float:
    """Faulkner plate effective-width ratio b_eff/b.

    Parameters
    ----------
    beta :
        Plate slenderness (b/t)·√(σ_y/E).

    Returns
    -------
    float
        Effective width ratio in [0, 1].
    """
    if beta <= 0:
        raise ValueError("Plate slenderness must be positive")
    if beta <= 1.0:
        return 1.0
    return min(1.0, max(0.0, 2.0 / beta - 1.0 / beta ** 2))


@dataclass
class PlatePanelStructural:
    """Structural properties of a plate panel for degradation analysis.

    Parameters
    ----------
    name :
        Panel identifier.
    b :
        Panel breadth (stiffener spacing) (m).
    t :
        Plate thickness (m).
    sigma_y :
        Yield stress (Pa).
    E :
        Young's modulus (Pa).
    z_centroid :
        Vertical centroid above baseline (m).
    A_effective_intact :
        Effective area of panel + stiffener when intact (m²).
    load_ratio :
        Applied compressive load / yield load (0 to 1+).
        Values > 1 indicate post-yield.
    """
    name: str
    b: float
    t: float
    sigma_y: float
    E: float
    z_centroid: float
    A_effective_intact: float
    load_ratio: float = 0.0

    @property
    def beta(self) -> float:
        """Plate slenderness."""
        return (self.b / self.t) * math.sqrt(self.sigma_y / self.E)

    @property
    def phi_faulkner(self) -> float:
        """Faulkner effective-width reduction factor."""
        return faulkner_effective_width(self.beta)


@dataclass
class StiffnessDegradationResult:
    """Result of stiffness degradation analysis."""
    damage_parameter: float
    I_elastic_m4: float
    I_effective_m4: float
    BM_intact_m: float
    BM_degraded_m: float
    GM_intact_m: float
    GM_degraded_m: float
    GM_reduction_pct: float
    panel_details: List[Dict[str, float]]
    faulkner_asymptote_GM_m: float
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "damage_parameter": self.damage_parameter,
            "I_elastic_m4": self.I_elastic_m4,
            "I_effective_m4": self.I_effective_m4,
            "BM_intact_m": self.BM_intact_m,
            "BM_degraded_m": self.BM_degraded_m,
            "GM_intact_m": self.GM_intact_m,
            "GM_degraded_m": self.GM_degraded_m,
            "GM_reduction_pct": self.GM_reduction_pct,
            "panel_details": self.panel_details,
            "faulkner_asymptote_GM_m": self.faulkner_asymptote_GM_m,
            "notes": self.notes,
        }


def compute_degradation(
    panels: Sequence[PlatePanelStructural],
    I_wp_elastic: float,
    nabla: float,
    KG: float,
    KB: float,
    A_deck_total: float,
    A_bottom_total: float,
) -> StiffnessDegradationResult:
    """Compute stiffness degradation from panel yielding.

    Parameters
    ----------
    panels :
        List of structural plate panels.
    I_wp_elastic :
        Intact waterplane second moment (m⁴).
    nabla :
        Displaced volume (m³).
    KG :
        Vertical centre of gravity (m).
    KB :
        Vertical centre of buoyancy (m).
    A_deck_total :
        Total deck area contributing to hull girder (m²).
    A_bottom_total :
        Total bottom area contributing to hull girder (m²).

    Returns
    -------
    StiffnessDegradationResult
    """
    if nabla <= 0:
        raise ValueError("Displaced volume must be positive")

    # Compute effective areas for each panel
    panel_details = []
    I_reduction = 0.0

    for panel in panels:
        phi = panel.phi_faulkner
        # Effective area reduction
        A_reduced = panel.A_effective_intact * phi
        delta_A = panel.A_effective_intact - A_reduced

        # Contribution to I reduction (parallel axis theorem)
        # δI = δA · (z - NA)² where NA is neutral axis
        # Simplified: use z_centroid relative to mid-depth
        z_na = 0.5 * (KB + KG)  # approximate neutral axis
        delta_I = delta_A * (panel.z_centroid - z_na) ** 2
        I_reduction += delta_I

        panel_details.append({
            "name": panel.name,
            "beta": panel.beta,
            "phi_faulkner": phi,
            "load_ratio": panel.load_ratio,
            "A_intact_m2": panel.A_effective_intact,
            "A_effective_m2": A_reduced,
            "delta_A_m2": delta_A,
            "delta_I_m4": delta_I,
        })

    # Effective waterplane second moment
    I_eff = max(0.0, I_wp_elastic - I_reduction)

    # Damage parameter
    D = 1.0 - I_eff / I_wp_elastic if I_wp_elastic > 0 else 0.0
    D = max(0.0, min(1.0, D))

    # BM values
    BM_intact = I_wp_elastic / nabla
    BM_degraded = I_eff / nabla

    # GM values
    GM_intact = KB + BM_intact - KG
    GM_degraded = KB + BM_degraded - KG

    # Faulkner asymptote: fully yielded panels (all phi → 0)
    # This is the negative-control limit
    I_faulkner_limit = 0.0  # extreme case: all deck/bottom ineffective
    BM_faulkner = I_faulkner_limit / nabla
    GM_faulkner = KB + BM_faulkner - KG

    GM_reduction_pct = (
        100.0 * (GM_intact - GM_degraded) / GM_intact
        if GM_intact > 0 else 0.0
    )

    return StiffnessDegradationResult(
        damage_parameter=D,
        I_elastic_m4=I_wp_elastic,
        I_effective_m4=I_eff,
        BM_intact_m=BM_intact,
        BM_degraded_m=BM_degraded,
        GM_intact_m=GM_intact,
        GM_degraded_m=GM_degraded,
        GM_reduction_pct=GM_reduction_pct,
        panel_details=panel_details,
        faulkner_asymptote_GM_m=GM_faulkner,
        notes="Degradation from Faulkner effective-width reduction of yielded panels",
    )


def degraded_gm(
    GM_intact: float,
    damage_parameter: float,
    alpha: float = 0.3,
) -> float:
    """Compute degraded GM from damage parameter.

    Simplified model: GM_degraded = GM_intact · (1 - α · D)

    where α is the structural contribution factor (typically 0.2-0.4
    for conventional merchant ships).

    Parameters
    ----------
    GM_intact :
        Intact metacentric height (m).
    damage_parameter :
        Damage parameter D ∈ [0, 1].
    alpha :
        Structural contribution factor.

    Returns
    -------
    float
        Degraded GM (m).
    """
    return GM_intact * (1.0 - alpha * damage_parameter)
