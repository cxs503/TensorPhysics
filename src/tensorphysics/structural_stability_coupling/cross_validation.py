"""Cross-validation: 3D waterline → plate load vs. ultimate capacity.

Physics background
------------------
The structural–stability coupling requires consistency between:

1. **N06 output**: 3D non-symmetric waterline geometry at heel angle φ,
   from the inclining-test inversion.
2. **N05 output**: Plate-panel ultimate strength under combined loading.

The cross-validation closes the loop:

    3D waterline → hydrostatic pressure on wet surface → plate-panel load
    → compare with N05 ultimate capacity → consistency ratio

If the consistency ratio exceeds 3%, the coupling is broken and the
analysis chain must be re-examined.

The pressure distribution on the wet surface is computed from the
hydrostatic pressure at each submerged point::

    p(x, y, z) = ρ · g · (T - z + y · tan(φ))

where z is measured from the baseline, y from the centreline (+ to port),
and φ is the heel angle.

References
----------
- IACS UR I, Unified requirements concerning strength of ships.
- DNV-RP-C201, Design of plated structures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CrossValidationResult",
    "cross_validate",
    "WaterlinePanelLoad",
]


@dataclass
class WaterlinePanelLoad:
    """Load on a plate panel from 3D waterline geometry.

    Parameters
    ----------
    name :
        Panel identifier.
    panel_area :
        Panel area (m²).
    p_mean :
        Mean hydrostatic pressure on panel (Pa).
    p_max :
        Maximum pressure on panel (Pa).
    F_total :
        Total force on panel (N).
    z_centroid :
        Vertical centroid above baseline (m).
    y_centroid :
        Transverse centroid from centreline (m).
    """
    name: str
    panel_area: float
    p_mean: float
    p_max: float
    F_total: float
    z_centroid: float
    y_centroid: float


@dataclass
class CrossValidationResult:
    """Result of cross-validation between waterline loads and plate capacity."""
    phi_deg: float
    panel_loads: List[WaterlinePanelLoad]
    panel_capacities: Dict[str, float]  # name → p_ult (Pa)
    consistency_ratios: Dict[str, float]  # name → load/capacity
    max_ratio: float
    max_ratio_panel: str
    passed: bool
    threshold: float
    summary: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "phi_deg": self.phi_deg,
            "panel_loads": [
                {
                    "name": pl.name,
                    "panel_area_m2": pl.panel_area,
                    "p_mean_Pa": pl.p_mean,
                    "p_max_Pa": pl.p_max,
                    "F_total_N": pl.F_total,
                }
                for pl in self.panel_loads
            ],
            "panel_capacities_Pa": self.panel_capacities,
            "consistency_ratios": self.consistency_ratios,
            "max_ratio": self.max_ratio,
            "max_ratio_panel": self.max_ratio_panel,
            "passed": self.passed,
            "threshold": self.threshold,
            "summary": self.summary,
        }


def _compute_panel_pressure(
    panel_name: str,
    panel_area: float,
    z_centroid: float,
    y_centroid: float,
    T: float,
    phi_rad: float,
    rho_water: float = 1025.0,
    g: float = 9.81,
) -> Tuple[float, float, float]:
    """Compute hydrostatic pressure on a panel at heel φ.

    Returns (p_mean, p_max, F_total).
    """
    # Pressure at centroid: p = ρg·d where d is depth below waterline
    # At heel φ, the waterline shifts: d = T - z + y·tan(φ)
    depth = T - z_centroid + y_centroid * math.tan(phi_rad)
    depth = max(0.0, depth)

    p_mean = rho_water * g * depth

    # Maximum pressure at panel bottom edge (approximate)
    # Assume panel height h ≈ √(panel_area)
    h_panel = math.sqrt(panel_area)
    depth_max = depth + 0.5 * h_panel
    p_max = rho_water * g * max(0.0, depth_max)

    # Total force
    F_total = p_mean * panel_area

    return p_mean, p_max, F_total


def cross_validate(
    panel_geometries: Sequence[Dict],
    phi_deg: float,
    T: float,
    panel_capacities_Pa: Dict[str, float],
    threshold: float = 0.03,
    rho_water: float = 1025.0,
) -> CrossValidationResult:
    """Cross-validate 3D waterline loads against plate ultimate capacity.

    Parameters
    ----------
    panel_geometries :
        List of dicts with keys: name, panel_area, z_centroid, y_centroid.
    phi_deg :
        Heel angle (degrees).
    T :
        Draught (m).
    panel_capacities_Pa :
        Dict mapping panel name → ultimate pressure capacity (Pa).
    threshold :
        Maximum allowed consistency ratio (default 3%).
    rho_water :
        Water density (kg/m³).

    Returns
    -------
    CrossValidationResult
    """
    phi_rad = math.radians(phi_deg)

    panel_loads = []
    consistency_ratios = {}

    for pg in panel_geometries:
        name = pg["name"]
        area = pg["panel_area"]
        z_c = pg["z_centroid"]
        y_c = pg["y_centroid"]

        p_mean, p_max, F_total = _compute_panel_pressure(
            name, area, z_c, y_c, T, phi_rad, rho_water
        )

        panel_loads.append(WaterlinePanelLoad(
            name=name,
            panel_area=area,
            p_mean=p_mean,
            p_max=p_max,
            F_total=F_total,
            z_centroid=z_c,
            y_centroid=y_c,
        ))

        # Consistency ratio: load / capacity
        if name in panel_capacities_Pa and panel_capacities_Pa[name] > 0:
            ratio = p_max / panel_capacities_Pa[name]
            consistency_ratios[name] = ratio
        else:
            consistency_ratios[name] = float('inf')

    # Find maximum ratio
    max_ratio = max(consistency_ratios.values()) if consistency_ratios else 0.0
    max_panel = max(consistency_ratios, key=consistency_ratios.get) if consistency_ratios else ""

    # Pass/fail: all ratios must be below threshold
    passed = all(r <= threshold for r in consistency_ratios.values() if r != float('inf'))

    summary = {
        "n_panels": len(panel_geometries),
        "phi_deg": phi_deg,
        "T_m": T,
        "max_consistency_ratio": max_ratio,
        "threshold": threshold,
        "mean_ratio": float(np.mean([r for r in consistency_ratios.values() if r != float('inf')])) if consistency_ratios else 0.0,
    }

    return CrossValidationResult(
        phi_deg=phi_deg,
        panel_loads=panel_loads,
        panel_capacities=panel_capacities_Pa,
        consistency_ratios=consistency_ratios,
        max_ratio=max_ratio,
        max_ratio_panel=max_panel,
        passed=passed,
        threshold=threshold,
        summary=summary,
    )
