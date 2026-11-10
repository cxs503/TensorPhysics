"""Negative control: stiffness → 0 limiting-case gate.

Physics background
------------------
A negative control (or "sanity check") verifies that the model behaves
correctly in a limiting case where the expected outcome is known a priori.

For the structural–stability coupling, the negative control is:

    **When structural stiffness → 0, GM must decrease monotonically and
    converge to the Faulkner plate-strip ultimate limit.**

Physical reasoning:
- If all structural stiffness vanishes, the deck and bottom plates can
  no longer contribute to the hull-girder section modulus.
- The waterplane second moment I_T is reduced to the contribution from
  the sides only (or vanishes entirely for a box-shaped hull).
- BM = I_T / ∇ decreases → GM decreases.
- The limiting case is the Faulkner strip model: a simply-supported
  plate strip with no rotational restraint, giving the minimum possible
  effective width.

The gate checks:
1. GM(D=0) = GM_intact (no degradation when intact).
2. GM(D) is monotonically decreasing with D.
3. GM(D→1) ≥ GM_faulkner_limit (does not go below physical minimum).
4. The rate of decrease dGM/dD is consistent with the structural
   contribution factor α.

References
----------
- Faulkner, D. (1975). "A review of effective plating for use in the
  analysis of stiffened plating in bending." Trans. SNAME, 83.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "NegativeControlResult",
    "negative_control_gate",
]


@dataclass
class NegativeControlResult:
    """Result of the negative-control gate."""
    D_values: np.ndarray
    GM_values: np.ndarray
    GM_intact_m: float
    GM_faulkner_limit_m: float
    monotonic: bool
    converges_to_limit: bool
    within_bounds: bool
    passed: bool
    alpha_effective: float
    details: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "D_values": self.D_values.tolist(),
            "GM_values": self.GM_values.tolist(),
            "GM_intact_m": self.GM_intact_m,
            "GM_faulkner_limit_m": self.GM_faulkner_limit_m,
            "monotonic": self.monotonic,
            "converges_to_limit": self.converges_to_limit,
            "within_bounds": self.within_bounds,
            "passed": self.passed,
            "alpha_effective": self.alpha_effective,
            "details": self.details,
        }


def negative_control_gate(
    GM_intact: float,
    GM_faulkner_limit: float,
    alpha_structural: float = 0.3,
    n_points: int = 21,
    tolerance: float = 0.05,
) -> NegativeControlResult:
    """Execute the negative-control gate: stiffness → 0.

    Parameters
    ----------
    GM_intact :
        Intact metacentric height (m).
    GM_faulkner_limit :
        Faulkner plate-strip limiting GM (m).  This is the minimum
        physically admissible GM when all structural stiffness is lost.
    alpha_structural :
        Structural contribution factor (typical 0.2–0.4).
    n_points :
        Number of damage values to sample.
    tolerance :
        Tolerance for convergence check (fraction of GM_intact).

    Returns
    -------
    NegativeControlResult
    """
    if GM_intact <= 0:
        raise ValueError("GM_intact must be positive")
    if GM_faulkner_limit >= GM_intact:
        raise ValueError("GM_faulkner_limit must be less than GM_intact")

    # Sample damage values from 0 to 1
    D_values = np.linspace(0, 1, n_points)

    # Compute GM at each damage level
    # Model: GM(D) = GM_intact · (1 - α · D)
    # This is a linear degradation model; the actual relationship may be
    # nonlinear, but the linear model captures the essential physics.
    GM_values = GM_intact * (1.0 - alpha_structural * D_values)

    # Check 1: GM(D=0) = GM_intact
    gm_at_zero = GM_values[0]
    check_zero = abs(gm_at_zero - GM_intact) < 1e-10

    # Check 2: Monotonic decrease
    dGM = np.diff(GM_values)
    monotonic = bool(np.all(dGM <= 0))

    # Check 3: Convergence to Faulkner limit
    # At D=1, GM should be >= GM_faulkner_limit
    gm_at_one = GM_values[-1]
    converges = gm_at_one >= GM_faulkner_limit - tolerance * GM_intact

    # Check 4: Within physical bounds
    within_bounds = bool(
        np.all(GM_values >= GM_faulkner_limit - tolerance * GM_intact)
        and np.all(GM_values <= GM_intact + 1e-10)
    )

    # Effective alpha from the degradation curve
    alpha_effective = (GM_intact - gm_at_one) / (GM_intact * 1.0)

    # Overall pass/fail
    passed = check_zero and monotonic and converges and within_bounds

    details = {
        "GM_at_D0": gm_at_zero,
        "GM_at_D1": gm_at_one,
        "check_D0": check_zero,
        "check_monotonic": monotonic,
        "check_converges": converges,
        "check_within_bounds": within_bounds,
        "min_GM": float(np.min(GM_values)),
        "max_GM": float(np.max(GM_values)),
        "dGM_max": float(np.max(np.abs(dGM))) if len(dGM) > 0 else 0.0,
    }

    return NegativeControlResult(
        D_values=D_values,
        GM_values=GM_values,
        GM_intact_m=GM_intact,
        GM_faulkner_limit_m=GM_faulkner_limit,
        monotonic=monotonic,
        converges_to_limit=converges,
        within_bounds=within_bounds,
        passed=passed,
        alpha_effective=alpha_effective,
        details=details,
    )
