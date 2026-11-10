"""ITTC (2017) recommended method comparison for ice breaking force scaling.

The ITTC Ice Committee recommends a specific procedure for extrapolating
model-scale ice resistance to full scale. This module implements that
procedure and compares it with the size effect scaling law.

ITTC Recommended Procedure (simplified)
----------------------------------------
1. Model test at Froude scale (λ = L_full / L_model)
2. Measure model ice resistance R_m
3. Apply size effect scaling: R_full = R_m · λ^n
   where n ∈ [2.5, 3.0] per ITTC recommendations
4. Apply minor correction factors for:
   - Friction differences (typically < 5%)
   - Geometry differences (typically < 5%)

The ITTC-recommended exponent n is typically 2.7–2.8, reflecting the
quasi-brittle nature of ice.

This module ensures that the size effect scaling and ITTC method agree
within 15% when using consistent parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ITTCRecommendedMethod:
    """ITTC (2017) recommended method parameters.

    Attributes
    ----------
    n_ittc : float
        ITTC-recommended effective exponent (typically 2.7–2.8).
    C_friction : float
        Friction correction factor (typically 0.98–1.05).
    C_geometry : float
        Geometry correction factor (typically 0.98–1.02).
    """
    n_ittc: float = 2.75
    C_friction: float = 1.0
    C_geometry: float = 1.0

    @property
    def C_ITTC(self) -> float:
        """Total ITTC correction factor."""
        return self.C_friction * self.C_geometry


def ittc_correction_factor(
    lambda_ratio: float,
    n_ittc: float = 2.75,
    mu_model: float = 0.15,
    mu_full: float = 0.15,
) -> float:
    """Compute the ITTC correction factor for model-to-full extrapolation.

    The ITTC correction accounts for:
    1. The difference between ITTC exponent and pure Froude (n=3.0)
    2. Friction coefficient differences

    C_ITTC = λ^(n_ittc - 3.0) · (μ_full / μ_model)

    When n_ittc = 3.0 and μ_model = μ_full, C_ITTC = 1.0 (pure Froude).
    When n_ittc = 2.75 and λ = 30, C_ITTC ≈ 30^(-0.25) ≈ 0.43.

    Parameters
    ----------
    lambda_ratio : float
        Scale ratio λ = L_full / L_model.
    n_ittc : float
        ITTC-recommended exponent.
    mu_model : float
        Model-scale friction coefficient.
    mu_full : float
        Full-scale friction coefficient.

    Returns
    -------
    float
        ITTC correction factor C_ITTC.
    """
    # Exponent correction: ITTC uses n_ittc instead of 3.0
    exponent_correction = lambda_ratio ** (n_ittc - 3.0)

    # Friction ratio
    friction_ratio = mu_full / mu_model if mu_model != 0 else 1.0

    return exponent_correction * friction_ratio


def compute_ittc_full_scale(
    R_model: float,
    L_model: float,
    L_full: float,
    n_ittc: float = 2.75,
    mu_model: float = 0.15,
    mu_full: float = 0.15,
) -> Dict[str, float]:
    """Compute full-scale ice resistance using ITTC recommended method.

    The ITTC method uses size effect scaling with exponent n_ittc:
    R_full = R_model · λ^n_ittc · C_friction · C_geometry

    Parameters
    ----------
    R_model : float
        Model-scale ice resistance (N).
    L_model : float
        Model-scale ship length (m).
    L_full : float
        Full-scale ship length (m).
    n_ittc : float
        ITTC-recommended exponent (default 2.75).
    mu_model : float
        Model-scale friction coefficient.
    mu_full : float
        Full-scale friction coefficient.

    Returns
    -------
    dict
        Full-scale resistance and breakdown.
    """
    lambda_ratio = L_full / L_model

    # Pure Froude scaling (n=3.0)
    R_froude = R_model * lambda_ratio ** 3

    # ITTC size effect scaling
    R_ittc = R_model * lambda_ratio ** n_ittc

    # ITTC correction factor (relative to Froude)
    C_ITTC = ittc_correction_factor(lambda_ratio, n_ittc, mu_model, mu_full)

    # Verify: R_ittc = R_froude * C_ITTC
    R_via_correction = R_froude * C_ITTC

    return {
        "R_model_N": R_model,
        "R_froude_N": R_froude,
        "R_ittc_N": R_ittc,
        "R_via_correction_N": R_via_correction,
        "C_ITTC": C_ITTC,
        "lambda_ratio": lambda_ratio,
        "n_ittc": n_ittc,
        "deviation_froude_vs_ittc": abs(R_ittc - R_froude) / R_froude if R_froude != 0 else 0.0,
    }


def compare_scaling_methods(
    R_model: float,
    L_model: float,
    L_full: float,
    n_range: Tuple[float, float] = (2.5, 3.0),
    n_steps: int = 20,
) -> Dict[str, list]:
    """Compare different scaling methods over a range of exponents.

    Parameters
    ----------
    R_model : float
        Model-scale resistance (N).
    L_model : float
        Model-scale length (m).
    L_full : float
        Full-scale length (m).
    n_range : tuple
        (n_min, n_max) range to sweep.
    n_steps : int
        Number of steps in the sweep.

    Returns
    -------
    dict
        Mapping 'n', 'R_scaled' to lists.
    """
    lambda_ratio = L_full / L_model
    n_values = [n_range[0] + i * (n_range[1] - n_range[0]) / (n_steps - 1)
                for i in range(n_steps)]

    results = {"n": [], "R_scaled": [], "R_froude": []}
    R_froude = R_model * lambda_ratio ** 3

    for n in n_values:
        R_scaled = R_model * lambda_ratio ** n
        results["n"].append(n)
        results["R_scaled"].append(R_scaled)
        results["R_froude"].append(R_froude)

    return results


def validate_against_ittc(
    R_predicted: float,
    R_ittc: float,
    tolerance: float = 0.15,
) -> Dict[str, float]:
    """Validate a prediction against ITTC recommended value.

    Parameters
    ----------
    R_predicted : float
        Predicted full-scale resistance (N).
    R_ittc : float
        ITTC recommended full-scale resistance (N).
    tolerance : float
        Maximum acceptable relative error (default 15%).

    Returns
    -------
    dict
        Validation results.
    """
    rel_error = abs(R_predicted - R_ittc) / R_ittc if R_ittc != 0 else float("inf")
    passed = rel_error <= tolerance

    return {
        "R_predicted_N": R_predicted,
        "R_ittc_N": R_ittc,
        "relative_error": rel_error,
        "tolerance": tolerance,
        "passed": passed,
    }
