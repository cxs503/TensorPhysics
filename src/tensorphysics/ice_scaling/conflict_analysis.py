"""Conflict analysis for ice model-test scaling.

The fundamental problem: not all dimensionless groups can be simultaneously
satisfied when scaling from model to full scale. This module identifies
which groups conflict, quantifies the cost of each concession, and
determines feasible parameter windows for given scale ratios.

Conflict structure
------------------
Under Froude scaling:
  - Fr is automatically preserved (by construction).
  - Geometric ratios (B/L, h/L) are preserved (by construction).
  - ρ_i/ρ_w is preserved (material constants).
  - μ is preserved (assumed).
  - n is preserved (assumed).

Groups that CANNOT be preserved:
  - η = σ_i/(ρ_i g L): requires σ_i ∝ 1/λ, but ice strength is a
    material property that doesn't scale with size.
  - Ca = ρ_i U²/E: requires E ∝ λ, but modulus is a material property.
  - σ_i/E: requires both σ_i and E to scale consistently, but neither
    scales with size.
  - K_IC* = K_IC/(E√L): requires K_IC ∝ √L, but toughness is a
    material property.

The "conflict dimensions" are those groups that cannot be preserved.
For each, we quantify the mismatch and determine which to sacrifice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dimensionless import ALL_GROUPS, DimensionlessGroup, compute_all_groups
from .scaling_laws import (
    ScalingLaw, FROUDE_LAW, HALF_LAW, CAUCHY_LAW,
    STRENGTH_LAWS, scale_property,
)


@dataclass
class ConflictDimension:
    """A dimensionless group that cannot be simultaneously satisfied.

    Attributes
    ----------
    group : DimensionlessGroup
        The conflicting dimensionless group.
    model_value : float
        Value at model scale.
    full_value : float
        Value at full scale (with unscaled material properties).
    mismatch_ratio : float
        Ratio model/full (deviation from 1.0 = conflict severity).
    sacrifice_priority : int
        Priority for sacrifice (1 = first to sacrifice, higher = more important).
    cost_description : str
        Physical consequence of sacrificing this group.
    """
    group: DimensionlessGroup
    model_value: float
    full_value: float
    mismatch_ratio: float
    sacrifice_priority: int
    cost_description: str


def analyse_conflicts(
    lambda_ratio: float,
    full_scale_params: Dict[str, float],
    scaling_law: ScalingLaw = FROUDE_LAW,
) -> List[ConflictDimension]:
    """Analyse which dimensionless groups conflict for a given scale ratio.

    Parameters
    ----------
    lambda_ratio : float
        Scale ratio λ = L_full / L_model.
    full_scale_params : dict
        Full-scale physical parameters.
    scaling_law : ScalingLaw
        The strength scaling law to use.

    Returns
    -------
    list of ConflictDimension
        Conflicting groups sorted by sacrifice priority.
    """
    L_f = full_scale_params["L"]
    B_f = full_scale_params["B"]
    h_f = full_scale_params["h"]
    U_f = full_scale_params["U"]
    sigma_c_f = full_scale_params["sigma_c"]
    sigma_f_f = full_scale_params["sigma_f"]
    E_f = full_scale_params.get("E", 5e9)
    rho_i = full_scale_params.get("rho_i", 900.0)
    rho_w = full_scale_params.get("rho_w", 1025.0)
    g = full_scale_params.get("g", 9.81)
    mu = full_scale_params.get("mu", 0.15)
    porosity = full_scale_params.get("porosity", 0.3)
    K_IC = full_scale_params.get("K_IC", 100e3)

    # Full-scale groups
    groups_full = compute_all_groups(
        L=L_f, U=U_f, h=h_f, B=B_f,
        sigma_i=sigma_f_f, E=E_f,
        rho_i=rho_i, rho_w=rho_w, g=g,
        mu=mu, porosity=porosity, K_IC=K_IC,
    )

    # Model-scale parameters
    L_m = L_f / lambda_ratio
    B_m = B_f / lambda_ratio
    h_m = h_f / lambda_ratio
    U_m = U_f / math.sqrt(lambda_ratio)

    # Ice strength at model scale (depends on scaling law)
    sigma_f_m = scale_property(scaling_law, sigma_f_f, lambda_ratio)

    # E, K_IC don't scale (material properties)
    E_m = E_f
    K_IC_m = K_IC

    # Model-scale groups
    groups_model = compute_all_groups(
        L=L_m, U=U_m, h=h_m, B=B_m,
        sigma_i=sigma_f_m, E=E_m,
        rho_i=rho_i, rho_w=rho_w, g=g,
        mu=mu, porosity=porosity, K_IC=K_IC_m,
    )

    conflicts = []

    # η (ice strength number)
    eta_full = groups_full["eta"]
    eta_model = groups_model["eta"]
    eta_mismatch = eta_model / eta_full if eta_full != 0 else 1.0
    conflicts.append(ConflictDimension(
        group=ALL_GROUPS[1],  # eta
        model_value=eta_model,
        full_value=eta_full,
        mismatch_ratio=eta_mismatch,
        sacrifice_priority=1,  # First to sacrifice: strength scaling is the main compromise
        cost_description="Ice crushing/breaking pattern changes. "
                         "Model ice may be too strong or too weak relative to gravity.",
    ))

    # Ca (Cauchy number)
    Ca_full = groups_full["Ca"]
    Ca_model = groups_model["Ca"]
    Ca_mismatch = Ca_model / Ca_full if Ca_full != 0 else 1.0
    conflicts.append(ConflictDimension(
        group=ALL_GROUPS[2],  # Ca
        model_value=Ca_model,
        full_value=Ca_full,
        mismatch_ratio=Ca_mismatch,
        sacrifice_priority=2,  # Second: modulus effects are secondary
        cost_description="Elastic deformation pattern changes. "
                         "Ice appears more or less compliant at model scale.",
    ))

    # σ_i/E (strength-modulus ratio)
    sigE_full = groups_full["sigma_over_E"]
    sigE_model = groups_model["sigma_over_E"]
    sigE_mismatch = sigE_model / sigE_full if sigE_full != 0 else 1.0
    conflicts.append(ConflictDimension(
        group=ALL_GROUPS[3],  # sigma_over_E
        model_value=sigE_model,
        full_value=sigE_full,
        mismatch_ratio=sigE_mismatch,
        sacrifice_priority=3,  # Third: brittle/ductile transition
        cost_description="Brittle/ductile transition shifts. "
                         "Model ice may fail in a different mode than full-scale ice.",
    ))

    # K_IC* (fracture toughness number)
    KIC_full = groups_full["K_IC_star"]
    KIC_model = groups_model["K_IC_star"]
    KIC_mismatch = KIC_model / KIC_full if KIC_full != 0 else 1.0
    conflicts.append(ConflictDimension(
        group=ALL_GROUPS[9],  # K_IC_star
        model_value=KIC_model,
        full_value=KIC_full,
        mismatch_ratio=KIC_mismatch,
        sacrifice_priority=4,  # Fourth: fracture pattern
        cost_description="Fracture pattern (crack spacing, size) changes. "
                         "Model ice breaks into different piece sizes.",
    ))

    # Sort by priority
    conflicts.sort(key=lambda c: c.sacrifice_priority)

    return conflicts


def feasible_parameter_window(
    lambda_ratio: float,
    full_scale_params: Dict[str, float],
    max_mismatch: float = 0.15,
) -> Dict[str, Tuple[float, float]]:
    """Determine feasible parameter windows for model ice properties.

    Given a maximum allowable mismatch in each dimensionless group,
    determine what range of model ice properties is acceptable.

    Parameters
    ----------
    lambda_ratio : float
        Scale ratio.
    full_scale_params : dict
        Full-scale parameters.
    max_mismatch : float
        Maximum allowable relative mismatch (default 15%).

    Returns
    -------
    dict
        Mapping property name → (min_value, max_value) feasible window.
    """
    sigma_f_f = full_scale_params["sigma_f"]
    sigma_c_f = full_scale_params["sigma_c"]
    E_f = full_scale_params.get("E", 5e9)
    L_f = full_scale_params["L"]
    rho_i = full_scale_params.get("rho_i", 900.0)
    g = full_scale_params.get("g", 9.81)

    windows = {}

    # For η to be within max_mismatch:
    # η_m / η_f = (σ_m / σ_f) * (L_f / L_m) = (σ_m / σ_f) * λ
    # |η_m / η_f - 1| ≤ max_mismatch
    # → σ_m / σ_f ∈ [(1 - max_mismatch)/λ, (1 + max_mismatch)/λ]
    sigma_f_min = sigma_f_f * (1 - max_mismatch) / lambda_ratio
    sigma_f_max = sigma_f_f * (1 + max_mismatch) / lambda_ratio
    windows["sigma_f"] = (sigma_f_min, sigma_f_max)

    # Same for σ_c
    sigma_c_min = sigma_c_f * (1 - max_mismatch) / lambda_ratio
    sigma_c_max = sigma_c_f * (1 + max_mismatch) / lambda_ratio
    windows["sigma_c"] = (sigma_c_min, sigma_c_max)

    # For Ca to be within max_mismatch:
    # Ca_m / Ca_f = (U_m² / U_f²) * (E_f / E_m) = (1/λ) * (E_f / E_m)
    # Since E_m = E_f (material property): Ca_m / Ca_f = 1/λ
    # This is ALWAYS violated for λ > 1. No feasible window for E.
    # We record this as an infeasible constraint.
    Ca_mismatch = 1.0 / lambda_ratio  # Always < 1 for λ > 1
    if abs(Ca_mismatch - 1.0) > max_mismatch:
        windows["E"] = (float("nan"), float("nan"))  # Infeasible
    else:
        windows["E"] = (E_f, E_f)  # Trivially feasible

    return windows


def lambda_sweep(
    lambda_ratios: List[float],
    full_scale_params: Dict[str, float],
    scaling_laws: List[ScalingLaw] = None,
    resistance_model: callable = None,
) -> Dict[str, List[float]]:
    """Sweep over scale ratios and scaling laws to produce deviation surface.

    Parameters
    ----------
    lambda_ratios : list of float
        Scale ratios to sweep (e.g., [20, 30, 40]).
    full_scale_params : dict
        Full-scale parameters.
    scaling_laws : list of ScalingLaw, optional
        Scaling laws to compare. Defaults to all three.
    resistance_model : callable, optional
        Resistance model function.

    Returns
    -------
    dict
        Mapping (law_name, lambda) → total deviation.
        Also includes arrays for plotting.
    """
    if scaling_laws is None:
        scaling_laws = STRENGTH_LAWS

    from .budget import compute_error_budget

    results = {
        "lambda_ratios": lambda_ratios,
        "law_names": [law.name for law in scaling_laws],
        "deviations": {},  # (law_name, lambda) → deviation
        "deviation_matrix": [],  # for heatmap: rows=laws, cols=lambdas
    }

    for law in scaling_laws:
        devs = []
        for lam in lambda_ratios:
            budget = compute_error_budget(
                lambda_ratio=lam,
                full_scale_params=full_scale_params,
                scaling_law=law,
                resistance_model=resistance_model,
            )
            dev = abs(budget.total_deviation)
            results["deviations"][(law.name, lam)] = dev
            devs.append(dev)
        results["deviation_matrix"].append(devs)

    return results
