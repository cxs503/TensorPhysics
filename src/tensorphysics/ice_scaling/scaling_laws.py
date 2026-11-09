"""Scaling laws for ice model-test extrapolation.

Primary law: Froude similitude
-------------------------------
Under Froude scaling with scale ratio λ = L_full / L_model:

    Length:     L_f = λ · L_m
    Time:       T_f = √λ · T_m
    Velocity:   U_f = √λ · U_m
    Acceleration: g_f = g_m  (gravity is invariant)
    Force:      F_f = λ³ · F_m   (since F ∝ ρ L² U² ∝ λ² · λ = λ³)
    Mass:       M_f = λ³ · M_m
    Pressure:   P_f = P_m        (ρ g h → same if ρ, g same and h ∝ λ)
    Stress:     σ_f = σ_m        (same as pressure)

The critical insight: Froude scaling says stress should be the SAME at
model and full scale. But ice strength is a material property that does
NOT scale with size. This is the fundamental conflict.

Alternative strength scaling laws
----------------------------------
Different researchers have proposed different compromises:

1. **λ⁻¹ law** (strength must decrease at model scale):
   σ_m = σ_f / λ  → preserves the ice strength number η

2. **λ⁻⁰·⁵ law** (partial scaling):
   σ_m = σ_f / √λ  → compromise between strength and fracture mechanics

3. **Cauchy law / no scaling** (strength same at all scales):
   σ_m = σ_f  → simplest but violates η similarity

Each law produces different extrapolation errors when applied to
resistance predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ScalingLaw:
    """A scaling law for a specific property.

    Attributes
    ----------
    name : str
        Identifier.
    property_name : str
        Which physical property this law applies to.
    exponent : float
        σ_model = σ_full * λ^exponent  (λ = L_full/L_model > 1)
        exponent = 0 means no scaling (Cauchy law).
        exponent = -1 means full Froude-compatible scaling.
    description : str
        Physical rationale.
    """
    name: str
    property_name: str
    exponent: float
    description: str


# ── Standard scaling laws ─────────────────────────────────────────

FROUDE_LAW = ScalingLaw(
    name="froude",
    property_name="ice_strength",
    exponent=-1.0,
    description="Full Froude-compatible: σ_m = σ_f / λ. "
                "Preserves ice strength number η = σ/(ρg L).",
)

HALF_LAW = ScalingLaw(
    name="half_froude",
    property_name="ice_strength",
    exponent=-0.5,
    description="Half-Froude: σ_m = σ_f / √λ. "
                "Compromise between strength and fracture similarity.",
)

CAUCHY_LAW = ScalingLaw(
    name="cauchy",
    property_name="ice_strength",
    exponent=0.0,
    description="Cauchy (no scaling): σ_m = σ_f. "
                "Simplest but violates η similarity. "
                "Used as negative control.",
)

STRENGTH_LAWS: List[ScalingLaw] = [FROUDE_LAW, HALF_LAW, CAUCHY_LAW]


# ── Froude scaling core relations ─────────────────────────────────

def froude_scale_length(lambda_ratio: float, model_length: float) -> float:
    """Scale length from model to full: L_f = λ · L_m."""
    return lambda_ratio * model_length


def froude_scale_time(lambda_ratio: float, model_time: float) -> float:
    """Scale time from model to full: T_f = √λ · T_m."""
    return math.sqrt(lambda_ratio) * model_time


def froude_scale_velocity(lambda_ratio: float, model_velocity: float) -> float:
    """Scale velocity from model to full: U_f = √λ · U_m."""
    return math.sqrt(lambda_ratio) * model_velocity


def froude_scale_force(lambda_ratio: float, model_force: float) -> float:
    """Scale force from model to full: F_f = λ³ · F_m."""
    return lambda_ratio**3 * model_force


def froude_unscale_length(lambda_ratio: float, full_length: float) -> float:
    """Unscale length from full to model: L_m = L_f / λ."""
    return full_length / lambda_ratio


def froude_unscale_time(lambda_ratio: float, full_time: float) -> float:
    """Unscale time from full to model: T_m = T_f / √λ."""
    return full_time / math.sqrt(lambda_ratio)


def froude_unscale_velocity(lambda_ratio: float, full_velocity: float) -> float:
    """Unscale velocity from full to model: U_m = U_f / √λ."""
    return full_velocity / math.sqrt(lambda_ratio)


def froude_unscale_force(lambda_ratio: float, full_force: float) -> float:
    """Unscale force from full to model: F_m = F_f / λ³."""
    return full_force / lambda_ratio**3


def froude_round_trip(
    lambda_ratio: float,
    length: float,
    time: float,
    velocity: float,
    force: float,
) -> Dict[str, float]:
    """Round-trip Froude scaling: model → full → model.

    Verifies that scaling up then back down recovers the original values.

    Parameters
    ----------
    lambda_ratio : float
        Scale ratio λ = L_full / L_model.
    length, time, velocity, force : float
        Model-scale values.

    Returns
    -------
    dict
        Residuals (returned - original) for each quantity.
        Should be ≤ 1e-12 relative error for machine-precision consistency.
    """
    # Scale up
    L_f = froude_scale_length(lambda_ratio, length)
    T_f = froude_scale_time(lambda_ratio, time)
    U_f = froude_scale_velocity(lambda_ratio, velocity)
    F_f = froude_scale_force(lambda_ratio, force)

    # Scale back down
    L_m_rt = froude_unscale_length(lambda_ratio, L_f)
    T_m_rt = froude_unscale_time(lambda_ratio, T_f)
    U_m_rt = froude_unscale_velocity(lambda_ratio, U_f)
    F_m_rt = froude_unscale_force(lambda_ratio, F_f)

    return {
        "length_residual": abs(L_m_rt - length),
        "time_residual": abs(T_m_rt - time),
        "velocity_residual": abs(U_m_rt - velocity),
        "force_residual": abs(F_m_rt - force),
        "length_rel_error": abs(L_m_rt - length) / abs(length) if length != 0 else 0.0,
        "time_rel_error": abs(T_m_rt - time) / abs(time) if time != 0 else 0.0,
        "velocity_rel_error": abs(U_m_rt - velocity) / abs(velocity) if velocity != 0 else 0.0,
        "force_rel_error": abs(F_m_rt - force) / abs(force) if force != 0 else 0.0,
        "max_rel_error": max(
            abs(L_m_rt - length) / abs(length) if length != 0 else 0.0,
            abs(T_m_rt - time) / abs(time) if time != 0 else 0.0,
            abs(U_m_rt - velocity) / abs(velocity) if velocity != 0 else 0.0,
            abs(F_m_rt - force) / abs(force) if force != 0 else 0.0,
        ),
    }


def froude_internal_consistency(lambda_ratio: float) -> Dict[str, float]:
    """Check internal consistency of Froude scaling relations.

    Verifies: U = L/T, F = M L/T² = ρ L³ · L/T² = ρ L² U²
    These must hold exactly under the scaling exponents.

    Returns
    -------
    dict
        Consistency checks with residuals.
    """
    # Base exponents
    alpha_L = 1.0       # length ∝ λ^1
    alpha_T = 0.5       # time ∝ λ^0.5
    alpha_U = 0.5       # velocity ∝ λ^0.5
    alpha_F = 3.0       # force ∝ λ^3
    alpha_M = 3.0       # mass ∝ λ^3

    # Check U = L/T: exponent should be alpha_L - alpha_T = 0.5
    vel_from_L_T = alpha_L - alpha_T
    # Check F = M L / T²: exponent should be alpha_M + alpha_L - 2*alpha_T = 3
    force_from_MLT = alpha_M + alpha_L - 2 * alpha_T
    # Check F = ρ L² U²: exponent should be 2*alpha_L + 2*alpha_U = 3
    force_from_rhoLU = 2 * alpha_L + 2 * alpha_U

    return {
        "velocity_from_L_T_exponent": vel_from_L_T,
        "velocity_expected_exponent": alpha_U,
        "velocity_consistency": abs(vel_from_L_T - alpha_U),
        "force_from_MLT_exponent": force_from_MLT,
        "force_expected_exponent": alpha_F,
        "force_MLT_consistency": abs(force_from_MLT - alpha_F),
        "force_from_rhoLU_exponent": force_from_rhoLU,
        "force_rhoLU_consistency": abs(force_from_rhoLU - alpha_F),
        "max_consistency_error": max(
            abs(vel_from_L_T - alpha_U),
            abs(force_from_MLT - alpha_F),
            abs(force_from_rhoLU - alpha_F),
        ),
    }


def scale_property(
    law: ScalingLaw,
    full_scale_value: float,
    lambda_ratio: float,
) -> float:
    """Apply a scaling law to get model-scale property value.

    σ_model = σ_full × λ^exponent

    Parameters
    ----------
    law : ScalingLaw
        The scaling law to apply.
    full_scale_value : float
        Full-scale property value.
    lambda_ratio : float
        Scale ratio λ = L_full / L_model (> 1).

    Returns
    -------
    float
        Model-scale property value.
    """
    return full_scale_value * (lambda_ratio ** law.exponent)


def unscale_property(
    law: ScalingLaw,
    model_scale_value: float,
    lambda_ratio: float,
) -> float:
    """Reverse a scaling law: get full-scale from model-scale.

    σ_full = σ_model × λ^(-exponent)
    """
    return model_scale_value * (lambda_ratio ** (-law.exponent))
