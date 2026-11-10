"""Core size effect scaling law for ice breaking force.

Implements the F ∝ L^n scaling with n ∈ [2.5, 3.0], including the
Bažant-type quasi-brittle size effect law adapted for ice-ship interaction.

The key insight: ice breaking force does NOT scale purely as λ³ (Froude)
because ice is a quasi-brittle material with a finite fracture process zone.
The effective exponent n depends on the relative importance of:
- Gravity-driven bending (n → 3.0)
- Fracture mechanics (n → 2.5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class SizeEffectLaw:
    """Size effect law parameters for ice breaking force scaling.

    Attributes
    ----------
    n : float
        Size effect exponent. Literature range: 2.5–3.0.
        n = 3.0: pure Froude (gravity-dominated).
        n = 2.5: fracture-dominated (LEFM-like).
    L_ch : float
        Characteristic length (m) related to fracture process zone.
        Typical range: 0.5–5.0 m for sea ice.
    delta : float
        Transition sharpness parameter. δ = 0.5 gives smooth transition.
    F_ref : float
        Reference force (N) at reference length L_ref.
    L_ref : float
        Reference length (m) at which F_ref is defined.
    """
    n: float = 2.75
    L_ch: float = 2.0
    delta: float = 0.5
    F_ref: float = 1.0
    L_ref: float = 1.0

    def __post_init__(self):
        if not (2.3 <= self.n <= 3.2):
            raise ValueError(f"Size effect exponent n={self.n} outside plausible range [2.3, 3.2]")
        if self.L_ch <= 0:
            raise ValueError(f"Characteristic length L_ch={self.L_ch} must be positive")
        if self.F_ref <= 0:
            raise ValueError(f"Reference force F_ref={self.F_ref} must be positive")
        if self.L_ref <= 0:
            raise ValueError(f"Reference length L_ref={self.L_ref} must be positive")


# Default parameters based on literature synthesis
DEFAULT_SIZE_EFFECT_PARAMS = SizeEffectLaw(
    n=2.75,       # Mid-range of literature values (2.5–3.0)
    L_ch=2.0,     # Typical fracture process zone scale for sea ice
    delta=0.5,    # Smooth transition
    F_ref=1.0,    # Normalized
    L_ref=1.0,    # Normalized
)


def compute_size_effect_force(
    L: float,
    law: SizeEffectLaw = DEFAULT_SIZE_EFFECT_PARAMS,
) -> float:
    """Compute ice breaking force at length L using size effect law.

    F(L) = F_ref · (L/L_ref)^n

    This is the pure power-law form, appropriate when L >> L_ch
    (gravity-dominated regime).

    Parameters
    ----------
    L : float
        Characteristic length (m) — typically ship length or ice feature size.
    law : SizeEffectLaw
        Size effect law parameters.

    Returns
    -------
    float
        Ice breaking force (N).
    """
    return law.F_ref * (L / law.L_ref) ** law.n


def compute_size_effect_exponent(
    F_model: float,
    F_full: float,
    L_model: float,
    L_full: float,
) -> float:
    """Back-calculate the effective size effect exponent from force measurements.

    n = log(F_full / F_model) / log(L_full / L_model)

    Parameters
    ----------
    F_model : float
        Model-scale breaking force (N).
    F_full : float
        Full-scale breaking force (N).
    L_model : float
        Model-scale characteristic length (m).
    L_full : float
        Full-scale characteristic length (m).

    Returns
    -------
    float
        Effective size effect exponent n.
    """
    lambda_ratio = L_full / L_model
    if lambda_ratio <= 1.0:
        raise ValueError(f"Scale ratio λ={lambda_ratio} must be > 1 for model→full extrapolation")
    force_ratio = F_full / F_model
    if force_ratio <= 0:
        raise ValueError(f"Force ratio must be positive, got {force_ratio}")
    return math.log(force_ratio) / math.log(lambda_ratio)


def bazant_type_size_effect(
    L: float,
    law: SizeEffectLaw = DEFAULT_SIZE_EFFECT_PARAMS,
) -> float:
    """Bažant-type quasi-brittle size effect law for ice breaking force.

    F(L) = F_ref · (L/L_ref)^n · [1 + (L/L_ch)]^(-δ)

    This captures the transition from fracture-dominated (small L) to
    gravity-dominated (large L) behavior:
    - For L << L_ch: F ∝ L^(n-δ) ≈ L^2.25 (fracture-dominated)
    - For L >> L_ch: F ∝ L^n ≈ L^2.75 (gravity-dominated)

    Parameters
    ----------
    L : float
        Characteristic length (m).
    law : SizeEffectLaw
        Size effect law parameters.

    Returns
    -------
    float
        Ice breaking force (N).
    """
    ratio = L / law.L_ref
    transition = 1.0 + (L / law.L_ch)
    return law.F_ref * (ratio ** law.n) * (transition ** (-law.delta))


def froude_vs_fracture_transition(
    L: float,
    L_ch: float = 2.0,
    n_froude: float = 3.0,
    n_fracture: float = 2.5,
) -> Tuple[float, float, float]:
    """Compute the effective exponent as a function of length.

    Returns the effective exponent n_eff(L) that transitions smoothly
    from n_fracture (small L) to n_froude (large L).

    n_eff(L) = n_fracture + (n_froude - n_fracture) · L/(L + L_ch)

    Parameters
    ----------
    L : float
        Characteristic length (m).
    L_ch : float
        Transition length scale (m).
    n_froude : float
        Exponent in gravity-dominated regime (default 3.0).
    n_fracture : float
        Exponent in fracture-dominated regime (default 2.5).

    Returns
    -------
    tuple
        (n_eff, weight_froude, weight_fracture)
        n_eff: effective exponent at length L
        weight_froude: contribution weight of Froude regime
        weight_fracture: contribution weight of fracture regime
    """
    w_f = L / (L + L_ch)
    w_fr = 1.0 - w_f
    n_eff = n_fracture * w_fr + n_froude * w_f
    return n_eff, w_f, w_fr


def scale_force_model_to_full(
    F_model: float,
    L_model: float,
    L_full: float,
    n: float = 2.75,
) -> float:
    """Scale ice breaking force from model to full scale.

    F_full = F_model · (L_full / L_model)^n

    Parameters
    ----------
    F_model : float
        Model-scale breaking force (N).
    L_model : float
        Model-scale characteristic length (m).
    L_full : float
        Full-scale characteristic length (m).
    n : float
        Size effect exponent (default 2.75, literature range 2.5–3.0).

    Returns
    -------
    float
        Full-scale breaking force (N).
    """
    lambda_ratio = L_full / L_model
    return F_model * (lambda_ratio ** n)


def scale_force_full_to_model(
    F_full: float,
    L_model: float,
    L_full: float,
    n: float = 2.75,
) -> float:
    """Scale ice breaking force from full to model scale (inverse).

    F_model = F_full · (L_model / L_full)^n

    Parameters
    ----------
    F_full : float
        Full-scale breaking force (N).
    L_model : float
        Model-scale characteristic length (m).
    L_full : float
        Full-scale characteristic length (m).
    n : float
        Size effect exponent.

    Returns
    -------
    float
        Model-scale breaking force (N).
    """
    lambda_ratio = L_full / L_model
    return F_full / (lambda_ratio ** n)


def effective_exponent_sweep(
    L_values: list,
    L_ch: float = 2.0,
    n_froude: float = 3.0,
    n_fracture: float = 2.5,
) -> Dict[str, list]:
    """Sweep effective exponent over a range of lengths.

    Parameters
    ----------
    L_values : list of float
        Lengths to evaluate (m).
    L_ch : float
        Transition length scale (m).
    n_froude : float
        Gravity-dominated exponent.
    n_fracture : float
        Fracture-dominated exponent.

    Returns
    -------
    dict
        Mapping 'L', 'n_eff', 'w_froude', 'w_fracture' to lists.
    """
    results = {"L": [], "n_eff": [], "w_froude": [], "w_fracture": []}
    for L in L_values:
        n_eff, w_f, w_fr = froude_vs_fracture_transition(L, L_ch, n_froude, n_fracture)
        results["L"].append(L)
        results["n_eff"].append(n_eff)
        results["w_froude"].append(w_f)
        results["w_fracture"].append(w_fr)
    return results
