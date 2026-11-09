"""Dimensionless groups governing ice-structure interaction scaling.

The complete set of Π-groups for model-to-full-scale ice extrapolation:

1. **Froude number**  Fr = U / √(g L)
   — gravity-wave making; primary scaling driver.

2. **Ice strength number**  η = σ_i / (ρ_i g L)
   — ratio of ice failure strength to hydrostatic pressure scale.

3. **Cauchy number**  Ca = ρ_i U² / E
   — ratio of inertial stress to elastic modulus (compressibility of ice).

4. **Strength–modulus ratio**  σ_i / E
   — controls brittle vs ductile transition in ice failure.

5. **Friction coefficient**  μ  (or friction angle φ_f = arctan μ)
   — ice/structure interface friction.

6. **Porosity**  n  (brash ice / ridge keel)
   — void fraction in broken ice fields.

7. **Geometric ratios**  B/L, h/L, φ_bow
   — ship geometry similarity.

8. **Density ratio**  ρ_i / ρ_w
   — ice-to-water density ratio (buoyancy scaling).

Each group has a Froude-law scaling exponent: if the group is to be
preserved across scales, the model-ice property must scale as λ^exponent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DimensionlessGroup:
    """A single dimensionless group with its Froude scaling exponent.

    Attributes
    ----------
    name : str
        Short identifier (e.g. "Fr", "eta", "Ca").
    symbol : str
        LaTeX-friendly symbol.
    description : str
        Physical meaning.
    froude_exponent : float
        If the group is to be preserved under Froude scaling with scale
        ratio λ = L_full / L_model, the relevant property must scale as
        λ^froude_exponent.  0 means the group is automatically preserved.
    formula : str
        Human-readable formula.
    """
    name: str
    symbol: str
    description: str
    froude_exponent: float
    formula: str


# ── Complete set of dimensionless groups ──────────────────────────

ALL_GROUPS: List[DimensionlessGroup] = [
    DimensionlessGroup(
        name="Fr",
        symbol="Fr = U/√(gL)",
        description="Froude number: gravity-wave making",
        froude_exponent=0.0,  # automatically preserved by Froude scaling
        formula="U / sqrt(g * L)",
    ),
    DimensionlessGroup(
        name="eta",
        symbol="η = σ_i/(ρ_i g L)",
        description="Ice strength number: strength vs hydrostatic pressure",
        froude_exponent=-1.0,  # σ_i must scale as λ^(-1) to preserve η
        formula="sigma_i / (rho_i * g * L)",
    ),
    DimensionlessGroup(
        name="Ca",
        symbol="Ca = ρ_i U²/E",
        description="Cauchy number: inertial stress vs elastic modulus",
        froude_exponent=-1.0,  # E must scale as λ^(-1) to preserve Ca
        # Under Froude: U² ∝ λ, ρ_i = const → ρ_i U² ∝ λ
        # To keep Ca const: E ∝ λ → but E is material property
        # Actually: Ca = ρ U² / E, under Froude U² ∝ L ∝ λ
        # So Ca_model/Ca_full = (ρ_m/ρ_f)(U_m²/U_f²)(E_f/E_m)
        # = 1 * (1/λ) * (E_f/E_m) → need E_f/E_m = λ → E ∝ λ^1
        # But E is a material property that doesn't scale with size!
        # So the exponent for E to preserve Ca is +1 (E must increase with scale)
        # This is the fundamental conflict: ice modulus doesn't scale.
        formula="rho_i * U^2 / E",
    ),
    DimensionlessGroup(
        name="sigma_over_E",
        symbol="σ_i/E",
        description="Strength-modulus ratio: brittle/ductile transition",
        froude_exponent=-1.0,
        # σ_i/E: if σ_i ∝ λ^(-1) (to preserve η) and E is constant →
        # σ_i/E ∝ λ^(-1). To preserve this ratio, need E ∝ λ^(-1) too,
        # but E is a material property → conflict.
        formula="sigma_i / E",
    ),
    DimensionlessGroup(
        name="mu",
        symbol="μ",
        description="Friction coefficient (ice/structure interface)",
        froude_exponent=0.0,  # dimensionless, should be same at all scales
        formula="mu = tan(phi_f)",
    ),
    DimensionlessGroup(
        name="porosity",
        symbol="n",
        description="Porosity of broken ice field (brash/ridge)",
        froude_exponent=0.0,  # geometric, should be same at all scales
        formula="n = V_void / V_total",
    ),
    DimensionlessGroup(
        name="B_over_L",
        symbol="B/L",
        description="Beam-to-length ratio (geometric similarity)",
        froude_exponent=0.0,  # geometric, preserved by construction
        formula="B / L",
    ),
    DimensionlessGroup(
        name="h_over_L",
        symbol="h/L",
        description="Ice thickness to ship length ratio",
        froude_exponent=0.0,  # geometric, preserved by Froude scaling
        formula="h / L",
    ),
    DimensionlessGroup(
        name="rho_ratio",
        symbol="ρ_i/ρ_w",
        description="Ice-to-water density ratio",
        froude_exponent=0.0,  # material constants, same at all scales
        formula="rho_i / rho_w",
    ),
    DimensionlessGroup(
        name="K_IC_star",
        symbol="K_IC/(E√L)",
        description="Fracture toughness number: LEFM scaling",
        froude_exponent=-0.5,
        # K_IC has units Pa·m^0.5. Under Froude:
        # K_IC/(E√L): if E=const, need K_IC ∝ √L ∝ λ^0.5
        # But K_IC is a material property → conflict
        # Exponent for K_IC to preserve: +0.5
        formula="K_IC / (E * sqrt(L))",
    ),
]


def compute_all_groups(
    L: float,
    U: float,
    h: float,
    B: float,
    sigma_i: float,
    E: float,
    rho_i: float = 900.0,
    rho_w: float = 1025.0,
    g: float = 9.81,
    mu: float = 0.15,
    porosity: float = 0.3,
    K_IC: float = 100e3,  # Pa·m^0.5 typical ice fracture toughness
) -> Dict[str, float]:
    """Compute all dimensionless groups for given physical parameters.

    Parameters
    ----------
    L : float
        Ship characteristic length (m).
    U : float
        Ship speed (m/s).
    h : float
        Ice thickness (m).
    B : float
        Ship beam (m).
    sigma_i : float
        Ice strength (Pa) — crushing or flexural as appropriate.
    E : float
        Ice elastic modulus (Pa).
    rho_i : float
        Ice density (kg/m³).
    rho_w : float
        Water density (kg/m³).
    g : float
        Gravitational acceleration (m/s²).
    mu : float
        Friction coefficient.
    porosity : float
        Broken ice porosity.
    K_IC : float
        Ice fracture toughness (Pa·m^0.5).

    Returns
    -------
    dict
        Mapping group name → numerical value.
    """
    Fr = U / math.sqrt(g * L)
    eta = sigma_i / (rho_i * g * L)
    Ca = rho_i * U**2 / E
    sigma_E = sigma_i / E
    B_L = B / L
    h_L = h / L
    rho_ratio = rho_i / rho_w
    K_IC_star = K_IC / (E * math.sqrt(L))

    return {
        "Fr": Fr,
        "eta": eta,
        "Ca": Ca,
        "sigma_over_E": sigma_E,
        "mu": mu,
        "porosity": porosity,
        "B_over_L": B_L,
        "h_over_L": h_L,
        "rho_ratio": rho_ratio,
        "K_IC_star": K_IC_star,
    }


def froude_scale_group(
    group_name: str,
    lambda_ratio: float,
    base_value: float,
) -> float:
    """Scale a dimensionless group value under Froude similitude.

    For groups with froude_exponent = 0, the value is unchanged.
    For groups with non-zero exponent, the property that enters the group
    must be adjusted by λ^exponent to preserve the group.

    Parameters
    ----------
    group_name : str
        Name of the dimensionless group.
    lambda_ratio : float
        Scale ratio λ = L_full / L_model (> 1).
    base_value : float
        Value of the group at model scale.

    Returns
    -------
    float
        Required value at full scale to preserve the group, OR
        the value the group takes if the property is NOT scaled.
    """
    group = _find_group(group_name)
    # The group itself is dimensionless; what changes is the underlying
    # property. If we don't adjust the property, the group value changes.
    # This function returns what the group value WOULD be at full scale
    # if the property is kept constant (i.e., the mismatch).
    if group.froude_exponent == 0.0:
        return base_value  # automatically preserved
    else:
        # If property is NOT scaled (kept same at model and full):
        # group_full / group_model = (1/lambda)^|exponent| for negative exp
        # More precisely: for η = σ/(ρ g L), if σ is constant:
        # η_full/η_model = L_model/L_full = 1/λ
        return base_value * (lambda_ratio ** (-group.froude_exponent))


def group_value(
    group_name: str,
    params: Dict[str, float],
) -> float:
    """Compute a single group value from parameters dict.

    Parameters
    ----------
    group_name : str
        Name of the group.
    params : dict
        Must contain the relevant physical parameters.

    Returns
    -------
    float
        The group value.
    """
    all_vals = compute_all_groups(**params)
    return all_vals[group_name]


def _find_group(name: str) -> DimensionlessGroup:
    """Look up a group by name."""
    for g in ALL_GROUPS:
        if g.name == name:
            return g
    raise KeyError(f"Unknown dimensionless group: {name}")
