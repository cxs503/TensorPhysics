"""Automatic failure-mode identification for Smith analyses.

The Smith method gives, for free, the state of every element at the ultimate
curvature.  The dominant failure mode of the section is the mechanism that
removes the largest amount of moment capacity from the *compression* side at
collapse:

``capacity loss`` of element ``e``  ``L_e = A_e (sigma_y - sigma_u,e) |z_e - z_NA|``

with ``sigma_u,e = phi(beta_e) sigma_y`` for plating and
``sigma_u,e = min(sigma_y, sigma_column, sigma_tripping)`` for stiffeners.  The
element mechanism labels come from the element law construction
(:meth:`.solver.SmithSolver._build_models`), so the attribution is a
bookkeeping operation on documented element capacities, not a fit.

When the runner-up mechanism carries >= ``interaction_threshold`` of the
dominant loss the mode is reported as an *interaction* (e.g.
``plate_buckling+stiffener_tripping``), which is the classical
plate-stiffener interaction of the Smith method.
"""

from __future__ import annotations

from typing import Dict, List, Optional

__all__ = ["identify_failure_mode", "MODE_LABELS"]


MODE_LABELS = {
    "plate_buckling": "compression plating buckles (beta > 1, phi < 1)",
    "yielding": "compression elements reach yield without buckling (beta <= 1)",
    "stiffener_column_buckling": "stiffener column buckling (Euler + Johnson-Ostenfeld)",
    "stiffener_tripping": "stiffener lateral-torsional (tripping) buckling",
    "tension_yield": "tension flange yields (tension-side mechanism)",
}


def _mechanism_of(element: Dict[str, object], compression: bool) -> str:
    mechs = list(element.get("mechanisms") or [])
    if not compression:
        return "tension_yield"
    for key in ("plate_buckling", "stiffener_tripping", "stiffener_column_buckling", "yielding"):
        if key in mechs:
            return key
    return "yielding"


def identify_failure_mode(
    elements: Dict[str, Dict[str, object]],
    sigma_y: float,
    z_na: float,
    loading: str = "bending",
    bending: str = "sagging",
    interaction_threshold: float = 0.6,
) -> Dict[str, object]:
    """Identify the dominant failure mode from the element states at collapse.

    Parameters
    ----------
    elements : dict
        Per-element response from a :class:`~.solver.SmithResult`.
    sigma_y : float
        Yield stress [Pa] (yield of the section material).
    z_na : float
        Neutral axis at ultimate [m] (unused in the axial mode).
    loading : str
        ``"bending"`` or ``"axial"``.
    bending : str
        Bending direction for the bending mode (``"sagging"``/``"hogging"``).
    interaction_threshold : float
        Runner-up / dominant loss ratio above which the mode is an interaction.

    Returns
    -------
    dict
        ``dominant``, ``mechanisms`` (loss shares), ``interaction``,
        ``top_elements`` and the per-element loss table.
    """
    comp_high = bending == "sagging"
    shares: Dict[str, float] = {}
    table: List[Dict[str, object]] = []
    for elem_id, el in elements.items():
        strain = float(el.get("strain", 0.0))
        if loading == "axial":
            compression = strain > 0.0
        else:
            z = float(el["z_m"])
            if z >= z_na:
                compression = comp_high
            else:
                compression = not comp_high
        kind = el.get("kind", "plate")
        cap = float(el.get("compression_capacity_MPa", el.get("capacity_MPa", 0.0))) * 1e6
        area = float(el.get("area_m2", 0.0))
        z = float(el.get("z_m", 0.0))
        lever = abs(z - z_na) if loading == "bending" else 1.0
        loss = area * max(0.0, sigma_y - cap) * lever if compression else 0.0
        mech = _mechanism_of(el, compression)
        shares[mech] = shares.get(mech, 0.0) + loss
        table.append({
            "id": elem_id, "kind": kind, "mechanism": mech,
            "compression": bool(compression),
            "capacity_MPa": cap / 1e6,
            "sigma_y_MPa": sigma_y / 1e6,
            "utilization": float(el.get("utilization", 0.0)),
            "strain": strain,
            "z_m": z,
            "lever_m": lever,
            "capacity_loss": loss,
        })
    total = sum(shares.values())
    if total <= 0.0:
        return {
            "dominant": "yielding",
            "label": MODE_LABELS["yielding"],
            "interaction": False,
            "mechanisms": {},
            "shares": {},
            "secondary": None,
            "top_elements": sorted(table, key=lambda r: -abs(float(r["utilization"])))[:3],
            "elements": table,
            "note": "no capacity loss (every compression element reaches yield): "
                    "the section is compact",
        }
    frac = {k: v / total for k, v in shares.items()}
    ranked = sorted(frac.items(), key=lambda kv: -kv[1])
    dominant = ranked[0][0]
    secondary: Optional[str] = ranked[1][0] if len(ranked) > 1 else None
    interaction = bool(secondary is not None and ranked[1][1] >= interaction_threshold)
    label = MODE_LABELS.get(dominant, dominant)
    if interaction:
        label = f"{dominant} + {secondary}: {MODE_LABELS.get(dominant, dominant)} with " \
                f"{MODE_LABELS.get(secondary, secondary)}"
    return {
        "dominant": dominant,
        "secondary": secondary,
        "interaction": interaction,
        "label": label,
        "mechanisms": {k: v for k, v in shares.items()},
        "shares": frac,
        "top_elements": sorted(table, key=lambda r: -float(r["capacity_loss"]))[:3],
        "elements": table,
    }