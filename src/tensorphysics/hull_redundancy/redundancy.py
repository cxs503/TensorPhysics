"""Redundancy Index Factor (RIF) machinery: matrix, ranking, sensitivity.

RIF definition
--------------
``RIF(c, m) = M_u(damage: component c, mode m) / M_u(intact)``

with ``M_u`` the section-level ultimate bending moment (magnitude) from the
incremental Smith solver, evaluated separately for **sagging** (deck in
compression) and **hogging** (bottom in compression).

Deliverables
------------
- the full component x failure-mode RIF matrix per bending direction,
- the component-wise worst-case RIF (``rif_min``) and the redundancy
  classification (``rif_min >= 0.80`` -> redundancy class "redundant"),
- the top-3 most dangerous components,
- the RIF sensitivity to the failure position (Spearman rank correlation
  between the lever arm about the ultimate neutral axis and RIF; the physical
  requirement is a strong negative correlation),
- the negative control (components sitting on the neutral axis must not matter),
- the discretisation convergence study (doubling the fiber/curvature
  discretisation must change RIF by <= 3 %).

A subtlety worth stating: the **elastic** neutral axis and the **ultimate**
neutral axis are different objects.  When the compression flange sheds load the
zero-strain point migrates towards the tension side (sagging: 9.53 m -> 5.69 m),
so a strip that sits on the elastic neutral axis can still be several metres away
from the plastic one.  Both references are therefore reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .damage import DamageSpec, SEVERITY_RANK, apply_damage
from .section import HullSection
from .smith import SmithResult, SmithSectionSolver

__all__ = [
    "RIFEntry",
    "spearman_rho",
    "rif_matrix",
    "worst_case_by_component",
    "most_dangerous",
    "redundancy_set",
    "negative_control",
    "sensitivity_analysis",
    "convergence_study",
]

DEFAULT_MODES_BY_KIND: Dict[str, Tuple[str, ...]] = {
    "plate": ("remove",),                       # plate strips have one tearing mode
    "stiffener": ("remove", "trip", "debond"),
}


@dataclass(frozen=True)
class RIFEntry:
    component_id: str
    kind: str
    mode: str
    z: float
    area: float
    M_u_damaged: float
    M_u_intact: float
    rif: float
    z_NA_elastic: float
    z_NA_ultimate: float
    bending: str

    @property
    def lever_arm_elastic(self) -> float:
        return abs(self.z - self.z_NA_elastic)

    @property
    def lever_arm_ultimate(self) -> float:
        return abs(self.z - self.z_NA_ultimate)

    def as_dict(self) -> Dict[str, object]:
        return {
            "component_id": self.component_id,
            "kind": self.kind,
            "mode": self.mode,
            "bending": self.bending,
            "z_m": self.z,
            "area_m2": self.area,
            "lever_arm_elastic_m": self.lever_arm_elastic,
            "lever_arm_ultimate_m": self.lever_arm_ultimate,
            "M_u_damaged_MNm": self.M_u_damaged / 1e6,
            "M_u_intact_MNm": self.M_u_intact / 1e6,
            "rif": self.rif,
            "delta_M_MNm": (self.M_u_intact - self.M_u_damaged) / 1e6,
        }


# ---------------------------------------------------------------------------
# statistics helpers (no scipy dependency)
# ---------------------------------------------------------------------------
def _average_ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with averaged ranks for ties."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3:
        raise ValueError("need at least 3 pairs")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


# ---------------------------------------------------------------------------
# RIF matrix
# ---------------------------------------------------------------------------
def rif_matrix(
    section: HullSection,
    modes_by_kind: Optional[Dict[str, Tuple[str, ...]]] = None,
    component_ids: Optional[Iterable[str]] = None,
    solver_kwargs: Optional[Dict[str, object]] = None,
    intact_result: Optional[SmithResult] = None,
    bending: str = "sagging",
) -> Dict[str, object]:
    """Component x failure-mode RIF matrix for one section and bending sense."""
    kw = dict(solver_kwargs or {})
    kw["bending"] = bending
    modes_by_kind = modes_by_kind or DEFAULT_MODES_BY_KIND

    if intact_result is None:
        intact_result = SmithSectionSolver(
            apply_damage(section, DamageSpec(section.component_ids()[0], "none")), **kw
        ).solve()
    M_intact = intact_result.M_u
    z_na_el = section.neutral_axis()
    z_na_u = intact_result.z_NA_u

    ids = list(component_ids) if component_ids is not None else section.component_ids()
    entries: List[RIFEntry] = []
    for cid in ids:
        kind = section.component_kind(cid)
        comp = section.plate(cid) if kind == "plate" else section.stiffener(cid)
        for mode in modes_by_kind[kind]:
            res = SmithSectionSolver(apply_damage(section, DamageSpec(cid, mode)), **kw).solve()
            entries.append(
                RIFEntry(
                    component_id=cid,
                    kind=kind,
                    mode=mode,
                    z=float(comp.z),
                    area=float(comp.area),
                    M_u_damaged=float(res.M_u),
                    M_u_intact=float(M_intact),
                    rif=float(res.M_u / M_intact),
                    z_NA_elastic=float(z_na_el),
                    z_NA_ultimate=float(z_na_u),
                    bending=bending,
                )
            )

    # negative control: re-running the intact section must reproduce RIF = 1
    rif_none = (
        SmithSectionSolver(
            apply_damage(section, DamageSpec(section.component_ids()[0], "none")), **kw
        ).solve().M_u
        / M_intact
    )

    return {
        "section": section.name,
        "bending": bending,
        "M_u_intact_Nm": float(M_intact),
        "M_u_intact_MNm": float(M_intact / 1e6),
        "z_NA_elastic_m": float(z_na_el),
        "z_NA_ultimate_m": float(z_na_u),
        "kappa_u_intact": float(intact_result.kappa_u),
        "n_entries": len(entries),
        "entries": [e.as_dict() for e in entries],
        "rif_negative_control": float(rif_none),
        "rif_in_unit_interval": bool(
            all(0.0 <= e.rif <= 1.0 + 1e-12 for e in entries)
        ),
        "rif_min": float(min(e.rif for e in entries)),
        "rif_max": float(max(e.rif for e in entries)),
        "_entry_objects": entries,
        "_intact_result": intact_result,
    }


def worst_case_by_component(matrix: Dict[str, object]) -> List[Dict[str, object]]:
    """Worst (lowest) RIF over the failure modes of each component."""
    by_comp: Dict[str, Dict[str, object]] = {}
    for e in matrix["entries"]:                       # type: ignore[index]
        cid = e["component_id"]                       # type: ignore[index]
        rec = by_comp.get(cid)
        if rec is None or e["rif"] < rec["rif_min"]:   # type: ignore[index]
            by_comp[cid] = {
                "component_id": cid,
                "kind": e["kind"],                    # type: ignore[index]
                "z_m": e["z_m"],                      # type: ignore[index]
                "area_m2": e["area_m2"],              # type: ignore[index]
                "lever_arm_ultimate_m": e["lever_arm_ultimate_m"],  # type: ignore[index]
                "rif_min": e["rif"],                  # type: ignore[index]
                "governing_mode": e["mode"],          # type: ignore[index]
            }
    out = list(by_comp.values())
    out.sort(key=lambda r: float(r["rif_min"]))
    return out


def most_dangerous(matrix: Dict[str, object], k: int = 3) -> List[Dict[str, object]]:
    """The ``k`` components with the smallest worst-case RIF."""
    return worst_case_by_component(matrix)[:k]


def redundancy_set(
    matrix: Dict[str, object], threshold: float = 0.80
) -> Dict[str, object]:
    """Components whose single failure leaves RIF >= threshold (redundancy class)."""
    worst = worst_case_by_component(matrix)
    redundant = [r for r in worst if float(r["rif_min"]) >= threshold]
    critical = [r for r in worst if float(r["rif_min"]) < threshold]
    return {
        "bending": matrix["bending"],
        "threshold": float(threshold),
        "redundant_components": redundant,
        "critical_components": critical,
        "n_redundant": len(redundant),
        "n_critical": len(critical),
        "all_single_failures_redundant": bool(len(critical) == 0),
    }


def negative_control(
    matrix: Dict[str, object], n_closest: int = 2, threshold: float = 0.98
) -> Dict[str, object]:
    """Removal of the components closest to the neutral axis must barely matter.

    Two equivalent readings are reported because they can disagree:

    ``ultimate_NA``
        components with the smallest lever arm about the *ultimate* (plastic)
        neutral axis - these really are the ones the collapse mechanism does not
        use, and their removal must leave ``RIF >= threshold``;
    ``elastic_NA``
        components with the smallest lever arm about the elastic neutral axis -
        reported for comparison, since a mid-side shell strip can be 0.5 m from
        the elastic axis and 4 m from the plastic one.
    """
    entries = [e for e in matrix["entries"] if e["mode"] == "remove"]  # type: ignore[index]
    if not entries:
        raise ValueError("matrix has no 'remove' entries")
    picks_u = sorted(entries, key=lambda e: float(e["lever_arm_ultimate_m"]))[:n_closest]  # type: ignore[index]
    picks_e = sorted(entries, key=lambda e: float(e["lever_arm_elastic_m"]))[:n_closest]   # type: ignore[index]

    def _pack(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return [
            {
                "component_id": r["component_id"],
                "lever_arm_ultimate_m": r["lever_arm_ultimate_m"],
                "lever_arm_elastic_m": r["lever_arm_elastic_m"],
                "rif": r["rif"],
            }
            for r in rows
        ]

    pack_u = _pack(picks_u)
    pack_e = _pack(picks_e)
    return {
        "bending": matrix["bending"],
        "threshold": float(threshold),
        "z_NA_ultimate_m": matrix["z_NA_ultimate_m"],
        "z_NA_elastic_m": matrix["z_NA_elastic_m"],
        "ultimate_NA_picks": pack_u,
        "elastic_NA_picks": pack_e,
        "ultimate_NA_pass": bool(
            all(float(r["rif"]) >= threshold for r in pack_u)
        ),
        "elastic_NA_pass": bool(all(float(r["rif"]) >= threshold for r in pack_e)),
    }


def sensitivity_analysis(
    matrix: Dict[str, object],
    modes: Sequence[str] = ("remove",),
    reference: str = "ultimate",
    compression_side_only: bool = True,
) -> Dict[str, object]:
    """RIF versus failure position (lever arm about the neutral axis).

    ``reference = "ultimate"`` uses the plastic neutral axis at collapse,
    ``"elastic"`` the transformed-section axis.  ``compression_side_only``
    restricts the population to the components that are actually compressed.
    Returns the Spearman rank correlation and the least-squares slope of RIF
    against the lever arm, pooled and per component class, so that the
    correlation cannot be confounded by the very different areas of plating and
    stiffener groups.
    """
    if reference not in ("ultimate", "elastic"):
        raise ValueError("reference must be 'ultimate' or 'elastic'")
    z_ref = float(matrix["z_NA_ultimate_m"] if reference == "ultimate"
                  else matrix["z_NA_elastic_m"])
    entries = [e for e in matrix["entries"] if e["mode"] in modes]  # type: ignore[index]
    if compression_side_only:
        if matrix["bending"] == "sagging":
            entries = [e for e in entries if e["z_m"] >= z_ref]     # type: ignore[index]
        else:
            entries = [e for e in entries if e["z_m"] <= z_ref]     # type: ignore[index]

    def _lever(e: Dict[str, object]) -> float:
        return abs(float(e["z_m"]) - z_ref)                          # type: ignore[index]

    def _corr(rows: List[Dict[str, object]]) -> Dict[str, object]:
        if len(rows) < 3:
            return {"n": len(rows), "spearman_rho": None, "slope_per_m": None,
                    "lever_arm_m": [], "rif": []}
        x = [_lever(r) for r in rows]
        y = [float(r["rif"]) for r in rows]
        slope = float(np.polyfit(x, y, 1)[0]) if len(set(x)) > 1 else 0.0
        return {
            "n": len(rows),
            "spearman_rho": spearman_rho(x, y),
            "slope_per_m": slope,
            "lever_arm_m": x,
            "rif": y,
        }

    pooled = _corr(entries)
    plates = _corr([e for e in entries if e["kind"] == "plate"])
    stiff = _corr([e for e in entries if e["kind"] == "stiffener"])
    pooled_rho = pooled["spearman_rho"]
    plate_rho = plates["spearman_rho"]
    return {
        "bending": matrix["bending"],
        "reference": reference,
        "modes": list(modes),
        "compression_side_only": bool(compression_side_only),
        "z_reference_m": z_ref,
        "pooled": pooled,
        "plating": plates,
        "stiffeners": stiff,
        "monotone_ok": bool(
            pooled_rho is not None and float(pooled_rho) <= -0.8
        ),
        "plating_monotone_ok": bool(
            plate_rho is not None and float(plate_rho) <= -0.8
        ),
        "stiffener_class_degenerate": bool(
            stiff["spearman_rho"] is not None and abs(float(stiff["spearman_rho"])) < 0.3
        ),
    }


def convergence_study(
    section: HullSection,
    spec: DamageSpec,
    base_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """RIF convergence under doubling of the element/curvature discretisation."""
    kw = dict(base_kwargs or {})
    n_web = int(kw.pop("n_web_fibers", 2))
    n_steps = int(kw.pop("n_steps", 60))
    bending = str(kw.pop("bending", "sagging"))

    def _rif(nw: int, ns: int) -> float:
        kk = dict(kw)
        kk["bending"] = bending
        intact = SmithSectionSolver(
            apply_damage(section, DamageSpec(section.component_ids()[0], "none")),
            n_web_fibers=nw, n_steps=ns, **kk,
        ).solve()
        damaged = SmithSectionSolver(
            apply_damage(section, spec), n_web_fibers=nw, n_steps=ns, **kk
        ).solve()
        return float(damaged.M_u / intact.M_u)

    rif_coarse = _rif(n_web, n_steps)
    rif_fine = _rif(2 * n_web, 2 * n_steps)
    rel = abs(rif_fine - rif_coarse) / rif_coarse
    return {
        "component_id": spec.component_id,
        "mode": spec.mode,
        "bending": bending,
        "discretisation": {
            "coarse": {"n_web_fibers": n_web, "n_steps": n_steps},
            "fine": {"n_web_fibers": 2 * n_web, "n_steps": 2 * n_steps},
        },
        "rif_coarse": rif_coarse,
        "rif_fine": rif_fine,
        "relative_change": rel,
        "converged_at_3pct": bool(rel <= 0.03),
    }