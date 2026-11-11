"""Unified TensorPhysics entry point for the Smith method (maturation layer).

One call analyses *any* of the supported section archetypes:

======================  =========================  ==============================
``section.kind``        loading automatically set  independent reference
======================  =========================  ==============================
``stiffened_panel``     axial (uniform shortening) panel strength formulae
``plate``               axial                      panel strength formulae
``box_girder``          bending (sagging/hogging)  exact plastic + effective width
``double_bottom``       bending (sagging/hogging)  exact plastic + effective width
``hull_girder``         bending (sagging/hogging)  exact plastic + effective width
======================  =========================  ==============================

The returned payload is JSON-serialisable (except for the ``result`` object,
which can be dropped) so that campaign nodes can persist it as evidence without
knowing anything about the solver internals.  ``loading="auto"`` is the default
and uses :func:`~.idealisation.classify_section`; calling with an explicit
``loading`` overrides the dispatch and is recorded in the payload.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .benchmarks import (
    csr_min_of_modes_panel_strength,
    effective_width_ultimate_moment,
    paik_thayamballi_panel_strength,
)
from .convergence import grid_convergence_study
from .idealisation import GirderSection
from .laws import faulkner_plate_strength_ratio, plate_slenderness
from .modes import identify_failure_mode
from .solver import SmithResult, SmithSolver

__all__ = ["SmithAnalysis", "analyse_section", "analyse_with_convergence", "AUTO_LOADING"]

#: section kinds analysed as panels (uniform compression) rather than in bending
AUTO_LOADING = {
    "stiffened_panel": "axial",
    "plate": "axial",
    "box_girder": "bending",
    "double_bottom": "bending",
    "hull_girder": "bending",
}


class SmithAnalysis:
    """Container for one mature Smith analysis (result + references + mode)."""

    def __init__(
        self,
        section: GirderSection,
        result: SmithResult,
        mode: Dict[str, object],
        reference: Dict[str, object],
        loading: str,
        auto_loading: bool,
    ) -> None:
        self.section = section
        self.result = result
        self.mode = mode
        self.reference = reference
        self.loading = loading
        self.auto_loading = auto_loading

    def as_dict(self) -> Dict[str, object]:
        sec = self.section
        res = self.result
        payload: Dict[str, object] = {
            "section": sec.describe(),
            "loading": self.loading,
            "loading_auto_selected": self.auto_loading,
            "bending": self.bending,
            "reference": self.reference,
            "failure_mode": {
                "dominant": self.mode.get("dominant"),
                "secondary": self.mode.get("secondary"),
                "interaction": self.mode.get("interaction"),
                "label": self.mode.get("label"),
                "shares": self.mode.get("shares"),
                "top_elements": self.mode.get("top_elements"),
            },
            "converged": res.converged,
            "n_fibers": res.n_fibers,
            "diagnostics": res.diagnostics,
            "elements": res.elements,
        }
        if self.loading == "axial":
            payload.update({
                "sigma_u_MPa": res.sigma_u_avg / 1e6,
                "sigma_u_over_sigma_y": res.sigma_u_avg / sec.material.sigma_y0,
                "eps_u": res.eps_u,
                "load_shortening": [[e, s / 1e6] for e, s in res.curve],
            })
        else:
            payload.update({
                "M_u_MNm": res.M_u / 1e6,
                "kappa_u": res.kappa_u,
                "z_NA_m": res.z_NA_u,
                "moment_curvature": [[k, m / 1e6] for k, m in res.curve],
            })
        return payload

    @property
    def bending(self) -> str:
        return self.result.bending


def _panel_reference(section: GirderSection) -> Dict[str, object]:
    """Independent panel-strength references for a stiffened panel."""
    plate = section.plates[0]
    mat = section.material
    beta = plate_slenderness(plate.spacing_b, plate.thickness, mat)
    ref: Dict[str, object] = {
        "beta": beta,
        "faulkner_phi": faulkner_plate_strength_ratio(beta),
    }
    if section.stiffeners:
        s = section.stiffeners[0]
        csr = csr_min_of_modes_panel_strength(
            beta=beta, h=s.h, t_w=s.t_w, span=s.span_a, spacing_b=s.spacing_b,
            plate_t=plate.thickness, material=mat,
        )
        # like-for-like closed form: area-weighted effective-width panel strength
        #   sigma_u = [A_p phi(beta) sigma_y + A_s min(sigma_y, sigma_col, sigma_trip)]
        #             / (A_p + A_s)
        # Same element strengths as the solver, single step instead of progressive
        # averaging, so a mismatch points at the discretisation, not at the physics.
        phi_p = faulkner_plate_strength_ratio(beta)
        A_p = plate.area
        A_s = s.area
        sigma_cap = min(mat.sigma_y0, csr["column_ratio"] * mat.sigma_y0,
                        csr["tripping_ratio"] * mat.sigma_y0)
        ref["area_weighted_effective_width_ratio"] = float(
            (A_p * phi_p * mat.sigma_y0 + A_s * sigma_cap) / ((A_p + A_s) * mat.sigma_y0)
        )
        ref.update({
            "csr_min_of_modes_ratio": csr["sigma_u_over_sigma_y"],
            "csr_governing_mode": csr["governing_mode"],
            "csr_modes": {
                "plate": csr["plate_ratio"],
                "column": csr["column_ratio"],
                "tripping": csr["tripping_ratio"],
            },
            "lambda_column": csr["lambda_column"],
            "paik_thayamballi_ratio_reference_only": paik_thayamballi_panel_strength(
                beta, csr["lambda_column"]
            ),
        })
    return ref


def analyse_section(
    section: GirderSection,
    n_sub: int = 4,
    n_steps: int = 120,
    loading: str = "auto",
    bending: str = "sagging",
    interaction_threshold: float = 0.6,
) -> SmithAnalysis:
    """Mature Smith analysis of one section (auto dispatch + references + mode)."""
    kind = section.kind
    auto = loading == "auto"
    if auto:
        if kind not in AUTO_LOADING:
            raise ValueError(f"no automatic loading rule for section kind {kind!r}")
        loading = AUTO_LOADING[kind]
    solver = SmithSolver(section, n_sub=n_sub, n_steps=n_steps, loading=loading,
                         bending=bending)
    result = solver.solve()
    mode = identify_failure_mode(
        result.elements, sigma_y=section.material.sigma_y0, z_na=result.z_NA_u,
        loading=loading, bending=bending, interaction_threshold=interaction_threshold,
    )
    if loading == "axial":
        reference = _panel_reference(section)
    else:
        reference = effective_width_ultimate_moment(section, bending=bending)
        reference["M_plastic_exact_MNm"] = section.plastic_moment() / 1e6
    return SmithAnalysis(section, result, mode, reference, loading=loading,
                         auto_loading=auto)


def analyse_with_convergence(
    section_factory,
    n_sub_levels: Sequence[int] = (2, 4, 8),
    loading: str = "auto",
    bending: str = "sagging",
    n_steps: int = 120,
) -> Dict[str, object]:
    """Grid-convergence study of ``M_u`` / ``sigma_u`` over a refinement ladder.

    ``section_factory(level)`` must return the section idealisation built at the
    requested refinement index ``level`` (e.g. number of web strips), so that
    only the discretisation - and not the physics - changes between levels.
    """
    values: List[float] = []
    stored: List[SmithAnalysis] = []
    for level in n_sub_levels:
        analysis = analyse_section(
            section_factory(int(level)), n_sub=int(level), loading=loading,
            bending=bending, n_steps=n_steps,
        )
        stored.append(analysis)
        values.append(analysis.result.M_u if analysis.loading == "bending"
                      else analysis.result.sigma_u_avg)
    study = grid_convergence_study(lambda i: values[n_sub_levels.index(i)],
                                   levels=n_sub_levels, ratio=2.0)
    finest = stored[-1]
    study.update({
        "quantity": "M_u [N m]" if finest.loading == "bending" else "sigma_u [Pa]",
        "loading": finest.loading,
        "bending": bending,
        "finest": finest.as_dict(),
        "finest_relative_to_extrapolated": (
            abs(values[-1] - study["extrapolated"]) / abs(study["extrapolated"])
            if study["extrapolated"] else float("inf")
        ),
    })
    return study