"""Generalised Smith-method progressive collapse: sections, benchmarks, handoff.

This package generalises the Smith (1977) progressive-collapse method from the
single-stiffened-panel setting of
:mod:`tensorphysics.elastoplastic_collapse.progressive_collapse` to **arbitrary
plated sections** and provides a benchmark + acceptance layer around it:

==========================  ==================================================
module                      content
==========================  ==================================================
:mod:`.section`             component model (plating strips, stiffener groups,
                            inclined plates) + rational fiberisation
                            (``n_fibers`` / ``n_web_fibers`` = mesh parameters)
:mod:`.solver`              incremental-curvature bending solver and uniform
                            axial panel solver, shared vectorised element-curve
                            kernel, failure-mode attribution
:mod:`.strength_curves`     element strength curves (von Karman / Faulkner
                            plating, Johnson-Ostenfeld columns) and the
                            independent CSR-type panel formula
:mod:`.convergence`         Richardson extrapolation and the grid-convergence
                            criterion (observed order, ``h -> 0`` residual)
:mod:`.benchmarks`          ISSC-2009-style double bottom, Paik-style ``beta``
                            series, box girder with closed-form limit-analysis
                            reference, **KCS real midship section** with an
                            ``Cm`` geometry gate
:mod:`.handoff`             load-capacity handoff gate (Smith ``M_u`` vs
                            wave-induced extreme bending moment)
:mod:`.acceptance`          the C1-C6 criterion runner and report builder
==========================  ==================================================

Status of the evidence
----------------------
* Bending/axial solvers, element curves and the mesh-convergence machinery are
  **REAL** (incremental equilibrium analysis, self-converged to 1% by
  Richardson extrapolation — see the acceptance report).
* Benchmark *geometries* are reconstructed parameter sets (declared per case);
  the literature experimental reference values are **not** reproduced offline
  and are reported as such (``NOT_SUPPORTED`` / ``WITHHELD``), never fabricated.
* The KCS section is a real ship's midship section in the sense that the
  principal dimensions and ``Cm`` are the published values and the structural
  arrangement (double bottom, hopper, deck strips, hatch coaming, single skin)
  is the ship's; the individual scantlings are representative values of the
  class, which is declared in the provenance block of every result.
"""

from .section import (  # noqa: F401
    STEEL_E,
    STEEL_NU,
    PlateStrip,
    SectionFiber,
    SectionModel,
    StiffenerGroup,
    stiffener_strength_caps,
)
from .solver import (  # noqa: F401
    AxialPanelResult,
    SmithSectionResult,
    SmithSectionSolver,
    axial_ultimate_strength,
    element_stress,
)
from .strength_curves import (  # noqa: F401
    csr_type_panel_strength,
    faulkner_plate_ratio,
    von_karman_plate_ratio,
)
from .convergence import (  # noqa: F401
    grid_convergence_criterion,
    mesh_convergence_study,
    richardson_extrapolate,
)
from .benchmarks import (  # noqa: F401
    KCSMidshipSection,
    box_girder,
    box_girder_plastic_moment,
    issc2009_double_bottom_unit,
    kcs_midship_section,
    kcs_particulars,
    paik_style_beta_series,
    paik_style_panel,
)
from .handoff import (  # noqa: F401
    kcs_girder,
    load_capacity_handoff,
    wave_demand,
)
from .acceptance import (  # noqa: F401
    run_acceptance,
    write_acceptance_report,
)

__all__ = [
    "STEEL_E",
    "STEEL_NU",
    "PlateStrip",
    "SectionFiber",
    "SectionModel",
    "StiffenerGroup",
    "stiffener_strength_caps",
    "AxialPanelResult",
    "SmithSectionResult",
    "SmithSectionSolver",
    "axial_ultimate_strength",
    "element_stress",
    "csr_type_panel_strength",
    "faulkner_plate_ratio",
    "von_karman_plate_ratio",
    "grid_convergence_criterion",
    "mesh_convergence_study",
    "richardson_extrapolate",
    "KCSMidshipSection",
    "box_girder",
    "box_girder_plastic_moment",
    "issc2009_double_bottom_unit",
    "kcs_midship_section",
    "kcs_particulars",
    "paik_style_beta_series",
    "paik_style_panel",
    "kcs_girder",
    "load_capacity_handoff",
    "wave_demand",
    "run_acceptance",
    "write_acceptance_report",
]