"""Component-level structural redundancy (RIF) of ice-class hull girders.

Campaign node **p351w1_n08** - "冰区船体结构冗余度指标 RIF 与破损后极限承载力".

The package answers the question *which single-component failure still leaves an
ice-class hull girder above 80 % of its intact ultimate bending capacity*, with
a component-level failure spectrum built on the verified elastoplastic
progressive-collapse kernel of :mod:`tensorphysics.elastoplastic_collapse`.

Contents
--------
``section``     component idealisation of the P350W1-N03 plate-beam combination
                closed by a double-hull side, plus a stand-alone double-hull
                side section, with independent geometry validation.
``elements``    element strength laws: plate load-end shortening (re-used from
                ``elastoplastic_collapse``), J2 uniaxial law, tripping stress,
                bare-strut (debonded stiffener) stress, Johnson-Ostenfeld.
``damage``      the three component failure modes (plate-band tearing /
                stiffener tripping / loss of composite action) and the mapping
                onto element properties.
``smith``       incremental-curvature section-level Smith solver -> M(kappa),
                M_u, and the elastic/plastic consistency checks.
``redundancy``  RIF matrix, worst-case ranking, redundancy set, sensitivity
                (Spearman rho of RIF vs failure position) and the
                discretisation convergence study.
``reference``   independent closed-form (Faulkner / Paik & Thayamballi
                simplified method) residual-strength benchmark.

Evidence level: **REAL** algorithms (no hard-coded or random values); the
benchmark comparison is method-level (published closed-form), which is declared
in the report.
"""

from .damage import (  # noqa: F401
    MODES,
    MODE_DESCRIPTION,
    DamageSpec,
    DamagedPlate,
    DamagedSection,
    DamagedStiffener,
    apply_damage,
)
from .elements import (  # noqa: F401
    bare_strut_stress,
    effective_plate_buckling_stress,
    faulkner_plate_strength_ratio,
    johnson_ostenfeld,
    plate_buckling_stress,
    plate_element_stress,
    strut_element_stress,
    tripping_stress,
)
from .reference import (  # noqa: F401
    benchmark_vs_smith,
    faulkner_plate_strength,
    simplified_ultimate_moment,
)
from .redundancy import (  # noqa: F401
    RIFEntry,
    convergence_study,
    most_dangerous,
    negative_control,
    redundancy_set,
    rif_matrix,
    sensitivity_analysis,
    spearman_rho,
    worst_case_by_component,
)
from .section import (  # noqa: F401
    DEFAULT_MATERIAL,
    HullSection,
    PlateStrip,
    StiffenerGroup,
    double_hull_side_section,
    p350w1_n03_combined_section,
)
from .smith import (  # noqa: F401
    SmithResult,
    SmithSectionSolver,
    ultimate_moment,
)

__all__ = [
    "MODES",
    "MODE_DESCRIPTION",
    "DamageSpec",
    "DamagedPlate",
    "DamagedStiffener",
    "DamagedSection",
    "apply_damage",
    "bare_strut_stress",
    "effective_plate_buckling_stress",
    "faulkner_plate_strength_ratio",
    "johnson_ostenfeld",
    "plate_buckling_stress",
    "plate_element_stress",
    "strut_element_stress",
    "tripping_stress",
    "faulkner_plate_strength",
    "simplified_ultimate_moment",
    "benchmark_vs_smith",
    "RIFEntry",
    "rif_matrix",
    "worst_case_by_component",
    "most_dangerous",
    "redundancy_set",
    "negative_control",
    "sensitivity_analysis",
    "convergence_study",
    "spearman_rho",
    "DEFAULT_MATERIAL",
    "HullSection",
    "PlateStrip",
    "StiffenerGroup",
    "p350w1_n03_combined_section",
    "double_hull_side_section",
    "SmithResult",
    "SmithSectionSolver",
    "ultimate_moment",
]