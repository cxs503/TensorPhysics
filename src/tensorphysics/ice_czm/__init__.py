"""
Ice-Steel Cohesive Zone Model (CZM)

Physics-based traction-separation law for ice-structure interface.
Calibrated against Frederking 1984 (Mode I) and Timco 1986 (Mode II).
Mixed-mode via Benzeggagh-Kenane criterion.

References:
    [1] Frederking, R.M.W. (1984). "Adhesion of ice to steel."
        Cold Regions Science and Technology, 8(3), 277-284.
    [2] Timco, G.W. (1986). "Shear strength of ice."
        Cold Regions Science and Technology, 12(2), 137-145.
    [3] Benzeggagh, M.L., Kenane, M. (1996). "Characterization of mixed mode
        delamination in carbon/epoxy laminates." Composites Science and Technology.
"""

from .traction_separation import (
    BilinearTractionSeparation,
    TractionSeparationState,
)
from .calibration import (
    IceSteelCalibration,
    Frederking1984,
    Timco1986,
    calibrate_ice_steel_czm,
)
from .bk_mixed_mode import (
    BenzeggaghKenaneCriterion,
    MixedModeCZM,
)
from .interface_element import (
    ZeroThicknessInterfaceElement,
    InterfaceElementState,
)
from .dem_mapping import (
    AdfreezeBondMapper,
    AdfreezeBondState,
)

__all__ = [
    "BilinearTractionSeparation",
    "TractionSeparationState",
    "IceSteelCalibration",
    "Frederking1984",
    "Timco1986",
    "calibrate_ice_steel_czm",
    "BenzeggaghKenaneCriterion",
    "MixedModeCZM",
    "ZeroThicknessInterfaceElement",
    "InterfaceElementState",
    "AdfreezeBondMapper",
    "AdfreezeBondState",
]
