"""Cross-repository interface contracts (P27W3 integration surface).

This package is the *receiving* half of three contracts that TensorPhysics
signed with its neighbours.  Each contract has a versioned JSON payload, an
explicit unit/normalisation register, and a fail-closed validator: a payload
that does not declare its conventions is rejected rather than silently
interpreted.

============================  ============================  ==================
contract                      peer (sender)                 consumed by
============================  ============================  ==================
``rao-v1``                    ship-performance-platform     :mod:`.rao`
                              ``SeakeepingPhysics`` STF     (response
                              strip theory                  reconstruction)
``stress-psd-v1``             ship-performance-platform     :mod:`.spectral`
                              ``backend.modules.             (spectral fatigue)
                              spectral_fatigue``
``ice-bond-v1``               TensorDEM                     :mod:`.ice_bond`
                              ``IceBeamSpec`` lattice       (cohesive bond
                              bond model                    strength)
============================  ============================  ==================

Evidence labels used in this package follow the repository convention:
REAL = independent implementation exercised on the contract payload;
CONTRACT = the quantity is fixed by the payload itself (transport fidelity);
WITHHELD = not decidable from the available evidence.
"""

from .interchange import (
    ContractViolation,
    IceBondTransfer,
    RAOTransfer,
    StressSpectrumTransfer,
    load_golden,
)
from .ice_bond import (
    BondMapping,
    cohesive_energy_error,
    dem_bond_to_cohesive,
    dem_failure_load,
    tensorphysics_failure_load,
)
from .rao import (
    ResponseComparison,
    compare_responses,
    geometry_audit,
    long_wave_limit,
    pchip_interp,
    reconstruct_response,
    sectional_added_mass_crosscheck,
)
from .spectral import (
    DamageComparison,
    compare_damage,
    damage_by_method,
    moments_of,
)

__all__ = [
    "BondMapping",
    "ContractViolation",
    "DamageComparison",
    "IceBondTransfer",
    "RAOTransfer",
    "ResponseComparison",
    "StressSpectrumTransfer",
    "cohesive_energy_error",
    "compare_damage",
    "compare_responses",
    "geometry_audit",
    "damage_by_method",
    "dem_bond_to_cohesive",
    "dem_failure_load",
    "load_golden",
    "long_wave_limit",
    "moments_of",
    "pchip_interp",
    "reconstruct_response",
    "sectional_added_mass_crosscheck",
    "tensorphysics_failure_load",
]