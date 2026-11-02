"""Physics model contracts: declared-method naming gate + kernel registry.

Stage-2 deliverable of the 2026-11-02 owner task book (统一阻力计算链路):
every physics model exposed by TensorPhysics must declare an explicit
contract -- inputs with units, method components actually implemented, and
reference bands with provenance.  The naming gate rejects the audited
mislabel where an ITTC-1957 + empirical-residual composition was published
under the name "Holtrop-Mennen" (audit m2086, crosscheck kernel A..J).

Design rules (from tensorphysics.research_contracts precedent):
- frozen dataclasses, validation in __post_init__ (fail at construction),
- explicit SI units in field names,
- no silent defaults that change physics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

SCHEMA_VERSION = 1


class PhysicsContractError(ValueError):
    """Raised when a physics model contract is self-inconsistent."""


# Roles that only the *full* Holtrop & Mennen (1984) method may claim.
_HM1984_REQUIRED_ROLES = frozenset({
    "friction_ittc1957",
    "form_factor_holtrop",
    "wave_holtrop1984",
    "bulbous_bow_holtrop1984",
    "transom_holtrop1984",
    "correlation_allowance_holtrop1984",
})


@dataclass(frozen=True)
class InputSpec:
    name: str
    unit: str
    required: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise PhysicsContractError("input spec needs name and unit")
        for ch in self.name:
            if not (ch.isalnum() or ch == "_"):
                raise PhysicsContractError(f"bad input name {self.name!r}")


@dataclass(frozen=True)
class MethodComponent:
    role: str
    formula: str
    provenance: str  # citation or "fit: <dataset>" -- never empty

    def __post_init__(self) -> None:
        if not self.role or not self.formula or not self.provenance:
            raise PhysicsContractError(
                f"component {self.role!r} needs role, formula and provenance"
            )


@dataclass(frozen=True)
class ReferenceBand:
    """A benchmark reference value with explicit provenance class.

    provenance:
      EXPERIMENTAL -- independently measured (tank test / published data)
      DERIVED      -- back-computed from another published quantity
      WITHHELD     -- source withheld or unverifiable; NOT admissible as an
                      acceptance anchor (audit m2087: 8 of 12 benchmark rows)
    """
    case_id: str
    quantity: str
    value: float
    unit: str
    provenance: str
    tolerance_rel: float
    admissible_for_judgment: bool = True

    def __post_init__(self) -> None:
        if self.provenance not in ("EXPERIMENTAL", "DERIVED", "WITHHELD"):
            raise PhysicsContractError(
                f"provenance must be EXPERIMENTAL|DERIVED|WITHHELD, "
                f"got {self.provenance!r}"
            )
        if self.provenance == "WITHHELD" and self.admissible_for_judgment:
            raise PhysicsContractError(
                f"{self.case_id}: WITHHELD reference is not admissible for "
                "judgment -- refusing to register a self-certifying anchor"
            )
        if self.tolerance_rel <= 0:
            raise PhysicsContractError("tolerance_rel must be positive")


@dataclass(frozen=True)
class PhysicsModelContract:
    key: str                      # e.g. "resistance.suboff_shared_core"
    declared_method: str          # human-facing label -- the naming gate target
    inputs: Tuple[InputSpec, ...]
    components: Tuple[MethodComponent, ...]
    reference_kernel: str         # dotted path inside tensorphysics
    references: Tuple[ReferenceBand, ...] = field(default_factory=tuple)
    aliases: Tuple[str, ...] = () # audited production kernels mapped here
    limitations: str = ""

    def __post_init__(self) -> None:
        if not self.key or "." not in self.key:
            raise PhysicsContractError(f"contract key {self.key!r} needs '<domain>.<name>'")
        label = self.declared_method.lower()
        roles = {c.role for c in self.components}
        is_hm_name = "holtrop" in label and ("mennen" in label or "hm1984" in label
                                             or "hm-1984" in label or "hm 1984" in label)
        if is_hm_name and not _HM1984_REQUIRED_ROLES <= roles:
            missing = sorted(_HM1984_REQUIRED_ROLES - roles)
            raise PhysicsContractError(
                f"naming violation: {self.key!r} declares {self.declared_method!r} "
                f"but is missing full-method components {missing}. Rename to the "
                "actual composition (e.g. 'ITTC-1957 friction + SUBOFF residual fit') "
                "or implement the full Holtrop & Mennen (1984) method."
            )
        if not self.components:
            raise PhysicsContractError(f"{self.key}: at least one component required")
        if not self.reference_kernel:
            raise PhysicsContractError(f"{self.key}: reference_kernel required")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CONTRACTS: Dict[str, PhysicsModelContract] = {}
_ALIASES: Dict[str, str] = {}


def register(contract: PhysicsModelContract) -> PhysicsModelContract:
    if contract.key in _CONTRACTS:
        raise PhysicsContractError(f"duplicate contract key {contract.key}")
    _CONTRACTS[contract.key] = contract
    for a in contract.aliases:
        if a in _ALIASES and _ALIASES[a] != contract.key:
            raise PhysicsContractError(
                f"alias {a!r} already bound to {_ALIASES[a]!r}"
            )
        _ALIASES[a] = contract.key
    return contract


def get_contract(key_or_alias: str) -> PhysicsModelContract:
    key = _ALIASES.get(key_or_alias, key_or_alias)
    try:
        return _CONTRACTS[key]
    except KeyError:
        raise PhysicsContractError(
            f"unknown physics contract {key_or_alias!r}; "
            f"registered: {sorted(_CONTRACTS)}"
        ) from None


def all_contracts() -> Dict[str, PhysicsModelContract]:
    return dict(_CONTRACTS)


# ---------------------------------------------------------------------------
# Canonical contracts for the audited resistance chain
# ---------------------------------------------------------------------------

SUBOFF_SHARED_CORE_LABEL = "ITTC-1957 friction + SUBOFF-calibrated residual fit"

register(PhysicsModelContract(
    key="resistance.suboff_shared_core",
    declared_method=SUBOFF_SHARED_CORE_LABEL,
    inputs=(
        InputSpec("length_m", "m"),
        InputSpec("beam_m", "m"),
        InputSpec("cb", "-"),
        InputSpec("speed_ms", "m/s"),
        InputSpec("rho_kg_m3", "kg/m^3"),
        InputSpec("nu_m2_s", "m^2/s"),
        InputSpec("wetted_surface_m2", "m^2",
                  description="REQUIRED -- no silent empirical fallback"),
        InputSpec("form_factor", "-", required=False,
                  description="(1+k); 1.0 means no form correction"),
    ),
    components=(
        MethodComponent("friction_ittc1957", "Cf = 0.075/(log10(Re)-2)^2",
                        "ITTC-1957 correlation line"),
        MethodComponent("residual_empirical", "Cr = 0.002*(1+2*Cb)*Fn^2.5, Fn>0.05",
                        "fit: SUBOFF-like axisymmetric bodies, no published provenance"),
    ),
    reference_kernel="tensorphysics.physics_kernels_resistance.shared_core_resistance",
    references=(
        ReferenceBand("SUBOFF-01", "ct", 0.003461889103158906, "-",
                      provenance="DERIVED",
                      # six audited kernels agree bitwise on this composition
                      tolerance_rel=1e-9),
        ReferenceBand("SUBOFF-01", "ct", 0.0038, "-",
                      provenance="EXPERIMENTAL", tolerance_rel=0.15),
    ),
    aliases=("A_endpoint_inline", "B_ship_resistance", "C_services_resistance_core",
             "D_api_domain_resistance_core", "E_resistance_model",
             "I_ship_resistance_k115", "J_endpoint_k115"),
    limitations="Residual fit calibrated on axisymmetric SUBOFF family; not a "
                "general merchant-ship method. Do not relabel as Holtrop-Mennen.",
))

register(PhysicsModelContract(
    key="resistance.holtrop1984_full",
    declared_method="Holtrop & Mennen (1984) full method",
    inputs=(
        InputSpec("length_m", "m"), InputSpec("beam_m", "m"),
        InputSpec("draft_m", "m"), InputSpec("cb", "-"),
        InputSpec("speed_ms", "m/s"), InputSpec("rho_kg_m3", "kg/m^3"),
        InputSpec("nu_m2_s", "m^2/s"),
        InputSpec("wetted_surface_m2", "m^2"),
        InputSpec("lcb_from_midship_percent_l", "%L"),
        InputSpec("abt_m2", "m^2", required=False,
                  description="immersed transverse bulb area, 0 = none"),
        InputSpec("at_m2", "m^2", required=False,
                  description="immersed transom area, 0 = none"),
    ),
    components=(
        MethodComponent("friction_ittc1957", "Cf = 0.075/(log10(Re)-2)^2",
                        "ITTC-1957 correlation line"),
        MethodComponent("form_factor_holtrop", "(1+k) via Holtrop 1984 hull-form",
                        "Holtrop & Mennen (1984), Eq. (2)-(8)"),
        MethodComponent("wave_holtrop1984", "Rw, h1/h2 wave-making split",
                        "Holtrop & Mennen (1984), Eq. (9)-(24)"),
        MethodComponent("bulbous_bow_holtrop1984", "Rb bulb resistance",
                        "Holtrop & Mennen (1984), Eq. (25)-(27)"),
        MethodComponent("transom_holtrop1984", "Rtr transom resistance",
                        "Holtrop & Mennen (1984), Eq. (28)-(29)"),
        MethodComponent("correlation_allowance_holtrop1984", "Ra = 0.5*rho*S*Ca*V^2",
                        "Holtrop & Mennen (1984), Eq. (30)-(31)"),
    ),
    reference_kernel="tensorphysics.physics_kernels_holtrop1984.holtrop1984_resistance",
    aliases=("F_resistance_holtrop", "G_holtrop1984"),
    limitations="Recovered deploy-tree copy (main lost it in merge 2f3caffb). "
                "Documented simplifications: rhow=1 above Fn~0.4, LR~=L, no "
                "low-length Ca correction; not valid for slender axisymmetric bodies.",
))

register(PhysicsModelContract(
    key="friction.ittc1957",
    declared_method="ITTC-1957 model-ship correlation line",
    inputs=(InputSpec("re", "-"),),
    components=(MethodComponent("friction_ittc1957",
                                "Cf = 0.075/(log10(Re)-2)^2", "ITTC-1957"),),
    reference_kernel="tensorphysics.physics_kernels_resistance.ittc_1957_cf",
))

register(PhysicsModelContract(
    key="friction.schoenherr1932",
    declared_method="Schoenherr (1932) friction line",
    inputs=(InputSpec("re", "-"),),
    components=(MethodComponent("friction_schoenherr",
                                "0.242/sqrt(Cf) = log10(Re*Cf)", "Schoenherr 1932"),),
    reference_kernel="tensorphysics.physics_kernels_resistance.schoenherr_1932_cf",
    limitations="Historical independent cross-check of ITTC-1957, not an "
                "acceptance anchor by itself.",
))

register(PhysicsModelContract(
    key="wetted_surface.axisymmetric_empirical",
    declared_method="Axisymmetric empirical wetted-surface estimate",
    inputs=(InputSpec("length_m", "m"), InputSpec("beam_m", "m"),
            InputSpec("cb", "-")),
    components=(MethodComponent("wetted_surface_empirical",
                                "S = pi*B*L*(0.5 + 0.25*Cb)", "internal fit"),),
    reference_kernel="tensorphysics.physics_kernels_resistance."
                     "axisymmetric_wetted_surface",
    references=(
        ReferenceBand("KVLCC2", "wetted_surface_m2", 40958.0, "m^2",
                      provenance="DERIVED",
                      # S(from this fit) vs published 27194 m^2 -> +50.6% bias
                      tolerance_rel=0.51),
        ReferenceBand("KVLCC2", "wetted_surface_m2", 27194.0, "m^2",
                      provenance="EXPERIMENTAL", tolerance_rel=0.01,
                      admissible_for_judgment=False),
    ),
    aliases=("D2_domain_same_wetted_surface",),
    limitations="Bias measured -30.4% (cylinder) / -8.7% (prolate spheroid) on "
                "closed forms; KVLCC2 +50.6% vs published. Callers MUST declare "
                "this derivation; acceptance anchors require a geometry-validated S.",
))
