"""Typed interface contracts for the LBM/FEM/DEM three-field coupling loop.

An interface contract is the *statement* that makes an energy ledger checkable:
which quantity crosses the interface, in which direction, in which unit, on
which sign convention, and what the two solvers are each required to book.  If
the contract is not written down, an energy residual can only be compared
against a bookkeeping identity; with the contract written down the residual can
be attributed to a specific crossing.

The three crossings of the loop close a directed cycle over the three sides::

    fluid (LBM) --heat flux--> solid (FEM) --contact force--> particles (DEM)
        ^                                                          |
        +----------------------- drag power ----------------------+

Contract keys are stable identifiers used by the acceptance gates and by the
``verification.json`` record, so they must not be renamed silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

__all__ = [
    "ContractError",
    "Quantity",
    "InterfaceContract",
    "CONTRACTS",
    "CONTRACT_INDEX",
    "contract",
    "coupling_graph",
    "validate_contracts",
    "assert_reciprocity",
]


class ContractError(RuntimeError):
    """A transfer was booked in violation of its declared interface contract."""


class Quantity(str, Enum):
    """Physical quantity carried across an interface."""

    HEAT_RATE = "heat_rate"  # [W]
    FORCE = "force"  # [N]
    POWER = "power"  # [W]


@dataclass(frozen=True)
class InterfaceContract:
    """One declared crossing between two solvers.

    Attributes
    ----------
    key:
        Stable identifier (used by gates and records).
    source, target:
        Side names, matching the solver roles of the loop.
    quantity, unit:
        What is carried and in which unit.
    sign_convention:
        The one convention both sides must use; stated so that a sign error is
        a contract violation rather than a modelling choice.
    booked_by_source, booked_by_target:
        What each side is required to book.  For a conservative crossing the two
        entries are the same number (double entry); for a *reconstructed*
        crossing they are two independent discretisations of one physical
        quantity and their difference is the interface imbalance.
    conservation:
        The physical statement the pair of bookings must satisfy.
    exact:
        ``True`` when the contract requires an *identical* number on both sides
        at the exchange instant (reciprocity is machine-checkable with a
        tolerance), ``False`` when the two sides evaluate the same physical
        quantity independently (only a flux-continuity tolerance is meaningful).
    """

    key: str
    source: str
    target: str
    quantity: Quantity
    unit: str
    sign_convention: str
    booked_by_source: str
    booked_by_target: str
    conservation: str
    exact: bool

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "source": self.source,
            "target": self.target,
            "quantity": self.quantity.value,
            "unit": self.unit,
            "sign_convention": self.sign_convention,
            "booked_by_source": self.booked_by_source,
            "booked_by_target": self.booked_by_target,
            "conservation": self.conservation,
            "exact": self.exact,
        }


CONTRACTS: tuple[InterfaceContract, ...] = (
    InterfaceContract(
        key="lbm_fem_heat_flux",
        source="fluid",
        target="solid",
        quantity=Quantity.HEAT_RATE,
        unit="W",
        sign_convention=(
            "positive = heat flows from the fluid into the solid; the coupled "
            "face is the plate top row / lattice south row"
        ),
        booked_by_source=(
            "fluid books -C_f * injected: the scalar its Dirichlet wall removed "
            "or supplied, integrated over the lattice step"
        ),
        booked_by_target=(
            "solid removes the same heat rate as a nodal sink on its top row; "
            "the rate it can know at its own step is a degree-p reconstruction "
            "of the fluid's samples"
        ),
        conservation=(
            "the fluid's measured wall heat balance and the solid's one-sided "
            "surface flux are two discretisations of one flux: their difference "
            "is the interface imbalance (criterion: < 2 % L2)"
        ),
        exact=False,
    ),
    InterfaceContract(
        key="fem_dem_contact_force",
        source="solid",
        target="particles",
        quantity=Quantity.FORCE,
        unit="N",
        sign_convention=(
            "positive = compression: the particle is pushed along +y, the plate "
            "node is pushed along -y"
        ),
        booked_by_source=(
            "solid applies +f_node on its own top-row node from the contact "
            "force reconstructed out of the DEM's samples"
        ),
        booked_by_target=(
            "particles apply -f_node on the particle from the same law "
            "evaluated against the reconstructed plate surface position/velocity"
        ),
        conservation=(
            "Newton's third law at the exchange instant: |f_node + sum f_p| = 0 "
            "to round-off (reciprocity, machine-checkable), so the interface "
            "cannot create momentum"
        ),
        exact=True,
    ),
    InterfaceContract(
        key="dem_lbm_drag",
        source="particles",
        target="fluid",
        quantity=Quantity.POWER,
        unit="W",
        sign_convention=(
            "positive = kinetic energy leaves the particle; the carrier fluid is "
            "prescribed quiescent, so all of it becomes internal energy"
        ),
        booked_by_source=(
            "particles book gamma * |v - u|^2 * dt measured at every DEM step"
        ),
        booked_by_target=(
            "fluid books the degree-p reconstruction of the particle power "
            "history, integrated over its own lattice step, as a scalar source "
            "in the cell that contains the particle"
        ),
        conservation=(
            "the dissipated power must arrive as internal energy: the mismatch "
            "is the DEM->LBM transfer imbalance (part of the total residual)"
        ),
        exact=False,
    ),
)

CONTRACT_INDEX: Mapping[str, InterfaceContract] = {c.key: c for c in CONTRACTS}


def contract(key: str) -> InterfaceContract:
    try:
        return CONTRACT_INDEX[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ContractError(f"unknown interface contract {key!r}") from exc


def coupling_graph(contracts: Iterable[InterfaceContract] = CONTRACTS) -> dict[str, set[str]]:
    """Directed adjacency of the declared crossings."""
    graph: dict[str, set[str]] = {}
    for c in contracts:
        graph.setdefault(c.source, set()).add(c.target)
    return graph


def validate_contracts(contracts: Sequence[InterfaceContract] = CONTRACTS) -> None:
    """Structural audit of a contract set.

    Rejects: duplicate keys, self-loops, unknown units, empty booking
    statements, and a contract cycle that does not visit every side at least
    once (an open chain is not a *loop* and cannot close the energy books of a
    three-field system by itself).
    """
    if not contracts:
        raise ContractError("empty contract set")
    seen: set[str] = set()
    sides: set[str] = set()
    for c in contracts:
        if c.key in seen:
            raise ContractError(f"duplicate contract key {c.key!r}")
        seen.add(c.key)
        if c.source == c.target:
            raise ContractError(f"contract {c.key!r} couples a side to itself")
        if c.unit not in {"W", "N"}:
            raise ContractError(f"contract {c.key!r} has unknown unit {c.unit!r}")
        if not c.booked_by_source.strip() or not c.booked_by_target.strip():
            raise ContractError(f"contract {c.key!r} does not state both bookings")
        sides.update((c.source, c.target))
    covered = set().union(*coupling_graph(contracts).values()) | set(coupling_graph(contracts))
    if sides - covered:  # pragma: no cover - defensive
        raise ContractError(f"sides {sorted(sides - covered)} never appear as an endpoint")
    # every side must both produce and receive, otherwise the graph is a chain
    for side in sides:
        produces = any(c.source == side for c in contracts)
        receives = any(c.target == side for c in contracts)
        if not (produces and receives):
            raise ContractError(
                f"side {side!r} is not both a source and a target: the declared "
                "crossings form a chain, not a closed loop"
            )


def assert_reciprocity(key: str, source_value: float, target_value: float,
                       rtol: float = 1e-12, atol: float = 0.0) -> None:
    """Enforce the reciprocity clause of an ``exact`` contract.

    For an ``exact`` crossing the *same* number must be booked on both sides at
    the exchange instant.  ``source_value`` is the value the producer booked and
    ``target_value`` the value the receiver booked with the sign convention of
    the contract already applied (so an action-reaction pair is passed as
    ``+f`` and ``-(-f)``).
    """
    c = contract(key)
    if not c.exact:
        raise ContractError(
            f"contract {key!r} is a reconstructed crossing; reciprocity is not "
            "the right check (use the flux/energy continuity gate)"
        )
    scale = max(abs(source_value), abs(target_value), atol)
    mismatch = abs(source_value - target_value)
    if scale > 0.0 and mismatch > max(rtol * scale, atol):
        raise ContractError(
            f"contract {key!r} violated: source booked {source_value!r}, target "
            f"booked {target_value!r} (relative mismatch {mismatch / scale:.3e})"
        )