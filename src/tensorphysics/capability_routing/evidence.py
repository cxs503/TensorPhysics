"""Evidence taxonomy and admissibility policy for capability-aware routing.

The routing layer never decides *by name* whether a solver may run: it asks the
capability ledger for the evidence grade of the (capability, solver) pair and
compares it against an explicit admissibility policy.  Everything here is data
and ordinal arithmetic -- there is no capability name, axis name or domain word
in this module.

Grades are the platform vocabulary (REAL / MOCK / STUB / WITHHELD /
NOT_SUPPORTED) ordered by *how much verified evidence* stands behind the
implementation:

===============  =====  ====================================================
grade            rank   meaning
===============  =====  ====================================================
REAL                4   validated against experiment / reference solution
WITHHELD            3   implemented, evidence insufficient or pending
MOCK                2   hard-coded or randomised placeholder result
STUB                1   shell: signature only, no numerics
NOT_SUPPORTED       0   capability does not exist
===============  =====  ====================================================

Fail-closed rules enforced here:

* an unknown grade string raises :class:`UnknownEvidenceLevel` (a typo can never
  silently demote a solver to "admissible");
* an unknown tier name raises :class:`PolicyError`;
* ``NOT_SUPPORTED`` is admissible in *no* tier -- the invariant is encoded once
  in :meth:`AdmissibilityPolicy.admits` and checked exhaustively in tests;
* ``MOCK``/``STUB`` are admissible only in the ``development`` tier *and* only
  when the caller passes ``allow_mock=True`` (explicit consent, recorded in the
  routing decision as ``mock_consumed``).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "EvidenceLevel",
    "UnknownEvidenceLevel",
    "PolicyError",
    "AdmissibilityPolicy",
    "TIER_ADMISSIBLE",
]


class UnknownEvidenceLevel(ValueError):
    """A grade string outside the closed platform vocabulary."""


class EvidenceLevel(str, Enum):
    """Platform evidence grading of one (capability, solver) pair."""

    REAL = "REAL"
    WITHHELD = "WITHHELD"
    MOCK = "MOCK"
    STUB = "STUB"
    NOT_SUPPORTED = "NOT_SUPPORTED"

    @property
    def rank(self) -> int:
        """Higher is better; the total order drives fallback selection."""
        return _RANK[self]

    @property
    def verified(self) -> bool:
        """True only for grades backed by an executed verification case."""
        return self is EvidenceLevel.REAL

    @property
    def fabricated(self) -> bool:
        """True for grades whose numerical output is not produced by physics."""
        return self is EvidenceLevel.MOCK or self is EvidenceLevel.STUB

    @classmethod
    def parse(cls, value: "EvidenceLevel | str") -> "EvidenceLevel":
        if isinstance(value, EvidenceLevel):
            return value
        if not isinstance(value, str):
            raise UnknownEvidenceLevel(f"evidence grade must be a string, got {type(value).__name__}")
        key = value.strip().upper()
        try:
            return cls(key)
        except ValueError:
            raise UnknownEvidenceLevel(
                f"unknown evidence grade {value!r}; allowed: {[g.value for g in cls]}"
            ) from None


_RANK = {
    EvidenceLevel.REAL: 4,
    EvidenceLevel.WITHHELD: 3,
    EvidenceLevel.MOCK: 2,
    EvidenceLevel.STUB: 1,
    EvidenceLevel.NOT_SUPPORTED: 0,
}

#: Grade order from best to worst evidence (deterministic, used for ranking).
GRADE_ORDER: tuple[EvidenceLevel, ...] = tuple(
    sorted(EvidenceLevel, key=lambda g: -g.rank)
)

#: Admissible grades per tier.  ``NOT_SUPPORTED`` is absent from every tier by
#: construction -- that is the hard fail-closed invariant of this module.
TIER_ADMISSIBLE: dict[str, frozenset[EvidenceLevel]] = {
    "production": frozenset({EvidenceLevel.REAL}),
    "staged": frozenset({EvidenceLevel.REAL, EvidenceLevel.WITHHELD}),
    "development": frozenset(
        {EvidenceLevel.REAL, EvidenceLevel.WITHHELD, EvidenceLevel.MOCK, EvidenceLevel.STUB}
    ),
}


class PolicyError(ValueError):
    """An admissibility policy that cannot be interpreted must not run."""


@dataclass(frozen=True)
class AdmissibilityPolicy:
    """Which evidence grades may be routed, and in which operating tier.

    ``tier``         production | staged | development (see ``TIER_ADMISSIBLE``)
    ``allow_mock``   explicit consent to consume placeholder grades; only legal
                     in the ``development`` tier
    ``min_rank``     optional stricter floor (e.g. 4 == REAL only) that is
                     intersected with the tier set
    ``preference``   ``evidence_first`` (default: rank decides, the requested
                     solver only breaks ties) or ``preferred_first`` (the
                     requested solver is examined first, then the evidence order)
    """

    tier: str = "production"
    allow_mock: bool = False
    min_rank: "int | None" = None
    preference: str = "evidence_first"

    def __post_init__(self) -> None:
        if self.tier not in TIER_ADMISSIBLE:
            raise PolicyError(
                f"unknown tier {self.tier!r}; allowed: {sorted(TIER_ADMISSIBLE)}"
            )
        if not isinstance(self.allow_mock, bool):
            raise PolicyError("allow_mock must be a bool")
        if self.allow_mock and self.tier != "development":
            raise PolicyError(
                f"allow_mock=True requires tier='development', got tier={self.tier!r}"
            )
        if self.min_rank is not None:
            if isinstance(self.min_rank, bool) or not isinstance(self.min_rank, int):
                raise PolicyError("min_rank must be an int or None")
            if self.min_rank < 0 or self.min_rank > 4:
                raise PolicyError(f"min_rank out of range [0, 4]: {self.min_rank}")
        if self.preference not in ("evidence_first", "preferred_first"):
            raise PolicyError(
                f"unknown preference mode {self.preference!r}; "
                "allowed: ['evidence_first', 'preferred_first']"
            )

    @property
    def admissible(self) -> frozenset[EvidenceLevel]:
        """Effective admissible grade set (tier set, mock consent, rank floor)."""
        base = TIER_ADMISSIBLE[self.tier]
        if not self.allow_mock:
            base = frozenset(g for g in base if not g.fabricated)
        if self.min_rank is not None:
            base = frozenset(g for g in base if g.rank >= self.min_rank)
        return base

    def admits(self, level: "EvidenceLevel | str") -> bool:
        """Fail-closed membership test: unparseable grades are never admitted."""
        try:
            parsed = EvidenceLevel.parse(level)
        except UnknownEvidenceLevel:
            return False
        if parsed is EvidenceLevel.NOT_SUPPORTED:
            return False
        return parsed in self.admissible

    def as_dict(self) -> dict:
        return {
            "tier": self.tier,
            "allow_mock": self.allow_mock,
            "min_rank": self.min_rank,
            "preference": self.preference,
            "admissible": sorted(g.value for g in self.admissible),
        }