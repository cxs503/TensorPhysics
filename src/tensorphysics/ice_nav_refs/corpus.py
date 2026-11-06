"""sha256-anchored reference corpus for ice-navigation performance validation.

Why this module lives in TensorPhysics
=====================================
TensorPhysics owns *scientific acceptance and reference validation*: deciding
whether a computed number is acceptable against an external datum, and carrying
the provenance of that datum.  The reference corpus below is exactly that
artefact -- a set of frozen, content-addressed reference values that the
ice-navigation campaign nodes (resistance / propulsion / structural ice load)
consume from ``pytest`` instead of re-typing literature numbers inside tests.
It is *not* a ship-design feature (those belong to the domain ship platform)
and it is *not* a solver kernel.

Corpus layout on disk
=====================
``benchmarks/ice_nav_refs/``

* ``assets/<family>.json``  -- one file per source family, each
  ``{"schema", "family", "assets": [asset, ...]}``.
* ``MANIFEST.json``         -- sha256 of every asset file plus an aggregate
  digest over ``(relative path, sha256)`` pairs.

Each *asset* carries the four mandatory fields
``source_citation``, ``values``, ``tolerance``, ``applicable_regime`` plus the
audit metadata ``asset_id``, ``family``, ``evidence_tier``, ``role``,
``primary_source_verified``, ``verification_note`` and ``value_sha256`` (a
content address of ``{asset_id, values, tolerance}`` so that a consumer can
pin the *numbers* independently of the file that stores them).

Evidence tiers (what a consumer is allowed to conclude)
======================================================
``published_formula``
    Closed-form expression published in the cited work; the recorded value is
    that expression evaluated at the documented reference condition.  Anyone
    can re-derive it, so the tolerance can be tight (pure arithmetic).
``repo_transcription``
    A value produced by the repository's own documented transcription of a
    cited formula.  Use it as a *regression anchor* (drift detection), not as
    independent physical truth.
``derived_from_rule_form``
    Rule-form pressure law (power law) evaluated with a reference pressure that
    is itself a nominal engineering value; tolerances are wide.
``literature_band``
    A published scatter band, recorded as lower/nominal/upper.
``standards_clause``
    Qualitative/numeric requirement transcribed from a classification-society
    or IMO instrument clause.

Roles
=====
``accuracy_reference``
    The number may be used to *judge* a computation (the error budget is
    meaningful).
``regression_anchor``
    The number may only be used to *detect drift* in this repository's own
    implementation; it is not an external truth and must never be quoted as
    validation evidence on its own.

``primary_source_verified`` is ``False`` for every asset whose value could not
be checked against the primary document inside the build environment (no access
to paywalled standards / monographs).  Consumers must propagate that flag into
their own evidence grading; see ``docs/ICE_NAV_REFS_CORPUS.md``.

Usage
-----
>>> from tensorphysics.ice_nav_refs import load_corpus
>>> corpus = load_corpus()                      # doctest: +SKIP
>>> verdict = corpus.check("LIN_C_CRUSH_PHI30", {"C_c": 2.1558})  # doctest: +SKIP
>>> verdict["pass"]                                            # doctest: +SKIP
True
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SCHEMA",
    "REQUIRED_FIELDS",
    "EVIDENCE_TIERS",
    "ROLES",
    "TOLERANCE_MODES",
    "CorpusError",
    "MissingFieldError",
    "MissingCitationError",
    "CorpusIntegrityError",
    "ToleranceError",
    "Tolerance",
    "ReferenceAsset",
    "ReferenceCorpus",
    "canonical_json",
    "sha256_digest",
    "value_digest",
    "asset_from_dict",
    "default_root",
    "load_corpus",
    "verify_manifest",
]

SCHEMA = "ice_nav_refs.v1"

#: The four fields every reference asset must carry.
REQUIRED_FIELDS: Tuple[str, ...] = (
    "source_citation",
    "values",
    "tolerance",
    "applicable_regime",
)

EVIDENCE_TIERS = frozenset(
    {
        "published_formula",
        "repo_transcription",
        "derived_from_rule_form",
        "literature_band",
        "standards_clause",
    }
)

ROLES = frozenset({"accuracy_reference", "regression_anchor"})

TOLERANCE_MODES = frozenset({"relative", "absolute", "band", "exact"})

#: Shortest acceptable citation; anything shorter is treated as a placeholder.
CITATION_MIN_CHARS = 12

#: Strings that look like a citation but carry no provenance.  A corpus entry
#: carrying one of these is rejected exactly like a missing citation.
PLACEHOLDER_CITATIONS = frozenset(
    {
        "",
        "-",
        "?",
        "n/a",
        "n.a.",
        "na",
        "none",
        "null",
        "nil",
        "tba",
        "tbd",
        "todo",
        "fixme",
        "unknown",
        "see above",
        "reference",
        "literature",
        "textbook",
    }
)

#: Keys every ``applicable_regime`` block must define.
REGIME_REQUIRED_KEYS: Tuple[str, ...] = ("ice_regime", "ice_thickness_m")


class CorpusError(Exception):
    """Base class for every corpus-level failure."""


class MissingFieldError(CorpusError):
    """A mandatory asset field is absent or empty."""


class MissingCitationError(CorpusError):
    """A reference asset carries no usable ``source_citation``."""


class CorpusIntegrityError(CorpusError):
    """A file or value digest does not match its recorded sha256 anchor."""


class ToleranceError(CorpusError):
    """A tolerance block is missing, malformed or inconsistent."""


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding used for every digest in this corpus."""
    return (
        json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def sha256_digest(payload: bytes) -> str:
    """Hex sha256 of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def value_digest(asset_id: str, values: Mapping[str, Any], tolerance: Any) -> str:
    """Content address of the numeric payload of an asset.

    Deliberately excludes prose (citation, notes): editing wording must not
    break a consumer's pinned reference value, while editing a number must.
    """
    payload = {
        "asset_id": asset_id,
        "values": dict(values),
        "tolerance": tolerance if isinstance(tolerance, dict) else tolerance.to_dict(),
    }
    return sha256_digest(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _require_text(value: Any, field_name: str) -> str:
    if value is None:
        raise MissingFieldError(f"missing required field {field_name!r}")
    if not isinstance(value, str):
        raise MissingFieldError(f"field {field_name!r} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise MissingFieldError(f"field {field_name!r} is empty")
    return text


def _check_citation(citation: Any) -> str:
    """Validate a ``source_citation`` and reject placeholder provenance."""
    if citation is None:
        raise MissingCitationError("reference asset has no source_citation (None)")
    if not isinstance(citation, str):
        raise MissingCitationError(
            f"source_citation must be a string, got {type(citation).__name__}"
        )
    text = citation.strip()
    if text.lower() in PLACEHOLDER_CITATIONS:
        raise MissingCitationError(
            f"source_citation {text!r} is a placeholder, not provenance"
        )
    if len(text) < CITATION_MIN_CHARS:
        raise MissingCitationError(
            f"source_citation {text!r} is shorter than {CITATION_MIN_CHARS} characters"
        )
    if not any(ch.isdigit() for ch in text):
        # Every real citation in this corpus carries a year/volume/clause
        # number; a bare name is not checkable provenance.
        raise MissingCitationError(
            f"source_citation {text!r} carries no year/volume/clause identifier"
        )
    return text


@dataclass(frozen=True)
class Tolerance:
    """Acceptance tolerance attached to a reference value.

    ``mode``
        * ``relative`` -- ``|computed - nominal| / |nominal| <= value``
        * ``absolute`` -- ``|computed - nominal| <= value``
        * ``band``     -- ``lower <= computed <= upper``
        * ``exact``    -- categorical equality (``value`` must be 0.0)
    """

    mode: str
    value: float
    lower: Optional[float] = None
    upper: Optional[float] = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.mode not in TOLERANCE_MODES:
            raise ToleranceError(
                f"tolerance mode {self.mode!r} not in {sorted(TOLERANCE_MODES)}"
            )
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ToleranceError(f"tolerance value must be a number, got {self.value!r}")
        if not math.isfinite(float(self.value)):
            raise ToleranceError(f"tolerance value must be finite, got {self.value!r}")
        if self.mode == "exact":
            if float(self.value) != 0.0:
                raise ToleranceError("exact tolerance must have value 0.0")
        elif float(self.value) <= 0.0:
            raise ToleranceError(f"tolerance value must be > 0, got {self.value!r}")
        if self.mode == "band":
            if self.lower is None or self.upper is None:
                raise ToleranceError("band tolerance requires lower and upper")
            if float(self.lower) > float(self.upper):
                raise ToleranceError(
                    f"band tolerance has lower {self.lower} > upper {self.upper}"
                )

    @classmethod
    def from_dict(cls, payload: Any) -> "Tolerance":
        if payload is None:
            raise ToleranceError("missing tolerance block")
        if isinstance(payload, Tolerance):
            return payload
        if not isinstance(payload, Mapping):
            raise ToleranceError(f"tolerance must be a mapping, got {type(payload).__name__}")
        if "mode" not in payload:
            raise ToleranceError("tolerance block has no 'mode'")
        if "value" not in payload:
            raise ToleranceError("tolerance block has no 'value'")
        return cls(
            mode=str(payload["mode"]),
            value=float(payload["value"]),
            lower=None if payload.get("lower") is None else float(payload["lower"]),
            upper=None if payload.get("upper") is None else float(payload["upper"]),
            note=str(payload.get("note", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"mode": self.mode, "value": self.value}
        if self.lower is not None:
            out["lower"] = self.lower
        if self.upper is not None:
            out["upper"] = self.upper
        if self.note:
            out["note"] = self.note
        return out

    def describe(self) -> str:
        if self.mode == "band":
            return f"band [{self.lower}, {self.upper}]"
        if self.mode == "exact":
            return "exact equality"
        unit = "" if self.mode == "relative" else " (absolute)"
        return f"+-{self.value:g}{unit} {self.mode}"

    def accepts(self, computed: Any, nominal: Any) -> Tuple[bool, float]:
        """Return ``(pass, relative_error)`` for ``computed`` against ``nominal``."""
        if self.mode == "exact":
            ok = computed == nominal
            return ok, 0.0 if ok else float("inf")
        try:
            c = float(computed)
            n = float(nominal)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ToleranceError(f"non-numeric value {computed!r}/{nominal!r}: {exc}") from exc
        if self.mode == "band":
            ok = float(self.lower) <= c <= float(self.upper)
            denom = abs(n) if n != 0.0 else 1.0
            return ok, abs(c - n) / denom
        rel = abs(c - n) / abs(n) if n != 0.0 else abs(c - n)
        if self.mode == "relative":
            return rel <= self.value, rel
        return abs(c - n) <= self.value, rel


@dataclass(frozen=True)
class ReferenceAsset:
    """One sha256-anchored reference datum."""

    asset_id: str
    family: str
    source_citation: str
    values: Dict[str, Any]
    tolerance: Tolerance
    applicable_regime: Dict[str, Any]
    units: Dict[str, str]
    evidence_tier: str
    role: str
    primary_source_verified: bool
    verification_note: str
    value_sha256: str = ""
    reference_condition: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise MissingFieldError("asset_id must not be empty")
        _check_citation(self.source_citation)
        if not isinstance(self.values, Mapping) or not self.values:
            raise MissingFieldError(f"{self.asset_id}: 'values' must be a non-empty mapping")
        for key, val in self.values.items():
            if isinstance(val, bool) or not isinstance(val, (int, float, str)):
                raise MissingFieldError(
                    f"{self.asset_id}: value {key!r} must be a number or string, got {type(val).__name__}"
                )
            if isinstance(val, float) and not math.isfinite(val):
                raise MissingFieldError(f"{self.asset_id}: value {key!r} is not finite")
        if not isinstance(self.tolerance, Tolerance):
            raise ToleranceError(f"{self.asset_id}: tolerance must be a Tolerance")
        regime = self.applicable_regime
        if not isinstance(regime, Mapping) or not regime:
            raise MissingFieldError(f"{self.asset_id}: 'applicable_regime' must be a non-empty mapping")
        for key in REGIME_REQUIRED_KEYS:
            if key not in regime:
                raise MissingFieldError(
                    f"{self.asset_id}: applicable_regime is missing {key!r}"
                )
        span = regime["ice_thickness_m"]
        if (
            not isinstance(span, Sequence)
            or isinstance(span, (str, bytes))
            or len(span) != 2
            or float(span[0]) > float(span[1])
        ):
            raise MissingFieldError(
                f"{self.asset_id}: applicable_regime['ice_thickness_m'] must be [min, max]"
            )
        if self.evidence_tier not in EVIDENCE_TIERS:
            raise CorpusError(
                f"{self.asset_id}: evidence_tier {self.evidence_tier!r} not in {sorted(EVIDENCE_TIERS)}"
            )
        if self.role not in ROLES:
            raise CorpusError(f"{self.asset_id}: role {self.role!r} not in {sorted(ROLES)}")
        expected = value_digest(self.asset_id, self.values, self.tolerance)
        if self.value_sha256 and self.value_sha256 != expected:
            raise CorpusIntegrityError(
                f"{self.asset_id}: recorded value_sha256 does not match the recorded numbers"
            )
        object.__setattr__(self, "value_sha256", expected)

    # -- consumption API -------------------------------------------------
    def check(self, computed: Mapping[str, Any] | float, key: Optional[str] = None) -> Dict[str, Any]:
        """Compare computed value(s) against this reference datum."""
        return check_against(self, computed, key=key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "family": self.family,
            "source_citation": self.source_citation,
            "values": self.values,
            "units": self.units,
            "tolerance": self.tolerance.to_dict(),
            "applicable_regime": self.applicable_regime,
            "evidence_tier": self.evidence_tier,
            "role": self.role,
            "primary_source_verified": self.primary_source_verified,
            "verification_note": self.verification_note,
            "value_sha256": self.value_sha256,
            "reference_condition": self.reference_condition,
        }


def asset_from_dict(payload: Mapping[str, Any]) -> ReferenceAsset:
    """Validate one serialised asset and build a :class:`ReferenceAsset`."""
    if not isinstance(payload, Mapping):
        raise MissingFieldError(f"asset must be a mapping, got {type(payload).__name__}")
    for field_name in REQUIRED_FIELDS:
        if field_name not in payload:
            raise MissingFieldError(f"asset {payload.get('asset_id', '<no id>')}: missing {field_name!r}")
    return ReferenceAsset(
        asset_id=_require_text(payload.get("asset_id"), "asset_id"),
        family=_require_text(payload.get("family"), "family"),
        source_citation=_check_citation(payload["source_citation"]),
        values=dict(payload["values"]),
        tolerance=Tolerance.from_dict(payload["tolerance"]),
        applicable_regime=dict(payload["applicable_regime"]),
        units=dict(payload.get("units", {})),
        evidence_tier=_require_text(payload.get("evidence_tier", "repo_transcription"), "evidence_tier"),
        role=_require_text(payload.get("role", "regression_anchor"), "role"),
        primary_source_verified=bool(payload.get("primary_source_verified", False)),
        verification_note=str(payload.get("verification_note", "")),
        value_sha256=str(payload.get("value_sha256", "")),
        reference_condition=dict(payload.get("reference_condition", {})),
    )


def check_against(
    asset: ReferenceAsset,
    computed: Mapping[str, Any] | float,
    key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare ``computed`` against ``asset``'s reference value(s)."""
    if isinstance(computed, Mapping):
        if key is not None:
            keys = [key]
        else:
            keys = list(asset.values)
        verdicts: Dict[str, Any] = {}
        all_pass = True
        worst = 0.0
        for k in keys:
            if k not in asset.values:
                raise CorpusError(f"{asset.asset_id}: no reference value for key {k!r}")
            if k not in computed:
                raise CorpusError(f"{asset.asset_id}: computed value for {k!r} not supplied")
            ok, rel = asset.tolerance.accepts(computed[k], asset.values[k])
            verdicts[k] = {
                "nominal": asset.values[k],
                "computed": computed[k],
                "rel_error": rel,
                "pass": ok,
            }
            all_pass = all_pass and ok
            if math.isfinite(rel):
                worst = max(worst, rel)
        return {
            "asset_id": asset.asset_id,
            "pass": all_pass,
            "worst_rel_error": worst,
            "tolerance": asset.tolerance.to_dict(),
            "tolerance_text": asset.tolerance.describe(),
            "values": verdicts,
        }

    if key is None:
        if len(asset.values) != 1:
            raise CorpusError(
                f"{asset.asset_id} has {len(asset.values)} values; pass a mapping or a key"
            )
        key = next(iter(asset.values))
    if key not in asset.values:
        raise CorpusError(f"{asset.asset_id}: no reference value for key {key!r}")
    ok, rel = asset.tolerance.accepts(computed, asset.values[key])
    return {
        "asset_id": asset.asset_id,
        "pass": ok,
        "worst_rel_error": rel,
        "tolerance": asset.tolerance.to_dict(),
        "tolerance_text": asset.tolerance.describe(),
        "values": {
            key: {
                "nominal": asset.values[key],
                "computed": computed,
                "rel_error": rel,
                "pass": ok,
            }
        },
    }


@dataclass(frozen=True)
class ReferenceCorpus:
    """A verified set of reference assets plus the manifest that anchors them."""

    assets: Tuple[ReferenceAsset, ...]
    manifest: Dict[str, Any]
    root: Path

    def __len__(self) -> int:
        return len(self.assets)

    def __iter__(self) -> Iterator[ReferenceAsset]:
        return iter(self.assets)

    @property
    def asset_ids(self) -> Tuple[str, ...]:
        return tuple(a.asset_id for a in self.assets)

    @property
    def families(self) -> Tuple[str, ...]:
        return tuple(sorted({a.family for a in self.assets}))

    @property
    def aggregate_sha256(self) -> str:
        return str(self.manifest.get("aggregate_sha256", ""))

    def get(self, asset_id: str) -> ReferenceAsset:
        for asset in self.assets:
            if asset.asset_id == asset_id:
                return asset
        raise KeyError(f"unknown reference asset {asset_id!r}")

    def select(
        self,
        *,
        family: Optional[str] = None,
        role: Optional[str] = None,
        evidence_tier: Optional[str] = None,
        regime: Optional[str] = None,
        verified_only: bool = False,
    ) -> Tuple[ReferenceAsset, ...]:
        out = []
        for asset in self.assets:
            if family is not None and asset.family != family:
                continue
            if role is not None and asset.role != role:
                continue
            if evidence_tier is not None and asset.evidence_tier != evidence_tier:
                continue
            if regime is not None and asset.applicable_regime.get("ice_regime") != regime:
                continue
            if verified_only and not asset.primary_source_verified:
                continue
            out.append(asset)
        return tuple(out)

    def check(self, asset_id: str, computed: Mapping[str, Any] | float, key: Optional[str] = None) -> Dict[str, Any]:
        return check_against(self.get(asset_id), computed, key=key)

    def certify(self, computed: Mapping[str, Mapping[str, Any] | float]) -> Dict[str, Any]:
        """Check a batch ``{asset_id: computed}`` payload; return an audit block."""
        results = []
        for asset_id in sorted(computed):
            verdict = self.check(asset_id, computed[asset_id])
            results.append(verdict)
        failed = [r["asset_id"] for r in results if not r["pass"]]
        return {
            "asset_count": len(results),
            "pass": not failed,
            "failed": failed,
            "worst_rel_error": max((r["worst_rel_error"] for r in results), default=0.0),
            "results": results,
        }

    def value_anchors(self) -> Dict[str, str]:
        return {a.asset_id: a.value_sha256 for a in self.assets}


def default_root() -> Path:
    """Locate ``benchmarks/ice_nav_refs`` in this checkout.

    Resolution order: ``TENSORPHYSICS_ICE_REFS_DIR`` env var, then an upward
    search from this module and from the current working directory.
    """
    env = os.environ.get("TENSORPHYSICS_ICE_REFS_DIR")
    if env:
        candidate = Path(env)
        if (candidate / "MANIFEST.json").is_file():
            return candidate
        raise CorpusError(
            f"TENSORPHYSICS_ICE_REFS_DIR={env!r} does not contain MANIFEST.json"
        )
    starts = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for start in starts:
        for ancestor in (start, *start.parents):
            candidate = ancestor / "benchmarks" / "ice_nav_refs"
            if (candidate / "MANIFEST.json").is_file():
                return candidate
    raise CorpusError(
        "cannot locate benchmarks/ice_nav_refs/MANIFEST.json; set TENSORPHYSICS_ICE_REFS_DIR"
    )


def verify_manifest(root: Path, manifest: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Recompute the sha256 of every asset file and compare with the manifest."""
    root = Path(root)
    if manifest is None:
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    errors: list[str] = []
    records: Dict[str, Any] = {}
    for rel_path in sorted(files):
        entry = files[rel_path]
        path = root / rel_path
        if not path.is_file():
            errors.append(f"missing file {rel_path}")
            continue
        payload = path.read_bytes()
        digest = sha256_digest(payload)
        ok = digest == entry.get("sha256")
        if not ok:
            errors.append(f"sha256 mismatch for {rel_path}: {digest} != {entry.get('sha256')}")
        if int(entry.get("bytes", -1)) != len(payload):
            errors.append(f"size mismatch for {rel_path}")
        records[rel_path] = {"sha256": digest, "match": ok, "bytes": len(payload)}
    recomputed_aggregate = sha256_digest(
        "\n".join(f"{rel}:{records[rel]['sha256']}" for rel in sorted(records)).encode("utf-8")
    )
    if recomputed_aggregate != manifest.get("aggregate_sha256"):
        errors.append("aggregate_sha256 mismatch")
    return {
        "ok": not errors,
        "errors": errors,
        "files": records,
        "aggregate_sha256": recomputed_aggregate,
    }


def load_corpus(root: Optional[Path | str] = None, *, verify: bool = True) -> ReferenceCorpus:
    """Load, integrity-check and validate the reference corpus."""
    root_path = Path(root) if root is not None else default_root()
    manifest_path = root_path / "MANIFEST.json"
    if not manifest_path.is_file():
        raise CorpusError(f"no MANIFEST.json under {root_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise CorpusError(
            f"manifest schema {manifest.get('schema')!r} != expected {SCHEMA!r}"
        )
    if verify:
        report = verify_manifest(root_path, manifest)
        if not report["ok"]:
            raise CorpusIntegrityError("; ".join(report["errors"]))

    assets: list[ReferenceAsset] = []
    seen: Dict[str, str] = {}
    for rel_path in sorted(manifest.get("files", {})):
        doc = json.loads((root_path / rel_path).read_text(encoding="utf-8"))
        if doc.get("schema") != SCHEMA:
            raise CorpusError(f"{rel_path}: schema {doc.get('schema')!r} != {SCHEMA!r}")
        for payload in doc.get("assets", []):
            asset = asset_from_dict(payload)
            if asset.asset_id in seen:
                raise CorpusError(
                    f"duplicate asset_id {asset.asset_id!r} in {rel_path} and {seen[asset.asset_id]}"
                )
            seen[asset.asset_id] = rel_path
            assets.append(asset)
    if not assets:
        raise CorpusError(f"corpus at {root_path} contains no assets")
    if len(assets) != int(manifest.get("asset_count", -1)):
        raise CorpusError(
            f"manifest declares {manifest.get('asset_count')} assets, files contain {len(assets)}"
        )
    return ReferenceCorpus(assets=tuple(assets), manifest=manifest, root=root_path)


def describe_corpus(corpus: ReferenceCorpus) -> Dict[str, Any]:
    """Audit summary: coverage by family / tier / role and verification status."""
    by_family: Dict[str, int] = {}
    by_tier: Dict[str, int] = {}
    by_role: Dict[str, int] = {}
    for asset in corpus:
        by_family[asset.family] = by_family.get(asset.family, 0) + 1
        by_tier[asset.evidence_tier] = by_tier.get(asset.evidence_tier, 0) + 1
        by_role[asset.role] = by_role.get(asset.role, 0) + 1
    return {
        "schema": SCHEMA,
        "asset_count": len(corpus),
        "family_count": len(by_family),
        "assets_by_family": dict(sorted(by_family.items())),
        "assets_by_evidence_tier": dict(sorted(by_tier.items())),
        "assets_by_role": dict(sorted(by_role.items())),
        "primary_source_verified": sum(1 for a in corpus if a.primary_source_verified),
        "primary_source_unverified": sum(1 for a in corpus if not a.primary_source_verified),
        "aggregate_sha256": corpus.aggregate_sha256,
        "value_anchors": corpus.value_anchors(),
    }