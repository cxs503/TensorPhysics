"""Curated parameter library for the ice pressure-area relation.

Every record is a *source-level* datum: one published/repository value of
``ex`` (and optionally ``p0``) together with the units, the area definition,
the footprint geometry and the test/rule conditions under which it holds.

Why a library and not a constant
================================
The gap ``GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-CALIB`` records that ``ex`` is
currently carried as a declared scan range ``0.30-0.70`` with no primary source
and no unified area convention, so the ship-structure demand band
(``1.32x-1.74x``) cannot be narrowed to a quotable value.  This module makes the
evidence explicit and machine-checkable:

* ``area_definition`` is mandatory and comes from a controlled vocabulary
  (:data:`~tensorphysics.ice_pressure_area.conventions.AREA_CONVENTIONS`), so an
  exponent can never again be quoted bare;
* ``role`` separates :data:`ACCURACY_REFERENCE` (may judge a computation) from
  :data:`REGRESSION_ANCHOR` (this repository's own transcription -- drift only);
* ``primary_source_verified`` records honestly whether the primary document was
  opened (in this offline build: never) -- the flag is *not* upgradable
  silently because the entry's digest is anchored in ``MANIFEST.json``;
* :func:`detect_conflicts` fails the library if two records of the same family
  disagree beyond their bands, which is exactly how the in-repo contradiction
  ``ex = 0.15`` (``ice_modules/local_pressure.py``) vs ``ex = 0.45``
  (``benchmarks/ice_nav_refs``) surfaces.

CLI
---
    python -m tensorphysics.ice_pressure_area.library --check    # verify only
    python -m tensorphysics.ice_pressure_area.library --seal     # rewrite MANIFEST
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .conventions import (
    AREA_CONVENTIONS,
    DEFAULT_DAMAGE_RATIO_SWEEP,
    REFERENCE_CONVENTION,
    ConventionError,
    PressureAreaLaw,
    check_convention,
    to_reference,
)

SCHEMA = "ice_pressure_area_library.v1"

ACCURACY_REFERENCE = "accuracy_reference"
REGRESSION_ANCHOR = "regression_anchor"
ROLES = (ACCURACY_REFERENCE, REGRESSION_ANCHOR)

EVIDENCE_CLASSES = (
    "field_measurement",
    "medium_scale_test",
    "model_test",
    "data_compilation",
    "standard_rule",
    "repo_transcription",
)

REQUIRED_FIELDS = (
    "asset_id",
    "family",
    "scale",
    "citation",
    "evidence_class",
    "role",
    "primary_source_verified",
    "verification_note",
    "area_definition",
    "conditions",
    "reported",
)

PLACEHOLDER_CITATIONS = {"n/a", "na", "tbd", "todo", "", "-", "unknown"}


class LibraryError(ValueError):
    """Raised when an entry cannot enter the library."""


class LibraryIntegrityError(LibraryError):
    """Raised when the on-disk library drifted from its manifest."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def default_root() -> Path:
    """Directory holding ``pressure_area_sources.json``."""
    here = Path(__file__).resolve()
    candidates = [parent / "data" / "ice_pressure_area" for parent in here.parents]
    # also allow running from a checkout whose package is installed elsewhere
    candidates += [Path.cwd() / "data" / "ice_pressure_area"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise LibraryError(
        "data/ice_pressure_area not found above "
        + str(here)
        + " nor under the current working directory"
    )


def value_digest(entry: Mapping[str, Any]) -> str:
    """Digest over the load-bearing part of an entry (numbers + conventions)."""
    payload = {
        "asset_id": entry["asset_id"],
        "area_definition": entry["area_definition"],
        "reported": entry["reported"],
        "conditions": entry["conditions"],
        "primary_source_verified": entry["primary_source_verified"],
    }
    return sha256_digest(canonical_json(payload))


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PressureAreaEntry:
    asset_id: str
    family: str
    scale: str
    citation: str
    evidence_class: str
    role: str
    primary_source_verified: bool
    verification_note: str
    area_definition: str
    conditions: Mapping[str, Any]
    ex: float
    ex_band: Tuple[float, float]
    p_ref_MPa: Optional[float]
    A_ref_m2: Optional[float]
    units: Mapping[str, str]
    geometry: Mapping[str, Any]
    data_points: Optional[Mapping[str, Any]] = None
    notes: str = ""
    value_sha256: Optional[str] = None

    # -- derived ---------------------------------------------------------
    @property
    def sigma(self) -> float:
        """One-sigma uncertainty implied by the 90 % band half-width."""
        lo, hi = self.ex_band
        return (hi - lo) / (2.0 * 1.6448536269514722)

    @property
    def sigma90(self) -> float:
        return 0.5 * (self.ex_band[1] - self.ex_band[0])

    def normalized_ex(self, delta_over_sqrtA: float = 0.0) -> float:
        """``ex`` mapped to :data:`REFERENCE_CONVENTION` with law (T)."""
        return to_reference(
            self.ex,
            self.area_definition,
            self.A_ref_m2 if self.A_ref_m2 else 1.0,
            lambda_perimeter=self.geometry.get("perimeter_factor_lambda") or 4.0,
            delta_over_sqrtA=delta_over_sqrtA,
        )

    def conventional_shift(self, delta_over_sqrtA: float) -> float:
        return self.normalized_ex(delta_over_sqrtA) - self.ex

    @property
    def is_anchor(self) -> bool:
        return self.role == REGRESSION_ANCHOR

    @property
    def stratum(self) -> str:
        """Pooling stratum: scale class x role.

        Regression anchors get their own (never pooled) stratum because they are
        this repository's transcription, not external evidence.
        """
        if self.is_anchor:
            return f"anchor::{self.scale}"
        conditions_scale = str(self.conditions.get("scale", self.scale))
        if conditions_scale == "design_rule":
            return "design_rule"
        if self.scale == "global":
            return "global"
        return "measured_local"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "family": self.family,
            "scale": self.scale,
            "citation": self.citation,
            "evidence_class": self.evidence_class,
            "role": self.role,
            "primary_source_verified": self.primary_source_verified,
            "verification_note": self.verification_note,
            "area_definition": self.area_definition,
            "conditions": dict(self.conditions),
            "reported": {
                "ex": self.ex,
                "ex_band": list(self.ex_band),
                "p_ref_MPa": self.p_ref_MPa,
                "A_ref_m2": self.A_ref_m2,
                "units": dict(self.units),
            },
            "geometry": dict(self.geometry),
            "data_points": dict(self.data_points) if self.data_points else None,
            "notes": self.notes,
            "value_sha256": self.value_sha256,
        }


def entry_from_dict(payload: Mapping[str, Any]) -> PressureAreaEntry:
    if not isinstance(payload, Mapping):
        raise LibraryError(f"entry must be a mapping, got {type(payload).__name__}")
    for field_name in REQUIRED_FIELDS:
        if field_name not in payload:
            raise LibraryError(
                f"entry {payload.get('asset_id', '?')} missing required field {field_name!r}"
            )

    citation = str(payload["citation"]).strip()
    if citation.lower() in PLACEHOLDER_CITATIONS:
        raise LibraryError(f"{payload['asset_id']}: placeholder citation {citation!r}")
    if not any(ch.isdigit() for ch in citation) or len(citation) < 20:
        raise LibraryError(
            f"{payload['asset_id']}: citation must name a source with a year and venue"
        )
    if payload["evidence_class"] not in EVIDENCE_CLASSES:
        raise LibraryError(
            f"{payload['asset_id']}: unknown evidence_class {payload['evidence_class']!r}"
        )
    if payload["role"] not in ROLES:
        raise LibraryError(f"{payload['asset_id']}: unknown role {payload['role']!r}")
    if not isinstance(payload["primary_source_verified"], bool):
        raise LibraryError(f"{payload['asset_id']}: primary_source_verified must be bool")
    if not str(payload["verification_note"]).strip():
        raise LibraryError(f"{payload['asset_id']}: verification_note must not be empty")
    check_convention(payload["area_definition"])

    reported = payload["reported"]
    for key in ("ex", "ex_band", "units"):
        if key not in reported:
            raise LibraryError(f"{payload['asset_id']}: reported.{key} missing")
    ex = float(reported["ex"])
    band = tuple(float(x) for x in reported["ex_band"])
    if len(band) != 2 or band[0] >= band[1]:
        raise LibraryError(f"{payload['asset_id']}: ex_band must be an increasing pair")
    if not band[0] <= ex <= band[1]:
        raise LibraryError(
            f"{payload['asset_id']}: ex={ex} outside its own band {band}"
        )
    units = dict(reported["units"])
    for key in ("ex",):
        if key not in units:
            raise LibraryError(f"{payload['asset_id']}: units for {key!r} missing")
    if reported.get("p_ref_MPa") is not None and "p_ref_MPa" not in units:
        raise LibraryError(f"{payload['asset_id']}: units for p_ref_MPa missing")
    if reported.get("A_ref_m2") is not None and "A_ref_m2" not in units:
        raise LibraryError(f"{payload['asset_id']}: units for A_ref_m2 missing")

    conditions = dict(payload["conditions"])
    for key in ("scale", "area_range_m2"):
        if key not in conditions:
            raise LibraryError(f"{payload['asset_id']}: conditions.{key} missing")

    geometry = dict(payload.get("geometry") or {})
    return PressureAreaEntry(
        asset_id=str(payload["asset_id"]),
        family=str(payload["family"]),
        scale=str(payload["scale"]),
        citation=citation,
        evidence_class=str(payload["evidence_class"]),
        role=str(payload["role"]),
        primary_source_verified=bool(payload["primary_source_verified"]),
        verification_note=str(payload["verification_note"]),
        area_definition=str(payload["area_definition"]),
        conditions=conditions,
        ex=ex,
        ex_band=(band[0], band[1]),
        p_ref_MPa=None if reported.get("p_ref_MPa") is None else float(reported["p_ref_MPa"]),
        A_ref_m2=None if reported.get("A_ref_m2") is None else float(reported["A_ref_m2"]),
        units={k: str(v) for k, v in units.items()},
        geometry=geometry,
        data_points=dict(payload["data_points"]) if payload.get("data_points") else None,
        notes=str(payload.get("notes", "")),
        value_sha256=payload.get("value_sha256"),
    )


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------


class PressureAreaLibrary:
    def __init__(self, entries: Sequence[PressureAreaEntry], meta: Mapping[str, Any]):
        self._entries: Tuple[PressureAreaEntry, ...] = tuple(entries)
        self.meta = dict(meta)
        ids = [e.asset_id for e in self._entries]
        if len(set(ids)) != len(ids):
            raise LibraryError(f"duplicate asset_id in library: {sorted(ids)}")

    # -- container protocol ---------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def asset_ids(self) -> Tuple[str, ...]:
        return tuple(e.asset_id for e in self._entries)

    def get(self, asset_id: str) -> PressureAreaEntry:
        for entry in self._entries:
            if entry.asset_id == asset_id:
                return entry
        raise KeyError(asset_id)

    def families(self) -> Tuple[str, ...]:
        return tuple(sorted({e.family for e in self._entries}))

    # -- views -----------------------------------------------------------
    def accuracy_references(self) -> Tuple[PressureAreaEntry, ...]:
        return tuple(e for e in self._entries if e.role == ACCURACY_REFERENCE)

    def anchors(self) -> Tuple[PressureAreaEntry, ...]:
        return tuple(e for e in self._entries if e.role == REGRESSION_ANCHOR)

    def by_stratum(self) -> Dict[str, Tuple[PressureAreaEntry, ...]]:
        out: Dict[str, List[PressureAreaEntry]] = {}
        for entry in self._entries:
            out.setdefault(entry.stratum, []).append(entry)
        return {k: tuple(v) for k, v in sorted(out.items())}

    def unverified(self) -> Tuple[PressureAreaEntry, ...]:
        return tuple(e for e in self._entries if not e.primary_source_verified)

    # -- checks ----------------------------------------------------------
    def detect_conflicts(self, k_sigma: float = 2.0) -> List[Dict[str, Any]]:
        """Pairs inside one family whose exponents disagree beyond their bands.

        ``z = |ex_i - ex_j| / sqrt(sigma_i^2 + sigma_j^2) > k_sigma`` -> conflict.
        """
        conflicts: List[Dict[str, Any]] = []
        entries = list(self._entries)
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a, b = entries[i], entries[j]
                if a.family != b.family:
                    continue
                denom = (a.sigma ** 2 + b.sigma ** 2) ** 0.5
                if denom == 0.0:
                    continue
                z = abs(a.ex - b.ex) / denom
                if z > k_sigma:
                    conflicts.append(
                        {
                            "family": a.family,
                            "asset_ids": [a.asset_id, b.asset_id],
                            "ex": [a.ex, b.ex],
                            "bands": [list(a.ex_band), list(b.ex_band)],
                            "delta_ex": abs(a.ex - b.ex),
                            "z": z,
                            "k_sigma": k_sigma,
                            "severity": "blocking" if z > 2.0 * k_sigma else "warning",
                        }
                    )
        return conflicts

    def area_definition_coverage(self) -> Dict[str, int]:
        cov = {name: 0 for name in AREA_CONVENTIONS}
        for entry in self._entries:
            cov[entry.area_definition] += 1
        return cov

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": self.meta.get("schema", SCHEMA),
            "reference_convention": self.meta.get("reference_convention", REFERENCE_CONVENTION),
            "entries": len(self._entries),
            "families": list(self.families()),
            "strata": {k: [e.asset_id for e in v] for k, v in self.by_stratum().items()},
            "area_definition_coverage": self.area_definition_coverage(),
            "primary_source_verified": [
                e.asset_id for e in self._entries if e.primary_source_verified
            ],
            "unverified": [e.asset_id for e in self.unverified()],
            "conflicts": self.detect_conflicts(),
        }

    def blocking_conflicts(self, k_sigma: float = 2.0) -> List[Dict[str, Any]]:
        return [c for c in self.detect_conflicts(k_sigma) if c["severity"] == "blocking"]

    # -- products --------------------------------------------------------
    def law(
        self,
        stratum: str,
        *,
        ex: Optional[float] = None,
        ex_band: Optional[Tuple[float, float]] = None,
        p0_MPa: Optional[float] = None,
        p0_band_MPa: Optional[Tuple[float, float]] = None,
        source: str = "ice_pressure_area.library",
    ) -> PressureAreaLaw:
        """Assemble a consumable law from a stratum (bands default to its hull)."""
        members = [e for e in self._entries if e.stratum == stratum]
        if not members:
            raise LibraryError(f"empty stratum {stratum!r}; known: {sorted(self.by_stratum())}")
        if ex is None:
            ex = sum(m.ex for m in members) / len(members)
        if ex_band is None:
            ex_band = (min(m.ex_band[0] for m in members), max(m.ex_band[1] for m in members))
        if p0_MPa is None:
            cand = [m.p_ref_MPa for m in members if m.p_ref_MPa is not None]
            if not cand:
                raise LibraryError(f"stratum {stratum!r} carries no p_ref_MPa; supply p0_MPa")
            p0_MPa = sum(cand) / len(cand)
        return PressureAreaLaw(
            p0_MPa=p0_MPa,
            ex=ex,
            convention=REFERENCE_CONVENTION,
            p0_band_MPa=p0_band_MPa,
            ex_band=ex_band,
            source=source,
        )


# ---------------------------------------------------------------------------
# load / seal
# ---------------------------------------------------------------------------


def _library_path(root: Optional[Path | str] = None) -> Path:
    base = Path(root) if root is not None else default_root()
    path = base / "pressure_area_sources.json"
    if not path.is_file():
        raise LibraryError(f"library file not found: {path}")
    return path


def load_library(root: Optional[Path | str] = None, *, verify: bool = True) -> PressureAreaLibrary:
    path = _library_path(root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise LibraryError(f"schema mismatch: {raw.get('schema')!r} != {SCHEMA!r}")
    entries = [entry_from_dict(item) for item in raw["entries"]]

    if verify:
        manifest_path = path.parent / "MANIFEST.json"
        if not manifest_path.is_file():
            raise LibraryIntegrityError(
                f"MANIFEST.json missing next to {path}; run `python -m "
                "tensorphysics.ice_pressure_area.library --seal`"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("file_sha256") != sha256_digest(path.read_bytes()):
            raise LibraryIntegrityError("pressure_area_sources.json drifted from MANIFEST.json")
        for entry in entries:
            expected = manifest["entry_sha256"].get(entry.asset_id)
            if expected is None:
                raise LibraryIntegrityError(f"{entry.asset_id}: no digest in manifest")
            if value_digest(entry.to_dict()) != expected:
                raise LibraryIntegrityError(f"{entry.asset_id}: value digest mismatch")

    # A family conflict is *evidence*, not a load error: the library must be
    # loadable so that the conflict can be reported and consumed by the
    # calibration.  Callers that gate on it use blocking_conflicts()/--strict.
    return PressureAreaLibrary(entries, raw)


def seal(root: Optional[Path | str] = None) -> Dict[str, Any]:
    """Write ``MANIFEST.json``: per-entry value digests + file digest."""
    path = _library_path(root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = [entry_from_dict(item) for item in raw["entries"]]
    manifest = {
        "schema": SCHEMA,
        "file": path.name,
        "file_sha256": sha256_digest(path.read_bytes()),
        "entry_count": len(entries),
        "entry_sha256": {e.asset_id: value_digest(e.to_dict()) for e in entries},
        "aggregate_rule": "sha256 over the sorted 'asset_id:sha256' lines",
    }
    manifest["aggregate_sha256"] = sha256_digest(
        "\n".join(f"{k}:{v}" for k, v in sorted(manifest["entry_sha256"].items())).encode()
    )
    (path.parent / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ice pressure-area parameter library")
    parser.add_argument("--check", action="store_true", help="verify only (default)")
    parser.add_argument("--seal", action="store_true", help="rewrite MANIFEST.json")
    parser.add_argument("--json", action="store_true", help="print describe() as JSON")
    parser.add_argument(
        "--strict", action="store_true", help="exit 2 if a blocking family conflict exists"
    )
    args = parser.parse_args(argv)

    if args.seal:
        manifest = seal()
        print(json.dumps({"sealed": True, "aggregate_sha256": manifest["aggregate_sha256"]}, indent=2))
        return 0

    library = load_library()
    desc = library.describe()
    if args.json:
        print(json.dumps(desc, indent=2, ensure_ascii=False))
    else:
        print(f"entries={desc['entries']} families={desc['families']}")
        print(f"reference_convention={desc['reference_convention']}")
        print(f"area_definition_coverage={desc['area_definition_coverage']}")
        print(f"conflicts={len(desc['conflicts'])}")
        for c in desc["conflicts"]:
            print(f"  [{c['severity']}] {c['family']} {c['asset_ids']} ex={c['ex']} z={c['z']:.2f}")
    if args.strict and library.blocking_conflicts():
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())