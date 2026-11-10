"""Loader for the frozen FAIL corpus and the external vocabulary files.

Everything the taxonomy knows about wording lives under ``data/`` next to this
module and is loaded here through :mod:`importlib.resources`, so the package can
be installed read-only and the code still holds zero domain vocabulary.

The frozen corpus is *pinned*: ``frozen_manifest.json`` records the record
count and the SHA-256 of the JSONL payload.  :func:`load_frozen_set` verifies
the pin on every load, so an accidental edit of the back-test set raises instead
of silently changing the reported numbers (the "frozen" part of "frozen set").
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"

FROZEN_FILE = "frozen_fail_set.jsonl"
MANIFEST_FILE = "frozen_manifest.json"
DECOY_FILE = "decoy_pool.jsonl"
CUE_FILE = "cue_lexicon.json"
WEIGHTS_FILE = "rule_weights.json"
SEEDS_FILE = "seed_exemplars.json"
DOMAIN_FILE = "domain_lexicon.json"


class FrozenSetError(RuntimeError):
    """The frozen corpus no longer matches its pinned digest."""


@dataclass(frozen=True)
class FailRecord:
    """One graded FAIL record of the frozen set."""

    record_id: str
    body: str
    gold_class: str
    annotator: str
    provenance: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "body": self.body,
            "gold_class": self.gold_class,
            "annotator": self.annotator,
            "provenance": dict(self.provenance),
        }


def _read_json(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def digest_of(records: list[dict]) -> str:
    """SHA-256 of the canonical JSONL payload of *records*."""
    payload = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(name: str) -> list[dict]:
    path = DATA_DIR / name
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest() -> dict:
    return _read_json(MANIFEST_FILE)


def load_frozen_set(verify: bool = True) -> list[FailRecord]:
    rows = load_jsonl(FROZEN_FILE)
    if verify:
        manifest = load_manifest()
        if len(rows) != manifest["n_records"]:
            raise FrozenSetError(
                f"record count {len(rows)} != pinned {manifest['n_records']}"
            )
        if digest_of(rows) != manifest["sha256"]:
            raise FrozenSetError("frozen corpus digest mismatch: the back-test set was edited")
    return [
        FailRecord(
            record_id=r["record_id"],
            body=r["body"],
            gold_class=r["gold_class"],
            annotator=r.get("annotator", "unknown"),
            provenance=r.get("provenance", {}),
        )
        for r in rows
    ]


def load_raw_frozen_set() -> list[dict]:
    return load_jsonl(FROZEN_FILE)


def load_decoys() -> list[str]:
    return [r["body"] for r in load_jsonl(DECOY_FILE)]


def _strip_notes(data: dict) -> dict:
    """Drop ``_note`` documentation keys so data files stay self-describing."""
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def load_cue_lexicon() -> dict[str, list[str]]:
    return _strip_notes(_read_json(CUE_FILE))


def load_rule_weights() -> dict[str, dict[str, float]]:
    return _strip_notes(_read_json(WEIGHTS_FILE))


def load_seeds() -> dict[str, list[str]]:
    return _strip_notes(_read_json(SEEDS_FILE))


def load_domain_lexicon() -> tuple[str, ...]:
    data = _read_json(DOMAIN_FILE)
    terms = data["terms"] if isinstance(data, dict) else data
    return tuple(str(t) for t in terms)


def stratum_split(records: list[FailRecord], mod: int = 3, remainder: int = 0) -> tuple[list[FailRecord], list[FailRecord]]:
    """Deterministic stratified split: every ``mod``-th record of a class is held out.

    The split is a property of the frozen file order, not of the run, so train
    and test are reproducible across machines without storing an index file.
    """
    train, test = [], []
    counters: dict[str, int] = {}
    for rec in records:
        i = counters.get(rec.gold_class, 0)
        counters[rec.gold_class] = i + 1
        (test if i % mod == remainder else train).append(rec)
    return train, test


__all__ = [
    "DATA_DIR",
    "FailRecord",
    "FrozenSetError",
    "digest_of",
    "load_cue_lexicon",
    "load_decoys",
    "load_domain_lexicon",
    "load_frozen_set",
    "load_manifest",
    "load_rule_weights",
    "load_seeds",
    "stratum_split",
]