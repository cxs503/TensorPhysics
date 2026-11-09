"""Zero-domain-semantics scan for the criterion pre-registration package.

The package must stay domain-neutral: campaign criteria are data, never
engineering-domain vocabulary baked into the logic.  This scanner is the machine
check for that rule.  The vocabulary lives outside the code
(``domain_vocabulary.json`` next to this file) so the *list* is data and can be
extended without touching logic; a missing/corrupt vocabulary degrades to an
empty list (identity behaviour, never a crash) instead of silently passing with
a broken rule -- the caller sees ``vocabulary_loaded=False``.

Usage::

    python -m tensorphysics.campaign.criterion_prereg.scan src/tensorphysics/campaign/criterion_prereg

Zero domain semantics: this module contains no domain term; the terms are read
from the externalised JSON file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

VOCABULARY_FILE = Path(__file__).with_name("domain_vocabulary.json")
DEFAULT_TARGET = Path(__file__).parent


@dataclass
class ScanHit:
    path: str
    line: int
    term: str
    text: str

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "line": self.line, "term": self.term, "text": self.text}


@dataclass
class ScanResult:
    n_files: int
    hits: list[ScanHit] = field(default_factory=list)
    vocabulary_loaded: bool = True
    n_terms: int = 0

    @property
    def n_hits(self) -> int:
        return len(self.hits)

    @property
    def clean(self) -> bool:
        return self.n_hits == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "n_files": self.n_files,
            "n_hits": self.n_hits,
            "clean": self.clean,
            "vocabulary_loaded": self.vocabulary_loaded,
            "n_terms": self.n_terms,
            "hits": [h.to_dict() for h in self.hits],
        }


def load_vocabulary(path: Path | str | None = None) -> tuple[list[str], bool]:
    """Load the term list; returns ``(terms, loaded)`` and never raises."""
    target = Path(path) if path is not None else VOCABULARY_FILE
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], False
    terms: list[str] = []
    for section in ("terms", "cjk_terms"):
        values = payload.get(section) or []
        if isinstance(values, list):
            terms.extend(str(v) for v in values if str(v).strip())
    return terms, bool(terms)


def _matcher(term: str) -> re.Pattern[str]:
    if re.fullmatch(r"[A-Za-z0-9_\- ]+", term):
        return re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(re.escape(term))


def scan_text(text: str, terms: Sequence[str], *, path: str = "<text>") -> list[ScanHit]:
    hits: list[ScanHit] = []
    patterns = [(t, _matcher(t)) for t in terms]
    for lineno, line in enumerate(text.splitlines(), start=1):
        for term, pattern in patterns:
            if pattern.search(line):
                hits.append(ScanHit(path=path, line=lineno, term=term, text=line.strip()[:160]))
    return hits


def scan_paths(targets: Iterable[Path | str] | None = None, *, vocabulary: Path | str | None = None) -> ScanResult:
    """Scan ``*.py`` files under each target for domain vocabulary."""
    terms, loaded = load_vocabulary(vocabulary)
    files: list[Path] = []
    for target in targets or [DEFAULT_TARGET]:
        p = Path(target)
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(x for x in p.rglob("*.py")))
    hits: list[ScanHit] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        hits.extend(scan_text(text, terms, path=str(f)))
    return ScanResult(n_files=len(files), hits=hits, vocabulary_loaded=loaded, n_terms=len(terms))


__all__ = ["ScanHit", "ScanResult", "load_vocabulary", "scan_text", "scan_paths", "VOCABULARY_FILE"]