"""Zero-domain-semantics scan for the revival registry.

The registry must stay domain-neutral: missions, quantities, thresholds and
blocker names are *data* injected by artifacts.  This scanner is the machine
check.  It reuses the ecosystem-wide externalised vocabulary file
(``campaign/criterion_prereg/domain_vocabulary.json``) so there is one list, and
degrades to ``vocabulary_loaded=False`` (never a silent PASS) if the file is
missing or corrupt.

Only code is scanned (``*.py``); JSON artifacts and reports are data by
construction and are not part of the zero-domain-semantics contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Iterable, Sequence

VOCABULARY_FILE = (
    Path(__file__).resolve().parents[1] / "campaign" / "criterion_prereg" / "domain_vocabulary.json"
)


@dataclass(frozen=True)
class ScanHit:
    path: str
    line: int
    term: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "term": self.term, "text": self.text}


def load_vocabulary(path: Path | str | None = None) -> tuple[tuple[str, ...], bool]:
    p = Path(path) if path is not None else VOCABULARY_FILE
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (), False
    terms: list[str] = []
    for key in ("terms", "cjk_terms"):
        terms.extend(str(t) for t in (data.get(key) or ()))
    return tuple(dict.fromkeys(t for t in terms if t.strip())), bool(terms)


def scan_text(text: str, path: str, terms: Sequence[str]) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for i, line in enumerate(text.splitlines(), start=1):
        low = line.lower()
        for term in terms:
            t = term.lower()
            if re.search(r"[a-z]", t):
                pattern = r"(?<![a-z])" + re.escape(t) + r"(?![a-z])"
                if re.search(pattern, low):
                    hits.append(ScanHit(path=path, line=i, term=term, text=line.strip()[:160]))
            elif t in low:
                hits.append(ScanHit(path=path, line=i, term=term, text=line.strip()[:160]))
    return hits


def scan_paths(paths: Iterable[Path | str],
               vocabulary: Sequence[str] | None = None) -> dict[str, Any]:
    terms = tuple(vocabulary) if vocabulary is not None else load_vocabulary()[0]
    loaded = bool(terms)
    files: list[str] = []
    hits: list[ScanHit] = []
    for raw in paths:
        p = Path(raw)
        candidates = sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for f in candidates:
            if "__pycache__" in f.parts:
                continue
            files.append(str(f))
            hits.extend(scan_text(f.read_text(encoding="utf-8"), str(f), terms))
    return {
        "vocabulary_loaded": loaded,
        "vocabulary_size": len(terms),
        "files_scanned": len(files),
        "hits": [h.to_dict() for h in hits],
        "hit_count": len(hits),
        "clean": loaded and not hits,
        "paths": [str(p) for p in paths],
    }