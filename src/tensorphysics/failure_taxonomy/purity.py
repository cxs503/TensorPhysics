"""Domain-purity scanner (criterion C5).

The platform rule is that the failure taxonomy must not encode any physics or
application-domain vocabulary: the same classifiers are consumed by solver libraries
that share nothing but method-level mechanics.  This scanner enforces the rule
mechanically instead of by review:

* the forbidden vocabulary lives **outside** the code, in
  ``data/domain_lexicon.json`` (the "external word list");
* :func:`scan_text` reports every hit with file, line and term;
* :func:`assert_pure` raises when a scope contains a hit.

Tests keep their fixtures synthetic (``node-01``, ``body-01`` ...) so the
scanner can be pointed at the package and the test suite itself.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


class DomainPurityError(AssertionError):
    """A scanned file contains domain vocabulary."""


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    term: str
    text: str

    def as_dict(self) -> dict:
        return {"path": self.path, "line": self.line, "term": self.term, "text": self.text}


def load_lexicon(path: str | Path) -> tuple[str, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    terms = data["terms"] if isinstance(data, dict) else data
    return tuple(str(t) for t in terms)


_ASCII_TERM = re.compile(r"^[a-z0-9][a-z0-9 \-\.]*$")


def _matcher(term: str) -> "re.Pattern[str]":
    """CJK terms match as substrings, ASCII terms with word boundaries."""
    if _ASCII_TERM.match(term.lower()):
        return re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    return re.compile(re.escape(term), re.IGNORECASE)


def scan_text(text: str, terms: Sequence[str], path: str = "<text>") -> list[Hit]:
    hits: list[Hit] = []
    matchers = [(t, _matcher(t)) for t in terms if t]
    for lineno, line in enumerate(text.splitlines(), start=1):
        for term, rx in matchers:
            if rx.search(line):
                hits.append(Hit(path=path, line=lineno, term=term, text=line.strip()[:160]))
    return hits


def scan_paths(paths: Iterable[str | Path], terms: Sequence[str]) -> list[Hit]:
    hits: list[Hit] = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            hits.extend(scan_paths(sorted(p.rglob("*.py")), terms))
            continue
        if p.suffix != ".py":
            continue
        hits.extend(scan_text(p.read_text(encoding="utf-8"), terms, path=str(p)))
    return hits


def assert_pure(paths: Iterable[str | Path], terms: Sequence[str]) -> None:
    hits = scan_paths(paths, terms)
    if hits:
        detail = "; ".join(f"{h.path}:{h.line}:{h.term}" for h in hits[:10])
        raise DomainPurityError(f"domain vocabulary found in scanned scope ({len(hits)} hits): {detail}")


__all__ = ["DomainPurityError", "Hit", "assert_pure", "load_lexicon", "scan_paths", "scan_text"]