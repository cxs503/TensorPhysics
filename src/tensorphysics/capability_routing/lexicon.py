"""Domain-lexicon scanner: prove that a code surface carries no domain semantics.

The routing layer is platform mechanism, so it must not smuggle domain meaning
into code (a domain word in an ``if`` branch is exactly how a platform module
starts to depend on one engineering discipline).  This module is the mechanical
guard: it tokenises source text and reports every token matching a lexicon that
is **injected as data** -- the words themselves live in a JSON file, never in
the code checked by the scan.

Fail-closed behaviour: an unreadable lexicon, a lexicon without ``terms``, or a
scan of zero files is an error, because "nothing was checked" must not look like
"the check passed".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "LexiconError",
    "ScanResult",
    "load_terms",
    "tokenize",
    "scan_text",
    "scan_files",
    "find_repo_root",
    "DEFAULT_LEXICON_RELPATH",
]

DEFAULT_LEXICON_RELPATH = "data/capability_routing/lexicon_v1.json"
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


class LexiconError(ValueError):
    """The lexicon payload or the scan request cannot be interpreted."""


@dataclass(frozen=True)
class ScanResult:
    """Outcome of scanning a file set against a data-injected lexicon."""

    files_scanned: int
    hits: tuple[dict, ...]
    terms: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.hits

    def as_dict(self) -> dict:
        return {
            "files_scanned": self.files_scanned,
            "terms": list(self.terms),
            "hits": list(self.hits),
            "clean": self.clean,
        }


def find_repo_root(start: "str | Path") -> Path:
    """Walk upwards until a directory holding ``pyproject.toml`` is found."""
    path = Path(start).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise LexiconError(f"no repository root (pyproject.toml) above {path}")


def load_terms(path: "str | Path") -> frozenset[str]:
    """Load the stranded word list (data only, never inline code literals)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "terms" not in payload:
        raise LexiconError(f"{path}: lexicon must be an object with a 'terms' list")
    terms = payload["terms"]
    if not isinstance(terms, list) or not terms:
        raise LexiconError(f"{path}: 'terms' must be a non-empty list")
    cleaned = set()
    for term in terms:
        if not isinstance(term, str) or not term.strip():
            raise LexiconError(f"{path}: invalid term {term!r}")
        cleaned.add(term.strip().casefold())
    return frozenset(cleaned)


def tokenize(text: str) -> list[tuple[int, int, str]]:
    """``[(line_no, column, token), ...]`` with camelCase/underscore splitting."""
    tokens: list[tuple[int, int, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for match in _TOKEN.finditer(line):
            token = match.group(0)
            offset = match.start()
            for piece in re.split(r"_+", token):
                for part in _CAMEL.split(piece):
                    if not part:
                        continue
                    tokens.append((line_no, offset, part))
                    offset += len(part)
    return tokens


#: Inflections accepted on top of an exact token match (identifiers are usually
#: snake_case, so a domain word shows up as its own token; plurals are the only
#: common spelling variation that survives tokenisation).
_INFLECTIONS = ("", "s", "es", "ed", "d", "ing")


def _expansion(terms: Iterable[str]) -> frozenset[str]:
    folded = frozenset(t.casefold() for t in terms)
    return frozenset(term + suffix for term in folded for suffix in _INFLECTIONS)


def scan_text(text: str, terms: Iterable[str], *, prefix_match: bool = False) -> list[dict]:
    """Find lexicon hits in one source string (case-insensitive, token based).

    ``prefix_match=True`` additionally flags any token that starts with a term
    of three or more characters (stricter, but it also flags compound tokens
    that merely reference an existing module path).
    """
    folded = frozenset(t.casefold() for t in terms)
    expanded = _expansion(folded)
    hits = []
    for line_no, column, token in tokenize(text):
        low = token.casefold()
        matched = low if low in expanded else None
        if matched is None and prefix_match:
            matched = next((term for term in folded if len(term) >= 3 and low.startswith(term)),
                           None)
        if matched is not None:
            hits.append({"line": line_no, "column": column, "token": token, "term": matched})
    return hits


def scan_files(paths: Sequence["str | Path"], terms: Iterable[str],
               *, suffixes: Sequence[str] = (".py",),
               prefix_match: bool = False) -> ScanResult:
    """Scan a file list; each file must exist and match one of ``suffixes``."""
    term_set = tuple(sorted(frozenset(t.casefold() for t in terms)))
    if not term_set:
        raise LexiconError("lexicon is empty: an empty scan cannot certify anything")
    files = [Path(p) for p in paths]
    if not files:
        raise LexiconError("no files to scan: 'zero files scanned' is not a pass")
    hits: list[dict] = []
    scanned = 0
    for file in files:
        if not file.is_file():
            raise LexiconError(f"cannot scan missing file {file}")
        if suffixes and file.suffix not in suffixes:
            continue
        scanned += 1
        for hit in scan_text(file.read_text(encoding="utf-8"), term_set,
                             prefix_match=prefix_match):
            hits.append({"file": str(file), **hit})
    if scanned == 0:
        raise LexiconError("no file matched the requested suffixes: nothing was certified")
    return ScanResult(files_scanned=scanned, hits=tuple(hits), terms=term_set)


def default_lexicon_path(start: "str | Path | None" = None) -> Path:
    """Resolve the repository checkout's injected lexicon file."""
    root = find_repo_root(start or Path(__file__))
    return root / DEFAULT_LEXICON_RELPATH