"""Zero-domain-semantics self-check: the logic must not contain domain words.

The ledger's value depends on the vocabulary being *injected*: if a domain word
creeps into ``.py`` logic, the module silently becomes single-domain and the
replay of another wave would be biased by the first one.  This module scans the
package sources (and optionally any extra files) for the configured deny list
using word-boundary matching, and returns every hit with file/line/word so the
check is a hard, reproducible gate rather than a code-review habit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import VocabConfig

__all__ = ["Hit", "package_sources", "scan_sources", "selfcheck"]


@dataclass(frozen=True)
class Hit:
    path: str
    line_no: int
    word: str
    line: str

    def as_dict(self) -> dict:
        return {"path": self.path, "line_no": self.line_no, "word": self.word, "line": self.line}


def package_sources(root: Path | None = None) -> list[Path]:
    base = Path(root) if root is not None else Path(__file__).resolve().parent
    return sorted(p for p in base.rglob("*.py") if p.is_file())


def _pattern(word: str) -> re.Pattern:
    if re.fullmatch(r"[A-Za-z0-9_+. -]+", word):
        return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])", re.IGNORECASE)
    return re.compile(re.escape(word))


def scan_sources(paths: Iterable[Path], deny_list: Iterable[str], root: Path | None = None) -> list[Hit]:
    base = Path(root) if root is not None else Path.cwd()
    pats = [(w, _pattern(w)) for w in deny_list if w]
    hits: list[Hit] = []
    for path in sorted(Path(p) for p in paths):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for word, pat in pats:
                if pat.search(line):
                    try:
                        rel = str(path.resolve().relative_to(base.resolve()))
                    except ValueError:
                        rel = str(path)
                    hits.append(Hit(path=rel, line_no=line_no, word=word, line=line.strip()))
    hits.sort(key=lambda h: (h.path, h.line_no, h.word))
    return hits


def selfcheck(cfg: VocabConfig, root: Path | None = None, extra_paths: Iterable[Path] = ()) -> dict:
    base = Path(root) if root is not None else Path(__file__).resolve().parent
    paths = list(package_sources(base)) + [Path(p) for p in extra_paths]
    hits = scan_sources(paths, cfg.deny_list, root=base)
    return {
        "schema": "lineage_debt_selfcheck.v1",
        "files_scanned": len(paths),
        "deny_list_size": len(cfg.deny_list),
        "hits": [h.as_dict() for h in hits],
        "verdict": "pass" if not hits else "fail",
    }