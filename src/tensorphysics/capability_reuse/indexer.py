"""Repository / module index and the two mechanical extractors.

Universe of evidence = every Python / C / C++ / CUDA file tracked in the
repositories listed by the snapshot.  Two *independent mechanisms* read the
same universe:

* ``ast``  -- the parser reads ``import``/``from ... import`` statements and
  ``#include`` directives (structural dependency).
* ``ref``  -- a lexical scan finds module tokens (``import x``, ``from x.y
  import``, ``importlib.import_module("x.y")``, ``#include "x.h"``) and
  distinctive symbol references (a name defined in exactly one module of the
  ecosystem, used in another module).

Both are projected onto the same capability nodes downstream; the comparison
of the two projections is the cross-validation required by the node.
"""
from __future__ import annotations

import ast
import bisect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .model import SNAPSHOT_SCHEMA

CODE_SUFFIXES = (".py",)
HEADER_SUFFIXES = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp")

_IMPORT_RE = re.compile(r"^[ \t]*import[ \t]+([\w.]+(?:[ \t]*,[ \t]*[\w.]+)*)", re.M)
_FROM_RE = re.compile(r"^[ \t]*from[ \t]+([\w.]+|\.+[\w.]*)[ \t]+import[ \t]+([^\n#]+)", re.M)
_DYN_RE = re.compile(r"(?:import_module|__import__)\(\s*[\"']([\w.]+)[\"']")
_INCLUDE_RE = re.compile(r"^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]+)[>\"]", re.M)
_QUALIFIED_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)")
_TOKEN_RE = re.compile(r"\b([A-Za-z_]\w{5,})\b")


@dataclass
class RepoIndex:
    name: str
    root: str
    head: str
    dirty: bool
    import_roots: Tuple[str, ...]
    code_files: Tuple[str, ...]        # repo-relative python paths
    header_files: Tuple[str, ...]      # repo-relative c/c++/cuda paths
    module_map: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "root": self.root, "head": self.head, "dirty": self.dirty,
            "import_roots": list(self.import_roots),
            "code_files": list(self.code_files), "header_files": list(self.header_files),
        }


def _module_names(relpath: str, import_roots: Sequence[str]) -> List[str]:
    """All module aliases of a path: every import root it lives under.

    A repository can be imported both as ``pkg.mod`` (root ``src/``) and as
    ``src.pkg.mod`` (root ``""``); both spellings occur in the wild, so both are
    registered and resolution accepts either.
    """
    if not relpath.endswith(".py"):
        return []
    roots = [r for r in import_roots
             if not r or relpath.startswith(r.rstrip("/") + "/")]
    roots.sort(key=len, reverse=True)
    out: List[str] = []
    for root in roots:
        rel = relpath[len(root):].lstrip("/") if root else relpath
        mod = rel[:-3].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        if mod and mod not in out:
            out.append(mod)
    return out


def _module_name(relpath: str, import_roots: Sequence[str]) -> Optional[str]:
    """Canonical (longest import root) module name of a repo-relative path."""
    names = _module_names(relpath, import_roots)
    return names[0] if names else None


def build_index(snapshot: dict) -> Dict[str, RepoIndex]:
    repos: Dict[str, RepoIndex] = {}
    for name, info in sorted(snapshot["repos"].items()):
        roots = tuple(info.get("import_roots") or [""])
        code = tuple(sorted(p for p in info["files"] if p.endswith(CODE_SUFFIXES)))
        hdr = tuple(sorted(p for p in info["files"] if p.endswith(HEADER_SUFFIXES)))
        mm: Dict[str, List[str]] = {}
        for rel in code:
            for mod in _module_names(rel, roots):
                mm.setdefault(mod, []).append(rel)
        repos[name] = RepoIndex(
            name=name, root=info["root"], head=info.get("head", ""), dirty=bool(info.get("dirty")),
            import_roots=roots, code_files=code, header_files=hdr,
            module_map={k: tuple(sorted(v)) for k, v in sorted(mm.items())},
        )
    return repos


def global_module_index(repos: Dict[str, RepoIndex]) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {}
    for rname, repo in sorted(repos.items()):
        for mod, rels in sorted(repo.module_map.items()):
            for rel in rels:
                out.setdefault(mod, []).append((rname, rel))
    return {k: sorted(v) for k, v in sorted(out.items())}


def resolve_module(repos: Dict[str, RepoIndex], gidx: Dict[str, List[Tuple[str, str]]],
                   from_repo: str, module: str) -> List[Tuple[str, str]]:
    """Resolve an absolute module name: own repository first, then ecosystem-wide."""
    own = repos.get(from_repo)
    if own is not None and module in own.module_map:
        return [(from_repo, r) for r in own.module_map[module]]
    return list(gidx.get(module, []))


_MOD_CACHE: Dict[Tuple[str, str], Optional[str]] = {}


def _module_of_file(repos: Dict[str, RepoIndex], repo: str, relpath: str) -> Optional[str]:
    key = (repo, relpath)
    if key in _MOD_CACHE:
        return _MOD_CACHE[key]
    ri = repos.get(repo)
    val = None if ri is None else _module_name(relpath, ri.import_roots)
    _MOD_CACHE[key] = val
    return val


def _resolve_relative(repos: Dict[str, RepoIndex], repo: str, relpath: str,
                      module: str, level: int) -> Optional[str]:
    cur = _module_of_file(repos, repo, relpath)
    if cur is None:
        return None
    parts = cur.split(".")
    if relpath.endswith("__init__.py"):
        base = parts
    else:
        base = parts[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)] if len(base) >= level - 1 else []
    tail = [p for p in module.split(".") if p] if module else []
    cand = ".".join(base + tail)
    return cand or None


def resolve_relative_module(repos: Dict[str, RepoIndex], repo: str, relpath: str,
                            module: str, level: int) -> Optional[str]:
    """Public wrapper: absolute module name of a relative import."""
    return _resolve_relative(repos, repo, relpath, module, level)


@dataclass(frozen=True)
class RawDep:
    target: str        # absolute module name (may not resolve to a file)
    line: int
    kind: str          # import | from | dynamic | include
    raw: str


def ast_dependencies(repos: Dict[str, RepoIndex], repo: str, relpath: str, text: str) -> List[RawDep]:
    """Structural extraction: ``ast`` for Python, include directives for C/C++."""
    out: List[RawDep] = []
    if relpath.endswith(".py"):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return out
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    out.append(RawDep(a.name, node.lineno, "import", f"import {a.name}"))
            elif isinstance(node, ast.ImportFrom):
                lvl = node.level or 0
                mod = node.module or ""
                if lvl:
                    abs_mod = _resolve_relative(repos, repo, relpath, mod, lvl)
                else:
                    abs_mod = mod
                if abs_mod:
                    out.append(RawDep(abs_mod, node.lineno, "from",
                                      f"from {'.' * lvl}{mod} import ..."))
                    # ``from pkg import sub`` may import a *module*, not a symbol
                    for a in node.names:
                        out.append(RawDep(f"{abs_mod}.{a.name}", node.lineno, "from_name",
                                          f"from {mod} import {a.name}"))
            elif isinstance(node, ast.Call):
                fn = node.func
                fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                if fname in ("import_module", "__import__") and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        out.append(RawDep(arg.value, node.lineno, "dynamic",
                                          f'{fname}("{arg.value}")'))
    else:
        for m in _INCLUDE_RE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            out.append(RawDep(m.group(1), line, "include", f'#include "{m.group(1)}"'))
    return out


@dataclass(frozen=True)
class RawRef:
    token: str
    line: int
    kind: str          # module | symbol
    raw: str


_NEWLINE_RE = re.compile(r"\n")


def line_index(text: str) -> List[int]:
    return [m.start() for m in _NEWLINE_RE.finditer(text)]


def lexical_references(text: str, symbols: Dict[str, List[str]], df_limit: int,
                       module_tokens: Optional[Set[str]] = None) -> List[RawRef]:
    """Mechanical extraction: module tokens and distinctive symbol references."""
    out: List[RawRef] = []
    nl = line_index(text)

    def _line(pos: int) -> int:
        return bisect.bisect_right(nl, pos) + 1

    for m in _IMPORT_RE.finditer(text):
        for part in m.group(1).split(","):
            tok = part.strip()
            if tok:
                out.append(RawRef(tok, _line(m.start()), "module", m.group(0).strip()))
    for m in _FROM_RE.finditer(text):
        tok = m.group(1)
        if tok:      # pure-dot tokens are relative imports ("from . import x")
            out.append(RawRef(tok, _line(m.start()), "module", m.group(0).strip()[:120]))
    for m in _DYN_RE.finditer(text):
        out.append(RawRef(m.group(1), _line(m.start()), "module", m.group(0)))
    for m in _INCLUDE_RE.finditer(text):
        out.append(RawRef(m.group(1), _line(m.start()), "module", m.group(0)))
    if module_tokens:
        for m in _QUALIFIED_RE.finditer(text):
            tok = m.group(1)
            if tok in module_tokens:
                out.append(RawRef(tok, _line(m.start()), "module", tok))
    # symbol references (distinctive names only: defined once, not ubiquitous)
    seen: Set[str] = set()
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(1)
        if tok in seen or tok not in symbols:
            continue
        owners = symbols[tok]
        if len(owners) != 1 or df_limit and _df(tok, owners) > df_limit:
            continue
        seen.add(tok)
        out.append(RawRef(tok, _line(m.start()), "symbol", tok))
    return out


_DF_CACHE: Dict[str, int] = {}


def _df(tok: str, owners: Sequence[str]) -> int:
    return _DF_CACHE.get(tok, 1)


def set_df_cache(cache: Dict[str, int]) -> None:
    _DF_CACHE.clear()
    _DF_CACHE.update(cache)


def open_text(root: str, relpath: str) -> str:
    try:
        return (Path(root) / relpath).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


class PathResolver:
    """Resolve repo-relative path tokens with an O(1) basename index and a cache."""

    def __init__(self, manifest: Dict[str, Set[str]]):
        from collections import defaultdict as _dd
        self.manifest = manifest
        self.repos = sorted(manifest)
        self.by_base: Dict[str, List[Tuple[str, str]]] = _dd(list)
        for r in self.repos:
            for p in sorted(manifest[r]):
                self.by_base[p.rsplit("/", 1)[-1]].append((r, p))
        self._cache: Dict[Tuple[str, str], Optional[Tuple[str, str]]] = {}

    def resolve(self, token: str, own_repo: str) -> Optional[Tuple[str, str]]:
        key = (own_repo, token)
        if key in self._cache:
            return self._cache[key]
        tok = token.strip().lstrip("./").replace("\\", "/")
        hit: Optional[Tuple[str, str]] = None
        if tok and not tok.startswith("/"):
            if tok in self.manifest.get(own_repo, ()):
                hit = (own_repo, tok)
            else:
                exact = [(r, tok) for r in self.repos if tok in self.manifest.get(r, ())]
                if len(exact) == 1:
                    hit = exact[0]
                elif "/" not in tok:
                    cand = self.by_base.get(tok, ())
                    if len(cand) == 1:
                        hit = cand[0]
        self._cache[key] = hit
        return hit


def load_snapshot(path: str | Path) -> dict:
    snap = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = snap.get("schema")
    if schema != SNAPSHOT_SCHEMA:
        raise ValueError(f"unexpected snapshot schema {schema!r}, want {SNAPSHOT_SCHEMA!r}")
    return snap


def strip_comments(text: str, suffix: str = ".py") -> str:
    """Remove source comments but keep string literals (docstrings are evidence).

    Comments are not dependencies: a lexical scanner that reads them invents
    edges the code does not support.  Quotes are tracked so a hash inside a
    string literal survives.
    """
    if suffix != ".py":
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        return re.sub(r"//[^\n]*", "", text)
    trip_a = chr(34) * 3
    trip_b = chr(39) * 3
    in_tri = None
    out_lines = []
    for line in text.splitlines():
        buf = []
        i = 0
        n = len(line)
        while i < n:
            if in_tri:
                if line.startswith(in_tri, i):
                    buf.append(in_tri)
                    i += 3
                    in_tri = None
                    continue
                buf.append(line[i])
                i += 1
                continue
            if line.startswith(trip_a, i):
                in_tri = trip_a
                buf.append(trip_a)
                i += 3
                continue
            if line.startswith(trip_b, i):
                in_tri = trip_b
                buf.append(trip_b)
                i += 3
                continue
            if line[i] == "#":
                break
            buf.append(line[i])
            i += 1
        out_lines.append("".join(buf))
    return "\n".join(out_lines)

