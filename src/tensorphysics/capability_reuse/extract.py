"""Capability <-> code-path mapping and module dependency projection."""
from __future__ import annotations

import ast
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .indexer import (CODE_SUFFIXES, HEADER_SUFFIXES, PathResolver, RepoIndex,
                      RawDep, RawRef, ast_dependencies, build_index,
                      global_module_index, lexical_references, open_text,
                      resolve_module, resolve_relative_module, set_df_cache,
                      strip_comments)
from .model import Capability, Edge, dedupe_edges

#: evidence tokens: relative paths with a known suffix
_PATH_RE = re.compile(
    r"(?<![\w/.\-])((?:[\w.\-]+/)*[\w.\-]+\.(?:py|c|cc|cpp|cu|cuh|h|hpp|md|json|ya?ml|txt|png|npy))(?![\w/.\-])")
#: capability names are word tokens (node ids, feature slugs)
_NAME_TOKEN_RE = re.compile(r"[A-Za-z][\w.\-]{2,}")
#: files larger than this are not evidence (generated data dumps, vendored code)
MAX_FILE_BYTES = 400_000
#: directories that are never part of a capability's implementation surface
SKIP_PARTS = ("/.git/", "/node_modules/", "/__pycache__/", "/site-packages/", "/.venv/",
              "/venv/", "/build/", "/dist/", "/.pytest_cache/", "/migrations/versions/")

CODE_EXT = CODE_SUFFIXES
HDR_EXT = HEADER_SUFFIXES
ARTIFACT_EXT = (".md", ".json", ".yaml", ".yml", ".txt", ".png", ".npy")
MAX_WORKERS = int(os.environ.get("CAP_REUSE_WORKERS", "0") or 0) or min(32, os.cpu_count() or 1)


def _skip(rel: str) -> bool:
    p = "/" + rel
    return any(s in p for s in SKIP_PARTS)


def repo_files(info: dict) -> Set[str]:
    return set(info.get("files") or ())


def resolve_token(token: str, own_repo: str, repos: Sequence[str],
                  manifest: Dict[str, Set[str]]) -> Optional[Tuple[str, str]]:
    """Backwards-compatible wrapper around :class:`PathResolver` (uncached)."""
    return PathResolver(manifest).resolve(token, own_repo)


def _is_test_like(rel: str) -> bool:
    p = "/" + rel
    base = rel.rsplit("/", 1)[-1]
    return ("/tests/" in p or "/test/" in p or "/testing/" in p
            or base.startswith("test_") or base.endswith("_test.py"))


def _cap_id(name: str, node_id: int, dup_names: Dict[str, int]) -> str:
    return name if dup_names.get(name, 0) <= 1 else f"{name}#{node_id}"


def _area_of(rel: str, import_roots: Sequence[str]) -> str:
    """Implementation area of a file: first directory under the import root."""
    best = ""
    for root in import_roots:
        if root and not rel.startswith(root.rstrip("/") + "/"):
            continue
        best = root
        break
    rel2 = rel[len(best):].lstrip("/") if best else rel
    parts = rel2.split("/")
    if len(parts) == 1:
        return "root"
    if parts[0] == "src" and len(parts) > 2:
        return parts[1]
    return parts[0]


def map_capabilities(snapshot: dict, repos: Dict[str, RepoIndex],
                     params: Optional[Dict[str, object]] = None
                     ) -> Tuple[List[Capability], Dict[Tuple[str, str], str], Dict[str, Dict[str, List]]]:
    """Map every delivered capability to its code / header / artefact paths."""
    params = params or {}
    max_claims = int(params.get("max_claims_per_file", 2))
    names = Counter(c["name"] for c in snapshot["capabilities"])
    manifest = {r: repo_files(info) for r, info in snapshot["repos"].items()}
    resolver = PathResolver(manifest)
    caps: List[Capability] = []
    pathmap: Dict[str, Dict[str, List]] = {}
    claim: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    for entry in snapshot["capabilities"]:
        cid = entry.get("cid") or _cap_id(entry["name"], int(entry.get("node_id") or 0), names)
        own_repo = entry.get("repo") or ""
        rows = [line.get("text", "") for line in (entry.get("evidence") or [])]
        rows.append(entry.get("goal") or "")
        rows.extend(entry.get("acceptance_criteria") or [])
        found: Dict[str, Set[str]] = {"code": set(), "header": set(), "artifact": set()}
        for text in rows:
            for tok in _PATH_RE.findall(text or ""):
                hit = resolver.resolve(tok, own_repo)
                if not hit:
                    continue
                repo, rel = hit
                if _skip(rel):
                    continue
                if rel.endswith(CODE_EXT):
                    found["code"].add(f"{repo}::{rel}")
                elif rel.endswith(HDR_EXT):
                    found["header"].add(f"{repo}::{rel}")
                elif rel.endswith(ARTIFACT_EXT):
                    found["artifact"].add(f"{repo}::{rel}")
        code = sorted(x for x in found["code"] if x.startswith(own_repo + "::"))
        cross = sorted(x for x in found["code"] if not x.startswith(own_repo + "::"))
        # capability class = declared domain, else implementation area of its code
        area_rows = [(item.split("::", 1)[0], item.split("::", 1)[1]) for item in code
                     if item.split("::", 1)[0] in repos]
        non_test = [(r, p_) for r, p_ in area_rows if not _is_test_like(p_)]
        chosen = non_test or area_rows
        areas = Counter(_area_of(rel, repos[r].import_roots) for r, rel in chosen)
        cap_class = entry.get("domain") or (areas.most_common(1)[0][0] if areas else "unclassified")
        caps.append(Capability(
            cid=cid, name=entry["name"], repo=own_repo,
            domain=entry.get("domain"), cap_class=cap_class,
            mission_id=entry.get("mission_id"), status=entry.get("status", "done"),
            origin=entry.get("origin", "node"),
        ))
        pathmap[cid] = {
            "code": code,
            "cross_repo_code": cross,
            "header": sorted(found["header"]),
            "artifact": sorted(found["artifact"]),
        }
        for item in code + cross + sorted(found["header"]):
            repo, rel = item.split("::", 1)
            claim[(repo, rel)][cid] += 1

    done_at = {c["cid"]: (c.get("done_at") or "9999") for c in snapshot["capabilities"]}
    cap_repo = {c["cid"]: (c.get("repo") or "") for c in snapshot["capabilities"]}
    ownership: Dict[Tuple[str, str], str] = {}
    shared: List[Dict[str, object]] = []
    foreign: List[Dict[str, object]] = []
    for key, counter in sorted(claim.items()):
        if len(counter) > max_claims:
            # a file several capabilities claim is shared infrastructure: it
            # cannot be evidence that *one* capability was reused
            shared.append({"file": f"{key[0]}::{key[1]}", "claimants": len(counter),
                           "capabilities": sorted(counter)[:5]})
            continue
        # owner = a capability of the file's own repository (a foreign claimant
        # is a *reuser*, not an owner), earliest delivery first
        repo_of_file = key[0]
        same_repo = [cid for cid in counter if cap_repo.get(cid, "") == repo_of_file]
        if not same_repo:
            # nobody in the file's own repository ever delivered it: the claims
            # all come from other repositories citing it -> ownership unknown
            foreign.append({"file": f"{key[0]}::{key[1]}", "claimants": sorted(counter)[:5]})
            continue
        best = sorted((kv for kv in counter.items() if kv[0] in same_repo),
                      key=lambda kv: (cap_repo.get(kv[0], "") != repo_of_file,
                                      done_at.get(kv[0], "9999"), -kv[1], kv[0]))[0][0]
        ownership[key] = best
    if shared:
        pathmap.setdefault("__shared__", {})["files"] = shared
    if foreign:
        pathmap.setdefault("__foreign_only__", {})["files"] = foreign
    return caps, ownership, pathmap


@dataclass
class ExtractionResult:
    ast_edges: List[Edge] = field(default_factory=list)
    ref_edges: List[Edge] = field(default_factory=list)
    external: List[Dict] = field(default_factory=list)      # reuse from unowned code
    stats: Dict[str, object] = field(default_factory=dict)


_DEF_RE = re.compile(r"^(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_]\w*)", re.M)


def _top_defs(text: str) -> List[Tuple[str, int]]:
    """Top-level definitions by lexical scan (col-0 def/class), no extra AST pass."""
    out = []
    for m in _DEF_RE.finditer(text):
        out.append((m.group(1), text.count("\n", 0, m.start()) + 1))
    return out


def _iter_files(repos: Dict[str, RepoIndex], snapshot: dict):
    for rname, repo in sorted(repos.items()):
        for rel in repo.code_files + repo.header_files:
            if _skip(rel):
                continue
            yield rname, repo, rel


_TEXT_CACHE: Dict[Tuple[str, str], Optional[str]] = {}
_CACHE_BUDGET = 400_000_000
_CACHE_USED = [0]


def _read(repo: RepoIndex, rel: str) -> Optional[str]:
    key = (repo.name, rel)
    if key in _TEXT_CACHE:
        return _TEXT_CACHE[key]
    try:
        path = os.path.join(repo.root, rel)
        if os.path.getsize(path) > MAX_FILE_BYTES:
            _TEXT_CACHE[key] = None
            return None
    except OSError:
        _TEXT_CACHE[key] = None
        return None
    text = open_text(repo.root, rel)
    if _CACHE_USED[0] + len(text) < _CACHE_BUDGET:
        _TEXT_CACHE[key] = text
        _CACHE_USED[0] += len(text)
    return text


# --------------------------------------------------------------------------
# parallel workers (fork inherits the read-only globals set by the caller)
# --------------------------------------------------------------------------
_G: Dict[str, object] = {}


def _w_deps(item):
    rname, rel = item
    repos: Dict[str, RepoIndex] = _G["repos"]           # type: ignore[assignment]
    repo = repos[rname]
    text = _read(repo, rel)
    if text is None:
        return (rname, rel), [], []
    deps = [(d.target, d.line, d.kind, d.raw) for d in ast_dependencies(repos, rname, rel, text)]
    defs = _top_defs(text) if rel.endswith(".py") else []
    return (rname, rel), deps, defs


def _w_df(item):
    rname, rel = item
    repos: Dict[str, RepoIndex] = _G["repos"]           # type: ignore[assignment]
    text = _read(repos[rname], rel)
    if text is None:
        return ()
    syms: Set[str] = _G["symbols"]                     # type: ignore[assignment]
    return tuple(set(re.findall(r"\b([A-Za-z_]\w{5,})\b", text)) & syms)


def _w_refs(item):
    rname, rel = item
    repos: Dict[str, RepoIndex] = _G["repos"]           # type: ignore[assignment]
    text = _read(repos[rname], rel)
    if text is None:
        return (rname, rel), []
    syms: Dict[str, List[str]] = _G["symbol_owner"]     # type: ignore[assignment]
    mods: Set[str] = _G["module_tokens"]                # type: ignore[assignment]
    text = strip_comments(text, os.path.splitext(rel)[1])
    min_len = int(_G.get("min_symbol_len", 8))
    out = []
    for r in lexical_references(text, syms, int(_G["df_limit"]), mods):
        if r.kind == "symbol":
            if len(r.token) >= min_len:
                out.append((r.token, r.line, r.kind, r.raw))
            continue
        tok = r.token
        if tok.startswith("."):     # relative import: same context as the AST calibre
            level = len(tok) - len(tok.lstrip("."))
            abs_mod = resolve_relative_module(repos, rname, rel, tok.lstrip("."), level)
            if not abs_mod:
                continue
            tok = abs_mod
        out.append((tok, r.line, r.kind, r.raw))
    return (rname, rel), out


def _w_external(item):
    rname, rel = item
    repos: Dict[str, RepoIndex] = _G["repos"]           # type: ignore[assignment]
    repo = repos[rname]
    text = _read(repo, rel)
    if text is None or len(text) > 200_000:
        return (rname, rel), []
    alt: re.Pattern = _G["alt"]                        # type: ignore[assignment]
    if not alt.search(text):
        return (rname, rel), []
    out = [(d.target, d.line, d.kind) for d in ast_dependencies(repos, rname, rel, text)]
    return (rname, rel), out


def _imap(func, items, workers: int):
    """Deterministic parallel map (results ordered by input order)."""
    if workers <= 1:
        return [func(it) for it in items]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        return pool.map(func, items, chunksize=max(1, len(items) // (workers * 8) or 1))


def extract_code_edges(snapshot: dict, repos: Dict[str, RepoIndex],
                       ownership: Dict[Tuple[str, str], str],
                       full_universe: bool = True,
                       workers: Optional[int] = None,
                       params: Optional[Dict[str, object]] = None) -> ExtractionResult:
    """Project code dependencies onto capabilities with two independent calibers.

    The *source universe* of both calibers is identical -- the implementation
    files owned by a capability -- so their capability-level projections are
    comparable.  References coming from files nobody owns (glue code, scripts)
    are counted separately as ``external`` evidence and never mixed into the
    dual-caliber comparison.
    """
    workers = MAX_WORKERS if workers is None else workers
    params = params or {}
    res = ExtractionResult()
    gidx = global_module_index(repos)
    manifest = {r: repo_files(info) for r, info in snapshot["repos"].items()}
    resolver = PathResolver(manifest)
    prefix: Dict[str, List[str]] = defaultdict(list)
    for mod in gidx:
        parts = mod.split(".")
        for i in range(1, len(parts) + 1):
            prefix[".".join(parts[:i])].append(mod)

    owned: List[Tuple[str, str]] = []
    others: List[Tuple[str, str]] = []
    for rname, repo, rel in _iter_files(repos, snapshot):
        (owned if (rname, rel) in ownership else others).append((rname, rel))

    def _targets(mod: str, from_repo: str) -> List[Tuple[str, str]]:
        if mod.endswith((".h", ".hpp", ".cuh", ".hxx")) or "/" in mod:
            hit = resolver.resolve(mod, from_repo)
            return [hit] if hit else []
        n = mod
        while n:
            hit = resolve_module(repos, gidx, from_repo, n)
            if hit:
                return hit
            if "." not in n:
                break
            n = n.rsplit(".", 1)[0]
        # No prefix fan-out: a package-level import is attributed only when the
        # package itself resolves to a file (its __init__); fanning out to every
        # child module invents edges that the source line does not support.
        return []

    _G["repos"] = repos
    # ---- pass 1: AST dependencies + top-level definitions -------------------
    dep_rows = _imap(_w_deps, owned, workers)
    deps: Dict[Tuple[str, str], List[Tuple[str, int, str, str]]] = {}
    symbols: Dict[str, List[str]] = defaultdict(list)
    for key, dlist, defs in dep_rows:
        deps[key] = dlist
        for name, _line in defs:
            if len(name) >= 8 and not name.startswith("_"):
                symbols[name].append(f"{key[0]}::{key[1]}")
    symbol_owner = {k: sorted(v) for k, v in symbols.items() if len(v) == 1}
    _G["symbols"] = set(symbol_owner)
    _G["symbol_owner"] = symbol_owner
    _G["df_limit"] = 10 ** 9
    _G["module_tokens"] = set(gidx) | set(prefix)

    # symbol ubiquity is controlled by the strict lexical rule inside the emission
    # step (snake_case, length >= min_symbol_len_strict, defined exactly once,
    # non-test module); a separate document-frequency pass is therefore not needed
    df_limit = 10 ** 9
    _G["min_symbol_len"] = int(params.get("min_symbol_len", 8))
    _G["df_limit"] = df_limit

    # ---- pass 3: lexical calibre + ast projection --------------------------
    ref_rows = _imap(_w_refs, owned, workers)
    refs: Dict[Tuple[str, str], List[Tuple[str, int, str, str]]] = dict(ref_rows)

    for (rname, rel), dlist in sorted(deps.items()):
        src_cap = ownership.get((rname, rel))
        if not src_cap:
            continue
        # a ``from pkg import name`` statement already carries a from_name dep for
        # the concrete module; the bare package edge would double count it
        # resolution rule shared with the lexical calibre: ``from pkg import name``
        # denotes the module ``pkg`` when pkg resolves, and ``pkg.name`` only when
        # pkg itself is not a module (sub-module import of a namespace package)
        base_by_line = {line: target for target, line, kind, raw in dlist if kind == "from"}
        from_names = {line for target, line, kind, raw in dlist if kind == "from_name"}
        for target, line, kind, raw in dlist:
            if kind == "from_name":
                base = base_by_line.get(line)
                if base and _targets(base, rname):
                    continue
            for tgt in _targets(target, rname):
                dst_cap = ownership.get(tgt)
                if dst_cap is None or dst_cap == src_cap:
                    continue
                res.ast_edges.append(Edge(src=src_cap, dst=dst_cap,
                                          kind="c_include" if kind == "include" else "ast_import",
                                          caliber="ast", file=f"{rname}::{rel}", line=line,
                                          detail=raw[:160]))
    for (rname, rel), rlist in sorted(refs.items()):
        src_cap = ownership.get((rname, rel))
        if not src_cap:
            continue
        for token, line, kind, raw in rlist:
            if kind == "symbol":
                # strict symbol calibre: a distinctive snake_case identifier
                # (df filter already applied) defined in a non-test module
                if "_" not in token or len(token) < int(params.get("min_symbol_len_strict", 10)):
                    res.stats["symbol_refs_dropped_generic"] = int(res.stats.get("symbol_refs_dropped_generic", 0)) + 1
                    continue
                tgts = [tuple(o.split("::", 1)) for o in (symbol_owner.get(token) or [])]
                if any(_is_test_like(t[1]) for t in tgts):
                    res.stats["symbol_refs_dropped_test"] = int(res.stats.get("symbol_refs_dropped_test", 0)) + 1
                    continue
            else:
                tgts = _targets(token, rname)
            for tgt in tgts:
                dst_cap = ownership.get(tgt)
                if dst_cap is None or dst_cap == src_cap:
                    continue
                res.ref_edges.append(Edge(
                    src=src_cap, dst=dst_cap,
                    kind="c_include" if raw.lstrip().startswith("#") else (
                        "ref_symbol" if kind == "symbol" else "ref_module"),
                    caliber="ref", file=f"{rname}::{rel}", line=line, detail=raw[:160]))

    # ---- pass 4: external reuse from unowned files --------------------------
    if full_universe and others:
        target_tops = sorted({mod.split(".")[0] for mod in gidx
                              if any(ownership.get((r, p)) for r, p in gidx[mod][:1])})
        if target_tops:
            _G["alt"] = re.compile(r"\b(" + "|".join(re.escape(x) for x in target_tops) + r")\b")
            for (rname, rel), dlist in _imap(_w_external, others, workers):
                for target, line, kind in dlist:
                    for tgt in _targets(target, rname):
                        dst_cap = ownership.get(tgt)
                        if dst_cap is None:
                            continue
                        res.external.append({
                            "dst": dst_cap, "file": f"{rname}::{rel}", "line": line,
                            "kind": "c_include" if kind == "include" else "ast_import",
                            "caliber": "ast"})

    res.ast_edges = dedupe_edges(res.ast_edges)
    res.ref_edges = dedupe_edges(res.ref_edges)
    res.stats = {
        "workers": workers,
        "owned_files": len(owned),
        "unowned_files_scanned": len(others),
        "files_parsed": len(deps),
        "distinctive_symbols": len(symbol_owner),
        "symbol_df_limit": df_limit,
        "ast_edges": len(res.ast_edges),
        "ref_edges": len(res.ref_edges),
        "external_refs": len(res.external),
    }
    return res


def evidence_edges(snapshot: dict, repos: Dict[str, RepoIndex],
                   ownership: Dict[Tuple[str, str], str],
                   pathmap: Dict[str, Dict[str, List]]) -> List[Edge]:
    """Non-code reuse evidence: artefacts, report mentions and declared dependencies.

    * ``artifact_ref``    -- capability X's evidence cites a *file* owned by Y
      (a later report re-uses Y's deliverable / fixture).
    * ``report_ref``      -- capability X's evidence *mentions* capability Y by
      name (the seeding text itself points at the earlier capability).
    * ``dependency_hint`` -- node metadata declares an upstream dependency.
    """
    out: List[Edge] = []
    manifest = {r: repo_files(info) for r, info in snapshot["repos"].items()}
    resolver = PathResolver(manifest)
    name_to_cid: Dict[str, str] = {}
    for entry in snapshot["capabilities"]:
        name_to_cid.setdefault(entry["name"], entry.get("cid") or entry["name"])
    for entry in snapshot["capabilities"]:
        cid = entry.get("cid") or entry["name"]
        own_repo = entry.get("repo") or ""
        for hint in entry.get("dependency_hints") or []:
            other = name_to_cid.get(hint)
            if other and other != cid:
                out.append(Edge(src=cid, dst=other, kind="dependency_hint",
                                caliber="evidence", file="campaign_nodes.metadata",
                                line=0, detail=str(hint)[:120]))
        for line in entry.get("evidence") or []:
            text = line.get("text", "")
            rid = line.get("report_id")
            ln = int(line.get("line") or 0)
            for tok in _PATH_RE.findall(text):
                hit = resolver.resolve(tok, own_repo)
                if not hit:
                    continue
                repo, rel = hit
                dst_cap = ownership.get((repo, rel))
                if dst_cap and dst_cap != cid:
                    out.append(Edge(src=cid, dst=dst_cap, kind="artifact_ref", caliber="evidence",
                                    file=f"report:{rid}", line=ln, detail=f"{repo}::{rel}"))
            for tok in _NAME_TOKEN_RE.findall(text):
                other = name_to_cid.get(tok)
                if other and other != cid and len(tok) >= 8:
                    out.append(Edge(src=cid, dst=other, kind="report_ref", caliber="evidence",
                                    file=f"report:{rid}", line=ln, detail=tok))
    return dedupe_edges(out)