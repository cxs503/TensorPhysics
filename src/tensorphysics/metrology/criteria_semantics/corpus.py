"""Historical criterion corpus: harvest, freeze, load.

The corpus is the evidence base for the replay re-adjudicator.  It is built from the
platform's own published reports (TensorPlatform ``/tensor/reports``): each report's
Markdown six-element table ⑥ is parsed into criterion rows and frozen to JSONL with the
SHA-256 of the source Markdown, so any row can be traced back to a byte-exact artefact
(zero hand-copying, the corpus is regenerable).

Two ingest paths are supported, for the same downstream format:

``harvest_report_docs``
    live platform API (needs a bearer token); results are cached on disk by report id.
``corpus_from_docs``
    a list of ``{report_id, mission_id, title, markdown}`` dicts — used by the offline
    tests and by report files already checked out in a worktree.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from .parser import CriterionRow, criterion_from_dict, parse_report_criteria

SCHEMA_VERSION = "criteria-corpus/1.0"


def md_sha256(md: str) -> str:
    return hashlib.sha256((md or "").encode("utf-8")).hexdigest()


@dataclass
class ReportDoc:
    report_id: int | None
    mission_id: int | None
    title: str
    markdown: str
    repo: str | None = None
    created_at: str | None = None

    def source(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "mission_id": self.mission_id,
            "title": self.title,
            "repo": self.repo,
            "created_at": self.created_at,
            "markdown_sha256": md_sha256(self.markdown),
        }


def _api_get(base: str, token: str, path: str, params: str = "") -> Any:
    url = base.rstrip("/") + "/" + path.lstrip("/") + (("?" + params) if params else "")
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - local service
        return json.loads(r.read().decode())


def harvest_report_docs(base: str, token: str, *, max_reports: int = 200,
                        cache_dir: str = "/tmp/tp_report_cache",
                        mission_ids: Iterable[int] = ()) -> list[ReportDoc]:
    """Fetch published reports (Markdown) from the platform, with an on-disk cache."""
    os.makedirs(cache_dir, exist_ok=True)
    docs: list[ReportDoc] = []
    refs: list[dict[str, Any]] = []

    if mission_ids:
        for mid in mission_ids:
            d = _api_get(base, token, "tensor/reports", f"mission_id={mid}")
            refs.extend(d.get("items", []) if isinstance(d, dict) else d)
    else:
        offset = 0
        while len(refs) < max_reports:
            d = _api_get(base, token, "tensor/reports", f"limit=100&offset={offset}")
            items = d.get("items", []) if isinstance(d, dict) else d
            if not items:
                break
            refs.extend(items)
            offset += len(items)
            if len(items) < 100:
                break

    seen = set()
    for ref in refs[:max_reports]:
        rid = ref.get("id")
        if rid in seen:
            continue
        seen.add(rid)
        cache = os.path.join(cache_dir, f"{rid}.json")
        doc = None
        if os.path.exists(cache):
            with open(cache) as fh:
                doc = json.load(fh)
        if not doc:
            try:
                det = _api_get(base, token, f"tensor/reports/{rid}")
            except Exception:  # pragma: no cover - network flake
                det = ref
            if isinstance(det, dict) and "report" in det:
                det = det["report"]
            doc = {"id": rid, "mission_id": det.get("mission_id") or ref.get("mission_id"),
                   "title": det.get("title") or ref.get("title"),
                   "markdown": det.get("markdown") or "",
                   "created_at": det.get("created_at") or ref.get("created_at")}
            with open(cache, "w") as fh:
                json.dump(doc, fh, ensure_ascii=False)
        if doc.get("markdown"):
            docs.append(ReportDoc(report_id=doc["id"], mission_id=doc.get("mission_id"),
                                  title=doc.get("title") or "", markdown=doc["markdown"],
                                  created_at=doc.get("created_at")))
    return docs


def corpus_from_docs(docs: Iterable[ReportDoc]) -> list[CriterionRow]:
    """Parse every report doc into criterion rows (flat list, source kept per row)."""
    rows: list[CriterionRow] = []
    for doc in docs:
        rows.extend(parse_report_criteria(doc.markdown, source=doc.source()))
    return rows


def write_corpus(rows: Iterable[CriterionRow], path: str, *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Freeze the corpus to JSONL; return a manifest with the file's SHA-256."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        if meta:
            fh.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for row in rows:
            fh.write(json.dumps({"kind": "criterion", **row.as_dict()},
                                ensure_ascii=False) + "\n")
            n += 1
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return {"path": path, "n_criteria": n, "sha256": digest,
            "schema_version": SCHEMA_VERSION}


def load_corpus(path: str) -> list[CriterionRow]:
    """Load a frozen JSONL corpus back into criterion rows."""
    rows: list[CriterionRow] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "kind" in d and d.get("kind") == "criterion":
                d = {k: v for k, v in d.items() if k != "kind"}
            rows.append(criterion_from_dict(d))
    return rows


def score_docs(docs: Iterable[ReportDoc]) -> dict[str, Any]:
    """Quick coverage probe: how many criterion rows each report yields."""
    out = []
    for doc in docs:
        rows = parse_report_criteria(doc.markdown, source=doc.source())
        out.append({"report_id": doc.report_id, "mission_id": doc.mission_id,
                    "title": doc.title, "n_criteria": len(rows)})
    return {"n_reports": len(out), "n_criteria": sum(r["n_criteria"] for r in out),
            "reports": out}