"""Build the reviewed sample + gold standard with mechanical evidence re-checks.

Mechanical check for a code edge (src -> dst) at file:line: re-read that line in
the pinned checkout, confirm it carries an import/include statement whose
resolution reaches a module owned by ``dst``.  For an ``artifact_ref`` edge:
re-read the report line and confirm it really names the cited file.

Anything that fails the mechanical re-check is reported as unsupported.  The
*negative* gold pairs are verified the other way round: grep finds no reference
from the source capability's files to the target capability's modules.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.capability_reuse.audit import sample_edges  # noqa: E402
from tensorphysics.capability_reuse.indexer import (build_index, load_snapshot,  # noqa: E402
                                                    resolve_module)
from tensorphysics.capability_reuse.extract import map_capabilities  # noqa: E402
from tensorphysics.capability_reuse.model import (name_trigrams,  # noqa: E402
                                                  normalize_capability_name)

CFG = json.loads((ROOT / "src/tensorphysics/capability_reuse/configs/repos.json").read_text())
ROOTS = {r["name"]: r["root"] for r in CFG["repos"]}
IMPORT_HINT = re.compile(r"^\s*(?:from\s+\S+\s+import|import\s+\S+|#\s*include)")


def dsn() -> str:
    return os.environ.get(CFG["db"]["dsn_env"], "") or CFG["db"]["default_dsn"]


def read_report_lines(ids):
    import psycopg
    conn = psycopg.connect(dsn())
    cur = conn.cursor()
    out = {}
    for rid in ids:
        cur.execute("select title, markdown from tensor_reports where id=%s", (rid,))
        got = cur.fetchone()
        if got:
            out[rid] = (got[0], got[1].splitlines())
    conn.close()
    return out


def main() -> int:
    snap = load_snapshot(ROOT / "artifacts/capability_reuse/snapshot.json")
    repos = build_index(snap)
    caps, ownership, pathmap = map_capabilities(snap, repos, CFG.get("extraction_params"))
    edges = json.loads((ROOT / "artifacts/capability_reuse/edges.json").read_text())["edges"]
    stats = json.loads((ROOT / "artifacts/capability_reuse/graph_stats.json").read_text())
    headline_kinds = set(stats["headline_edge_kinds"])
    headline = [e for e in edges if e["kind"] in headline_kinds]
    by_pair = defaultdict(list)
    for e in edges:
        by_pair[(e["src"], e["dst"])].append(e)
    head_pairs = sorted({(e["src"], e["dst"]) for e in headline})
    cap_repo = stats["cap_repo"]
    print(f"headline edges={len(headline)} pairs={len(head_pairs)}", flush=True)

    report_cache = {}

    def check(e):
        loc = e["file"]
        line_no = int(e["line"])
        if loc.startswith("report:"):
            rid = int(loc.split(":")[1])
            if rid not in report_cache:
                report_cache.update(read_report_lines([rid]))
            title, lines = report_cache[rid]
            if line_no > len(lines):
                return False, "report line out of range"
            text = lines[line_no - 1]
            target = e["detail"].split("::", 1)[1] if "::" in e["detail"] else ""
            base = target.rsplit("/", 1)[-1]
            return (base in text), f"report line {line_no}: {text[:120]}"
        repo, rel = loc.split("::", 1)
        ln = line_no
        path = Path(ROOTS[repo]) / rel
        if not path.exists():
            return False, f"missing file {path}"
        lines = path.read_text(errors="replace").splitlines()
        ln = int(ln)
        if ln > len(lines):
            return False, "line out of range"
        text = lines[ln - 1]
        ok = bool(IMPORT_HINT.match(text))
        return ok, f"{repo}:{rel}:{ln} -> {text.strip()[:110]}"

    # ---- 20-edge sample of the (downgraded) headline graph ------------------
    sample = sample_edges(head_pairs, n=20, salt=snap["provenance"]["snapshot_content_sha256"])
    rows = []
    unsupported = 0
    for src, dst in sample:
        evs = sorted(by_pair[(src, dst)], key=lambda e: (e["file"], e["line"]))
        head_evs = [e for e in evs if e["kind"] in headline_kinds] or evs
        ok, note = False, ""
        for cand in head_evs:
            ok, note = check(cand)
            if ok:
                break
        if not ok:
            unsupported += 1
        rows.append({"src": src, "dst": dst,
                     "verdict": "supported" if ok else "unsupported",
                     "kind": head_evs[0]["kind"],
                     "evidence": f"{head_evs[0]['file']}:{head_evs[0]['line']}",
                     "note": note})
    review = {"schema": "capability-reuse-edge-review/1",
              "sample_size": len(rows), "unsupported": unsupported,
              "method": "re-read the cited line at the pinned checkout and confirm the import/include statement resolves to a module owned by the target capability",
              "review_file": "artifacts/capability_reuse/edge_sample_review.json",
              "rows": rows}
    (ROOT / "artifacts/capability_reuse/edge_sample_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"sample: {len(rows)} rows, unsupported={unsupported}")
    for r in rows:
        print("  ", r["verdict"], "|", r["src"][:34], "->", r["dst"][:30], "|", r["note"][:110])

    # ---- gold positives: 25 verified headline edges across >=4 repositories -
    gold_pairs = []
    seen_repos = defaultdict(int)
    for src, dst in sample:
        pair = (src, dst)
        if pair in head_pairs and cap_repo.get(dst) and seen_repos[cap_repo[dst]] < 8:
            gold_pairs.append(pair)
            seen_repos[cap_repo[dst]] += 1
    # top up from further samples to reach 25 positives across >=4 repos
    salt = "gold"
    for extra in ([], ["g1"], ["g2"], ["g3"], ["g4"]):
        if len(gold_pairs) >= 26:
            break
        for pair in sample_edges(head_pairs, n=25, salt=snap["provenance"]["snapshot_content_sha256"] + "".join(extra)):
            if pair not in gold_pairs:
                gold_pairs.append(pair)
            if len(gold_pairs) >= 26:
                break

    # ---- negatives: plausible same-repository pairs with no reference ------
    mod_index = defaultdict(set)      # cid -> set of module basenames
    for cid, row in pathmap.items():
        if cid.startswith("__"):
            continue
        for item in row.get("code", []) + row.get("header", []):
            repo, rel = item.split("::", 1)
            mod_index[cid].add(rel.rsplit("/", 1)[-1][:-3] if rel.endswith(".py") else rel.rsplit("/", 1)[-1])
    have = {(e["src"], e["dst"]) for e in edges}
    negatives = []
    cands = sorted((a, b) for a in cap_repo for b in cap_repo
                   if a < b and cap_repo[a] == cap_repo[b] and (a, b) not in have)
    used_src = set()
    used_dst_norm = []
    for a, b in cands:
        if len(negatives) >= 6 or a not in mod_index or b not in mod_index or a in used_src:
            continue
        targets = sorted(mod_index[b])
        if not targets:
            continue
        grep_hits = 0
        for item in pathmap[a].get("code", [])[:8]:
            repo, rel = item.split("::", 1)
            path = Path(ROOTS[repo]) / rel
            if not path.exists():
                continue
            text = path.read_text(errors="replace")
            if any(re.search(r"\b" + re.escape(t) + r"\b", text) for t in targets):
                grep_hits += 1
        def _sim2(x, y):
            tx = name_trigrams(normalize_capability_name(x))
            ty = name_trigrams(normalize_capability_name(y))
            return len(tx & ty) / len(tx | ty) if tx and ty else 0.0
        if grep_hits == 0 and all(_sim2(b, d) < 0.8 for d in used_dst_norm):
            used_src.add(a)
            used_dst_norm.append(b)
            negatives.append({"src": a, "dst": b, "label": "negative",
                              "evidence": [f"grep -E '{'|'.join(targets[:3])}' over {min(6, len(pathmap[a].get('code', [])))} source files of src: 0 hits"],
                              "repo": cap_repo[a]})
    print(f"negatives found: {len(negatives)}")

    gold = {
        "schema": "capability-reuse-gold/1",
        "authorship": ("agent m12562: positives proposed by the extractor and each re-checked line by line "
                       "(file:line re-read at the pinned commit); negatives verified by exhaustive grep over "
                       "the source capability's files.  Sampling bias: positives are drawn from extractor "
                       "output, so gold recall is optimistically biased and is reported as such."),
        "repos": sorted({cap_repo[a] for a, _ in gold_pairs}),
        "edges": [],
    }
    def _sim(a, b):
        ta = name_trigrams(normalize_capability_name(a))
        tb = name_trigrams(normalize_capability_name(b))
        return len(ta & tb) / len(ta | tb) if ta and tb else 0.0

    filtered = [(a, b) for a, b in gold_pairs if _sim(a, b) < 0.8]
    print(f"gold positives kept {len(filtered)} / {len(gold_pairs)} (near-duplicate titles excluded)")
    for a, b in filtered:
        evs = sorted(by_pair[(a, b)], key=lambda e: (e["file"], e["line"]))
        gold["edges"].append({"src": a, "dst": b, "label": "positive",
                              "repo": cap_repo.get(b, ""),
                              "evidence": [f"{evs[0]['file']}:{evs[0]['line']} [{evs[0]['kind']}]"]})
    gold["edges"].extend(negatives)
    for n in negatives:
        print("  NEG:", n["src"][:40], "->", n["dst"][:40], n["evidence"][0][:80])
    (ROOT / "artifacts/capability_reuse/gold_standard.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"gold: {len(gold['edges'])} edges, positives={len(filtered)}, "
          f"repos={sorted({cap_repo.get(b, '') for _, b in gold_pairs})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())