#!/usr/bin/env python3
"""m1830 test-baseline drift patrol — figure generator (zero hand-copy: numbers come from artifacts).

Inputs:
  artifacts/tests/junit_*.xml             raw pytest results (run1 = baseline archive, run2 = patrol)
  scenarios/suboff_icebreaking/prereg/sir0_baseline.json  contract baseline triples
Outputs: docs/reports/m1830/fig_*.png  (+ figure_data.json)

Labels are English on purpose: the render host ships no CJK font (DejaVu/Noto Sans only),
so Chinese labels would render as tofu boxes. Report prose stays Chinese.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]  # docs/reports/m1830/make_figures.py -> repo root
ART = ROOT / "artifacts" / "tests"
OUT = ROOT / "docs" / "reports" / "m1830"
BASELINE = ROOT / "scenarios/suboff_icebreaking/prereg/sir0_baseline.json"
EXTERNAL_ROOTS = ("/nfs/",)
OUT.mkdir(parents=True, exist_ok=True)


def parse(junit: Path) -> dict:
    root = ET.parse(junit).getroot()
    if root.tag == "testsuites":
        root = root.find("testsuite")
    counts = {"passed": 0, "failed": 0, "skipped": 0, "env_red": 0}
    durations: dict[str, float] = {}
    for case in root.iter("testcase"):
        nodeid = f"{case.get('classname', '')}::{case.get('name', '')}"
        durations[nodeid] = float(case.get("time") or 0.0)
        if case.find("failure") is not None or case.find("error") is not None:
            counts["failed"] += 1
        elif (skip := case.find("skipped")) is not None:
            counts["env_red" if (skip.get("message") or "").startswith("环境红") else "skipped"] += 1
        else:
            counts["passed"] += 1
    counts["total"] = sum(counts.values())
    return {"counts": counts, "durations": durations}


def fig_suite_status(runs: list[tuple[str, Path]]) -> dict:
    labels = [lbl for lbl, _ in runs]
    series = {k: [] for k in ("passed", "env_red", "failed", "skipped")}
    for _, path in runs:
        c = parse(path)["counts"]
        for k in series:
            series[k].append(c[k])
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
    colors = {"passed": "#2e7d32", "env_red": "#f9a825", "failed": "#c62828", "skipped": "#607d8b"}
    names = {"passed": "passed", "env_red": "env-gated (external evidence offline)",
             "failed": "failed", "skipped": "skipped"}
    bottom = [0] * len(labels)
    for k in ("passed", "env_red", "failed", "skipped"):
        vals = series[k]
        ax.bar(labels, vals, bottom=bottom, color=colors[k], label=names[k], width=0.55)
        for i, (b, v) in enumerate(zip(bottom, vals)):
            if v:
                ax.text(i, b + v / 2, str(v), ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, tot in enumerate(bottom):
        ax.text(i, tot + 2, f"total={tot}", ha="center", fontsize=9)
    ax.set_ylabel("testcases")
    ax.set_title("TensorPhysics core suite status signature (read from junit XML)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, max(bottom) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_suite_status.png")
    plt.close(fig)
    return series


def _triples(node, acc):
    if isinstance(node, dict):
        if {"v", "f", "k"} <= set(node):
            acc.append(node)
            return acc
        for v in node.values():
            _triples(v, acc)
    elif isinstance(node, list):
        for x in node:
            _triples(x, acc)
    return acc


def fig_triples() -> dict:
    baseline = json.loads(BASELINE.read_text("utf-8"))
    sections: dict[str, dict[str, int]] = {}
    for name, block in baseline["gaps"].items():
        for t in _triples(block, []):
            kind = "external" if str(t["f"]).startswith(EXTERNAL_ROOTS) else "repo"
            sections.setdefault(name, {"repo": 0, "external": 0})[kind] += 1
    for t in _triples(baseline["reused_verified_anchors"], []):
        kind = "external" if str(t["f"]).startswith(EXTERNAL_ROOTS) else "repo"
        sections.setdefault("reused_verified_anchors", {"repo": 0, "external": 0})[kind] += 1
    repo = sum(v["repo"] for v in sections.values())
    ext = sum(v["external"] for v in sections.values())

    fig, ax = plt.subplots(figsize=(7.8, 3.9), dpi=160)
    names_ = list(sections)
    r = [sections[g]["repo"] for g in names_]
    e = [sections[g]["external"] for g in names_]
    ax.barh(names_, r, color="#1565c0", label=f"repo-local, hard-verified (n={repo})")
    ax.barh(names_, e, left=r, color="#f9a825",
            label=f"external /nfs, env-gated offline (n={ext})")
    for i, (a, b) in enumerate(zip(r, e)):
        if a:
            ax.text(a / 2, i, str(a), ha="center", va="center", color="white", fontsize=9)
        if b:
            ax.text(a + b / 2, i, str(b), ha="center", va="center", color="black", fontsize=9)
    ax.set_xlabel("baseline triples (value-file-key)")
    ax.set_title(f"SIR-0 contract triples by evidence reachability (n={repo + ext})")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_triple_availability.png")
    plt.close(fig)
    return {"total": repo + ext, "repo": repo, "external": ext, "sections": sections}


def fig_durations(junit: Path, top: int = 15) -> list[dict]:
    dur = parse(junit)["durations"]
    items = sorted(dur.items(), key=lambda kv: kv[1], reverse=True)[:top]
    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=160)
    names = [k.split("::")[-1] for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    ax.barh(names, vals, color="#4527a0")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f}s", va="center", fontsize=8)
    ax.set_xlabel("wall clock (s)")
    ax.set_title("Top-15 slowest testcases (junit read; patrol budget localization)")
    ax.set_xlim(0, max(vals) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_top_durations.png")
    plt.close(fig)
    return [{"test": k, "s": round(v, 2)} for k, v in items]


def main() -> None:
    junits = sorted(ART.glob("junit_*.xml"))
    if not junits:
        raise SystemExit("no junit artifacts under artifacts/tests")
    labels = ["run1 (baseline archive)", "run2 (drift patrol)"][: len(junits)]
    runs = list(zip(labels, junits))
    status = fig_suite_status(runs)
    cls = fig_triples()
    slowest = fig_durations(junits[0])
    probe = ART / "probe_prefix_contract.xml"
    meta = {
        "junits": [j.name for j in junits],
        "status_counts": status,
        "triple_classification": cls,
        "slowest_top15_run1": slowest,
        "prefix_probe": parse(probe)["counts"] if probe.is_file() else None,
        "figures": ["fig_suite_status.png", "fig_triple_availability.png",
                    "fig_top_durations.png"],
    }
    (OUT / "figure_data.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "triples": {k: cls[k] for k in ("total", "repo", "external")},
                      "prefix_probe": meta["prefix_probe"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()