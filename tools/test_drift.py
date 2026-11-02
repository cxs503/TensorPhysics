#!/usr/bin/env python3
"""测试基线漂移巡更工具（TensorPhysics）。

做三件事，全部零手抄（数字只从 pytest 的 junit XML 读）：

1. 跑测试套件，落 junit XML 到 `artifacts/tests/`；
2. 把结果压成「通过/失败签名」`artifacts/tests/baseline_signature.json`
   （nodeid → outcome），供后续巡更逐条 diff；
3. 与既有基线比对，输出新增真红 / 已修复红 / 新消失用例 / 环境红清单
   （环境红 = skip 原因以 `环境红` 开头的用例，即外部证据根不可见一类）。

用法：
    python3 tools/test_drift.py                    # 跑全量并与基线比对（无基线则建档）
    python3 tools/test_drift.py --update-baseline   # 跑全量并刷新基线签名
    python3 tools/test_drift.py --strict            # 外部证据不可见亦判红（集群模式）
    python3 tools/test_drift.py -- tests/test_x.py -k foo   # 只跑子集（不更新基线）
    python3 tools/test_drift.py --compare-only      # 不跑测试，仅重放最近一次 junit

退出码：0=无新增真红；1=有新增真红（或 --strict 下环境红转红）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "tests"
SIG = ART / "baseline_signature.json"
ENV_RED_PREFIX = "环境红"
SCHEMA = "tensorphysics-test-signature-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - 无 git 环境
        return "unknown"


def tool_versions() -> dict:
    import pytest  # 局部导入：仅上报版本
    return {"python": sys.version.split()[0], "pytest": pytest.__version__}


def run_pytest(extra: list[str], junit: Path, strict: bool) -> int:
    env = dict(os.environ)
    if strict:
        env["TP_STRICT_EXTERNAL_EVIDENCE"] = "1"
    cmd = [
        sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
        f"--junitxml={junit}",
    ] + extra
    print("[drift] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def parse_junit(junit: Path) -> dict:
    tree = ET.parse(junit)
    suite = tree.getroot()
    if suite.tag == "testsuites":  # 兼容多 suite
        suite = suite.find("testsuite")
    outcomes: dict[str, str] = {}
    env_gated: dict[str, str] = {}
    durations: dict[str, float] = {}
    for case in suite.iter("testcase"):
        # 口径：沿用 junit 原生 classname::name，全仓工具（巡更/报告）统一，不做二次改写
        nodeid = f"{case.get('classname', '')}::{case.get('name', '')}"
        durations[nodeid] = float(case.get("time") or 0.0)
        bad = None
        for tag in ("failure", "error"):
            node = case.find(tag)
            if node is not None:
                bad = tag
                break
        if bad:
            outcomes[nodeid] = "failed"
            continue
        skip = case.find("skipped")
        if skip is not None:
            msg = (skip.get("message") or "")
            if msg.startswith(ENV_RED_PREFIX):
                outcomes[nodeid] = "env_red"
                env_gated[nodeid] = msg
            else:
                outcomes[nodeid] = "skipped"
            continue
        outcomes[nodeid] = "passed"
    summary = {
        "total": len(outcomes),
        "passed": sum(v == "passed" for v in outcomes.values()),
        "failed": sum(v == "failed" for v in outcomes.values()),
        "skipped": sum(v == "skipped" for v in outcomes.values()),
        "env_gated": sum(v == "env_red" for v in outcomes.values()),
        "duration_s": round(sum(durations.values()), 2),
    }
    return {"outcomes": outcomes, "env_gated": env_gated,
            "durations": durations, "summary": summary}


def red_set(outcomes: dict) -> set[str]:
    return {k for k, v in outcomes.items() if v == "failed"}


def load_baseline() -> dict | None:
    if not SIG.is_file():
        return None
    return json.loads(SIG.read_text(encoding="utf-8"))


def diff(baseline: dict, current: dict) -> dict:
    prev = baseline.get("outcomes", {})
    cur = current["outcomes"]
    prev_red, cur_red = red_set(prev), red_set(cur)
    return {
        "new_red": sorted(cur_red - prev_red),
        "fixed_red": sorted((prev_red & set(cur)) - cur_red),
        "still_red": sorted(prev_red & cur_red),
        "new_tests": sorted(set(cur) - set(prev)),
        "removed_tests": sorted(set(prev) - set(cur)),
        "env_gated": sorted(current["env_gated"]),
        "summary": current["summary"],
        "baseline_summary": baseline.get("summary", {}),
        "baseline_utc": baseline.get("updated_utc"),
    }


def write_signature(result: dict, source: str, history: list) -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    sig = {
        "schema": SCHEMA,
        "updated_utc": utc_now(),
        "git_rev": git_rev(),
        "versions": tool_versions(),
        **result,
        "history": history,
    }
    sig.pop("durations_full", None)
    SIG.write_text(json.dumps(sig, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"[drift] 签名已写入 {SIG.relative_to(ROOT)}（source={source}）")
    return sig


def print_diff(d: dict) -> None:
    s = d["summary"]
    print("\n=== 通过/失败签名对照 ===")
    print(f"本次：passed={s['passed']} failed={s['failed']} skipped={s['skipped']} "
          f"env_red={s['env_gated']} total={s['total']} duration={s['duration_s']}s")
    b = d["baseline_summary"]
    if b:
        print(f"基线({d['baseline_utc']})：passed={b.get('passed')} failed={b.get('failed')} "
              f"skipped={b.get('skipped')} env_red={b.get('env_gated')} total={b.get('total')}")
    for label, key in (("新增真红", "new_red"), ("已修复红", "fixed_red"),
                       ("延续红", "still_red"), ("新增用例", "new_tests"),
                       ("消失用例", "removed_tests"), ("环境红(外部证据不可见)", "env_gated")):
        items = d[key]
        print(f"{label}: {len(items)}")
        for it in items:
            print(f"  - {it}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="测试基线漂移巡更")
    ap.add_argument("--update-baseline", action="store_true", help="刷新基线签名")
    ap.add_argument("--strict", action="store_true", help="外部证据不可见亦判红")
    ap.add_argument("--compare-only", action="store_true", help="不跑测试，仅重放最近 junit")
    ap.add_argument("extra", nargs="*", help="透传给 pytest（用 -- 分隔）")
    args = ap.parse_args(argv)

    ART.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    junit = ART / f"junit_{stamp}.xml"
    baseline = load_baseline()

    if args.compare_only:
        candidates = sorted(ART.glob("junit_*.xml"))
        if not candidates:
            print("[drift] 无 junit 可重放，先跑一次", file=sys.stderr)
            return 2
        junit = candidates[-1]
    else:
        rc = run_pytest(args.extra, junit, args.strict)
        print(f"[drift] pytest exit={rc} junit={junit.relative_to(ROOT)}")

    current = parse_junit(junit)
    if baseline is None:
        print("[drift] 仓库内无既有基线记录 → 建档")
        write_signature(current, junit.name, history=[{
            "utc": utc_now(), "git_rev": git_rev(), "summary": current["summary"],
            "source": junit.name, "note": "建档（首次巡更）",
        }])
        print_diff(diff({"outcomes": {}, "summary": {}}, current))
        return 1 if current["summary"]["failed"] else 0

    d = diff(baseline, current)
    print_diff(d)
    history = list(baseline.get("history", []))
    history.append({"utc": utc_now(), "git_rev": git_rev(),
                    "summary": current["summary"], "source": junit.name,
                    "new_red": d["new_red"]})
    if args.update_baseline:
        write_signature(current, junit.name, history=history)
    else:
        ART.joinpath(f"drift_{stamp}.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[drift] 未刷新基线（基线只由 --update-baseline 覆盖）；"
              f"diff 落 {ART.relative_to(ROOT)}/drift_{stamp}.json")

    if d["new_red"]:
        return 1
    if args.strict and current["summary"]["env_gated"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))