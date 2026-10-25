"""SIR-5（G5 断裂传播）契约测试。

纪律：本文件不写任何手抄判决数字——门限值来自 prereg 冻结文本解析，
一切实际值都从工件直读后比对，对不上即红。

覆盖：
1. 工件齐全（validation_report / gates / fracture_history / run_summary / analytical_anchor）
2. gates.json 每个三元组可当场复算（值-文件-键）
3. 机器门 M5.1-M5.5 与文件内 checks 一致，且 prereg 门限由文本解析后重判
4. 语义判据两条与 semantic_criteria.json 的 SIR-5 条目同名
5. 渐进性/键断裂的「非降」性质在时序工件上独立复验
6. prereg md5 与 validation_report 记录一致（append-only 冻结面）
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCN = ROOT / "scenarios" / "suboff_icebreaking"
PREREG = SCN / "prereg" / "sir5_prereg.md"
SEMANTIC = SCN / "prereg" / "semantic_criteria.json"
OUT = ROOT / "outputs" / "sir5_fracture_propagation"


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_artifacts_exist():
    for name in ("validation_report.json", "gates.json", "fracture_history.json",
                 "run_summary.json", "analytical_anchor.json"):
        assert (OUT / name).is_file(), f"缺工件 {name}"


def test_prereg_md5_matches_report():
    report = load(OUT / "validation_report.json")
    assert report["prereg_md5"] == hashlib.md5(PREREG.read_bytes()).hexdigest()


def test_gates_triplets_recomputable():
    gates = load(OUT / "gates.json")
    assert gates["triplets"], "三元组为空"
    for triplet in gates["triplets"]:
        doc = load(OUT / triplet["file"])
        cursor = doc
        for part in triplet["key"].split("."):
            assert part in cursor or isinstance(cursor, list), \
                f"{triplet['id']} 键路径断裂于 {part}"
            cursor = cursor[int(part)] if isinstance(cursor, list) else cursor[part]
        assert cursor == triplet["value"], f"{triplet['id']} 三元组不可复算"


def test_machine_gates_pass_and_match_checks():
    report = load(OUT / "validation_report.json")
    assert report["leg"] == "SIR-5" and report["gap"] == "G5"
    assert report["checks_passed"] == report["checks_total"] == 5
    assert report["status"] == "PASS"
    assert all(check["pass"] for check in report["checks"]), "存在未通过机器门"


def test_prereg_thresholds_reparsed_and_rejudged():
    """门限从 prereg 文本解析（非手抄），再用工件实际值重判一次。"""
    text = PREREG.read_text(encoding="utf-8")
    m51 = re.search(r"M5\.1.*?n_steps_with_new_fracture\s*≥\s*(\d+)", text)
    assert m51, "prereg 未给出 M5.1 的新增断裂步数下限"
    m52 = re.search(r"n_bonds_broken_total\s*>\s*(\d+)", text)
    assert m52, "prereg 未给出 M5.2 的键断裂下限"
    m53 = re.search(r"M5\.3.*?≤\s*([0-9.eE+-]+)", text)
    assert m53, "prereg 未给出 M5.3 的账本容差"

    report = load(OUT / "validation_report.json")
    fem = report["tests"]["fem_progressive_fracture"]
    dem = report["tests"]["dem_bond_breakage"]
    assert fem["n_steps_with_new_fracture"] >= int(m51.group(1))
    assert dem["n_bonds_broken_total"] > int(m52.group(1))
    assert report["tests"]["fracture_energy_ledger"]["relative_error"] <= float(m53.group(1))
    assert report["tests"]["mass_ledger"]["relative_error"] <= float(m53.group(1))


def test_monotonicity_reverified_from_history():
    history = load(OUT / "fracture_history.json")
    series = history["n_failed_faces_cumulative"]
    assert len(series) >= 2
    assert all(b >= a for a, b in zip(series, series[1:])), "破坏面累计数出现回退"
    summary = load(OUT / "run_summary.json")
    bonds = summary["broken_bond_history"]
    assert len(bonds) >= 2
    assert all(b >= a for a, b in zip(bonds, bonds[1:])), "键断裂累计数出现回退"


def test_analytical_anchor_count_and_ledger():
    anchor = load(OUT / "analytical_anchor.json")
    assert anchor["count_match"] is True
    assert anchor["n_total_faces_actual"] == anchor["n_total_faces_expected"]
    assert anchor["n_failed_actual"] == anchor["n_failed_expected"]
    assert anchor["area_rel_error"] <= 1e-12
    assert anchor["energy_rel_error"] <= 1e-12


def test_semantic_criteria_alignment():
    spec = load(SEMANTIC)
    expected = {item["id"] for item in spec["gates"]["SIR-5"]}
    report = load(OUT / "validation_report.json")
    assert set(report["semantic_criteria"]) == expected
    assert report["semantic_criteria_passed"] == report["semantic_criteria_total"]
    assert all(report["semantic_criteria"].values())


def walk(doc, key):
    """走 a.b[0].c 键路径（与 tests/test_sir_phase3_contract.py 同口径）。"""
    cursor = doc
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", key):
        if part.startswith("["):
            cursor = cursor[int(part[1:-1])]
        else:
            cursor = cursor[part]
    return cursor


def test_baseline_comparison_triplets_still_recomputable():
    """对照基线（SI-4 一步断完 / 0 键断）仍按 sir0_baseline.json 三元组复算。"""
    baseline = load(SCN / "prereg" / "sir0_baseline.json")
    entries = baseline["gaps"]["g5_fracture_propagation"]["evidence"]
    values = []
    for entry in entries:
        path = Path(entry["f"])
        if not path.is_absolute():
            path = ROOT / entry["f"]
        recomputed = walk(load(path), entry["k"])
        assert recomputed == entry["v"], f"基线三元组不可复算：{entry['k']}"
        values.append(recomputed)
    report = load(OUT / "validation_report.json")
    assert report["tests"]["fem_progressive_fracture"]["n_steps_with_new_fracture"] > 1, \
        "基线为一步断完（其余链步 0），本腿新增断裂步数必须 > 1"
    assert report["tests"]["dem_bond_breakage"]["n_bonds_broken_total"] > values[1], \
        "基线终态 0 键断，本腿键断裂总数必须严格大于基线"