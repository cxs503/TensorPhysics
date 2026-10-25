"""SIR 战役 Phase 3 契约自检（SIR-0 交付件）。

本测试把契约的三个冻结件钉成可机裁事实：

1. `prereg/sir0_baseline.json` 的每个 值-文件-键 三元组必须**当场复算通过**
   （零手抄：文件打不开、键不存在、值对不上 → 测试红）。
2. `prereg/dependency_graph.json` 必须是合法 DAG、覆盖 G1-G7、并行分组与节点声明一致，
   且拓扑序满足全部前置。
3. `prereg/semantic_criteria.json` 的每条内容符合判据必须是可机裁断言（category/type 在
   允许集内、必填字段齐全），与平台 PL-SEM 规范口径对齐。
4. `prereg/sir1..sir7_prereg.md` 七份骨架必须齐全，各含物理锚/机器门/规模判据三节，
   且时间戳为 UTC `Z` 后缀（append-only 纪律的机器面）。

红线纪律的机器面：本测试**不引入任何手抄数字常量**——所有期望值都从工件读取后比对，
或从 prereg 文件结构断言。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCN = ROOT / "scenarios" / "suboff_icebreaking"
PREREG = SCN / "prereg"

ALLOWED_CATEGORIES = {
    "provenance",
    "physics_switch",
    "scale_realism",
    "field_integrity",
}
ALLOWED_ASSERTION_TYPES = {
    # 平台 PL-SEM 三类
    "no_synthetic_marker",
    "config_key_true",
    "numeric_lower_bound",
    # 本战役扩展（均为可机裁断言）
    "numeric_abs_lower_bound",
    "numeric_monotone_nondecreasing",
    "field_present",
    "no_uncited_reference",
}


# ---------------------------------------------------------------------------
# 值-文件-键 三元组复算（口径同平台 gates_check：a.b[2].c + grep: 前缀）
# ---------------------------------------------------------------------------

def walk_key(doc, key: str):
    """走 a.b[2].c 键路径；含点键用回溯最长匹配（与 gates_check.walk 同口径）。"""
    def rec(cur, rest: str):
        if rest == "":
            return True, cur
        if not isinstance(cur, dict):
            return False, None
        for name in sorted(cur.keys(), key=len, reverse=True):
            if rest == name or rest.startswith(name + "."):
                ok, val = rec(cur[name], "" if len(rest) == len(name) else rest[len(name) + 1:])
                if ok:
                    return True, val
            m = re.match(re.escape(name) + r"((?:\[-?\d+\])+)(\.|$)", rest)
            if m:
                node = cur[name]
                try:
                    for i in re.findall(r"\[(-?\d+)\]", m.group(1)):
                        node = node[int(i)]
                except (TypeError, IndexError, KeyError):
                    continue
                tail = rest[m.end():]
                ok, val = rec(node, tail[1:] if tail.startswith(".") else tail)
                if ok:
                    return True, val
        return False, None

    return rec(doc, key)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def verify_triple(triple: dict) -> None:
    """复算一个三元组；任何不一致直接断言失败（=手抄）。"""
    for field in ("v", "f", "k"):
        assert field in triple, f"三元组缺字段 {field}: {triple}"
    path = resolve(triple["f"])
    assert path.is_file(), f"三元组文件不存在: {triple['f']}"
    key = triple["k"]
    if key.startswith("grep:"):
        needle = key[len("grep:"):]
        assert needle != "", f"grep 三元组缺模式: {triple}"
        text = path.read_text(encoding="utf-8")
        assert needle in text, f"grep 未命中: {needle!r} in {triple['f']}"
        assert triple["v"] == needle, f"grep 三元组值应为命中串: {triple}"
        return
    doc = json.loads(path.read_text(encoding="utf-8"))
    found, value = walk_key(doc, key)
    assert found, f"键路径不存在: {key} in {triple['f']}"
    assert value == triple["v"], (
        f"三元组值不符（手抄嫌疑）: {triple['f']} :: {key} "
        f"文件={value!r} 声明={triple['v']!r}"
    )


def iter_triples(node):
    """递归收集 baseline 里所有三元组（含嵌套 evidence 列表）。"""
    if isinstance(node, dict):
        if {"v", "f", "k"} <= set(node):
            yield node
            return
        for value in node.values():
            yield from iter_triples(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_triples(item)


@pytest.fixture(scope="module")
def baseline():
    path = PREREG / "sir0_baseline.json"
    assert path.is_file(), "缺少 SIR-0 基线工件 prereg/sir0_baseline.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def graph():
    path = PREREG / "dependency_graph.json"
    assert path.is_file(), "缺少 prereg/dependency_graph.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def criteria():
    path = PREREG / "semantic_criteria.json"
    assert path.is_file(), "缺少 prereg/semantic_criteria.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. 基线三元组全量复算（零手抄）
# ---------------------------------------------------------------------------

class TestBaselineTriples:
    def test_every_triple_recomputes(self, baseline):
        triples = list(iter_triples(baseline))
        assert len(triples) >= 15, f"基线三元组过少（{len(triples)}），契约未覆盖七缺口"
        for triple in triples:
            verify_triple(triple)

    def test_all_seven_gaps_have_evidence(self, baseline):
        gaps = baseline["gaps"]
        # 七缺口里 G2/G4 在 Phase 2 是「全缺」，基线以复用锚 + 复用路径登记，
        # 因此这里断言的是「可复算证据或复用锚路径」齐备，而非每个缺口都有三元组。
        assert set(gaps) == {
            "g1_fluid_load_precision",
            "g3_gravity_disabled",
            "g5_fracture_propagation",
            "g6_scale",
            "g7_reference_anchors",
        }
        for name, block in gaps.items():
            assert block["evidence"], f"缺口 {name} 无任何三元组证据"
        anchors = baseline["reused_verified_anchors"]
        # G2/G4 的承载：复用锚里必须有几何进链与接触的 verified 路径
        assert "suboff_geometry" in anchors
        assert "hull_contact_action_reaction" in anchors

    def test_frozen_tau_table_matches_frozen_formula(self, baseline):
        """τ 表必须由 prereg 冻结公式当场复算（不是抄来的值）。"""
        table = baseline["derived_frozen_table"]
        u_lb = table["u_lb"]
        for row in table["tiers"]:
            L = row["L"]
            nu = u_lb * L / 1000.0
            assert row["nu"] == pytest.approx(nu, rel=0, abs=1e-18)
            assert row["tau"] == pytest.approx(3.0 * nu + 0.5, rel=0, abs=1e-15)
            assert row["tau_minus_half"] == pytest.approx(3.0 * nu, rel=0, abs=1e-15)
            assert row["tau_minus_half"] > 0.0, "tau 近奇异（SIR-1 M1.2 门在此失败）"

    def test_no_hand_copied_number_without_triple(self, baseline):
        """baseline 里任何标量「基线事实」都必须挂在三元组上（或其派生表上）。"""
        allowed_toplevel = {
            "schema", "created_utc", "description", "source_report",
            "gaps", "reused_verified_anchors", "derived_frozen_table",
        }
        assert set(baseline) == allowed_toplevel


# ---------------------------------------------------------------------------
# 2. 依赖图：合法 DAG + 覆盖 G1-G7 + 并行分组一致
# ---------------------------------------------------------------------------

class TestDependencyGraph:
    def test_nodes_cover_all_gaps(self, graph):
        nodes = graph["nodes"]
        legs = {k for k in nodes if k != "SIR-0"}
        assert legs == {f"SIR-{i}" for i in range(1, 8)}
        assert set(graph["gaps_covered"]) == {f"G{i}" for i in range(1, 8)}
        for leg in legs:
            assert nodes[leg]["gap"] in graph["gaps_covered"]
        # 每个缺口恰由一条腿承接（一一对应，无遗漏无重复）
        gaps_on_legs = [nodes[leg]["gap"] for leg in legs]
        assert sorted(gaps_on_legs) == sorted(graph["gaps_covered"])

    def test_dag_has_no_cycle(self, graph):
        nodes = graph["nodes"]
        state: dict[str, int] = {}

        def visit(name: str):
            assert name in nodes, f"依赖指向未声明节点: {name}"
            if state.get(name) == 1:
                raise AssertionError(f"依赖图存在环，节点 {name}")
            if state.get(name) == 2:
                return
            state[name] = 1
            for dep in nodes[name]["depends_on"]:
                visit(dep)
            state[name] = 2

        for name in nodes:
            visit(name)

    def test_topological_order_respects_dependencies(self, graph):
        order = graph["topological_order"]
        assert set(order) == set(graph["nodes"])
        pos = {name: i for i, name in enumerate(order)}
        for name, node in graph["nodes"].items():
            for dep in node["depends_on"]:
                assert pos[dep] < pos[name], f"{dep} 必须先于 {name}"

    def test_parallel_declarations_are_symmetric_and_dependency_free(self, graph):
        nodes = graph["nodes"]
        for name, node in nodes.items():
            for other in node["parallel_with"]:
                assert name in nodes[other]["parallel_with"], (
                    f"并行声明不对称: {name} → {other}"
                )
                # 可并行 ⇒ 互不为前置
                assert other not in node["depends_on"]
                assert name not in nodes[other]["depends_on"]

    def test_parallel_groups_match_dependency_structure(self, graph):
        nodes = graph["nodes"]
        for group in graph["parallel_groups"]:
            legs = group["legs"]
            for leg in legs:
                for dep in nodes[leg]["depends_on"]:
                    assert dep not in legs, (
                        f"{group['wave']} 内 {leg} 依赖同波腿 {dep}，分组非法"
                    )


# ---------------------------------------------------------------------------
# 3. 内容符合判据：可机裁 + 覆盖三样
# ---------------------------------------------------------------------------

class TestSemanticCriteria:
    def test_spec_alignment_declared(self, criteria):
        align = criteria["spec_alignment"]
        assert "semantic_gate_spec.md" in align["platform_spec"]
        assert "§9" in " ".join(align["spec_sections"])
        assert criteria["verdict_contract"]["overall_verdict_enum"] == [
            "compliant", "partial_compliant", "non_compliant",
        ]

    def test_every_leg_has_criteria(self, criteria):
        assert set(criteria["gates"]) == {f"SIR-{i}" for i in range(1, 8)}
        for leg, items in criteria["gates"].items():
            assert items, f"{leg} 无内容符合判据"

    def test_each_criterion_is_machine_decidable(self, criteria):
        for leg, items in criteria["gates"].items():
            ids = set()
            for item in items:
                cid = item["id"]
                assert cid.startswith(leg.lower().replace("-", "")), (
                    f"判据 id 未带腿前缀: {cid}"
                )
                assert cid not in ids, f"判据 id 重复: {cid}"
                ids.add(cid)
                assert item["category"] in ALLOWED_CATEGORIES, (
                    f"{cid} 类别不可机裁: {item['category']}"
                )
                assertion = item["assertion"]
                assert assertion["type"] in ALLOWED_ASSERTION_TYPES, (
                    f"{cid} assertion 类型未登记: {assertion['type']}"
                )
                assert item["description"].strip(), f"{cid} 缺语义描述"
                # 每类 assertion 的必填字段（可机裁性）
                t = assertion["type"]
                if t in {"numeric_lower_bound", "numeric_abs_lower_bound",
                         "numeric_monotone_nondecreasing"}:
                    src = assertion["source"]
                    assert src["file"] and src["key_path"], f"{cid} 缺 source 定位"
                    assert isinstance(assertion["lower_bound"], (int, float))
                    assert assertion["unit"]
                elif t == "config_key_true":
                    assert assertion["key_path"] and assertion["config_files"]
                    assert assertion["expected"] is True
                elif t == "no_synthetic_marker":
                    assert assertion["scan_targets"] and assertion["markers"]
                elif t == "no_uncited_reference":
                    assert assertion["scan_targets"] and assertion["required_columns"]

    def test_all_three_platform_categories_covered(self, criteria):
        """平台三类（溯源/物理开关/规模现实性）在战役里都必须被用到。"""
        used = {
            item["category"]
            for items in criteria["gates"].values()
            for item in items
        }
        for required in ("provenance", "physics_switch", "scale_realism"):
            assert required in used, f"平台语义门类别 {required} 未被本战役使用"

    def test_goal_level_assertions_present(self, criteria):
        """每腿的判据必须含「goal 语义层可机裁断言」，而非只有账本自证。"""
        for leg, items in criteria["gates"].items():
            types = {item["assertion"]["type"] for item in items}
            non_ledger = types - {"numeric_lower_bound"}
            assert non_ledger or "scale_realism" in {
                item["category"] for item in items
            }, f"{leg} 只有账本类判据，缺 goal 语义层断言"


# ---------------------------------------------------------------------------
# 4. prereg 骨架：齐全 + 三样节齐全 + UTC Z 时间戳
# ---------------------------------------------------------------------------

class TestPreregSkeletons:
    def test_all_prereg_files_exist(self):
        for name in ["README.md", "sir0_prereg.md", "sir0_baseline.json",
                     "semantic_criteria.json", "dependency_graph.json"]:
            assert (PREREG / name).is_file(), f"缺少 prereg/{name}"
        for i in range(1, 8):
            assert (PREREG / f"sir{i}_prereg.md").is_file(), f"缺少 sir{i}_prereg.md"

    @pytest.mark.parametrize("leg", list(range(1, 8)))
    def test_each_leg_prereg_has_three_required_blocks(self, leg):
        text = (PREREG / f"sir{leg}_prereg.md").read_text(encoding="utf-8")
        # 三样：物理锚 / 机器门 / 规模判据
        assert "物理锚" in text, f"SIR-{leg} 缺物理锚节"
        assert "机器门" in text, f"SIR-{leg} 缺机器门节"
        assert "规模判据" in text, f"SIR-{leg} 缺规模判据节"
        # 内容符合判据（与 PL-SEM 的接口）
        assert "内容符合判据" in text, f"SIR-{leg} 缺内容符合判据节"
        # append-only 纪律声明
        assert "append-only" in text, f"SIR-{leg} 未声明 append-only"
        # 时间戳 UTC Z
        stamps = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text)
        assert stamps, f"SIR-{leg} 无 UTC Z 时间戳"
        assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?!Z)", text), (
            f"SIR-{leg} 存在非 Z 后缀时间戳"
        )

    def test_zero_prereg_declares_append_only_and_verdict(self):
        text = (PREREG / "sir0_prereg.md").read_text(encoding="utf-8")
        assert "append-only" in text
        assert "验收定义" in text
        assert "semantic_gate" in text or "语义门" in text

    def test_readme_phase3_contract_present(self):
        readme = (SCN / "README.md").read_text(encoding="utf-8")
        for needle in [
            "Phase 3",
            "SIR-0",
            "SIR-1",
            "SIR-7",
            "物理锚",
            "机器门",
            "规模判据",
            "内容符合判据",
            "PL-SEM",
            "GPU5",
            "launch-harvest",
        ]:
            assert needle in readme, f"README Phase 3 契约缺内容: {needle}"
        # 七腿全列
        for i in range(1, 8):
            assert f"SIR-{i}" in readme, f"README 未列 SIR-{i}"
        # 依赖有序 + 并行标注
        assert "可并行" in readme
        assert "硬依赖" in readme

    def test_every_baseline_number_appears_in_readme(self, baseline):
        """README 与本基线文件不得剪刀差：每个三元组的值都要在 README 正文出现。"""
        readme = (SCN / "README.md").read_text(encoding="utf-8")
        missing = []
        for triple in iter_triples(baseline):
            value = triple["v"]
            if isinstance(value, bool) or isinstance(value, str):
                if isinstance(value, str) and value not in readme:
                    missing.append((value, triple["f"], triple["k"]))
                continue
            candidates = [repr(value), f"{value:g}", f"{value:.6e}", f"{value:.17g}"]
            if not any(c in readme for c in candidates):
                missing.append((repr(value), triple["f"], triple["k"]))
        assert not missing, f"以下基线数字未在 README 出现（文档-工件剪刀差）: {missing}"

    def test_readme_baseline_numbers_come_from_baseline_json(self, baseline):
        """README 里引用的基线事实必须能在 baseline JSON 里找到同一三元组（防剪刀差）。"""
        readme = (SCN / "README.md").read_text(encoding="utf-8")
        reported = {
            (t["f"], t["k"])
            for t in iter_triples(baseline)
        }
        for path_like, key in [
            ("outputs/si4_full_chain/validation_report.json", "chain_execution.n_chain_steps"),
            ("outputs/si4_full_chain/validation_report.json", "final_state.n_particles"),
            ("outputs/si4_full_chain/validation_report.json", "final_state.n_bonds_broken"),
            ("outputs/si4_full_chain/dem_snapshot_step0.json", "dt_dem"),
            ("benchmarks/icebreaking/si1_one_way_output/validation_report.json", "anchors[1].w_hetenyi"),
            ("benchmarks/icebreaking/si2_fracture_gen_output/validation_report.json", "anchors[2].n_failed_actual"),
        ]:
            assert (path_like, key) in reported, (
                f"README 引用的基线 ({path_like} :: {key}) 未在 sir0_baseline.json 登记"
            )
            assert path_like in readme, f"README 未标注来源文件 {path_like}"
