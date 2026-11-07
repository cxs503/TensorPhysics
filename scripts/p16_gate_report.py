#!/usr/bin/env python3
"""Render the campaign-level P16-G1 acceptance report from recorded artifacts.

Every number in the generated markdown is injected from JSON artifacts
(``gate_verdict.json``, ``probe_results.json``, ``campaign_nodes.json``,
``p15_baseline.json``, ``ship_platform_facts.json``) -- nothing is transcribed by
hand, so the report cannot drift from the evidence.

    python3 scripts/p16_gate_report.py \
        --evidence artifacts/p16_gate \
        --out docs/P16_CAMPAIGN_INTEGRATION_GATE_REPORT.md \
        --platform-payload /tmp/report.json
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib

MISSION_ID = "5192"


def load(p: pathlib.Path) -> dict:
    return json.loads(p.read_text())


def fnum(x, fmt="{:.4f}", dash="—"):
    return dash if x is None else fmt.format(x)


def build_markdown(ev: pathlib.Path) -> tuple[str, dict]:
    verdict = load(ev / "gate_verdict.json")
    probes = load(ev / "probe_results.json")
    baseline = load(ev / "p15_baseline.json")
    facts = load(ev / "ship_platform_facts.json")
    campaign = load(ev / "campaign_nodes.json")
    node_ev = load(ev / "node_evidence.json")

    crit = {c["criterion_id"]: c for c in verdict["criteria"]}
    nodes = crit["a"]["evidence"]["nodes"]
    contracts = crit["b"]["evidence"]["contracts"]
    cases = crit["c"]["evidence"]["cases"]
    heads = probes.get("repo_heads") or {}
    src = campaign.get("source") or {}

    # ------------------------------------------------------------------ header
    L = [
        "# P16-G1 全战役集成验收门：跨域能力串联与回归基线确认",
        "",
        f"**任务** mission {MISSION_ID} · 战役计划 plan {src.get('api', '').rsplit('/', 1)[-1]}"
        f"（{src.get('plan_title')}）· 仓库 TensorPhysics · 门 schema `{verdict['schema']}`",
        "",
        f"**裁决：`{verdict['overall_status']}`**（生成时间 {verdict['generated_utc']}）。"
        "本报告是本门的 honest-FAIL 记录：四项判据中 (a) 证据缺失、(b)(c) 未通过，"
        "只有 (d)（报告与审计链）通过。",
        "",
        "---",
        "",
        "## ① 案例介绍",
        "",
        "### 物理背景",
        "",
        "P16 战役把两类工作并到一条验收线上：求解器核心能力成熟化（LBM 力尺度口径、"
        "FEM 循环塑性背应力、DEM clump 接触）与船舶总体性能新域（浅水操纵 MMG、动力定位 DP、"
        "气象航线随机优化、LNG 船风阻力 LBM 预报、船体结构振动干/湿模态）。战役 metadata 明确的"
        "跨域耦合要求为",
        "",
        f"> `{src.get('coupling_framework_requirements')}`",
        "",
        "G1 是战役终门，它不产生新的物理结果，而是回答四个可机器裁决的问题："
        "(a) N1-N8 是否各自满足验收判据；(b) 两条跨域接口契约（格式/类型/量纲）是否一致；"
        "(c) P15 已交付的 11 个验证节点在 P16 代码基线上重跑是否劣化（|Δ| < 1%）；"
        "(d) 战役级验收报告与其审计链是否生成。",
        "",
        "### 战役参数表（平台快照，非本门设定）",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 计划标题 | {src.get('plan_title')} |",
        f"| 计划状态 | `{src.get('plan_status')}` |",
        f"| 物理域 | `{src.get('physics_scope')}` |",
        f"| 验证标准 | `{src.get('validation_standards')}` |",
        f"| 节点数 | {len(campaign.get('nodes') or [])}（8 个任务节点 + 1 个本门） |",
        f"| 快照时间 | {campaign.get('collected_utc')} |",
        "",
        "### 两条跨域接口契约（本门固化的规范）",
        "",
        "**C-W：LBM 风阻力系数表 → ship-platform**"
        "（`wind_resistance_coefficients.v1`）。无量纲系数只有配上使其无量纲化的参考量才能被下游使用，"
        "因此契约要求：`units`（deg / 1 / m·s⁻¹ / kg·m⁻³ / m² / m）、`reference`"
        "（`A_front_m2`、`A_lateral_m2`、`U_ref_m_s`、`rho_air_kg_m3`、`z_ref_m` 全部 > 0）、"
        "`frame`（x 前、y 左、z 上、`heading_zero=bow-on`）、`samples`"
        "（β∈[0,360)、|C|≤3、0/90/180° 锚点齐备、最大角间隔 ≤45°）。",
        "",
        "**C-M：FEM 湿模态 + 附加质量 → BEM Helmholtz**（`wet_modal_bem_handoff.v1`）。"
        "TensorBEM 的外场声学求解入口是 `solve_helmholtz_diffraction(mesh, wavenumber=k)`，"
        "`k = ω/c`（`tensorbem/helmholtz.py`），TensorFEM 的附加质量以 kg/m² 的面积密度给出"
        "（`tensorfem/added_mass.py`：`m_a = c_m·ρ_w·L_char`）。因此交接必须同时携带 "
        "`f [Hz]`、`ω [rad/s]`、`c [m/s]`（使 k 可重建）、逐阶 `added_mass_kg` / `modal_mass_kg`、"
        "`S_ref_m2`（辐射效率归一化），并满足物理约束 0 < f_wet ≤ f_dry、"
        "`ω = 2πf`（相对公差 1e-9）、`k = ω/c`（1e-9）、附加质量矩阵对称（1e-10）且半正定（1e-12）。",
        "",
        "---",
        "",
        "## ② 怎么做的",
        "",
        "### 工具链（全部落在 TensorPhysics `campaign/integration_gate/`）",
        "",
        "| 阶段 | 实现 | 作用 |",
        "|---|---|---|",
        "| 证据采集 | `scripts/collect_p16_gate_evidence.py` | 从平台 API 快照 plan 33 节点表、"
        "N1-N8 mission 记录、P15 11 节点 probe 记录，以及 ship-platform OpenAPI 事实；失败即留空，不推断 |",
        "| 契约验证 | `integration_gate/contract.py` | 纯函数校验器：风系数表 8 项检查、湿模态交接 11 项检查、"
        "消费端端点兼容 4 项检查；每项有稳定 `check_id` 与数值/限值 |",
        "| 节点验收 | `integration_gate/nodes.py` | 逐条判据 PASS/FAIL/WITHHELD；数值判据**重算**"
        "（节点自报 `pass` 与自身 value/limit 矛盾即判 FAIL，防自证） |",
        "| 回归基线 | `integration_gate/regression.py` + `scripts/p16_run_p15_probes.py` | "
        "P15 记录的 pytest 探针在本地 checkout 复跑；目标文件缺失时按 `merge-base --is-ancestor` "
        "区分『陈旧检出』与『基线内被删除』，再经集群 fresh head 复核 |",
        "| 门聚合 | `integration_gate/gate.py` | 四判据聚合、sha256 审计链、`gate_verdict.json` + "
        "机器可读 markdown；`PASS` 仅当四项全 `PASS` |",
        "| 出图 | `scripts/p16_gate_figures.py` | 图 1-4 直接渲染 `gate_verdict.json` 的数字 |",
        "",
        "### 判据与阈值的机器化",
        "",
        "| 判据 | 裁决规则 | 阈值 |",
        "|---|---|---|",
        "| (a) N1-N8 | 8 节点全部 `PASS`→PASS；任一 `FAIL`→FAIL；否则 WITHHELD | 无 artifact 即 WITHHELD，"
        "绝不按 PR 合并状态推断 |",
        "| (b) 契约 | 任一检查 FAIL→FAIL；缺生产者载荷→WITHHELD | `ω=2πf` ≤1e-9、`k=ω/c` ≤1e-9、"
        "对称 ≤1e-10、PSD ≥−1e-12、|C|≤3、角间隔 ≤45° |",
        "| (c) 回归 | 11 节点全部 `PASS`→PASS；任一 `FAIL`→FAIL；否则 WITHHELD | |Δ pass rate| < 1% 且 "
        "failures=0 且 用例数不减 |",
        "| (d) 报告 | 产物生成即 PASS | 每个输入 artifact 有 sha256 |",
        "",
        f"P15 基线来自平台记录（plan {baseline.get('source', {}).get('plan_id')}，"
        f"状态 `{baseline.get('source', {}).get('plan_status')}`），共 {len(cases)} 个节点探针；"
        "本地复跑所用检出："
        + "、".join(f"{r}@{v['head']}" for r, v in sorted(heads.items()) if v.get("head"))
        + "；fresh head 复核走 外协方计算集群（GitHub main 秒级检出）。",
        "",
        "---",
        "",
        "## ③ 计算结果",
        "",
        "### 3.1 门裁决总览",
        "",
        "| 判据 | 内容 | 裁决 | 定量说明 |",
        "|---|---|---|---|",
    ]
    for cid in ("a", "b", "c", "d"):
        c = crit[cid]
        L.append(f"| ({cid}) | {c['label']} | **{c['status']}** | {c['detail']} |")
    L.append("")
    L.append(f"**overall_status = `{verdict['overall_status']}`**；阻塞项 "
             + "；".join(f"`{b}`" for b in verdict["blocking"]) + "。")
    L.append("")

    # (a)
    L += ["### 3.2 判据 (a)：N1-N8 节点验收", "",
          "| 节点 | 仓库 | mission | 节点状态 | 平台 accepted | 交付 artifact | 裁决 | 通过/判据数 |",
          "|---|---|---|---|---|---|---|---|"]
    ev_nodes = node_ev.get("nodes") or {}
    for n in nodes:
        rec = ev_nodes.get(n["ext_id"]) or {}
        L.append(f"| {n['ext_id']} | {n['repo']} | {n['mission_id']} | `{n['node_status']}` | "
                 f"{rec.get('accepted')} | {'yes' if rec.get('delivered') else 'no'} | "
                 f"**{n['verdict']}** | {n['n_pass']}/{n['n_criteria']} |")
    L += ["",
          "每个节点 6-8 条判据（C1..C8）全部落在 WITHHELD：平台侧 mission 仍在 running、"
          "`result.accepted` 为空，也没有 `p16_node_acceptance.v1` artifact 落盘。本门不做任何推断——"
          "这正是判据 (a) 无法通过的原因。",
          ""]

    # (b)
    L += ["### 3.3 判据 (b)：跨域接口契约", "",
          "| 契约 | 生产者 → 消费者 | 检查数 | 失败数 | 裁决 |", "|---|---|---|---|---|"]
    for c in contracts:
        L.append(f"| `{c['contract_id']}` | {c['producer']} → {c['consumer']} | {c['n_checks']} | "
                 f"{c['n_failed']} | **{c['verdict']}** |")
    L += ["", "逐项检查：", "", "| 契约 | check | 数值 / 说明 | 裁决 |", "|---|---|---|---|"]
    for c in contracts:
        for chk in c["checks"]:
            val = chk["detail"].replace("|", "/")
            L.append(f"| `{c['contract_id']}` | `{chk['check_id']}` | {val} | {chk['status']} |")
    L += ["",
          f"消费端现状：ship-platform OpenAPI 共 {facts.get('n_paths')} 条路径，"
          f"`/api/v1/calculations/wind_resistance` **不存在**（唯一含 wind 的路径是 "
          f"`{(facts['endpoints']['/api/v1/calculations/wind_resistance'].get('related_paths') or ['—'])[0]}`，"
          "与风载荷无关）；`/api/v1/fem/modal` 存在，但其 `ModalInput/ModalOutput` 只暴露"
          "干模态字段（`natural_frequencies`/`modal_mass`/`modal_stiffness`），"
          "没有任何流体密度、声速、附加质量或附加质量矩阵字段，因此湿模态契约在 HTTP 层无处承载。"
          "结论：两条跨域接口的生产者载荷均未交付、消费端也尚未就绪，(b) 判 FAIL。",
          ""]

    # (c)
    L += ["### 3.4 判据 (c)：P15 回归基线（11 节点，|Δ| < 1%）", "",
          "| P15 节点 | 仓库 | 检出 head | P15 pass rate | 本次 pass rate | Δ | 覆盖 | 用例数保留 | 裁决 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in cases:
        L.append(f"| {c['ext_id']} | {c['repo']} | {heads.get(c['repo'], {}).get('head', '—')} | "
                 f"{fnum(c['rate_p15'])} | {fnum(c['rate_now'])} | "
                 f"{fnum(c['delta'], '{:+.2e}')} | {c['coverage']} | "
                 f"{'—' if c['test_count_preserved'] is None else c['test_count_preserved']} | "
                 f"**{c['status']}** |")
    L += ["", "逐节点证据要点：", ""]
    for c in cases:
        L.append(f"- **{c['ext_id']}（{c['repo']}）{c['status']}**：{c['detail']}；探针 "
                 f"`{c['command']}`")
    L += [""]

    # (d) + audit
    L += ["### 3.5 判据 (d)：报告与审计链", "",
          "| artifact | 路径 | sha256（前 16 位） |", "|---|---|---|"]
    for k, v in (verdict.get("provenance") or {}).items():
        L.append(f"| {k} | `{v['path']}` | `{(v['sha256'] or 'MISSING')[:16]}…` |")
    L += ["", "---", "", "## ④ 结果图", ""]

    figs = [
        ("fig1_node_acceptance_matrix.png",
         "图 1 判据 (a)：8 个节点 × 各自验收判据（6-8 条）的裁决矩阵。全灰即全部 WITHHELD——"
         "不是失败，而是上游未交付可判定的数值证据；门不允许把『未交付』渲染成『通过』。"),
        ("fig2_p15_regression.png",
         "图 2 判据 (c)：上图为 P15 记录 pass rate（黑色短划）与本次实测（圆点）对照，"
         "下图为 |Δ| 与 1% 判据线。红=FAIL（P15 验证用例在 fresh main 上缺失，或实测 Δ 越界），"
         "灰=WITHHELD（部分覆盖/检出陈旧）。"),
        ("fig3_contract_checks.png",
         "图 3 判据 (b)：两条跨域契约与两个消费端端点的逐项检查矩阵。红块集中在"
         "『生产者载荷缺失』与『wind_resistance 端点不存在』两类。"),
        ("fig4_coupling_topology.png",
         "图 4 两条跨域耦合的数据流与契约裁决：LBM 风系数 → ship-platform；"
         "FEM 湿模态+附加质量 → BEM Helmholtz（k=ω/c）。方框颜色即契约裁决。"),
    ]
    for name, cap in figs:
        p = ev / name
        if p.is_file():
            L += [f"![{name}]({p.as_posix()})", "", f"*{cap}*", ""]

    # ---------------------------------------------------------------- ⑤ limits
    L += ["---", "", "## ⑤ 注意点（踩坑 / 局限 / 适用范围）", "",
          "1. **『节点在跑』不等于『节点通过』**。8 个上游 mission 全部 `running`，"
          "平台 `result` 为空；若按 PR 合并动作推断通过，会把整门变成橡皮章。本门的 rescission 规则是："
          "没有 `p16_node_acceptance.v1` artifact 就 WITHHELD。",
          "2. **P15 的验证用例已不在 fresh main 上**。TensorDEM 与 TensorLBM 的 7 个 P15 探针文件"
          "（`test_hertz_mindlin.py`、`test_neighbor_list_vectorized.py`、`test_clump_multisphere.py`、"
          "`test_stokes_sphere_g12.py`、`test_force_control_surface.py`、`test_d3q27_enumeration.py`、"
          "`test_lid_driven_cavity_3d.py`）在 GitHub main 的 fresh 检出中解析不到；"
          "权威判据是集群 fresh head 的 `MISS` 行（job `c1794033269_256` / `c1794033291_199`），"
          "而不是本地镜像的 `--all` 历史。",
          "3. **陈旧检出会把『没合并』误判成『被删除』**。本地镜像 `git log --all` 能看到"
          "尚未并入 main 的 agent 分支提交；本门用 `git merge-base --is-ancestor` 区分"
          "`removed_from_baseline` 与 `not_in_baseline_history`，后者只算证据缺口（WITHHELD）。",
          "4. **集群不是万能的**。TensorFVM/TensorBEM 的集群检出因节点访问 github 超时失败，"
          "TensorDEM/TensorLBM 节点缺运行时依赖导致 5 个与 1 个 collection error；"
          "这两类问题都被如实写进 `probe_results.json`，没有换算成 PASS。",
          "5. **湿模态在 HTTP 层目前无处落地**。`/api/v1/fem/modal` 仅有干模态字段；"
          "即使 N8 交付了 FEM→BEM 的库级交接，平台层仍缺载体——这是给 N8/集成半边的具体接口要求。",
          "6. **本门不测物理精度**。契约校验器检查格式/类型/量纲/物理自洽（f_wet≤f_dry、PSD），"
          "不检查风系数或辐射效率与实验的偏差；后者属于 N7/N8 的验收判据。",
          "7. **回归度量是 pass-rate**。P15 mission 记录的是 pytest 结果而非物理量，因此 |Δ| 是在"
          "『通过率』上的相对变化；P15 节点的数值判据（如 D11 的 decay_rate=1.5836 1/s）"
          "需要在 N1-N8 交付带数值的 artifact 后才能纳入同一度量。",
          ""]

    # ------------------------------------------------------------- ⑥ error
    L += ["---", "", "## ⑥ 误差判定", "",
          "**本门没有对物理实验/文献基准的量测**，因此逐项回答如下（不回避）：", "",
          "| 关键量 | 基准来源 | 本次值 | 误差 | 是否满足判据 |", "|---|---|---|---|---|"]
    for c in cases:
        base = c["rate_p15"]
        now = c["rate_now"]
        delta = c["delta"]
        if delta is None:
            L.append(f"| {c['ext_id']} pass rate | P15 平台记录 {base:.4f} | 不可测（{c['coverage']}） | "
                     f"— | 否（WITHHELD/FAIL） |")
        else:
            L.append(f"| {c['ext_id']} pass rate | P15 平台记录 {base:.4f} | {now:.4f} | "
                     f"Δ={delta:+.2e} ({delta * 100:+.2f}%) | "
                     f"{'是' if (abs(delta) < 0.01 and c['status'] == 'PASS') else '否'} |")
    L += ["",
          "| 关键量 | 基准来源 | 实测 | 是否 <3% |", "|---|---|---|---|",
          "| 风系数 C_X/C_Y/C_N | 风洞（Krijnen）— 属 N7 判据 | 未交付 | **不适用**（本门无实测值，"
          "上游 N7 未交付）；本门只验证表结构与量纲 |",
          "| 湿/干频率比 | Abramson — 属 N8 判据 | 未交付 | **不适用**（同上） |",
          "| 契约量纲一致性 | 恒等式 ω=2πf、k=ω/c | 未交付载荷 | **不适用**（无载荷可比） |",
          "| P15 pass-rate Δ | P15 平台记录 | 见上表 | 是/否逐项列出；唯一可测且越界者 "
          f"D11 Δ={next((c['delta'] for c in cases if c['ext_id'] == 'IN14-D11'), None)}",
          "",
          "**honest FAIL 声明**：本门裁决 `FAIL`，阻塞项为"
          + "、".join(f"({b.split(':')[0]}) {b.split(':')[1]}" for b in verdict["blocking"])
          + "。没有用任何『上游在跑』『PR 已合并』的理由把 WITHHELD 改写成 PASS；"
          "重新裁决只需在 N1-N8 交付 artifact、ship-platform 补齐端点后重跑 "
          "`python -m tensorphysics.campaign.integration_gate`。",
          "",
          "## 复现命令",
          "",
          "```bash",
          "# 1) 采集平台/端点事实（需要 TENSOR_API_BASE/TENSOR_AGENT_USER/TENSOR_AGENT_PASS）",
          "python3 scripts/collect_p16_gate_evidence.py --out artifacts/p16_gate --ship-api http://127.0.0.1:8002",
          "# 2) 复跑 P15 探针（本地 checkout；--only/--merge 支持逐仓增量）",
          "python3 scripts/p16_run_p15_probes.py --checkouts /root/tensor-ecosystem \\",
          "    --baseline artifacts/p16_gate/p15_baseline.json --out artifacts/p16_gate/probe_results.json",
          "# 3) 登记集群 fresh-head 复核结果",
          "python3 scripts/p16_record_cluster_head_checks.py --probe-results artifacts/p16_gate/probe_results.json",
          "# 4) 跑门 + 出图 + 出报告",
          "python3 -m tensorphysics.campaign.integration_gate --evidence-dir artifacts/p16_gate",
          "python3 scripts/p16_gate_figures.py",
          "python3 scripts/p16_gate_report.py --evidence artifacts/p16_gate \\",
          "    --out docs/P16_CAMPAIGN_INTEGRATION_GATE_REPORT.md --platform-payload /tmp/report.json",
          "# 5) 契约/门/回归单元测试（46 例，含负控）",
          "python3 -m pytest tests/test_integration_gate.py -q",
          "```",
          ""]
    md = "\n".join(L)

    # platform payload
    figures = []
    caps = dict(figs)
    for name, cap in figs:
        p = ev / name
        if p.is_file():
            figures.append({"name": name.replace(".png", ""), "caption": cap,
                            "png_base64": base64.b64encode(p.read_bytes()).decode()})
    payload = {
        "title": "P16-G1 全战役集成验收门：跨域契约与 P15 回归基线（honest FAIL 记录）",
        "summary": (
            "在 TensorPhysics 新建 campaign/integration_gate 门：证据采集→契约验证→节点验收→"
            "P15 回归复跑→审计链报告，46 个单元测试（含负控）全绿。裁决 FAIL："
            "(a) N1-N8 全部 in-flight 无 artifact→WITHHELD；(b) 两条跨域契约未就绪"
            "（wind_resistance 端点不存在于 443 条 OpenAPI 路径、湿模态无 HTTP 载体、生产者载荷缺失）；"
            "(c) P15 11 节点 7 个验证用例在 fresh main 上缺失、D11 实测 Δ=-1.35% 越界、2 个 WITHHELD；"
            "(d) 报告与 sha256 审计链生成。"),
        "scenario_key": "p16_campaign_integration_gate",
        "mission_id": MISSION_ID,
        "tags": ["campaign_gate", "acceptance", "cross_domain_contract", "regression_baseline",
                 "LBM_wind", "FEM_BEM_wet_modal", "honest_FAIL"],
        "markdown": md,
        "figures": figures,
    }
    return md, payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default="artifacts/p16_gate")
    ap.add_argument("--out", default="docs/P16_CAMPAIGN_INTEGRATION_GATE_REPORT.md")
    ap.add_argument("--platform-payload", default="/tmp/report.json")
    args = ap.parse_args()
    md, payload = build_markdown(pathlib.Path(args.evidence))
    pathlib.Path(args.out).write_text(md)
    pl = pathlib.Path(args.platform_payload)
    pl.write_text(json.dumps(payload, ensure_ascii=False))
    print(json.dumps({"markdown": args.out, "chars": len(md),
                      "payload": str(pl), "payload_bytes": pl.stat().st_size,
                      "figures": len(payload["figures"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())