# IN40-G40 核销门：节点递交契约与复跑配方

本文件是 `IN40-G40`（plan 84 波4核销门）的操作契约。核销门的裁决完全由证据决定，
因此「节点怎么把证据交给门」必须固定，否则同一份物理结论会因递交格式不同而得到不同裁决。

## 1. 递交位置与格式

节点把自己的验收 JSON 放到：

```
artifacts/in40_g40_wave4_gate/raw/acceptance/<ext_id>.json
```

内容（`node_from_ledger` 的唯一输入契约，见 `tensorphysics/campaign/wave4_gate.py`）：

```json
{
  "node_id": 548,
  "evidence_grade": "REAL",
  "rationale": "三网格收敛 p=1.6，形式因子 1+k=1.28 vs ITTC'78 1.27（+0.8%）",
  "artifacts": ["artifacts/in40_n01/form_factor.json", "tests/test_twin_hull_form_factor.py"],
  "criteria": {
    "C1": {"verdict": "accept", "measured": 1.8, "reference": 1.9, "unit": "%",
           "evidence": "三网格 C_T 相对差 1.8% ≤ 2%"},
    "C2": {"verdict": "accept", "measured": 1.28, "reference": 1.27, "unit": "-",
           "evidence": "ITTC'78 经验式对拍"},
    "C5": {"verdict": "fail", "measured": 0.0, "reference": 1.0, "unit": "pass",
           "evidence": "pytest 未全绿：2 个用例失败"}
  }
}
```

规则（与 `wave3_gate` 共享语义，不另立一套）：

| 项 | 语义 |
|---|---|
| `criteria` 键 | 必须覆盖该节点 `campaign_nodes.metadata.acceptance_criteria` 的全部 `C1..Cn`；缺项记 `withheld` |
| `verdict` | 只能是 `accept` / `partial` / `fail` / `withheld` |
| `measured`/`reference` | 有基准就必须给数；只写文字等于没有误差可判 |
| `evidence_grade` | `REAL` / `MOCK` / `STUB` / `WITHHELD` / `NOT_SUPPORTED`，**MOCK 不得报 REAL** |
| `rationale` | 谱系继承理由；`accept` 且无实测判据时门会直接报错，不允许「内部自洽式通过」 |

节点级裁决的滚动规则：全部 `accept` → `accept`；任一条 `fail` → `fail`（实测未达标是决定性的）；
全 `withheld` → `withheld`；其余 → `partial`。`fail` 同时构成 **critical failure**，即使通过率 ≥80% 也不许关计划。

## 2. 复跑配方（三步，全部可重复）

```bash
python scripts/run_in40_g40_wave4_gate.py collect    # 只读快照 plan 84 节点/任务/guide_tracker + sha256
python scripts/run_in40_g40_wave4_gate.py evaluate   # 离线重算矩阵/裁决/谱系/图，写 JSON + MD
python scripts/run_in40_g40_wave4_gate.py publish    # 发布 tensor_reports（同 md 幂等）
```

`evaluate` 不读网络、不读数据库、不读时钟，产物见：

- `artifacts/in40_g40_wave4_gate/wave4_gate_report.json`（机器报告：矩阵、判据数字、check 数字、audit_chain、产物路径）
- `docs/IN40_G40_WAVE4_GATE.md`（收官报告：六节 + 局限节 + guide_tracker 缺口）
- `artifacts/in40_g40_wave4_gate/figures/*.png`（矩阵热图 / 通过率 / guide 缺口）

## 3. 不可覆盖的边界

`verify_report_integrity` 会从证据快照重算 `final_verdict`、`plan_decision`、`pass_rate` 三个数；
写进磁盘或发布出去的版本只要比重算结果乐观，就抛 `GateIntegrityError`。
`plan_update_sql()` 只**计算**计划状态语句（≥80% 且无 critical 时为 `completed`），
应用与否是独立动作：未达门槛时语句为空操作，计划保持 `active`。

当前（证据截止 2026-11-07T13:25:26Z）实测：10 节点 / 54 条判据，通过 0/10，
`final_verdict=NOT_ACCEPTED_NODES_OPEN`，计划 `retained_active`，下一波提案未触发。