# SIR-1 预注册：G1 流体载荷精度封口

- **创建时间**：2026-10-25T05:10:21Z
- **缺口**：G1 —— `suboff_sail` 负阻力 honest-FAIL 遗留，τ 近奇异根因链（#256-#278 线）
- **依赖**：SIR-0（本契约）；可与 SIR-2 / SIR-3 / SIR-5 并行
- **算力/预算**：GPU5（单档先落）；M（1–2 班次）
- **append-only**：冻结后只允许 `## Addendum <UTC Z>` 追加

## 1. 科学目标

把 SUBOFF 破冰链路里的**流体载荷**从「可复现的数值量」提升到「有物理精度封口的量」：
在 SIR-1 出口，链路使用的流体力必须能对上已冻结的外部参考（ITTC-1957 / Blasius /
DARPA AFF-8 实验表），且 τ 冻结表逐档可复算、非近奇异；honest-FAIL 的历史事实保留不改写。

## 2. 物理锚（文献出处 / 已验证锚路径）

| 锚 | 内容 | 出处 | 机器可裁性 |
|---|---|---|---|
| A1.1 | ITTC-1957 `Cf = 0.075/(log10(Re) − 2)²` | ITTC Recommended Procedures 7.5-02-02-01；库内 `TensorLBM/src/tensorlbm/hydrodynamics.py:23` | 直接调用库函数，值可复算 |
| A1.2 | Blasius `Cf = 1.328/√Re`（Re=1000 → 0.0420） | `TensorLBM/benchmarks/pending/suboff_re1000/README.md`；`suboff_sail` 冻结 `Cf_ref` | 三元组 `prereg.md` grep |
| A1.3 | DARPA SUBOFF 实验阻力（AFF-1/AFF-8，6 航速） | Liu & Huang, CRDKNSWC/HD-1298-11, ADA359226, Table 14, p.23 | 复制到 `sir1_reference.csv` 后三元组化 |

**不接受**：只用本链路自己的收敛窗口证明「载荷对了」（纯自证闭环）。

## 3. 机器门（值-文件-键 + 双账本）

| 门 | 判据 | 判据来源 |
|---|---|---|
| M1.1 | 载荷双账本 `max|A−B| ≤ 1e-12`（流场侧发出 vs 结构侧接收） | SI-1 锚1 口径，`benchmarks/icebreaking/si1_one_way_output/validation_report.json` ← `anchors[0].value` |
| M1.2 | τ 冻结表 `tau = 3·u_lb·L/1000 + 0.5`，L∈{112,144,176}；`tau − 0.5 > 0` 且逐档登记 | `sir0_baseline.json` ← `derived_frozen_table` |
| M1.3 | honest-FAIL 保留：G1 门 FAIL 事实（`OVERALL=FAIL`、`G1_pass=false`）不得改写 | `sir0_baseline.json` ← `gaps.g1_fluid_load_precision.evidence` |
| M1.4 | 外锚误差门：最细档 `|Cd_tot/Cf_ref − 1| ≤ 3%`（沿用主梯 G1 门限） | `/nfs/wangxi/runs/suboff_sail_20261008/prereg.md` 门表 |

## 4. 规模判据

- ≥2 档网格（首选 3 档 L ∈ {112,144,176}，全域胞数 37,745,568 / 80,414,160 / 146,869,248）。
- 力采样末窗 ≥1000 样本；L=112 档步数 = 150·L = 16800 步。
- 物理时间：单档 ≥ 1.25 流贯时程（`6L/u_lb = 120L` 步，与 verified B6 等价）。

## 5. 内容符合判据（语义门，见 `semantic_criteria.json` → `SIR-1`）

- `sir1_provenance_load_source`（provenance）：载荷源必须是流场工件，不得含合成标记。
- `sir1_physics_tau_frozen_non_singular`（physics_switch）：`tau - 0.5 > 0` 且三元组可复算。

## 6. 交付物

`benchmarks/icebreaking/sir1_load_precision/validation_report.json`（机器门逐条 + 语义门结论 +
规模量 + 资源披露）+ `gates.json`（值-文件-键三元组）+ 必要时 `sir1_reference.csv`（带出处列）。

## 7. 停机条件（fail-closed）

任一机器门红 → 停，不消耗剩余预算；判词入 `validation_report.json.decision`。

## Addendum 记录

（暂无）
