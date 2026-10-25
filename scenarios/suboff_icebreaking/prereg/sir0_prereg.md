# SIR-0 预注册：SUBOFF 破冰真实化战役契约冻结（Phase 3 总预注册）

- **创建时间**：2026-10-25T05:10:21Z
- **冻结范围**：SIR-1 … SIR-7 七腿的靶标、依赖、纪律、验收定义（本文件 + 同目录骨架）
- **发起**：owner 2026-10-25 定向；执行 mission m302（薄编排腿，零求解器代码）
- **append-only**：本文件冻结后只允许 `## Addendum <UTC Z>` 追加

## 1. 起点事实（零手抄，三元组见 `sir0_baseline.json`）

Phase 1/2 已收官（单库四腿能力 + 耦合链 smoke，平台报告 **id=51**）。id=51 的「局限（如实
披露）」三条即本战役全部靶标：

| # | id=51 局限 | 转成的缺口 |
|---|---|---|
| L-1 | smoke 级规模；12 颗粒 / 30 键 / 断键 0；断裂传播未被演练 | G5、G6 |
| L-2 | 残差恰为 0 是机器精度表现，非物理精度证明 | G1、G7 |
| L-3 | 后向耦合为登记点，物理强度待放大验证 | G2、G3、G4 |

本战役不接受「形式闭环即达标」：每腿判据必须包含 (a) 可追溯的物理锚（文献/解析/已 verified
锚复用）与 (b) goal 语义层面的可机裁断言（§4 内容符合判据）。

## 2. 七腿与依赖（详见 `dependency_graph.json` 与 README §3.1）

| 腿 | 缺口 | 硬依赖 | 并行 |
|---|---|---|---|
| SIR-1 | G1 流体载荷精度封口 | SIR-0 | 可与 SIR-2/3/5 并行 |
| SIR-2 | G2 真流场入链 | SIR-0 | 可与 SIR-1/3/5 并行 |
| SIR-3 | G3 重力/浮力/自由面入链 | SIR-0 | 可与 SIR-1/2/5 并行 |
| SIR-4 | G4 船体上浮运动 + 船-冰接触 | SIR-2 + SIR-3 | 串行咽喉 |
| SIR-5 | G5 断裂传播 | SIR-0 | 可与 SIR-1/2/3 并行 |
| SIR-6 | G6 规模化 | SIR-1..SIR-5 | 串行 |
| SIR-7 | G7 文献/实验对照 | SIR-5 + SIR-6 | 串行（战役出口） |

波次：W1 = {SIR-1,SIR-2,SIR-3,SIR-5}；W2 = {SIR-4}；W3 = {SIR-6}；W4 = {SIR-7}。

## 3. 每腿必含三样（缺一即 prereg 不合格）

1. **物理锚**：文献出处（DOI/报告号/页码/表号）或已 verified 锚路径（库内 file:line 或库测试名）。
   **纯自证闭环不收**。
2. **机器门**：值-文件-键三元组 + 双账本范式（一侧发出 vs 另一侧接收，机器精度 ≤1e-12）。
3. **规模判据**：物理时长 / 网格量 / 颗粒数量的量级下限（相对 id=51 的 smoke 规模向上）。

## 4. 内容符合判据（与平台 PL-SEM 语义门对齐）

每腿在 `semantic_criteria.json` 里冻结一组判据，schema 与
`TensorPlatform/docs/specs/semantic_gate_spec.md`（PL-SEM-1/2）的 `semantic-criteria` 块一致：
`{id, category ∈ {provenance, physics_switch, scale_realism, field_integrity},
description, assertion}`。腿跑完由语义门裁定，`overall_verdict` 写入该腿
`validation_report.json` 的 `semantic_gate` 节。

**语义门失败不阻断交付，但阻断「succeeded=目标达成」表述**；差距入报告交 owner。

## 5. 验收定义（本战役整体）

SIR 线整体完成的判定（全部满足；任一不满足即「未完成」，不得以形式闭环宣称达标）：

1. 七腿各有 `sir<n>_prereg.md` 冻结 + 跑后 `validation_report.json`（含机器门逐条 +
   语义门逐条 + 规模量 + 资源披露）；
2. 七腿的机器门全部 PASS（honest-FAIL 计为事实披露，但该腿判「未达标」）；
3. 语义门 `overall_verdict` 全为 `compliant`（任一 `partial/non_compliant` 则该腿判未完成）；
4. SIR-6 的规模判据四项下限全达标（物理时长 ≥1 s、单元 ≥1e4、颗粒 ≥1e3、断键 >0）；
5. SIR-7 至少一条 P-h 曲线关键点对照达标（出处三元组齐全）；
6. 报告（论文级）含物理表述 + 曲线/云图 + 判据表 + 审计溯源 + 局限披露。

## 6. 纪律（红线，全文见 README §3.7）

不 git push；薄编排零求解器代码；数字零手抄；锚必须给文献出处或 verified 锚路径；
honest-FAIL 不假绿；prereg append-only；时间戳 UTC Z；预算纪律交付优先。

## 7. 既有冻结件引用（不得改写）

- Phase 1/2 prereg：`si1_one_way/prereg.json`、`si2_fracture_gen/prereg.json`、
  `si3_two_way/coupling_loop.py` 同目录账本、`si4_full_chain/prereg.json`。
- 流体精度根因链：`/nfs/wangxi/runs/suboff_sail_20261008/prereg.md`（append-only，
  含 Extension E2/E3/E5 与冻结 Cf_ref）。
- SI-4 收官工件：`outputs/si4_full_chain/validation_report.json` 与
  `outputs/si4_full_chain/full_chain_summary.json`。

## Addendum 记录

（暂无；后续追加格式：`## Addendum 2026-XX-XXTXX:XX:XXZ —— <主题>`）
