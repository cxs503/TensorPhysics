# suboff_icebreaking / prereg —— SIR 战役预注册（append-only）

本目录是 Phase 3（SIR 线）全部预注册文件的唯一存放点。

## 纪律

1. **append-only**：每份 prereg 冻结后只允许以 `## Addendum <UTC 时间戳>` 段落追加，
   不改写既有条款、不删除既有判据。任何口径变更必须新开 Addendum 并注明原条款编号。
2. **判据先冻结后跑**：腿的判据（物理锚 / 机器门 / 规模判据 / 内容符合判据）必须先于
   该腿的计算落盘并 commit；判决数字不得在跑后凭观测回填。
3. **时间戳 UTC Z**：本目录所有时间戳一律 `YYYY-MM-DDTHH:MM:SSZ`（CST 标 Z 教训）。
4. **数字零手抄**：一切基线/判决数字以 值-文件-键 三元组形式声明；本体由
   `tests/test_sir_phase3_contract.py` 打开文件复算，对不上即测试红。

## 文件清单

| 文件 | 内容 |
|---|---|
| `sir0_prereg.md` | 战役总预注册（起点局限、靶标、纪律、验收定义） |
| `sir0_baseline.json` | 基线数字三元组（id=51 / SI-1 / SI-2 / suboff_sail 事实机器直读） |
| `sir1_prereg.md` … `sir7_prereg.md` | 七腿 prereg 骨架（跑前冻结，跑后 Addendum 追加） |
| `semantic_criteria.json` | 七腿「内容符合判据」（与平台 PL-SEM 语义门 schema 对齐） |
| `dependency_graph.json` | 腿间依赖图 + 并行分组（机器可读，含 DAG 校验字段） |

## 与既有 Phase 1/2 prereg 的关系

Phase 1/2 的 prereg 分布在场景各阶目录（`si1_one_way/prereg.json`、`si2_fracture_gen/prereg.json`、
`si3_two_way/…`、`si4_full_chain/prereg.json`），**保持原位不动**（历史冻结件）；
Phase 3 起统一收敛到本目录，并在 `sir0_prereg.md` 中引用 Phase 1/2 的既有冻结件路径。
