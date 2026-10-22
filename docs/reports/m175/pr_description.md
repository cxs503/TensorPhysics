# PR 草稿：SI-5 FVM 可交换对照腿交付物收口（m166 收尾）

> **注意**：本文件为 PR 描述草稿，owner 在 GitHub 上创建 PR 时可直接复制以下内容。
> 因环境无 `gh` CLI 且 push 已禁用，PR 需由 owner 手动在 GitHub 创建。

---

## PR 标题

`SI-5: FVM 可交换对照腿交付物收口（m166 收尾）`

## 源分支

`agent/m166` → `main`

## PR 描述

### 背景

本 PR 为 **m166 收尾单**，不涉及新的科学计算或代码变更。

m166（SI-5 FVM 可交换对照腿）核心工作已由前任 agent 完成：
- **核心 commit**: `630875d` — 实现 FVM 流解替换 LBM 入 SI-1 单向耦合
- **验证结果**: 四锚判据 4/4 PASS（validation_report.json）

但 m166 最终停在 `finished_without_acceptance`，原因是以下交付物**留在暂存区/未跟踪状态，未被 commit**：
- `docs/reports/si5_fvm_exchange_report.md` — 四锚验证报告
- `docs/reports/figures/` — LBM/FVM 对照图件（PDF + PNG）与快照（npz）
- `benchmarks/icebreaking/generate_si5_figures.py` — 图件生成脚本

### 本次变更

仅一个 commit（`40dd5c2`），将上述遗留交付物 `git add` + `git commit` 到 `agent/m166` 分支：

```
40dd5c2 SI-5: 收口交付物——验证报告、图件与生成脚本落盘（m166 收尾）
```

**世系**：`40dd5c2` → `630875d`（m166 核心 commit）→ `f2962d5`（main），未重写历史。

### Gates 验证（复跑确认）

| 锚点 | 判据 | 实测值 | 状态 |
|------|------|--------|------|
| 锚1: 传递守恒 | max\|A−B\| ≤ 1e-12 | 0.000e+00 | ✓ PASS |
| 锚2: Hetényi 复现 | 相对误差 < 2% | 1.667e-02 | ✓ PASS |
| 锚3: 线性叠加 | max\|Δw\| ≤ 1e-10 | 4.461e-17 | ✓ PASS |
| 锚4: 可交换性 | 相对差异 < 5% | 1.119e-06 | ✓ PASS |

**总体**: 4/4 PASS — 与 `validation_report.json` 完全一致。

### 报告与图件一致性

报告 `docs/reports/si5_fvm_exchange_report.md` 中所有数字均与 `validation_report.json` 逐字对得上（零手抄），已用脚本自动核对。

### 披露

- **内容主体归属**: m166（前任 agent）
- **本单角色**: 收尾——仅 `git add` + `git commit` 遗留交付物
- **Merge 决策**: 留 owner

### 文件清单

```
benchmarks/icebreaking/generate_si5_figures.py           (新增 263 行)
docs/reports/si5_fvm_exchange_report.md                  (新增 225 行)
docs/reports/figures/fvm_snapshot_uniform.npz            (53 KB)
docs/reports/figures/lbm_snapshot_uniform.npz            (104 KB)
docs/reports/figures/si5_difference_analysis.pdf         (20 KB)
docs/reports/figures/si5_difference_analysis.png         (92 KB)
docs/reports/figures/si5_lbmfvm_comparison.pdf           (29 KB)
docs/reports/figures/si5_lbmfvm_comparison.png           (151 KB)
```
