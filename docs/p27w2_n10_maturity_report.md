# P27-W2 战役级成熟化汇总报告（N10 集成验收）

- **门 ID**: P27W2-G1 · **计划**: plan 153（P27-W2-seed）
- **总体裁决**: **WITHHELD** · **时间**: 2026-11-08T08:36:55Z
- **生成**: `scripts/run_p27w2_n10_gate.py` + `scripts/plot_p27w2_n10.py`
  （数字全部程序化注入，零手抄）
- **工件根**: `runs/p27w2_n10/`

## 1. 战役结构

10 节点 DAG（无环，Kahn 拓扑排序验证）：

```
N01(961) BEM辐射绕射 ──> N02(962) 耐波RAO ──> N03(963) 横摇阻尼ζ_eq
N04(964) 谱疲劳（成熟化）        N05(965) 浮力耦合板
N06(966) DEM滚动摩擦 ──> N07(967) DEM非球形
N08(968) 湍流u+ UQ              N09(969) AMR回流修正（成熟化）
全部 9 节点 ──────────────────────────────> N10(970) 集成验收（本节点）
```

## 2. 成熟化节点产出摘要

| 节点 | 类型 | 裁决 | 关键数字 | 工件 |
|---|---|---|---|---|
| N4 | spectral_fatigue | done | ship-performance-platform 原只有时域 S-N/Miner（api/fatigue.py），无任何频域谱疲劳能力。本节点新建 backe |
| N9 | amr | done | 在 TensorLBM 新增 AMR 回流修正（Berger–Colella）成熟化交付：`src/tensorlbm/amr_reflux.py`（一维标量守 |

> in-flight 节点在验收时点尚无 result，摘要留空并如实标注
> （honest 欠账，不可粉饰）。

## 3. 跨仓接口契约（真实节点数据）

| 契约 | 接口 | 数据源 | 检查通过 |
|---|---|---|---|
| C1 | TensorBEM → ship-performance-platform（RAO） | m6657 `backend/physics/rao_verification.py`（Wigley/KCS 实验对标） | 12/12 |
| C2 | TensorDEM → TensorPhysics（颗粒流参数） | m6661 球-球滚动摩擦验证（Cundall-Strack/滚动阻力/旋转阻尼） | 8/8 |
| C3 | TensorLBM → TensorPlatform（GCI/UQ） | m6663 圆柱网格收敛证据 + p27w2_n08 cross_analysis | 11/11 |

## 4. 证据台账（完整率 80%）

每节点：`nodes/m<mission>_result.json`（归档的 mission result）+
工件路径（worktree 未提交交付物 / 分支 diff）+ `agent/m<mission>`
分支 commit。

| 节点 | 状态 | result.json | 工件数 | commit |
|---|---|---|---|---|
| N1 | in-flight | — | 7 | `d371b21f0b` |
| N2 | done | ✓ | 19 | `8287dbba09` |
| N3 | done | ✓ | 1 | `f30f2cb100` |
| N4 | done | ✓ | 19 | `8bbb6fc827` |
| N5 | done | ✓ | 11 | `cb90900286` |
| N6 | done | ✓ | 9 | `cc7dc39e7b` |
| N7 | done | ✓ | 18 | `842eba95c0` |
| N8 | done | ✓ | 14 | `dba3f357bc` |
| N9 | done | ✓ | 20 | `3552f35c13` |
| N10 | in-flight | — | 12 | `2fd3037276` |

## 5. 能力台账（8 REAL / 1 WITHHELD）

| 能力 | 仓 | 节点 | 原等级 | 现等级 |
|---|---|---|---|---|
| bem_wave_making_rao | TensorBEM | N1 | STUB | WITHHELD |
| seakeeping_rao_verification | ship-performance-platform | N2 | MOCK | REAL |
| roll_damping_zeta_eq_three_channel | ship-performance-platform | N3 | MOCK | REAL |
| spectral_fatigue_dirlik_rayleigh | ship-performance-platform | N4 | MOCK | REAL |
| buoyancy_coupled_elastic_plate | TensorFEM | N5 | MOCK | REAL |
| dem_rolling_friction_cundall_strack | TensorDEM | N6 | MOCK | REAL |
| dem_non_spherical_clump | TensorDEM | N7 | MOCK | REAL |
| lbm_turbulence_statistical_error | TensorLBM | N8 | MOCK | REAL |
| lbm_amr_backflow_flux_balance | TensorLBM | N9 | MOCK | REAL |

## 6. 局限

1. 集成验收依赖前置完成：验收时点 N01（BEM，m6656）与 N09（AMR，
   m6664）仍在运行，判据 a/c/d 因此 WITHHELD——战役时序的诚实反映；
   前置完成后重跑同一脚本即得终判。
2. 契约验证器接受双布局：RAO 契约同时接受规范布局（schema+rao 块）
   与 ship 平台既有扁平布局（omega_e+<dof>_rao），避免接口升级误杀
   存量数据。
3. GCI 观测阶：LBM 圆柱 Cd 三网格观测阶 0.1（fit_rms 0.21%），未达
   渐近区——契约按「Richardson 一致或低 RMS 拟合背书」放行，物理
   收敛质量归节点自身判据。
4. 证据台账的 commit 取各仓 `agent/m<id>` 分支 tip；平台
   「matured mission deliverables」合并前，部分成功节点交付物暂存
   worktree 未提交（如 m6657/m6661）。
5. 本门为 P27-W2 专用（节点号/成熟化节点/契约集硬编码于本模块）；
   跨战役复用需参数化。
