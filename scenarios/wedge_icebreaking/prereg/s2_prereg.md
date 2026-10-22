# S2 预注册：S1 冰-船反力历史 → TensorFEM 梁/壳准静态（单向）

> Append-only 注册文件。先注册后派发，后续只允许追加（## Addendum 段落）。

## 1. 科学假设
S2 验证：把 S1 产出的冰-船反力历史作为外载施加到 TensorFEM 准静态结构模型上，
力、力矩、功三共轭账本闭合（管道级耦合正确性，非结构 Benchmark）。
- 物理范围：单向（DEM→FEM），结构不反馈回 DEM。
- 成熟度声明：Implemented 级证据，不得凭 S2 晋级 Benchmark-verified 或更高。
- fail-closed：任一机器门失败即停。

## 2. 输入契约
- S1 工件（已存在，只读）：/nfs/wangxi/runs/wedge_s1_20261018/outputs/wedge-icebreaking-s1-repair/dem_wedge/solver-output/history.csv
  反力列：reaction_x, reaction_y（冰对船体，N）；time 列（s）。
- TensorFEM：/nfs/wangxi/tensor_eco/TensorFEM（共享克隆），venv tensor_eco 可执行
  /nfs/wangxi/venvs/tensor_eco/bin/tensorfem。运行前记录其 git rev 进 manifest。

## 3. 机器判据（全部工件级，不依赖求解器内部）
三腿顺序：leg_loads（提取载荷）→ leg_fem（准静态求解）→ leg_gates（账本裁判）。
### 3.1 载荷提取门
- loads.json 可载入：frames == history.csv 数据行数、time 单调、力值全部有限。
### 3.2 FEM 求解门
- 退出码 ∈ {0}；输出 fem_out.json：每载荷步 {step, F_ext_x, F_ext_y, M_ext,
  R_x, R_y, M_R, u_ref_x, u_ref_y, U_strain_J, W_ext_J} 全部有限、步数==载荷帧数（或按注册的抽稀因子 k 整除抽样，k 必须在 fem_out.json 里显式记录）。
### 3.3 力平衡共轭门
- 每步 |F_ext + R| / max(|F_ext|, |R|, 1e-12) ≤ 1e-6。
### 3.4 力矩平衡共轭门
- 每步 |M_ext + M_R| / max(|M_ext|, |M_R|, 1e-12) ≤ 1e-6。
### 3.5 功共轭门
- 每步 |W_ext_J - U_strain_J| / max(|W_ext_J|, |U_strain_J|, 1e-12) ≤ 1e-5
  （准静态弹性：外力功==应变能；W_ext 用梯形积分 F·du，允许积分误差故容差 1e-5）。
### 3.6 工件完整性门（S0/S1 模式）
- history/loads/fem_out 有限可载、时间单调；RunManifest + LocalArtifactStore 校验和全过。
- manifest 记录 TensorFEM 与编排器自身（TensorPhysics）双 rev。

## 4. 决策词表
- decision ∈ {s2_structural_one_way_passed, stop_s2_and_repair_failed_gate}
- scientific_status = implemented_not_benchmark_verified

## Addendum (m46 复跑，2026-10-18)

### 背景
m45 交付的工件（validation_report.json）三门全绿，但验收摘要中将功共轭门误写为首跑中间值
「1.23e-02（≤1e-5 ✓）」，与工件终值 6.111693e-11 自相矛盾，审批器正确驳回。
缺陷仅在验收证据转录，不在交付本身。

### 复跑配置
- work-dir: `/nfs/wangxi/runs/wedge_s1_20261018/outputs/m46`
- scenario-id: `wedge-icebreaking-s2-m46`
- 命令: `python3 scenarios/wedge_icebreaking/s2.py --work-dir /nfs/wangxi/runs/wedge_s1_20261018/outputs/m46 --scenario-id wedge-icebreaking-s2-m46`
- m45 原始工件（leg_loads/leg_fem/validation_report.json）时间戳 14:06 未变，未覆盖。

### 复跑三门终值（逐字抄录自 validation_report.json）
来源文件: `/nfs/wangxi/runs/wedge_s1_20261018/outputs/m46/validation_report.json`

| 门 | 字段名 | 终值 | 容差 | 通过 |
|---|---|---|---|---|
| 力平衡 | force_balance | 8.480668e-16 | ≤1e-6 | ✓ |
| 力矩平衡 | moment_balance | 3.957861e-16 | ≤1e-6 | ✓ |
| 功共轭 | work_conjugacy | 6.111693e-11 | ≤1e-5 | ✓ |

### 决策
- decision: `s2_structural_one_way_passed`
- scientific_status: `implemented_not_benchmark_verified`
