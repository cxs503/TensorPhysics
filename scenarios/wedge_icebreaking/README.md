# Scenario: wedge_icebreaking — 二维楔形船首连续破冰联调阶梯

跨库联合调试（TensorLBM / TensorFVM / TensorFEM / TensorDEM / TensorPhysics / TensorPlatform）
的首个测试对象与分级耦合阶梯。对应 `docs/SYSTEM_OVERVIEW.md` §6 分阶段路径的落地实现。

## 为什么选这个案例

联调要测的是**管道**（进程调用、场交换、时钟、清单、注册），不是物理精度。
楔形破冰是目标应用（破冰船-水-冰闭环）的 2D 截片，每一段在各库中都有已验证的
独立实现，单轮迭代秒到分钟级，判据全部可机器裁定（守恒/账本门）。

## 阶梯

| 级 | 链路 | 状态 |
|---|---|---|
| **S0** | 四库冒烟：各求解器在统一 fail-closed 控制器下跑自己的分钟级已验证入口，RunManifest/ArtifactRef 全链 | `s0.py`（本目录） |
| **S1** | DEM 冰层 + 预设楔形船体（`tensordem --hull-profile`），反力冲量/断键能账本门 | `s1.py`（本目录） |
| **S2** | +FEM 结构单向（S1 冰-船反力历史 → TensorFEM 梁/壳准静态），力/力矩/功共轭门 | 计划 |
| **S3** | +流体载荷单向（TensorFVM 壁面力 / TensorLBM 力历史回放），总力守恒+时间对齐门 | 计划 |
| **S4** | 闭环双向交换，时钟同步+子循环（TensorDEMAdapter 模式），守恒+回滚+检查点链 | 计划 |

## S0 设计

- 四腿顺序执行，**fail-closed**：任一腿校验失败即停，不消耗剩余预算。
- 每腿：子进程调用求解器已验证入口 → 独立输出完整性检查（exit code、JSON/CSV
  可载、数值有限、时间单调）→ `ValidationReport` → `LocalArtifactStore`
  校验和入库 → `RunManifest`（含 config 摘要与 `code_revisions` 双库 SHA）。
- 判据词表：`decision ∈ {s0_smoke_complete_integrity_gates_passed,
  stop_s0_and_repair_failed_gate}`；`scientific_status = not_a_physics_benchmark`。
- 成熟度：本场景属管道完整性冒烟（Implemented 级证据），**不得**凭其通过晋级
  Benchmark-verified 或更高标签（见 `docs/SYSTEM_OVERVIEW.md` §7）。
- FVM 腿允许返回码 {0, 2}：2 = SIMPLE 迭动上限但产物完整（物理收敛非 S0 门），
  在该腿 config 与检查明细中披露。

## 运行

```bash
# 六库工作区（tensor_eco）与两个 venv 就绪后：
cd <tensor_eco>/TensorPhysics
/nfs/wangxi/venvs/tensor_eco/bin/python scenarios/wedge_icebreaking/s0.py \
    --work-dir outputs/wedge-icebreaking-s0 \
    --scenario-id wedge-icebreaking-s0-001
```

产物布局（`--work-dir` 下）：

```
research_goal.json      # 场景级 ResearchGoal
lbm/ fvm/ fem/ dem/     # 每腿：solver-output/ + validation_report.json
                        #        + run_manifest.json + research_memory.json
                        #        + leg_result.json + artifact-store/<run_id>/
s0_report.json          # 场景级报告：legs + decision + scientific_status
```

## 平台注册边界

S0 停在 RunManifest + LocalArtifactStore（与 `examples/research_loop_dem.py`
同一边界）。在线 TensorPlatform 注册需要非 `file://` 的共享 URI
（`platform_manifest._artifact_record` 拒绝本地 URI），待本地 Platform +
对象存储部署后随 S1+ 升级接入。
