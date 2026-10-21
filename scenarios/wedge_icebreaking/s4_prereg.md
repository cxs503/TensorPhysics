# wedge_icebreaking S4 预注册：闭环双向交换（fluid ↔ structure two-way）

## 目标
在 S3（流体→结构单向）基础上实现**闭环双向耦合**：
- **流体 → 结构**：TensorFVM 壁面力 F_fluid(t) 作为结构外载一部分
- **结构 → 流体**：结构自由端位移 u_ref(t) 反馈为流体壁面法向速度边界条件
- **时钟同步 + 子循环**：主时钟 = 结构步 dt_struct；每结构步内流体跑 n_sub 子步
- **守恒门**：每交换回合双向冲量/能量账本闭合
- **回滚 + 检查点链**：每回合保存检查点；守恒门红 → 回滚上一检查点并停（fail-closed）

全程双向：流体感知结构运动（通过壁面速度边界条件），结构感知流体载荷。

## 耦合策略（准稳态松耦合）

由于 TensorFVM 无 2D ALE 动网格，采用**固定几何 + 壁面速度边界条件**模拟双向：
- 流体网格不变（楔形 mask 固定）
- 楔形壁面处的速度边界条件从"无滑移 u=0"改为"法向速度 = v_wall(t)"
- v_wall(t) = du_ref_y/dt（结构自由端横向速度，均匀分布到壁面节点）
- 流体求解感受结构运动，结构求解感受变化的流体载荷

这是**一阶近似**：真实 FSI 需动网格或浸没边界法，本场景验证管道完整性（时钟同步、守恒、回滚），不追求物理精度。

## 腿与工件（全部 JSON，落 work-dir 下本场景子目录）

### leg_fluid: fluid_out.json
每耦合回合 k：
```json
{
  "coupling_step": k,
  "t": t_k,
  "F_fluid_x": F_fluid_x(t_k),
  "F_fluid_y": F_fluid_y(t_k),
  "F_wall_vectorized_x": ...,
  "F_wall_vectorized_y": ...,
  "F_wall_node_loop_x": ...,
  "F_wall_node_loop_y": ...,
  "F_total_reported_x": ...,
  "F_total_reported_y": ...,
  "v_wall": v_wall(t_k),
  "sub_steps": n_sub
}
```

### leg_struct: struct_out.json
每耦合回合 k：
```json
{
  "coupling_step": k,
  "t": t_k,
  "F_ext_x": F_dem_x + F_fluid_x,
  "F_ext_y": F_dem_y + F_fluid_y,
  "M_ext": ...,
  "u_ref_x": ...,
  "u_ref_y": ...,
  "v_ref_x": ...,
  "v_ref_y": ...,
  "R_x": ...,
  "R_y": ...,
  "M_R": ...,
  "U_strain_J": ...,
  "K_struct_J": ...,
  "W_ext_J": ...
}
```

### leg_exchange: exchange_log.json
每交换回合的双向数据：
```json
{
  "coupling_step": k,
  "t_start": t_k,
  "t_end": t_{k+1},
  "fluid_to_struct": {
    "F_fluid_x": ...,
    "F_fluid_y": ...,
    "impulse_x": ∫F_fluid_x dt,
    "impulse_y": ∫F_fluid_y dt,
    "work_J": ∫F_fluid · v_struct dt
  },
  "struct_to_fluid": {
    "u_ref_x": ...,
    "u_ref_y": ...,
    "v_wall": du_ref_y/dt,
    "displacement_work_J": ∫F_struct · v_wall dt
  },
  "checkpoint_id": "ckpt_k"
}
```

### leg_conservation: conservation_ledger.json
每回合的守恒账本：
```json
{
  "coupling_step": k,
  "momentum_balance": {
    "fluid_impulse_x": ...,
    "struct_momentum_change_x": ...,
    "reaction_impulse_x": ...,
    "residual": |fluid_impulse - struct_Δp - reaction_impulse|
  },
  "energy_balance": {
    "fluid_work_J": ...,
    "struct_strain_energy_J": ...,
    "struct_kinetic_energy_J": ...,
    "residual": |fluid_work - U_strain - K_struct|
  },
  "passed": true/false
}
```

### validation_report.json
checks 数组（name/passed/detail）+ decision 字段。

## 机器判据（artifact 级，全部通过才 passed）

1. **fluid_ledger**: max |F_vec − F_node| / scale ≤ 1e-10（两套壁面力算法一致）
2. **time_sync**: 每耦合回合，流体子步数 = n_sub，t_fluid_end − t_fluid_start = dt_struct ± 1e-12
3. **bidirectional_flow**: 
   - fluid→struct: max |F_fluid| > 0（流体力真实进入结构）
   - struct→fluid: max |v_wall| > 0（结构速度真实影响流体边界）
4. **momentum_conservation**: 每回合 |fluid_impulse − struct_Δp − reaction_impulse| / scale ≤ 1e-6
5. **energy_conservation**: 每回合 |fluid_work − U_strain − K_struct| / scale ≤ 1e-5
6. **checkpoint_chain**: 检查点 ID 单调递增，回滚次数 = 0（或注明回滚原因）
7. **decision == "s4_bidirectional_coupled_passed"**

## 时钟同步与子循环

- **主时钟**：结构求解器 Newmark-β 时间步 dt_struct（由 S2/S3 继承）
- **子循环**：每结构步内流体跑 n_sub = 10 子步，dt_fluid = dt_struct / n_sub
- **交换节拍**：
  - t_k：结构步 k 开始
  - 结构求解 → u_ref(t_k), v_ref(t_k)
  - 计算 v_wall = v_ref_y(t_k)
  - 流体子循环 n_sub 步（壁面边界条件含 v_wall）→ F_fluid(t_{k+1})
  - t_{k+1}：结构步 k+1 开始，用 F_fluid(t_{k+1})
- **记录**：exchange_log.json 记录每回合的 t_start, t_end, n_sub

## 守恒门设计

### 冲量守恒
流体给结构的冲量 = 结构动量变化 + 固定端反力冲量：
```
∫_{t_k}^{t_{k+1}} F_fluid dt = M * (v_struct(t_{k+1}) − v_struct(t_k)) + ∫_{t_k}^{t_{k+1}} R dt
```
其中 M = 结构等效质量（由 FEM 模型隐含），R 为固定端反力。

残差：|fluid_impulse − struct_Δp − reaction_impulse| / max(|fluid_impulse|, |struct_Δp|, 1e-12)

### 能量守恒
流体对结构做的功 = 结构应变能 + 结构动能：
```
∫_{t_k}^{t_{k+1}} F_fluid · v_struct dt = U_strain(t_{k+1}) − U_strain(t_k) + K_struct(t_{k+1}) − K_struct(t_k)
```

残差：|fluid_work − ΔU_strain − ΔK_struct| / max(|fluid_work|, |ΔU_strain|, 1e-12)

## 回滚 + 检查点链

- **检查点**：每耦合回合结束保存 (u_ref, v_ref, F_fluid, F_dem, fluid_state, ckpt_id)
- **守恒门红**：若某回合 momentum_conservation 或 energy_conservation 失败：
  - 回滚到上一检查点（恢复 u_ref, v_ref, fluid_state）
  - 停止后续耦合（fail-closed，不烧剩余预算）
  - validation_report 记录回滚原因与回合号
- **正常路径**：无回滚，检查点链完整

## 自由度与强制记录

- 流体求解器：TensorFVM SIMPLE（与 S3 一致）
- 结构求解器：TensorFEM frame2d（与 S2/S3 一致）
- DEM 载荷：读 S1 工件 loads.json（与 S3 一致）
- 耦合回合数：n_coupling = 20（可调，但必须记录）
- 子循环数：n_sub = 10（每结构步内流体子步数）
- 壁面速度分布：均匀（v_wall 均匀施加到楔形壁面所有节点）

## 禁止事项

- 不得修改 S1/S2/S3 已有工件
- 不得为过门反向调门：判据阈值固定如上，Fail 就如实 Fail
- 不得修改 TensorFVM/TensorFEM 核心代码（只用 Python API）
- 不得粉饰守恒门失败：回滚就记录回滚，不停就继续跑

## 实现要求

- 场景编排器 scenarios/wedge_icebreaking/s4.py，argparse 与 s3.py 对齐：
  --work-dir（默认 outputs/wedge-icebreaking-s4）、--scenario-id、--eco-root、--venv-eco
- 实跑 work-dir 固定 /nfs/wangxi/runs/wedge_s1_20261018/outputs、
  scenario-id wedge-icebreaking-s4-m124（独立目录，不覆盖任何既有工件）
- S1 反力历史读 outputs/leg_loads/loads.json
- 结构腿复用 S2/S3 的 TensorFEM frame2d 用法
- 流体腿复用 S3 的 TensorFVM SIMPLE + 壁面力计算（两套算法）
- 新增：壁面速度边界条件、子循环、检查点、守恒账本

## 验收纪律（m45/m49 驳回教训）

- 提请验收时，一切判据数字必须当场用工具直读 validation_report.json 等工件抄录
- checks 如实：过就过，不过就不过
- 守恒门失败允许 ≤3 轮自修复（调整耦合参数、检查点策略），仍不过则 honest FAIL
- 回滚必须记录：回滚回合号、原因、恢复的检查点 ID

## 预算

≤120 工具调用；墙钟 ≤10800s；完成后 commit（中文提交信息）再提验收。
