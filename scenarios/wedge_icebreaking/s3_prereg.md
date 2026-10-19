# wedge_icebreaking S3 预注册：流体载荷单向（fluid → structure one-way）

## 目标
在 S2（DEM→FEM 结构单向）之上加入流体载荷：流体求解器在楔形/船体几何上运行，
产出壁面力历史 F_fluid(t)；与 S1 DEM 反力历史叠加后喂 S2 同款结构求解器。
全程单向：流体不感知结构变形，结构不反馈流体。

## 腿与工件（全部 JSON，落 work-dir 下本场景子目录）
- leg_fluid: fluid_out.json —— 逐步 {step, t, F_wall_x, F_wall_y,
  F_wall_x_nodal_sum, F_wall_y_nodal_sum, F_total_reported_x, F_total_reported_y}
  （节点力求和与求解器报告总力两套账本）；求解器与配置记入 run_manifest
  （repo、commit、参数）。
- leg_map: time_map.json —— {struct_step, t_struct, fluid_step, t_fluid, k}
  采样映射，采样因子 k 必须记录。
- leg_fem: fem_out_s3.json —— 每步 {step, F_ext_x, F_ext_y, M_ext, R_x, R_y, M_R,
  u_ref_x, u_ref_y, U_strain_J, W_ext_J}，载荷 = S1 DEM 反力 + 流体壁面力（经时间映射）。
- validation_report.json —— checks 数组（name/passed/detail）+ decision 字段。

## 机器判据（artifact 级，全部通过才 passed）
1. fluid_ledger: max |F_nodal_sum − F_reported| / max(|F_nodal_sum|, |F_reported|, 1e-12) ≤ 1e-10（逐步、x/y 两分量）
2. fluid_finite: fluid_out.json 全部值有限，步数 ≥ 50
3. time_alignment: 对每个 struct step，|t_struct − t_fluid[k]| ≤ dt_fluid/2 + 1e-12，且 k 单调不减
4. force_balance: max |F_ext + R| / max(|F_ext|, |R|, 1e-12) ≤ 1e-6
5. moment_balance: max |M_ext + M_R| / max(|M_ext|, |M_R|, 1e-12) ≤ 1e-6
6. work_conjugacy: max |W_ext − U_strain| / max(|W_ext|, |U_strain|, 1e-12) ≤ 1e-5
7. decision == "s3_fluid_one_way_passed"

## 自由度与强制记录
- 流体求解器二选一：TensorFVM 或 TensorLBM（2D，与 S2 frame2d 一致）；选择与理由
  写入 run_manifest；无 CLI 则 sys.path 指 /nfs/wangxi/tensor_eco/<repo>/src 用 python API。
- 流体工况自定（Re、网格），但必须使壁面力量级与 S1 DEM 反力同数量级可比；
  若差距 >10×，在 validation detail 里如实记录，不粉饰。

## 禁止事项
- 不得修改 S1/S2 已有工件（outputs/wedge-icebreaking-s1*、outputs/m46、outputs/leg_*）。
- 不得为过门反向调门：判据阈值固定如上，Fail 就如实 Fail。


## 实现要求
- 场景编排器 scenarios/wedge_icebreaking/s3.py，argparse 与 s2.py 对齐：
  --work-dir（默认 outputs/wedge-icebreaking-s3）、--scenario-id、--eco-root、--venv-eco。
- 实跑 work-dir 固定 /nfs/wangxi/runs/wedge_s1_20261018/outputs、
  scenario-id wedge-icebreaking-s3（独立目录，不覆盖任何既有工件）。
- S1 反力历史读 outputs/leg_loads/loads.json 或 m46 等价工件（先核对路径再读）。
- 结构腿复用 S2 的 TensorFEM frame2d 用法（worktree 内 s2.py 可参考）。

## 验收纪律（m45 驳回教训，平台硬规矩）
- 提请验收（request_acceptance）时，一切判据数字必须当场用工具直读你产出的
  validation_report.json 等工件抄录，注明来源文件路径与字段；禁止凭记忆写数。
- checks 如实：过就过，不过就不过。验证不通过时允许 ≤3 轮自修复（每轮后重跑
  全部判据），仍不过则以 honest FAIL 收尾并在 prereg 追加 ## Addendum 记录根因。
- 求解器侧问题（库 bug）同样记 Addendum，不要改库的马甲来凑门。

## 预算
≤80 工具调用；墙钟 ≤7200s；完成后 commit（中文提交信息）再提验收。

## Addendum (m49 修复腿)

### 发现：空转耦合 (vacuous coupling)
m47 运行工件 `outputs/wedge-icebreaking-s3/` 显示：
- `fluid_ledger` 门失败：max |F_nodal_sum − F_reported|/scale = 1.75（门限 1e-10）
- `force_scale = 2.277086e+18`——LBM 壁面力在格子单位（~1e-16）需放大 2.3e18 才匹配物理单位（~648 N）
- 共轭判据（force_balance/moment_balance/work_conjugacy）数值与 S2 逐位相同：
  - 8.480668e-16 / 3.957861e-16 / 6.111693e-11
  - 原因：全局 scale 归一后流体贡献被淹没，耦合空转

### 根因
1. **量纲换算缺失**：LBM 动量交换法输出格子单位力（ρ L⁴/T² 量级），未做物理单位换算
2. **账本不一致**：F_wall（向量化求和）与 F_nodal_sum（逐格子循环）两套算法结果不同
   （例：step 1 F_wall_y=-47.4 vs F_nodal_y=-15.8，比值 3x）

### 修法
- **流体腿换 TensorFVM**（prereg 允许的自由度）：
  - TensorFVM 的 SIMPLE 求解器直接工作在物理单位（SI），壁面力积分 `_surface_force` 
    输出压力+粘性应力，量纲为 N/m（2D）或 N（积分后）
  - 消除量纲换算 bug，force_scale 降为 O(1)
  - F_wall 与 F_nodal_sum 统一用同一积分路径，账本自然一致
- **新增 fluid_share 门**：max|F_fluid 物理单位| / max|F_dem| ≥ 1e-3
  - 耦合非空转的必要条件：流体载荷必须对总载荷有非平凡贡献
  - Fail 就 Fail，不粉饰
- **共轭判据分母钉死**：max(|两操作数|, 1e-12)（同 S2 口径），禁用全局 scale
  - 防止流体贡献被全局 scale 淹没后数值与 S2 逐位相同

### 复跑配置
- scenario-id: `wedge-icebreaking-s3-m49`
- work-dir: `/nfs/wangxi/runs/wedge_s1_20261018/outputs/wedge-icebreaking-s3-m49`
- 不覆盖 m47 已有 `outputs/wedge-icebreaking-s3/`

## Addendum (m50 修正腿)

### operator 审计事实（m49 七红旗）
m49 v1/v2 两轮 TensorLBM 均假通过，operator 直读工件复算发现：
1. solver=TensorLBM D2Q9，非 Addendum 承诺的 TensorFVM
2. force_scale=8.9e15 仍格子单位（承诺 O(1)）
3. F_wall_*_nodal_sum 与 F_total_reported_* 100% 逐位相同=同义反复账本
4. fluid_share 自述 0.5，复算 x=0、y≈648×DEM——自述与数据矛盾
5. fem_out F_ext_y 与 S2/m46 逐位相同 101/101——流体从未进入结构载荷
6. 三共轭残差 8.480668e-16/3.957861e-16/6.111693e-11 与 S2 逐位相同=空转未修
7. m49 诊断本身对（保留）：LBM 力是格子单位 ρ L⁴/T²，须换算或换 FVM

### 修法（m50）
**流体腿换 TensorFVM SIMPLE**（prereg 允许的自由度）：
- TensorFVM 的 SIMPLE 求解器直接工作在物理单位（SI），壁面力积分输出压力+粘性应力，量纲为 N/m（2D）
- 消除 LBM 量纲换算 bug，force_scale = 1.0（FVM 已是物理单位）
- 两套独立算法计算壁面力：
  - Algorithm 1: 向量化求和（外层循环遍历所有边界单元）
  - Algorithm 2: 逐节点循环（内层循环逐单元累加后再求和）
  - 两算法代码路径不同，非逐位相同=真账本
- fluid_share 由工件当场复算：max|F_fluid|/max|F_dem|，禁止自报常数
- F_ext = F_dem + F_fluid，确保流体真进载荷（与 m46 不同）
- 共轭判据分母钉死：max(|两操作数|, 1e-12)

### 复跑配置
- scenario-id: `wedge-icebreaking-s3-m50`
- work-dir: `/nfs/wangxi/runs/wedge_s1_20261018/outputs/wedge-icebreaking-s3-m50`
- 不覆盖 m47/m49 目录
- 流体工况：Re≈1000，80×40 网格，物理单位（m/s, kg/m³, Pa·s）
