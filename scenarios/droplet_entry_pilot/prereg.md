# MP-DROP-0 预注册：液滴入水小规模证伪先导

**版本**: v1.0  
**日期**: 2026-10-25  
**状态**: 冻结（md5 见下方）  
**Mission**: m303

---

## 1. 案例定义

### 1.1 物理问题
单液滴（2D 圆形截面）自静止状态入静水池，2D 平面域。

### 1.2 求解器
- **TensorLBM**: `multiphase_water_entry.py` (2D cylinder, Color-Gradient 模型)
- **TensorSPH**: `droplet_impact.py` (2D WCSPH, 无表面张力)

### 1.3 关键限制（诚实披露）
1. **表面张力缺失**：
   - TensorLBM CG 模型：表面张力隐含在颜色梯度中，但未显式控制
   - TensorSPH：明确无表面张力模型（压力钳制在零，抑制拉伸不稳定但不模拟毛细效应）
2. **We 数匹配受限**：由于两边均无显式表面张力控制，Weber 数无法精确匹配
3. **密度比受限**：
   - LBM: rho_water/rho_air = 0.8/0.4 = 2.0（默认，可调）
   - SPH: 单流体密度 rho0=1000，空气相通过自由表面处理（无显式空气粒子）
4. **几何差异**：
   - LBM: 刚性圆柱（bounce-back 边界）
   - SPH: 液滴（流体粒子团）

**结论**：本先导为"形态一致性"测试，非严格无量纲匹配。重点考察：
- 穿透深度时间演化是否定性一致
- 液滴/圆柱形状演化是否定性一致
- 关键无量纲数（Re, Fr）是否在合理范围内

---

## 2. 参数冻结

### 2.1 TensorLBM 参数
```yaml
mode: 2d
model: cg  # Color-Gradient（最稳定）
nx: 256
ny: 512
radius: 16  # 液滴半径（格子单位）
water_level: 256  # 水面位置（y 索引）
clearance: 8  # 圆柱底部到水面间隙
rho_water: 0.8
rho_air: 0.4
density_ratio: 2.0
G: 0.9  # SC 耦合常数（CG 模型不使用）
tau: 1.0  # 松弛时间
g: 5e-5  # 重力（格子单位）
n_steps: 3000
output_interval: 300
device: cpu
```

**导出物理量**：
- 特征长度 D = 2*radius = 32 格子
- 特征速度 U = sqrt(g*D) ≈ sqrt(5e-5 * 32) ≈ 0.04 格子/步
- 运动粘度 nu = (tau - 0.5) / 3 = 0.5/3 ≈ 0.167 格子²/步
- Re_d = U*D/nu ≈ 0.04*32/0.167 ≈ 7.7
- Fr = U/sqrt(g*D) = 1.0（定义）

### 2.2 TensorSPH 参数
```yaml
dx: 0.005  # 粒子间距（米）
h_factor: 1.3
rho0: 1000.0  # 参考密度（kg/m³）
sound_speed: 20.0  # 声速（m/s）
gamma: 7.0  # Tait 指数
gravity: 0.5  # 重力（m/s²）——调低以匹配 LBM 时间尺度
viscosity: 0.001  # 运动粘度（m²/s）
dt: 1.0e-4  # 时间步（s）
steps: 1500
output_every: 50
pool_width: 1.28  # 水池宽度（m）——匹配 LBM 256*dx
pool_depth: 0.64  # 水池深度（m）——匹配 LBM water_level
drop_radius: 0.08  # 液滴半径（m）——匹配 LBM radius
drop_center_y: 0.80  # 液滴中心 y（m）
device: auto
boundary: wall
wall_layers: 3
```

**导出物理量**：
- 特征长度 D = 2*drop_radius = 0.16 m
- 特征速度 U = sqrt(g*D) = sqrt(0.5*0.16) ≈ 0.283 m/s
- Re_d = U*D/nu = 0.283*0.16/0.001 ≈ 45
- Fr = U/sqrt(g*D) = 1.0（定义）
- We = rho*U²*D/sigma → **未定义**（无表面张力）

### 2.3 无量纲数对比
| 无量纲数 | TensorLBM | TensorSPH | 匹配状态 |
|---------|-----------|-----------|---------|
| Fr | 1.0 | 1.0 | ✅ 匹配 |
| Re_d | ~7.7 | ~45 | ⚠️ 量级差异（LBM 更低） |
| We | N/A | N/A | ❌ 均无显式表面张力 |
| 密度比 | 2.0 | ∞（单流体） | ❌ 本质差异 |

**披露**：Re 数差异约 6 倍，密度比本质不同。本先导为"定性形态"测试，非定量验证。

---

## 3. 观测量定义

### 3.1 穿透深度 d(t)/D
**定义**：液滴/圆柱质心相对于初始位置的下降距离，无量纲化为 D。

**LBM 抽取**：
- 从密度场识别圆柱（高密度区域，rho > 0.5*(rho_water+rho_air)）
- 计算高密度区域的质心 (cx, cy)
- d(t) = cy_initial - cy(t)（y 向下为正）
- d(t)/D = d(t) / (2*radius)

**SPH 抽取**：
- 从粒子位置识别液滴（y > pool_depth 的粒子）
- 计算液滴粒子的质心 (cx, cy)
- d(t) = cy_initial - cy(t)
- d(t)/D = d(t) / (2*drop_radius)

**标记时刻**：t* = t*U/D = 0.5, 1.0, 1.5, 2.0

### 3.2 形状比 λ(t) = W/H
**定义**：液滴/圆柱的宽度 W 与高度 H 之比。

**LBM 抽取**：
- 从密度场识别液滴区域（阈值同上）
- 计算区域的 bounding box：x_min, x_max, y_min, y_max
- W = x_max - x_min, H = y_max - y_min
- λ = W/H

**SPH 抽取**：
- 从液滴粒子位置计算 bounding box
- W = x_max - x_min, H = y_max - y_min
- λ = W/H

**标记时刻**：t* = 0.5, 1.0, 1.5

### 3.3 （可选）空腔/冠最大宽度
**定义**：入水后水面空腔或溅射冠的最大宽度。

**LBM 抽取**：
- 从密度场识别水面（rho ≈ 0.5*(rho_water+rho_air) 的等值线）
- 在圆柱下方寻找空腔区域（低密度区）
- 计算空腔最大宽度

**SPH 抽取**：
- 从粒子位置识别自由表面（y ≈ pool_depth 附近的粒子）
- 计算溅射冠的最大宽度

**状态**：可选，若抽取不稳定则跳过。

---

## 4. 判决规则

### 4.1 定性形态一致
- 两边均应观察到：液滴/圆柱入水 → 穿透 → 空腔形成 → 可能的水冠溅射
- 若一边完全无穿透或数值爆炸，判 FAIL

### 4.2 定量容差
**逐观测量相对偏差** ≤ 25% @ 匹配 t*

相对偏差定义：
```
error = |obs_LBM - obs_SPH| / max(|obs_LBM|, |obs_SPH|, epsilon)
```
其中 epsilon = 1e-6（避免除零）。

**判决**：
- PASS：所有标记时刻的所有观测量相对偏差 ≤ 25%
- FAIL：任一观测量在任一时刻偏差 > 25%

### 4.3 失败归因
若 FAIL，需归因到哪个求解器的已知局限：
- LBM 局限：CG 模型界面扩散、低 Re 数值粘性、密度比受限
- SPH 局限：无表面张力、WCSPH 可压缩性、拉伸不稳定、单流体近似

文献锚：
- LBM 多相流：Chen & Doolen (1998) Annu. Rev. Fluid Mech.
- SPH 液滴入水：Antuono et al. (2012) JCP

---

## 5. 规模纪律

- **CPU 优先**：GPU5 被 SIR-1b τ 阶梯占用
- LBM: 256×512 格子，CPU，预计 < 5 分钟
- SPH: ~20k 粒子，CPU/GPU auto，预计 < 2 分钟

---

## 6. 工件命名

- 预注册：`scenarios/droplet_entry_pilot/prereg.md`（本文件）
- LBM 适配器：`scenarios/droplet_entry_pilot/run_lbm.py`
- SPH 适配器：`scenarios/droplet_entry_pilot/run_sph.py`
- 观测量抽取：`scenarios/droplet_entry_pilot/extract_observables.py`
- 对比脚本：`scenarios/droplet_entry_pilot/compare.py`
- 验证报告：`scenarios/droplet_entry_pilot/validation_report_m303.json`
- 判决：`scenarios/droplet_entry_pilot/verdict.json`

---

## 7. 冻结确认

**冻结时间**: 2026-10-25T13:00:00Z  
**冻结人**: tensor-expand (m303)  
**MD5**: 待计算（本文件最终版本）

```bash
md5sum scenarios/droplet_entry_pilot/prereg.md
```

---

## 8. 扩编决策

- **PASS** → 附 MP-DROP 扩编提案（仅报告，不开战役）
  - 建议：扩展到 3D、更高 Re、显式表面张力模型
- **FAIL** → 偏差归因 + 是否值得扩编的建议
  - 若归因到已知局限且有文献支持 → 记录差距，不建议扩编
  - 若归因到未知 bug → 建议扩编为战役深入调查
