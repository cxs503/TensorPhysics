# P84-W2C 战役集成验收门（P84W2C-N11）——全节点验收汇总与跨域接口互裁

- 产物 schema：`campaign_gate_report.v1` v1.0.0 → fingerprint `801f46d8d01979b6ec5e3f4530b4870a09b20b0c72632358b81cb2524e53a7a2`
- 战役计划：plan_id=385 · P84-W2C seed (谱系继承波·Plan 84 Wave 2C)
- 生成时间（UTC）：`2026-11-09T04:40:27.795539+00:00`
- **总裁决：`FAIL` · 阻塞项 11 条**

## 1. 案例介绍

P84-W2C 是谱系继承波（Wave 2C）：十个并行研究节点覆盖求解器验证前沿（TensorLBM 翼型气动、LES 湍流、悬沙冲刷；TensorBEM 阴极保护；TensorDEM 颗粒卸料）与船舶工程成熟化（岸电排放折减、ORC 余热回收 EEDI credit、货物系固、进坞墩木、谷物稳性）。节点之间不共享代码，只通过**产物**消费彼此结果，因此「战役是否可交付」不能由 PR 是否合并判断，只能由「产物是否落盘、pytest 是否可解析且绿、跨域接口是否真的对得上」判断。

本门汇总 10 个节点的验收结果：节点裁决 PASS 5 / FAIL 4 / WITHHELD 1；节点级判据 41 条中 PASS 17、FAIL 0、WITHHELD 24。产物 16 个、可解析 16 个；pytest 套件执行 7 个、全绿 5 个。

| 节点 | 仓库 | 物理域 | 节点裁决 | 平台状态 | 判据通过 |
|---|---|---|---|---|---|
| P84W2C-N01 | `TensorLBM` | 空气动力学 | ❌ FAIL | `running` | 0/6 |
| P84W2C-N02 | `TensorLBM` | 湍流建模 | ❌ FAIL | `running` | 0/5 |
| P84W2C-N03 | `TensorBEM` | 电化学 | ⛔ NOT_SUPPORTED | `queued` | 0/4 |
| P84W2C-N04 | `TensorLBM` | 多相流 | ❌ FAIL | `running` | 0/4 |
| P84W2C-N05 | `TensorDEM` | 颗粒力学 | ❌ FAIL | `running` | 0/5 |
| P84W2C-N06 | `ship-performance-platform` | 绿色航行 | ✅ PASS | `succeeded` | 4/4 |
| P84W2C-N07 | `ship-performance-platform` | 绿色航行 | ✅ PASS | `succeeded` | 4/4 |
| P84W2C-N08 | `ship-performance-platform` | 舾装 | ✅ PASS | `succeeded` | 3/3 |
| P84W2C-N09 | `ship-performance-platform` | 结构 | ✅ PASS | `succeeded` | 3/3 |
| P84W2C-N10 | `ship-performance-platform` | 稳性 | ✅ PASS | `succeeded` | 3/3 |

### 1.1 谱系继承裁决

父计划 plan 365 的同名节点被逐条对表：`inherited_verified` 4 个、`inherited_unverified` 6 个、`seeded_new` 0 个。判定语义：父计划同名节点 done → inherited_verified；否则 inherited_unverified。

| 节点 | 父节点 | 父 mission | 父状态 | 谱系裁决 |
|---|---|---|---|---|
| P84W2C-N01 | N01: TensorLBM NACA0012 翼型气动验证面 | 7747 | blocked | `inherited_unverified` |
| P84W2C-N02 | N02: TensorLBM Smagorinsky LES 通道流验证面 | 7748 | done | `inherited_verified` |
| P84W2C-N03 | N03: TensorBEM 船体阴极保护电流分布验证面 | 8197 | in-flight | `inherited_unverified` |
| P84W2C-N04 | N04: TensorLBM 悬沙输运与床面冲刷验证面 | 7750 | blocked | `inherited_unverified` |
| P84W2C-N05 | N05: TensorDEM 散货舱颗粒卸料动力学验证面 | 7751 | blocked | `inherited_unverified` |
| P84W2C-N06 | N06: 船舶岸电(AMP)靠港排放折减可算面·成熟化 | 7752 | done | `inherited_verified` |
| P84W2C-N07 | N07: 主机余热回收 ORC 闭环 + EEDI credit·成熟化 | 8268 | in-flight | `inherited_unverified` |
| P84W2C-N08 | N08: 货物系固系统验证面·成熟化 | 8221 | in-flight | `inherited_unverified` |
| P84W2C-N09 | N09: 进坞/坞墩工况可判面·成熟化 | 7755 | done | `inherited_verified` |
| P84W2C-N10 | N10: 谷物稳性(International Grain Code)可算面·成熟化 | 7756 | done | `inherited_verified` |

## 2. 怎么做的

判据实现在 `src/tensorphysics/campaign/p84w2c_gate/`（stdlib-only，不 import 任何求解器栈，也不 import 消费方运行时）：`spec.py` 只声明节点/判据/契约，`collect.py` 采集磁盘证据并**亲自执行**节点的 pytest 套件（junit XML → 归一化 JSON），`contracts.py` 用 `ast` 逐侧检查消费方声明并做数值锚对比，`criteria.py` 判 C1–C4，`lineage.py` 做谱系继承裁决。

| 战役判据 | 内容 | 判据数字 | 裁决 |
|---|---|---|---|
| C1 | 所有节点 pytest 结果可解析 (JSON) | pytest JSON 可解析 9/10；执行 7；全绿 5 | ❌ FAIL |
| C2 | 跨域接口契约验证通过 | 必需契约 2/2 通过；数据级一致性 PASS 0 | ✅ PASS |
| C3 | 战役报告 markdown ≥ 3000 字符 | 字符数见 JSON markdown.chars（下限 3000）；章节 6/6；图件 True | ✅ PASS |
| C4 | 端到端输出战役验收 JSON（schema 校验 + 指纹落盘） | 裁决 JSON schema 指纹 78929781ef79554de7acd16d91db1d19cea777412aba9da5a65b068c41c538e3；节点 10、判据 41 | ✅ PASS |

复现命令（证据全部落在 `artifacts/p84w2c_gate/`）：

```bash
python -m tensorphysics.campaign.p84w2c_gate \
    --out artifacts/p84w2c_gate --doc docs/P84W2C_CAMPAIGN_GATE.md \
    --auto-root-template '/root/tp-agent/work/m{mission_id}' \
    --consumer-root ship-performance-platform=<ship-platform 检出> \
    --plan-id 385 --parent-plan-id 365 --api auto
python -m pytest tests/test_p84w2c_gate.py -q
```

## 3. 计算结果

### 3.1 战役判据裁决与数字

| 判据 | 陈述 | 裁决 | 判据数字 |
|---|---|---|---|
| C1 | 所有节点 pytest 结果可解析 (JSON) | ❌ FAIL | pytest JSON 可解析 9/10；执行 7；全绿 5 |
| C2 | 跨域接口契约验证通过 | ✅ PASS | 必需契约 2/2 通过；数据级一致性 PASS 0 |
| C3 | 战役报告 markdown ≥ 3000 字符 | ✅ PASS | 字符数见 JSON markdown.chars（下限 3000）；章节 6/6；图件 True |
| C4 | 端到端输出战役验收 JSON（schema 校验 + 指纹落盘） | ✅ PASS | 裁决 JSON schema 指纹 78929781ef79554de7acd16d91db1d19cea777412aba9da5a65b068c41c538e3；节点 10、判据 41 |

### 3.2 跨域接口契约（双侧验证）

| 契约 | 生产侧节点 | 生产 schema | 消费侧 | 契约裁决 | 数据级一致性 | 检查项通过 |
|---|---|---|---|---|---|---|
| `aero-wind` | P84W2C-N01 | `aero_polar.v1` | `ship-performance-platform:backend/api/dynamic_stability.py` | ✅ PASS | `WITHHELD` | 9/10 |
| `sediment-ukc` | P84W2C-N04 | `sediment_summary.v1` | `ship-performance-platform:backend/api/calculations.py` | ⚪ WITHHELD | `WITHHELD` | 2/4 |
| `amp-eexi` | P84W2C-N06 | `amp_emissions.v1` | `ship-performance-platform:backend/api/green_shipping.py` | ✅ PASS | `WITHHELD` | 4/5 |

#### 契约 `aero-wind` 检查明细（指纹 `615f7798609264f8`）

| 检查 | 裁决 | 数值 | 阈值 | 说明 |
|---|---|---|---|---|
| consumer_module_present | ✅ PASS | — | — | /root/tp-agent/work/m8018/backend/api/dynamic_stability.py |
| consumer_symbols | ✅ PASS | — | — | found ['DynamicStabilityInput'] |
| consumer_fields | ✅ PASS | — | — | all 3 required inputs declared |
| constant:RHO_AIR | ✅ PASS | 0 | 1.000e-12 | consumer=1.225 contract=1.225 |
| constant:WIND_PRESSURE_IS_CODE | ✅ PASS | 0 | 1.000e-12 | consumer=504.0 contract=504.0 |
| consumer_law_shape | ✅ PASS | — | — | 0.5 * self.rho_air * input_data.wind_speed ** 2 |
| producer_schema_fields | ✅ PASS | — | — | schema aero_polar.v1 carries all 5 fields |
| numeric_anchor | ✅ PASS | 0 | 1.000e-12 | probes (10.0, 15.7, 25.7) m/s, rho=1.225 |
| is_code_cross_anchor | ✅ PASS | 1.128e-16 | 1.000e-12 | 504 Pa -> V*=28.685487 m/s -> 504.000000000 Pa |
| producer_data_conformance | ⚪ WITHHELD | — | — | producer artifact with declared schema not delivered yet |

#### 契约 `sediment-ukc` 检查明细（指纹 `895befa5586a2a03`）

| 检查 | 裁决 | 数值 | 阈值 | 说明 |
|---|---|---|---|---|
| consumer_module_present | ✅ PASS | — | — | /root/tp-agent/work/m8018/backend/api/calculations.py |
| consumer_fields | ⚪ WITHHELD | — | — | consumer endpoint not implemented yet, missing identifiers: ['ukc'] |
| producer_schema_fields | ✅ PASS | — | — | schema sediment_summary.v1 carries all 5 fields |
| producer_data_conformance | ⚪ WITHHELD | — | — | producer artifact with declared schema not delivered yet |

缺口：
- 消费侧缺少标识符 ['ukc']

#### 契约 `amp-eexi` 检查明细（指纹 `c1499bae69f525a7`）

| 检查 | 裁决 | 数值 | 阈值 | 说明 |
|---|---|---|---|---|
| consumer_module_present | ✅ PASS | — | — | /root/tp-agent/work/m8018/backend/api/green_shipping.py |
| consumer_symbols | ✅ PASS | — | — | found ['EediRequest', 'EexiRequest'] |
| consumer_fields | ✅ PASS | — | — | all 4 required inputs declared |
| producer_schema_fields | ✅ PASS | — | — | schema amp_emissions.v1 carries all 5 fields |
| producer_data_conformance | ⚪ WITHHELD | — | — | producer artifact with declared schema not delivered yet |

### 3.3 逐节点验收明细

#### P84W2C-N01 · TensorLBM NACA0012 翼型气动验证面

- 仓库：`TensorLBM` · 物理域：空气动力学 · mission=8013 · 平台状态：`running`
- 节点裁决：**❌ FAIL** · 产物家族 0 个（可解析 0） · pytest `FAIL`（mode=collect-only, collected=9991, passed=0, failed=0, errors=2）

| 判据 | 陈述 | 值 | 阈值 | 裁决 |
|---|---|---|---|---|
| C1 | 升力线斜率 dC_L/dα 与薄翼理论 2π 相对偏差 < 5% | — | 0.05 | ⚪ WITHHELD |
| C2 | α=0° 对称翼型零升力 |C_L| < 1e-4 | — | 1.000e-04 | ⚪ WITHHELD |
| C4 | 细网格 C_L 的 GCI < 2% | — | 0.02 | ⚪ WITHHELD |
| C5 | 进出口质量流量不平衡 < 1e-6 | — | 1.000e-06 | ⚪ WITHHELD |
| C2b | 零升阻力 C_D0 与 Abbott 表值相对偏差 < 15% | — | 0.15 | ⚪ WITHHELD |
| C3 | 失速角与实验值 (14°±2°) 偏差 < 3° | — | 3 | ⚪ WITHHELD |

未通过项 / gap：
- 未落盘任何 JSON 产物 (artifacts/naca0012_aero/*.json, artifacts/naca0012_aero/*.png)
- 声明的 pytest 套件未执行 (mode=collect-only, declared=['tests/test_naca0012_aero.py'])
- 判据 C1 无数字证据
- 判据 C2 无数字证据
- 判据 C4 无数字证据
- 判据 C5 无数字证据
- 判据 C2b 无数字证据
- 判据 C3 无数字证据

#### P84W2C-N02 · TensorLBM Smagorinsky LES 通道流验证面

- 仓库：`TensorLBM` · 物理域：湍流建模 · mission=8014 · 平台状态：`running`
- 节点裁决：**❌ FAIL** · 产物家族 0 个（可解析 0） · pytest `FAIL`（mode=executed, collected=21, passed=15, failed=6, errors=0）

| 判据 | 陈述 | 值 | 阈值 | 裁决 |
|---|---|---|---|---|
| C1 | 对数区 U+ 与 log-law 偏差 < 5% | — | 0.05 | ⚪ WITHHELD |
| C2 | 摩擦雷诺数 Re_tau 与目标 180 偏差 < 3% | — | 0.03 | ⚪ WITHHELD |
| C4 | 三档网格 U+ 剖面 GCI < 5% | — | 0.05 | ⚪ WITHHELD |
| C6 | 通道中心 SGS 应力占比 < 30% | — | 0.3 | ⚪ WITHHELD |
| C7 | 周期边界质量流量不平衡 < 1e-8 | — | 1.000e-08 | ⚪ WITHHELD |

未通过项 / gap：
- 未落盘任何 JSON 产物 (artifacts/channel_les/*.json, artifacts/channel_les/*.png)
- pytest 未全绿: passed=15 failed=6 errors=0
- 判据 C1 无数字证据
- 判据 C2 无数字证据
- 判据 C4 无数字证据
- 判据 C6 无数字证据
- 判据 C7 无数字证据

#### P84W2C-N03 · TensorBEM 船体阴极保护电流分布验证面

- 仓库：`TensorBEM` · 物理域：电化学 · mission=8362 · 平台状态：`queued`
- 节点裁决：**⛔ NOT_SUPPORTED** · 产物家族 0 个（可解析 0） · pytest `NOT_SUPPORTED`（mode=missing, collected=0, passed=0, failed=0, errors=0）

| 判据 | 陈述 | 值 | 阈值 | 裁决 |
|---|---|---|---|---|
| C1 | BEM 线性系统相对残差 < 1e-10 | — | 1.000e-10 | ⚪ WITHHELD |
| C2 | 阳极输出总电流 = 阴极吸收总电流，相对误差 < 1e-8 | — | 1.000e-08 | ⚪ WITHHELD |
| C3 | DNV 保护电位区间覆盖面积比 ≥ 95% | — | 0.95 | ⚪ WITHHELD |
| C5 | 三档 BEM 网格关键点电位 GCI < 3% | — | 0.03 | ⚪ WITHHELD |

未通过项 / gap：
- 节点工作树缺失: /root/tp-agent/work/m8362

#### P84W2C-N04 · TensorLBM 悬沙输运与床面冲刷验证面

- 仓库：`TensorLBM` · 物理域：多相流 · mission=8016 · 平台状态：`running`
- 节点裁决：**❌ FAIL** · 产物家族 0 个（可解析 0） · pytest `FAIL`（mode=executed, collected=1, passed=0, failed=0, errors=1）

| 判据 | 陈述 | 值 | 阈值 | 裁决 |
|---|---|---|---|---|
| C1 | 数值起动 Shields 曲线与 Soulsby–Whitehouse 偏差 ≤ 15% | — | 0.15 | ⚪ WITHHELD |
| C2 | Rouse 指数 Z 与解析值相对误差 < 5% | — | 0.05 | ⚪ WITHHELD |
| C3 | 泥沙总量（悬移+床面）漂移 < 1e-6 | — | 1.000e-06 | ⚪ WITHHELD |
| C5 | 单桩平衡冲刷 S/D 落在 Melville–Chiew 区间 [1.0, 2.4] | — | [1.0, 2.4] | ⚪ WITHHELD |

未通过项 / gap：
- 未落盘任何 JSON 产物 (artifacts/p84_w1/*.json)
- pytest 未全绿: passed=0 failed=0 errors=1
- 判据 C1 无数字证据
- 判据 C2 无数字证据
- 判据 C3 无数字证据
- 判据 C5 无数字证据

#### P84W2C-N05 · TensorDEM 散货舱颗粒卸料动力学验证面

- 仓库：`TensorDEM` · 物理域：颗粒力学 · mission=8017 · 平台状态：`running`
- 节点裁决：**❌ FAIL** · 产物家族 0 个（可解析 0） · pytest `FAIL`（mode=collect-only, collected=4576, passed=0, failed=0, errors=2）

| 判据 | 陈述 | 值 | 阈值 | 裁决 |
|---|---|---|---|---|
| C1 | 休止角与 atan(mu) 偏差 < 10° 且随 mu 单调 | — | 10 | ⚪ WITHHELD |
| C2 | Beverloo 标度律拟合 R² > 0.95 | — | 0.95 | ⚪ WITHHELD |
| C3 | Beverloo 常数 C 与 0.58 偏差 < 20% | — | 0.2 | ⚪ WITHHELD |
| C5 | 卸料总质量 = 初始填充质量，相对误差 < 1e-6 | — | 1.000e-06 | ⚪ WITHHELD |
| C6 | 能量账本残差 < 1e-8 | — | 1.000e-08 | ⚪ WITHHELD |

未通过项 / gap：
- 未落盘任何 JSON 产物 (artifacts/hopper_discharge/*.json, artifacts/hopper_discharge/*.png)
- 声明的 pytest 套件未执行 (mode=collect-only, declared=['tests/test_hopper_discharge.py'])
- 判据 C1 无数字证据
- 判据 C2 无数字证据
- 判据 C3 无数字证据
- 判据 C5 无数字证据
- 判据 C6 无数字证据

#### P84W2C-N06 · 船舶岸电(AMP)靠港排放折减可算面·成熟化

- 仓库：`ship-performance-platform` · 物理域：绿色航行 · mission=8018 · 平台状态：`succeeded`
- 节点裁决：**✅ PASS** · 产物家族 2 个（可解析 2） · pytest `PASS`（mode=executed, collected=15, passed=15, failed=0, errors=0）

| 判据 | 陈述 | 值 | 阈值 | 裁决 |
|---|---|---|---|---|
| C1 | 辅机负荷平衡相对误差 < 1e-9 | 0 | 1.000e-09 | ✅ PASS |
| C2 | CO2 = fuel × EF 质量守恒相对误差 < 1e-10 | 0 | 1.000e-10 | ✅ PASS |
| C3 | 岸电折减 CO2 非负 | 0 | 0 | ✅ PASS |
| C5 | 港口电网因子与公开数据偏差 < 5% | 0.0198675 | 0.05 | ✅ PASS |

| 产物 | 类型 | 可解析 | sha256(前16) |
|---|---|---|---|
| `/root/tp-agent/work/m8018/artifacts/shore_power/case.json` | json | True | `6b388f0a828e7d3e` |
| `/root/tp-agent/work/m8018/artifacts/shore_power/results.json` | json | True | `9c2e3f9c68c8b6d2` |

#### P84W2C-N07 · 主机余热回收 ORC 闭环 + EEDI credit·成熟化

- 仓库：`ship-performance-platform` · 物理域：绿色航行 · mission=8019 · 平台状态：`succeeded`
- 节点裁决：**✅ PASS** · 产物家族 3 个（可解析 3） · pytest `PASS`（mode=executed, collected=23, passed=23, failed=0, errors=0）

| 判据 | 陈述 | 值 | 阈值 | 裁决 |
|---|---|---|---|---|
| C1 | 循环能量守恒残差 < 1e-6·Q_in | -2.548e-16 | 1.000e-06 | ✅ PASS |
| C2 | η_th / η_Carnot ∈ [0.35, 0.70] | 0.567513 | [0.35, 0.7] | ✅ PASS |
| C3 | 75% 负荷最小换热温差 ≥ 10 K | 10 | 10 | ✅ PASS |
| C5 | EEDI 折减双算法相对差 < 1e-6 | 1.471e-16 | 1.000e-06 | ✅ PASS |

| 产物 | 类型 | 可解析 | sha256(前16) |
|---|---|---|---|
| `/root/tp-agent/work/m8019/artifacts/whr_orc/case.json` | json | True | `7541dee3cf262918` |
| `/root/tp-agent/work/m8019/artifacts/whr_orc/end_to_end_payload.json` | json | True | `90cdc7045cc96a30` |
| `/root/tp-agent/work/m8019/artifacts/whr_orc/results.json` | json | True | `161ef2c8fa9b33f2` |

#### P84W2C-N08 · 货物系固系统验证面·成熟化

- 仓库：`ship-performance-platform` · 物理域：舾装 · mission=8020 · 平台状态：`succeeded`
- 节点裁决：**✅ PASS** · 产物家族 3 个（可解析 3） · pytest `PASS`（mode=executed, collected=16, passed=16, failed=0, errors=0）

| 判据 | 陈述 | 值 | 阈值 | 裁决 |
|---|---|---|---|---|
| C1 | 加速度椭圆双实现相对误差 < 1e-12 | 2.209e-16 | 1.000e-12 | ✅ PASS |
| C2 | 每工况 ΣF=0、ΣM=0 残差 < 1e-9 | 4.030e-16 | 1.000e-09 | ✅ PASS |
| C3 | 绑扎数 2→6 单根最大张力单调不增 (R² > 0.99) | 1 | 0.99 | ✅ PASS |

| 产物 | 类型 | 可解析 | sha256(前16) |
|---|---|---|---|
| `/root/tp-agent/work/m8020/artifacts/cargo_securing/case.json` | json | True | `804d422d351cc0c4` |
| `/root/tp-agent/work/m8020/artifacts/cargo_securing/metrics.json` | json | True | `5998c660b90ba9a5` |
| `/root/tp-agent/work/m8020/artifacts/cargo_securing/verdict.json` | json | True | `8cabaa325051137f` |

#### P84W2C-N09 · 进坞/坞墩工况可判面·成熟化

- 仓库：`ship-performance-platform` · 物理域：结构 · mission=8021 · 平台状态：`succeeded`
- 节点裁决：**✅ PASS** · 产物家族 6 个（可解析 6） · pytest `PASS`（mode=executed, collected=12, passed=12, failed=0, errors=0）

| 判据 | 陈述 | 值 | 阈值 | 裁决 |
|---|---|---|---|---|
| C1 | ΣR_i = W 静力平衡残差 < 1e-6·W | 3.641e-15 | 1.000e-06 | ✅ PASS |
| C2 | Winkler 与三弯矩法 M_max 相对差 < 1% | 0.00549853 | 0.01 | ✅ PASS |
| C4 | 全墩脱空退化到自由-自由梁解析弯矩，误差 < 1% | 1.000e-09 | 0.01 | ✅ PASS |

| 产物 | 类型 | 可解析 | sha256(前16) |
|---|---|---|---|
| `/root/tp-agent/work/m8021/artifacts/drydock/convergence.json` | json | True | `8ce28790eb3a024a` |
| `/root/tp-agent/work/m8021/artifacts/drydock/fields.json` | json | True | `24493e73cef2cc58` |
| `/root/tp-agent/work/m8021/artifacts/drydock/landing_scan.json` | json | True | `217f164e420cfd86` |
| `/root/tp-agent/work/m8021/artifacts/drydock/layout.json` | json | True | `f2ad7e502ed40d55` |
| `/root/tp-agent/work/m8021/artifacts/drydock/results.json` | json | True | `5b8d9337f63114ea` |
| `/root/tp-agent/work/m8021/artifacts/drydock/sensitivity.json` | json | True | `fe7a2545a6164977` |

#### P84W2C-N10 · 谷物稳性(International Grain Code)可算面·成熟化

- 仓库：`ship-performance-platform` · 物理域：稳性 · mission=8022 · 平台状态：`succeeded`
- 节点裁决：**✅ PASS** · 产物家族 2 个（可解析 2） · pytest `PASS`（mode=executed, collected=25, passed=25, failed=0, errors=0）

| 判据 | 陈述 | 值 | 阈值 | 裁决 |
|---|---|---|---|---|
| C1 | VHM→SF→λ 单位链双实现相对误差 < 1e-12 | 8.910e-16 | 1.000e-12 | ✅ PASS |
| C2 | 左右舷镜像配载倾侧力矩相对差 < 1e-9 | 1.732e-16 | 1.000e-09 | ✅ PASS |
| C3 | 部分装载比例 0→60% 倾侧力矩单调增 (R² > 0.99) | 1 | 0.99 | ✅ PASS |

| 产物 | 类型 | 可解析 | sha256(前16) |
|---|---|---|---|
| `/root/tp-agent/work/m8022/artifacts/grain_stability/case.json` | json | True | `1afabccbbc8b4412` |
| `/root/tp-agent/work/m8022/artifacts/grain_stability/verdict.json` | json | True | `32d845cd565bf182` |

## 4. 结果图

![P84-W2C gate matrix](artifacts/p84w2c_gate/p84w2c_gate_matrix.png)

左图：十个节点各自声明的验收判据按 PASS/FAIL/WITHHELD 堆叠，直接暴露每个节点的「黑箱程度」——灰色段越长说明该节点交付的数字越少。右图：战役级证据覆盖率，四根柱分别是「产物家族可解析率」「pytest 套件执行且全绿率」「跨域契约通过率」「战役判据 C1–C4 通过率」，全部由本 JSON 的计数现算，没有手抄数字。

## 5. 注意点

- **honest-FAIL 语义**：任一节点未交付即计入 gap 分析，`WITHHELD` 永不等于 PASS；因此本门在节点在飞阶段必然给出 FAIL，这不是门坏了，而是战役尚未可交付。
- **只在磁盘上取证**：门不跑节点的物理算例、不重算流场，只解析产物并执行节点自己的pytest 套件；节点若在报告里自称达标却未落盘 JSON，判据一律 WITHHELD。
- **外部基准由节点自报**：Abbott 表值、Soulsby–Whitehouse 曲线、DNV 电网因子等外部表格无法在本仓复算，判据要求节点在产物中显式给出 deviation 字段（来源写进 `source`），门只复核该数字是否越过阈值——这是「自报 + 双口径」的折中，局限性已记录。
- **消费方静态检查而非运行**：消费侧（ship-platform）与生产侧不同运行时，门用 `ast` 检查类字段、模块常量与动压定律 AST 形状，并做数值锚对比；它证明两侧契约一致，但不证明消费方端到端 API 调用成功（该证据应由消费方自己的集成测试提供）。
- **pytest 时间上限**：节点套件在门内执行时有超时上限，超时记为 FAIL(mode=executed,exit_code=124) 而不是静默跳过；大算例应下沉到集群并把 JSON 结果落盘。
- **谱系裁决依赖平台证据**：父计划节点状态来自 `campaign_plans`；离线模式下继承一律降级为 `inherited_unverified`，不会凭标题同名给 `inherited_verified`。

## 6. 误差判定

逐项误差与阈值对照如下（键量为节点声明的「关键量 vs 基准」）；`是否<3%` 列显式回答每个关键量是否落在 3% 以内，无独立基准的项写明「无独立基准」及其判据类型，不含糊。

| 节点 | 判据 | 关键量 | 值 | 误差 / 偏差 | 基准/阈值 | 是否<3% | 裁决 |
|---|---|---|---|---|---|---|---|
| P84W2C-N01 | C1 | 升力线斜率 dC_L/dα 与薄翼理论 2π 相对偏差 < 5% | — | 无数字证据（判据 WITHHELD） | thin-airfoil theory 2*pi | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N01 | C2 | α=0° 对称翼型零升力 |C_L| < 1e-4 | — | 无数字证据（判据 WITHHELD） | symmetry exact solution | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N01 | C4 | 细网格 C_L 的 GCI < 2% | — | 无数字证据（判据 WITHHELD） | Roache GCI | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N01 | C5 | 进出口质量流量不平衡 < 1e-6 | — | 无数字证据（判据 WITHHELD） | continuity | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N01 | C2b | 零升阻力 C_D0 与 Abbott 表值相对偏差 < 15% | — | 无数字证据（判据 WITHHELD） | Abbott & von Doenhoff (1959) (external) | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N01 | C3 | 失速角与实验值 (14°±2°) 偏差 < 3° | — | 无数字证据（判据 WITHHELD） | Abbott & von Doenhoff (1959) (external) | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N02 | C1 | 对数区 U+ 与 log-law 偏差 < 5% | — | 无数字证据（判据 WITHHELD） | kappa=0.41, B=5.2 | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N02 | C2 | 摩擦雷诺数 Re_tau 与目标 180 偏差 < 3% | — | 无数字证据（判据 WITHHELD） | MKM1999 Re_tau=180 (external) | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N02 | C4 | 三档网格 U+ 剖面 GCI < 5% | — | 无数字证据（判据 WITHHELD） | Roache GCI | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N02 | C6 | 通道中心 SGS 应力占比 < 30% | — | 无数字证据（判据 WITHHELD） | resolved-scale dominance | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N02 | C7 | 周期边界质量流量不平衡 < 1e-8 | — | 无数字证据（判据 WITHHELD） | continuity | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N03 | C1 | BEM 线性系统相对残差 < 1e-10 | — | 无数字证据（判据 WITHHELD） | Laplace/Green identity | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N03 | C2 | 阳极输出总电流 = 阴极吸收总电流，相对误差 < 1e-8 | — | 无数字证据（判据 WITHHELD） | charge conservation | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N03 | C3 | DNV 保护电位区间覆盖面积比 ≥ 95% | — | 无数字证据（判据 WITHHELD） | DNV-RP-B401 | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N03 | C5 | 三档 BEM 网格关键点电位 GCI < 3% | — | 无数字证据（判据 WITHHELD） | Roache GCI | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N04 | C1 | 数值起动 Shields 曲线与 Soulsby–Whitehouse 偏差 ≤ 15% | — | 无数字证据（判据 WITHHELD） | Soulsby & Whitehouse (1997) (external) | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N04 | C2 | Rouse 指数 Z 与解析值相对误差 < 5% | — | 无数字证据（判据 WITHHELD） | Z = w_s / (kappa u*) | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N04 | C3 | 泥沙总量（悬移+床面）漂移 < 1e-6 | — | 无数字证据（判据 WITHHELD） | Exner flux closure | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N04 | C5 | 单桩平衡冲刷 S/D 落在 Melville–Chiew 区间 [1.0, 2.4] | — | 无数字证据（判据 WITHHELD） | Melville & Chiew (1999) (external) | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N05 | C1 | 休止角与 atan(mu) 偏差 < 10° 且随 mu 单调 | — | 无数字证据（判据 WITHHELD） | Coulomb friction cone | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N05 | C2 | Beverloo 标度律拟合 R² > 0.95 | — | 无数字证据（判据 WITHHELD） | Beverloo et al. (1961) | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N05 | C3 | Beverloo 常数 C 与 0.58 偏差 < 20% | — | 无数字证据（判据 WITHHELD） | Beverloo et al. (1961) (external) | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N05 | C5 | 卸料总质量 = 初始填充质量，相对误差 < 1e-6 | — | 无数字证据（判据 WITHHELD） | mass ledger | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N05 | C6 | 能量账本残差 < 1e-8 | — | 无数字证据（判据 WITHHELD） | gravity/kinetic/dissipation ledger | 无法判定（未交付数字） | ⚪ WITHHELD |
| P84W2C-N06 | C1 | 辅机负荷平衡相对误差 < 1e-9 | 0 | |值| = 0 vs 阈 1.000e-09 | power ledger | 是（量级 <3%） | ✅ PASS |
| P84W2C-N06 | C2 | CO2 = fuel × EF 质量守恒相对误差 < 1e-10 | 0 | |值| = 0 vs 阈 1.000e-10 | MGO CF = 3.114 | 是（量级 <3%） | ✅ PASS |
| P84W2C-N06 | C3 | 岸电折减 CO2 非负 | 0 | |值| = 0 vs 阈 0 | grid EF <= aux EF | 是（量级 <3%） | ✅ PASS |
| P84W2C-N06 | C5 | 港口电网因子与公开数据偏差 < 5% | 0.0198675 | |值| = 0.0198675 vs 阈 0.05 | IEA/EEA grid emission factors (external) | 是（量级 <3%） | ✅ PASS |
| P84W2C-N07 | C1 | 循环能量守恒残差 < 1e-6·Q_in | -2.548e-16 | |值| = 2.548e-16 vs 阈 1.000e-06 | first law | 是（量级 <3%） | ✅ PASS |
| P84W2C-N07 | C2 | η_th / η_Carnot ∈ [0.35, 0.70] | 0.567513 | 值 0.567513 ∈ [0.35, 0.7] | second law bound | 无独立基准（区间判据） | ✅ PASS |
| P84W2C-N07 | C3 | 75% 负荷最小换热温差 ≥ 10 K | 10 | |值| = 10 vs 阈 10 | pinch analysis | 无独立基准（阈判据） | ✅ PASS |
| P84W2C-N07 | C5 | EEDI 折减双算法相对差 < 1e-6 | 1.471e-16 | |值| = 1.471e-16 vs 阈 1.000e-06 | MEPC.1/Circ.896 | 是（量级 <3%） | ✅ PASS |
| P84W2C-N08 | C1 | 加速度椭圆双实现相对误差 < 1e-12 | 2.209e-16 | |值| = 2.209e-16 vs 阈 1.000e-12 | CSS Code Annex 13 | 是（量级 <3%） | ✅ PASS |
| P84W2C-N08 | C2 | 每工况 ΣF=0、ΣM=0 残差 < 1e-9 | 4.030e-16 | |值| = 4.030e-16 vs 阈 1.000e-09 | static equilibrium | 是（量级 <3%） | ✅ PASS |
| P84W2C-N08 | C3 | 绑扎数 2→6 单根最大张力单调不增 (R² > 0.99) | 1 | |值| = 1 vs 阈 0.99 | minimum-norm lashing solution | 无独立基准（阈判据） | ✅ PASS |
| P84W2C-N09 | C1 | ΣR_i = W 静力平衡残差 < 1e-6·W | 3.641e-15 | |值| = 3.641e-15 vs 阈 1.000e-06 | static equilibrium | 是（量级 <3%） | ✅ PASS |
| P84W2C-N09 | C2 | Winkler 与三弯矩法 M_max 相对差 < 1% | 0.00549853 | |值| = 0.00549853 vs 阈 0.01 | two independent models | 是（量级 <3%） | ✅ PASS |
| P84W2C-N09 | C4 | 全墩脱空退化到自由-自由梁解析弯矩，误差 < 1% | 1.000e-09 | |值| = 1.000e-09 vs 阈 0.01 | free-free beam analytic | 是（量级 <3%） | ✅ PASS |
| P84W2C-N10 | C1 | VHM→SF→λ 单位链双实现相对误差 < 1e-12 | 8.910e-16 | |值| = 8.910e-16 vs 阈 1.000e-12 | International Grain Code | 是（量级 <3%） | ✅ PASS |
| P84W2C-N10 | C2 | 左右舷镜像配载倾侧力矩相对差 < 1e-9 | 1.732e-16 | |值| = 1.732e-16 vs 阈 1.000e-09 | mirror symmetry | 是（量级 <3%） | ✅ PASS |
| P84W2C-N10 | C3 | 部分装载比例 0→60% 倾侧力矩单调增 (R² > 0.99) | 1 | |值| = 1 vs 阈 0.99 | Grain Code heeling moment | 无独立基准（阈判据） | ✅ PASS |

汇总：显式落在 3% 以内的关键量 0 项，显式超过 3% 的 0 项，其余 41 项因判据为区间/量级/布尔型或数字未交付而不适用 3% 门槛（逐行已注明原因）。本门不把「未交付」粉饰成「达标」：24 条判据以 WITHHELD 落表。

## 7. Gap 分析（未通过节点与阻塞项）

阻塞项 11 条，按节点归并如下；每条都给出可执行的补齐动作。

| 节点 | 严重度 | 缺口 | 补齐动作 |
|---|---|---|---|
| P84W2C-N01 | blocker | 裁决 FAIL；未落盘任何 JSON 产物 (artifacts/naca0012_aero/*.json, artifacts/naca0012_aero/*.png)；声明的 pytest 套件未执行 (mode=collect-only, declared=['tests/test_naca0012_aero.py'])；判据 C1 无数字证据；判据 C2 无数字证据；判据 C4 无数字证据；判据 C5 无数字证据；判据 C2b 无数字证据；判据 C3 无数字证据 | 按节点交付物 glob 落盘 case/results JSON（含判据数字字段） |
| P84W2C-N02 | blocker | 裁决 FAIL；未落盘任何 JSON 产物 (artifacts/channel_les/*.json, artifacts/channel_les/*.png)；pytest 未全绿: passed=15 failed=6 errors=0；判据 C1 无数字证据；判据 C2 无数字证据；判据 C4 无数字证据；判据 C6 无数字证据；判据 C7 无数字证据 | 按节点交付物 glob 落盘 case/results JSON（含判据数字字段） |
| P84W2C-N03 | blocker | 裁决 NOT_SUPPORTED；节点工作树缺失: /root/tp-agent/work/m8362 | 创建节点工作树（检出 base_ref）并按交付物清单落盘产物 |
| P84W2C-N04 | blocker | 裁决 FAIL；未落盘任何 JSON 产物 (artifacts/p84_w1/*.json)；pytest 未全绿: passed=0 failed=0 errors=1；判据 C1 无数字证据；判据 C2 无数字证据；判据 C3 无数字证据；判据 C5 无数字证据 | 按节点交付物 glob 落盘 case/results JSON（含判据数字字段） |
| P84W2C-N05 | blocker | 裁决 FAIL；未落盘任何 JSON 产物 (artifacts/hopper_discharge/*.json, artifacts/hopper_discharge/*.png)；声明的 pytest 套件未执行 (mode=collect-only, declared=['tests/test_hopper_discharge.py'])；判据 C1 无数字证据；判据 C2 无数字证据；判据 C3 无数字证据；判据 C5 无数字证据；判据 C6 无数字证据 | 按节点交付物 glob 落盘 case/results JSON（含判据数字字段） |
| CAMPAIGN-C1 | blocker | P84W2C-N01: pytest 状态 FAIL (mode=collect-only) | 补齐该缺口并把证据落盘到节点 worktree，然后重跑本门 |
| CAMPAIGN-C1 | blocker | P84W2C-N02: pytest 状态 FAIL (mode=executed) | 补齐该缺口并把证据落盘到节点 worktree，然后重跑本门 |
| CAMPAIGN-C1 | blocker | P84W2C-N03: pytest 记录为占位（mode='missing'），不计为可解析证据 | 补齐该缺口并把证据落盘到节点 worktree，然后重跑本门 |
| CAMPAIGN-C1 | blocker | P84W2C-N03: pytest 状态 NOT_SUPPORTED (mode=missing) | 补齐该缺口并把证据落盘到节点 worktree，然后重跑本门 |
| CAMPAIGN-C1 | blocker | P84W2C-N04: pytest 状态 FAIL (mode=executed) | 补齐该缺口并把证据落盘到节点 worktree，然后重跑本门 |
| CAMPAIGN-C1 | blocker | P84W2C-N05: pytest 状态 FAIL (mode=collect-only) | 补齐该缺口并把证据落盘到节点 worktree，然后重跑本门 |

## 8. 审计链与可复现性

- 裁决 JSON：`artifacts/p84w2c_gate/campaign_gate_report.json` · schema 指纹 `78929781ef79554de7acd16d91db1d19cea777412aba9da5a65b068c41c538e3`
- 证据目录：`artifacts/p84w2c_gate/node_evidence`（逐节点 pytest 归一化记录）
- 契约指纹：`aero-wind=615f7798609264f8`、`sediment-ukc=895befa5586a2a03`、`amp-eexi=c1499bae69f525a7`
- 平台证据：available=True · plan_nodes=11 · missions=10
- 本门实现文件 sha256：`__init__.py=e689359b2a1aa50a`；`__main__.py=d283d74eaefd4bbb`；`collect.py=5f161c0ceb1aa986`；`contracts.py=bd5486017efe939e`；`criteria.py=418ea103cd63e93b`；`figures.py=84d24766cb6c32d1`；`gate.py=f0555465fbee2dbd`；`lineage.py=690b59650333e4b7`；`report.py=8628b54572d5755a`；`schemas.py=71483b6654ee6995`；`spec.py=6b753054272ecbc8`

