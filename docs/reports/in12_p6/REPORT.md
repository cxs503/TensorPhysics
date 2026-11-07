# IN12-P6 · 战役编排契约与跨库对账账本成熟化（TensorPhysics）

**节点**：IN12-P6 / 纵贯军 P6 · 冰区战役薄编排（库优先教义：场景零求解器内核）
**分支**：`agent/m4934` · **库模块**：`src/tensorphysics/campaign/`
**审计状态**：`pass`（7 项机器判据 + pytest 全绿，详见 `artifacts/in12_p6/criterion_report.json`）

## 1. 案例介绍

冰区船舶总体性能战役把多个求解器库串在同一时间轴上：刚体/水动力主求解器给出冰载荷包络与
接触刚度（主步长 `dt_master = 0.004 s`），格子 Boltzmann / 显式接触子求解器因稳定性
必须使用更细的步长（`dt_sub = 0.0004 s`）。M12 处置要求**主/子时钟比率与交换节拍是显式契约字段**，
而不是隐式循环嵌套——否则耦合量的信息年龄（information age）在新库、重启、跨仓复算之间不可复现。

本节点交付的是**契约层**，不含求解器内核（库优先教义）：内核在 TensorLBM/TensorFVM/TensorFEM，
本层只固定「多库战役如何被机器判定为一致」。三条契约：

1. **耦合时钟**：整数比率 + 强制显式交换节拍 + 整数子步时间格（零漂移）。
2. **跨库对账账本**：质量 / 能量 / 冲量双账本（producer / consumer 两侧），残差判据为机器精度。
3. **几何契约**：`geom_pc6.json` 消费端注册（Cb 闭合）与 `failure_faces` v2 接口扩展（闭合曲面恒等式）。

几何与物理参数（全部来自 `artifacts/in12_p6/geom_pc6.json`）：

| 量 | 值 |
|---|---|
| 船级 | PC6（Polar Class 6 冰区加强） |
| Lpp / B / T | 120.0 m / 20.0 m / 7.5 m |
| 方形系数 Cb | 0.72 |
| 排水体积 ∇ | 12960.0 m³ |
| 排水量 Δ（ρ=1025 kg/m³） | 13284 t |
| failure_faces 接口 | v2 packet，4 面四面体接触控制体（船首肩部） |
| 设计水密度 | 1025.0 kg/m³ |

参考工况（解析已知输入，非求解器输出）：质量流率 12.3 kg/s、摩擦/挤压功率 4200 W、
接触力矢量 (1.2e5, 0, −3.4e4) N，在 0.2 s 水平上积分；二进制对照工况用 2 的幂步长与通量，
使两侧账本在浮点意义下**必须逐位相等**。

## 2. 方法与流程

### 2.1 耦合时钟（M12）

- 契约字段：`ratio`（整数，来自 `dt_master/dt_sub`，非整数比率直接拒绝）、`exchange_every`
  （**无默认值**，调用方必须显式给出）、`exchange_interval_s`、`sub_steps_per_exchange`、
  `max_information_age_s`。
- 时间基：`integer_sub_step`。交换时刻为整数 `master_tick * ratio`，物理时间只做**一次**乘法舍入，
  因此长战役零漂移；朴素 `t += dt` 累积的漂移在同一图（图 1 右）中对比。
- 消费端交叉校验：artifact 里既有的派生量（ratio/interval/sub_steps）会用原始量重算并比对，
  篡改 `exchange_every` 而不改派生量会被判为「internally inconsistent」。

本工况：ratio = 10，每 5 主步交换一次，交换间隔
0.02 s，每次交换 50 个子步，
信息年龄上界 = 0.02 s；0.2 s 内 10 次交换，
子步残差 0，时间残差 0.0e+00 s，
节拍残差 0.0e+00 s（判据：全部为 0）。

### 2.2 双账本对账（质量 / 能量 / 冲量）

- 每一侧（producer / consumer）按**交换窗口**记账；两侧窗口标签必须一致，否则对账直接失败。
- 生产者对窗口用 `math.fsum`（正确舍入），消费者用顺序累加（增量积分器的真实舍入模式），
  两侧总量因此只差求和舍入。
- 逐分量容差模型：`scale_i = max(|P_i|, |C_i|, Σ|p_ki|, Σ|c_ki|, tiny)`，
  `tol_i = 8.0 · eps · scale_i`；`scale` 取两侧条目 L1 和，作为抵消条件数代理——
  正负大额抵消的账本不能用总量的绝对量级来判定。
- 分级：`exact`（残差严格 0）/ `machine_precision`（落在窗口内）/ `out_of_tolerance`（账本不实）。

### 2.3 几何契约与接口扩展

- `geom_pc6.json` 经 schema（`additionalProperties: false`）校验后注册，注册即验证：
  Cb 闭合 `|Cb·L·B·T − ∇|/∇ ≤ 1e-12`，`L > B > T`，文件名与 `geometry_id` 一致。
- `failure_faces` 由 v1 扩展为 v2：保留 v1 四个必填字段，新增 centroid / failure_mode /
  contact_region / load_share / redundancy_group；v1 packet 被 v2 消费端接受，v2 可无损降级到 v1
  （文档化有损），版本窗口由契约里的 `consumer_contract` 显式协商（本实例 [1, 2]）。
- 闭合曲面恒等式（散度定理离散形式）`Σ A_i n_i = 0`：四面体接触控制体的面面积是无理数，
  因此闭合只能抵消到舍入底噪——这正是契约要认证的机器精度残差；删面或翻转法向立刻开裂（负控测试）。

### 2.4 契约 schema 双侧测试锁

6 个 artifact schema 各自指纹化（canonical JSON 的 sha256），指纹被**写死在测试里**
（`tests/test_campaign_contract_lock.py::PINNED_SCHEMA_FINGERPRINTS`）。生产者测试（对象→校验→JSON）
与消费者测试（JSON→校验→对象）引用同一组指纹，任一侧契约漂移都会同时打破两侧测试。
负控覆盖：多余字段、缺必填、NaN、类型错误、版本越界、未注册 schema 名。

### 2.5 薄编排自证

库代码 1791 行（非空非注释，含 docstring），编排驱动
`scripts/run_in12_p6_campaign.py` 102 行，占比
**5.70%**（判据 < 10%）；测试 556 行不计入编排。
驱动脚本内**零数值实现**：所有量来自库函数，脚本只做接线、落盘与审计链登记。

## 3. 计算结果

### 3.1 判据总表（机器判据，全部通过）

| 判据 | 内容 | 数值 | 结论 |
|---|---|---|---|
| C1 | 时钟比率/节拍显式 + 整数格残差 | ratio=10，节拍=5，子步/时间/节拍残差=0 | PASS |
| C2 | 已知输入对账残差 ≤ 8.0·eps | 最大相对残差 1.083e-15（窗口 1.776e-15） | PASS |
| C3 | 二进制工况逐位相等 | 最大相对残差 0.0（exact） | PASS |
| C4 | 同批判据 diff = 0 | 驱动 diff=0，消费端重算逐字节一致=True | PASS |
| C5 | 契约 schema 双侧锁 | 6 个 schema 指纹双侧一致 | PASS |
| C6 | 几何契约闭合 | Cb 残差 1.404e-16，曲面闭合 1.418e-17（阈值 1e-12） | PASS |
| C7 | 薄编排 < 10% | 5.70% | PASS |

### 3.2 双账本数字（decimal 工况，3 本账 / 5 个分量）

| 账本 | 侧 | 总量 | 单位 |
|---|---|---|---|
| mass | producer | 2.4600000000000004 | kg |
| mass | consumer | 2.460000000000003 | kg |
| energy | producer | 840.0000000000001 | J |
| energy | consumer | 840.0000000000007 | J |
| impulse x | producer / consumer | 24000.0 / 24000.0 | N·s |
| impulse z | producer / consumer | -6800.000000000001 / -6800.0000000000055 | N·s |

解析参考：m = 12.3 × 0.2 = 2.46 kg，E = 4200 × 0.2 = 840 J，Jx = 1.2e5 × 0.2 = 24000 N·s，
Jz = −3.4e4 × 0.2 = −6800 N·s。

### 3.3 漂移与几何

- 整数时间格：5000 子步后漂移严格 0.0 s；同工况朴素累加漂移
  1.610e-15 s（相对 8.05e-15，约 83 ulp）——图 1 右。
- `geom_pc6`：Cb 闭合残差 1.404e-16，Cb 由体积反算 = 0.72
  （声明值一致）；四面体接触控制体（4 面，总面积 62.65 m²，总承载
  8.8e+07 N）
  曲面闭合残差 1.418e-17。

## 4. 图

1. **图 1** `figures/fig01_clock_lattice_drift.png` —— M12 时间格：主/子刻度与交换时刻（左）；
   整数格零漂移 vs 朴素累加漂移随战役长度增长（右，对数坐标）。
2. **图 2** `figures/fig02_ledger_residuals.png` —— 双账本逐分量相对残差与 8 eps 容差线（左，
   绿色=machine_precision、蓝色=exact）；账本总量 vs 解析闭式的相对误差与 3% 验收线（右）。
3. **图 3** `figures/fig03_geometry_closure.png` —— Cb 闭合与曲面闭合残差（左，对比 1e-12 与 1 eps）；
   PC6 船首接触控制体面法向/面积/失效模式（右）。
4. **图 4** `figures/fig04_thin_orchestration.png` —— 库/测试/编排代码行数与 10% 判据对比（薄编排自证）。

## 5. 注意点、局限与适用范围

- **不是求解器**：本层只认证契约与对账，不产生物理场；因此不存在"精度对标实验"意义上的误差，
  全部判据是数学恒等式（闭合、守恒记账）与确定性重放。证据分级：**REAL（闭式恒等式验证）**，
  无 MOCK/STUB 成分——但也不覆盖真实求解器输出的数值误差。
- **恒定通量模板**：账本模板用常通量实例化（模板实例化要求已知输入）。时变通量的抵消风险由
  L1 `scale` 项覆盖并在测试中以定标扰动负控（1e-6 相对扰动 → out_of_tolerance）验证；
  尚未接入真实 LBM/FEM 交换数据。
- **非整数比率被拒绝**：目前只支持整数 ratio。真实多物理比率若是无理数（例如 CFL 自适应），
  需要引入有理逼近 + 显式守恒修正项，本节点未覆盖。
- **几何是理想化**：PC6 主尺度与排水量为代表性取值（非某实船线型），接触控制体为四面体等参理想化；
  `failure_faces` 只承载面积/法向/承载与失效模式，不含完整外板曲面或有限元网格。
- **闭合恒等式的适用范围**：`Σ A_i n_i = 0` 只对闭合控制体成立；开放补丁（例如仅首部外板）
  必须显式标注为开放并跳过该判据，当前版本把非闭合一律判为违约（偏严格）。
- **薄编排计量口径**：按非空非注释行统计（docstring 计入库），编排驱动 97 行 vs 库 1791 行 = 5.42%。
  若换用原始行数口径，比例会略有变化（库含大量注释/空行），结论不变。

## 6. 误差判定（vs 基准，逐项）

| 关键量 | 基准 | 本实现 | 相对误差 | < 3% ? |
|---|---|---|---|---|
| 质量账本总量 | 闭式 12.3×0.2=2.46 kg | 2.4600000000000004 / 2.460000000000003 kg | 1.805e-16 / 1.264e-15 | 是 |
| 能量账本总量 | 闭式 4200×0.2=840 J | 840.0000000000001 / 840.0000000000007 J | 1.353e-16 / 8.120e-16 | 是 |
| 冲量 Jx | 闭式 1.2e5×0.2=24000 N·s | 24000.0 / 24000.0 | 0.000e+00 / 0.000e+00 | 是 |
| 冲量 Jz | 闭式 −3.4e4×0.2=−6800 N·s | -6800.000000000001 / -6800.0000000000055 | 1.337e-16 / 8.025e-16 | 是 |
| 双账本残差 | 0（同物理量两侧） | 最大相对残差 1.083e-15 | 见左 | 是 |
| Cb 闭合 | ∇ = Cb·L·B·T | 残差 1.404e-16 | 同上 | 是 |
| 曲面闭合 | Σ A_i n_i = 0 | 残差 1.418e-17 | 同上 | 是 |
| 时钟漂移 | 零漂移（整数格） | 0.0 s（朴素累加 1.610e-15 s） | 0 / 8.049e-15 | 是 |

**逐项结论：所有关键量的相对误差均 < 3%（实测处于 1e-16~1e-17 量级，即浮点舍入底噪），判据为机器精度级。**
未用实测/文献基准的量（如 failure-face 承载能力）属代表性输入而非预测结果，故不作误差对标，
在此显式声明原因。

## 7. 审计链与复现

```bash
python3 scripts/run_in12_p6_campaign.py     # 生成 artifacts/in12_p6/*.json（含 canonical digest）
python3 scripts/in12_p6_audit.py            # 消费端独立重算 + 判据报告 + pytest 证据
python3 scripts/in12_p6_report.py           # 本报告与图
python3 -m pytest tests/test_campaign_*.py -q
```

产物 digest（canonical JSON 的 sha256）：

| artifact | sha256 (canonical JSON) |
|---|---|
| `artifacts/in12_p6/clock_binary_clock.json` | `ab48d66c7bcfe7fdbb1f8d6b80b788d0c4cf359dd421a88daeb875ed3105bb70` |
| `artifacts/in12_p6/clock_decimal_clock.json` | `134f65174cdf997e2157f49d593383c006b54eb35c37308411865294b000f6c6` |
| `artifacts/in12_p6/failure_faces_pc6.json` | `55eeded39939e4d085654ab935560ef8329c87fc02a253c70dde9a5edca08f9f` |
| `artifacts/in12_p6/geom_pc6.json` | `2befb83f79f5779e906693b54a9e498cf5da741076ea53e740104472b656d599` |
| `artifacts/in12_p6/ledger_binary_clock_energy_consumer.json` | `0c74d766f006b555e3534a9742bc9c808b8df76a5029c10b12855ce97cf6353a` |
| `artifacts/in12_p6/ledger_binary_clock_energy_producer.json` | `656bbf8582d647bf1c82e5a1a276595ad79ea59860affe838130f2b56d657600` |
| `artifacts/in12_p6/ledger_binary_clock_impulse_consumer.json` | `fe78da9b5c4afab454d4e9a02d3a487197efa19760c4d18d019f0f13e4625114` |
| `artifacts/in12_p6/ledger_binary_clock_impulse_producer.json` | `904d71448f3ae36df7ca67a0f3c59e5e60a5ca1c90844b7cf5fcb773569bcd26` |
| `artifacts/in12_p6/ledger_binary_clock_mass_consumer.json` | `b0ededc0a9ca54ad4f7815c7ac19701e2ea15ccf5a91f82d18a9bb764aad1ccc` |
| `artifacts/in12_p6/ledger_binary_clock_mass_producer.json` | `ae4b1f0e95b9b76c04115bcad8174389aebca8b72631c23bb8ae0a36dc6fccf4` |
| `artifacts/in12_p6/ledger_decimal_clock_energy_consumer.json` | `1a29f1e5ef528e1f8c34c6f005e9871f69e2cd112227b42332152f43553bc98e` |
| `artifacts/in12_p6/ledger_decimal_clock_energy_producer.json` | `bb92034b7a913c9fec3882b68747dffad5c0a124823d946c628dc8a6d5d6f6f5` |
| `artifacts/in12_p6/ledger_decimal_clock_impulse_consumer.json` | `e04d7a946de24261b2db0ee38c97a7a9354aa3e0425f574f3414ade7fee2c707` |
| `artifacts/in12_p6/ledger_decimal_clock_impulse_producer.json` | `6501253ea7104f445a5252890fb27d21d801fad812424a51e42ac656943b7912` |
| `artifacts/in12_p6/ledger_decimal_clock_mass_consumer.json` | `1844af11e953d8bddabfe8c07fd24460cf469f7ae34523cbf8939c080ad96aac` |
| `artifacts/in12_p6/ledger_decimal_clock_mass_producer.json` | `4b88d09237d774ba52cb374fd38a60d7caf5c0e6e7b885b062dafab6dc114e11` |
| `artifacts/in12_p6/ledger_reconciliation.json` | `5d291d9c63dec96016e4d70bf6ea175267ca4925905f0d8ce6ef6cc116c4fd3d` |
| `artifacts/in12_p6/ledger_reconciliation_binary.json` | `94c2d869ddfa53e118137f87b2123125e909c6913a63de36e7ebbd4008310e1d` |

契约 schema 指纹：6 个，记录于 `artifacts/in12_p6/schema_fingerprints.json`
并被 `tests/test_campaign_contract_lock.py` 双侧锁定。
pytest 证据：`56 passed, 1 warning`（returncode=0）。
