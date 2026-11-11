# m15005 · 交差审计：GAP-CROSS-CLOCK（m14792）跨库时钟语义交付

**审计单** m15005 · **被审单** m14792（TensorPhysics，succeeded，PR #393 merged，报告 id=13896）·
**锚点** GAP-CROSS-CLOCK（第 2 轮；第 1 轮 m14677 honest-FAIL）· **仓** TensorPhysics ·
**判级** **L2**（方案累积式；首个未通过 rung = L3）·
**取证工件** `runs/m15005_audit_cross_clock/audit_checks.json`（本报告全部数字的程序化来源）·
**复核脚本** `scripts/audit_m15005_cross_clock.py`、`scripts/make_m15005_audit_figure.py`

> 结论一句话：m14792 的**数字全部为真**（18 测试重跑全绿、三个时钟比下的账本/动量/ZOH 误差逐位复现），
> 但交付物**没有任何主求解器/平台侧调用方**，按平台累积式达标方案只能判 **L2**
> ——L4 级的验证强度与 L3 级的接线缺失同时成立，判级以累积序为准，不以验证强度为准。

---

## 1. 案例介绍（被审对象是什么）

船舶/海洋工程里 LBM 流体与 DEM 颗粒的**时钟不同步**耦合：流体大步长（h = 1/5/10 ms）、颗粒小步长
（dt_dem = 50 µs），两腿在同步点上交换冲量。跨库耦合的三类漂移源是 **时钟**（`+= dt` 累加、同步点不重合）、
**单位**（lattice 与 SI 混用、动量与力的量纲指数不一致）、**交换**（冲量丢/重、把积分量当点值插值）。
m14792 把这三点写成了实现级冻结契约：契约模块 + 两库（tensorlbm 0.3.0 / tensordem 0.1.0）最小耦合 demo
+ 版本冻结 pin。

几何/参数（取自被审工件 `results.json`，程序化注入）：

| 参数 | 值 |
| --- | --- |
| `dx_m` | 0.0001 |
| `tau` | 0.8 |
| `nu_si_m2_s` | 1e-06 |
| `rho0_kg_m3` | 1000 |
| `dt_lbm_s` | 0.001 |
| `dt_dem_s` | 5e-05 |
| `nx` | 64 |
| `ny` | 8 |
| `u_fluid0_m_s` | 0.001 |
| `particle_radius_m` | 0.0021 |
| `particle_density_kg_m3` | 1000 |
| `mu_pa_s` | 0.001 |
| `t_final_s` | 0.2 |

库锚（腿级，第三方闭式解）：tensorlbm D2Q9 Taylor-Green 衰减率 vs 解析 `2νk²`；tensordem Stokes 曳力松弛 vs
解析 `u + (v0-u)e^{-t/τp}`。

## 2. 怎么做的（审计方法与独立性限制）

审计设计：**不读结论、只重算**。`scripts/audit_m15005_cross_clock.py` 依次执行 7 项机器可判检查：

| 检查 | 判据（机器） |
| --- | --- |
| A 工件完整性 | 重算 `runs/m14792_cross_clock_library/` 三件工件的 SHA-256，与自带 `SHA256SUMS` 比对 |
| B 证据指针 | impl/entry/tests/contract 四文件存在；impl 主体 >40 行且无 `NotImplementedError`/占位 `pass` |
| C 冻结 pin | `verify_pins() == []`：耦合的两条腿是**已发布库**（版本 + 源码 sha256 + API 签名），不是仓内替身 |
| D 测试证据 | 就地重跑 `pytest tests/test_cross_clock_library_binding.py`，从报告里解析 passed/failed（不抄记录） |
| E 可复现 | 在**一次性副本**里执行 repro_cmd，逐字段比对物理量与已提交 `results.json`（仅容忍 timestamp/wall_seconds） |
| F 集成调用方 | 全仓 grep 该模块的外部调用方（排除模块自身、自身测试、demo 入口） |
| G 判级 | 按平台累积式达标方案 L1→L5 逐级核，首级未过即封顶 |

复现命令（本次实际执行）：

```
python scripts/audit_m15005_cross_clock.py          # A–G 全量，~30 s
python -m pytest tests/test_cross_clock_library_binding.py -q   # 18 passed
python scripts/run_cross_clock_library_demo.py      # repro_cmd（在临时副本里执行）
```

**独立性限制（如实）**：①我复核的是已合并工件，不是重新推导耦合数学——除 ZOH 闭式递推与两条腿的解析解外，
没有独立第三方求解器参与；②两条腿来自**别的任务工作树的 editable 安装**
（`tensorlbm` → `/root/tp-agent/work/m13963/src`，`tensordem` → `/root/tp-agent/work/m14461/src`），
pin 校验用源码 sha256 兜住这一点，但环境本身是脆的（见 §5 F5）；③我未复核 m14792 写进平台的
`oss_solutions`（已抽查：`id=376, repo=TensorPhysics, feature_slug=GAP-CROSS-CLOCK`，存在）。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 工件完整性（A）

| 工件 | sha256(前 12) | bytes | SHA256SUMS 核对 |
| --- | --- | --- | --- |
| `reference_drift_vs_window.png` | `6ff68b234212` | 57148 | OK |
| `results.json` | `4f882ebc7d1b` | 51377 | OK |
| `velocity_vs_time.png` | `688a9a7275b6` | 62667 | OK |

### 3.2 证据指针（B）

| 角色 | 路径 | 行数 | bytes | sha256(前 16) | placeholder |
| --- | --- | --- | --- | --- | --- |
| impl_files | `src/tensorphysics/cross_clock/library_binding.py` | 774 | 32882 | `471ed5ed1072e078` | 0 |
| entry | `scripts/run_cross_clock_library_demo.py` | 355 | 14136 | `bb8d5ae96ffe5c29` | 0 |
| tests | `tests/test_cross_clock_library_binding.py` | 255 | 10003 | `c995694fc760253d` | 0 |
| contract_doc | `docs/CROSS_CLOCK_LIBRARY_CONTRACT.md` | 112 | 6289 | `4f22067e99a9c85b` | 0 |

### 3.3 冻结 pin（C）

| 库 | 版本 | 源码 sha256(前 16) | drift |
| --- | --- | --- | --- |
| tensorlbm | 0.3.0 | `9fe8ab2ca05a271a` | 0 |
| tensordem | 0.1.0 | `28c3bb41c68396fb` | 0 |

### 3.4 测试重跑（D）

`18 passed, 0 failed`（returncode 0），
与 m14792 记录声称的 18 passed **一致**。

### 3.5 可复现（E）

repro_cmd 在临时副本执行 14.58 s；与已提交 `results.json` 的**物理量差异条目数 = 0**
（仅 `wall_seconds`/`generated_utc` 变化）。已提交工件 sha256 `4f882ebc7d1bde18`，
复现值 `1ab167ad8e69abcb`（整文件哈希因 `wall_seconds` 不同而不同，物理字段逐位相同）。

### 3.6 被审算例的数字（直接读取被审工件，含交叉核对）

| case | h [s] | r (lbm:dem) | windows | ledger drift | 系统动量残差 | stagger 残差 | vs 精确 ZOH | 参考漂移 | clock round-trip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fine | 1e-03 | 1:20 | 200 | 0.00e+00 | 1.05e-11 | 7.73e-06 | 1.85e-05 | reference | yes |
| medium | 5e-03 | 5:100 | 40 | 7.22e-18 | 1.04e-11 | 3.86e-05 | 4.63e-05 | 1.55e-05 | yes |
| coarse | 1e-02 | 10:200 | 20 | 2.17e-17 | 1.04e-11 | 7.69e-05 | 8.12e-05 | 3.47e-05 | yes |

时钟 round-trip 明细（同一工件内）：三例 `fluid_steps = 200`、
`particle_steps = 4000`（因 `dt_lbm` 固定 1 ms、只有耦合窗口 h 变化），
`sync_coincidence_error_s` 最大 5.55e-17 s
——与 m14792 记录声称的 "≤5.6e-17 s" 复算一致。

### 3.7 基准对照 ref vs got（L4 判据）

| 量 | reference | got | 相对误差 | <3%? |
| --- | --- | --- | --- | --- |
| tensorlbm Taylor-Green decay rate (measured/analytic 2*nu*k^2) | 1 | 1.00384 | 3.84e-03 | YES |
| tensordem Stokes relaxation v at t=0.2 s [m/s] | 0.000184604 | 0.000184606 | 1.15e-05 | YES |
| coupled particle velocity vs exact ZOH, h=1e-03 s | exact ZOH recursion from measured u_k | 1.84669e-05 | 1.85e-05 | YES |
| momentum bookkeeping residual, h=1e-03 s | 0 | 1.0469e-11 | 1.05e-11 | YES |
| coupled particle velocity vs exact ZOH, h=5e-03 s | exact ZOH recursion from measured u_k | 4.63447e-05 | 4.63e-05 | YES |
| momentum bookkeeping residual, h=5e-03 s | 0 | 1.04209e-11 | 1.04e-11 | YES |
| coupled particle velocity vs exact ZOH, h=1e-02 s | exact ZOH recursion from measured u_k | 8.11946e-05 | 8.12e-05 | YES |
| momentum bookkeeping residual, h=1e-02 s | 0 | 1.03565e-11 | 1.04e-11 | YES |

## 4. 结果图

![audit_evidence](fig:audit_evidence)

左：本次审计重新导出的 8 个误差量（对数轴），全部远低于 3% 验收线，最大 3.84e-3（Taylor-Green 衰减率，
即库锚本身，不是耦合误差）。右：累积式达标 rung 状态——L1/L2/L4 通过，**L3 集成失败**，L5 无外部锚。

![reference_drift_vs_window](fig:reference_drift_vs_window)

（被审工件自带图，未重绘）耦合窗口 h 增大时参考漂移与 stagger 残差同步上升，而账本残差停在舍入级
（1e-17）——说明"账本对上"与"物理对上"是两件事，两者被分别计量，这点 m14792 做对了。

## 5. 注意点（审计发现，按严重度排序）

**F1【阻断 L3｜严重度高】没有任何求解器/平台侧调用方。**
证据：F 检查在全仓 grep `library_binding` 的外部引用 = **空**
（模块自身、其测试、demo 入口之外为零）。含义：耦合器当前只能被 demo 脚本驱动，任何"下一次跨时钟耦合需求"
都要重新写一遍驱动。**具体返修路径**：`src/tensorphysics/coupling.py` 的 `CouplingManager.register(phase, callback,
*, solver_name, priority)`（该文件第 ~54 行）就是现成的接线点，把 `CoupledClockRun` 包成 `AFTER_SOLVER` 回调即可
在平台耦合相位里被调用；这也是把它从 L2 推到 L3 的最小改动。

**F2【平台侧｜严重度中】合并门把 run 工件滤掉了，main 上不存在被审工件。**
证据：m14792 合并 commit `d8f92af "m14792: matured mission deliverables（4 个文件被门禁滤除）"`；
`on_disk_changes` 列了 4 个 `runs/` 文件，而 TensorPhysics main 的 `runs/` 里没有它们（本次审计因此从工作树
m14792 把工件搬到本审计树并**用 SHA256SUMS 重新验过**）。后果：任何在合并态 main 上做的 L2+ 机器复核都会因
"工件缺失"降级——这是审计链的结构性漏点，建议交付门对 `runs/*/results.json`、`SHA256SUMS`、`figures/*.png`
开白名单（证据类工件不是构建产物）。

**F3【记录卫生｜严重度低】被审单完成时没带 maturity evidence。**
证据：m14792 记录 `server_quality_audit = {gate_valid: false, key_present: false, gate_missing_warning: true}`。
即"验收三件套"是写在 goal 里、落在报告正文里，而不是落在 evidence 键上——所以才需要本交差单补 verdict。

**F4【科学面｜严重度低】耦合算例的参照半解析、无外部锚，L5 不可宣称。**
证据：`v_vs_zoh_rel_error` 的参考解是**同一次运行测得的 u_k** 代入解析 ZOH 递推——它独立于 DEM 积分器
（能定量分离出积分/交换误差，这是它的价值），但不独立于流体腿；空间分辨率固定 64×8、dx = 1e-4 m，无网格加密研究。
因此"耦合轨迹"没有跨程序（preCICE 等）或实验对照 → L5 的 `external_anchor` 键只能指向两条腿的解析解，不足以升 L5。

**F5【复现健壮性｜严重度低】pin 是好机制，但 demo 不 fail-closed。**
证据：`library_binding.verify_pins()` 返回 drift 列表，demo 把 drift **记进** `results.json` 的 `contract_pin_drift`
而不中止；真正断言 drift 为空的是测试。建议 demo 在 drift 非空时 `raise SystemExit(2)`，让"漂移发生"不可能产出
看起来正常的工件。

**F6【无夸大｜信息】被审单的声明逐条为真。**
18 passed、ledger drift 0 / 7.22e-18 / 2.17e-17、
系统动量残差 ~1.05e-11、库锚 0.00384 与
1.15e-05、`verify_pins()==[]`——全部复现。且被审单**主动声明未做**的
（immersed boundary 空间分布 / 壁面 / 多粒子 / Schiller-Naumann 路径）在代码里确实不存在，属诚实范围声明。
未发现过度宣称。

## 6. 误差判定（逐项回答"是否 <3%"）

| 量 | reference | got | 相对误差 | <3%? |
| --- | --- | --- | --- | --- |
| tensorlbm Taylor-Green decay rate (measured/analytic 2*nu*k^2) | 1 | 1.00384 | 3.84e-03 | YES |
| tensordem Stokes relaxation v at t=0.2 s [m/s] | 0.000184604 | 0.000184606 | 1.15e-05 | YES |
| coupled particle velocity vs exact ZOH, h=1e-03 s | exact ZOH recursion from measured u_k | 1.84669e-05 | 1.85e-05 | YES |
| momentum bookkeeping residual, h=1e-03 s | 0 | 1.0469e-11 | 1.05e-11 | YES |
| coupled particle velocity vs exact ZOH, h=5e-03 s | exact ZOH recursion from measured u_k | 4.63447e-05 | 4.63e-05 | YES |
| momentum bookkeeping residual, h=5e-03 s | 0 | 1.04209e-11 | 1.04e-11 | YES |
| coupled particle velocity vs exact ZOH, h=1e-02 s | exact ZOH recursion from measured u_k | 8.11946e-05 | 8.12e-05 | YES |
| momentum bookkeeping residual, h=1e-02 s | 0 | 1.03565e-11 | 1.04e-11 | YES |

- **是否 <3%：是，全部项目 <3%**，最大项 3.84e-3（0.384%，Taylor-Green 衰减率，属库锚/单位换算校验），
  耦合相关量 1.8e-5 … 9.0e-5，账本与动量残差 1e-17 / 1e-11 量级。
- 必须说清的边界：这些是**对解析解 / 精确 ZOH 递推的数值一致性误差**，不是"对实验的误差"。
  跨时钟耦合轨迹**无实验/文献基准**，所以本单不对"耦合模型是否物理正确"下 3% 结论——那需要
  独立程序对照或实验（见 F4）。
- 非数值判据同样成立：时钟 round-trip 三例全 True（步数 = n_windows × r，两腿时间与调度时间差 <1e-15 s）。

## 7. 判级与落表

| rung | 判据 | 机器复核 |
| --- | --- | --- |
| L1 | 数学/单元验证（tests） | PASS |
| L2 | 独立功能实现（impl+entry+run_artifacts） | PASS |
| L3 | 系统集成（主求解器/平台调用方） | FAIL |
| L4 | 基准验证（ref vs got + repro_cmd） | PASS |
| L5 | 科研工程（外部锚：文献/实验） | FAIL |

**判级 = L2（L0–L5 制）**：功能实质为"真代码 + 真库 + 真跑数 + 逐位可复现"，
但**未接入主求解器/平台**，按累积序卡在 L3 之前。附报告长度不改变级别。
machine gate 预期：本 verdict 的 `integration_callers` 为空 → gate 逐级复核将得 gate_level = L2
（≥2）→ `confirmed`（消账），返修建议（F1 接线）可作为后续独立提案/任务，不必重开 m14792。

**处置（disposition）：approve with notes。** 无阻断科学正确性的缺陷；阻断 L3 的是**接线缺失**，已给出可执行返修点。

---

*工件：`runs/m15005_audit_cross_clock/{audit_checks.json, audit_summary.txt, figures/audit_evidence.png}`；
被审工件副本：`runs/m14792_cross_clock_library/`（SHA256SUMS 已复验）。报告生成时间 2026-11-11T14:31:40Z。*
