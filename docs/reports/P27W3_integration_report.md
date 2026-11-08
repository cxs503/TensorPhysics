# P27W3 集成验收报告：跨仓接口契约收敛 + 成熟化汇总

**节点**：P27W3-N08（mission 6734，仓 TensorPhysics，campaign plan 165 = `P27-W3-seed`）  
**生成时刻**：2026-11-08T08:27:10Z　**代码基线**：`d2251a82d68a`  
**判定文件**：`artifacts/p27w3_contracts/P27W3_verdict.json`（本报告全部数字由该 JSON 程序化注入，零手抄）  
**判据总览**：4 pass / 3 fail / 0 withheld

## 1 案例介绍

本节点是 P27W3 战役 DAG 的收敛节点（N08），本身不产生新的物理能力，而是回答三个工程问题：**（a）**上游戏 7 个节点的状态是否已收敛；**（b）**TensorPhysics 与两个邻居仓（ship-performance-platform、TensorDEM）的接口契约是否真正一致——同一船型、同一谱、同一强度语义经过仓边界后数值是否还原；**（c）**成熟化单（evidence_level 升级）是否落账。为此定义了三条版本化契约，每条都有明确的单位/归一化登记表和 fail-closed 校验：

| 契约 | 发送方（对端仓） | 载荷 | 验收量 | 判据 |
|---|---|---|---|---|
| `rao-v1` | ship-performance-platform `SeakeepingPhysics` STF 切片理论 | 规则波运动传递函数（ω 轴，rad/s）+ 船型标识 + 归一化登记表 | 接收端仅凭载荷重建 3 个频率点的垂荡/纵摇/横摇幅值 | <1% |
| `stress-psd-v1` | ship-performance-platform `backend.modules.spectral_fatigue` | 单边热点应力 PSD（MPa²/Hz）+ S-N 律（S^m N = C, MPa）+ 时长 | 接收端独立重算谱矩/Dirlik 损伤 | <1% |
| `ice-bond-v1` | TensorDEM `IceBeamSpec` 晶格键模型 | 试件几何 + 键强度/键刚度/断裂应变 + 晶格常数 | 映射为 TensorPhysics 内聚律 (τ_p, G_c) 并反向还原 σ_b、能量积分、首纤维破坏载荷 | <2% |

标准算例（两侧共用同一组数字）：

* **Wigley 船型**：L=100 m，B=10 m，T=5 m，Cb=Cwp=2/3，∇=3333.33 m³，Δ=3416.7 t（ρ=1025 kg/m³），GM=1.0 m，顶浪 β=180°，航速 0/5 kn。
* **应力谱**：双峰高斯（f₁=0.12 Hz σ₁=35 MPa；f₂=0.85 Hz σ₂=9 MPa），m₀=154.59 MPa²，RMS=12.43 MPa，带宽 α₁=0.653（宽带），S-N 单斜率 m=3、C=1e12（MPa 单位），时长 3600 s。
* **冰梁试件**：L=2.0 m，b=0.5 m，h=0.05 m，晶格 d=12.5 mm，E=5.0 GPa，σ_b=1.5 MPa，ρ=917 kg/m³。

对端证据（golden corpus 由 `scripts/p27w3_capture_peer_goldens.py` 用 `git show <sha>:<path>` 物化源码后现场计算，不写对端仓）：

| 契约 | 对端仓 | 提交 | 提交说明 | 源文件 sha256（前 16 位） |
|---|---|---|---|---|
| `rao-v1` | ship-performance-platform | `6d8ec823cb` | Merge #16 (m6621) via push-merge | `5ed53b3d30855ff6` |
| `stress-psd-v1` | ship-performance-platform | `8bbb6fc827` | P27W3-N04 deliverable (mission worktree, | `3c8cd73614c5fbc8` |
| `ice-bond-v1` | TensorDEM | `e6989a29d9` | Merge #507 (m6617) via merge-master | `7d9dbf258c216d2e` |

## 1.1 工作包分解与依赖（父目标、接口、验收）

**父目标**（campaign plan 165 / mission 6734）：P27W3 全流程收敛——三条跨仓接口契约机器可裁、成熟化单落账、战役级报告可复现。

| 工作包 | 交付物 | 验收测试 | 依赖 | 写权属 |
|---|---|---|---|---|
| WP1 契约定义（`integration/interchange.py`） | 三条版本化 schema + 单位登记表 + fail-closed 校验 | 拒收负控 5 条（缺归一化/非单调轴/负 PSD/零能量/缺 provenance） | 无（先定接口再并行） | 本节点 |
| WP2 三通道验证（`rao.py`/`spectral.py`/`ice_bond.py`） | 传输保真 + 模型无关不变量 + 抽样充分性 | `tests/test_p27w3_integration_contracts.py` 中 15 条 | WP1 的 schema 冻结 | 本节点 |
| WP3 DAG 与成熟化审计（`dag_audit.py`/`p27w3_gate.py`） | 逐条判据 JSON（数字+阈值+证据路径） | gate 结构测试 + 幂等重跑 | 只读平台表；N01–N07 的上游结果 | 本节点（只读上游） |
| WP4 报告与图（`scripts/p27w3_report.py`） | `P27W3_integration_report.md` + 5 图 + 平台报告 | C6 字数 + 数字全部由 JSON 注入 | WP2/WP3 的 JSON | 本节点 |

**依赖图**：WP1 → {WP2, WP3 并行} → WP4；对端仓（ship-platform、TensorDEM）只作为**只读输入**，不参与写。**集成点**：golden corpus（载荷+响应+来源哈希）是唯一跨仓边界，因此对端仓不可达时验收仍可离线复现。**冲突风险**：上游节点（N01–N07）与本节点写同一`campaign_nodes` 行集合但不同行，本节点只读；报告与判定文件只由本节点写，无共享可变文件。

**风险登记（gate 时刻）**：①上游 5/7 未 done → C1 不可裁决（已如实 FAIL，机制化可重跑）；②N02 成熟化单未产出 → C5 FAIL；③对端 N04 闭合式缺陷 → C3 FAIL（已给返修式与复验数字）；④对端 API 在验收窗口不可达 → 改走源码物化（哈希可复核）；⑤浮体剖面 BEM 缺混合边界 → 相关量降级为诊断。**完成定义**：C1–C7 全部 PASS 且报告数字可一键重算；当前 4/7 PASS，其余三项均为**上游/对端**原因，非本节点可自修。

## 2 方法

**（1）采集—回放两段式。** 契约验证不能依赖对端服务在线（本轮实测 ship-platform API 在验收窗口内不可达）。因此分两段：采集段在能读到对端代码时物化源码、现场求解、把载荷/响应/来源哈希写成 golden corpus；回放段（pytest 与 gate）只读 golden，完全不 import 对端仓，保证验收可离线复现。

**（2）fail-closed 载荷校验。** 每条契约只接受登记表里声明过的单位串：垂荡 `m/m`、纵摇/横摇 `rad/(k a)`、频率 `rad/s`、PSD `MPa^2/Hz`、S-N `S^m N = C, S in MPa`。缺项、错项、非单调频率轴、负 PSD、零能量谱、缺 provenance 一律抛 `ContractViolation`，不做“猜测式兼容”。

**（3）三类检查分工。**

* **传输/约定保真（CONTRACT）**：接收端只用载荷声明的归一化重建物理幅值；任何单位漂移（rad/m vs rad/(ka)、deg vs rad、ω vs ω_e）会放大成 O(1) 误差。这是 C2/C3/C4 的机器判据。
* **模型无关不变量（REAL）**：长波极限 kL→0 时垂荡 RAO→1（Newman 1977 §6.10）、纵摇 RAO→1（波陡归一化）、短波衰减、顶浪横摇为零、深水色散 k=ω²/g、遭遇频率 ω_e=ω−kU·cosβ。这些对任何线性势流解都成立，因此可以对“外来”载荷断言。
* **抽样充分性（REAL）**：把传递函数当作表格来插值（PCHIP）会丢信息。用 JONSWAP 海况下的显著幅值 σ=√(∫|RAO|²S_ζ dω) 作为误差度量，比较对端默认 40 点表与本契约建议的加密表（158 点，含共振带 0.72–1.36 rad/s 加密）。选谱域而非逐点误差，是因为垂荡 RAO 在 ω≈1.45 rad/s 有过零点，逐点相对误差在节点附近无物理意义。

**（4）跨实现独立见证。** Dirlik 是经验闭合，两侧实现同一公式当然“一致”，但那证明的是接口而非法理。因此另跑一条独立路径：随机相位合成时域记录→ASTM E1049 雨流计数→Palmgren-Miner，对同一 PSD 给出第三种损伤估计。

**（5）DAG 与成熟化读数。** 直接以只读 `psql` 读平台表 `campaign_nodes`（plan 165）与 `factory_evidence_records`，不可达即判 WITHHELD——绝不把“读不到”写成“通过”。

## 3 计算结果

### 3.1 判据总表（数字取自 verdict JSON）

| 判据 | 内容 | 判据值 | 阈值 | 状态 |
|---|---|---|---|---|
| C1 | 7 个前置节点 status=done（plan 165） | done=2/7 | 7/7 | **FAIL** |
| C2 | BEM↔耐波 RAO 传递偏差（3 频点） | 0 | <0.01 | **PASS** |
| C3 | FEM↔疲劳 应力谱传递偏差（Dirlik 损伤） | 0.1964 | <0.01 | **FAIL** |
| C4 | DEM↔冰板 键强度映射偏差 | 1.819e-16 | <0.02 | **PASS** |
| C5 | 成熟化单 evidence_level 升级记录 | N02 0/3 字段，N04 16/5 字段 | 完整 | **FAIL** |
| C6 | 本报告字数 | 12577 字符 | ≥3000 | **PASS** |
| C7 | 集成契约 pytest | 23 条 | ≥10 | **PASS** |

### 3.2 C2：`rao-v1` 三频点传输还原

| ω [rad/s] | 垂荡 接收/发送 [m] | 相对偏差 | 纵摇 接收/发送 [°] | 相对偏差 |
|---|---|---|---|---|
| 0.60 | 0.913039 / 0.913039 | 0 | 2.536852 / 2.536852 | 0 |
| 1.00 | 1.638250 / 1.638250 | 0 | 1.291127 / 1.291127 | 0 |
| 1.60 | 0.001301 / 0.001301 | 0 | 0.104497 / 0.104497 | 0 |

长波不变量：ω=0.020 rad/s 时 kL=0.00408，垂荡 RAO=1.000407，与解析极限 1 的偏差 4.074e-04。

| 不变量 | 值 | 参考 | 相对偏差 | 判定 |
|---|---|---|---|---|
| long_wave_heave_rao_to_unity | 1.0004 | 1.0000 | 4.074e-04 | PASS |
| long_wave_pitch_rao_to_unity | 1.0004 | 1.0000 | 4.406e-04 | PASS |
| short_wave_heave_rao_decay | 0.0032 | 0 | 0.0032 | PASS |
| head_sea_roll_symmetry | 1.418e-16 | 0 | 1.418e-16 | PASS |
| encounter_frequency_convention | 0 | 0 | 0 | PASS |

### 3.3 C2 附加诊断：表格抽样充分性（基线 vs 契约）

| 表 | 点数 | 垂荡 σ 误差 | 纵摇 σ 误差 | 逐点最大相对误差 |
|---|---|---|---|---|
| 对端默认表（基线） | 44 | 0.437% | 0.867% | 1144.1% |
| 本契约建议表 | 158 | 0.092% | 0.006% | 37.4% |

即：谱域误差从 0.44%（垂荡）/0.87%（纵摇）降到 0.092% / 0.006%，分别改善约 5× 与 155×。

### 3.4 C3：`stress-psd-v1` 谱疲劳传递（含对端闭合式审计）

| 估计器 | 3600 s 损伤 | 相对雨流基准 | 说明 |
|---|---|---|---|
| 雨流计数（时域独立基准） | 4.261e-05 | 0（统计误差 0.56%） | 32768 点 @8 Hz，2 个随机相位实现 |
| 接收方 Dirlik（TensorPhysics） | 3.975e-05 | 6.72% | 闭式谱矩，Dirlik(1985) 原文系数 |
| 发送方 Dirlik（N04 交付件） | 3.322e-05 | 22.04% | 数值积分其 `dirlik_pdf` |
| Rayleigh 窄带 | 6.616e-05 | 55.27% | 宽带谱应显著高估 |

**传输偏差（C3 判据）= 0.1964（19.64%）→ FAIL。** 归因（机器生成）：closure bias, not transport: receiver Dirlik is 6.72% from the rainflow estimate, the peer is 22.04% from it (receiver/peer agreement 19.64%). The peer's dirlik_parameters() does not implement Dirlik (1985): it uses x = alpha1 instead of xm = (m1/m0)*sqrt(m2/m4) and sets R = alpha1, yielding D1 = 2x^2(1-x)/(1+x^2) instead of 2(xm - alpha2^2)/(1 + alpha2^2).

对端契约本身是对的（谱矩、循环率约定、S-N 单位全部一致），偏差 100% 来自**闭合式实现**。把两侧系数并排放出来即可判定谁对：

| 系数 | Dirlik(1985) 原文（接收方） | 对端 N04 `dirlik_parameters()` |
|---|---|---|
| D1 | 0.177381 | 0.207414 |
| D2 | 0.596806 | 0.430903 |
| D3 | 0.225813 | 0.361684 |
| Q | 0.221726 | 0.209682 |
| R | 0.226752 | 0.653327 |

原文系数为 xm=(m1/m0)√(m2/m4)、D1=2(xm−α2²)/(1+α2²)、R=(α2−xm−D1²)/(1−α2−D1+D1²)、D2=(1−α2−D1+D1²)/(1−R)、D3=1−D1−D2、Q=1.25(α2−D3−D2R)/D1。对端实现把 x 取成 α1、并令 R=α1，**完全丢掉了含 m1 的 xm**，因此其 D1/D2/D3/Q/R 与原文不符，3 阶矩 E[S³] 偏低 16.4%，损伤偏低 22.0%（相对雨流）。这不是接口问题，是对端交付件的物理缺陷——已按战役纪律如实上报，回单建议见 §6.1。

同时用**已提交的 P27W2 参考实现**（`backend/modules/spectral_fatigue` @ `8bbb6fc827`）做第二见证：Dirlik 损伤 3.975e-05，与本接收方偏差 3.410e-16；Rayleigh 偏差 4.097e-16。两者均用原文闭式，故一致到机器精度——说明契约与约定无歧义，问题只出在 N04 交付件的系数。

### 3.5 C4：`ice-bond-v1` 键强度映射

| 量 | 值 | 说明 |
|---|---|---|
| σ_b（DEM 键强度） | 1.500e+06 Pa | 载荷原值 |
| ε_b（断裂应变） | 3.000e-04 | = σ_b/E，E=5 GPa 晶格反演 |
| ℓ（特征长度） | 0.0125 m | DEM 晶格常数（唯一量纲桥） |
| τ_p（内聚峰值） | 1.500e+06 Pa | = σ_b |
| δ_c（失配位移） | 3.750e-06 m | = ε_b·ℓ |
| G_c（断裂能） | 2.8125 J/m² | = ½τ_pδ_c |
| 单键断裂力 | 86.8056 N | = k·ℓ·ε_b |

| 校核 | 参照 | 重算 | 相对偏差 |
|---|---|---|---|
| 内聚律能量积分 | G_c=2.8125 J/m² | 2.8125 J/m² | 1.579e-16 |
| 首纤维破坏载荷 | DEM 625.0000 N | TensorPhysics 625.0000 N | 1.819e-16 |

### 3.6 C5：成熟化单汇总（evidence_level 升级记录）

| 成熟化节点 | mission | 冻结记录数 | 结构化字段 | 要求 | evidence_level | 结论 |
|---|---|---|---|---|---|---|
| P27W3-N02 | 6728 | 0 | 0 | 3 | — | FAIL（—（mission 仍在运行，未产出成熟化单）） |
| P27W3-N04 | 6730 | 1 | 16 | 5 | REAL | PASS（docs/fatigue_verification_report.json） |

说明：C5 的机器口径是「成熟化单里结构化证据字段数」——`evidence_level`、`mission_id`、`title`、`all_pass`、`artifacts` 以及每条判据 `criteria.Cn` 各计 1 字段。平台表 `factory_evidence_records` 目前为空（0 行，按 mission_id 查也无记录），因此本节点改用节点工作树里冻结的成熟化单 JSON 作为主证据，并把平台表作为次证据并列上报；两种来源都在 verdict 里留了查询语句与结果，可复核。

### 3.7 图

![RAO 传递函数](fig1_rao_transfer.png)

*图 1　对端 STF 解（灰线，1200 点参考）、对端默认 40 点表（红方块）与本契约 158 点表（蓝点）。虚线为三个验收频率；垂荡在 ω≈1.45 rad/s 有过零点，是逐点插值误差爆炸的根源。*

![抽样充分性](fig2_sampling_fidelity.png)

*图 2　JONSWAP(Hs=2 m, Tp=8 s) 下显著幅值误差：基线表 vs 契约表，虚线为 1% 界面判据。*

![谱疲劳](fig3_spectral_fatigue.png)

*图 3　左：传递的双峰热点应力 PSD；右：同一 PSD 的三种损伤估计（Dirlik / 时域雨流 / Rayleigh）。*

![内聚律](fig4_ice_bond.png)

*图 4　左：由 DEM 晶格键映射出的双线性内聚律（阴影面积即 G_c）；右：两侧首纤维破坏载荷。*

![DAG](fig5_dag_state.png)

*图 5　gate 时刻 plan 165 的节点状态（本节点 N08 为执行者）。*

## 4 注意点（踩过的坑与局限）

1. **船型标识字段不足（真实缺陷，重要）。** 载荷只带 (L,B,T,Cb,Cwp)，对端用固定形状函数 √(1−|ξ|⁶) 重建水线。结果水线最大半宽 3.642 m（建模船宽 7.284 m），声明 B=10.0 m，偏差 27.2%；中站剖面面积 36.42 m² vs 墙侧 Wigley 的 50.0 m²。排水体积仍严格等于 3333.3 m³（=Cb·L·B·T），所以**仅靠 Cb/L/B/T 的几何门抓不到这个偏差**——几何契约必须携带站线剖面或有形状族标识。
2. **截面附加质量只能作诊断，不能作真值。** TensorPhysics 的二维 BEM 只对「深没」剖面有闭式锚（ρπa²）；浮体墙侧剖面需要在自由面用 Dirichlet、在水线盖用 Neumann 的混合边界，本模块未实现。因此把对端经验因子 f_a(ν)≈0.75 与 BEM 的差（23.8%）当模型形态差异记录，不进入判据。
3. **RAO 表不能逐点插值。** 基线表逐点最大相对误差 1144%，但谱域误差只有 0.44%/0.87%——因为过零点附近的大误差被谱加权压掉了。结论：接收端应做**带内能量平均**，不要逐点取值；同时建议把共振带加密写进契约。
4. **Dirlik 是经验闭合，必须用独立基准仲裁。** 接收方与对端 N04 传输偏差 19.6%，而对照时域雨流基准分别是 6.7%（接收方）与 22.0%（对端，MC 统计误差仅 0.56%）：偏差在闭合式一侧，不在接口一侧。Rayleigh 窄带在 α₁=0.653 的宽带上高估 55%（教材量级），说明这套三方对照确实能分辨“经验闭合 vs 时域真值”。
5. **本次顺带修掉一个真实缺陷（有基线对照）。** 跨仓比对暴露 TensorPhysics `fatigue.spectral` 把窄带 Rayleigh 损伤的循环率写成 m1/m0（平均频率），而 Rice 上跨零率应为 (1/2π)√(m2/m0)。窄带谱两者重合，故旧测试全绿；在本契约的宽带谱（α₁=0.653）上两者相差 34.7%（=1−α₁），修复后与对端偏差降到 4.097e-16。修复方式：新增 `SpectralMoments.nu_zero` 属性并让 `rayleigh_damage_rate` 改用它，保留 `nu0`=m1/m0 并改正其文档；189 条既有疲劳/水弹性/冰振测试全绿。
6. **C1/C5 是上游未收敛，不是本节点可修。** gate 时刻前置节点 2/7 已 done；成熟化单方面 N04 已产出 `evidence_level` 记录（16 字段，≥5 达标），N02 尚无记录（0 字段，mission 6728 仍在运行）。按纪律如实记 FAIL，不覆盖；N02 落地后重跑 gate 即自动收敛（判定幂等、只读）。
7. **对端 API 不可达。** 验收窗口内 ship-platform :8001/:8002 均返回连接拒绝，故全部对端数值来自源码物化 + 现场求解（提交哈希可复核），而非 HTTP 端点；接口语义（单位/归一化）从端点 schema 与内核实现双向核对。

## 5 误差判定（逐项，显式回答“是否 <3%”）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| 垂荡/纵摇/横摇幅值（3 频点） | 对端 motion_amplitudes | 0 | 是 |
| 长波极限垂荡 RAO | 解析极限 1（Newman） | 4.074e-04 | 是 |
| 长波极限纵摇 RAO | 解析极限 1（波陡归一化） | 4.406e-04 | 是 |
| RAO 表谱域保真（契约表） | 1200 点对端解 | 0.092% | 是 |
| Dirlik 损伤传递（对已提交 P27W2 实现） | 对端 Dirlik 闭式 | 3.410e-16 | 是 |
| Dirlik 损伤传递（对 N04 W3 交付件） | 对端 Dirlik 数值积分 | 0.1964 | **否**（对端闭合式缺陷，已上报返修） |
| Dirlik vs 时域雨流 | 雨流 MC | 6.72% | 否（模型误差，如实记录） |
| 键强度往返 σ_b | DEM σ_b | 0 | 是 |
| 内聚律能量积分 | 解析 G_c | 1.579e-16 | 是 |
| 首纤维破坏载荷 | DEM beam theory | 1.819e-16 | 是 |
| 船型建模船宽 | 声明 B | 27.2% | **否**（对端几何缺陷，已上报） |
| 截面附加质量（诊断） | TensorPhysics BEM | 23.8% | **否**（混边界差异，不作判据） |

## 6 遗留问题与下一步

1. **几何契约升级为 `rao-v2`**：载荷必须携带站线 (x_i, b_i, a_i) 或形状族标识 + 版本，并在接收端重算 Cb/Cwp/∇ 做几何门；本文 §4.1 的 −27% 船宽偏差就是现成的负控算例。
2. **RAO 表分辨率写进契约**：建议 ≥120 点且共振带 Δω≤0.02 rad/s（本轮 158 点表的谱域误差 0.092% 可作验收下限参考）。
3. **浮体剖面混合边界 BEM**：给 `hull_hydroelastic.sections` 增加水线盖 Neumann 条件，才能把对端经验因子 f_a(ν) 的模型误差定成权威数字（当前只能定性）。
4. **C1/C5 建议由战役调度器自动复跑**：本 gate 已是幂等、只读、可离线；N01–N07 转 done 后重跑 `python -m tensorphysics.integration.p27w3_gate` 即可自动收敛，无需人工改报告。
5. **成熟化单落账口径**：建议把 N02/N04 的 maturity 记录写成 `factory_evidence_records.metrics` 的结构化字段（evidence_level_from/to、criterion、value、threshold、artifact），本 gate 的 C5 判据即按 metrics 字段数计（3/5），可直接对接。

### 6.1 回单建议：ship-performance-platform / P27W3-N04（谱疲劳闭合式）

`backend/physics/spectral_fatigue_core.py::dirlik_parameters` 未实现 Dirlik (1985)。现实现把 `x = alpha1` 当作主参数、并令 `R = alpha1`，丢掉了含 m1 的 `xm`。建议替换为原文六式（与本仓 `tensorphysics.fatigue.spectral.dirlik_parameters` 逐行一致）：

```python
xm = (m1 / m0) * sqrt(m2 / m4)
D1 = 2.0 * (xm - alpha2**2) / (1.0 + alpha2**2)
R  = (alpha2 - xm - D1**2) / (1.0 - alpha2 - D1 + D1**2)
D2 = (1.0 - alpha2 - D1 + D1**2) / (1.0 - R)
D3 = 1.0 - D1 - D2
Q  = 1.25 * (alpha2 - D3 - D2 * R) / D1
```

修复后本契约的 C3 判据可在同一 PSD 上直接复验：宽带双峰谱（α1=0.653、α2=0.393、xm=0.2565）的 Dirlik 损伤应从 3.322e-05 变为 3.975e-05（+19.6%），与雨流基准的偏差从 22.0% 收敛到 6.7%；其自身 C2 判据（Dirlik vs 雨流 K-S）建议改为对**损伤**而非仅对 PDF 形状断言，否则系数错误仍可被 K-S 放过。

## 7 复现

```bash
# 1) 采集对端 golden（需要能读对端仓；不写对端仓）
python scripts/p27w3_capture_peer_goldens.py
# 2) 跑契约测试（hermetic，只读 golden）
PYTHONPATH=src python -m pytest tests/test_p27w3_integration_contracts.py -q
# 3) 生成判据 JSON / 报告 / 图
PYTHONPATH=src python -m tensorphysics.integration.p27w3_gate
python scripts/p27w3_report.py
```

