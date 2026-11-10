# FEM → DEM 断裂交接协议（`handoff.fem_to_dem`）

**状态** Implemented（节点 p273w3-N03）· **上游契约** `failure_faces` v3（冻结）
**代码** `src/tensorphysics/failure_handoff/{mixed_mode,transfer,protocol}.py`
**验证锚** 冰梁三点弯曲（混合模式）断裂 → 碎片 → DEM 接续
**报告** `docs/reports/p273w3_n03/REPORT.md` · **工件** `artifacts/p273w3_n03/`

## 1. 这层协议解决什么

`failure_faces` v3 只回答"材料在哪里分开、消耗了多少内聚能"。DEM 需要的却是**物体**：
多面体几何、质量、惯量张量、初始速度与自旋。两者之间的差距不能靠"文档约定"补齐，
必须是可执行、可审计、带判据的转换：

```
FEM 求解（位移控制三点弯曲，混合模式）
  → 裂纹尖端 K_I/K_II（位移相关法，r→0 外推）        mixed_mode.stress_intensity_factors
  → MTS 拐折角（判据） + Williams 应力场反演（独立）   mixed_mode.mts_kink_angle_deg / williams_stress_fit
  → 单元删除式裂纹扩展（第三路方向估计）              mixed_mode.propagate_crack
  → 失效面网络 → 封印 failure_faces v3 包             fem_export / schema（冻结，不改）
  → 多面体碎片：几何重建（散度定理）+ 质量 + 惯量      transfer.build_transfer_fragment / fragment_inertia
  → 初始速度/自旋继承（8 节点刚体最小二乘投影）        transfer.rigid_body_fit
  → 体积 / 动量 / 能量账本四门                        transfer.volume_budget / momentum_budget / energy_budget
  → DEM 接续（继承速度 + 自由落体 + 落地）             dem_rigid.RigidDEM
```

## 2. 契约边界（重要）

* **`failure_faces` v3 不改**：新增数据（裂纹尖端状态、失效单元体积、单元节点速度、
  碎片惯量与运动状态）写入伴生 `handoff_bundle.json`，用
  `packet_sha256` 绑定到被消费的那一版包。老消费者（只吃 v3 包）行为不变。
* **消费侧不采信生产侧数字**：碎片体积由自身顶点用散度定理重算，面积由 Newell 面积向量重算，
  内聚能取包内声明值但**与消费者记账分开**（否则账本误差会在两边同时抵消，门形同虚设）。

## 3. 判据（全部内联求值，数字进 `criteria_report.json`）

| 判据 | 定义 | 容差 |
|---|---|---|
| C1 | FEM 侧 MTS（应力场 Williams 反演 K）与判据侧 MTS（位移相关 K 闭式）的拐折角差 | ≤ 10° |
| C2 | 重建多面体体积 vs FEM 失效区（单元 Jacobian）体积，逐碎片 + 总量 | ≤ 3 % |
| C3 | 碎片线动量 vs FEM 裂纹面节点动量（角动量作副指标） | ≤ 2 % |
| C4 | 断裂内能 + 动能（含自旋）+ 势能账本，FEM 侧 vs DEM 侧闭合 | ≤ 5 % |

外加 `gate_sensitivity()` 负控矩阵（层厚 +5 %、丢碎片、扣 20 % 断裂能、角度偏 15°），
证明每个门都会红。

## 4. 用法

```python
from tensorphysics.failure_handoff import ICE_SPEC, run_protocol, write_protocol_artifacts

run = run_protocol(spec=ICE_SPEC, dims=(48, 12, 4), load_offset_m=0.024)
print(run.criteria()["C2_volume_relative_error"])
write_protocol_artifacts(run, "artifacts/p273w3_n03")
```

复现完整节点（协议 + 测试 + 图 + 报告 + 发布）：

```bash
python scripts/run_p273w3_n03.py --dimensions 48,12,4 --post
```

## 5. 适用范围与已知局限

* 准脆性、小应变、线弹性裂尖场；单裂纹；穿厚直裂纹前缘。
* 裂纹面必须是坐标平面（v3 的 `faces_on_plane` 导出方式决定），拐折裂纹的真实空间曲面
  目前用竖直平面近似 —— 这是**契约边界**，不是实现缺陷。
* 裂纹扩展方向在 5 mm 网格上的单元删除路径量化带约 ±45°/步，只作支撑证据；
  C1 以应力场 MTS 为准。
* DEM 侧仍是平动刚体积分器（无转动积分、无碎块互碰），自旋只在交接账本中记账。
* 无实验基准，故为 Implemented，不主张 Verified。