# 交差审计报告 — m14679 GAP-CROSS-MAP（TensorPhysics）

审计单 m15004 · verdict_by `board-audit` · 被审对象 m14679（已 succeeded，PR #391 merged）

## 0. 被审身份（可核对的 commit/run 身份）

| 项 | 值 |
|---|---|
| 合并提交 | `3568988` Merge #391 (m14679) via push-merge；内容提交 `8b6807a` "m14679: matured mission deliverables（7 个文件被门禁滤除）" |
| 被审源码 | `src/tensorphysics/grid_to_grid_remap/{overlap.py,remap.py,acceptance.py,__init__.py}`（1404 行，本审计工作树 main 分支同一内容） |
| 交付工件 | `/root/tp-agent/work/m14679/{runs,artifacts}/m14679/*`（**未进合并提交**，见 M1） |
| 复现工件（本审计） | `runs/m15004_audit_m14679/checks.json`、`artifacts/m15004/m14679_convergence_audit.png` |
| 复现命令 | `python scripts/audit_m14679/audit_m14679.py`（探针 A–F）、`python generate_m14679_artifacts.py`（交付自带的复现命令）、`python -m pytest tests/test_grid_to_grid_remap.py -q` |

## 1. 三件套核验（逐项，非重做）

| 三件套 | 交付声明 | 审计核实 | 结论 |
|---|---|---|---|
| run_artifacts | `runs/m14679/*.json` + `artifacts/m14679/*.png` | 文件真实存在且非空；重新执行 `generate_m14679_artifacts.py` 后 4 份 JSON 与交付副本 **逐位相同**（逐文件结果：benchmark:identical | conservation:identical | convergence:identical | symmetry:identical；返回码 0） | ✅ 真工件、真可复现 |
| benchmark ref vs got | 守恒缺陷 ~1e-15；收敛阶 1.31 / 2.11；对称往返 L2 2.64e-2 | 数值全部复现；但收敛阶的"参考解"与文档不符、阶数非渐近（B1/B2） | ⚠️ 数字真，**结论被夸大** |
| repro_cmd | `generate_m14679_artifacts.py` + pytest | 本审计实跑 `pytest tests/test_grid_to_grid_remap.py -q` → **14 passed, 1 warning in 17.83s**（rc=0）；工件逐位复现 | ✅ 可执行 |
| external_anchor | 解析积分 4/π² | 真：`acceptance.py:167` 用 `4/π²` 作 smooth_sin 精确积分 | ✅ 真（但仅此一项，见 N1） |

## 2. 独立探针结果（另一条路线重算，不只信交付自测）

### A. 几何核 — 三角形求交面积 vs 独立光栅法
200 对随机 CCW 三角形（过滤退化），交付的 Sutherland–Hodgman 内核面积与 800×800 光栅二重积分的交点面积对照：
用对 114 对，最大相对偏差 **3.03e-03**，中位数 2.50e-03，超容差(1e-2)对数 0 → **PASS**（偏差量级=光栅法本身的边界误差）。
方向健壮性：把源网格连接关系翻转成 CW 后，交面积最大变化 6.9e-17，场值差 2.3e-13，`total_overlap` CW 1.000000000 / CCW 1.000000000 → 未见方向敏感缺陷。

### B. 代数核 — 稀疏算子的双侧单位分解
n_src=32×32（2048 单元）→ n_tgt=48×48（4608 单元）扰动网格：
行和 vs |T_j| 最大相对误差 1.13e-12；列和 vs |S_i| 1.22e-12；Σ重叠面积 − 域面积 = -5.6e-16 → **PASS**。
即"守恒是代数恒等式"这一声明成立，**但它是结构性质**：只证明算子行随机/列守恒，不证明映射精度（见 N2）。

### C/D. 收敛性 — 参考解与阶数重算（本审计核心发现）

参考解口径：交付代码 `acceptance.py:296-303` 注释称"exact cell average of the analytic field"，实际用**单元形心点值** `exact_fn(tgt_centroids)`。用 5×5 子三角形积分构造近似精确单元均值作对照，形心点值参考自身的 L2 误差 = **8.381e-05**，相当于被审最细点（n_src=48）bounded 误差的 **8.7%**、n_src=96 时的 **57%**。

| n_src | h | avg L2 (点值参考) | bounded L2 (点值参考) | bounded L2 (精确均值参考) | bounded 守恒缺陷 |
|---|---|---|---|---|---|
| 8 | 0.1250 | 6.345e-02 | 5.099e-02 | 5.099e-02 | 1.2e-14 |
| 12 | 0.0833 | 4.167e-02 | 2.990e-02 | 2.990e-02 | 3.0e-15 |
| 16 | 0.0625 | 2.809e-02 | 1.743e-02 | 1.743e-02 | 6.8e-16 |
| 24 | 0.0417 | 1.722e-02 | 7.715e-03 | 7.714e-03 | 2.2e-15 |
| 32 | 0.0312 | 1.159e-02 | 4.802e-03 | 4.805e-03 | 2.5e-15 |
| 48 | 0.0208 | 5.862e-03 | 9.678e-04 | 9.771e-04 | 4.1e-16 |
| 64 | 0.0156 | 3.185e-03 | 3.353e-04 | 3.535e-04 | 8.2e-16 |
| 96 | 0.0104 | 1.505e-03 | 9.767e-05 | 1.461e-04 | 5.5e-16 |

全局拟合阶（log-log 最小二乘）：

| 口径 | average | bounded |
|---|---|---|
| 交付口径 levels=(8,16,24,32,48) 点值参考 | 1.308 | 2.109 |
| 仅 levels=(8,16,24,32) | 1.217 | 1.719 |
| 全档 (8…96) 精确均值参考 | 1.504 | 2.462 |

逐档局部阶：average [1.04, 1.37, 1.21, 1.38, 1.68, 2.12, 1.85]；bounded [1.32, 1.88, 2.01, 1.65, 3.95, 3.68, 3.04]。
→ 交付的 **2.11** 恰好是"含 n_src=48 那一点 + 点值参考"的拟合值；去掉该点降为 1.72，而局部阶在 n_src≥48 后跳到 3–4（bounded 理论只有 2 阶）——这是**网格族同构导致的超收敛/参考解污染**，不是渐近阶。**"二阶收敛已建立"不能成立，可支持的说法是 1.7–2.1 区间内 1<阶≤2。**

### F. 复现口径
`generate_m14679_artifacts.py` 在审计工作树内重跑（返回码 0）：`{'benchmark': 'identical', 'conservation': 'identical', 'convergence': 'identical', 'symmetry': 'identical'}`。交付自带的 pytest 套件在本审计工作树全绿：**14 passed, 1 warning in 17.83s**（rc=0）；`runs/m14679/benchmark.json` 2634 B 与交付副本同源。

## 3. 判级（L0–L5）与理由

**判定：L2 = 独立功能实现。** gate 侧可达最高级也是 L2（L3 需 `integration_callers`，本模块无生产调用方）。

| 级 | 判据 | 是否成立 | 证据 |
|---|---|---|---|
| L1 数学/单元验证 | tests 非空且断言不变量 | ✅ | `tests/test_grid_to_grid_remap.py` 14 passed；守恒 gate 1e-12、单调性 n_cells_above_max==0、收敛阶断言 |
| L2 独立功能实现 | 真代码 输入→计算→输出，有入口与运行工件 | ✅ | `overlap.py:132/251`、`remap.py:63/255`（BJ 限制器 193 + 守恒修复 213）、入口 `__init__.py` 懒加载 + `acceptance.py:353 run_acceptance`；工件逐位复现 |
| L3 系统集成 | 主求解器/平台侧调用方 | ❌ | 全仓 grep：仅 `tests/test_grid_to_grid_remap.py` 与 `generate_m14679_artifacts.py` 引用，无求解器/耦合器调用 |
| L4 基准验证 | L3 全部 + 基准 ref/got + repro | ❌（材料半备） | 有解析场收敛研究 + 守恒基准 + 可复现命令，但缺 L3，且基准参考解口径有瑕（B2）、阶数非渐近（B1） |
| L5 科研工程 | L4 + external_anchor | ❌ | 仅一个解析积分锚（4/π²），无文献/实验对照 |

> 说明：L4 的**材料要求**（解析基准 + 误差分析 + repro_cmd）实际大半具备，但按门禁阶梯 L4 需先过 L3；把 L4 材料记为潜能力，不上调级别。

## 4. 严重度排序的发现

| 编号 | 严重度 | 发现 | 证据/复现 |
|---|---|---|---|
| B1 | 阻断（对结论，不对功能） | "bounded 观测阶 2.11 ≥ 2.0" 是拟合口径产物：4 档 8–32 为 1.72；局部阶 [1.32, 1.88, 2.01, 1.65, 3.95, 3.68, 3.04]，n_src≥48 后 3–4 阶（同构网格对齐超收敛） | `runs/m15004_audit_m14679/checks.json:C_convergence`；图 `artifacts/m15004/m14679_convergence_audit.png` |
| B2 | 主要（验证口径/文档） | `case_convergence` 注释称参考=精确单元均值，实为形心点值；污染 8.38e-05 = 最细点误差的 8.7%(n=48)/57%(n=96) | `acceptance.py:296-303`；checks.json:C_convergence |
| M1 | 中（证据保管） | 交付的 7 个工件（3 png + 4 json）被合并门禁全部滤除，`git ls-files runs/m14679 artifacts/m14679` 为空 → 仓库内无留存，只在 m14679 工作树 | `git ls-files runs/m14679 artifacts/m14679` 空输出；`8b6807a` 提交标题"7 个文件被门禁滤除" |
| M2 | 中（能力范围） | 无任何生产调用方，库孤岛（FSI/耦合流程未接） | 全仓 grep 仅测试+生成器 |
| N1 | 轻（外部锚） | external_anchor 仅解析积分 4/π²，无文献/独立实现对照（supermesh CSR 类方法有成熟文献可对标） | `acceptance.py:167` |
| N2 | 轻（判据口径） | "映射误差<1%"只对**守恒泛函**成立（≤1e-14）；**场量精度**不满足：average 模式 n=48 时 L2 相对 1.17e-2，对称往返 L2 相对 2.64e-2 | `runs/m14679/benchmark.json:convergence/symmetry` |

**未发现造假迹象**：所有声明数字逐位复现；测试断言的是不变量（1e-12 守恒门、单调性零越限、阶数下界），不是"跑一遍就算过"。

## 5. 未执行的检查（明确 not_run，非 pass）

- 与公开文献/第三方实现（Farrell 2009 supermesh、Ullrich CSR、ESMF）的数值对照：未做（本机无 shapely/参考数据）。
- 三维映射、曲面/球面网格、非凸域与带孔网格：未测（模块为 2D 平面三角形，已在 docstring 说明）。
- 性能/内存标定（bbox 过滤 vs O(N²)，大网格 1e6 单元）：未做。
- `p1_recon`（无限制器）二阶模式的越限行为：未量化（测试仅覆盖 bounded）。
- 与 TensorLBM/FEM 真耦合路径的端到端守恒验证：无调用方，不可测。
- external anchor 的文献核验：未做。

## 6. 处置（Disposition）

**approve with notes（L2 confirmed）**：功能实质为真（真几何核、真守恒算子、真限制器与守恒修复、14 测试全绿、工件逐位可复现），守恒硬门通过且余量极大；但交付的两条**结论性声明被夸大**（收敛阶渐近性 B1、参考解口径 B2），工件未入库（M1），能力未集成（M2）。建议返修（非阻断本单消账）：

1. `case_convergence` 参考解改为子三角形积分（或明确写"形心点值近似"），并把重跑阶数写进报告；
2. 收敛阶表述改为"测试区间内 1.7–2.1，n_src≥48 出现网格同构超收敛，渐近阶未建立"；
3. 至少一个生产侧调用（例：把 `remap_cell_average` 接到某耦合/后处理流程）以够到 L3；
4. 把 `runs/m14679`、`artifacts/m14679` 纳入受版本管理的证据目录（避免门禁滤除后证据只存在于工作树）。

## 7. 独立性声明与局限

- 审计者阅读的是合并后 main 上的同一份源码；复现时执行的是**交付自带的**生成脚本，因此"复现"证明的是可重复性，不是独立实现正确性。
- 几何探针用光栅二重积分（独立路线，但精度 O(1/N)，仅点对点；未做全网格级面积审计）。
- 收敛重算共用交付的 `case_convergence` 网格族与场函数，只替换参考解口径与档位集合；因此 B1/B2 是"口径敏感性"结论，不含新物理。
- 本审计未获取任何外部文献数据，N1 的"应对标文献"是建议而非已完成的对照。
