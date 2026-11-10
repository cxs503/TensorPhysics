# 多物理场耦合守恒 Mortar 插值与非匹配界面守恒残差机器门禁（P162W1-N06）

**campaign node** P162W1-N06 · **repo** TensorPhysics · **module**
`tensorphysics.multiphysics_mortar` · **operator** P1 迹空间上的 L2 投影 mortar，
对偶（双正交）基用于对角化耦合 · 本文件由
`scripts/run_p162_n06_mortar_acceptance.py` 生成，文中所有数字来自同一次运行的
`artifacts/p162_n06/acceptance.json`，无手抄。

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

多物理场分区耦合的界面是两个求解器的公共边界：流体侧（master，细网格）把
质量/动量/能量交给结构或冰侧（slave，粗网格）。两侧网格不匹配（节点数不同、
一侧余弦聚站、或三角形对角线交叉），因此界面传递算子必须是**守恒**的——否则
耦合系统在界面上凭空产生或吞掉质量、动量与能量，而结果看上去仍然"合理"。

三个验证面（数值全部落在同一次 acceptance 运行里）：

| 项 | 取值 |
|---|---|
| C1 interface | xi in [0, 1] (advection interface) |
| C1 master/slave nodes | 17 / 8 |
| C1 rho | 1.0 + 0.35 sin(2 pi x) + 0.12 x^2  [kg/m^3] |
| C1 u | 0.80 + 0.20 cos(3 pi x)  [m/s] |
| C1 T | 285 + 15 cos(2 pi x) + 4 x  [K] |
| C2 interface | y = 0.25 sin(2 pi x / 4), x in [0, 4] m |
| C2 master/slave nodes | 41 / 13 |
| C2 slave grading | cosine clustering at both ends (structure mesh) |
| C2 rho | 1025 (1 + 0.02 sin(3s) + 0.01 x/L)  [kg/m^3], s = 2 pi x / L |
| C2 u | (0.42 sin s + 0.08, 0.16 cos 2s)  [m/s] |
| C2 T | 283.15 + 6 cos s - 1.5 x/L  [K] |
| C3 interface | conical hull patch, chart (theta, z), theta in [0, 0.9], z in [0, 6] m |
| C3 radius | r(z) = 5.0 - 0.2 z  [m]  ->  J = r sqrt(1 + r'^2) affine in z |
| C3 master/slave tri | 14 x 9 (flow side, flip=True) / 5 x 4 (ice side, flip=False) |
| C3 rho | 1025 (1 + 0.02 cos(2 pi z / L) sin(3 theta))  [kg/m^3] |
| C3 u | (0.60 sin(pi z/L), 0.25 cos(4 theta), 0.08)  [m/s] |
| C3 T | 271.35 + 0.8 sin(3.5 theta) - 0.15 z/L  [K] |
| specific heat | C1 1005.0 J/(kg K), C2/C3 3993.0 J/(kg K) |

单位：$\rho$ kg/m$^3$，$u$ m/s，$T$ K；$c_v\approx c_p$ 为弱可压液体的量热
替身（海水 10 °C 量级），**不是**可压缩气体热力学模型。

## 2. 怎么做的（方法 / 流程 / 关键设置）

**(a) 迹空间**：两侧都是 P1 分片线性迹空间，构造上是单位分解
$\sum_i\varphi_i=1$（`spaces.LineTraceSpace` / `TriTraceSpace`）。
1D 侧用弧长 Jacobian $d\Gamma/d\xi=\sqrt{1+y'^2}$；3D 侧用回转面
Jacobian $J=r\sqrt{1+r'^2}$，所以积分是真实面积（m$^2$）而不是参数面积。

**(b) 对偶基与耦合矩阵**：$M_{ij}=\int_\Gamma\varphi_i\varphi_j\,d\Gamma$，
对偶基 $\psi_i=\sum_j(M^{-1})_{ij}\varphi_j$，交叉矩阵
$C_{jk}=\int_\Gamma\varphi_j^{slave}\varphi_k^{master}\,d\Gamma$。
传输有两条代数上等价的路：矩阵路 $u_s=M^{-1}Cu_m$，对偶路
$c_i=\int_\Gamma u_m\psi_i$。两条路的数值一致性在测试里验证到 $10^{-12}$。

**(c) 守恒为什么是结构性的**：因为接%收基是单位分解，

$$\int_\Gamma u_s=\sum_j\int_\Gamma u_s\varphi_j=\sum_j b_j=\int_\Gamma u_m\sum_j\varphi_j^{slave}=\int_\Gamma u_m.$$

注意这用的是**原始基**的单位分解，**不是**对偶基的单位分解——L2 对偶基一般
不是单位分解（本实现在 C2 界面上测得
`dual_partition_of_unity_defect = 37.971`，
远大于 1），若按"对偶基构成单位分解"去推导守恒就会漏质量。这条先验被实测否定，
写在 §5。

**(d) 残差**：把 $\rho$、$\rho u$、$\rho E$（$E=c_vT+|u|^2/2$）分别送过算子，
在接收侧重建 $(\rho,u,T)$，再用接收侧**自己的**测度独立积分：

$$R_q=\frac{\|\int_\Gamma q_s\,d\Gamma-\int_\Gamma q_m\,d\Gamma\|}{\|\int_\Gamma q_m\,d\Gamma\|}.$$

**(e) 门禁**：`gate.ConservationGate` 对声明的全部残差做阈值判定，**fail-closed**
（未声明名称的测量直接报错，不允许默认通过）；超限抛 `ConservationGateError`。

**(f) 几何与性能**：重叠多边形裁剪复用
`interface_operators.polygons`（全仓唯一几何核）；组装按顶点数分组做批量
求积，并把裁剪写成纯 float 的快速版（与 canonical 内核逐面积比对为 0，见 §5）。

## 3. 计算结果（机器判据）

| 判据 | 测量 | 阈值 | 判定 |
|---|---|---|---|
| C1 1D 标量质量残差 | 0.000e+00 | 1.000e-12 | PASS |
| C2 2D FSI 动量残差 | 5.379e-16 | 1.000e-08 | PASS |
| C2 2D FSI 能量残差 | 1.978e-16 | 1.000e-06 | PASS |
| C3 3D 流-冰 质量残差 | 1.465e-16 | 1.000e-06 | PASS |
| C3 3D 流-冰 动量残差 | 5.018e-16 | 1.000e-05 | PASS |
| C4 匹配网格 1D 恒等 | 3.101e-16 | 1.000e-12 | PASS |
| C4 匹配网格 2D 恒等 | 6.556e-16 | 1.000e-12 | PASS |
| C5 加密单调（质量/动量） | True / True，观测阶 6.07 | 严格下降 | PASS |
| C6 vs 最近邻最小比值 | 1.530e+11 | ≥ 1000 | PASS |
| C7 构造时间占比 | 0.0160 | ≤ 0.10 | PASS |

门禁逐项（`gate` 段）：

| 名称 | 残差 | 阈值 | 判定 |
|---|---|---|---|
| C1_1d_mass | 0.000e+00 | 1.000e-12 | PASS |
| C2_2d_energy | 1.978e-16 | 1.000e-06 | PASS |
| C2_2d_momentum | 5.379e-16 | 1.000e-08 | PASS |
| C3_3d_mass | 1.465e-16 | 1.000e-06 | PASS |
| C3_3d_momentum | 5.018e-16 | 1.000e-05 | PASS |
| C4_identity_1d | 3.101e-16 | 1.000e-12 | PASS |
| C4_identity_2d | 6.556e-16 | 1.000e-12 | PASS |

三个物理面的双向与基线残差：

| case | 量 | master→slave R | slave→master R | 最近邻 R | 比值 |
|---|---|---|---|---|---|
| C1 1D 标量传输 | mass | 0.000e+00 | 0.000e+00 | 2.922e-04 | 2.92e+11 |
| C1 1D 标量传输 | momentum | 0.000e+00 | 1.364e-16 | 6.374e-04 | 6.37e+11 |
| C1 1D 标量传输 | energy | 0.000e+00 | 0.000e+00 | 1.530e-04 | 1.53e+11 |
| C2 2D FSI 面板 | mass | 0.000e+00 | 6.419e-13 | 5.576e-04 | 5.58e+11 |
| C2 2D FSI 面板 | momentum | 5.379e-16 | 1.505e-11 | 6.006e-02 | 6.01e+13 |
| C2 2D FSI 面板 | energy | 1.978e-16 | 7.346e-13 | 1.218e-03 | 1.22e+12 |
| C3 3D 流-冰 patch | mass | 1.465e-16 | 7.324e-16 | 8.251e-04 | 8.25e+11 |
| C3 3D 流-冰 patch | momentum | 5.018e-16 | 3.613e-16 | 2.311e-02 | 2.31e+13 |
| C3 3D 流-冰 patch | energy | 4.246e-16 | 8.492e-16 | 8.882e-04 | 8.88e+11 |

C5 加密族（曲率段 $r=5.0-0.2z+0.06z^2$，$J$ 非多项式，残差由求积缺陷主导）：

| slave $n_\theta$ | $h$ [m] | 重叠片数 | $R_m$ | $R_p$ |
|---|---|---|---|---|
| 6 | 1.500e-01 | 92 | 9.804e-09 | 5.329e-08 |
| 12 | 7.500e-02 | 344 | 1.483e-10 | 7.746e-10 |
| 24 | 3.750e-02 | 1440 | 2.302e-12 | 1.194e-11 |
| 48 | 1.875e-02 | 7392 | 3.222e-14 | 1.896e-13 |

C7 性能（`steps=200`，定义：coupled block-Gauss-Seidel loop: 1 thermal solve per side (red-black GS, 60 sweeps, 64x64 grid) + mortar exchange both ways per step）：
构造 0.150 s，耦合循环 9.245 s，
单步 46.22 ms，占比 0.0160，
摊平临界步数 $N^*=9t_{build}/t_{step}\approx29$ 步。
最坏情形（只算裸矩阵乘交换、不算求解器工作）占比
0.896——这是**上界**，说明构造一次后必须
被至少约 $N^*$ 步复用，见 §5。

一致性（非守恒）判据——传递误差 vs 解析场：

| master/slave 节点 | $h$ | mortar $L_2$ | 最近邻 $L_2$ |
|---|---|---|---|
| 16/8 | 0.1250 | 1.895e-02 | 3.030e-02 |
| 32/16 | 0.0625 | 4.773e-03 | 1.496e-02 |
| 64/32 | 0.0312 | 1.189e-03 | 6.143e-03 |
| 128/64 | 0.0156 | 2.969e-04 | 3.049e-03 |

mortar 观测阶 2.00，最近邻 1.12。

## 4. 结果图

| 文件 | 内容 |
|---|---|
| `artifacts/p162_n06/fig1_1d_transfer.png` | C1：解析/主/从密度、逐 DOF 质量平衡缺陷、mortar vs 最近邻残差 |
| `artifacts/p162_n06/fig2_2d_fsi_panel.png` | C2：非匹配面板几何与节点、$\rho u_x$ 传递、残差对比 |
| `artifacts/p162_n06/fig3_3d_residual_map.png` | C3：冰侧 patch 上传递后的 $\rho E$ 云图 + 局部质量平衡缺陷图 |
| `artifacts/p162_n06/fig4_refinement_accuracy.png` | C5 加密递减曲线 + 一致性（$L_2$ 阶）曲线 |
| `artifacts/p162_n06/fig5_criteria_margin.png` | 七条判据的满足裕度（测量/允许） |

## 5. 注意点（坑 / 局限 / 适用范围）

1. **守恒来自原始基的单位分解，不是对偶基**。L2 对偶基双正交但非单位分解
   （实测缺陷 37.97），按后者推导会
   漏质量。测试 `test_dual_basis_is_biorthogonal_but_not_a_partition_of_unity`
   把这条钉住。
2. **曲率段残差是求积缺陷，不是守恒缺陷**。锥形段 $J$ 仿射 → 求积精确 → 残差
   到舍入；曲率段 $J$ 非多项式 → 残差由两侧测度求积不一致主导，随 $h$ 以
   6.1 阶下降（C5），这既是加密性的证据也是适用范围
   的边界：若几何只给离散点云（无解析 $J$），测度闭合误差就是这个量级。
3. **局部不平衡是正常的**。粗侧单元无法复现细侧分布，逐 DOF 缺陷图（图 3）在
   $10^{-15}$–$10^{-16}$ 量级非零；门禁判的是**全局**积分残差，不是逐单元。
4. **性能**：单次构造必须被复用。裸交换占比
   0.896 说明构造摊销需要约
   29 步；短耦合（<20 步）应把算子缓存到运行期。
   本节点把组装批量向量化 + 快速裁剪后，3D 面构造从 0.94 s 降到
   44 ms。
5. **`scipy.linalg.cho_factor` 在本机对 33–65 DOF 的 SPD 系统直接段错误**
   （进程 139），因此 `spd_solve` 走 NumPy LAPACK；对 150 DOF 质量矩阵约 42 ms，
   占单步耦合成本 ~2%。这条是环境级坑，不得在库路径上引入 scipy 的该入口。
6. **适用范围**：P1 连续迹空间（FE/CG 类求解器）。单元中心（FV）迹空间的守恒
   重叠算子已存在于 `interface_operators`（IN19-T2 / IN50-N08），两者共用几何核，
   本模块不重复实现。$c_v\approx c_p$ 的量热替身仅适用弱可压液体；气体需换成
   $\rho e$ 的真实状态方程。

## 6. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 误差 | 是否 < 3 % |
|---|---|---|---|
| C1 质量守恒 | 精确守恒律（0） | 0.000e+00 | 是（机器精度，$\ll$ 3 %） |
| C2 动量守恒 | 精确守恒律（0） | 5.379e-16 | 是 |
| C2 能量守恒 | 精确守恒律（0） | 1.978e-16 | 是 |
| C3 质量守恒 | 精确守恒律（0） | 1.465e-16 | 是 |
| C3 动量守恒 | 精确守恒律（0） | 5.018e-16 | 是 |
| 传递精度（最细网格） | 解析 $\rho(x)$ | 2.969e-04 | 是（0.0297 %） |
| 传递精度（最粗网格） | 解析 $\rho(x)$ | 1.895e-02 | 是（1.90 %，粗网格 8 单元） |
| 最近邻基线（同网格） | 解析 $\rho(x)$ | 3.049e-03 | 是（0.30 %，为 mortar 的 10.3 倍且仅一阶收敛） |

结论：守恒残差相对精确守恒律为机器精度（全部 $\le 10^{-15}$ 量级，判据通过）；
传递精度在最粗的 8 单元网格为 1.90 %、最细网格
0.0297 %，两者都 < 3 %（粗网格偏差来自离散分辨率而非
算子，二阶收敛见 §3 一致性表）。最近邻插值同样是"< 3 %"量级
（0.30 %），但它是 mortar 的
10.3 倍且只有
1.12 阶——**守恒差距（C6：3 个数量级以上）比精度差距更
能区分两类算子**，这也说明"精度不超标"不等于"耦合可信"。

## 7. 复现

```bash
PYTHONPATH=src python -m pytest tests/test_multiphysics_mortar.py -q
PYTHONPATH=src python scripts/run_p162_n06_mortar_acceptance.py
```

判据总状态：**ALL PASS**
（`all_criteria_passed = True`）。
