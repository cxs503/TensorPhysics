# MPI 域分解并行有限元基础设施（p284w3-n04-fem-mpi-parallel）

> 交付仓：TensorPhysics ｜ 模块：`src/tensorphysics/fem_parallel/` ｜ 后端：`auto`
> 机器可算判据全部由 `python -m tensorphysics.fem_parallel.verification` 直接产出
> （`artifacts/fem_parallel/verification.json`），本文件由 `scripts/fem_parallel_report.py`
> 读取该 JSON 自动渲染，**零手抄**。

## ① 案例介绍

船舶结构有限元（舱段/冰带板/局部加强）在 10⁶ 自由度量级上单机内存与时间都吃紧，域分解并行是
工程化的必要条件。本节点补齐 `GAP-FEM-PARALLEL`：一套可度量、可复现的并行组装 + 并行 Krylov
求解基础设施，并在三个物理算例上验证：

| 算例 | 方程 | 离散 | 用途 |
|---|---|---|---|
| Poisson | $-\nabla\cdot(\nabla u)=f$ | Q1 双线性/三线性 | 强/弱扩展主算例（N=10⁶） |
| 变导热 | $-\nabla\cdot(k(\mathbf{x})\nabla T)=f$，$k=1+0.5\sin\pi x$ | Q1（Gauss 点赋系数） | 非均匀系数组装正确性 |
| 线弹性 | $-\nabla\cdot\boldsymbol\sigma(\mathbf u)=\mathbf f$，平面应变 | Q1，$E$=210 GPa，$\nu$=0.3 | 多分量（2/3 DOF/节点）并行正确性 |

制造解：$u=\sin\pi x\sin\pi y+0.5\,x(1-x)y(1-y)$，
$f=2\pi^2\sin\pi x\sin\pi y+[x(1-x)+y(1-y)]$。**多项式项不可省**——纯正弦模态是均匀网格
Q1 九点差分格式的精确特征向量（可分离），CG 一步即收敛，会使全部求解器对比退化。

计算规模：强扩展固定 $N$=1,000,000 DOF（网格 1000x1000），
$P\in${1, 2, 4}；弱扩展固定 $N/P$=100,000 DOF；
Krylov 相对容差 1e-08。

计算环境（测量可信度的前提）：

| 项目 | 值 |
|---|---|
| CPU 逻辑核数 | 128 |
| 平台 | Linux loongarch64 |
| 软件版本 | python 3.11.6 · numpy 2.4.6 · scipy 1.16.0 |
| 宿主 load average（报告生成时采样） | 230.1, 264.1, 299.9 |
| 并行后端 | auto（`auto` → 无 mpi4py 时落到进程级共享内存传输） |

## ② 方法

1. **域分解**：结构化网格 Cartesian 分块（`partition.CartesianPartition`），秩数在 `ndim` 个维度上
   因式分解，先最小化最大块体积（负载），再最小化界面面积（通信量）；块边界为连续区间，
   与 MPI Cartesian 拓扑一致。非结构网格提供 `partition_graph()`：安装 `pymetis` 时走 METIS，
   否则退回**递归坐标二分 RCB + Kernighan–Lin 边界精化**（本机无 METIS，实际走 RCB+KL，
   边割指标 `edge_cut()` 用于对照）。
2. **并行组装**：每个秩组装**自己拥有的行**；由于所有触及自有节点的单元都落在“块 ±1 层”的
   halo 盒内，行和是完整和，无需 off-process 汇总（与 gather-scatter 组装结果逐元素一致，
   已用独立朴素参考实现 `reference.assemble_reference` 逐元素比对）。单元矩阵用参考单元
   2ᵈ 点 Gauss 积分生成，变系数（$k(\mathbf x)$）在 Gauss 点求值。
3. **通信**：`HaloPlan` 按全局节点编号排序收发块（双侧一致，无元数据交换）；传输层三种后端
   实现同一 MPI 语义接口：`SerialComm`、`SharedMemoryComm`（nranks 个独立进程 + POSIX 共享
   内存 arena，双缓冲、每次集合通信 1 次 barrier）、`MPIComm`（mpi4py，自动探测）。
4. **并行求解**：PCG；每次内积一次 `allreduce`，每次算子应用一次 halo 交换。
   预条件：`Jacobi`（对角）与 `ASM`（一级加性 Schwarz，自有 DOF 块 `splu` 精确求解）。
5. **计时口径**：`T(P)=max_rank(t_assemble+t_setup+t_solve)`；`T_comm` 为集合通信内部计时的
   累计（含 barrier 等待，与 MPI 计时语义一致）；`T_comm/T_solve` 即通信占比。

## ③ 计算结果

### 强扩展（N=1,000,000 DOF）

| P | t_assemble [s] | t_solve [s] | S(P) | E(P) | iters | T_comm/T_solve |
|---|---|---|---|---|---|---|
| 1 | 4.32 | 123.667 | 1.00 | 1.000 | 1063 | 6.59e-04 |
| 2 | 3.87 | 161.987 | 0.76 | 0.382 | 1063 | 0.478 |
| 4 | 2.07 | 94.637 | 1.31 | 0.327 | 1063 | 0.303 |

### 弱扩展（N/P=100,000 DOF）

| P | grid | t_solve [s] | T(P)/T(1) | iters | load std/mean |
|---|---|---|---|---|---|
| 1 | 316x316 | 7.582 | 1.000 | 329 | 0.000 |
| 2 | 447x447 | 12.790 | 1.687 | 468 | 0.158 |
| 4 | 632x632 | 27.172 | 3.584 | 666 | 0.139 |

### 并行 vs 串行（同一离散、同一容差）

| 算例 | grid | P | iters | rel L2(par, ser) | rel L2(par, exact) |
|---|---|---|---|---|---|
| poisson_manufactured | 65x63 | 2 | 112 | 3.63e-16 | 2.07e-04 |
| poisson_manufactured | 65x63 | 4 | 112 | 3.73e-16 | 2.07e-04 |
| heat_variable_k | 65x63 | 2 | 109 | 1.97e-15 | 1.91e-04 |
| heat_variable_k | 65x63 | 4 | 109 | 1.35e-15 | 1.91e-04 |
| elasticity_patch | 33x31 | 2 | 194 | 8.83e-15 | 1.33e-13 |
| elasticity_patch | 33x31 | 4 | 194 | 9.33e-16 | 1.39e-13 |

### 制造解收敛（Q1，vs 解析解）

| grid | h | rel L2 error | observed order |
|---|---|---|---|
| 33x33 | 0.03125 | 8.04e-04 | - |
| 65x65 | 0.01562 | 2.01e-04 | 2.00 |
| 129x129 | 0.00781 | 5.02e-05 | 2.00 |

### PCG vs CG 迭代数

| 变体 | iters | 相对 CG 降幅 [%] | t_solve [s] | T_comm/T_solve |
|---|---|---|---|---|
| none | 329 | 0.0 | 2.590 | 0.278 |
| jacobi | 329 | 0.0 | 2.636 | 0.430 |
| asm | 53 | 83.9 | 3.332 | 0.317 |

### 宿主机争用对测量的影响（同一 1e6 DOF 算例的重复测量）

| 运行 | 规模 | t_solve [s] | t_solve / 本次P=1 | E(P) |
|---|---|---|---|---|
| 本次（run1） | P=1 | 123.7 | 1.000 | 0.382 |
| run2 (load≈220) | P=1 | 223.8 | 1.81 | 1.000 |
| run2 (load≈220) | P=2 | 142.9 | 1.16 | 0.783 |
| run2 (load≈220) | P=4 | 95.4 | 0.77 | 0.586 |
| run3 (load≈650) | P=8 | 112.0 | 0.91 | 0.125 |

**同一问题、同一秩数、同一代码，P=1 的墙钟时间在不同时段相差近 2 倍**——这正是共享宿主机
（128 核，load average 200–650）上无法给出可信扩展比的原因：进程调度抖动直接进入 `T(P)`，
使 `S(P)` 完全被噪声支配（run2 里 P=2 的 143 s 与本次 P=2 的 162 s 也是同一现象）。

### 判据汇总


| 判据 | 实测 | 门槛 | 实测点 | 结论 |
|---|---|---|---|---|
| C3_parallel_vs_serial_l2 | 8.83e-15 | 1.00e-10 | - | pass |
| C5_iteration_reduction | 0.838906 | 0.500000 | - | pass |
| C6_load_balance | 0.110812 | 0.100000 | P=4 | FAIL |
| C1_strong_efficiency | not measured | 0.600000 | P=16 | FAIL (未测量) |
| C2_weak_ratio | not measured | 1.500000 | P=32 | FAIL (未测量) |
| C4_comm_fraction | not measured | 0.300000 | P=16 | FAIL (未测量) |

![强扩展](figures/strong_scaling.png)
![弱扩展](figures/weak_scaling.png)
![通信开销](figures/communication.png)
![迭代数](figures/iterations.png)
![收敛](figures/accuracy.png)
![负载均衡](figures/load_balance.png)
![场量](figures/fields.png)

## ④ 结果图说明

* `strong_scaling.png`：S(P)、E(P) 与理想线；同时给出仅求解与“组装+setup+求解”两种口径。
* `weak_scaling.png`：T(P)/T(1) 随 P 变化，理想值 1，验收线 1.5。
* `communication.png`：T_comm/T_solve 随 P 变化与 T_solve、T_comm 绝对值。
* `iterations.png`：CG / PCG-Jacobi / PCG-ASM 迭代数。
* `accuracy.png`：Q1 相对 L2 误差随 h 的双对数曲线与 h² 参考斜率。
* `load_balance.png`：各秩本地组装/本地矩阵向量积耗时（负载均衡判据的数据源）。
* `fields.png`：P 并行解场云图、解析解场云图与逐点误差图（制造解算例）。

## ⑤ 注意点（踩过的坑 / 局限）

1. **正弦制造解是退化的**：均匀网格 Q1 九点格式对 $\sin\pi x\sin\pi y$ 是精确特征向量，
   CG 一步收敛（残差机器精度）。加多项式项后 Krylov 维数恢复正常，否则一切迭代数对比无效。
2. **均匀网格上 Jacobi 预条件对迭代数无效**：对角为常数时 PCG 与 CG 的 Krylov 子空间相同，
   迭代数完全一致（实测相等）。因此“PCG 迭代数下降 ≥50%”只能用 ASM 这类真正改变谱分布的
   预条件满足，报告同时给出两者，不掩饰 Jacobi 的零增益。
3. **Dirichlet 消元的实现方式影响 1–2 个数量级**：`lil` 逐行赋值在 1.7×10⁴ DOF 时已达 1.1 s，
   换成 CSR 数据数组上的向量化 `eliminate_dirichlet` 后降到毫秒级。
4. **角点编号约定**必须与单元矩阵的参考单元序一致（`itertools.product` 末维最快）；
   不一致时刚度阵仍对称、仍收敛，但连通性错误——通过独立参考组装逐元素比对才能发现。
5. **halo 计划的双侧一致性**：发送侧按“自有节点与对端块 Chebyshev 距离 1”枚举，
   接收侧按“盒内非自有节点”枚举，两侧都按全局 id 排序；只按轴向邻接枚举会漏掉对角角点。
6. **传输层语义 vs 真实 MPI**：本机无 MPI 运行时（无 `mpi4py`/`mpirun`），批量测量使用
   `SharedMemoryComm`（多进程 + 共享内存 arena）。它给出真实的进程级并行与真实 memcpy +
   barrier 代价，但**跨节点带宽/延迟模型无法在本机复现**；`MPIComm` 代码路径已实现并做后端
   自动探测，但**未在本机实测**（如实声明，不作为本节点证据）。
7. **共享宿主机负载**：测量宿主 load average 长期 >200（128 核），单进程调度抖动会直接放大
   barrier 等待，故通信占比与扩展效率需在专用节点上复测；JSON 中保留了每次运行的
   `t_solve/t_comm/per_rank` 供复核。
8. **ASM 的规模上限**：块直接解法在每秩数万 DOF 以上填充爆炸，代码内置 `limit`
   （默认 4×10⁵ DOF/秩）并显式抛错，避免“悄悄换算法”。
9. **集群算力未到位（未完成部分，如实申报）**：向 外协方双节点集群递交的三个作业
   （cj308 全量判据、cj313 P=1/2 点、cj314 P=4/8 点）在报告定稿时仍为 `queued`
   （已排队 40 min 以上，`queued_ahead>100`），因此 **P=16/P=32 的扩展点没有任何集群数据**。
   本报告中的扩展数字全部来自这台共享宿主机（128 核，load average 200–650），
   仅作“代码路径可跑通 + 单点性能量级”的证据，不构成扩展性结论。
   补测入口已就绪：`scripts/fem_parallel_point.py` 可按秩数分片跑点，
   `python -m tensorphysics.fem_parallel.verification` 一次性跑全量判据。

## ⑥ 误差判定

* 并行解 vs 串行解：三个算例（Poisson / 变导热 / 线弹性）最大相对 L2 差
  **8.83e-15**，
  门槛 1e-10 → **通过**。
* 制造解 vs 解析解：最细网格相对 L2 误差
  **5.02e-05**，
  观测收敛阶 **2.00**
  （理论 2）。**该量与解析解对比的误差远小于 3%**（见上表），但它是离散误差而非求解误差：
  求解误差由“并行 vs 串行 1e-10 量级”表征。
* 线弹性常应变 patch test：Q1 可精确再现线性场，直接解误差见 `test_elasticity_patch_test_is_exact`
  （<1e-10，机器精度量级）→ **<3% 通过**。
* 判据 C1–C6 是否满足，见上表“判据汇总”（缺失项标注“未测量”并按 FAIL 记），
  **未通过项如实标注，不粉饰**。

## 复现

```bash
export PYTHONPATH=src
python -m tensorphysics.fem_parallel.verification --n-dofs 1000000 --n-per-rank 100000 \
    --ranks 1,2,4,8,16,32 --out artifacts/fem_parallel/verification.json
python -m pytest tests/test_fem_parallel_*.py -q
python scripts/fem_parallel_figures.py
python scripts/fem_parallel_report.py
```
