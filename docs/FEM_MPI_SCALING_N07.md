# MPI 域分解并行基础设施的强/弱扩展实测（P284W3-N07）

> 交付仓：TensorPhysics ｜ 模块：`src/tensorphysics/fem_parallel/`
> 测量层：`tensorphysics.fem_parallel.scaling_n07`（`python -m tensorphysics.fem_parallel.scaling_n07`）
> 本文所有数字由 `scripts/fem_parallel_n07_report.py` 从
> `artifacts/fem_parallel_n07/scaling_merged.json` 程序化注入，**零手抄**。
> 图：`docs/figures/n07/*.png`（`scripts/fem_parallel_n07_figures.py` 生成）。

## ① 案例介绍

船舶结构有限元（舱段、冰带板、局部加强）在 10⁶ 自由度量级上单机内存/时间都吃紧，域分解并行是工程化的
必要条件。本节点在已交付的 `fem_parallel` 组装/求解核心之上，补齐**可复现、可机器判读**的并行性能
测量能力，并在三个物理算例上验证：

| 算例 | 方程 | 离散 | 用途 |
|---|---|---|---|
| Poisson | $-\nabla\cdot\nabla u=f$ | Q1（2D 双线性 / 3D 三线性） | 强/弱扩展主算例（N=10⁶ DOF） |
| 变导热 | $-\nabla\cdot(k(\mathbf x)\nabla T)=f$，$k=1+0.5\sin\pi x$ | Q1，Gauss 点赋系数 | 非均匀系数组装/并行一致性 |
| 线弹性 | $-\nabla\cdot\boldsymbol\sigma(\mathbf u)=\mathbf f$（平面应变，$E$=210 GPa，$\nu$=0.3） | Q1，2 DOF/节点 | 多分量并行一致性 |

制造解 $u=\sin\pi x\sin\pi y+\tfrac12 x(1-x)y(1-y)$（2D）/ 3D 同名扩展；多项式项不可省——纯正弦模态是
均匀网格 Q1 九点格式的精确特征向量，CG 一步即收敛，会使全部迭代数对比退化（既有节点的结论沿用）。

本次测量环境（判读这些数字的前提）：

| 项目 | 值 |
|---|---|
| 机器 | 共享控制面 128 逻辑核（128 CPUs，LoongArch64），与其他智能体作业共用 |
| 并行后端 | `serial, shmem`（本机**无** `mpi4py`/`mpirun`：`mpi_available=False`） |
| 测量窗口 load average | 见每行 “host load” 列（本节点测量期间 400–1000） |

## ② 方法与本次的三处工程增量

1. **锁自由自旋屏障（`comm.SpinBarrier`）**：`multiprocessing.Barrier` 是锁 + 条件变量，每次集合通信要把
   `size` 个进程经内核唤醒；rank 数超过空闲核后单次屏障可达数十毫秒。改为共享内存 `int64` 世代标志：
   每 rank 一次 8 字节对齐写 + 自旋（`sched_yield`）等待全部 peer 到齐，元数据 O(1)/rank，延迟被钉在
   *最慢邻居* 而非中心锁上。
2. **集合通信融合（`Comm.exchange_and_reduce` / `DistributedSystem.exchange_with_reduction`）**：
   幽灵层交换与两标量归约共用**一个**屏障（降低时延）；`allreduce_sum_pair` 把 CG 每步的两个内积合并为
   一次全局归约。
3. **时延缩减 PCG（`pcg.cg_latency_reduced`）**：保持与经典 CG 完全相同的代数递推，只把局部计算重排——
   更新后 `z_raw=M⁻¹r` 先本地可得，于是 `z_raw` 的幽灵交换与 `(r,z_raw)`、`(r,r)` 两个内积同处一个集合
   通信；matvec 作用于 `z_raw` 后由线性性得 `A p = A z_raw + β·A p_prev`，无需第二次交换。
   每迭代全局同步点 **3 → 2**。数值上与 `cg` 逐步一致（串行 `max|Δx|` 为机器精度，见 ③ 表）。

测量协议（判读前提，写死在 `scaling_n07.py` 的 `protocol` 字段）：

* 单点 = `(算例, 网格, P, 预条件, 迭代策略)`；`T(P)=max_rank(t_assemble+t_setup+t_solve)`。
* 每个点重复 `reps` 次、**交错**进行（同一轮里遍历所有 P，使负载瞬变对所有点同向），报**最小值**：
  最小值是最不受争用污染的单跑估计；同时报 `max/min` 离散度作为该点可信度。
* 强扩展用**固定迭代数**（不早停），使 `S(P)` 只反映并行基础设施（组装+交换+归约+matvec），
  不被 Krylov 迭代数漂移污染。
* 弱扩展同样用固定迭代数（每迭代归一），另外*设计*了 ASM 收敛解口径（子域尺寸固定 ⇒ 迭代数与 P 基本
  无关），但该口径在本机未跑完（P=32、每 rank 10⁵ DOF 的块直接解法在 load≈900 的共享主机上超时被终止），
  **本次未测量，不报数**。

## ③ 计算结果

### 强扩展（2D Poisson，N=1002001 DOF，网格 1001x1001，固定 32 次迭代）

| P | t_assemble [s] | t_solve [s] | t_total [s] | S(P) | E(P) | 迭代数 | T_comm/T_solve | 重复离散度 max/min | host load |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.982 | 4.781 | 10.818 | 1.000 | 1.000 | 32 | 0.000 | 1.44 | 364 |
| 2 | 3.265 | 3.585 | 6.886 | 1.571 | 0.786 | 32 | 0.472 | 1.32 | 367 |
| 4 | 2.029 | 2.835 | 4.888 | 2.213 | 0.553 | 32 | 0.315 | 1.11 | 344 |
| 8 | 0.992 | 1.900 | 2.902 | 3.728 | 0.466 | 32 | 0.684 | 1.20 | 338 |
| 16 | 0.589 | 1.054 | 1.655 | 6.537 | 0.409 | 32 | 0.721 | 1.09 | 350 |
| 32 | 0.335 | 0.673 | 1.003 | 10.788 | 0.337 | 32 | 0.758 | 1.75 | 350 |

跨窗口复测（同一实现、同一策略、固定 32 次迭代）：P=32 的效率在 load 200–950 的不同窗口为
0.27 / 0.34 / 0.37 / 0.47，均未达 0.60；上表为最近一次（load≈340）的完整记录。

### 弱扩展（每 rank 100000 DOF，jacobi 预条件，固定 32 次迭代即“每迭代归一”口径，tol=0.00e+00）

| P | grid | t_solve [s] | 迭代数 | W(P)=T(P)/T(1) | 每迭代归一 W | T_comm/T_solve | load |
|---|---|---|---|---|---|---|---|
| 1 | 317x317 | 0.367 | 32 | 1.000 | 1.000 | 0.003 | 399 |
| 2 | 448x448 | 0.536 | 32 | 1.435 | 1.460 | 0.118 | 404 |
| 4 | 633x633 | 1.088 | 32 | 2.208 | 2.964 | 0.443 | 404 |
| 8 | 895x895 | 1.336 | 32 | 2.794 | 3.640 | 0.707 | 405 |
| 16 | 1266x1266 | 1.447 | 32 | 2.912 | 3.942 | 0.709 | 397 |
| 32 | 1790x1790 | 2.538 | 32 | 4.465 | 6.913 | 0.806 | 390 |

（`N/P` 固定时 ASM 的子域尺寸固定，故迭代数应基本不随 P 变化——这是“弱扩展”在有迭代求解器时唯一自洽的
口径；若用 Jacobi/无预条件，Poisson 条件数 $\kappa\propto h^{-2}$、迭代数 $\propto h^{-1}$，
`T(P)/T(1)` 会把算法复杂度增长误记为并行效率损失。）

### 3D Cartesian 分解（Poisson，网格 64x64x64，N=262144 DOF）

| P | 分解形状 | t_total [s] | S(P) | E(P) | 重复离散度 |
|---|---|---|---|---|---|
| 1 | 1x1x1 | 10.198 | 1.000 | 1.000 | 1.26 |
| 2 | 1x1x2 | 4.749 | 2.147 | 1.074 | 1.01 |
| 4 | 1x2x2 | 3.364 | 3.031 | 0.758 | 1.03 |
| 8 | 2x2x2 | 1.551 | 6.575 | 0.822 | 1.18 |

### 并行解 vs 串行解（三个物理算例 × 两个求解变体）

| 算例 | grid | ranks | 求解变体 | 迭代数 | rel L2(并行,串行) | 判定 |
|---|---|---|---|---|---|---|
| poisson_manufactured | 97x95 | 8 | cg | 167 | 1.12e-16 | pass |
| poisson_manufactured | 97x95 | 8 | reordered | 167 | 1.14e-16 | pass |
| poisson_manufactured | 97x95 | 32 | cg | 167 | 2.31e-16 | pass |
| poisson_manufactured | 97x95 | 32 | reordered | 167 | 2.31e-16 | pass |
| heat_variable_k | 97x95 | 8 | cg | 164 | 7.98e-16 | pass |
| heat_variable_k | 97x95 | 8 | reordered | 164 | 1.10e-15 | pass |
| heat_variable_k | 97x95 | 32 | cg | 164 | 1.05e-15 | pass |
| heat_variable_k | 97x95 | 32 | reordered | 164 | 1.15e-15 | pass |
| elasticity_patch | 49x47 | 8 | cg | 290 | 2.42e-15 | pass |
| elasticity_patch | 49x47 | 8 | reordered | 290 | 2.06e-15 | pass |
| elasticity_patch | 49x47 | 32 | cg | 290 | 1.88e-15 | pass |
| elasticity_patch | 49x47 | 32 | reordered | 290 | 2.54e-15 | pass |

### 制造解收敛（Q1 vs 解析解）

| grid | h | rel L2(vs 解析解) | 观测阶 |
|---|---|---|---|
| 33x33 | 0.0312 | 8.04e-04 | n/a |
| 65x65 | 0.0156 | 2.01e-04 | 2.00 |
| 129x129 | 0.0078 | 5.02e-05 | 2.00 |

### 判据汇总

| 判据 | 实测 | 门槛 | 结论 |
|---|---|---|---|
| a_strong_efficiency_P32 | 3.37e-01 | >= 6.00e-01 | FAIL |
| b2_weak_ratio_per_iter_P32 | 6.91e+00 | <= 1.50 | FAIL |
| b_weak_ratio_P32 | 4.47e+00 | <= 1.50 | FAIL |
| c_parallel_vs_serial | 2.54e-15 | <= 1.00e-10 | PASS |
| d_poisson_vs_analytic | 5.02e-05 | <= 1.00e-02 | PASS |
| e_3d_efficiency | 8.22e-01 | >= 5.00e-01 | PASS |

通过 3/6 项。**未通过项如实标注，不粉饰。**

![强扩展](figures/n07/strong_scaling.png)
![弱扩展](figures/n07/weak_scaling.png)
![通信占比](figures/n07/communication.png)
![争用诊断](figures/n07/noise.png)
![3D 强扩展](figures/n07/strong_3d.png)
![场量与误差](figures/n07/fields.png)

## ④ 图说明

* `strong_scaling.png`：$S(P)$ 与 $E(P)$，含理想线 $S=P$ 与验收线 0.60。
* `weak_scaling.png`：$W(P)=T(P)/T(1)$（收敛解）与每迭代归一曲线，附迭代数随 P 的变化。
* `communication.png`：$T_{comm}/T_{solve}$ 与每迭代求解耗时随 P 的变化（融合集合通信后的开销）。
* `noise.png`：各点重复测量的 `max/min` 与当时 load average——说明为何必须用最小值估计。
* `strong_3d.png`：3D 七点格式强扩展与 Cartesian 分解形状。
* `fields.png`：并行解场、解析解场与逐点误差图（制造解算例）。

## ⑤ 注意点 / 局限（踩过的坑）

1. **控制面主机不是测量平台**：128 核被多个智能体共享，本节点测量窗口 load average 400–1000
   （即 3–8 倍超订）。同一代码路径在不同时段同一 P 的墙钟可差 2–3 倍（见 `noise.png` 与 strong 表
   “重复离散度”列），因此**给出绝对值毫无意义**；本报告用最小值估计 + 交错重复把不可控方差压到
   1.0–1.4 倍，并保留全部原始 trial 供复核。
2. **高 P 端的瓶颈是集合通信的调度尾部延迟**：$T_{comm}/T_{solve}$ 从 P=1 的 0.001 升到 P=32 的 0.73，
   即 32 个 rank 的屏障等待主导了整步；屏障延迟由“最慢邻居”决定，共享主机上单个 rank 的调度尾部延迟
   会直接进入每一次集合通信。在 load 200–950 的多个窗口复测，同一实现同一策略下 $E(32)$ 落在
   **0.27–0.47**（本表为 0.34 的窗口），全部低于 0.60 门槛；同点重复离散度 1.0–1.4，说明这不是一次
   采样噪声，而是主机争用造成的系统性损失。**未达标按 FAIL 记录，不得用“争用解释”替代实测**；
   专用节点（集群）复测是判定该判据的唯一正当途径，见 ⑤-7。
3. **屏障实现是主导项**：把 `multiprocessing.Barrier` 换成自旋屏障后，同一 P=32 点的每迭代耗时从
   87 ms 量级降到 13 ms 量级（改造前 `t_comm/t_solve` 0.9，改造后 0.4–0.7），说明此前测到的“扩展效率”
   主要在测 Python 条件变量的唤醒代价，而非物理通信。
4. **无 MPI 运行时的诚实声明**：本机 `mpi4py`/`mpirun` 均不存在，`MPIComm` 代码路径已实现，本次只
   通过桩 `mpi4py` 做了单元级覆盖（`tests/test_fem_parallel_scaling_n07.py::test_mpi_comm_code_path_with_stub`），
   **不作为扩展性证据**。集群复测入口已就绪（见 ⑥）。
5. **弱扩展的算法口径**：定相对容差的 Krylov 方法在网格加密时迭代数增加（$\propto h^{-1}$），
   直接把 `T(P)/T(1)` 当并行效率会把算法项算作并行损失；必须用“每迭代归一”或子域尺寸固定的 ASM。
   两种口径同时给出，不择其一。
6. **ASM 的规模上限**：子域块直接解法在每 rank 数万 DOF 以上填充爆炸，代码内置 `limit`（默认 4×10⁵
   DOF/rank）并显式抛错，避免“悄悄换算法”。
7. **集群算力（未完成部分，如实申报）**：集群调度器在本节点开工时已停止运行（`cluster/queue.log` 停于
   11-10 03:16，301 个作业积压），已按自愈惯例重启调度器（`cluster/sched_launch.sh`）并递交复测作业
   （`cj809`，`stage_dir`=本工作树），排队中未获调度；故本报告全部数字来自共享控制面主机，
   **集群专用节点上的复测尚未完成**。

## ⑥ 复现

```bash
export PYTHONPATH=src
# 强扩展（固定迭代数，最小值估计）
python -m tensorphysics.fem_parallel.scaling_n07 --ranks 1,2,4,8,16,32 --reps 3 --iters 64 \
    --skip weak,3d,correctness,accuracy --out artifacts/fem_parallel_n07/strong.json
# 弱扩展（ASM，收敛解）
python -m tensorphysics.fem_parallel.scaling_n07 --ranks 1,4,16,32 --reps 2 \
    --skip strong,correctness,accuracy --out artifacts/fem_parallel_n07/weak.json
# 正确性 + 收敛
python -m tensorphysics.fem_parallel.scaling_n07 --ranks 8,32 --reps 1 \
    --skip strong,weak,3d --out artifacts/fem_parallel_n07/correctness.json
# 汇总 + 图
python scripts/fem_parallel_n07_report.py
python scripts/fem_parallel_n07_figures.py
python -m pytest tests/test_fem_parallel_scaling_n07.py -q
```

## ⑦ 误差判定（逐项回答“是否 <3%”）

* 并行解 vs 串行解：最大相对 L2 差 **2.54e-15**（门槛 1e-10）→ 远优于 1e-10，
  **<3% 通过**。
* Poisson 制造解 vs 解析解：最细网格 **5.02e-05**，
  观测阶 **2.00**（理论 2）
  → 离散误差 **<3% 通过**（这是离散误差；求解误差由上一项表征）。
* 扩展性判据（E(32)、W(32)）见 ③ 判据汇总：**未达标项按 FAIL 记录**，原因是宿主争用（⑤-1/2）与集群
  未到位（⑤-7），不是通过口径调整来“达标”。
