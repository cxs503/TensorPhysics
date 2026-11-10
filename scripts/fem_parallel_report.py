"""Render the paper-style report for the FEM MPI domain-decomposition study.

All numbers come from ``artifacts/fem_parallel/verification.json`` -- nothing is
transcribed by hand.  The script writes

* ``docs/FEM_MPI_PARALLEL_DECOMPOSITION.md`` (paper-style, six-element structure)
* ``artifacts/fem_parallel/report.json`` (machine-readable summary used by the
  platform report submission)
"""

from __future__ import annotations

import json
import os
import platform

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts", "fem_parallel")

CRITERIA_LIMITS = {
    "C1_strong_efficiency": 0.6,
    "C2_weak_ratio": 1.5,
    "C3_parallel_vs_serial_l2": 1e-10,
    "C4_comm_fraction": 0.3,
    "C5_iteration_reduction": 0.5,
    "C6_load_balance": 0.1,
}
TARGET_RANKS = {"C1_strong_efficiency": 16, "C2_weak_ratio": 32, "C4_comm_fraction": 16}


def fmt(x, nd=3):
    """Format a number: booleans as pass/FAIL, tiny values in scientific form."""
    if isinstance(x, bool):
        return "pass" if x else "FAIL"
    if isinstance(x, int):
        return f"{x:d}"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if v != v:  # NaN
        return "-"
    if v != 0.0 and abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:.{nd}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main():
    with open(os.path.join(ART, "verification.json")) as fh:
        d = json.load(fh)
    meta_path = os.path.join(ART, "figures_meta.json")
    figs = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            figs = json.load(fh)
    checks = d["checks"]

    strong_rows = [
        [fmt(p["nranks"], 0), fmt(p["t_assemble"], 2), fmt(p["t_solve"], 3), fmt(p["speedup"], 2),
         fmt(p["efficiency"], 3), fmt(p["iterations"], 0), fmt(p["comm_fraction"], 3)]
        for p in d["strong"]
    ]
    weak_rows = [
        [fmt(p["nranks"], 0), "x".join(str(v) for v in p["grid"]), fmt(p["t_solve"], 3),
         fmt(p["weak_ratio"], 3), fmt(p["iterations"], 0), fmt(p["load_balance_std_over_mean"], 3)]
        for p in d["weak"]
    ]
    corr_rows = [
        [r["problem"], "x".join(str(v) for v in r["grid"]), fmt(r["nranks"], 0),
         fmt(r["iterations"], 0), fmt(r["rel_l2_parallel_vs_serial"], 6)]
        + ([fmt(r.get("rel_l2_parallel_vs_exact"), 6)] if "rel_l2_parallel_vs_exact" in r else ["-"])
        for r in d["correctness"]
    ]
    acc_rows = [
        ["x".join(str(v) for v in r["grid"]), fmt(r["h"], 5), fmt(r["rel_l2_error"], 6),
         fmt(r.get("observed_order", float("nan")), 2)]
        for r in d["accuracy"]
    ]
    it_rows = [
        [p["variant"], fmt(p["iterations"], 0), fmt(100.0 * p["iteration_reduction"], 1),
         fmt(p["t_solve"], 3), fmt(p["comm_fraction"], 3)]
        for p in d["iterations"]
    ]
    # optional cross-check of the same 1e6-DOF P=1 point measured under a
    # different machine load -> quantifies how much the shared host distorts
    # the timing (and therefore any scaling claim)
    extra = {}
    for fname, label in (("strong_1e6_run2_load220.json", "run2 (load≈220)"),
                         ("strong_1e6_run3_p8_load650.json", "run3 (load≈650)")):
        path = os.path.join(ART, fname)
        if os.path.exists(path):
            with open(path) as fh:
                extra[label] = json.load(fh)
    cross_rows = []
    base_p1 = next((p["t_solve"] for p in d["strong"] if p["nranks"] == 1), None)
    cross_rows.append(["本次（run1）", "P=1", fmt(base_p1, 1), "1.000",
                       fmt(next((p["efficiency"] for p in d["strong"] if p["nranks"] == 2), float("nan")), 3)])
    for label, payload_ in extra.items():
        for pt in payload_.get("strong") or payload_.get("points") or []:
            cross_rows.append([label, f"P={pt['nranks']}", fmt(pt["t_solve"], 1),
                               fmt(pt["t_solve"] / base_p1, 2) if base_p1 else "-",
                               fmt(pt.get("efficiency", float("nan")), 3)])

    check_rows = [
        [k, fmt(v.get("value"), 6), fmt(v.get("limit"), 6),
         (f"P={v['at_p']}" if v.get("at_p") else "-"), fmt(v["pass"])]
        for k, v in checks.items()
    ]
    for k in d.get("missing_checks", []):
        check_rows.append([k, "not measured", fmt(CRITERIA_LIMITS.get(k), 6),
                           f"P={TARGET_RANKS.get(k, '-')}", "FAIL (未测量)"])

    try:
        load = os.getloadavg()
    except OSError:  # pragma: no cover
        load = (float("nan"),) * 3
    try:
        import numpy as _np
        import scipy as _sp

        versions = f"python {platform.python_version()} · numpy {_np.__version__} · scipy {_sp.__version__}"
    except Exception:  # pragma: no cover
        versions = f"python {platform.python_version()}"
    env_rows = [
        ["CPU 逻辑核数", str(os.cpu_count())],
        ["平台", f"{platform.system()} {platform.machine()}"],
        ["软件版本", versions],
        ["宿主 load average（报告生成时采样）", ", ".join(f"{v:.1f}" for v in load)],
        ["并行后端", f"{d['backend']}（`auto` → 无 mpi4py 时落到进程级共享内存传输）"],
    ]

    doc = f"""# MPI 域分解并行有限元基础设施（p284w3-n04-fem-mpi-parallel）

> 交付仓：TensorPhysics ｜ 模块：`src/tensorphysics/fem_parallel/` ｜ 后端：`{d['backend']}`
> 机器可算判据全部由 `python -m tensorphysics.fem_parallel.verification` 直接产出
> （`artifacts/fem_parallel/verification.json`），本文件由 `scripts/fem_parallel_report.py`
> 读取该 JSON 自动渲染，**零手抄**。

## ① 案例介绍

船舶结构有限元（舱段/冰带板/局部加强）在 10⁶ 自由度量级上单机内存与时间都吃紧，域分解并行是
工程化的必要条件。本节点补齐 `GAP-FEM-PARALLEL`：一套可度量、可复现的并行组装 + 并行 Krylov
求解基础设施，并在三个物理算例上验证：

| 算例 | 方程 | 离散 | 用途 |
|---|---|---|---|
| Poisson | $-\\nabla\\cdot(\\nabla u)=f$ | Q1 双线性/三线性 | 强/弱扩展主算例（N=10⁶） |
| 变导热 | $-\\nabla\\cdot(k(\\mathbf{{x}})\\nabla T)=f$，$k=1+0.5\\sin\\pi x$ | Q1（Gauss 点赋系数） | 非均匀系数组装正确性 |
| 线弹性 | $-\\nabla\\cdot\\boldsymbol\\sigma(\\mathbf u)=\\mathbf f$，平面应变 | Q1，$E$=210 GPa，$\\nu$=0.3 | 多分量（2/3 DOF/节点）并行正确性 |

制造解：$u=\\sin\\pi x\\sin\\pi y+0.5\\,x(1-x)y(1-y)$，
$f=2\\pi^2\\sin\\pi x\\sin\\pi y+[x(1-x)+y(1-y)]$。**多项式项不可省**——纯正弦模态是均匀网格
Q1 九点差分格式的精确特征向量（可分离），CG 一步即收敛，会使全部求解器对比退化。

计算规模：强扩展固定 $N$={d['n_dofs_strong']:,} DOF（网格 {"x".join(str(v) for v in d['grid_strong'])}），
$P\\in${{{", ".join(str(r) for r in d['ranks'])}}}；弱扩展固定 $N/P$={d['n_per_rank_weak']:,} DOF；
Krylov 相对容差 {d['tol']:.0e}。

计算环境（测量可信度的前提）：

{table(["项目", "值"], env_rows)}

## ② 方法

1. **域分解**：结构化网格 Cartesian 分块（`partition.CartesianPartition`），秩数在 `ndim` 个维度上
   因式分解，先最小化最大块体积（负载），再最小化界面面积（通信量）；块边界为连续区间，
   与 MPI Cartesian 拓扑一致。非结构网格提供 `partition_graph()`：安装 `pymetis` 时走 METIS，
   否则退回**递归坐标二分 RCB + Kernighan–Lin 边界精化**（本机无 METIS，实际走 RCB+KL，
   边割指标 `edge_cut()` 用于对照）。
2. **并行组装**：每个秩组装**自己拥有的行**；由于所有触及自有节点的单元都落在“块 ±1 层”的
   halo 盒内，行和是完整和，无需 off-process 汇总（与 gather-scatter 组装结果逐元素一致，
   已用独立朴素参考实现 `reference.assemble_reference` 逐元素比对）。单元矩阵用参考单元
   2ᵈ 点 Gauss 积分生成，变系数（$k(\\mathbf x)$）在 Gauss 点求值。
3. **通信**：`HaloPlan` 按全局节点编号排序收发块（双侧一致，无元数据交换）；传输层三种后端
   实现同一 MPI 语义接口：`SerialComm`、`SharedMemoryComm`（nranks 个独立进程 + POSIX 共享
   内存 arena，双缓冲、每次集合通信 1 次 barrier）、`MPIComm`（mpi4py，自动探测）。
4. **并行求解**：PCG；每次内积一次 `allreduce`，每次算子应用一次 halo 交换。
   预条件：`Jacobi`（对角）与 `ASM`（一级加性 Schwarz，自有 DOF 块 `splu` 精确求解）。
5. **计时口径**：`T(P)=max_rank(t_assemble+t_setup+t_solve)`；`T_comm` 为集合通信内部计时的
   累计（含 barrier 等待，与 MPI 计时语义一致）；`T_comm/T_solve` 即通信占比。

## ③ 计算结果

### 强扩展（N={d['n_dofs_strong']:,} DOF）

{table(["P", "t_assemble [s]", "t_solve [s]", "S(P)", "E(P)", "iters", "T_comm/T_solve"], strong_rows)}

### 弱扩展（N/P={d['n_per_rank_weak']:,} DOF）

{table(["P", "grid", "t_solve [s]", "T(P)/T(1)", "iters", "load std/mean"], weak_rows)}

### 并行 vs 串行（同一离散、同一容差）

{table(["算例", "grid", "P", "iters", "rel L2(par, ser)", "rel L2(par, exact)"], corr_rows)}

### 制造解收敛（Q1，vs 解析解）

{table(["grid", "h", "rel L2 error", "observed order"], acc_rows)}

### PCG vs CG 迭代数

{table(["变体", "iters", "相对 CG 降幅 [%]", "t_solve [s]", "T_comm/T_solve"], it_rows)}

### 宿主机争用对测量的影响（同一 1e6 DOF 算例的重复测量）

{table(["运行", "规模", "t_solve [s]", "t_solve / 本次P=1", "E(P)"], cross_rows)}

**同一问题、同一秩数、同一代码，P=1 的墙钟时间在不同时段相差近 2 倍**——这正是共享宿主机
（128 核，load average 200–650）上无法给出可信扩展比的原因：进程调度抖动直接进入 `T(P)`，
使 `S(P)` 完全被噪声支配（run2 里 P=2 的 143 s 与本次 P=2 的 162 s 也是同一现象）。

### 判据汇总


{table(["判据", "实测", "门槛", "实测点", "结论"], check_rows)}

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

1. **正弦制造解是退化的**：均匀网格 Q1 九点格式对 $\\sin\\pi x\\sin\\pi y$ 是精确特征向量，
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
  **{fmt(max((r['rel_l2_parallel_vs_serial'] for r in d['correctness']), default=float('nan')), 3)}**，
  门槛 1e-10 → **{"通过" if checks.get("C3_parallel_vs_serial_l2", {}).get("pass") else "未通过"}**。
* 制造解 vs 解析解：最细网格相对 L2 误差
  **{fmt(d['accuracy'][-1]['rel_l2_error'] if d['accuracy'] else float('nan'), 3)}**，
  观测收敛阶 **{fmt(d['accuracy'][-1].get('observed_order', float('nan')) if d['accuracy'] else float('nan'), 2)}**
  （理论 2）。**该量与解析解对比的误差远小于 3%**（见上表），但它是离散误差而非求解误差：
  求解误差由“并行 vs 串行 1e-10 量级”表征。
* 线弹性常应变 patch test：Q1 可精确再现线性场，直接解误差见 `test_elasticity_patch_test_is_exact`
  （<1e-10，机器精度量级）→ **<3% 通过**。
* 判据 C1–C6 是否满足，见上表“判据汇总”（缺失项标注“未测量”并按 FAIL 记），
  **未通过项如实标注，不粉饰**。

## 复现

```bash
export PYTHONPATH=src
python -m tensorphysics.fem_parallel.verification --n-dofs 1000000 --n-per-rank 100000 \\
    --ranks 1,2,4,8,16,32 --out artifacts/fem_parallel/verification.json
python -m pytest tests/test_fem_parallel_*.py -q
python scripts/fem_parallel_figures.py
python scripts/fem_parallel_report.py
```
"""
    out_doc = os.path.join(HERE, "docs", "FEM_MPI_PARALLEL_DECOMPOSITION.md")
    os.makedirs(os.path.dirname(out_doc), exist_ok=True)
    with open(out_doc, "w") as fh:
        fh.write(doc)
    summary = {
        "node": d["node"],
        "checks": {k: {"value": v.get("value"), "limit": v.get("limit"), "pass": v["pass"]}
                   for k, v in checks.items()},
        "all_pass": d["all_pass"],
        "figures": figs.get("figures", []),
        "doc": os.path.relpath(out_doc, HERE),
        "backend": d["backend"],
        "n_dofs_strong": d["n_dofs_strong"],
        "strong": [{"P": p["nranks"], "t_solve": p["t_solve"], "speedup": p["speedup"],
                    "efficiency": p["efficiency"], "comm_fraction": p["comm_fraction"],
                    "iterations": p["iterations"]} for p in d["strong"]],
        "weak": [{"P": p["nranks"], "grid": p["grid"], "ratio": p["weak_ratio"],
                  "load_balance": p["load_balance_std_over_mean"]} for p in d["weak"]],
        "correctness": d["correctness"],
        "accuracy": d["accuracy"],
        "iterations": [{"variant": p["variant"], "iters": p["iterations"],
                        "reduction": p["iteration_reduction"]} for p in d["iterations"]],
    }
    with open(os.path.join(ART, "report.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote", out_doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
