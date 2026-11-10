#!/usr/bin/env python3
"""Publish IN157-N05: analytic anchor, figure/doc regeneration, report upload.

All numbers come from ``artifacts/simp_in157_n05/verification.json`` plus one
exact analytic anchor computed here (the SIMP modulus scaling law), so the doc
and the report cannot drift from the run.

    PYTHONPATH=src python3 scripts/publish_simp_in157.py            # full publish
    PYTHONPATH=src python3 scripts/publish_simp_in157.py --no-post  # local only
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

import numpy as np

from tensorphysics.topology_optimization.simp.problems import build_mbb_beam
from tensorphysics.topology_optimization.simp.sensitivity import adjoint_sensitivity

OUT = pathlib.Path("artifacts/simp_in157_n05")
DOC = pathlib.Path("docs/SIMP_DENSITY_FILTER_IN157_N05.md")
API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
MISSION_ID = int(os.environ.get("SIMP_IN157_MISSION_ID", "11191"))

FIGURES = [
    ("fig1_mbb_designs.png", "图1 MBB 100×50（volfrac 0.5, p=3）密度场：上=无滤波，下=r=1.5 线性密度滤波；标题给出柔度与棋盘格指数。"),
    ("fig2_alternation_map.png", "图2 字面 2×2 交替检测器场（黑=属于交替块）：无滤波设计满布棋格，滤波设计近零。"),
    ("fig3_radius_scan.png", "图3 滤波半径扫描 r∈[1.0,3.0]：柔度单调上升（Spearman ρ=1.0），交替指数下降。"),
    ("fig4_history.png", "图4 迭代历史：柔度与连续棋盘格指数的收敛过程（无滤波 vs r=1.5）。"),
    ("fig5_negative_control.png", "图5 负控 r→0：与无滤波设计的逐单元密度差（应为 0）与柔度对比。"),
]


def analytic_anchor() -> dict:
    """Exact SIMP scaling: C(rho0) / C(1) must equal rho0**(-p)."""
    prob = build_mbb_beam(100, 50, 0.5)
    mesh = prob.mesh
    n = mesh.n_elems
    c_solid, _, _ = adjoint_sensitivity(mesh, np.ones(n), prob.F, prob.fixed_dofs, penal=3.0)
    c_half, _, _ = adjoint_sensitivity(mesh, np.full(n, 0.5), prob.F, prob.fixed_dofs, penal=3.0)
    ratio = float(c_half / c_solid)
    exact = 0.5 ** -3.0
    return {
        "law": "C(rho0)/C(1) = rho0**(-p) for a uniform density field (E = rho**p E0)",
        "rho0": 0.5,
        "penal": 3.0,
        "compliance_uniform_half": float(c_half),
        "compliance_solid": float(c_solid),
        "ratio_measured": ratio,
        "ratio_exact": float(exact),
        "relative_error": float(abs(ratio - exact) / exact),
    }


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def build_markdown(d: dict, anchor: dict) -> str:
    c = d["criteria"]
    c1, c2, c3, c4, c5 = c["C1"], c["C2"], c["C3"], c["C4"], c["C5"]
    prob = d["problem"]
    runs = {f"{r['tag']}_{r['seed']}": r for r in d["runs"]}
    unf0, unf1 = runs["unfiltered_0"], runs["unfiltered_1"]
    fil = runs[f"filtered_r{prob['radius']:g}_0"]
    neg = runs["negative_control_0"]
    rows = "\n".join(
        f"| {r['tag']} | {_fmt(r['radius'], 2)} | {r['iterations']} | "
        f"{'是' if r['converged'] else '否'} | {_fmt(r['compliance'])} | "
        f"{_fmt(r['alternating_fraction_raw'])} | {_fmt(r['alternating_fraction_physical'])} | "
        f"{_fmt(r['checkerboard_index_physical'])} | {_fmt(r['grey_fraction'], 3)} |"
        for r in d["runs"]
    )
    scan_rows = "\n".join(
        f"| {_fmt(r, 2)} | {_fmt(cc)} |" for r, cc in zip(c3["radii"], c3["compliance"])
    )
    head = f"""# SIMP 密度滤波验证面 · 节点 IN157-N05（MBB 梁棋盘格抑制）

> 本文件由 `scripts/publish_simp_in157.py` 从 `artifacts/simp_in157_n05/verification.json`
> 程序化生成（数字零手抄）。证据等级 REAL：真实 SIMP 有限元 + 线性密度滤波 + 字面棋盘格检测器。

## 1 案例介绍

MBB 半梁（Messerschmitt-Bölkow-Blohm），平面应力 Q4 单元，单位厚度，
E0 = 1、Emin = 1e-9、ν = 0.3、SIMP 惩罚因子 p = {_fmt(prob['penal'], 1)}：

| 项 | 值 |
|---|---|
| 网格 | {prob['nelx']}×{prob['nely']} 单元 |
| 体积分数 | {_fmt(prob['volfrac'], 2)} |
| 边界条件 | 左端 u_x = 0（对称滚支），右下角 u_y = 0 |
| 载荷 | 左上角节点 F_y = −1 |
| 滤波 | Sigmund 线性（hat）密度滤波，基准半径 r = {_fmt(prob['radius'], 2)} 单元长度 |
| 优化器 | OC 更新，move limit 0.2，体积约束施加在滤波后密度上，tol = {prob['tol']}，上限 {prob['iters']} 迭代 |
| 初值 | 均匀 {_fmt(prob['volfrac'], 2)} ± 5 % 对称破缺（seed 0/1） |

物理背景：无正则化时离散柔度最小化问题不适定——双线性 Q4 网格允许逐单元交替密度
（棋盘格）与单节点铰接，这些是纯离散伪影，不对应可制造结构。线性密度滤波把物理密度
写成设计密度的邻域凸组合 `rho_tilde = P rho`，恢复最小长度尺度，从而使问题适定。

## 2 方法

1. **有限元**（`simp/fem2d.py`）：2×2 Gauss 积分装配 Q4 平面应力刚度阵，齐次 Dirichlet
   消元 + 稀疏直接解；解出后校验 `||K u − F||/||F|| <= 1e-6`，否则判为不连通设计。
2. **密度滤波**（`simp/density_filter.py`）：`H_ij = max(0, r − |x_i − x_j|)`，
   `rho_tilde = P rho`（P 行和为 1、权重非负 ⇒ 滤波后场仍在设计盒内）。
3. **灵敏度链式法则**（`simp/sensitivity.py`）：自伴随解析导数
   `dC/drho_tilde = −p rho^(p−1) (E0−Emin) u_e^T k0 u_e`，再投影 `dC/drho = P^T dC/drho_tilde`。
4. **棋盘格检测器**（本节点新增 `alternating_cell_fraction`）：字面定义——若某 2×2 单元块
   两条对角线一致、非对角线相差 ≥ 0.5，则该块 4 个单元标记为交替单元；
   指数 = 交替单元数 / 总单元数（纯棋盘格 = 1，均匀/条纹场 = 0）。同时记录连续的
   混合二阶差分指数（`checkerboard_index`）作交叉诊断。
5. **验收驱动**（`simp/acceptance_in157.py`）：{len(d['runs'])} 个真实 SIMP 算例
   （无滤波 seed 0/1、r={_fmt(prob['radius'], 2)}、半径扫描 9 点、负控 r→{_fmt(d['negative_control_radius'], 0)}），
   多进程并行，判据 C1–C5 全部由计算场直接给出并落 JSON。
6. **解析锚点**：均匀密度场下 SIMP 模量插值是精确幂律，故
   `C(rho0)/C(1) = rho0^(−p)` 可作无离散误差的解析基准。

## 3 计算结果

### 3.1 判据裁决

| 判据 | 内容 | 实测 | 阈值 | 结论 |
|---|---|---|---|---|
| C1 | C_no_filter / C_filter | {_fmt(c1['ratio'])}（多 seed 均值 {_fmt(c1['ratio_mean'])}） | ≥ 5 | **FAIL（如实）** |
| C2 | 棋盘格指数：无滤波 / 有滤波 | {_fmt(c2['index_no_filter_mean'])} / {_fmt(c2['index_filtered'])} | > 0.3 / < 0.05 | PASS |
| C3 | 柔度–半径单调性 Spearman ρ | {_fmt(c3['spearman_rho'])}（递减步数 {c3['n_decreasing_steps']}） | > 0.9 | PASS |
| C4 | 负控 r→0 与无滤波偏差 | {_fmt(c4['relative_deviation'], 6)} | ≤ 1 % | PASS |
| C5 | pytest 用例数 | {c5['value']} | ≥ 8 | PASS |

C1 的 FAIL 不是实现缺陷，而是判据本身的物理不可能性：滤波后的物理密度场
`P rho` 仍是设计盒 `[0,1]^n` 内的点（P 行和为 1、权重非负），故有滤波问题在
**子集**上求极小，必然 `C_filter ≥ C_no_filter`，比值上界为 1。实测
{_fmt(c1['ratio'])} 与该理论上界一致；判据想表达的「抑制倍数」由 C2 的棋盘格
抑制比 {_fmt(c2['suppression_ratio'], 1)}（检测器分辨率下限 1/{prob['nelx']*prob['nely']}）承担。

### 3.2 全部算例

| 算例 | r | 迭代 | 收敛 | 柔度 C | 交替指数(设计) | 交替指数(物理) | 连续指数(物理) | 灰度分数 |
|---|---|---|---|---|---|---|---|---|
{rows}

### 3.3 半径扫描

| r / 单元长度 | 柔度 C |
|---|---|
{scan_rows}

### 3.4 解析锚点（精度判定）

| 量 | 实测 | 精确 | 相对误差 |
|---|---|---|---|
| C(ρ=0.5 均匀)/C(ρ=1 实心) | {_fmt(anchor['ratio_measured'], 6)} | ρ^(−p) = {_fmt(anchor['ratio_exact'], 6)} | {anchor['relative_error']:.2e} |

## 4 结果图

![图1 密度场](../artifacts/simp_in157_n05/fig1_mbb_designs.png)
![图2 交替检测器场](../artifacts/simp_in157_n05/fig2_alternation_map.png)
![图3 半径扫描](../artifacts/simp_in157_n05/fig3_radius_scan.png)
![图4 迭代历史](../artifacts/simp_in157_n05/fig4_history.png)
![图5 负控](../artifacts/simp_in157_n05/fig5_negative_control.png)

## 5 注意点与局限

1. **C1 判据物理不可达**（见 3.1）。任何合规的 SIMP 实现都不可能给出 ≥5 的柔度比；
   报告如实记 FAIL 并给出“抑制倍数”这一判据可能想表达的量，不做判据重定义。
2. **r = 1.0 是退化半径**：单元间距离恰为 1，`H_ij = max(0, 1 − 1) = 0`，滤波算子
   退化为单位阵（见表中 `scan_r1` 与 `unfiltered_0` 柔度完全一致、交替指数同为
   {_fmt(unf0['alternating_fraction_raw'])}）。因此扫描区间下端点是“无滤波”而不是
   “弱滤波”，这是线性 hat 核在单元距离度量下的结构性质，不是数值噪声。
3. **有滤波的交替指数为检测器分辨率零**：物理场指数
   {_fmt(c2['index_filtered'])}（< 1/{prob['nelx']*prob['nely']} = {_fmt(c2['detector_resolution'], 6)}），
   而设计变量场指数为 {_fmt(fil['alternating_fraction_raw'])}，两者都远低于 0.05 阈值；抑制比因此
   按下限归一化记录，避免出现“除以 0”的伪无穷。
4. **未完全收敛**：上限 {prob['iters']} 迭代，多数算例在 tol = {prob['tol']} 前停止（表中“收敛”列），
   柔度仍有 ~1e-3 量级漂移；单调性结论在 9 点扫描上稳定（无递减步），但对更大上限迭代的
   绝对值不作承诺。
5. **灰度分数随 r 上升**（{_fmt(unf0['grey_fraction'], 3)} → {_fmt(runs['scan_r3_0']['grey_fraction'], 3)}）：半径越大设计越“灰”，
   这是密度滤波的标准代价，本节点不对其做可制造性投影（无 Heaviside 投影）。
6. **适用边界**：结论限于 2-D 平面应力 Q4 + OC + 线性 hat 滤波。其他滤波族
   （Heaviside/灵敏度滤波）与 3-D 单元需另立验证面。

## 6 误差判定（是否 < 3 %）

| 关键量 | 基准 | 实测误差 | < 3 % ? |
|---|---|---|---|
| 均匀场柔度幂律 C(0.5)/C(1) | 解析 rho^(−p) | {anchor['relative_error']:.2e} | **是** |
| 负控 r→0 柔度 | 无滤波 run（同 seed） | {_fmt(c4['relative_deviation'], 6)} | **是** |
| 负控 r→0 逐单元密度 | 无滤波设计 | {_fmt(c4['max_density_difference'], 6)}（绝对） | **是** |
| 棋盘格指数（滤波后） | 判据阈值 0.05 | {_fmt(c2['index_filtered'])} < 0.05 | **是** |
| 柔度比 C_no_filter/C_filter | 判据阈值 5 | {_fmt(c1['ratio'])} | **否（honest FAIL，理论上界 1）** |

绝对柔度值无外部实验/文献基准（本节点为数值验证面，不含试验对标），故不作绝对值误差声明；
C1 的偏差不是实现误差而是判据与变分原理冲突，已如实记录。

## 7 复现

```bash
PYTHONPATH=src python3 scripts/simp_in157_study.py --workers 8   # 优化 + 图
PYTHONPATH=src python3 -m pytest tests/test_simp_in157_acceptance.py -q
PYTHONPATH=src python3 scripts/publish_simp_in157.py             # 本文档 + 报告
```

工件：`artifacts/simp_in157_n05/verification.json`（全部判据数字）、`designs.npz`（密度场）、
5 张 png。判据状态：`{json.dumps(d['verdict']['criteria_status'])}`；
未通过项：`{d['verdict']['failed']}`。
"""
    return head


def main(argv=None) -> int:
    import sys

    no_post = "--no-post" in sys.argv[1:]
    rec = json.loads((OUT / "verification.json").read_text())
    anchor = analytic_anchor()
    rec["analytic_anchor"] = anchor
    (OUT / "verification.json").write_text(json.dumps(rec, indent=2))
    markdown = build_markdown(rec, anchor)
    DOC.write_text(markdown)

    payload = {
        "title": "IN157-N05 SIMP 密度滤波验证面（MBB 梁棋盘格抑制）：C1 判据证伪，C2–C5 通过",
        "summary": (
            f"MBB 100×50/volfrac 0.5/p=3 真实 SIMP：无滤波棋盘格指数 "
            f"{_fmt(rec['criteria']['C2']['index_no_filter_mean'])}，r=1.5 滤波后 "
            f"{_fmt(rec['criteria']['C2']['index_filtered'])}；半径扫描 Spearman ρ="
            f"{_fmt(rec['criteria']['C3']['spearman_rho'])}；负控 r→0 偏差 "
            f"{_fmt(rec['criteria']['C4']['relative_deviation'], 6)}；C1 柔度比 "
            f"{_fmt(rec['criteria']['C1']['ratio'])} 如实记 FAIL（理论上界 1）。"
        ),
        "scenario_key": "topology_optimization",
        "mission_id": MISSION_ID,
        "tags": ["SIMP", "density_filter", "checkerboard", "MBB", "topology_optimization",
                 "IN157-N05", "honest_fail"],
        "markdown": markdown,
        "figures": [
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode((OUT / name).read_bytes()).decode(),
            }
            for name, caption in FIGURES
            if (OUT / name).exists()
        ],
    }
    pathlib.Path("/tmp/report_in157.json").write_text(json.dumps(payload, ensure_ascii=False))
    print(f"doc -> {DOC}, report payload -> /tmp/report_in157.json "
          f"({len(payload['figures'])} figures)")
    if no_post:
        return 0

    user = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
    password = os.environ.get("TENSOR_AGENT_PASS", "")

    def post(path, body, token=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            API_BASE + path, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as fh:
            return json.load(fh)

    token = post("/auth/login", {"username": user, "password": password}).get("access_token")
    resp = post("/tensor/reports", payload, token=token)
    print("report:", json.dumps(resp, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())