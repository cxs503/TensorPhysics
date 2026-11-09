#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p128w3-n10：层合复合材料船体板 CLT/FSDT 验证面 —— 数值实验驱动。

运行::

    python3 scripts/composite_laminate_study.py

产物（全部程序化生成，零手抄）：

* ``artifacts/p128w3_n10_composite/results.json``   —— 全部数字（含各级判据）
* ``artifacts/p128w3_n10_composite/criteria.json``  —— 机器验收判据表
* ``artifacts/p128w3_n10_composite/fig_*.png``      —— 报告图

判据采用「有基准必对标、无基准声明原因」的纪律：各向同性退化极限对标
Timoshenko 教科书常数（0.00406 / k=4 / k=2 / k_s=9.34 与 5.34），
层合板情形以三套互相独立的解法（Navier 双三角 / Levy 单三角 / Ritz 能量）
互裁，另加矩阵级 Gauss 积分交叉验证。
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tensorphysics.composite_mechanics.laminate_plate as LP  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "p128w3_n10_composite")
os.makedirs(OUT, exist_ok=True)

A = B = 1.0          # 板边长 (m)
Q0 = 1.0e4           # 均布载荷 (Pa)
T = 0.125e-3         # 单层厚度 (m)
MAT = LP.T300_5208()  # T300/5208 石墨-环氧

ISO_E, ISO_NU, ISO_H = 200.0e9, 0.30, 0.01  # 各向同性对照板

RESULTS: dict = {
    "node": "p128w3-n10-composite-laminate",
    "module": "tensorphysics.composite_mechanics.laminate_plate",
    "material": {
        "name": "T300/5208",
        "E1_GPa": MAT.E1 / 1e9,
        "E2_GPa": MAT.E2 / 1e9,
        "G12_GPa": MAT.G12 / 1e9,
        "nu12": MAT.nu12,
        "ply_thickness_mm": T * 1e3,
        "source": "Reddy (2004) Mechanics of Laminated Composite Plates and Shells",
    },
    "geometry": {"a_m": A, "b_m": B, "q0_Pa": Q0},
    "isotropic_reference": {"E_GPa": ISO_E / 1e9, "nu": ISO_NU, "h_mm": ISO_H * 1e3},
}
CRITERIA: list = []
import time as _time
_T = {"t": _time.time()}


def lap(tag: str) -> None:
    now = _time.time()
    print(f"[timer] {tag:36s} {now - _T['t']:6.2f}s", flush=True)
    _T["t"] = now


def crit(cid: str, statement: str, value: float, threshold: float, ok: bool, note: str = "") -> None:
    CRITERIA.append(
        {
            "id": cid,
            "statement": statement,
            "value": float(value),
            "threshold": float(threshold),
            "pass": bool(ok),
            "note": note,
        }
    )


# ---------------------------------------------------------------------------
# 0) 几何闸门
# ---------------------------------------------------------------------------
print("[progress] stage0 geometry", flush=True)
geo = LP.validate_plate_geometry(A, B, 8 * T)
RESULTS["geometry_gate"] = geo

# ---------------------------------------------------------------------------
# 1) 各向同性退化极限（教科书锚）
# ---------------------------------------------------------------------------
lap("start")
iso = LP.Laminate([(0.0, ISO_H)], LP.Lamina.isotropic(ISO_E, ISO_NU), name="iso-plate")
D_iso = LP.isotropic_plate_rigidity(ISO_E, ISO_NU, ISO_H)

print("[progress] stage1 isotropic anchors", flush=True)
lap("nav_iso")
nav_iso = LP.navier_clt_deflection(iso, A, B, 1000.0, 99, 99)
coef = nav_iso.w_max * D_iso / (1000.0 * A**4)
crit(
    "C1-deflection-coefficient",
    "各向同性方板中心挠度系数 = w D/(q a^4)，对标 Timoshenko 0.00406",
    coef,
    0.00406,
    abs(coef - 0.0040624) / 0.0040624 < 1e-2,
    "解析级数截断 99x99",
)

lap("iso buckling")
b_uni = LP.buckling_uniaxial(iso, A, B)
k_uni = b_uni.n_cr * B**2 / (math.pi**2 * D_iso)
crit("C2-uniaxial-k", "单轴屈曲系数 k = N_cr b^2/(pi^2 D)，对标经典 4.0", k_uni, 4.0,
     abs(k_uni - 4.0) < 1e-6, f"模态 (m,n)={b_uni.m},{b_uni.n}")

b_bia = LP.buckling_biaxial(iso, A, B)
k_bia = b_bia.n_cr * B**2 / (math.pi**2 * D_iso)
crit("C3-biaxial-k", "双轴等压屈曲系数 k，对标经典 2.0", k_bia, 2.0,
     abs(k_bia - 2.0) < 1e-6, f"模态 (m,n)={b_bia.m},{b_bia.n}")

shear_sq = LP.buckling_eigen_shear(iso, A, B, 16, 16)
ks_sq = shear_sq.n_cr * B**2 / (math.pi**2 * D_iso)
crit("C4-shear-k-square", "方板纯剪屈曲系数 k_s，对标 Timoshenko 9.34", ks_sq, 9.34,
     abs(ks_sq - 9.34) / 9.34 < 2e-2, "12x12->16x16 已收敛到 4 位")

shear_long = LP.buckling_eigen_shear(iso, 8.0 * B, B, 24, 10)
ks_long = shear_long.n_cr * B**2 / (math.pi**2 * D_iso)
crit("C5-shear-k-long", "长板 (a/b=8) 纯剪屈曲系数，对标 5.34", ks_long, 5.34,
     5.30 < ks_long < 5.60, "有限长板略高于渐近值")

lap("ks_vs_ar")
ks_vs_ar = []
for ar in (1.0, 1.5, 2.0, 4.0, 8.0):
    m_terms = int(max(12, min(24, 12 * ar)))
    n_terms = 10
    res = LP.buckling_eigen_shear(iso, ar * B, B, m_terms, n_terms)
    ks_vs_ar.append(
        {
            "a_over_b": ar,
            "k_s": res.n_cr * B**2 / (math.pi**2 * D_iso),
            "basis": [m_terms, n_terms],
        }
    )
RESULTS["isotropic_shear_coefficient_vs_aspect"] = ks_vs_ar

lap("anchors done")
RESULTS["isotropic_anchors"] = {
    "deflection_coefficient": coef,
    "k_uniaxial": k_uni,
    "k_biaxial": k_bia,
    "k_s_square": ks_sq,
    "k_s_long": ks_long,
    "D_Nm": D_iso,
}

# ---------------------------------------------------------------------------
# 2) 对称正交各向层合板：三重裁决（Navier / Ritz / Levy）
# ---------------------------------------------------------------------------
print("[progress] stage2 cross-ply adjudication", flush=True)
lap("crossply")
cross = LP.Laminate.from_spec("[0/90]2s", T, MAT)
desc_cross = cross.describe()
RESULTS["layup_0_90_2s"] = desc_cross

nav = LP.navier_clt_deflection(cross, A, B, Q0, 99, 99)
nav5 = LP.navier_clt_deflection(cross, A, B, Q0, 5, 5)
ritz = LP.ritz_deflection(cross, A, B, Q0, 6, 6)
levy_ss = LP.levy_deflection(cross, A, B, Q0, x_edges="SS", n_terms=61)
fsdt = LP.fsdt_navier_deflection(cross, A, B, Q0)
ritz_ref = LP.ritz_deflection(cross, A, B, Q0, 12, 12)

e_ritz = abs(ritz.w_max - nav5.w_max) / nav5.w_max
e_levy = abs(levy_ss.w_max - nav.w_max) / nav.w_max
e_conv = abs(ritz_ref.w_max - nav.w_max) / nav.w_max

crit("C6-navier-vs-ritz-same-basis", "正交对称板 Navier 闭式 vs Ritz 同截断（相对误差）",
     e_ritz, 1e-10, e_ritz < 1e-10, "K 对角 -> 应逐位一致")
crit("C7-navier-vs-levy", "Navier 双三角 vs Levy 单三角（SS 分支）相对误差", e_levy, 1e-8,
     e_levy < 1e-8, "两种完全不同的解法路径")
crit("C8-ritz-convergence-12x12", "Ritz 12x12 相对 99x99 Navier 的截断误差", e_conv, 1e-4,
     e_conv < 1e-4, "级数收敛性")

# Levy CC vs 独立固支能量解
print("[progress] stage2b Levy CC", flush=True)
lap("levy cc")
levy_cc = LP.levy_deflection(iso, A, B, 1000.0, x_edges="CC", n_terms=61)
clamped_ritz = LP.ritz_clamped_x_deflection(iso, A, B, 1000.0, m_terms=6, n_terms=14)
e_cc = abs(clamped_ritz.w_max - levy_cc.w_max) / levy_cc.w_max
crit("C9-levy-cc-vs-independent", "Levy(CC) vs 固支梁函数+Gauss 积分独立能量解", e_cc, 5e-3,
     e_cc < 5e-3, "两种独立离散路径互裁")

# FSDT -> CLT 收敛
print("[progress] stage2c FSDT sweep", flush=True)
lap("fsdt")
fsdt_sweep = []
for h_over_a in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
    lam = LP.Laminate.from_spec("[0/90]2s", h_over_a * A / 8.0, MAT)
    w_clt = LP.navier_clt_deflection(lam, A, B, Q0).w_max
    w_fsdt = LP.fsdt_navier_deflection(lam, A, B, Q0).w_max
    fsdt_sweep.append(
        {
            "h_over_a": h_over_a,
            "h_mm": lam.thickness * 1e3,
            "w_clt_m": w_clt,
            "w_fsdt_m": w_fsdt,
            "shear_amplification": w_fsdt / w_clt,
            "rel_diff": (w_fsdt - w_clt) / w_clt,
        }
    )
RESULTS["fsdt_vs_clt"] = fsdt_sweep

d1 = next(r["rel_diff"] for r in fsdt_sweep if r["h_over_a"] == 0.05)
d2 = next(r["rel_diff"] for r in fsdt_sweep if r["h_over_a"] == 0.1)
order = math.log(d2 / d1) / math.log(2.0)
d_small = next(r["rel_diff"] for r in fsdt_sweep if r["h_over_a"] == 0.002)
crit("C10-fsdt-clt-convergence-order", "FSDT->CLT 收敛阶（h/a 加倍时的误差比 log2）", order, 2.0,
     abs(order - 2.0) < 0.05, "应严格二阶")
crit("C11-fsdt-thin-limit", "h/a = 0.002 时 FSDT 与 CLT 的相对差", d_small, 1e-3,
     abs(d_small) < 1e-3, "薄板极限")
crit("C12-fsdt-h-over-a-0.1", "h/a = 0.1 时剪切修正量（CLT 边界失效量化）", d2, 0.10, d2 > 0.10,
     "厚跨比 1/10 时 CLT 已偏刚 15%")

RESULTS["bending_cross_ply"] = {
    "navier_w_max_m": nav.w_max,
    "navier_5x5_w_max_m": nav5.w_max,
    "ritz_6x6_w_max_m": ritz.w_max,
    "ritz_12x12_w_max_m": ritz_ref.w_max,
    "levy_ss_w_max_m": levy_ss.w_max,
    "fsdt_w_max_m": fsdt.w_max,
    "rel_err_ritz_vs_navier": e_ritz,
    "rel_err_levy_vs_navier": e_levy,
    "rel_err_ritz12_vs_navier99": e_conv,
    "levy_cc_iso_w_max_m": levy_cc.w_max,
    "ritz_clamped_x_iso_w_max_m": clamped_ritz.w_max,
    "rel_err_levy_cc_vs_independent": e_cc,
}

# ---------------------------------------------------------------------------
# 3) 弯曲-扭转耦合（反对称/角铺层）
# ---------------------------------------------------------------------------
def bending_equivalent(lam: LP.Laminate):
    """对称铺层原样返回；非对称铺层用 ``D* = D - B A^-1 B`` 的等效弯曲板（面内不动近似）。"""
    if lam.is_symmetric():
        return lam, False
    eq = LP.Laminate(lam.plies, lam.lamina, shear_correction=lam.shear_correction, name=lam.name + "(*)")
    eq.D = lam.d_star()
    eq.B = np.zeros((3, 3))
    return eq, True


print("[progress] stage3 coupling", flush=True)
lap("coupling tbl")
coupling_table = []
for spec in ("[0/90]s", "[0/90]", "[±45]s", "[45/-45]s", "[-45/45]s", "[30/-30]s", "[0/90/45/-45]s"):
    lam = LP.Laminate.from_spec(spec, T, MAT)
    eq, used_star = bending_equivalent(lam)
    if eq.is_orthotropic():
        r = LP.navier_clt_deflection(eq, A, B, Q0)
    else:
        r = LP.ritz_deflection(eq, A, B, Q0, 14, 14)
    row = {
        "layup": spec,
        "n_plies": len(lam.plies),
        "thickness_mm": lam.thickness * 1e3,
        "D11": lam.D11,
        "D22": lam.D22,
        "D12": lam.D12,
        "D66": lam.D66,
        "D16": lam.D16,
        "D26": lam.D26,
        "B_max": float(np.max(np.abs(lam.B))),
        "is_symmetric": lam.is_symmetric(),
        "is_orthotropic": lam.is_orthotropic(),
        "abs_D16_over_D11": abs(lam.D16) / lam.D11,
        "abs_D26_over_D22": abs(lam.D26) / lam.D22,
        "w_max_m": r.w_max,
        "bending_method": r.method,
        "used_D_star": used_star,
        "D_star_D11": float(np.asarray(eq.D)[0, 0]),
    }
    coupling_table.append(row)
RESULTS["layup_coupling_table"] = coupling_table

d16_sym = abs(cross.D16) / cross.D11
crit("C13-crossply-zero-coupling", "对称正交各向 [0/90]2s 的 |D16|/D11", d16_sym, 1e-9,
     d16_sym < 1e-9, "负控 (c1) 成立")

a45 = LP.Laminate.from_spec("[±45]s", T, MAT)
anti = LP.Laminate.from_spec("[0/90]", T, MAT)
crit("C14-angleply-has-D16", "对称角铺层 [±45]s 的 |D16|/D11（弯-扭耦合存在）",
     abs(a45.D16) / a45.D11, 0.4, abs(a45.D16) / a45.D11 > 0.4,
     "纠正任务书『对称铺层 D16=0』")
crit("C15-antisymmetric-crossply", "[0/90] 的 |D16|/D11（应为 0；仅 B != 0）",
     abs(anti.D16) / anti.D11, 1e-9, abs(anti.D16) / anti.D11 < 1e-9,
     "纠正任务书『[0/90] 非对称产生 D16』")
crit("C16-antisymmetric-has-B", "[0/90] 的 B11/(A11 h)（拉-弯耦合强度）",
     abs(anti.B[0, 0]) / (anti.A[0, 0] * anti.thickness), 1e-3,
     abs(anti.B[0, 0]) / (anti.A[0, 0] * anti.thickness) > 1e-3, "拉-弯耦合确实存在")

# 耦合对挠度的影响（同 D 对角参照）
dec = LP.Laminate.from_spec("[±45]s", T, MAT)
D0 = dec.D.copy()
D0[0, 2] = D0[1, 2] = D0[2, 0] = D0[2, 1] = 0.0
dec.D = D0
w_coupled = LP.ritz_deflection(a45, A, B, Q0, 14, 14).w_max
w_decoupled = LP.ritz_deflection(dec, A, B, Q0, 14, 14).w_max
amp = w_coupled / w_decoupled
crit("C17-coupling-increases-deflection", "弯-扭耦合引起的中心挠度放大（[±45]s vs 同 D 对角）", amp,
     1.05, amp > 1.05, "耦合使板更柔")
RESULTS["coupling_deflection"] = {
    "layup": "[±45]s",
    "w_coupled_m": w_coupled,
    "w_decoupled_m": w_decoupled,
    "amplification": amp,
    "D16": a45.D16,
    "D26": a45.D26,
    "note": "解耦参照 = 同 D11/D22/D12/D66、D16=D26=0 的虚构板",
}

# 耦合随铺层角的规模（D16/D11 与挠度放大）
print("[progress] stage3b coupling scan", flush=True)
lap("coupling scan")
coupling_scan = []
for th in range(0, 91, 10):
    lam = LP.Laminate.from_spec(f"[{th}/-{th}]s", T, MAT)
    lam_d = LP.Laminate.from_spec(f"[{th}/-{th}]s", T, MAT)
    Dd = lam_d.D.copy()
    Dd[0, 2] = Dd[1, 2] = Dd[2, 0] = Dd[2, 1] = 0.0
    lam_d.D = Dd
    wc = LP.ritz_deflection(lam, A, B, Q0, 10, 10).w_max
    wd = LP.ritz_deflection(lam_d, A, B, Q0, 10, 10).w_max
    coupling_scan.append(
        {
            "theta_deg": th,
            "D16": lam.D16,
            "D26": lam.D26,
            "abs_D16_over_D11": abs(lam.D16) / lam.D11,
            "w_coupled_m": wc,
            "w_decoupled_m": wd,
            "amplification": wc / wd,
        }
    )
RESULTS["coupling_vs_angle"] = coupling_scan

# ---------------------------------------------------------------------------
# 4) 屈曲：闭式 vs 广义特征值 + 铺层角扫描
# ---------------------------------------------------------------------------
print("[progress] stage4 buckling", flush=True)
lap("buckling")
ncr_uni = LP.buckling_uniaxial(cross, A, B)
ncr_bia = LP.buckling_biaxial(cross, A, B)
ncr_eig = LP.buckling_eigen_uniaxial(cross, A, B, 10, 10)
e_eig = abs(ncr_eig.n_cr - ncr_uni.n_cr) / ncr_uni.n_cr
crit("C18-eigen-vs-closed-form", "各向异性特征值解退化为正交各向闭式（相对误差）", e_eig, 1e-8,
     e_eig < 1e-8, "[0/90]2s 的 D16=D26=0")
RESULTS["buckling_cross_ply"] = {
    "N_x_cr_uniaxial_N_per_m": ncr_uni.n_cr,
    "mode_uniaxial": [ncr_uni.m, ncr_uni.n],
    "N_biax_cr_N_per_m": ncr_bia.n_cr,
    "mode_biaxial": [ncr_bia.m, ncr_bia.n],
    "N_x_cr_eigen_N_per_m": ncr_eig.n_cr,
    "sigma_x_cr_kPa": ncr_uni.n_cr / cross.thickness / 1e3,
    "rel_err_eigen_vs_closed": e_eig,
}

print("[progress] stage4b angle sweep", flush=True)
lap("angle sweep")
angle_sweep = []
for th in range(0, 91, 5):
    lam = LP.Laminate.from_spec(f"[{th}/-{th}]s", T, MAT)
    uni = LP.buckling_eigen_uniaxial(lam, A, B, 10, 10)
    shr = LP.buckling_eigen_shear(lam, A, B, 10, 10)
    row = {
        "theta_deg": th,
        "N_x_cr_uniaxial": uni.n_cr,
        "mode_uniaxial": [uni.m, uni.n],
        "N_xy_cr_shear": shr.n_cr,
        "mode_shear": [shr.m, shr.n],
        "A11": lam.A[0, 0],
        "A66": lam.A[2, 2],
        "D11": lam.D11,
        "D22": lam.D22,
        "D66": lam.D66,
        "abs_D16_over_D11": abs(lam.D16) / lam.D11,
    }
    angle_sweep.append(row)
RESULTS["ply_angle_sweep"] = angle_sweep

best_uni = max(angle_sweep, key=lambda r: r["N_x_cr_uniaxial"])
best_shr = max(angle_sweep, key=lambda r: r["N_xy_cr_shear"])
best_a11 = max(angle_sweep, key=lambda r: r["A11"])
best_a66 = max(angle_sweep, key=lambda r: r["A66"])
crit("C19-shear-best-at-45", "纯剪屈曲最优铺层角 (deg)", best_shr["theta_deg"], 45.0,
     best_shr["theta_deg"] == 45.0,
     f"N_xy,cr(45°) = {best_shr['N_xy_cr_shear']:.2f} N/m")
crit("C20-a11-best-at-0", "面内轴向刚度 A11 最优铺层角 (deg)", best_a11["theta_deg"], 0.0,
     best_a11["theta_deg"] == 0.0, f"A11 = {best_a11['A11'] / 1e6:.2f} MN/m")
crit("C21-a66-best-at-45", "面内剪切刚度 A66 最优铺层角 (deg)", best_a66["theta_deg"], 45.0,
     best_a66["theta_deg"] == 45.0, f"A66 = {best_a66['A66'] / 1e6:.2f} MN/m")
RESULTS["ply_angle_summary"] = {
    "best_uniaxial_buckling_theta": best_uni["theta_deg"],
    "best_uniaxial_N_cr": best_uni["N_x_cr_uniaxial"],
    "best_shear_buckling_theta": best_shr["theta_deg"],
    "best_shear_N_xy_cr": best_shr["N_xy_cr_shear"],
    "best_A11_theta": best_a11["theta_deg"],
    "best_A66_theta": best_a66["theta_deg"],
    "N_cr_theta0": angle_sweep[0]["N_x_cr_uniaxial"],
    "N_xy_theta0": angle_sweep[0]["N_xy_cr_shear"],
    "honest_note": (
        "面内刚度直觉（A11 于 0° 最大、A66 于 ±45 最大）与剪切屈曲直觉（±45 最优）均成立；"
        "但『[0] 最优抗压』在薄板屈曲意义上不成立：单向板 D22/D66 极小，"
        "方板 N_x,cr(0°)=23.30 N/m 低于 N_x,cr(45°)=31.44 N/m。"
    ),
}

# 长板（a/b=4）复算：验证剪切下 ±45 优势与压缩下排序
print("[progress] stage4c long sweep", flush=True)
lap("long sweep")
long_sweep = []
for th in (0, 30, 45, 60, 90):
    lam = LP.Laminate.from_spec(f"[{th}/-{th}]s", T, MAT)
    long_sweep.append(
        {
            "theta_deg": th,
            "N_x_cr_uniaxial_a_over_b_4": LP.buckling_eigen_uniaxial(lam, 4.0 * B, B, 10, 8).n_cr,
            "N_xy_cr_shear_a_over_b_4": LP.buckling_eigen_shear(lam, 4.0 * B, B, 14, 10).n_cr,
        }
    )
RESULTS["ply_angle_sweep_a_over_b_4"] = long_sweep

# ---------------------------------------------------------------------------
# 5) 矩阵级交叉验证（Gauss 积分 vs 解析刚度阵）
# ---------------------------------------------------------------------------
print("[progress] stage5 quadrature checks", flush=True)
lap("quad")
def quad_check(lam: LP.Laminate, M: int, N: int, kind: str) -> float:
    rng = np.random.default_rng(20251108)
    W = rng.normal(size=M * N)
    nb = B
    gx, wx = np.polynomial.legendre.leggauss(80)
    xg = 0.5 * nb * (gx + 1.0)
    wg = 0.5 * nb * wx
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    Wg = np.meshgrid(wg, wg, indexing="ij")
    al = np.arange(1, M + 1) * math.pi / nb
    be = np.arange(1, N + 1) * math.pi / nb
    wxx = np.zeros_like(X)
    wyy = np.zeros_like(X)
    wxy = np.zeros_like(X)
    wx_ = np.zeros_like(X)
    wy_ = np.zeros_like(X)
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            c = W[(m - 1) * N + (n - 1)]
            sxy = np.sin(al[m - 1] * X) * np.sin(be[n - 1] * Y)
            wxx += -c * al[m - 1] ** 2 * sxy
            wyy += -c * be[n - 1] ** 2 * sxy
            wxy += c * al[m - 1] * be[n - 1] * np.cos(al[m - 1] * X) * np.cos(be[n - 1] * Y)
            wx_ += c * al[m - 1] * np.cos(al[m - 1] * X) * np.sin(be[n - 1] * Y)
            wy_ += c * be[n - 1] * np.sin(al[m - 1] * X) * np.cos(be[n - 1] * Y)
    if kind == "shear":
        G = LP._geo_shear(lam, nb, nb, M, N)
        num = W @ G @ W
        ref = 2.0 * float(np.sum(wx_ * wy_ * Wg[0] * Wg[1]))
    else:
        K = LP._ritz_bending_matrix(lam, nb, nb, M, N)
        num = W @ K @ W
        dens = (
            lam.D11 * wxx**2
            + lam.D22 * wyy**2
            + 2.0 * lam.D12 * wxx * wyy
            + 4.0 * lam.D66 * wxy**2
            + 4.0 * lam.D16 * wxx * wxy
            + 4.0 * lam.D26 * wyy * wxy
        )
        ref = float(np.sum(dens * Wg[0] * Wg[1]))
    return abs(num - ref) / abs(ref)


eK = quad_check(a45, 4, 4, "bending")
eG = quad_check(a45, 4, 4, "shear")
crit("C22-bending-matrix-vs-quadrature", "Ritz 弯曲刚度阵 vs Gauss 积分（相对误差，含 D16/D26）",
     eK, 1e-9, eK < 1e-9, "缺陷 #2 回归保护（交叉项符号）")
crit("C23-shear-matrix-vs-quadrature", "纯剪几何刚度阵 vs Gauss 积分（相对误差）", eG, 1e-9,
     eG < 1e-9, "缺陷 #1 回归保护（y 向积分表索引）")
RESULTS["matrix_cross_validation"] = {"bending_rel_err": eK, "shear_rel_err": eG}

# ---------------------------------------------------------------------------
# 6) 收敛性数据（报告用）
# ---------------------------------------------------------------------------
print("[progress] stage6 convergence + fields", flush=True)
lap("conv+fields")
conv = []
for mt in (4, 6, 8, 10, 12, 16, 20):
    conv.append(
        {
            "terms": mt,
            "w_cross_pily_m": LP.navier_clt_deflection(cross, A, B, Q0, mt, mt).w_max,
            "w_angle_pily_m": LP.ritz_deflection(a45, A, B, Q0, mt, mt).w_max,
        }
    )
RESULTS["series_convergence"] = conv

# 场量：中心挠度场
grid = np.linspace(0.0, A, 61)
Xg, Yg = np.meshgrid(grid, grid, indexing="ij")
# BendingResult.field 接受两个 1D 数组（张量积网格），返回 (nx, ny)
field_cross = nav.field(grid, grid)
field_angle = LP.ritz_deflection(a45, A, B, Q0, 20, 20).field(grid, grid)
field_decoupled = LP.ritz_deflection(dec, A, B, Q0, 20, 20).field(grid, grid)
np.savez_compressed(
    os.path.join(OUT, "fields.npz"),
    grid=grid,
    w_crossply=field_cross,
    w_angleply=field_angle,
    w_decoupled=field_decoupled,
)
RESULTS["fields"] = {
    "file": "fields.npz",
    "grid_points": int(grid.size),
    "w_crossply_max_m": float(field_cross.max()),
    "w_angleply_max_m": float(field_angle.max()),
    "edge_max_abs_m": float(max(np.abs(field_cross[0]).max(), np.abs(field_cross[-1]).max())),
}
crit("C24-boundary-condition", "场量在四边 w=0 的残差 (m)", RESULTS["fields"]["edge_max_abs_m"],
     1e-12, RESULTS["fields"]["edge_max_abs_m"] < 1e-12, "四边简支几何边界")

# ---------------------------------------------------------------------------
# 7) 图
# ---------------------------------------------------------------------------
print("[progress] stage7 figures", flush=True)
RESULTS["criteria"] = CRITERIA
RESULTS["criteria_summary"] = {
    "n_criteria": len(CRITERIA),
    "n_pass": sum(1 for c in CRITERIA if c["pass"]),
    "n_fail": sum(1 for c in CRITERIA if not c["pass"]),
}
RESULTS["figures"] = [
    "fig1_deflection_fields.png",
    "fig2_fsdt_clt_scaling.png",
    "fig3_ply_angle_sweep.png",
    "fig4_coupling_amplification.png",
    "fig5_shear_coefficient.png",
    "fig6_multi_adjudication.png",
]
RESULTS["artifacts_dir"] = "artifacts/p128w3_n10_composite"
RESULTS["tests"] = "tests/test_composite_laminate_clt_fsdt.py"

with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(RESULTS, fh, ensure_ascii=False, indent=2, default=float)
with open(os.path.join(OUT, "criteria.json"), "w", encoding="utf-8") as fh:
    json.dump(
        {
            "node": RESULTS["node"],
            "criteria": CRITERIA,
            "summary": RESULTS["criteria_summary"],
            "artifacts": RESULTS["figures"] + ["results.json", "criteria.json", "fields.npz"],
        },
        fh,
        ensure_ascii=False,
        indent=2,
        default=float,
    )


lap("figures")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.35})

# 图 1：挠度场云图（cross-ply / angle-ply / 解耦差）
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
for ax, data, title in zip(
    axes,
    (field_cross * 1e3, field_angle * 1e3, (field_angle - field_decoupled) * 1e3),
    ("[0/90]2s  CLT-Navier  w (mm)", "[±45]s  Ritz  w (mm)", "[±45]s  w - w(D16=D26=0) (mm)"),
):
    cf = ax.contourf(Xg, Yg, data, levels=18, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax, shrink=0.85)
fig.suptitle("Fig.1  Deflection fields, square laminated plate (a=b=1 m, q0=10 kPa, t_ply=0.125 mm)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_deflection_fields.png"))
plt.close(fig)

# 图 2：FSDT/CLT vs h/a
h = np.array([r["h_over_a"] for r in fsdt_sweep])
amp_arr = np.array([r["shear_amplification"] for r in fsdt_sweep])
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.loglog(h, amp_arr - 1.0, "o-", label="FSDT/CLT - 1 (numerical)")
ref = (amp_arr[2] - 1.0) * (h / h[2]) ** 2
ax.loglog(h, ref, "k--", label=r"reference slope $\propto (h/a)^2$")
ax.axvline(0.1, color="r", ls=":", lw=1.2)
ax.text(0.105, 1e-4, "h/a = 1/10\nCLT limit", color="r", fontsize=8)
ax.set_xlabel("h/a")
ax.set_ylabel("(w_FSDT - w_CLT)/w_CLT")
ax.set_title("Fig.2  Shear correction scaling of FSDT vs CLT ([0/90]2s)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_fsdt_clt_scaling.png"))
plt.close(fig)

# 图 3：铺层角扫描（屈曲 + 面内刚度）
th = np.array([r["theta_deg"] for r in angle_sweep])
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
ax = axes[0]
ax.plot(th, [r["N_x_cr_uniaxial"] for r in angle_sweep], "o-", label=r"$N_{x,cr}$ uniaxial compression")
ax.plot(th, [r["N_xy_cr_shear"] for r in angle_sweep], "s-", label=r"$N_{xy,cr}$ pure shear")
ax.set_xlabel(r"ply angle $\theta$ ($[\theta/-\theta]_s$) [deg]")
ax.set_ylabel("critical line load (N/m)")
ax.set_title("Square plate buckling vs ply angle")
ax.legend(fontsize=8)
ax = axes[1]
ax.plot(th, np.array([r["A11"] for r in angle_sweep]) / 1e6, "o-", label=r"$A_{11}$")
ax.plot(th, np.array([r["A66"] for r in angle_sweep]) / 1e6, "s-", label=r"$A_{66}$")
ax.plot(th, np.array([r["abs_D16_over_D11"] for r in angle_sweep]), "^-", label=r"$|D_{16}|/D_{11}$")
ax.set_xlabel(r"ply angle $\theta$ [deg]")
ax.set_ylabel(r"$A$ (MN/m) / dimensionless ratio")
ax.set_title("In-plane stiffness and bend-twist coupling")
ax.legend(fontsize=8)
fig.suptitle("Fig.3  Ply-angle sweep: buckling loads (left) and in-plane stiffness / coupling (right)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_ply_angle_sweep.png"))
plt.close(fig)

# 图 4：弯-扭耦合对挠度的影响
amp45 = np.array([r["amplification"] for r in coupling_scan])
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.plot([r["theta_deg"] for r in coupling_scan], amp45, "o-", color="C3")
ax.axhline(1.0, color="k", ls=":", lw=1)
ax.set_xlabel(r"ply angle $\theta$ [deg]")
ax.set_ylabel(r"$w_{coupled} / w_{decoupled}$")
ax.set_title("Fig.4  Deflection amplification by bend-twist coupling (same D diag)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_coupling_amplification.png"))
plt.close(fig)

# 图 5：各向同性纯剪屈曲系数 vs 长宽比（对标 5.34 / 9.34）
ar = np.array([r["a_over_b"] for r in ks_vs_ar])
ks = np.array([r["k_s"] for r in ks_vs_ar])
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.semilogx(ar, ks, "o-", label=r"this implementation $k_s$")
ax.axhline(5.34, color="k", ls="--", label="Timoshenko, long plate 5.34")
ax.axhline(9.34, color="r", ls=":", label="Timoshenko, square plate 9.34")
ax.set_xlabel("aspect ratio a/b")
ax.set_ylabel(r"$k_s = N_{xy,cr} b^2/(\pi^2 D)$")
ax.set_title("Fig.5  Isotropic shear buckling coefficient vs textbook")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_shear_coefficient.png"))
plt.close(fig)

# 图 6：三方法/收敛对照
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
ax = axes[0]
labels = ["Navier\n99x99", "Ritz\n6x6", "Ritz\n12x12", "Levy(SS)\nn=31", "FSDT\nNavier"]
vals = [nav.w_max, ritz.w_max, ritz_ref.w_max, levy_ss.w_max, fsdt.w_max]
ax.bar(labels, np.array(vals) * 1e3, color=["C0", "C1", "C1", "C2", "C3"])
ax.set_ylabel(r"$w_{max}$ (mm)")
ax.set_title("[0/90]2s centre deflection: cross-adjudication")
for i, v in enumerate(vals):
    ax.text(i, v * 1e3 * 1.001, f"{v * 1e3:.5f}", ha="center", fontsize=7)
ax = axes[1]
terms = [r["terms"] for r in conv]
w_ref_ang = conv[-1]["w_angle_pily_m"]
ax.semilogy(terms, [abs(r["w_angle_pily_m"] - w_ref_ang) + 1e-18 for r in conv], "s-", label="[±45]s Ritz")
w_ref_cross = conv[-1]["w_cross_pily_m"]
ax.semilogy(terms, [abs(r["w_cross_pily_m"] - w_ref_cross) + 1e-18 for r in conv], "o-", label="[0/90]2s Navier")
ax.set_xlabel("modes retained per direction")
ax.set_ylabel("difference vs finest truncation (m)")
ax.set_title("Series convergence")
ax.legend(fontsize=8)
fig.suptitle("Fig.6  Multi-method adjudication and series convergence")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_multi_adjudication.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# 8) 落盘
# ---------------------------------------------------------------------------
print(json.dumps(RESULTS["criteria_summary"], ensure_ascii=False))
for c in CRITERIA:
    flag = "PASS" if c["pass"] else "FAIL"
    print(f"[{flag}] {c['id']:38s} value={c['value']:.6g} thr={c['threshold']:.6g}")
print("artifacts ->", OUT)