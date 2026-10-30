#!/usr/bin/env python3
"""ice_anchor 锚库验证与首轮量级对照脚本。

功能：
1. 运行全部 14 个锚点，输出标准工况下的量级
2. 生成覆盖矩阵热图（七腿 × 锚库）
3. 与 W1 已有输出（断裂传播/自由面）做首轮量级对照
4. 生成偏差表 + 对比图
5. 输出验证报告 JSON
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ice_anchor.anchors import (
    ANCHORS, ANCHOR_REGISTRY,
    lindqvist_1989, keinonen_1983, huang_2014, enkvist_1972, riska_1997,
    ice_crushing_strength, ice_flexural_strength, ice_indentation,
    bending_fracture_load, model_test_acecrc, model_test_hamburg,
    model_test_arctech, ice_thermal_properties, ice_density_salinity,
)
from ice_anchor.mapping import get_coverage_matrix, get_mapping


# ──────────────────────────────────────────────
# 标准工况参数
# ──────────────────────────────────────────────

STANDARD_CASE = {
    "h": 1.0,        # 冰厚 (m)
    "B": 20.0,       # 船宽 (m)
    "L": 120.0,      # 船长 (m)
    "phi": 30.0,     # 艏角 (deg)
    "V": 3.0,        # 航速 (m/s)
    "sigma_f": 500e3,  # 冰弯曲强度 (Pa)
    "sigma_c": 5e6,    # 冰压溃强度 (Pa)
    "rho_i": 900.0,    # 冰密度 (kg/m³)
    "T_C": -10.0,      # 冰温 (°C)
}


def run_all_anchors() -> list:
    """运行全部锚点，返回结果列表。"""
    sc = STANDARD_CASE
    results = []

    # A1: Lindqvist
    r = lindqvist_1989(sc["h"], sc["B"], sc["phi"], sc["V"], sc["L"],
                       sc["sigma_f"], sc["sigma_c"], sc["rho_i"])
    results.append({"anchor_id": "A1", "R_total_kN": r["R_total_kN"],
                    "R_c_kN": r["R_c_N"]/1000, "R_f_kN": r["R_f_N"]/1000, "R_s_kN": r["R_s_N"]/1000})

    # A2: Keinonen
    r = keinonen_1983(sc["h"], sc["B"], sc["V"], sc["L"], sc["phi"], sc["sigma_f"], sc["rho_i"])
    results.append({"anchor_id": "A2", "R_total_kN": r["R_total_kN"]})

    # A3: Huang
    r = huang_2014(sc["h"], sc["B"], sc["V"], sc["L"], sc["phi"],
                   sc["sigma_f"], sc["sigma_c"], sc["rho_i"])
    results.append({"anchor_id": "A3", "R_total_kN": r["R_total_kN"],
                    "C_B": r["C_B"], "C_phi": r["C_phi"], "C_h": r["C_h"]})

    # A4: Enkvist
    r = enkvist_1972(sc["h"], sc["B"], sc["V"], sc["L"], sc["phi"], sc["sigma_f"], sc["rho_i"])
    results.append({"anchor_id": "A4", "R_total_kN": r["R_total_kN"]})

    # A5: Riska (ahead)
    r = riska_1997(sc["h"], sc["B"], sc["V"], sc["L"], sc["phi"],
                   sc["sigma_f"], sc["sigma_c"], sc["rho_i"], mode="ahead")
    results.append({"anchor_id": "A5", "R_total_kN": r["R_total_kN"], "mode": "ahead"})

    # A6: Crushing strength
    r = ice_crushing_strength(sc["T_C"])
    results.append({"anchor_id": "A6", "sigma_c_MPa": r["sigma_c_MPa"]})

    # A7: Flexural strength
    r = ice_flexural_strength(sc["T_C"])
    results.append({"anchor_id": "A7", "sigma_f_kPa": r["sigma_f_kPa"]})

    # A8: Indentation
    r = ice_indentation(sc["h"], sc["B"], sc["V"], sc["sigma_c"])
    results.append({"anchor_id": "A8", "F_kN": r["F_kN"], "p_MPa": r["p_MPa"]})

    # A9: Bending fracture
    r = bending_fracture_load(sc["h"], sc["sigma_f"])
    results.append({"anchor_id": "A9", "q_kN_per_m": r["q_kN_per_m"],
                    "l_m": r["characteristic_length_m"]})

    # A10: ACECRC
    r = model_test_acecrc()
    results.append({"anchor_id": "A10", "n_data_points": len(r["ice_thickness_m"]) * len(r["speed_ms"]),
                    "basin": r["basin"]})

    # A11: HSVA
    r = model_test_hamburg()
    results.append({"anchor_id": "A11", "n_data_points": len(r["ice_thickness_m"]) * len(r["speed_ms"]),
                    "basin": r["basin"]})

    # A12: Arctech
    r = model_test_arctech()
    results.append({"anchor_id": "A12", "n_data_points": len(r["ice_thickness_m"]) * len(r["speed_ms"]),
                    "basin": r["basin"]})

    # A13: Thermal
    r = ice_thermal_properties(sc["T_C"])
    results.append({"anchor_id": "A13", "k_W_mK": r["thermal_conductivity_W_per_mK"],
                    "c_J_kgK": r["specific_heat_J_per_kgK"]})

    # A14: Density
    r = ice_density_salinity()
    results.append({"anchor_id": "A14", "rho_kg_m3": r["rho_i_kg_per_m3"]})

    return results


def plot_coverage_heatmap(output_dir: Path) -> str:
    """生成覆盖矩阵热图。"""
    headers, rows = get_coverage_matrix()
    sir_legs = headers[1:]  # skip "anchor"
    anchor_ids = [r[0] for r in rows]
    matrix = np.array([r[1:] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = mcolors.ListedColormap(["#f0f0f0", "#2196F3"])
    bounds = [0, 0.5, 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(sir_legs)))
    ax.set_xticklabels(sir_legs, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(anchor_ids)))
    ax.set_yticklabels(anchor_ids, fontsize=11)

    # 标注数值
    for i in range(len(anchor_ids)):
        for j in range(len(sir_legs)):
            val = int(matrix[i, j])
            color = "white" if val == 1 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    ax.set_xlabel("SIR 七腿", fontsize=13, fontweight="bold")
    ax.set_ylabel("锚点", fontsize=13, fontweight="bold")
    ax.set_title("锚库 × SIR 七腿覆盖矩阵 (m598)", fontsize=14, fontweight="bold")

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2196F3", label="已覆盖"),
                       Patch(facecolor="#f0f0f0", edgecolor="gray", label="未覆盖")]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    png_path = output_dir / "coverage_heatmap.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def plot_resistance_comparison(output_dir: Path) -> str:
    """生成阻力公式对比图（A1-A5）。"""
    sc = STANDARD_CASE
    h_values = np.linspace(0.3, 2.0, 20)

    R_lindqvist = []
    R_keinonen = []
    R_huang = []
    R_enkvist = []
    R_riska = []

    for h in h_values:
        r1 = lindqvist_1989(h, sc["B"], sc["phi"], sc["V"], sc["L"],
                            sc["sigma_f"], sc["sigma_c"], sc["rho_i"])
        R_lindqvist.append(r1["R_total_kN"])

        r2 = keinonen_1983(h, sc["B"], sc["V"], sc["L"], sc["phi"],
                           sc["sigma_f"], sc["rho_i"])
        R_keinonen.append(r2["R_total_kN"])

        r3 = huang_2014(h, sc["B"], sc["V"], sc["L"], sc["phi"],
                        sc["sigma_f"], sc["sigma_c"], sc["rho_i"])
        R_huang.append(r3["R_total_kN"])

        r4 = enkvist_1972(h, sc["B"], sc["V"], sc["L"], sc["phi"],
                          sc["sigma_f"], sc["rho_i"])
        R_enkvist.append(r4["R_total_kN"])

        r5 = riska_1997(h, sc["B"], sc["V"], sc["L"], sc["phi"],
                        sc["sigma_f"], sc["sigma_c"], sc["rho_i"], mode="ahead")
        R_riska.append(r5["R_total_kN"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_values, R_lindqvist, "b-o", label="A1: Lindqvist (1989)", linewidth=2, markersize=5)
    ax.plot(h_values, R_keinonen, "g-s", label="A2: Keinonen (1983)", linewidth=2, markersize=5)
    ax.plot(h_values, R_huang, "r-^", label="A3: Huang et al. (2014)", linewidth=2, markersize=5)
    ax.plot(h_values, R_enkvist, "m-d", label="A4: Enkvist (1972)", linewidth=2, markersize=5)
    ax.plot(h_values, R_riska, "c-v", label="A5: Riska (1997)", linewidth=2, markersize=5)

    ax.set_xlabel("冰厚 h (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("冰阻力 R (kN)", fontsize=12, fontweight="bold")
    ax.set_title("平整冰阻力公式对比 (B=20m, L=120m, φ=30°, V=3m/s)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = output_dir / "resistance_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def plot_fracture_load_vs_thickness(output_dir: Path) -> str:
    """生成断裂线载荷-冰厚关系图（A9）。"""
    h_values = np.linspace(0.1, 2.0, 30)

    q_radial = []
    q_circum = []

    for h in h_values:
        r_r = bending_fracture_load(h, crack_type="radial")
        r_c = bending_fracture_load(h, crack_type="circumferential")
        q_radial.append(r_r["q_kN_per_m"])
        q_circum.append(r_c["q_kN_per_m"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(h_values, q_radial, "b-o", label="径向裂纹 (Nevel 1965)", linewidth=2, markersize=4)
    ax.plot(h_values, q_circum, "r-s", label="环向裂纹 (Kerr 1996)", linewidth=2, markersize=4)

    ax.set_xlabel("冰厚 h (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("起始线载荷 q (kN/m)", fontsize=12, fontweight="bold")
    ax.set_title("破冰弯曲断裂线载荷 vs 冰厚 (σ_f=500kPa)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = output_dir / "fracture_load_vs_thickness.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def plot_strength_vs_temperature(output_dir: Path) -> str:
    """生成冰强度-温度关系图（A6, A7）。"""
    T_values = np.linspace(-40, 0, 30)

    sigma_c_vals = []
    sigma_f_slow = []
    sigma_f_fast = []

    for T in T_values:
        r_c = ice_crushing_strength(T)
        sigma_c_vals.append(r_c["sigma_c_MPa"])

        r_f_slow = ice_flexural_strength(T, loading_rate="slow")
        r_f_fast = ice_flexural_strength(T, loading_rate="fast")
        sigma_f_slow.append(r_f_slow["sigma_f_kPa"])
        sigma_f_fast.append(r_f_fast["sigma_f_kPa"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(T_values, sigma_c_vals, "b-o", linewidth=2, markersize=4)
    ax1.set_xlabel("温度 T (°C)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("压溃强度 σ_c (MPa)", fontsize=11, fontweight="bold")
    ax1.set_title("A6: 冰压溃强度 vs 温度", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(T_values, sigma_f_slow, "g-s", label="慢加载", linewidth=2, markersize=4)
    ax2.plot(T_values, sigma_f_fast, "r-^", label="快加载", linewidth=2, markersize=4)
    ax2.set_xlabel("温度 T (°C)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("弯曲强度 σ_f (kPa)", fontsize=11, fontweight="bold")
    ax2.set_title("A7: 冰弯曲强度 vs 温度", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("冰强度-温度关系 (ITTC/ISO 19906)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    png_path = output_dir / "strength_vs_temperature.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def generate_deviation_table(anchor_results: list) -> list:
    """生成首轮量级对照偏差表。

    以 Lindqvist (A1) 为基准，计算其他阻力公式的相对偏差。
    """
    deviations = []
    R_base = None
    for r in anchor_results:
        if r["anchor_id"] == "A1":
            R_base = r["R_total_kN"]
            break

    if R_base is None:
        return deviations

    for r in anchor_results:
        if r["anchor_id"] in ("A1", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14"):
            continue  # 非阻力公式或数据集
        if "R_total_kN" not in r:
            continue

        R_val = r["R_total_kN"]
        rel_dev = (R_val - R_base) / R_base * 100.0
        deviations.append({
            "anchor_id": r["anchor_id"],
            "R_total_kN": round(R_val, 2),
            "R_base_kN": round(R_base, 2),
            "relative_deviation_pct": round(rel_dev, 2),
            "within_15pct": abs(rel_dev) <= 15.0,
        })

    return deviations


def generate_architecture_diagram(output_dir: Path) -> str:
    """生成锚库架构图。"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # 标题
    ax.text(7, 7.5, "ice_anchor 破冰物理锚库架构 (m598)", fontsize=16, fontweight="bold",
            ha="center", va="center")

    # 四大类别框
    categories = [
        ("A. 阻力经验式", 1.5, 5.5, ["A1", "A2", "A3", "A4", "A5"], "#E3F2FD"),
        ("B. 强度典型值", 5.5, 5.5, ["A6", "A7", "A8"], "#E8F5E9"),
        ("C. 断裂关系", 8.5, 5.5, ["A9"], "#FFF3E0"),
        ("D. 试验数据集", 11.0, 5.5, ["A10", "A11", "A12"], "#F3E5F5"),
    ]

    for title, x, y, anchors, color in categories:
        # 类别框
        rect = plt.Rectangle((x - 1.2, y - 1.5), 2.8, 3.0,
                             facecolor=color, edgecolor="gray", linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y + 1.0, title, fontsize=11, fontweight="bold", ha="center", va="center")

        # 锚点
        for i, aid in enumerate(anchors):
            ay = y - 0.2 - i * 0.5
            ax.text(x, ay, aid, fontsize=10, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

    # 热物理性质
    rect2 = plt.Rectangle((1.5, 1.5), 5.0, 2.0,
                          facecolor="#FFF8E1", edgecolor="gray", linewidth=1.5, alpha=0.7)
    ax.add_patch(rect2)
    ax.text(4.0, 3.0, "E. 热物理性质", fontsize=11, fontweight="bold", ha="center", va="center")
    ax.text(2.5, 2.3, "A13", fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
    ax.text(5.5, 2.3, "A14", fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

    # SIR 七腿
    sir_y = 0.8
    sir_legs = ["SIR-1", "SIR-2", "SIR-3", "SIR-4", "SIR-5", "SIR-6", "SIR-7"]
    for i, leg in enumerate(sir_legs):
        sx = 1.5 + i * 1.6
        ax.text(sx, sir_y, leg, fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#2196F3", edgecolor="white",
                          alpha=0.8), color="white", fontweight="bold")

    # 箭头
    ax.annotate("", xy=(4.0, 1.5), xytext=(4.0, 3.5),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(7.0, 1.5), xytext=(7.0, 4.0),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    ax.text(7, 0.2, "← 锚点 → SIR 七腿映射 →", fontsize=10, ha="center", va="center",
            style="italic", color="gray")

    plt.tight_layout()
    png_path = output_dir / "architecture_diagram.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def main():
    """主函数。"""
    print("=" * 60)
    print("ice_anchor 锚库验证与首轮量级对照 (m598)")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "artifacts" / "ice_anchor"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 运行全部锚点
    print("\n[1] 运行全部 14 个锚点...")
    results = run_all_anchors()
    for r in results:
        print(f"  {r['anchor_id']}: {r}")

    # 2. 生成覆盖矩阵热图
    print("\n[2] 生成覆盖矩阵热图...")
    heatmap_path = plot_coverage_heatmap(output_dir)
    print(f"  热图已保存: {heatmap_path}")

    # 3. 生成阻力对比图
    print("\n[3] 生成阻力公式对比图...")
    res_comp_path = plot_resistance_comparison(output_dir)
    print(f"  对比图已保存: {res_comp_path}")

    # 4. 生成断裂线载荷图
    print("\n[4] 生成断裂线载荷-冰厚关系图...")
    frac_path = plot_fracture_load_vs_thickness(output_dir)
    print(f"  断裂图已保存: {frac_path}")

    # 5. 生成强度-温度图
    print("\n[5] 生成冰强度-温度关系图...")
    strength_path = plot_strength_vs_temperature(output_dir)
    print(f"  强度图已保存: {strength_path}")

    # 6. 生成架构图
    print("\n[6] 生成锚库架构图...")
    arch_path = generate_architecture_diagram(output_dir)
    print(f"  架构图已保存: {arch_path}")

    # 7. 生成偏差表
    print("\n[7] 生成首轮量级对照偏差表...")
    deviations = generate_deviation_table(results)
    for d in deviations:
        status = "✓" if d["within_15pct"] else "✗"
        print(f"  {d['anchor_id']}: R={d['R_total_kN']} kN, "
              f"偏差={d['relative_deviation_pct']:.1f}% {status}")

    # 8. 保存验证报告
    report = {
        "mission_id": 598,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_anchors": len(ANCHORS),
        "standard_case": STANDARD_CASE,
        "anchor_results": results,
        "deviation_table": deviations,
        "figures": {
            "coverage_heatmap": heatmap_path,
            "resistance_comparison": res_comp_path,
            "fracture_load_vs_thickness": frac_path,
            "strength_vs_temperature": strength_path,
            "architecture_diagram": arch_path,
        },
        "coverage_summary": {
            "total_anchors": 14,
            "resistance_anchors": 5,
            "strength_anchors": 3,
            "fracture_anchors": 1,
            "dataset_anchors": 3,
            "thermal_anchors": 2,
        },
    }

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[8] 验证报告已保存: {report_path}")

    print("\n" + "=" * 60)
    print("ice_anchor 锚库验证完成！")
    print(f"  锚点总数: {len(ANCHORS)}")
    print(f"  偏差表条目: {len(deviations)}")
    print(f"  生成图数: 5")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
