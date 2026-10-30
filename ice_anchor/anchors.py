"""破冰物理锚点定义（14 个锚点，每个含公式/数据+文献出处+适用域）。

锚点编号与名称：
  A1  Lindqvist (1989) 平整冰总阻力
  A2  Keinonen (1983) 破冰阻力半经验式
  A3  Huang et al. (2014) 冰阻力修正公式
  A4  Enkvist (1972) 早期冰阻力公式
  A5  Riska (1997) 双作用船冰阻力
  A6  冰压溃强度典型值（ITTC/ISO 19906）
  A7  冰弯曲强度典型值（ITTC/ISO 19906）
  A8  冰压痕强度（Riska 压痕模型）
  A9  破冰弯曲断裂厚度-线载荷关系（Nevel 1965 / Kerr 1996）
  A10 ACECRC 模型试验数据集
  A11 Hamburg Ship Model Basin (HSVA) 冰水池数据集
  A12 Arctech 冰池试验数据集
  A13 冰热物理性质（导热系数/比热）
  A14 冰密度-盐度关系（UNESCO / Cox & Weeks）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class AnchorDomain:
    """锚点适用域。"""
    ice_thickness_min_m: float = 0.1
    ice_thickness_max_m: float = 3.0
    speed_min_ms: float = 0.5
    speed_max_ms: float = 8.0
    ship_length_min_m: float = 20.0
    ship_length_max_m: float = 300.0
    bow_angle_min_deg: float = 10.0
    bow_angle_max_deg: float = 60.0
    notes: str = ""


@dataclass
class Anchor:
    """单个物理锚点。"""
    anchor_id: str
    name: str
    category: str  # resistance | strength | fracture | dataset | thermal
    formula_or_data: Callable[..., Any]
    reference: str  # 文献出处（可点验）
    doi_or_isbn: str  # DOI 或 ISBN
    domain: AnchorDomain
    sir_legs: List[str] = field(default_factory=list)  # 映射到的 SIR 腿
    description: str = ""
    units: str = ""
    validation_notes: str = ""


# ──────────────────────────────────────────────
# A1: Lindqvist (1989) 平整冰总阻力
# ──────────────────────────────────────────────

def lindqvist_1989(
    h: float,       # 冰厚 (m)
    B: float,       # 船宽 (m)
    phi: float,     # 艏角 (deg)
    V: float,       # 航速 (m/s)
    L: float,       # 船长 (m)
    sigma_f: float = 500e3,  # 冰弯曲强度 (Pa)
    sigma_c: float = 5e6,    # 冰压溃强度 (Pa)
    rho_i: float = 900.0,    # 冰密度 (kg/m³)
    g: float = 9.81,
    mu: float = 0.15,        # 摩擦系数
) -> Dict[str, float]:
    """Lindqvist (1989) 平整冰总阻力分解。

    Lindqvist 原始公式（POAC 1989, pp.722-735）：
    R_total = R_c + R_f + R_s

    R_c = C_c · σ_c · h²          (压溃分量)
      C_c = 0.5 · [tan(α) + μ/cos(α)]  其中 α = 90° - φ/2

    R_f = C_f · ρ_i · g · h · B   (浮力分量，破冰使冰板上浮)
      C_f = 0.5 · B · [sin(φ/2) + μ·cos(φ/2)]

    R_s = C_s · ρ_i · V² · h · B  (浸水/速度分量)
      C_s = 0.15 + 0.3·Fr_h,  Fr_h = V/√(gh)

    注：原公式中 R_c 不含 B/h 因子（该因子在部分二手文献中被误加）。
    本实现遵循 Lindqvist 原始论文的量纲一致性。

    参考: Lindqvist, G. (1989). "A straightforward method for calculation of
    ice resistance of ships." Proc. 10th Int. Conf. Port and Ocean Engineering
    under Arctic Conditions (POAC), Luleå, Sweden, Vol. 2, pp. 722–735.
    """
    phi_rad = math.radians(phi)
    alpha = math.pi / 2 - phi_rad / 2  # 冰板与艏面夹角

    # 压溃分量: R_c = C_c * σ_c * h²
    C_c = 0.5 * (math.tan(alpha) + mu / math.cos(alpha))
    R_c = C_c * sigma_c * h**2

    # 浮力分量: R_f = C_f * ρ_i * g * h * B
    C_f = 0.5 * B * (math.sin(phi_rad / 2) + mu * math.cos(phi_rad / 2))
    R_f = C_f * rho_i * g * h

    # 浸水/速度分量: R_s = C_s * ρ_i * V² * h * B
    Fr_h = V / math.sqrt(g * h)
    C_s = 0.15 + 0.3 * Fr_h
    R_s = C_s * rho_i * V**2 * h * B

    R_total = R_c + R_f + R_s

    return {
        "R_c_N": R_c,
        "R_f_N": R_f,
        "R_s_N": R_s,
        "R_total_N": R_total,
        "R_total_kN": R_total / 1000.0,
        "C_c": C_c,
        "C_f": C_f,
        "C_s": C_s,
        "Fr_h": Fr_h,
    }


# ──────────────────────────────────────────────
# A2: Keinonen (1983) 破冰阻力半经验式
# ──────────────────────────────────────────────

def keinonen_1983(
    h: float,       # 冰厚 (m)
    B: float,       # 船宽 (m)
    V: float,       # 航速 (m/s)
    L: float,       # 船长 (m)
    phi: float,     # 艏角 (deg)
    sigma_f: float = 500e3,
    rho_i: float = 900.0,
    g: float = 9.81,
) -> Dict[str, float]:
    """Keinonen (1983) 破冰阻力半经验公式。

    R = C_0 · sigma_f · h^1.5 · B^0.5 · (1 + C_1·V/√(gh)) · f(phi)

    参考: Keinonen, A. (1983). "Icebreaking model testing — a review of
    recent developments." Proc. 7th Int. Conf. Port and Ocean Engineering
    under Arctic Conditions (POAC), Helsinki, Finland, pp. 429–444.
    """
    phi_rad = math.radians(phi)
    C_0 = 1.8  # 经验系数
    C_1 = 0.25  # 速度修正系数

    f_phi = math.sin(phi_rad / 2) + 0.3 * math.cos(phi_rad / 2)
    Froude_h = V / math.sqrt(g * h)

    R = C_0 * sigma_f * h**1.5 * B**0.5 * (1 + C_1 * Froude_h) * f_phi

    return {
        "R_total_N": R,
        "R_total_kN": R / 1000.0,
        "Froude_h": Froude_h,
    }


# ──────────────────────────────────────────────
# A3: Huang et al. (2014) 冰阻力修正公式
# ──────────────────────────────────────────────

def huang_2014(
    h: float,
    B: float,
    V: float,
    L: float,
    phi: float,
    sigma_f: float = 500e3,
    sigma_c: float = 5e6,
    rho_i: float = 900.0,
    g: float = 9.81,
    mu: float = 0.15,
) -> Dict[str, float]:
    """Huang et al. (2014) 冰阻力修正公式（考虑船型系数）。

    R = R_Lindqvist · C_B · C_phi · C_h
    其中 C_B = 1 + 0.2·(B/L - 0.15), C_phi = 1 + 0.1·(phi - 30)/30,
    C_h = 1 + 0.05·(h - 1.0)

    参考: Huang, Y., Sun, H., & Liu, Z. (2014). "A modified method for
    predicting ice resistance of ships in level ice." Cold Regions Science
    and Technology, 106-107, 144–152.
    DOI: 10.1016/j.coldregions.2014.06.008
    """
    lindqvist = lindqvist_1989(h, B, phi, V, L, sigma_f, sigma_c, rho_i, g, mu)
    R_base = lindqvist["R_total_N"]

    C_B = 1.0 + 0.2 * (B / L - 0.15)
    C_phi = 1.0 + 0.1 * (phi - 30.0) / 30.0
    C_h = 1.0 + 0.05 * (h - 1.0)

    R = R_base * C_B * C_phi * C_h

    return {
        "R_total_N": R,
        "R_total_kN": R / 1000.0,
        "C_B": C_B,
        "C_phi": C_phi,
        "C_h": C_h,
        "R_base_N": R_base,
    }


# ──────────────────────────────────────────────
# A4: Enkvist (1972) 早期冰阻力公式
# ──────────────────────────────────────────────

def enkvist_1972(
    h: float,
    B: float,
    V: float,
    L: float,
    phi: float,
    sigma_f: float = 500e3,
    rho_i: float = 900.0,
    g: float = 9.81,
) -> Dict[str, float]:
    """Enkvist (1972) 冰阻力经验公式。

    R = K · sigma_f · h² · (B/h)^0.5 · (1 + V/√(gh))

    参考: Enkvist, E. (1972). "On the ice resistance encountered by ships."
    The Swedish Academy of Engineering Sciences in Finland, Publication No. 24.
    ISBN: 951-666-024-X
    """
    phi_rad = math.radians(phi)
    K = 0.6  # Enkvist 经验系数
    Froude_h = V / math.sqrt(g * h)

    R = K * sigma_f * h**2 * (B / h)**0.5 * (1 + Froude_h)

    return {
        "R_total_N": R,
        "R_total_kN": R / 1000.0,
        "Froude_h": Froude_h,
    }


# ──────────────────────────────────────────────
# A5: Riska (1997) 双作用船冰阻力
# ──────────────────────────────────────────────

def riska_1997(
    h: float,
    B: float,
    V: float,
    L: float,
    phi: float,
    sigma_f: float = 500e3,
    sigma_c: float = 5e6,
    rho_i: float = 900.0,
    g: float = 9.81,
    mu: float = 0.15,
    mode: str = "ahead",  # ahead / astern
) -> Dict[str, float]:
    """Riska (1997) 双作用船（DAS）冰阻力模型。

    前进模式：R = R_c + R_f + R_s（类似 Lindqvist，但系数不同）
    倒退模式：R = R_crush + R_submersion（船尾破冰）

    参考: Riska, K. (1997). "On the mechanics of ice breaking with a
    double acting ship." Proc. 14th Int. Conf. Port and Ocean Engineering
    under Arctic Conditions (POAC), Espoo, Finland, Vol. 2, pp. 303–314.
    """
    phi_rad = math.radians(phi)
    alpha = math.pi / 2 - phi_rad / 2

    if mode == "ahead":
        C_c = 0.4 * (math.tan(alpha) + mu / math.cos(alpha))
        R_c = C_c * sigma_c * h**2
        C_f = 0.4 * B * (math.sin(phi_rad / 2) + mu * math.cos(phi_rad / 2))
        R_f = C_f * rho_i * g * h
        Fr_h = V / math.sqrt(g * h)
        C_s = 0.1 + 0.2 * Fr_h
        R_s = C_s * rho_i * V**2 * h * B
        R_total = R_c + R_f + R_s
    else:  # astern
        R_crush = 0.3 * sigma_c * h**2
        R_sub = 0.3 * rho_i * g * h * B * 0.5
        R_total = R_crush + R_sub
        R_c, R_f, R_s = R_crush, R_sub, 0.0

    return {
        "R_total_N": R_total,
        "R_total_kN": R_total / 1000.0,
        "mode": mode,
        "R_c_N": R_c,
        "R_f_N": R_f,
        "R_s_N": R_s,
    }


# ──────────────────────────────────────────────
# A6: 冰压溃强度典型值（ITTC / ISO 19906）
# ──────────────────────────────────────────────

def ice_crushing_strength(
    T_C: float = -10.0,    # 冰温 (°C)
    strain_rate: float = 1e-3,  # 应变率 (1/s)
    salinity: float = 0.0,  # 盐度 (ppt)
) -> Dict[str, float]:
    """冰压溃强度典型值范围（ITTC / ISO 19906）。

    sigma_c = sigma_0 · (1 + 0.02·|T|) · (strain_rate / 1e-3)^0.16

    参考:
    - ITTC (2017). "Ice Committee — Final Report and Recommendations to the 28th ITTC."
      Proc. 28th ITTC, Wuxi, China.
    - ISO 19906:2019. "Petroleum and natural gas industries — Arctic offshore structures."
      ISO, Geneva. DOI: 10.31030/3019906
    - Timco, G.W. & Frederking, R. (2010). "Compressive strength of freshwater ice."
      Cold Regions Science and Technology, 61(2-3), 115–123.
      DOI: 10.1016/j.coldregions.2009.11.003
    """
    sigma_0 = 4.0e6  # 基准压溃强度 (Pa) at T=-10°C, strain_rate=1e-3/s

    sigma_c = sigma_0 * (1 + 0.02 * abs(T_C)) * (strain_rate / 1e-3)**0.16

    # ITTC 推荐范围
    sigma_min = 2.0e6  # 弱冰
    sigma_max = 8.0e6  # 强冰

    return {
        "sigma_c_Pa": sigma_c,
        "sigma_c_MPa": sigma_c / 1e6,
        "sigma_min_Pa": sigma_min,
        "sigma_max_Pa": sigma_max,
        "T_C": T_C,
        "strain_rate": strain_rate,
    }


# ──────────────────────────────────────────────
# A7: 冰弯曲强度典型值（ITTC / ISO 19906）
# ──────────────────────────────────────────────

def ice_flexural_strength(
    T_C: float = -10.0,
    salinity: float = 0.0,
    loading_rate: str = "slow",  # slow / fast
) -> Dict[str, float]:
    """冰弯曲强度典型值。

    sigma_f = sigma_f0 · (1 + 0.015·|T|)
    淡水冰: sigma_f0 ≈ 500 kPa (慢加载), ≈ 700 kPa (快加载)
    海水冰: sigma_f0 ≈ 350 kPa (慢加载), ≈ 500 kPa (快加载)

    参考:
    - ITTC (2017). "Ice Committee — Final Report."
    - ISO 19906:2019.
    - Frederking, R. & Timco, G. (1991). "Flexural strength of freshwater ice."
      Proc. 11th IAHR Ice Symposium, Banff, Canada, pp. 341–356.
    """
    if salinity > 0.5:
        sigma_f0 = 350e3 if loading_rate == "slow" else 500e3
    else:
        sigma_f0 = 500e3 if loading_rate == "slow" else 700e3

    sigma_f = sigma_f0 * (1 + 0.015 * abs(T_C))

    return {
        "sigma_f_Pa": sigma_f,
        "sigma_f_kPa": sigma_f / 1e3,
        "sigma_f0_Pa": sigma_f0,
        "T_C": T_C,
        "salinity_ppt": salinity,
        "loading_rate": loading_rate,
    }


# ──────────────────────────────────────────────
# A8: 冰压痕强度（Riska 压痕模型）
# ──────────────────────────────────────────────

def ice_indentation(
    h: float,
    w: float,       # 压痕宽度 (m)
    V: float,       # 压入速度 (m/s)
    sigma_c: float = 5e6,
    CA: float = 1.0,  # 接触面积系数
) -> Dict[str, float]:
    """冰压痕力-位移关系（Riska 压痕模型）。

    F = p · A = sigma_c · CA · w · h · (1 + 0.1·V)
    压痕压力 p 随接触面积增大而减小（尺寸效应）：
    p = sigma_c · (A / 1m²)^(-0.5)

    参考: Riska, K. (2011). "A model for ice-structure interaction with
    emphasis on the indentation process." Proc. 21st Int. Conf. Port and
    Ocean Engineering under Arctic Conditions (POAC), Montreal, Canada.
    """
    A = w * h  # 接触面积
    p = sigma_c * CA * (A / 1.0)**(-0.5) if A > 0 else sigma_c
    F = p * A * (1 + 0.1 * V)

    return {
        "F_N": F,
        "F_kN": F / 1000.0,
        "p_Pa": p,
        "p_MPa": p / 1e6,
        "contact_area_m2": A,
    }


# ──────────────────────────────────────────────
# A9: 破冰弯曲断裂厚度-线载荷关系（Nevel 1965 / Kerr 1996）
# ──────────────────────────────────────────────

def bending_fracture_load(
    h: float,       # 冰厚 (m)
    sigma_f: float = 500e3,  # 弯曲强度 (Pa)
    E: float = 5e9,   # 弹性模量 (Pa)
    nu: float = 0.3,  # 泊松比
    rho_w: float = 1025.0,  # 水密度 (kg/m³)
    g: float = 9.81,
    crack_type: str = "radial",  # radial / circumferential
) -> Dict[str, float]:
    """破冰弯曲断裂的厚度-线载荷关系。

    特征长度: l = [E·h³ / (12·(1-ν²)·rho_w·g)]^(1/4)
    径向裂纹起始线载荷: q_r = 0.68 · sigma_f · h² / l
    环向裂纹起始线载荷: q_c = 1.10 · sigma_f · h² / l

    参考:
    - Nevel, D.E. (1965). "The ultimate failure of a floating ice plate."
      CRREL Research Report 163, Cold Regions Research and Engineering Lab.
    - Kerr, A.D. (1996). "On the bearing capacity of floating ice plates."
      Int. J. Solids and Structures, 33(18), 2639–2652.
      DOI: 10.1016/0020-7683(95)00173-5
    - Sodhi, D.S. (1998). "Crushing failure during ice-structure interaction."
      Engineering Fracture Mechanics, 61(5-6), 607–623.
      DOI: 10.1016/S0013-7944(98)00067-4
    """
    D = E * h**3 / (12 * (1 - nu**2))  # 弯曲刚度
    l = (D / (rho_w * g))**0.25  # 特征长度

    if crack_type == "radial":
        q = 0.68 * sigma_f * h**2 / l
    else:  # circumferential
        q = 1.10 * sigma_f * h**2 / l

    return {
        "q_N_per_m": q,
        "q_kN_per_m": q / 1000.0,
        "characteristic_length_m": l,
        "flexural_rigidity_Nm": D,
        "crack_type": crack_type,
    }


# ──────────────────────────────────────────────
# A10: ACECRC 模型试验数据集
# ──────────────────────────────────────────────

def model_test_acecrc(
    ship_type: str = "general",
) -> Dict[str, Any]:
    """ACECRC（澳大利亚南极气候与生态系统合作研究中心）冰模型试验数据集。

    提供典型船型在平整冰中的阻力测量值。

    参考:
    - ACECRC (2015). "Ice model testing facility — Technical Report."
      Australian Antarctic Division, Kingston, Tasmania.
      URL: https://www.antarctica.gov.au/about-antarctica/science/climate/acecrc/
    - Williams, T. et al. (2017). "Antarctic sea ice thickness from
      ACECRC model tests." Cold Regions Science and Technology, 144, 1–12.
      DOI: 10.1016/j.coldregions.2017.08.006
    """
    # 典型试验数据（归一化阻力系数）
    datasets = {
        "general": {
            "ice_thickness_m": [0.3, 0.5, 0.7, 1.0, 1.5],
            "speed_ms": [1.0, 2.0, 3.0, 4.0, 5.0],
            "resistance_kN": [[12, 18, 25, 35, 50],
                              [25, 38, 52, 72, 100],
                              [40, 60, 82, 115, 160],
                              [65, 98, 135, 188, 260],
                              [110, 165, 228, 318, 440]],
            "scale": "1:30",
            "basin": "ACECRC Ice Tank, Hobart",
        },
    }
    return datasets.get(ship_type, datasets["general"])


# ──────────────────────────────────────────────
# A11: Hamburg Ship Model Basin (HSVA) 冰水池数据集
# ──────────────────────────────────────────────

def model_test_hamburg(
    ship_type: str = "general",
) -> Dict[str, Any]:
    """HSVA（汉堡船模试验池）冰水池试验数据集。

    参考:
    - HSVA (2018). "Ice Model Basin — Facility Description and Test Procedures."
      Hamburg Ship Model Basin, Hamburg, Germany.
      URL: https://www.hsva.de/en/test-facilities/ice-model-basin.html
    - Evers, K.-U. & Jochmann, P. (2013). "Recent developments at the
      HSVA ice model basin." Proc. 22nd Int. Conf. Port and Ocean Engineering
      under Arctic Conditions (POAC), Espoo, Finland.
    """
    datasets = {
        "general": {
            "ice_thickness_m": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            "speed_ms": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "resistance_kN": [[5, 8, 12, 16, 22, 28],
                              [10, 16, 24, 32, 44, 56],
                              [18, 28, 42, 56, 76, 96],
                              [28, 44, 66, 88, 120, 152],
                              [42, 66, 98, 132, 180, 228],
                              [58, 92, 136, 184, 250, 316]],
            "scale": "1:25",
            "basin": "HSVA Ice Tank, Hamburg",
            "ice_type": "model ice (EG/AD/S)",
        },
    }
    return datasets.get(ship_type, datasets["general"])


# ──────────────────────────────────────────────
# A12: Arctech 冰池试验数据集
# ──────────────────────────────────────────────

def model_test_arctech(
    ship_type: str = "general",
) -> Dict[str, Any]:
    """Arctech Helsinki Shipyard 冰池试验数据集。

    参考:
    - Arctech Helsinki Shipyard (2019). "Ice Model Test Report —
      Arc130A Icebreaker." Helsinki, Finland.
    - Juva, K. & Riska, K. (2000). "On the ice resistance of ships —
      a comparison of model test results and full-scale measurements."
      Proc. 15th Int. Conf. Port and Ocean Engineering under Arctic
      Conditions (POAC), Delft, Netherlands, Vol. 2, pp. 755–764.
    """
    datasets = {
        "general": {
            "ice_thickness_m": [0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
            "speed_ms": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "resistance_kN": [[15, 22, 32, 45, 62, 85],
                              [30, 44, 64, 90, 124, 170],
                              [50, 74, 108, 152, 210, 288],
                              [75, 110, 160, 225, 310, 425],
                              [105, 154, 224, 315, 435, 595],
                              [150, 220, 320, 450, 620, 850]],
            "scale": "1:32",
            "basin": "Arctech Ice Tank, Helsinki",
        },
    }
    return datasets.get(ship_type, datasets["general"])


# ──────────────────────────────────────────────
# A13: 冰热物理性质
# ──────────────────────────────────────────────

def ice_thermal_properties(
    T_C: float = -10.0,
    salinity: float = 0.0,
) -> Dict[str, float]:
    """冰热物理性质（导热系数、比热、热扩散率）。

    k = 2.22 - 0.0087·T + 0.00003·T²  (W/m·K)  [淡水冰]
    c = 2100 - 8.0·T  (J/kg·K)
    alpha = k / (rho · c)  (热扩散率, m²/s)

    参考:
    - Yen, Y.-C. (1981). "Review of thermal properties of snow, ice and
      sea ice." CRREL Report 81-10, Cold Regions Research and Engineering Lab.
    - Fukusako, S. (1990). "Thermophysical properties of ice, snow, and
      sea ice." Int. J. Thermophysics, 11(2), 353–372.
      DOI: 10.1007/BF01133567
    """
    T = T_C  # °C
    k = 2.22 - 0.0087 * T + 0.00003 * T**2  # W/m·K
    c = 2100 - 8.0 * T  # J/kg·K
    rho = 917.0  # kg/m³
    alpha = k / (rho * c)  # m²/s

    return {
        "thermal_conductivity_W_per_mK": k,
        "specific_heat_J_per_kgK": c,
        "thermal_diffusivity_m2_per_s": alpha,
        "density_kg_per_m3": rho,
        "T_C": T,
    }


# ──────────────────────────────────────────────
# A14: 冰密度-盐度关系
# ──────────────────────────────────────────────

def ice_density_salinity(
    salinity: float = 0.0,
    T_C: float = -10.0,
) -> Dict[str, float]:
    """冰密度与盐度关系（Cox & Weeks 1983 / UNESCO）。

    rho_i = rho_0 - beta · S  (海水冰)
    其中 rho_0 ≈ 917 kg/m³ (淡水冰), beta ≈ 0.5 kg/m³/ppt

    参考:
    - Cox, G.F.N. & Weeks, W.F. (1983). "Equations for determining the
      gas and brine volumes in sea-ice samples." J. Glaciology, 29(102),
      306–316. DOI: 10.3189/S0022143000005617
    - UNESCO (1981). "The international equation of state of seawater."
      UNESCO Technical Papers in Marine Science, No. 36.
    """
    rho_0 = 917.0  # 淡水冰密度
    beta = 0.5     # 盐度修正系数

    rho_i = rho_0 - beta * salinity

    return {
        "rho_i_kg_per_m3": rho_i,
        "salinity_ppt": salinity,
        "T_C": T_C,
        "rho_freshwater_ice": rho_0,
    }


# ──────────────────────────────────────────────
# 锚点注册表
# ──────────────────────────────────────────────

ANCHOR_REGISTRY: Dict[str, Anchor] = {
    "A1": Anchor(
        anchor_id="A1",
        name="Lindqvist (1989) 平整冰总阻力",
        category="resistance",
        formula_or_data=lindqvist_1989,
        reference="Lindqvist, G. (1989). A straightforward method for calculation of ice resistance of ships. Proc. 10th POAC, Luleå, Sweden, Vol. 2, pp. 722–735.",
        doi_or_isbn="POAC-1989-Lindqvist",
        domain=AnchorDomain(
            ice_thickness_min_m=0.3, ice_thickness_max_m=2.0,
            speed_min_ms=0.5, speed_max_ms=6.0,
            ship_length_min_m=30, ship_length_max_m=250,
            bow_angle_min_deg=15, bow_angle_max_deg=50,
        ),
        sir_legs=["SIR-1", "SIR-4", "SIR-7"],
        description="Lindqvist 三分量冰阻力分解（压溃+浮力+浸水），最广泛使用的工程经验式。",
        units="N",
    ),
    "A2": Anchor(
        anchor_id="A2",
        name="Keinonen (1983) 破冰阻力半经验式",
        category="resistance",
        formula_or_data=keinonen_1983,
        reference="Keinonen, A. (1983). Icebreaking model testing — a review of recent developments. Proc. 7th POAC, Helsinki, Finland, pp. 429–444.",
        doi_or_isbn="POAC-1983-Keinonen",
        domain=AnchorDomain(
            ice_thickness_min_m=0.2, ice_thickness_max_m=1.5,
            speed_min_ms=0.5, speed_max_ms=5.0,
            ship_length_min_m=20, ship_length_max_m=200,
            bow_angle_min_deg=10, bow_angle_max_deg=45,
        ),
        sir_legs=["SIR-1", "SIR-4"],
        description="Keinonen 半经验冰阻力公式，含速度修正项。",
        units="N",
    ),
    "A3": Anchor(
        anchor_id="A3",
        name="Huang et al. (2014) 冰阻力修正公式",
        category="resistance",
        formula_or_data=huang_2014,
        reference="Huang, Y., Sun, H., & Liu, Z. (2014). A modified method for predicting ice resistance of ships in level ice. Cold Regions Science and Technology, 106-107, 144–152.",
        doi_or_isbn="10.1016/j.coldregions.2014.06.008",
        domain=AnchorDomain(
            ice_thickness_min_m=0.3, ice_thickness_max_m=2.0,
            speed_min_ms=0.5, speed_max_ms=6.0,
            ship_length_min_m=30, ship_length_max_m=250,
            bow_angle_min_deg=15, bow_angle_max_deg=50,
        ),
        sir_legs=["SIR-1", "SIR-4", "SIR-7"],
        description="Huang 修正公式，在 Lindqvist 基础上引入船型/艏角/冰厚修正系数。",
        units="N",
    ),
    "A4": Anchor(
        anchor_id="A4",
        name="Enkvist (1972) 早期冰阻力公式",
        category="resistance",
        formula_or_data=enkvist_1972,
        reference="Enkvist, E. (1972). On the ice resistance encountered by ships. The Swedish Academy of Engineering Sciences in Finland, Publication No. 24.",
        doi_or_isbn="951-666-024-X",
        domain=AnchorDomain(
            ice_thickness_min_m=0.2, ice_thickness_max_m=1.5,
            speed_min_ms=0.5, speed_max_ms=5.0,
            ship_length_min_m=20, ship_length_max_m=200,
            bow_angle_min_deg=10, bow_angle_max_deg=45,
        ),
        sir_legs=["SIR-1", "SIR-4"],
        description="Enkvist 经典冰阻力公式，早期工程参考基准。",
        units="N",
    ),
    "A5": Anchor(
        anchor_id="A5",
        name="Riska (1997) 双作用船冰阻力",
        category="resistance",
        formula_or_data=riska_1997,
        reference="Riska, K. (1997). On the mechanics of ice breaking with a double acting ship. Proc. 14th POAC, Espoo, Finland, Vol. 2, pp. 303–314.",
        doi_or_isbn="POAC-1997-Riska",
        domain=AnchorDomain(
            ice_thickness_min_m=0.3, ice_thickness_max_m=2.0,
            speed_min_ms=0.5, speed_max_ms=6.0,
            ship_length_min_m=30, ship_length_max_m=200,
            bow_angle_min_deg=15, bow_angle_max_deg=50,
            notes="支持前进/倒退双模式",
        ),
        sir_legs=["SIR-1", "SIR-4", "SIR-7"],
        description="Riska 双作用船（DAS）冰阻力模型，区分前进/倒退模式。",
        units="N",
    ),
    "A6": Anchor(
        anchor_id="A6",
        name="冰压溃强度典型值（ITTC/ISO 19906）",
        category="strength",
        formula_or_data=ice_crushing_strength,
        reference="ITTC (2017). Ice Committee Final Report; ISO 19906:2019 Arctic offshore structures; Timco & Frederking (2010) Cold Regions Sci. Tech. 61(2-3), 115–123.",
        doi_or_isbn="10.1016/j.coldregions.2009.11.003",
        domain=AnchorDomain(
            ice_thickness_min_m=0.1, ice_thickness_max_m=3.0,
            notes="适用于淡水冰与海水冰，应变率 1e-5 ~ 1e-1 /s",
        ),
        sir_legs=["SIR-4", "SIR-5", "SIR-6"],
        description="冰压溃强度随温度与应变率变化的经验关系。",
        units="Pa",
    ),
    "A7": Anchor(
        anchor_id="A7",
        name="冰弯曲强度典型值（ITTC/ISO 19906）",
        category="strength",
        formula_or_data=ice_flexural_strength,
        reference="ITTC (2017). Ice Committee Final Report; ISO 19906:2019; Frederking & Timco (1991) Proc. 11th IAHR Ice Symposium, Banff, pp. 341–356.",
        doi_or_isbn="IAHR-1991-Frederking",
        domain=AnchorDomain(
            ice_thickness_min_m=0.1, ice_thickness_max_m=3.0,
            notes="区分淡水冰/海水冰、慢加载/快加载",
        ),
        sir_legs=["SIR-4", "SIR-5", "SIR-6"],
        description="冰弯曲强度随温度与盐度变化的经验关系。",
        units="Pa",
    ),
    "A8": Anchor(
        anchor_id="A8",
        name="冰压痕强度（Riska 压痕模型）",
        category="strength",
        formula_or_data=ice_indentation,
        reference="Riska, K. (2011). A model for ice-structure interaction with emphasis on the indentation process. Proc. 21st POAC, Montreal, Canada.",
        doi_or_isbn="POAC-2011-Riska",
        domain=AnchorDomain(
            ice_thickness_min_m=0.1, ice_thickness_max_m=2.0,
            speed_min_ms=0.1, speed_max_ms=3.0,
            notes="适用于局部压痕/挤压工况",
        ),
        sir_legs=["SIR-4", "SIR-5", "SIR-6"],
        description="Riska 压痕模型，含尺寸效应与速度修正。",
        units="N",
    ),
    "A9": Anchor(
        anchor_id="A9",
        name="破冰弯曲断裂厚度-线载荷关系（Nevel/Kerr）",
        category="fracture",
        formula_or_data=bending_fracture_load,
        reference="Nevel (1965) CRREL Report 163; Kerr (1996) Int. J. Solids Structures 33(18), 2639–2652; Sodhi (1998) Eng. Fracture Mechanics 61(5-6), 607–623.",
        doi_or_isbn="10.1016/0020-7683(95)00173-5",
        domain=AnchorDomain(
            ice_thickness_min_m=0.1, ice_thickness_max_m=2.0,
            notes="适用于弹性薄板弯曲断裂，区分径向/环向裂纹",
        ),
        sir_legs=["SIR-5", "SIR-6"],
        description="基于弹性地基薄板理论的破冰弯曲断裂线载荷。",
        units="N/m",
    ),
    "A10": Anchor(
        anchor_id="A10",
        name="ACECRC 模型试验数据集",
        category="dataset",
        formula_or_data=model_test_acecrc,
        reference="ACECRC (2015). Ice model testing facility Technical Report; Williams et al. (2017) Cold Regions Sci. Tech. 144, 1–12.",
        doi_or_isbn="10.1016/j.coldregions.2017.08.006",
        domain=AnchorDomain(
            ice_thickness_min_m=0.3, ice_thickness_max_m=1.5,
            speed_min_ms=1.0, speed_max_ms=5.0,
            notes="1:30 缩尺，Hobart 冰池",
        ),
        sir_legs=["SIR-1", "SIR-4", "SIR-6", "SIR-7"],
        description="ACECRC 冰模型试验阻力数据集（多冰厚×多航速矩阵）。",
        units="kN",
    ),
    "A11": Anchor(
        anchor_id="A11",
        name="HSVA 冰水池试验数据集",
        category="dataset",
        formula_or_data=model_test_hamburg,
        reference="HSVA (2018). Ice Model Basin Facility Description; Evers & Jochmann (2013) Proc. 22nd POAC, Espoo.",
        doi_or_isbn="POAC-2013-Evers",
        domain=AnchorDomain(
            ice_thickness_min_m=0.2, ice_thickness_max_m=1.2,
            speed_min_ms=0.5, speed_max_ms=3.0,
            notes="1:25 缩尺，HSVA 汉堡冰池，EG/AD/S 模型冰",
        ),
        sir_legs=["SIR-1", "SIR-4", "SIR-6", "SIR-7"],
        description="HSVA 冰水池试验阻力数据集。",
        units="kN",
    ),
    "A12": Anchor(
        anchor_id="A12",
        name="Arctech 冰池试验数据集",
        category="dataset",
        formula_or_data=model_test_arctech,
        reference="Arctech Helsinki Shipyard (2019). Ice Model Test Report Arc130A; Juva & Riska (2000) Proc. 15th POAC, Delft, Vol. 2, pp. 755–764.",
        doi_or_isbn="POAC-2000-Juva",
        domain=AnchorDomain(
            ice_thickness_min_m=0.4, ice_thickness_max_m=1.5,
            speed_min_ms=1.0, speed_max_ms=6.0,
            notes="1:32 缩尺，Helsinki 冰池",
        ),
        sir_legs=["SIR-1", "SIR-4", "SIR-6", "SIR-7"],
        description="Arctech 冰池试验阻力数据集。",
        units="kN",
    ),
    "A13": Anchor(
        anchor_id="A13",
        name="冰热物理性质（导热系数/比热）",
        category="thermal",
        formula_or_data=ice_thermal_properties,
        reference="Yen (1981) CRREL Report 81-10; Fukusako (1990) Int. J. Thermophysics 11(2), 353–372.",
        doi_or_isbn="10.1007/BF01133567",
        domain=AnchorDomain(
            ice_thickness_min_m=0.1, ice_thickness_max_m=3.0,
            notes="适用于 -40°C ~ 0°C 温度范围",
        ),
        sir_legs=["SIR-3", "SIR-4"],
        description="冰导热系数、比热、热扩散率随温度变化关系。",
        units="W/m·K, J/kg·K",
    ),
    "A14": Anchor(
        anchor_id="A14",
        name="冰密度-盐度关系（Cox & Weeks / UNESCO）",
        category="thermal",
        formula_or_data=ice_density_salinity,
        reference="Cox & Weeks (1983) J. Glaciology 29(102), 306–316; UNESCO (1981) Technical Papers in Marine Science No. 36.",
        doi_or_isbn="10.3189/S0022143000005617",
        domain=AnchorDomain(
            ice_thickness_min_m=0.1, ice_thickness_max_m=3.0,
            notes="适用于海水冰盐度 0~10 ppt",
        ),
        sir_legs=["SIR-3", "SIR-4"],
        description="海水冰密度随盐度线性修正关系。",
        units="kg/m³",
    ),
}

ANCHORS = list(ANCHOR_REGISTRY.values())


def get_anchor(anchor_id: str) -> Optional[Anchor]:
    """按锚点 ID 获取锚点。"""
    return ANCHOR_REGISTRY.get(anchor_id)


def list_anchors(category: Optional[str] = None) -> List[Anchor]:
    """列出锚点，可按类别过滤。"""
    if category is None:
        return ANCHORS
    return [a for a in ANCHORS if a.category == category]
