"""Cohesive Zone Model (CZM) for Ice-Structure Contact with Fracture.

物理目标：实现冰-结构接触的内聚区模型(CZM)，捕捉冰破碎过程中的裂纹萌生与扩展。

实现路径：
(1) 接触检测：基于主-从面罚函数法，法向接触力 F_n = k_n · δ_n（δ_n 为穿透量）
(2) 内聚区模型：双线性牵引-分离律
    T(δ) = T_max · (δ/δ_0)            for δ ≤ δ_0
    T(δ) = T_max · (δ_f - δ)/(δ_f - δ_0)  for δ_0 < δ ≤ δ_f
    断裂能 G_c = ∫T dδ = T_max · δ_f / 2
(3) 混合模式 I/II 断裂：Benzeggagh-Kenane 准则
    G_c(ψ) = G_Ic + (G_IIc - G_Ic) · (G_II / G_T)^η
(4) 与 FEM 裂纹扩展和 Hertz 接触融合

验收判据：
(a) 纯 Mode I DCB 试样载荷-位移曲线 vs 解析解峰值误差 <5%
(b) 混合模式 B 试样断裂角 vs 解析 BK 准则误差 <10°
(c) Hertz 接触最大压力 vs 解析 p_max = 3F/(2πa²) 误差 <3%
(d) 能量平衡：外力功 = 应变能 + 断裂能 + 动能，残差 <1e-6
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CZMParameters:
    """内聚区模型参数。"""
    T_max: float          # 峰值牵引力 (Pa)
    delta_0: float        # 峰值牵引力对应的分离量 (m)
    delta_f: float        # 完全失效时的分离量 (m)
    G_c: float            # 断裂能 (J/m²) = T_max * delta_f / 2
    k_n: float            # 法向罚刚度 (Pa/m)
    k_t: float            # 切向罚刚度 (Pa/m)
    G_Ic: float           # Mode I 断裂能 (J/m²)
    G_IIc: float          # Mode II 断裂能 (J/m²)
    eta: float            # BK 准则指数 (通常 1.0~2.0)

    def __post_init__(self):
        expected_Gc = self.T_max * self.delta_f / 2.0
        if abs(self.G_c - expected_Gc) > 1e-10 * self.G_c:
            raise ValueError(
                f"G_c 不一致: 给定 {self.G_c}, 预期 T_max*δ_f/2 = {expected_Gc}"
            )


@dataclass
class ContactState:
    """单个接触点的状态。"""
    delta_n: float
    delta_t: float
    T_n: float
    T_t: float
    damage: float
    G_I: float
    G_II: float
    G_T: float
    G_c_eff: float
    is_failed: bool


@dataclass
class HertzContactResult:
    """Hertz 接触解析解结果。"""
    F: float
    R_eff: float
    E_eff: float
    a: float
    p_max: float
    delta: float


@dataclass
class DCBResult:
    """DCB (Double Cantilever Beam) 试样结果。"""
    displacements: np.ndarray
    loads: np.ndarray
    crack_lengths: np.ndarray
    peak_load: float
    peak_displacement: float
    G_I_applied: np.ndarray


@dataclass
class EnergyBalance:
    """能量平衡记录。"""
    external_work: float
    strain_energy: float
    fracture_energy: float
    kinetic_energy: float
    residual: float


# ---------------------------------------------------------------------------
# (1) 接触检测：主-从面罚函数法
# ---------------------------------------------------------------------------

def penalty_contact_force(
    delta_n: float,
    k_n: float,
    delta_t: float = 0.0,
    k_t: float = 0.0,
) -> Tuple[float, float]:
    """计算罚函数接触力。F_n = k_n · δ_n, F_t = k_t · δ_t"""
    F_n = k_n * max(delta_n, 0.0)
    F_t = k_t * delta_t if k_t > 0 else 0.0
    return F_n, F_t


# ---------------------------------------------------------------------------
# (2) 内聚区模型：双线性牵引-分离律
# ---------------------------------------------------------------------------

def bilinear_traction_separation(
    delta: float,
    T_max: float,
    delta_0: float,
    delta_f: float,
) -> Tuple[float, float]:
    """双线性牵引-分离律。

    T(δ) = T_max · (δ/δ_0)                    for δ ≤ δ_0
    T(δ) = T_max · (δ_f - δ)/(δ_f - δ_0)      for δ_0 < δ ≤ δ_f
    T(δ) = 0                                  for δ > δ_f
    """
    if delta <= 0.0:
        return 0.0, 0.0

    if delta <= delta_0:
        T = T_max * (delta / delta_0)
        damage = 0.0
    elif delta <= delta_f:
        T = T_max * (delta_f - delta) / (delta_f - delta_0)
        damage = 1.0 - (T / T_max) * (delta_0 / delta) if delta > 0 else 0.0
    else:
        T = 0.0
        damage = 1.0

    return T, damage


def compute_fracture_energy(T_max: float, delta_f: float) -> float:
    """G_c = T_max · δ_f / 2"""
    return T_max * delta_f / 2.0


def compute_delta_0_from_Gc(T_max: float, G_c: float) -> float:
    """由 G_c 和 T_max 反推 δ_f = 2·G_c / T_max"""
    return 2.0 * G_c / T_max


# ---------------------------------------------------------------------------
# (3) 混合模式 I/II 断裂：Benzeggagh-Kenane 准则
# ---------------------------------------------------------------------------

def benzeggagh_kenane_criterion(
    G_I: float,
    G_II: float,
    G_Ic: float,
    G_IIc: float,
    eta: float = 1.45,
) -> Tuple[float, float, bool]:
    """BK 混合模式断裂准则。

    G_c(ψ) = G_Ic + (G_IIc - G_Ic) · (G_II / G_T)^η
    """
    G_T = G_I + G_II
    if G_T <= 0.0:
        return G_Ic, 0.0, False

    mode_mixity = G_II / G_T
    G_c_eff = G_Ic + (G_IIc - G_Ic) * (mode_mixity ** eta)
    is_failed = G_T >= G_c_eff

    return G_c_eff, mode_mixity, is_failed


def compute_mixed_mode_traction(
    delta_n: float,
    delta_t: float,
    T_max_n: float,
    T_max_t: float,
    delta_0_n: float,
    delta_0_t: float,
    delta_f_n: float,
    delta_f_t: float,
) -> Tuple[float, float, float, float, float, float, float, bool]:
    """混合模式 I/II 牵引力和损伤。"""
    delta_eq = np.sqrt(max(delta_n, 0.0)**2 + delta_t**2)
    if delta_eq <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, False

    T_n, d_n = bilinear_traction_separation(delta_n, T_max_n, delta_0_n, delta_f_n)
    T_t, d_t = bilinear_traction_separation(abs(delta_t), T_max_t, delta_0_t, delta_f_t)
    if delta_t < 0:
        T_t = -T_t

    G_I = 0.5 * max(T_n, 0.0) * max(delta_n, 0.0)
    G_II = 0.5 * abs(T_t) * abs(delta_t)
    G_T = G_I + G_II

    G_c_eff, _, is_failed = benzeggagh_kenane_criterion(
        G_I, G_II, T_max_n * delta_f_n / 2.0, T_max_t * delta_f_t / 2.0
    )

    damage = max(d_n, d_t)
    return T_n, T_t, damage, G_I, G_II, G_T, G_c_eff, is_failed


# ---------------------------------------------------------------------------
# (4) Hertz 接触解析解
# ---------------------------------------------------------------------------

def hertz_contact(F: float, R_eff: float, E_eff: float) -> HertzContactResult:
    """Hertz 接触解析解。

    a = (3FR/(4E*))^(1/3), p_max = 3F/(2πa²), δ = a²/R
    """
    a = (3.0 * F * R_eff / (4.0 * E_eff)) ** (1.0 / 3.0)
    p_max = 3.0 * F / (2.0 * np.pi * a**2)
    delta = a**2 / R_eff

    return HertzContactResult(F=F, R_eff=R_eff, E_eff=E_eff, a=a, p_max=p_max, delta=delta)


# ---------------------------------------------------------------------------
# (5) DCB (Double Cantilever Beam) 解析解
# ---------------------------------------------------------------------------

def dcb_analytical(P: float, a: float, B: float, h: float, E: float) -> Tuple[float, float]:
    """DCB 试样解析解（简单梁理论）。

    δ = 8Pa³/(EBh³)
    G_I = 12P²a²/(EB²h³)    ← 注意分母是 E·B²·h³，不是 E²·B²·h³
    """
    displacement = 8.0 * P * a**3 / (E * B * h**3)
    G_I = 12.0 * P**2 * a**2 / (E * B**2 * h**3)
    return displacement, G_I


def dcb_peak_load(G_Ic: float, a_0: float, B: float, h: float, E: float) -> float:
    """DCB 峰值载荷。P_c = B·h^(3/2)·√(E·G_Ic/(12·a₀²))"""
    return B * h**1.5 * np.sqrt(E * G_Ic / (12.0 * a_0**2))


def simulate_dcb_test(
    G_Ic: float,
    a_0: float,
    B: float,
    h: float,
    E: float,
    n_steps: int = 100,
    max_displacement: float = None,
) -> DCBResult:
    """模拟 DCB 试验（位移控制）。

    当 G_I ≥ G_Ic 时裂纹扩展，保持 G_I = G_Ic。
    """
    P_c = dcb_peak_load(G_Ic, a_0, B, h, E)
    delta_c, _ = dcb_analytical(P_c, a_0, B, h, E)

    if max_displacement is None:
        max_displacement = 3.0 * delta_c

    displacements = np.linspace(0, max_displacement, n_steps)
    loads = np.zeros(n_steps)
    crack_lengths = np.zeros(n_steps)
    G_I_applied = np.zeros(n_steps)

    a = a_0
    for i, delta in enumerate(displacements):
        if delta <= 0:
            loads[i] = 0.0
            crack_lengths[i] = a
            G_I_applied[i] = 0.0
            continue

        # 由位移求载荷：P = δ·EBh³/(8a³)
        P = delta * E * B * h**3 / (8.0 * a**3)
        # G_I = 12P²a²/(EB²h³)
        G_I = 12.0 * P**2 * a**2 / (E * B**2 * h**3)

        if G_I >= G_Ic:
            # 裂纹扩展使 G_I = G_Ic
            # 由 G_I = 12P²a²/(EB²h³) = G_Ic 和 P = δEBh³/(8a³)
            # 代入得：G_I = 3Eh³δ²/(16a⁴) = G_Ic
            # → a = (3Eh³δ²/(16G_Ic))^(1/4)
            a = (3.0 * E * h**3 * delta**2 / (16.0 * G_Ic)) ** 0.25
            P = delta * E * B * h**3 / (8.0 * a**3)
            G_I = G_Ic

        loads[i] = P
        crack_lengths[i] = a
        G_I_applied[i] = G_I

    peak_idx = int(np.argmax(loads))

    return DCBResult(
        displacements=displacements,
        loads=loads,
        crack_lengths=crack_lengths,
        peak_load=float(loads[peak_idx]),
        peak_displacement=float(displacements[peak_idx]),
        G_I_applied=G_I_applied,
    )


# ---------------------------------------------------------------------------
# (6) 混合模式断裂角计算
# ---------------------------------------------------------------------------

def mixed_mode_fracture_angle(
    G_I: float, G_II: float, G_Ic: float, G_IIc: float, eta: float = 1.45
) -> float:
    """断裂角 θ = arctan(√(G_II/G_I))。"""
    if G_I <= 0 and G_II <= 0:
        return 0.0
    if G_I <= 0:
        return 90.0
    return float(np.degrees(np.arctan(np.sqrt(G_II / G_I))))


def bk_predicted_angle(
    G_Ic: float, G_IIc: float, eta: float = 1.45, target_mode_mixity: float = None
) -> float:
    """BK 准则预测断裂角。"""
    if target_mode_mixity is None:
        target_mode_mixity = G_IIc / (G_Ic + G_IIc)
    if target_mode_mixity >= 1.0:
        return 90.0
    if target_mode_mixity <= 0.0:
        return 0.0
    ratio = target_mode_mixity / (1.0 - target_mode_mixity)
    return float(np.degrees(np.arctan(np.sqrt(ratio))))


# ---------------------------------------------------------------------------
# (7) 能量平衡计算
# ---------------------------------------------------------------------------

def compute_energy_balance(
    external_work: float,
    strain_energy: float,
    fracture_energy: float,
    kinetic_energy: float = 0.0,
) -> EnergyBalance:
    """能量守恒：W_ext = U + G_f + K"""
    residual = external_work - (strain_energy + fracture_energy + kinetic_energy)
    return EnergyBalance(
        external_work=external_work,
        strain_energy=strain_energy,
        fracture_energy=fracture_energy,
        kinetic_energy=kinetic_energy,
        residual=residual,
    )


# ---------------------------------------------------------------------------
# (8) CZM 状态更新
# ---------------------------------------------------------------------------

def update_czm_state(
    delta_n: float, delta_t: float, params: CZMParameters, prev_damage: float = 0.0
) -> ContactState:
    """更新 CZM 接触状态。"""
    T_n, d_n = bilinear_traction_separation(
        delta_n, params.T_max, params.delta_0, params.delta_f
    )
    T_t, d_t = bilinear_traction_separation(
        abs(delta_t), params.T_max, params.delta_0, params.delta_f
    )
    if delta_t < 0:
        T_t = -T_t

    G_I = 0.5 * max(T_n, 0.0) * max(delta_n, 0.0)
    G_II = 0.5 * abs(T_t) * abs(delta_t)
    G_T = G_I + G_II

    G_c_eff, _, is_failed = benzeggagh_kenane_criterion(
        G_I, G_II, params.G_Ic, params.G_IIc, params.eta
    )

    damage = max(d_n, d_t, prev_damage)

    return ContactState(
        delta_n=delta_n, delta_t=delta_t,
        T_n=T_n, T_t=T_t,
        damage=damage,
        G_I=G_I, G_II=G_II, G_T=G_T,
        G_c_eff=G_c_eff,
        is_failed=is_failed or damage >= 1.0,
    )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def czm_parameters_from_material(
    E: float,
    G_Ic: float,
    G_IIc: float = None,
    sigma_t: float = None,
    eta: float = 1.45,
    penalty_factor: float = 100.0,
) -> CZMParameters:
    """由材料参数生成 CZM 参数。"""
    if G_IIc is None:
        G_IIc = G_Ic
    if sigma_t is None:
        sigma_t = E / 10.0

    T_max = sigma_t
    delta_f = 2.0 * G_Ic / T_max
    delta_0 = delta_f / 2.0

    k_n = penalty_factor * E
    k_t = penalty_factor * E / (2.0 * (1.0 + 0.3))

    return CZMParameters(
        T_max=T_max, delta_0=delta_0, delta_f=delta_f, G_c=G_Ic,
        k_n=k_n, k_t=k_t, G_Ic=G_Ic, G_IIc=G_IIc, eta=eta,
    )
