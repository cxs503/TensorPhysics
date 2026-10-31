"""冰材料力学性质随温度变化的经验关联式（hull-ice friction 物理域）。

本模块只包含**有文献依据的经验关联式**，不含任何针对摩擦数据标定的参数：

* ``E_ice(theta)``          弹性模量，随过冷度增加而缓慢增大（8.5 GPa @0 °C → 11 GPa @-25 °C）
* ``sigma_c(theta)``        单轴抗压强度（实验室应变率量级），线性增长
* ``tau_shear(theta)``      界面剪切强度 = χ·σ_c（χ≈0.3，冰的剪切/压缩强度比）
* ``hardness(theta)``       压痕硬度（真实接触压力上限，限制单个微凸体的承载压力）

``theta = T_melt - T`` 为**过冷度**（K），恒 ≥ 0。使用过冷度而非摄氏温度，使得
盐水冰（melting point depression，``melting_temperature(salinity)``）可以直接复用。

所有关联式均为线性形式并在温度区间 [-5, -25] °C 内插值使用；区间外的外推由调用方
自行负责（``docs/HULL_ICE_FRICTION_TEMPERATURE.md`` 中说明了适用范围）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 淡水冰的熔点（°C）
T_MELT_FRESHWATER_C = 0.0

#: 盐度引起熔点降低的近似斜率（°C / psu），海水 ~35 psu → 约 -1.9 °C
MELTING_DEPRESSION_C_PER_PSU = 0.054


@dataclass(frozen=True)
class IceMaterialCorrelations:
    """冰的力学/热学性质经验关联式参数组。

    默认值取自冰力学文献中常用的量级：

    * ``E0_GPa``/``dE_dtheta_GPa``: 冰的杨氏模量 8–11 GPa，随过冷度约 +1 %/K
    * ``sigma_c0_MPa``/``dsigma_c_dtheta_MPa``: 抗压强度 ~0.4 MPa @0 °C，
      约 +0.16 MPa/K（-25 °C 时 ~4.4 MPa，处于实验室量级 1–5 MPa 的下半区）
    * ``hardness0_MPa``/``dhardness_dtheta_MPa``: 压痕硬度 15–45 MPa
    * ``tau_over_sigma``: 剪切强度/抗压强度比 0.3
    """

    E0_GPa: float = 8.5
    dE_dtheta_GPa: float = 0.10
    sigma_c0_MPa: float = 0.4
    dsigma_c_dtheta_MPa: float = 0.16
    hardness0_MPa: float = 15.0
    dhardness_dtheta_MPa: float = 1.2
    tau_over_sigma: float = 0.3

    # 热物性（用于界面闪温计算）
    conductivity_W_mK: float = 2.2
    density_kg_m3: float = 917.0
    specific_heat_J_kgK: float = 2000.0
    latent_heat_J_kg: float = 3.34e5

    def melting_temperature(self, salinity_psu: float = 0.0) -> float:
        """冰的平衡熔点（°C）。``salinity_psu = 0`` 时为淡水冰 0 °C。"""
        if salinity_psu < 0.0:
            raise ValueError("salinity must be non-negative")
        return T_MELT_FRESHWATER_C - MELTING_DEPRESSION_C_PER_PSU * float(salinity_psu)

    def undercooling(self, temperature_C, salinity_psu: float = 0.0):
        """过冷度 ``theta = T_melt - T``（K），矢量安全。"""
        theta = self.melting_temperature(salinity_psu) - np.asarray(temperature_C, dtype=float)
        return np.maximum(theta, 0.0)

    # -- 弹性模量 ---------------------------------------------------------
    def elastic_modulus_GPa(self, theta):
        theta = np.asarray(theta, dtype=float)
        return self.E0_GPa + self.dE_dtheta_GPa * theta

    def elastic_modulus_Pa(self, theta):
        return self.elastic_modulus_GPa(theta) * 1.0e9

    # -- 强度 -------------------------------------------------------------
    def compressive_strength_MPa(self, theta):
        theta = np.asarray(theta, dtype=float)
        return self.sigma_c0_MPa + self.dsigma_c_dtheta_MPa * theta

    def compressive_strength_Pa(self, theta):
        return self.compressive_strength_MPa(theta) * 1.0e6

    def shear_strength_Pa(self, theta):
        """界面剪切强度 τ_s(θ) = χ·σ_c(θ)（Pa）。"""
        return self.tau_over_sigma * self.compressive_strength_Pa(theta)

    def hardness_Pa(self, theta):
        """压痕硬度（Pa）——真实接触压力的物理上限。"""
        theta = np.asarray(theta, dtype=float)
        return (self.hardness0_MPa + self.dhardness_dtheta_MPa * theta) * 1.0e6

    # -- 热物性 -----------------------------------------------------------
    @property
    def thermal_diffusivity_m2_s(self) -> float:
        return self.conductivity_W_mK / (self.density_kg_m3 * self.specific_heat_J_kgK)

    @property
    def effusivity(self) -> float:
        """热渗透系数 sqrt(k ρ c)（J m^-2 K^-1 s^-1/2）。"""
        return float(
            np.sqrt(self.conductivity_W_mK * self.density_kg_m3 * self.specific_heat_J_kgK)
        )


DEFAULT_ICE = IceMaterialCorrelations()