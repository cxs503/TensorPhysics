"""冰-钢接触界面闪温：解析解 + 一维有限差分数值解（含网格收敛性）。

物理模型
--------
滑动接触中摩擦功 ``q = μ p v`` 在真实接触斑点上转化为热流。对半无限体在
恒定表面热流下的瞬态导热，表面温升为

.. math::  \\Delta T(t) = \\frac{2 q}{k}\\sqrt{\\frac{\\alpha t}{\\pi}}

两个接触体同时导热时（Blok 界面温度），界面温升由**热渗透系数之和**控制：

.. math::  \\Delta T_{int}(t) = \\frac{2 q}{S}\\sqrt{\\frac{t_c}{\\pi}},
           \\quad S = \\sqrt{k\\rho c}\\big|_{ice} + \\sqrt{k\\rho c}\\big|_{steel}

其中 ``t_c = 2 a / v`` 为单个微凸体的接触时间（a 为接触斑点半径，v 为滑移速度）。
本模块提供

* :func:`interface_heat_partition` —— 热流分配（进入冰侧的比例 ζ）
* :func:`flash_temperature` —— 解析界面闪温
* :func:`fd_surface_temperature` —— 无量纲一维有限差分（隐式 Euler，追赶法）
* :func:`fd_grid_convergence` —— 网格加密收敛性研究（对解析解 ``2/√π`` 的误差与观测阶数）

``fd_surface_temperature`` 是**独立数值解**，用于证明解析系数 ``2/√π`` 的正确性以及
本模拟所用解析公式的离散误差量级（见 docs 报告中的收敛性表）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .ice_materials import DEFAULT_ICE, IceMaterialCorrelations

#: 恒定热流半无限体表面的无量纲系数 2/sqrt(pi)
ANALYTIC_SURFACE_COEFFICIENT = 2.0 / np.sqrt(np.pi)


@dataclass(frozen=True)
class ThermalProperties:
    """刚性体热物性。"""

    conductivity_W_mK: float
    density_kg_m3: float
    specific_heat_J_kgK: float
    label: str = "body"

    @property
    def effusivity(self) -> float:
        return float(
            np.sqrt(self.conductivity_W_mK * self.density_kg_m3 * self.specific_heat_J_kgK)
        )

    @property
    def diffusivity(self) -> float:
        return self.conductivity_W_mK / (self.density_kg_m3 * self.specific_heat_J_kgK)


def ice_thermal_properties(material: IceMaterialCorrelations = DEFAULT_ICE) -> ThermalProperties:
    return ThermalProperties(
        conductivity_W_mK=material.conductivity_W_mK,
        density_kg_m3=material.density_kg_m3,
        specific_heat_J_kgK=material.specific_heat_J_kgK,
        label="ice",
    )


#: 船体钢（低合金高强钢）典型热物性
STEEL_THERMAL = ThermalProperties(
    conductivity_W_mK=50.0, density_kg_m3=7800.0, specific_heat_J_kgK=480.0, label="steel"
)


@dataclass(frozen=True)
class InterfaceHeatPartition:
    """两体接触热流分配（Blok）。"""

    zeta_ice: float
    zeta_steel: float
    effusivity_sum: float

    def __post_init__(self) -> None:
        if abs(self.zeta_ice + self.zeta_steel - 1.0) > 1e-9:
            raise ValueError("heat partition fractions must sum to 1")
        if self.effusivity_sum <= 0.0:
            raise ValueError("effusivity sum must be positive")


def interface_heat_partition(
    ice: ThermalProperties | None = None, steel: ThermalProperties = STEEL_THERMAL
) -> InterfaceHeatPartition:
    """按热渗透系数分配摩擦热（Blok）：ζ_ice = √(kρc)_ice / (√(kρc)_ice + √(kρc)_steel)。"""
    ice = ice_thermal_properties() if ice is None else ice
    total = ice.effusivity + steel.effusivity
    return InterfaceHeatPartition(
        zeta_ice=ice.effusivity / total,
        zeta_steel=steel.effusivity / total,
        effusivity_sum=total,
    )


def flash_temperature(
    heat_flux_W_m2,
    contact_time_s,
    effusivity_sum: float | None = None,
    partition: InterfaceHeatPartition | None = None,
):
    """界面闪温（K）。``heat_flux_W_m2`` 为**总**摩擦热流 ``μ p v``（可含数组）。

    返回 ``2 q / S * sqrt(t_c / π)``。
    """
    if effusivity_sum is None:
        effusivity_sum = (partition or interface_heat_partition()).effusivity_sum
    if effusivity_sum <= 0.0:
        raise ValueError("effusivity_sum must be positive")
    q = np.asarray(heat_flux_W_m2, dtype=float)
    t_c = np.asarray(contact_time_s, dtype=float)
    if np.any(t_c < 0.0):
        raise ValueError("contact time must be non-negative")
    return 2.0 * q / effusivity_sum * np.sqrt(t_c / np.pi)


# ---------------------------------------------------------------------------
# 一维有限差分数值解（无量纲）
# ---------------------------------------------------------------------------


class _TridiagonalSolver:
    """常系数三对角系统的追赶法求解器（前向消元系数只算一次）。

    纯 Python float 实现（比 numpy 标量操作快一个量级），用于时间推进内循环。
    """

    def __init__(self, diag: Sequence[float], lower: Sequence[float], upper: Sequence[float]):
        n = len(diag)
        self.n = n
        self.c = [0.0] * n
        self.m = [0.0] * n
        self.lower = list(lower)
        self.rhs_scale = 0.0
        # 前向消元
        m0 = float(diag[0])
        self.m[0] = m0
        if n > 1:
            self.c[0] = float(upper[0]) / m0
        for i in range(1, n):
            mi = float(diag[i]) - self.lower[i - 1] * self.c[i - 1]
            self.m[i] = mi
            if i < n - 1:
                self.c[i] = float(upper[i]) / mi

    def solve_into(self, rhs: list[float]) -> None:
        """就地求解；``rhs`` 被覆盖为解向量。"""
        n = self.n
        m = self.m
        c = self.c
        lower = self.lower
        # 前向
        rhs[0] /= m[0]
        for i in range(1, n):
            rhs[i] = (rhs[i] - lower[i - 1] * rhs[i - 1]) / m[i]
        # 回代
        for i in range(n - 2, -1, -1):
            rhs[i] -= c[i] * rhs[i + 1]


def fd_surface_temperature(
    fourier_number: float,
    n_nodes: int = 201,
    domain_factor: float = 4.0,
    time_step_factor: float = 0.5,
    max_steps: int = 200_000,
) -> float:
    """无量纲恒定热流半无限体表面的瞬态温度 θ(0, Fo)。

    无量纲化：``ξ = x/L``，``Fo = α t / L²``，``θ = T k / (q L)``，则
    ``θ_Fo = θ_ξξ``，表面 ``θ_ξ(0) = -1``（镜像节点 ``θ_{-1} = θ_1 + 2Δξ``），
    远场 ``θ(∞) = 0``。计算域 ``[0, domain_factor·√Fo]``；隐式 Euler 无条件稳定，
    步长 ``ΔFo = time_step_factor·Δξ²``。

    解析解为 ``θ(0, Fo) = 2√(Fo/π)``。
    """
    if fourier_number <= 0.0:
        raise ValueError("fourier_number must be positive")
    if n_nodes < 5:
        raise ValueError("n_nodes must be >= 5")

    length = domain_factor * float(np.sqrt(fourier_number))
    dxi = length / (n_nodes - 1)
    dfo = min(time_step_factor * dxi**2, fourier_number / 20.0)
    n_steps = int(np.ceil(fourier_number / dfo))
    if n_steps > max_steps:
        n_steps = max_steps
        dfo = fourier_number / n_steps

    r = dfo / dxi**2
    diag = [1.0 + 2.0 * r] * n_nodes
    upper = [-r] * (n_nodes - 1)
    lower = [-r] * (n_nodes - 1)
    # 表面行（镜像节点热流边界）
    upper[0] = -2.0 * r
    # 远场节点固定
    diag[-1] = 1.0
    solver = _TridiagonalSolver(diag, lower, upper)

    surf_source = 2.0 * r * dxi
    theta = [0.0] * n_nodes
    for _ in range(n_steps):
        theta[0] += surf_source
        solver.solve_into(theta)
        theta[-1] = 0.0

    return float(theta[0])


def fd_analytic_relative_error(fourier_number: float, n_nodes: int, **kwargs) -> float:
    """有限差分表面温度相对解析解的误差（(FD-解析)/解析）。"""
    fd = fd_surface_temperature(fourier_number, n_nodes=n_nodes, **kwargs)
    analytic = ANALYTIC_SURFACE_COEFFICIENT * np.sqrt(fourier_number)
    return float((fd - analytic) / analytic)


def fd_grid_convergence(
    fourier_number: float, node_counts: Sequence[int] = (41, 81, 161, 321, 641)
) -> list[dict]:
    """网格加密收敛性研究。

    返回每个网格的 ``abs_rel_error``，以及相邻网格间的观测收敛阶
    ``p = log2(e_{i-1}/e_i)``（空间二阶 + 时间一阶且 ΔFo∝Δξ² → 理论综合阶 ≈ 2）。
    """
    table: list[dict] = []
    previous_error: float | None = None
    for n_nodes in node_counts:
        error = abs(fd_analytic_relative_error(fourier_number, n_nodes))
        order = None
        if previous_error is not None and error > 0.0 and previous_error > 0.0:
            order = float(np.log(previous_error / error) / np.log(2.0))
        table.append(
            {
                "n_nodes": int(n_nodes),
                "abs_rel_error": error,
                "observed_order": order,
            }
        )
        previous_error = error
    return table


__all__ = [
    "ANALYTIC_SURFACE_COEFFICIENT",
    "STEEL_THERMAL",
    "InterfaceHeatPartition",
    "ThermalProperties",
    "fd_analytic_relative_error",
    "fd_grid_convergence",
    "fd_surface_temperature",
    "flash_temperature",
    "ice_thermal_properties",
    "interface_heat_partition",
]