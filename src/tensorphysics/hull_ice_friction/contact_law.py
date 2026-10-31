"""接触律：冰-钢界面混合润滑摩擦系数（温度相关）。

摩擦系数分解为两个并联的承载机制（Bowden–Tabor 型面积加权）：

.. math::
   \\mu(\\theta) = (1-\\lambda)\\,\\mu_{dry}(\\theta) + \\lambda\\,\\mu_{wet},

* **干摩擦（蠕变/剪切控制）** ``μ_dry = τ_s(θ) / p_real``：冰的界面剪切强度除以真实
  接触压力。冰在低速滑移下由蠕变/剪切破坏主导，故 μ_dry 随过冷度 θ 增大而上升。
* **融水润滑** ``μ_wet``：接触斑点表面出现融水膜时的润滑摩擦水平（文献 0.02–0.05）。
* **润滑覆盖度 λ** 由每个微凸体的界面闪温 ΔT_i 与当地熔点间距的比较给出

  .. math:: s_i = \\frac{1}{1+\\exp[-(\\Delta T_i - \\theta)/w]},
            \\quad \\lambda = \\frac{\\sum_i A_i s_i}{\\sum_i A_i}

  其中 ``w`` 为融膜转变宽度（K）。``s_i`` 与 ``μ_i`` 相互依赖（摩擦生热 → 闪温 →
  润滑 → 摩擦），故每个微凸体需解一个**不动点问题**：

  .. math:: \\mu_i = (1-s_i(\\mu_i))\\mu_{dry} + s_i(\\mu_i)\\,\\mu_{wet}

  本模块用带松弛的 Picard 迭代（``ω``）求解，收敛过程与残差历史作为数值证据输出。

参数分层（重要，见 docs 报告）：
* 文献固定常数：``μ_wet``、``w``、几何/统计参数、冰材料关联式；
* **标定参数**：``real_contact_pressure_Pa``（真实接触压力，唯一的本构标定量，
  由 :mod:`tensorphysics.hull_ice_friction.calibration` 从参考数据子集辨识）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .ice_materials import DEFAULT_ICE, IceMaterialCorrelations
from .thermal_contact import (
    InterfaceHeatPartition,
    flash_temperature,
    interface_heat_partition,
)


@dataclass(frozen=True)
class IceSteelContactParameters:
    """冰-钢滑动接触的工况与接触律参数。

    ``real_contact_pressure_Pa`` 为**标定量**（默认值来自
    ``artifacts/hull_ice_friction/calibration.json`` 的辨识结果）。
    """

    velocity_m_s: float = 1.0
    nominal_pressure_Pa: float = 1.0e6
    real_contact_pressure_Pa: float = 3.67e6
    local_to_mean_pressure_ratio: float = 3.0
    wet_friction_coefficient: float = 0.03
    melt_transition_width_K: float = 2.0
    relaxation: float = 0.5
    max_iterations: int = 500
    fixed_point_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.velocity_m_s <= 0.0:
            raise ValueError("velocity must be positive")
        if self.real_contact_pressure_Pa <= 0.0:
            raise ValueError("real contact pressure must be positive")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation must lie in (0, 1]")
        if self.melt_transition_width_K <= 0.0:
            raise ValueError("melt transition width must be positive")

    def replace(self, **changes) -> "IceSteelContactParameters":
        data = {
            "velocity_m_s": self.velocity_m_s,
            "nominal_pressure_Pa": self.nominal_pressure_Pa,
            "real_contact_pressure_Pa": self.real_contact_pressure_Pa,
            "local_to_mean_pressure_ratio": self.local_to_mean_pressure_ratio,
            "wet_friction_coefficient": self.wet_friction_coefficient,
            "melt_transition_width_K": self.melt_transition_width_K,
            "relaxation": self.relaxation,
            "max_iterations": self.max_iterations,
            "fixed_point_tolerance": self.fixed_point_tolerance,
        }
        data.update(changes)
        return IceSteelContactParameters(**data)

    @property
    def real_area_fraction(self) -> float:
        """真实接触面积分数 A_r/A_n = p_nominal / p_real（Bowden–Tabor）。"""
        return self.nominal_pressure_Pa / self.real_contact_pressure_Pa


@dataclass
class ContactSolution:
    """单个温度点上微凸体集合的接触律解。"""

    temperature_C: float
    undercooling_K: float
    mu: float
    lambda_wet: float
    dry_friction: float
    wet_friction: float
    iterations: int
    converged: bool
    residual_max: float
    residual_history: list[float] = field(default_factory=list)
    solver: str = "bisection"
    flash_temperature_K: dict[str, float] = field(default_factory=dict)
    local_mu: np.ndarray | None = field(default=None, repr=False)
    local_melt_fraction: np.ndarray | None = field(default=None, repr=False)

    def summary(self) -> dict:
        return {
            "temperature_C": self.temperature_C,
            "undercooling_K": self.undercooling_K,
            "mu": self.mu,
            "lambda_wet": self.lambda_wet,
            "dry_friction": self.dry_friction,
            "wet_friction": self.wet_friction,
            "iterations": self.iterations,
            "converged": self.converged,
            "solver": self.solver,
            "residual_max": self.residual_max,
            "flash_temperature_K": dict(self.flash_temperature_K),
        }


def dry_friction_coefficient(
    undercooling_K,
    real_contact_pressure_Pa: float,
    material: IceMaterialCorrelations = DEFAULT_ICE,
):
    """干摩擦分支 μ_dry(θ) = τ_s(θ)/p_real（矢量化）。"""
    if real_contact_pressure_Pa <= 0.0:
        raise ValueError("real contact pressure must be positive")
    return material.shear_strength_Pa(undercooling_K) / real_contact_pressure_Pa


def melt_fraction(flash_temperature_K, undercooling_K, transition_width_K: float = 2.0):
    """融膜覆盖度 s = 1/(1+exp(-(ΔT-θ)/w))。"""
    if transition_width_K <= 0.0:
        raise ValueError("transition width must be positive")
    dT = np.asarray(flash_temperature_K, dtype=float)
    theta = np.asarray(undercooling_K, dtype=float)
    return 1.0 / (1.0 + np.exp(-(dT - theta) / transition_width_K))


def solve_contact_ensemble(
    pressure_local_Pa,
    contact_radius_m,
    undercooling_K: float,
    parameters: IceSteelContactParameters,
    material: IceMaterialCorrelations = DEFAULT_ICE,
    partition: InterfaceHeatPartition | None = None,
    temperature_C: float | None = None,
    solver: str = "bisection",
) -> ContactSolution:
    """对一组微凸体解混合润滑接触律（每个微凸体一个对称不动点问题）。

    ``pressure_local_Pa``/``contact_radius_m`` 为同长度数组。摩擦热流
    ``q_i = μ_i p_i v``，接触时间 ``t_{c,i} = 2 a_i / v``。

    局部方程 ``μ_i = F(μ_i) = (1-s(μ_i))μ_dry + s(μ_i)μ_wet`` 中 ``F`` 单调递减、
    ``μ - F(μ)`` 单调递增，故根唯一。两种求解器：

    * ``"bisection"``（默认）：对称区间 ``[μ_wet, μ_dry]`` 上的向量化二分，**无条件收敛**，
      迭代历史为区间宽度（每次减半）；
    * ``"picard"``：带松弛参数 ``ω`` 的 Picard 迭代，用于对比（冷端可能出现极限环，
      见 ``docs/HULL_ICE_FRICTION_TEMPERATURE.md`` 的数值注意点）。
    """
    p = np.asarray(pressure_local_Pa, dtype=float)
    a = np.asarray(contact_radius_m, dtype=float)
    if p.shape != a.shape:
        raise ValueError("pressure and radius arrays must have the same shape")
    if np.any(p <= 0.0) or np.any(a <= 0.0):
        raise ValueError("pressures and radii must be positive")
    theta = float(undercooling_K)
    if theta < 0.0:
        raise ValueError("undercooling must be non-negative")

    partition = partition or interface_heat_partition()
    v = parameters.velocity_m_s
    contact_time = 2.0 * a / v
    mu_dry = float(dry_friction_coefficient(theta, parameters.real_contact_pressure_Pa, material))
    mu_dry = min(mu_dry, 1.5)  # 数值安全上限（μ_dry 物理上 < 1）
    mu_wet = float(parameters.wet_friction_coefficient)
    if not mu_wet < mu_dry:
        raise ValueError("wet friction coefficient must be smaller than the dry branch")
    width = parameters.melt_transition_width_K
    weights = a**2
    weights = weights / weights.sum()

    history: list[float] = []
    if solver == "bisection":
        lo = np.full(p.shape, mu_wet)
        hi = np.full(p.shape, mu_dry)
        iterations = 0
        residual = np.inf
        span = float(np.max(hi - lo))
        tol_effective = parameters.fixed_point_tolerance * max(1.0, span)
        for iterations in range(1, parameters.max_iterations + 1):
            mid = 0.5 * (lo + hi)
            dT = flash_temperature(mid * p * v, contact_time, partition=partition)
            s = melt_fraction(dT, theta, width)
            h = mid - ((1.0 - s) * mu_dry + s * mu_wet)
            lower_half = h < 0.0
            lo = np.where(lower_half, mid, lo)
            hi = np.where(lower_half, hi, mid)
            residual = float(np.max(hi - lo))
            history.append(residual)
            if residual <= tol_effective:
                break
        mu_local = 0.5 * (lo + hi)
    elif solver == "picard":
        mu_local = np.full(p.shape, mu_dry)
        iterations = 0
        residual = np.inf
        omega = parameters.relaxation
        tol_effective = parameters.fixed_point_tolerance
        for iterations in range(1, parameters.max_iterations + 1):
            dT = flash_temperature(mu_local * p * v, contact_time, partition=partition)
            s = melt_fraction(dT, theta, width)
            target = (1.0 - s) * mu_dry + s * mu_wet
            residual = float(np.max(np.abs(target - mu_local)))
            history.append(residual)
            mu_local = (1.0 - omega) * mu_local + omega * target
            if residual <= parameters.fixed_point_tolerance:
                break
    else:  # pragma: no cover - 参数校验
        raise ValueError("solver must be 'bisection' or 'picard'")

    converged = bool(residual <= tol_effective)

    # 用最终 μ 重新评估闪温/润滑状态（保证输出自洽）
    q = mu_local * p * v
    dT = flash_temperature(q, contact_time, partition=partition)
    s = melt_fraction(dT, theta, width)
    mu = float(np.sum(weights * mu_local))
    lam = float(np.sum(weights * s))
    flash_stats = {
        "mean_K": float(np.sum(weights * dT)),
        "median_K": float(np.median(dT)),
        "p90_K": float(np.percentile(dT, 90)),
        "max_K": float(np.max(dT)),
    }
    return ContactSolution(
        temperature_C=float(temperature_C if temperature_C is not None else -theta),
        undercooling_K=theta,
        mu=mu,
        lambda_wet=lam,
        dry_friction=mu_dry,
        wet_friction=mu_wet,
        iterations=iterations,
        converged=converged,
        residual_max=residual,
        residual_history=history,
        flash_temperature_K=flash_stats,
        local_mu=mu_local,
        local_melt_fraction=s,
        solver=solver,
    )


def area_weighted_mean(values: Sequence[float], radii: Sequence[float]) -> float:
    """接触面积（a²）加权平均。"""
    v = np.asarray(values, dtype=float)
    w = np.asarray(radii, dtype=float) ** 2
    if v.shape != w.shape:
        raise ValueError("values and radii must have the same shape")
    return float(np.sum(w * v) / np.sum(w))


__all__ = [
    "ContactSolution",
    "IceSteelContactParameters",
    "area_weighted_mean",
    "dry_friction_coefficient",
    "melt_fraction",
    "solve_contact_ensemble",
]