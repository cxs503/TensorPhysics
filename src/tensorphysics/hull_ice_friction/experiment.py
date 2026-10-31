"""冰-钢摩擦数值实验：微凸体集合 Monte-Carlo + 温度扫描（每次温度 10 重复）。

实验设置（对应实验室“钢滑块在冰面上恒速滑移”试验的数值等价）：

============  ====================================================
条件          取值
============  ====================================================
名义接触压力  1.0 MPa（船体-冰典型量级 0.5–2 MPa）
滑移速度      1.0 m/s（船体相对冰面滑移 0.5–2 m/s 的中值）
温度扫描      -5, -10, -15, -20, -25 °C（过冷度 5…25 K）
重复次数      每个温度 10 次（独立随机微凸体集合 → 均值/标准差/标准误）
冰材料        淡水冰（熔点 0 °C），:class:`IceMaterialCorrelations` 文献关联式
============  ====================================================

微凸体集合：接触斑点半径为对数正态分布，压力采样后再按 **冰压痕硬度** 截断
（``p_i ≤ H(θ)``，物理上限）；半径按赫兹弹性接触 ``a ∝ (F/E*)^{1/3}`` 随
``E_ice(T)`` 轻微缩放（E 随过冷度增大 → 接触斑点略变小）。

``build_ensemble_bank`` / ``evaluate_ensemble_bank`` 把“随机抽样”与“本构参数求值”
分离：标定阶段可复用同一批微凸体样本（仅改变本构参数），避免蒙特卡洛噪声掩盖
目标函数梯度（见 :mod:`tensorphysics.hull_ice_friction.calibration`）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contact_law import (
    ContactSolution,
    IceSteelContactParameters,
    solve_contact_ensemble,
)
from .ice_materials import DEFAULT_ICE, IceMaterialCorrelations
from .thermal_contact import InterfaceHeatPartition, interface_heat_partition

#: 半径缩放参考过冷度（-10 °C）
HERTZ_REFERENCE_UNDERCOOLING_K = 10.0


@dataclass(frozen=True)
class AsperityEnsembleParameters:
    """微凸体（接触斑点）集合的统计参数。"""

    n_asperities: int = 256
    radius_median_m: float = 4.0e-4
    radius_sigma_log: float = 0.6
    radius_min_m: float = 5.0e-6
    radius_max_m: float = 3.0e-3
    pressure_sigma_log: float = 0.5

    def __post_init__(self) -> None:
        if self.n_asperities < 8:
            raise ValueError("need at least 8 asperities")
        if self.radius_median_m <= 0.0:
            raise ValueError("radius median must be positive")
        if not (0.0 < self.radius_min_m < self.radius_max_m):
            raise ValueError("invalid radius bounds")


@dataclass(frozen=True)
class FrictionTestConfig:
    """数值摩擦试验配置（工况 + 接触律 + 统计集合 + 重复次数）。"""

    contact: IceSteelContactParameters = field(default_factory=IceSteelContactParameters)
    material: IceMaterialCorrelations = DEFAULT_ICE
    ensemble: AsperityEnsembleParameters = field(default_factory=AsperityEnsembleParameters)
    n_repetitions: int = 10
    base_seed: int = 20261031
    salinity_psu: float = 0.0

    def __post_init__(self) -> None:
        if self.n_repetitions < 1:
            raise ValueError("n_repetitions must be >= 1")

    def replace(self, **changes) -> "FrictionTestConfig":
        data = {
            "contact": self.contact,
            "material": self.material,
            "ensemble": self.ensemble,
            "n_repetitions": self.n_repetitions,
            "base_seed": self.base_seed,
            "salinity_psu": self.salinity_psu,
        }
        data.update(changes)
        return FrictionTestConfig(**data)

    def to_dict(self) -> dict:
        return {
            "velocity_m_s": self.contact.velocity_m_s,
            "nominal_pressure_Pa": self.contact.nominal_pressure_Pa,
            "real_contact_pressure_Pa": self.contact.real_contact_pressure_Pa,
            "real_area_fraction": self.contact.real_area_fraction,
            "local_to_mean_pressure_ratio": self.contact.local_to_mean_pressure_ratio,
            "wet_friction_coefficient": self.contact.wet_friction_coefficient,
            "melt_transition_width_K": self.contact.melt_transition_width_K,
            "relaxation": self.contact.relaxation,
            "n_asperities": self.ensemble.n_asperities,
            "asperity_radius_median_m": self.ensemble.radius_median_m,
            "asperity_radius_sigma_log": self.ensemble.radius_sigma_log,
            "pressure_sigma_log": self.ensemble.pressure_sigma_log,
            "n_repetitions": self.n_repetitions,
            "base_seed": self.base_seed,
            "salinity_psu": self.salinity_psu,
        }


def hertzian_radius_scale(
    undercooling_K: float, material: IceMaterialCorrelations = DEFAULT_ICE
) -> float:
    """赫兹弹性接触的接触斑尺寸温度缩放 ``(E(θ_ref)/E(θ))^{1/3}``。"""
    e_ref = material.elastic_modulus_Pa(HERTZ_REFERENCE_UNDERCOOLING_K)
    e_now = material.elastic_modulus_Pa(undercooling_K)
    return float((e_ref / e_now) ** (1.0 / 3.0))


def draw_asperity_ensemble(
    seed: int,
    undercooling_K: float,
    parameters: IceSteelContactParameters,
    ensemble: AsperityEnsembleParameters = AsperityEnsembleParameters(),
    material: IceMaterialCorrelations = DEFAULT_ICE,
) -> dict:
    """抽取一组微凸体（半径、压力），压力按冰硬度截断。"""
    rng = np.random.default_rng(seed)
    n = ensemble.n_asperities
    radius = rng.lognormal(
        mean=np.log(ensemble.radius_median_m), sigma=ensemble.radius_sigma_log, size=n
    )
    radius = np.clip(radius, ensemble.radius_min_m, ensemble.radius_max_m)
    radius = radius * hertzian_radius_scale(undercooling_K, material)

    p_median = parameters.local_to_mean_pressure_ratio * parameters.real_contact_pressure_Pa
    sigma = ensemble.pressure_sigma_log
    pressure = p_median * np.exp(sigma * rng.standard_normal(n) - 0.5 * sigma**2)
    hardness = float(material.hardness_Pa(undercooling_K))
    pressure_capped = np.minimum(pressure, hardness)
    return {
        "radius_m": radius,
        "pressure_Pa": pressure_capped,
        "hardness_Pa": hardness,
        "pressure_cap_fraction": float(np.mean(pressure >= hardness)),
    }


def build_ensemble_bank(temperatures_C, config: FrictionTestConfig) -> list[dict]:
    """按 :func:`run_temperature_scan` 的种子约定预抽微凸体样本库。

    返回 ``[{"temperature_C", "undercooling_K", "seed", "ensemble"}, ...]``，其中每个
    温度有 ``n_repetitions`` 个独立样本。
    """
    bank: list[dict] = []
    for index, temperature in enumerate(temperatures_C):
        theta = float(config.material.undercooling(temperature, config.salinity_psu))
        draws = []
        for rep in range(config.n_repetitions):
            seed = config.base_seed + 1000 * index + rep
            draws.append(
                {
                    "seed": int(seed),
                    "ensemble": draw_asperity_ensemble(
                        seed, theta, config.contact, config.ensemble, config.material
                    ),
                }
            )
        bank.append(
            {
                "temperature_C": float(temperature),
                "undercooling_K": theta,
                "draws": draws,
            }
        )
    return bank


def _evaluate_point(
    entry: dict,
    config: FrictionTestConfig,
    partition: InterfaceHeatPartition,
    collect_history: bool,
    solver: str = "bisection",
) -> dict:
    theta = entry["undercooling_K"]
    records: list[dict] = []
    history: list[float] = []
    for draw in entry["draws"]:
        solution: ContactSolution = solve_contact_ensemble(
            draw["ensemble"]["pressure_Pa"],
            draw["ensemble"]["radius_m"],
            theta,
            config.contact,
            config.material,
            partition=partition,
            temperature_C=entry["temperature_C"],
            solver=solver,
        )
        if collect_history:
            history = [float(x) for x in solution.residual_history]
        record = solution.summary()
        record["seed"] = draw["seed"]
        record["pressure_cap_fraction"] = draw["ensemble"]["pressure_cap_fraction"]
        record["hardness_Pa"] = draw["ensemble"]["hardness_Pa"]
        records.append(record)

    mu = np.array([r["mu"] for r in records], dtype=float)
    lam = np.array([r["lambda_wet"] for r in records], dtype=float)
    iterations = [r["iterations"] for r in records]
    point = {
        "temperature_C": entry["temperature_C"],
        "undercooling_K": theta,
        "mu_mean": float(mu.mean()),
        "mu_std": float(mu.std(ddof=1)) if mu.size > 1 else 0.0,
        "mu_stderr": float(mu.std(ddof=1) / np.sqrt(mu.size)) if mu.size > 1 else 0.0,
        "mu_repetitions": [float(x) for x in mu],
        "lambda_wet_mean": float(lam.mean()),
        "lambda_wet_std": float(lam.std(ddof=1)) if lam.size > 1 else 0.0,
        "dry_friction": float(records[0]["dry_friction"]),
        "wet_friction": float(records[0]["wet_friction"]),
        "elastic_modulus_GPa": float(config.material.elastic_modulus_GPa(theta)),
        "compressive_strength_MPa": float(config.material.compressive_strength_MPa(theta)),
        "shear_strength_MPa": float(config.material.shear_strength_Pa(theta) * 1e-6),
        "hardness_Pa": float(records[0]["hardness_Pa"]),
        "flash_temperature_K": records[0]["flash_temperature_K"],
        "iterations_max": int(max(iterations)),
        "iterations_mean": float(np.mean(iterations)),
        "converged_all": bool(all(r["converged"] for r in records)),
        "solver": records[0].get("solver", solver),
        "pressure_cap_fraction": float(records[0]["pressure_cap_fraction"]),
    }
    if collect_history:
        point["_residual_history"] = history
    return point


def evaluate_ensemble_bank(
    bank: list[dict],
    config: FrictionTestConfig,
    collect_history: bool = False,
    solver: str = "bisection",
) -> list[dict]:
    """用给定本构/接触律参数对样本库求值（可用于标定迭代与最终结果）。"""
    partition = interface_heat_partition()
    return [
        _evaluate_point(entry, config, partition, collect_history, solver) for entry in bank
    ]


def run_temperature_scan(
    temperatures_C,
    config: FrictionTestConfig | None = None,
    collect_history: bool = False,
    solver: str = "bisection",
) -> dict:
    """对一组温度做数值摩擦试验，每个温度 ``n_repetitions`` 次重复。

    返回可直接 JSON 序列化的结果字典（数字均由本函数计算，无手抄）。
    """
    config = config or FrictionTestConfig()
    bank = build_ensemble_bank(temperatures_C, config)
    points = evaluate_ensemble_bank(
        bank, config, collect_history=collect_history, solver=solver
    )
    result = {
        "config": config.to_dict(),
        "n_repetitions": config.n_repetitions,
        "points": points,
    }
    if collect_history:
        history = {}
        for point in points:
            key = f"{point['temperature_C']:g}"
            history[key] = point.pop("_residual_history", [])
        result["fixed_point_residual_history"] = history
    return result


def run_velocity_sweep(
    velocities_m_s,
    temperatures_C,
    config: FrictionTestConfig | None = None,
) -> dict:
    """滑移速度敏感性扫描（检验模型在高滑移速度下的融水润滑趋势）。"""
    config = config or FrictionTestConfig()
    rows: list[dict] = []
    for velocity in velocities_m_s:
        contact = config.contact.replace(velocity_m_s=float(velocity))
        sub = config.replace(contact=contact)
        scan = run_temperature_scan(temperatures_C, sub)
        for point in scan["points"]:
            rows.append(
                {
                    "velocity_m_s": float(velocity),
                    "temperature_C": point["temperature_C"],
                    "mu_mean": point["mu_mean"],
                    "lambda_wet_mean": point["lambda_wet_mean"],
                }
            )
    return {"velocities_m_s": [float(v) for v in velocities_m_s], "rows": rows}


__all__ = [
    "AsperityEnsembleParameters",
    "FrictionTestConfig",
    "build_ensemble_bank",
    "draw_asperity_ensemble",
    "evaluate_ensemble_bank",
    "hertzian_radius_scale",
    "run_temperature_scan",
    "run_velocity_sweep",
]