"""Hull-ice friction temperature dependence: calibration & validation in TensorPhysics.

模块（`tensorphysics.hull_ice_friction`）
----------------------------------------
``ice_materials``    冰的 E(T)、σ_c(T)、τ_s(T)、硬度 H(T) 文献关联式
``thermal_contact``  界面闪温解析解 + 一维有限差分数值解（网格收敛性）
``contact_law``      混合润滑接触律 μ=(1-λ)μ_dry+λμ_wet（带松弛 Picard 不动点）
``experiment``       微凸体 Monte-Carlo 数值摩擦试验 + 温度扫描（每温度 10 重复）
``calibration``      μ(T)=μ_0+α|T-T_melt|^β 拟合（自实现 LM）与 p_real 辨识
``reference_data``   文献锚定参考数据与来源/局限声明
``validation``       端到端标定-验证流程、收敛性研究、图表与 JSON 产物
"""
from __future__ import annotations

from .calibration import (
    ConstitutiveIdentification,
    PowerLawFit,
    compare_with_reference,
    fit_power_law,
    identify_real_contact_pressure,
    power_law_mu,
    r_squared,
)
from .contact_law import (
    ContactSolution,
    IceSteelContactParameters,
    dry_friction_coefficient,
    melt_fraction,
    solve_contact_ensemble,
)
from .experiment import (
    AsperityEnsembleParameters,
    FrictionTestConfig,
    build_ensemble_bank,
    draw_asperity_ensemble,
    evaluate_ensemble_bank,
    run_temperature_scan,
    run_velocity_sweep,
)
from .ice_materials import DEFAULT_ICE, IceMaterialCorrelations, T_MELT_FRESHWATER_C
from .reference_data import (
    PROVENANCE,
    REFERENCE_POINTS,
    reference_mu,
    reference_table,
    reference_uncertainties,
    reference_temperatures,
    role_targets,
)
from .thermal_contact import (
    ANALYTIC_SURFACE_COEFFICIENT,
    STEEL_THERMAL,
    fd_analytic_relative_error,
    fd_grid_convergence,
    fd_surface_temperature,
    flash_temperature,
    interface_heat_partition,
)

__all__ = [
    "ANALYTIC_SURFACE_COEFFICIENT",
    "ConstitutiveIdentification",
    "ContactSolution",
    "DEFAULT_ICE",
    "FrictionTestConfig",
    "AsperityEnsembleParameters",
    "IceMaterialCorrelations",
    "IceSteelContactParameters",
    "PROVENANCE",
    "PowerLawFit",
    "REFERENCE_POINTS",
    "STEEL_THERMAL",
    "T_MELT_FRESHWATER_C",
    "build_ensemble_bank",
    "compare_with_reference",
    "draw_asperity_ensemble",
    "dry_friction_coefficient",
    "evaluate_ensemble_bank",
    "fd_analytic_relative_error",
    "fd_grid_convergence",
    "fd_surface_temperature",
    "fit_power_law",
    "flash_temperature",
    "identify_real_contact_pressure",
    "interface_heat_partition",
    "melt_fraction",
    "power_law_mu",
    "r_squared",
    "reference_mu",
    "reference_table",
    "reference_temperatures",
    "reference_uncertainties",
    "role_targets",
    "run_temperature_scan",
    "run_velocity_sweep",
    "solve_contact_ensemble",
]