"""P84-W2C campaign specification: ten nodes and two cross-domain interfaces.

Everything the gate judges is *declared* here, transcribed from the campaign
node goals stored in ``campaign_plans`` (plan 385, P84-W2C seed).  The module is
deliberately free of numerical code: it says **what** must be delivered and
**how** it is judged, never how to compute it.  The numbers below are the
machine-checkable acceptance thresholds of the node goals, not tuned values.

Two kinds of declaration exist:

``NodeSpec``
    One of the ten research nodes: repository, physics domain, declared module /
    test / artifact paths, and its acceptance criteria.  A criterion is judged
    only from numbers the node itself reports in a parseable artifact; when the
    artifact is missing the criterion is ``WITHHELD`` (never silently green).
``ContractSpec``
    A cross-solver interface: the producer artifact schema (TensorLBM / the
    ship platform) and the consumer symbol that must actually exist in the
    consumer repository (ship-performance-platform).  Contract validation is
    bilateral: it inspects both sides, so a producer cannot "pass" by inventing
    a schema the consumer never asked for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

#: thin-airfoil lift slope, 2*pi per radian (the reference of node N01 C1)
TWO_PI = 6.283185307179586

COMPARATORS = (
    "rel_dev_le",   # |value - ref| / |ref| <= threshold
    "le",           # |value| <= threshold (or value <= threshold when ref None)
    "ge",           # value >= threshold
    "in_range",     # threshold is (lo, hi)
    "is_true",      # value is truthy
)


@dataclass(frozen=True)
class Criterion:
    """One machine-checkable acceptance criterion of a campaign node."""

    cid: str
    statement: str
    aliases: tuple[str, ...]
    comparator: str
    threshold: float | tuple[float, float] | bool
    ref: float | None = None
    unit: str = ""
    source: str = ""
    #: True when the reference value comes from an external table (Abbott,
    #: Soulsby-Whitehouse, DNV, ...).  The node must then report the deviation
    #: itself and name the table; the gate re-checks the reported number only.
    external: bool = False

    def __post_init__(self) -> None:
        if self.comparator not in COMPARATORS:
            raise ValueError(f"{self.cid}: unknown comparator {self.comparator!r}")
        if self.comparator == "in_range" and not (
            isinstance(self.threshold, tuple) and len(self.threshold) == 2
        ):
            raise ValueError(f"{self.cid}: in_range needs a (lo, hi) threshold")


@dataclass(frozen=True)
class NodeSpec:
    node_key: str
    mission_id: int
    repo: str
    title: str
    physics_domain: str
    modules: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    artifact_globs: tuple[str, ...] = ()
    criteria: tuple[Criterion, ...] = ()
    contract_id: str | None = None
    #: parent-plan node title used for the lineage adjudication (plan 365)
    lineage_title: str = ""


@dataclass(frozen=True)
class ContractSpec:
    contract_id: str
    producer_node: str
    producer_schema: str
    producer_required: tuple[str, ...]
    consumer_repo: str
    consumer_file: str
    consumer_symbols: tuple[str, ...]
    consumer_fields: tuple[str, ...]
    consumer_constants: Mapping[str, float] = field(default_factory=dict)
    #: producer-side transform that must agree numerically with the consumer
    numeric_anchor: str = ""
    tolerance: float = 1e-12
    #: advisory contracts (consumer interface not yet built) never block C2
    required: bool = True


# ---------------------------------------------------------------------------
# The ten P84-W2C nodes (plan 385, N01..N10).
# ---------------------------------------------------------------------------
NODES: tuple[NodeSpec, ...] = (
    NodeSpec(
        node_key="P84W2C-N01",
        mission_id=8013,
        repo="TensorLBM",
        title="TensorLBM NACA0012 翼型气动验证面",
        physics_domain="空气动力学",
        modules=("src/verification/aero/naca0012/",),
        tests=("tests/test_naca0012_aero.py",),
        artifact_globs=("artifacts/naca0012_aero/*.json", "artifacts/naca0012_aero/*.png"),
        criteria=(
            Criterion("C1", "升力线斜率 dC_L/dα 与薄翼理论 2π 相对偏差 < 5%",
                      ("lift_slope", "dcl_dalpha", "cl_alpha_slope", "cl_slope", "dCL_dalpha"),
                      "rel_dev_le", 0.05, ref=TWO_PI, unit="1/rad",
                      source="thin-airfoil theory 2*pi"),
            Criterion("C2", "α=0° 对称翼型零升力 |C_L| < 1e-4",
                      ("cl_at_zero", "cl0", "cl_zero", "C_L_at_0"), "le", 1e-4, unit="-",
                      source="symmetry exact solution"),
            Criterion("C4", "细网格 C_L 的 GCI < 2%",
                      ("grid_convergence_index", "gci", "GCI"), "le", 0.02, unit="-",
                      source="Roache GCI"),
            Criterion("C5", "进出口质量流量不平衡 < 1e-6",
                      ("mass_imbalance", "mass_conservation_error", "mass_flux_imbalance"),
                      "le", 1e-6, unit="-", source="continuity"),
            Criterion("C2b", "零升阻力 C_D0 与 Abbott 表值相对偏差 < 15%",
                      ("cd0_rel_error", "cd0_deviation", "cd0_error_vs_abbott"),
                      "le", 0.15, unit="-", source="Abbott & von Doenhoff (1959)",
                      external=True),
            Criterion("C3", "失速角与实验值 (14°±2°) 偏差 < 3°",
                      ("stall_angle_dev_deg", "stall_deviation_deg"), "le", 3.0, unit="deg",
                      source="Abbott & von Doenhoff (1959)", external=True),
        ),
        contract_id="aero-wind",
        lineage_title="N01: TensorLBM NACA0012 翼型气动验证面",
    ),
    NodeSpec(
        node_key="P84W2C-N02",
        mission_id=8014,
        repo="TensorLBM",
        title="TensorLBM Smagorinsky LES 通道流验证面",
        physics_domain="湍流建模",
        modules=("src/tensorlbm/verification/turbulence/",),
        tests=("tests/test_channel_les.py",),
        artifact_globs=("artifacts/channel_les/*.json", "artifacts/channel_les/*.png"),
        criteria=(
            Criterion("C1", "对数区 U+ 与 log-law 偏差 < 5%",
                      ("log_law_max_dev", "log_law_error", "ulog_deviation"), "le", 0.05,
                      unit="-", source="kappa=0.41, B=5.2"),
            Criterion("C2", "摩擦雷诺数 Re_tau 与目标 180 偏差 < 3%",
                      ("re_tau_rel_error", "retau_rel_dev", "friction_reynolds_rel_error"),
                      "le", 0.03, unit="-", source="MKM1999 Re_tau=180", external=True),
            Criterion("C4", "三档网格 U+ 剖面 GCI < 5%",
                      ("grid_convergence_index", "gci"), "le", 0.05, unit="-",
                      source="Roache GCI"),
            Criterion("C6", "通道中心 SGS 应力占比 < 30%",
                      ("sgs_stress_fraction", "sgs_fraction_center"), "le", 0.30, unit="-",
                      source="resolved-scale dominance"),
            Criterion("C7", "周期边界质量流量不平衡 < 1e-8",
                      ("mass_imbalance", "mass_conservation_error"), "le", 1e-8, unit="-",
                      source="continuity"),
        ),
        lineage_title="N02: TensorLBM Smagorinsky LES 通道流验证面",
    ),
    NodeSpec(
        node_key="P84W2C-N03",
        mission_id=8362,
        repo="TensorBEM",
        title="TensorBEM 船体阴极保护电流分布验证面",
        physics_domain="电化学",
        modules=("src/verification/corrosion/cathodic_protection/",),
        tests=("tests/test_cathodic_protection.py",),
        artifact_globs=("artifacts/cathodic_protection/*.json",
                        "artifacts/cathodic_protection/*.png"),
        criteria=(
            Criterion("C1", "BEM 线性系统相对残差 < 1e-10",
                      ("linear_residual", "laplace_residual", "axb_rel_residual"),
                      "le", 1e-10, unit="-", source="Laplace/Green identity"),
            Criterion("C2", "阳极输出总电流 = 阴极吸收总电流，相对误差 < 1e-8",
                      ("current_conservation_error", "current_imbalance"), "le", 1e-8,
                      unit="-", source="charge conservation"),
            Criterion("C3", "DNV 保护电位区间覆盖面积比 ≥ 95%",
                      ("protection_ratio", "protected_fraction_95", "protection_ratio_dnv"),
                      "ge", 0.95, unit="-", source="DNV-RP-B401"),
            Criterion("C5", "三档 BEM 网格关键点电位 GCI < 3%",
                      ("grid_convergence_index", "gci"), "le", 0.03, unit="-",
                      source="Roache GCI"),
        ),
        lineage_title="N03: TensorBEM 船体阴极保护电流分布验证面",
    ),
    NodeSpec(
        node_key="P84W2C-N04",
        mission_id=8016,
        repo="TensorLBM",
        title="TensorLBM 悬沙输运与床面冲刷验证面",
        physics_domain="多相流",
        modules=("src/tensorlbm/sediment/",),
        tests=("tests/test_sediment_scour.py",),
        artifact_globs=("artifacts/p84_w1/*.json",),
        criteria=(
            Criterion("C1", "数值起动 Shields 曲线与 Soulsby–Whitehouse 偏差 ≤ 15%",
                      ("shields_rel_error", "shields_deviation"), "le", 0.15, unit="-",
                      source="Soulsby & Whitehouse (1997)", external=True),
            Criterion("C2", "Rouse 指数 Z 与解析值相对误差 < 5%",
                      ("rouse_z_rel_error", "rouse_rel_error"), "le", 0.05, unit="-",
                      source="Z = w_s / (kappa u*)"),
            Criterion("C3", "泥沙总量（悬移+床面）漂移 < 1e-6",
                      ("mass_drift", "sediment_mass_drift", "mass_conservation_drift"),
                      "le", 1e-6, unit="-", source="Exner flux closure"),
            Criterion("C5", "单桩平衡冲刷 S/D 落在 Melville–Chiew 区间 [1.0, 2.4]",
                      ("scour_ratio", "s_over_d", "scour_depth_ratio"), "in_range",
                      (1.0, 2.4), unit="-", source="Melville & Chiew (1999)", external=True),
        ),
        contract_id="sediment-ukc",
        lineage_title="N04: TensorLBM 悬沙输运与床面冲刷验证面",
    ),
    NodeSpec(
        node_key="P84W2C-N05",
        mission_id=8017,
        repo="TensorDEM",
        title="TensorDEM 散货舱颗粒卸料动力学验证面",
        physics_domain="颗粒力学",
        modules=("src/tensordem/hopper_discharge/",),
        tests=("tests/test_hopper_discharge.py",),
        artifact_globs=("artifacts/hopper_discharge/*.json",
                        "artifacts/hopper_discharge/*.png"),
        criteria=(
            Criterion("C1", "休止角与 atan(mu) 偏差 < 10° 且随 mu 单调",
                      ("repose_angle_dev_deg", "repose_angle_deviation_deg"),
                      "le", 10.0, unit="deg", source="Coulomb friction cone"),
            Criterion("C2", "Beverloo 标度律拟合 R² > 0.95",
                      ("beverloo_r2", "beverloo_fit_r2"), "ge", 0.95, unit="-",
                      source="Beverloo et al. (1961)"),
            Criterion("C3", "Beverloo 常数 C 与 0.58 偏差 < 20%",
                      ("beverloo_c_rel_error", "beverloo_constant_rel_error"), "le", 0.20,
                      unit="-", source="Beverloo et al. (1961)", external=True),
            Criterion("C5", "卸料总质量 = 初始填充质量，相对误差 < 1e-6",
                      ("mass_conservation_error", "mass_balance_error"), "le", 1e-6,
                      unit="-", source="mass ledger"),
            Criterion("C6", "能量账本残差 < 1e-8",
                      ("energy_residual", "energy_balance_residual"), "le", 1e-8, unit="-",
                      source="gravity/kinetic/dissipation ledger"),
        ),
        lineage_title="N05: TensorDEM 散货舱颗粒卸料动力学验证面",
    ),
    NodeSpec(
        node_key="P84W2C-N06",
        mission_id=8018,
        repo="ship-performance-platform",
        title="船舶岸电(AMP)靠港排放折减可算面·成熟化",
        physics_domain="绿色航行",
        modules=("backend/physics/shore_power/",),
        tests=("tests/test_shore_power.py",),
        artifact_globs=("artifacts/shore_power/*.json",),
        criteria=(
            Criterion("C1", "辅机负荷平衡相对误差 < 1e-9",
                      ("aux_load_balance_error", "aux_power_balance_rel_error"), "le",
                      1e-9, unit="-", source="power ledger"),
            Criterion("C2", "CO2 = fuel × EF 质量守恒相对误差 < 1e-10",
                      ("co2_conservation_error", "co2_mass_error"), "le", 1e-10, unit="-",
                      source="MGO CF = 3.114"),
            Criterion("C3", "岸电折减 CO2 非负",
                      ("co2_reduction_t", "reduction_co2_t"), "ge", 0.0, unit="t",
                      source="grid EF <= aux EF"),
            Criterion("C5", "港口电网因子与公开数据偏差 < 5%",
                      ("grid_factor_rel_error", "grid_factor_deviation"), "le", 0.05,
                      unit="-", source="IEA/EEA grid emission factors", external=True),
        ),
        contract_id="amp-eexi",
        lineage_title="N06: 船舶岸电(AMP)靠港排放折减可算面·成熟化",
    ),
    NodeSpec(
        node_key="P84W2C-N07",
        mission_id=8019,
        repo="ship-performance-platform",
        title="主机余热回收 ORC 闭环 + EEDI credit·成熟化",
        physics_domain="绿色航行",
        modules=("backend/modules/waste_heat_orc/",),
        tests=("tests/test_waste_heat_orc.py",),
        artifact_globs=("artifacts/whr_orc/*.json",),
        criteria=(
            Criterion("C1", "循环能量守恒残差 < 1e-6·Q_in",
                      ("energy_residual", "cycle_energy_residual"), "le", 1e-6, unit="-",
                      source="first law"),
            Criterion("C2", "η_th / η_Carnot ∈ [0.35, 0.70]",
                      ("eta_over_carnot", "thermal_efficiency_ratio"), "in_range",
                      (0.35, 0.70), unit="-", source="second law bound"),
            Criterion("C3", "75% 负荷最小换热温差 ≥ 10 K",
                      ("min_pinch_k", "pinch_k", "pinch_temperature_k"), "ge", 10.0,
                      unit="K", source="pinch analysis"),
            Criterion("C5", "EEDI 折减双算法相对差 < 1e-6",
                      ("eedi_dual_rel_diff", "deduction_dual_rel_diff"), "le", 1e-6,
                      unit="-", source="MEPC.1/Circ.896"),
        ),
        lineage_title="N07: 主机余热回收 ORC 闭环 + EEDI credit·成熟化",
    ),
    NodeSpec(
        node_key="P84W2C-N08",
        mission_id=8020,
        repo="ship-performance-platform",
        title="货物系固系统验证面·成熟化",
        physics_domain="舾装",
        modules=("backend/physics/cargo_securing/",),
        tests=("tests/test_cargo_securing.py",),
        artifact_globs=("artifacts/cargo_securing/*.json",),
        criteria=(
            Criterion("C1", "加速度椭圆双实现相对误差 < 1e-12",
                      ("acceleration_dual_rel_error", "acceleration_crosscheck_error"),
                      "le", 1e-12, unit="-", source="CSS Code Annex 13"),
            Criterion("C2", "每工况 ΣF=0、ΣM=0 残差 < 1e-9",
                      ("force_moment_residual", "equilibrium_residual"), "le", 1e-9,
                      unit="-", source="static equilibrium"),
            Criterion("C3", "绑扎数 2→6 单根最大张力单调不增 (R² > 0.99)",
                      ("lashing_monotonic_r2", "lashing_r2"), "ge", 0.99, unit="-",
                      source="minimum-norm lashing solution"),
        ),
        lineage_title="N08: 货物系固系统验证面·成熟化",
    ),
    NodeSpec(
        node_key="P84W2C-N09",
        mission_id=8021,
        repo="ship-performance-platform",
        title="进坞/坞墩工况可判面·成熟化",
        physics_domain="结构",
        modules=("backend/physics/drydock/",),
        tests=("tests/test_drydock_strength.py",),
        artifact_globs=("artifacts/drydock/*.json",),
        criteria=(
            Criterion("C1", "ΣR_i = W 静力平衡残差 < 1e-6·W",
                      ("static_equilibrium_error", "reaction_balance_error", "equilibrium_residual"),
                      "le", 1e-6,
                      unit="-", source="static equilibrium"),
            Criterion("C2", "Winkler 与三弯矩法 M_max 相对差 < 1%",
                      ("two_model_mmax_rel_diff", "mmax_crosscheck_rel_diff",
                       "M_max_relative_difference", "max_relative_difference"), "le", 0.01,
                      unit="-", source="two independent models"),
            Criterion("C4", "全墩脱空退化到自由-自由梁解析弯矩，误差 < 1%",
                      ("degradation_rel_error", "free_free_moment_error",
                       "free_floating_relative_error"), "le", 0.01,
                      unit="-", source="free-free beam analytic"),
        ),
        lineage_title="N09: 进坞/坞墩工况可判面·成熟化",
    ),
    NodeSpec(
        node_key="P84W2C-N10",
        mission_id=8022,
        repo="ship-performance-platform",
        title="谷物稳性(International Grain Code)可算面·成熟化",
        physics_domain="稳性",
        modules=("backend/physics/grain_stability/",),
        tests=("tests/test_grain_stability.py",),
        artifact_globs=("artifacts/grain_stability/*.json",),
        criteria=(
            Criterion("C1", "VHM→SF→λ 单位链双实现相对误差 < 1e-12",
                      ("unit_chain_rel_error", "vhm_conversion_rel_error"), "le", 1e-12,
                      unit="-", source="International Grain Code"),
            Criterion("C2", "左右舷镜像配载倾侧力矩相对差 < 1e-9",
                      ("symmetry_rel_error", "mirror_rel_error"), "le", 1e-9, unit="-",
                      source="mirror symmetry"),
            Criterion("C3", "部分装载比例 0→60% 倾侧力矩单调增 (R² > 0.99)",
                      ("heeling_monotonic_r2", "vhm_monotonic_r2"), "ge", 0.99, unit="-",
                      source="Grain Code heeling moment"),
        ),
        lineage_title="N10: 谷物稳性(International Grain Code)可算面·成熟化",
    ),
)

NODE_BY_KEY: dict[str, NodeSpec] = {n.node_key: n for n in NODES}
NODE_BY_MISSION: dict[int, NodeSpec] = {n.mission_id: n for n in NODES}


# ---------------------------------------------------------------------------
# Cross-domain interface contracts (bilateral: producer schema + consumer code).
# ---------------------------------------------------------------------------
CONTRACTS: tuple[ContractSpec, ...] = (
    ContractSpec(
        contract_id="aero-wind",
        producer_node="P84W2C-N01",
        producer_schema="aero_polar.v1",
        producer_required=("alpha_deg", "C_L", "C_D", "rho_air_kg_m3", "chord_m"),
        consumer_repo="ship-performance-platform",
        consumer_file="backend/api/dynamic_stability.py",
        consumer_symbols=("DynamicStabilityInput",),
        consumer_fields=("wind_pressure", "wind_area", "wind_lever"),
        consumer_constants={"RHO_AIR": 1.225, "WIND_PRESSURE_IS_CODE": 504.0},
        numeric_anchor="wind_pressure=0.5*rho_air*V^2",
        tolerance=1e-12,
    ),
    ContractSpec(
        contract_id="sediment-ukc",
        producer_node="P84W2C-N04",
        producer_schema="sediment_summary.v1",
        producer_required=("shields_curve", "rouse_profiles", "bed_elevation",
                           "scour_depth", "mass_budget"),
        consumer_repo="ship-performance-platform",
        consumer_file="backend/api/calculations.py",
        consumer_symbols=(),
        consumer_fields=("ukc",),
        consumer_constants={},
        numeric_anchor="under-keel clearance margin in metres",
        tolerance=1e-12,
        required=False,
    ),
    ContractSpec(
        contract_id="amp-eexi",
        producer_node="P84W2C-N06",
        producer_schema="amp_emissions.v1",
        producer_required=("aux_power_kw", "fuel_t", "co2_t", "co2_reduction_t", "p_eff_kw"),
        consumer_repo="ship-performance-platform",
        consumer_file="backend/api/green_shipping.py",
        consumer_symbols=("EediRequest", "EexiRequest"),
        consumer_fields=("p_eff_kw", "p_me_kw", "v_ref_kn", "capacity_t"),
        consumer_constants={},
        numeric_anchor="innovative-energy-technology credit enters EEDI/EEXI as p_eff_kw (kW)",
        tolerance=1e-12,
    ),
)

CONTRACT_BY_ID: dict[str, ContractSpec] = {c.contract_id: c for c in CONTRACTS}

#: campaign-level machine criteria (the mission goal's C1..C4)
GATE_CRITERIA = {
    "C1": "所有节点 pytest 结果可解析 (JSON)",
    "C2": "跨域接口契约验证通过",
    "C3": "战役报告 markdown ≥ 3000 字符",
    "C4": "端到端输出战役验收 JSON（schema 校验 + 指纹落盘）",
}

MIN_REPORT_CHARS = 3000

#: parent plan used for the lineage adjudication (P84-W2 seed)
PARENT_PLAN_ID = 365