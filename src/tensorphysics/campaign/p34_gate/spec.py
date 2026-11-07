"""P34 campaign node table: the declarative contract the gate evaluates.

P34 is a seven node cross-repository research campaign.  The nodes live in
different repositories (TensorLBM x2, TensorDEM, ship-performance-platform x3,
TensorPlatform), so the only thing the acceptance gate can rely on is what each
node *leaves behind*: a git branch, source modules, a pytest file, and numeric
artifacts.  This module is the single source of truth for that contract.

Nothing here is domain physics: the table only names paths, minimum test counts
and the artifact families a node promised in its mission goal.  All evaluation
logic lives in ``collect.py`` / ``criteria.py`` and is written against the table,
never against a specific node.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CAMPAIGN_ID = "P34"
GATE_ID = "P34-G1"
CAMPAIGN_TITLE = "P34 跨域能力串联战役"

#: Criteria identifiers are part of the gate contract (verdict schema below).
CRITERIA = {
    "C1": "全部 7 个任务节点 pytest 全绿",
    "C2": "全部 artifacts 完整落盘 (JSON + PNG)",
    "C3": "既有模块回归测试全绿",
    "C4": "战役总结报告论文级质量 (六要素完整)",
    "C5": "零领域语义扫描通过",
}


@dataclass(frozen=True)
class ArtifactRequirement:
    """One artifact family a node promised to deliver."""

    kind: str          # "json" | "png"
    pattern: str       # glob relative to the node worktree root
    note: str = ""


@dataclass(frozen=True)
class NodeSpec:
    ext_id: str
    mission_id: int
    repo: str
    title: str
    modules: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    artifacts: tuple[ArtifactRequirement, ...] = ()
    min_tests: int = 1
    acceptance: tuple[tuple[str, str], ...] = ()


def _png(glob: str, note: str = "") -> ArtifactRequirement:
    return ArtifactRequirement("png", glob, note)


def _json(glob: str, note: str = "") -> ArtifactRequirement:
    return ArtifactRequirement("json", glob, note)


# --------------------------------------------------------------------------
# The seven P34 nodes, transcribed from the campaign mission goals.
# --------------------------------------------------------------------------
NODES: tuple[NodeSpec, ...] = (
    NodeSpec(
        ext_id="P34-N1",
        mission_id=5490,
        repo="TensorLBM",
        title="波浪水池基准: Stokes 二阶造波/消波 + 两点法反射系数 + 质量守恒",
        modules=(
            "src/tensorlbm/wave_tank/",
            "benchmarks/verified/wave_tank/",
        ),
        tests=("tests/test_wave_tank.py",),
        artifacts=(
            _json("artifacts/wave_tank/*.json", "色散/谐波/反射/质量守恒判据数字"),
            _png("artifacts/wave_tank/*.png", "波形与反射系数图"),
        ),
        min_tests=7,
        acceptance=(
            ("C1", "相速度 vs c²=(g/k)tanh(kh) 误差 < 1% (>=3 个 kh)"),
            ("C2", "一阶谐波幅值误差 < 3%，二阶谐波误差 < 10%"),
            ("C3", "消波反射系数 < 5%，负控关消波 > 30%"),
            ("C4", "质量相对漂移 < 0.5%/周期 x 5 周期"),
            ("C5", "网格收敛: dx 减半波形变化 < 3%"),
            ("C6", "纳入 benchmarks/verified 索引，现有 pytest 全绿"),
            ("C7", "单算例 < 20 min (loong64)"),
        ),
    ),
    NodeSpec(
        ext_id="P34-N2",
        mission_id=5491,
        repo="TensorLBM",
        title="附加质量不稳定性定标: 1D 解析收敛因子 + IQN-ILS/Aitken 稳定化",
        modules=(
            "src/tensorlbm/fsi_coupling.py",
            "benchmarks/pending/added_mass_instability/",
        ),
        tests=("tests/test_fsi_added_mass.py",),
        artifacts=(
            _json("artifacts/fsi/*.json", "收敛因子 / 稳定域 / 成本表"),
            _png("artifacts/fsi/*.png", "稳定域图与残差历史"),
        ),
        min_tests=12,
        acceptance=(
            ("D1", "1D 解析 vs 数值收敛因子 <= 5%，临界密度比判据 <= 10%"),
            ("D2", "负控: 显式 rho_s/rho_f=0.1 必须发散; IQN-ILS 20 次内残差 1e-6"),
            ("D3", "物理不变性 (自由衰减) <= 5%"),
            ("D4", "成本表落 JSON 且与报告一致"),
            ("D5", "pytest >= 12 条含负控与回归基线"),
        ),
    ),
    NodeSpec(
        ext_id="P34-N3",
        mission_id=5492,
        repo="TensorDEM",
        title="真形接触 superquadric vs 球并集 clump 误差定标",
        modules=(
            "src/tensordem/superquadric.py",
            "src/tensordem/shape_error_budget.py",
        ),
        tests=("tests/test_superquadric_contact.py",),
        artifacts=(
            _json("artifacts/dem/*.json", "退化检验/误差表/禁用比例表"),
            _png("artifacts/dem/*.png", "表面偏差与休止角图"),
        ),
        min_tests=14,
        acceptance=(
            ("E1", "球极限 Hertz 退化 <= 1%，立方体平压刚度 <= 2%"),
            ("E2", "无穿透 > 1e-3·D，纯弹性冲击能量漂移 < 1%/1000 步"),
            ("E3", "球颗粒休止角 22°±2°，theta 差与力链偏差落表，禁用表被 API 闸门读取"),
            ("E4", "真形接触单步成本 <= 3x clump (loong64 实测)"),
            ("E5", "pytest >= 14 条全绿"),
        ),
    ),
    NodeSpec(
        ext_id="P34-N4",
        mission_id=5493,
        repo="ship-performance-platform",
        title="砰击载荷与弹性响应耦合: Wagner 峰值 + 模态叠加 + springing/whipping",
        modules=(
            "backend/api/hydroelastic.py",
            "backend/physics/hydroelastic/springing_whipping.py",
            "tools/hydroelastic_rao.py",
        ),
        tests=("tests/test_hydroelastic.py",),
        artifacts=(
            _json("artifacts/hydroelastic/*.json", "湿模态/RAO 峰值/whipping 放大因子"),
            _png("artifacts/hydroelastic/*.png", "RAO 曲线 + springing 海况 + whipping 时域"),
        ),
        min_tests=10,
        acceptance=(
            ("F1", "湿模态 f_wet 在 [0.15,0.25] Hz 与文献误差 <= 10%"),
            ("F2", "2 节点 RAO 峰值与 ITTC 推荐值误差 <= 15%"),
            ("F3", "zeta_2=0.03 时 Delta_omega/omega_2 = 0.06 ± 0.005"),
            ("F4", "whipping 放大因子 1.5-3.0 且非硬编码"),
            ("F5", "pytest >= 10 条全绿 + 三类图落盘"),
        ),
    ),
    NodeSpec(
        ext_id="P34-N5",
        mission_id=5494,
        repo="ship-performance-platform",
        title="波浪载荷长期极值与规范设计波双轨对拍",
        modules=(
            "backend/api/wave_loads_extreme.py",
            "backend/data/wave_scatter_natl.json",
            "tools/longterm_extreme.py",
        ),
        tests=("tests/test_longterm_extreme.py",),
        artifacts=(
            _json("artifacts/extreme/*.json", "极值曲面 / 散布图表 / 差异台账"),
            _png("artifacts/extreme/*.png", "超越概率与极值曲面图"),
        ),
        min_tests=16,
        acceptance=(
            ("F1", "Rayleigh 3.717 sigma 复现 <= 0.5%，宽带修正 <= 2%"),
            ("F2", "长期两法 25 年极值 <= 5%"),
            ("F3", "与 CSR 设计波 <= 10% (超差留台账)"),
            ("F4", "负控 gamma 单调性通过，散布图 sum p = 1 ± 1e-9"),
            ("F5", "极值曲面 JSON + 图落盘，pytest >= 16 条全绿"),
        ),
    ),
    NodeSpec(
        ext_id="P34-N6",
        mission_id=5495,
        repo="ship-performance-platform",
        title="绿色航行: 空气润滑净节能闭环 + 粗糙度摩擦退化 ISO19030 标度律",
        modules=(
            "backend/api/air_lubrication.py",
            "backend/api/hull_roughness.py",
            "backend/physics/air_lubrication/bubble_cavity.py",
            "backend/physics/roughness/friction_degradation.py",
            "tools/air_lubrication_net_saving.py",
            "tools/roughness_energy_penalty.py",
        ),
        tests=("tests/test_air_lubrication.py", "tests/test_hull_roughness.py"),
        artifacts=(
            _json("artifacts/air_lubrication/*.json", "净节能闭环数字"),
            _png("artifacts/air_lubrication/*.png", "净节能曲面"),
            _png("artifacts/roughness/*.png", "粗糙度函数曲线与时变退化曲线"),
        ),
        min_tests=14,
        acceptance=(
            ("G1-G9", "净节能闭环 + 粗糙度退化标度律九项判据 (详见节点目标)"),
            ("G9", "pytest >= 14 条全绿 + 四类图落盘"),
        ),
    ),
    NodeSpec(
        ext_id="P34-N7",
        mission_id=5496,
        repo="TensorPlatform",
        title="平台报告归属门与判据变异门成熟化 (mission_id 归属 + PNG 魔数 + kill-rate)",
        modules=(
            "tools/report_payload.py",
            "tools/audit/criteria_mutation.py",
        ),
        tests=("tests/test_report_ownership.py", "tests/test_criteria_mutation.py"),
        artifacts=(
            _json("artifacts/criteria_mutation/*.json", "逐门 kill-rate 台账"),
            _png("artifacts/criteria_mutation/*.png", "kill-rate 曲线"),
        ),
        min_tests=12,
        acceptance=(
            ("C1", "越权提交 403 REPORT_MISSION_NOT_OWNED"),
            ("C2", "8 线程并发串号率 0/8"),
            ("C3", "JPEG/WebP -> 422 且错误信息含 PNG 魔数"),
            ("C6/C7", "PAPER-TRACE 门 kill-rate 49.5% -> >= 90%"),
            ("C10", "恒真判据 kill-rate 0/5 且标记不可证伪"),
        ),
    ),
)

NODE_BY_ID = {n.ext_id: n for n in NODES}

#: Regression families required by criterion C3.  Each family is satisfied by at
#: least one recorded suite whose ``family`` field matches and that is green.
REQUIRED_REGRESSION_FAMILIES = (
    "platform_core",
    "resistance",
    "propulsion",
    "seakeeping",
    "maneuvering",
    "stability",
)

#: Vocabulary that belongs to the *domain* repositories, not to workflow logic.
#: Criterion C5 proves these tokens do not leak into the gate's executable logic
#: (they may only appear in this declarative table and in the summary document).
DOMAIN_LEXICON = (
    "阻力", "推进", "耐波", "操纵", "稳性", "砰击", "波浪弯矩", "螺旋桨",
    "hull", "propeller", "slamming", "springing", "whipping", "seakeeping",
    "resistance", "roll damping", "added resistance",
)

#: Placeholder markers that turn a "delivered" artifact into an empty claim.
PLACEHOLDER_TOKENS = (
    "TODO", "FIXME", "TBD", "lorem ipsum", "待补", "占位", "placeholder",
    "XXX", "填这里",
)

#: Repositories whose kernels the gate must never import (boundary discipline).
FORBIDDEN_IMPORT_ROOTS = ("numpy", "scipy", "torch", "tensorlbm", "tensordem", "tensorfem", "backend")