"""P35 campaign node table: the declarative contract the acceptance gate judges.

P35 is a ten node cross-repository campaign (``P35-N1`` .. ``P35-N10``) plus the
integration gate ``P35-G1``.  The nodes live in TensorLBM (2), TensorDEM (1),
ship-performance-platform (5) and TensorPhysics (1), so the only thing a gate can
rely on is what each node *leaves behind* on the platform and on disk: a report,
a booked capability delta, and the artifact families its mission goal promised.

Nothing in this module is domain physics.  It only names identities, paths,
declared prerequisites and the acceptance-relevant paths, so that the evaluation
logic in :mod:`criteria` / :mod:`regression` stays node-agnostic.  Every entry is
transcribed from the mission goal of the corresponding ``campaign_plans`` node
(plan 179, ``P35_wave1``) and from the plan's ``physics_scope`` list.

``DECLARED_PREREQUISITES`` is a *declaration*, not a discovery: the platform's
``campaign_plans`` rows for P35 carry empty ``dependencies`` for all eleven nodes
(verified 2026-11-08), so the DAG that the campaign actually claims to execute is
the one written into the node goals ("复用 N1 参数", "DEM 冰 floe" ...).  The gate
records both views and reports the divergence instead of silently trusting one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CAMPAIGN_ID = "P35"
GATE_ID = "P35-G1"
CAMPAIGN_TITLE = "P35·多域求解器前沿与船舶深域拓展"
PLAN_ID = 179

#: criteria identifiers are part of the verdict/readiness contract.
CRITERIA = {
    "a": "10 个节点产物落盘率 100%",
    "b": "能力台账登记率 100% (ext_id/repo/physics_domain)",
    "c": "回归基线偏差 <1%",
    "d": "依赖链完整且无孤立节点",
    "e": "成熟化节点 (N4/N5/N6/N7) 门禁通过率 100%",
    "f": "campaign_readiness JSON 合法且通过 schema 校验",
    "g": "honest-FAIL 语义：任一节点未通过 → FAIL + 阻塞清单",
}

#: regression tolerance requested by the campaign for criterion (c).
REGRESSION_THRESHOLD = 0.01

#: TRL floor a "matured" node has to declare for criterion (e).
MATURITY_TRL_FLOOR = 6

#: physics domains the P35 plan declares, plus the one P35-N3 needs but the plan
#: does not list (recorded as a finding by the gate, not silently accepted).
PLAN_PHYSICS_SCOPE = (
    "ice_navigation",
    "green_navigation",
    "hydroelasticity",
    "multiphase_flow",
    "cargo_safety",
    "shaft_alignment",
)
UNSCOPED_PHYSICS_DOMAINS = ("wave_basin_inversion",)


@dataclass(frozen=True)
class ArtifactRequirement:
    """One artifact family a node promised to deliver."""

    kind: str      # "json" | "png"
    pattern: str   # glob relative to the node repository root
    note: str = ""


@dataclass(frozen=True)
class Prerequisite:
    """A declared upstream a node consumes.

    ``kind`` is ``"node"`` for an in-campaign edge (``ref`` is another ``ext_id``)
    or ``"interface"`` for an upstream that lives outside this campaign (an
    already-verified module or a named interface), where ``ref`` names the
    interface the node goal cites.
    """

    kind: str
    ref: str


@dataclass(frozen=True)
class NodeSpec:
    ext_id: str
    mission_id: int
    repo: str
    physics_domain: str
    title: str
    prerequisites: tuple[Prerequisite, ...] = ()
    artifact_dir: str = ""
    maturity: bool = False
    acceptance: tuple[str, ...] = ()
    notes: str = ""

    @property
    def artifacts(self) -> tuple[ArtifactRequirement, ...]:
        """Artifact families every node must leave in ``artifact_dir``.

        The directory is the node's own product folder (verified on the delivery
        worktrees of P35, e.g. ``artifacts/lng_rollover_m6800``); ``**`` keeps the
        check tolerant of the ``figures/`` sub-folder convention without
        accepting an empty family.
        """
        families = [
            _json(f"{self.artifact_dir}/**/*.json", "数值判据与账本 (程序生成, 零手抄)"),
            _png(f"{self.artifact_dir}/**/*.png", "场量云图/对标曲线"),
        ]
        if self.maturity:
            families.insert(1, _json(maturity_pattern(self.ext_id),
                                     "成熟化声明 (trl + 门禁记录)"))
        return tuple(families)


def _json(glob: str, note: str = "") -> ArtifactRequirement:
    return ArtifactRequirement("json", glob, note)


def _png(glob: str, note: str = "") -> ArtifactRequirement:
    return ArtifactRequirement("png", glob, note)


def maturity_pattern(ext_id: str) -> str:
    """Path of the maturity declaration a matured node has to leave on disk."""
    node = NODE_BY_ID.get(ext_id)
    base = node.artifact_dir if node else "artifacts"
    return f"{base}/maturity.json"


def _n(ext_id: str) -> Prerequisite:
    return Prerequisite("node", ext_id)


def _iface(ref: str) -> Prerequisite:
    return Prerequisite("interface", ref)


# --------------------------------------------------------------------------
# The ten P35 task nodes, transcribed from plan 179 node goals.
# --------------------------------------------------------------------------
NODES: tuple[NodeSpec, ...] = (
    NodeSpec(
        ext_id="P35-N1",
        artifact_dir="artifacts/thermal_multiphase",
        mission_id=6795,
        repo="TensorLBM",
        physics_domain="multiphase_flow",
        title="LBM 热多相流相变验证面：Lee 模型蒸发/冷凝 + 单气泡 R²∝t 标度律",
        prerequisites=(_iface("tensorlbm.thermal_multiphase"),),
        acceptance=("R²-t 拟合 R²>0.99", "相界面温度偏差<0.5K", "能量残差<2%",
                    "网格收敛阶>=1.5", "关闭相变源项回到纯导热<1%"),
        notes="战役根节点（唯一无 in-campaign 上游的节点之一）",
    ),
    NodeSpec(
        ext_id="P35-N2",
        artifact_dir="artifacts/wet_ice_cohesion",
        mission_id=6796,
        repo="TensorDEM",
        physics_domain="ice_navigation",
        title="DEM 湿冰毛细桥 cohesion：液桥力解析锚 + 团聚体尺寸分布 + 冰阻力增强因子",
        prerequisites=(_iface("tensordem.contact_dry_ice"),),
        acceptance=("液桥力峰值误差<5%", "<D> vs Smoluchowski<15%", "F_wet/F_dry 偏差<20%",
                    "γ=0 负控回到干冰<1%", "加密一倍变化<2%"),
    ),
    NodeSpec(
        ext_id="P35-N3",
        artifact_dir="artifacts/wave_probe_inversion",
        mission_id=6797,
        repo="TensorLBM",
        physics_domain="wave_basin_inversion",
        title="造波水池探针反演可辨识性定律：三点 Goda×粘性衰减×复波数联合反演",
        prerequisites=(_iface("benchmarks.verified.wave_tank"),),
        acceptance=("α*=0 合成靶 R̂ 误差<1e-3", "k̂ 相对误差<1e-4", "达标枚举包络落盘"),
        notes="该节点声明的物理域 wave_basin_inversion 不在 plan physics_scope 内（门记录为发现）",
    ),
    NodeSpec(
        ext_id="P35-N4",
        artifact_dir="artifacts/air_lubrication",
        mission_id=6798,
        repo="ship-performance-platform",
        physics_domain="green_navigation",
        title="船体空气润滑减阻成熟化：气腔形成判据 + 摩擦减阻量化 + 净节能闭环",
        prerequisites=(_iface("ship.resistance_core"), _iface("ship.ship_green")),
        maturity=True,
        acceptance=("气腔判据偏差<10%", "减阻率误差<15%", "净节能包络一致率>90%"),
    ),
    NodeSpec(
        ext_id="P35-N5",
        artifact_dir="artifacts/shaft_alignment",
        mission_id=6799,
        repo="ship-performance-platform",
        physics_domain="shaft_alignment",
        title="轴系校中成熟化：影响数法↔三弯矩互裁 + 轴承反力可容许域 + 冷/热态变换",
        prerequisites=(_iface("ship.shaft_model_beam"),),
        maturity=True,
        acceptance=("两法反力偏差<1%", "影响系数对称性<1e-6", "冷/热态解析偏差<2%"),
    ),
    NodeSpec(
        ext_id="P35-N6",
        artifact_dir="artifacts/lng_rollover_m6800",
        mission_id=6800,
        repo="ship-performance-platform",
        physics_domain="cargo_safety",
        title="LNG 舱分层-滚动 rollover 成熟化：双扩散能量账本 + Turner 角/N² 解析锚",
        prerequisites=(_iface("ship.bog_steady"),),
        maturity=True,
        acceptance=("Turner 判定一致率 100%", "N² 误差<2%", "ΔE_p 偏差<3%", "BOG 倍数偏差<25%"),
    ),
    NodeSpec(
        ext_id="P35-N7",
        artifact_dir="artifacts/alt_fuel_propulsion",
        mission_id=6801,
        repo="ship-performance-platform",
        physics_domain="green_navigation",
        title="替代燃料推进成熟化：LNG/H2/NH3 消耗-排放因子-EEXI/CII 合规回流",
        prerequisites=(_iface("ship.eexi_cii_engine"),),
        maturity=True,
        acceptance=("消耗率-功率 R²>0.99", "排放因子误差<5%", "传统燃油路径负控一致"),
    ),
    NodeSpec(
        ext_id="P35-N8",
        artifact_dir="artifacts/m6802_frank_kamenetskii",
        mission_id=6802,
        repo="ship-performance-platform",
        physics_domain="cargo_safety",
        title="散货自热 Frank-Kamenetskii 临界判据：δ_crit 解析锚 + 通风边界 + 临界堆高",
        prerequisites=(_iface("ship.effective_conductivity.k_eff"),),
        acceptance=("δ_crit 相对误差<1%", "网格收敛阶>=1.8", "Bi 律<2%", "k_eff 接口缺失判 BLOCKED"),
    ),
    NodeSpec(
        ext_id="P35-N9",
        artifact_dir="artifacts/lbm_dem_fsi",
        mission_id=6803,
        repo="TensorLBM",
        physics_domain="ice_navigation",
        title="LBM-DEM-FSI 三向耦合冰水相互作用：动量交换/能量交接/时间同步端到端链",
        prerequisites=(_n("P35-N1"), _n("P35-N2")),
        acceptance=("LBM→DEM 动量残差<2%", "DEM→FSI 能量残差<3%", "时间账本残差<3%",
                    "无 FSI 反馈负控<1e-6"),
    ),
    NodeSpec(
        ext_id="P35-N10",
        artifact_dir="artifacts/p35_chain",
        mission_id=6804,
        repo="TensorPhysics",
        physics_domain="hydroelasticity",
        title="P34 三段链端到端串联账本：造波→6DOF→砰击水弹性 力-能-时三重交接守恒门",
        prerequisites=(_n("P35-N1"), _iface("p34.wave_tank"), _iface("p34.fsi_added_mass"),
                      _iface("p34.hydroelastic")),
        acceptance=("面 A 力残差<2%", "面 C 广义力残差<3%", "能量账本<3%",
                    "Δt/2 偏移检出>10%", "缺面必须 WITHHELD"),
        notes="声明复用 P35-N1 波浪参数（goal 原文「复用N1参数」）",
    ),
)

NODE_BY_ID = {n.ext_id: n for n in NODES}
NODE_BY_MISSION = {n.mission_id: n for n in NODES}

#: Nodes the campaign designates as roots: they must still declare a
#: prerequisite interface, otherwise criterion (d) flags them as orphans.
ROOT_NODES = ("P35-N1",)

#: Matured nodes (criterion e) -- the campaign names N4/N5/N6/N7 explicitly.
MATURITY_NODES = tuple(n.ext_id for n in NODES if n.maturity)

#: The gate node itself depends on every task node.
GATE_UPSTREAMS = tuple(n.ext_id for n in NODES)

#: Every registered capability must name a domain from this set.
ALLOWED_PHYSICS_DOMAINS = PLAN_PHYSICS_SCOPE + UNSCOPED_PHYSICS_DOMAINS


# --------------------------------------------------------------------------
# Regression baseline families (criterion c)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RegressionCase:
    """One already-verified case the gate re-runs against its recorded baseline.

    ``baseline_path`` / ``metric_path`` are relative to ``repo_root`` and use
    dotted lookup; ``pattern`` locates the probe output JSON written by
    ``command`` (``{out}`` is substituted with a scratch directory).
    """

    case: str
    repo: str
    command: tuple[str, ...]
    baseline_path: str
    metric_path: str
    metric_pattern: str
    threshold: float = REGRESSION_THRESHOLD
    floor: float = 0.0
    budget_s: float | None = None
    estimated_cost_s: float | None = None
    note: str = ""


#: The three verified faces the campaign names, plus the analytic gate each case
#: already carries.  Commands are the producers' own entry points; the gate never
#: re-implements a solver kernel, it only runs the published runner.
REGRESSION_CASES: tuple[RegressionCase, ...] = (
    RegressionCase(
        case="couette_3d",
        repo="TensorLBM",
        command=("python3", "benchmarks/verified/couette_3d/run.py", "--dtype", "fp64",
                 "--device", "cpu", "--H", "16", "--out-dir", "{out}"),
        baseline_path="benchmarks/verified/couette_3d/result.json",
        metric_path="grids.H16.l2_rel_err",
        metric_pattern="case_H16_fp64.json:l2_rel_err",
        floor=1e-10,
        note="fp64 晋级位；l2 处于机器地板，比较用 floor 防止噪声被百分比放大",
    ),
    RegressionCase(
        case="cavity_natural_convection",
        repo="TensorLBM",
        command=("python3", "benchmarks/verified/cavity_natural_convection/run.py",
                 "--nx", "64", "--steps", "100000", "--out", "{out}/case_n64.json"),
        baseline_path="benchmarks/verified/cavity_natural_convection/result.json",
        metric_path="grids.64.nu",
        metric_pattern="case_n64.json:grids.64.nu",
        note="de Vahl Davis 1983 Ra=1e4 的 Nu；runner 入口在检出内缺失时判 WITHHELD",
    ),
    RegressionCase(
        case="cavity_3d_full",
        repo="TensorLBM",
        command=("python3", "benchmarks/verified/cavity_3d_full/run.py", "single", "64",
                 "{out}/case_64.json", "--device", "cpu", "--steps", "100000"),
        baseline_path="benchmarks/verified/cavity_3d_full/result.json",
        metric_path="grids.64.u_min",
        metric_pattern="case_64.json:u_min",
        budget_s=3600.0,
        estimated_cost_s=19400.0,
        note=("Ku/Hirsh/Taylor 1987 u_min；实测 500 步 97 s (x86 CPU, 8 线程, 64³) → 100k 步 "
              "≈1.94e4 s ≈5.4 h，超出本门 1 h 预算，按 WITHHELD 记账"),
    ),
)

REGRESSION_CASE_BY_NAME = {c.case: c for c in REGRESSION_CASES}