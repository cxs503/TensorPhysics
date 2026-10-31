"""雪-冰复合层弯曲力学模型 (Snow-Ice Composite Layer Mechanics).

物理背景
--------
冰盖上覆雪后形成 ``snow / snow-ice / ice`` 三层介质。雪层模量比冰低 2~3 个量级，
雪-冰（snow-ice，冻结雪泥）模量与强度也低于柱状冰，因此三层复合截面的弯曲刚度
``D_composite`` 与同总厚度的纯冰板不同；雪层还使中性轴偏离几何中面，改变受拉
边缘位置，从而改变板的抗弯承载能力。

本模块实现（均为单调、可机器复算的解析/半解析公式）：

1. 弹性加权中性轴
   ``z_na = Σ E'_i A_i z_i / Σ E'_i A_i``，``E'_i = E_i / (1 - ν_i²)``（平面应力模量）。

2. 复合弯曲刚度（单位宽度，Pa·m³）
   ``D_composite = Σ E'_i [ t_i³/12 + t_i (z_i - z_na)² ]``
   即 ``Σ E_i I_i`` 的平行轴展开；``h_snow = 0`` 时退化为
   ``E_ice h³ / (12 (1-ν_ice²))``（纯冰板）。

3. 弯曲强度折减系数
   以「下缘受拉」的正弯（sagging）为控制工况：曲率 κ 下第 i 层受拉最远纤维应变
   ``ε_i = κ (z_na - z_{i,bottom})``。首条裂缝由 ``κ_cr = min_i σ_f,i / (E'_i (z_na - z_{i,bot}))``
   给出，首裂弯矩 ``M_cr = D_composite · κ_cr``。把复合板按同总厚度均质板折算的
   表观弯曲强度
   ``σ_app = 6 (1 - ν²) M_cr / h_tot²``
   对纯冰板恰为 ``σ_f,ice``，故折减系数
   ``η_snow = σ_app / σ_f,ice``。

4. Winkler 弹性地基板
   ``D_composite ∇⁴w + k_f w = q``（简支矩形板，Navier 双三角级数解），
   用于给出「含雪 vs 无雪」的中心挠度对比。

单位约定：长度 m、模量 Pa、刚度 Pa·m³、弯矩 N·m/m、挠度 m。

文献与参数来源
--------------
- Saloranta, T. (2000), *Modeling the evolution of snow, snow ice and ice in the
  Baltic Sea*, Tellus A 52(1), doi:10.3402/tellusa.v52i1.12255 —— 该文是一维
  热力学（雪/雪冰/冰厚度演化）模型，提供 Baltic 海雪-冰层序与密度的量级；
  其出版记录中**没有复合梁弯曲实验数据**（见 docs/reports/m762 的审计节）。
  本模块把它仅作为层序/厚度的参考构型来源，不把它当作力学实验基准。
- Timco & Weeks (2010), *A review of the engineering properties of sea ice*,
  Cold Regions Science and Technology 60:107-129 —— 海冰 E 与弯曲强度的量级范围。
- 三层弯曲强度比率的真实三点弯曲实验锚点：*An Experimental Investigation of the
  Flexural Strength and Fracture Toughness of Granular Snow Ice Under a
  Three-Point Bending Test*, Water 2024, 16(23):3358, doi:10.3390/w16233358
  （−20 ℃ 粒状雪冰 σ_f = 2.85 MPa，柱状冰 1.93 MPa）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "IceLayer",
    "LayerStack",
    "DEFAULT_MATERIALS",
    "SALORANTA_2000",
    "SALORANTA_2000_CITATION",
    "GRANULAR_SNOW_ICE_BENDING_TEST",
    "default_layers",
    "snow_ice_composite",
    "snow_on_ice_fixed_total",
    "equivalent_homogeneous_modulus",
    "winkler_plate_center_deflection",
    "winkler_plate_deflection_profile",
    "winkler_plate_field",
]


# ---------------------------------------------------------------------------
# 材料常数（文献量级，可在调用处覆盖）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IceLayer:
    """单一材料层。

    Parameters
    ----------
    name : str
        层名（``snow`` / ``snow-ice`` / ``ice``）。
    thickness : float
        层厚 (m)，必须 >= 0。
    young : float
        杨氏模量 (Pa)。
    poisson : float
        泊松比，需满足 -1 < ν < 0.5。
    flexural_strength : float or None
        弯曲（受拉）强度 (Pa)。``None`` 表示该层无受拉承载能力（如新雪），
        若它落在受拉区会在 :meth:`LayerStack.first_crack_curvature` 中被显式
        当作 0 强度处理。
    density : float or None
        密度 (kg/m³)，仅用于自重载荷。
    """

    name: str
    thickness: float
    young: float
    poisson: float = 0.3
    flexural_strength: Optional[float] = None
    density: Optional[float] = None

    def __post_init__(self) -> None:
        if not (self.thickness >= 0.0):
            raise ValueError(f"thickness must be >= 0, got {self.thickness}")
        if not (self.young >= 0.0):
            raise ValueError(f"young must be >= 0, got {self.young}")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError(f"poisson must be in (-1, 0.5), got {self.poisson}")
        if self.flexural_strength is not None and self.flexural_strength < 0.0:
            raise ValueError("flexural_strength must be >= 0 when given")

    # -- 派生量 ---------------------------------------------------------
    @property
    def plane_stress_modulus(self) -> float:
        """平面应力模量 E' = E / (1 - ν²) (Pa)。"""
        return self.young / (1.0 - self.poisson**2)

    @property
    def with_material(self) -> str:
        return f"{self.name}(t={self.thickness:.4f}m, E={self.young:.3e}Pa)"


#: 文献量级默认材料参数（见模块 docstring 的来源说明）。
#: - 新雪：E ≈ 10 MPa（ρ≈300 kg/m³ 的雪，Mellor 型经验关系给 5~50 MPa），ν≈0.10，
#:   受拉强度视为 0（雪无抗拉能力），ρ≈300 kg/m³。
#: - 雪-冰（冻结雪泥/粒状冰）：E ≈ 4.0 GPa（孔隙率高于柱状冰，取海冰范围的低端），
#:   ν≈0.31，σ_f ≈ 0.60 MPa，ρ≈870 kg/m³。
#: - 柱状海冰：E ≈ 9.5 GPa，ν≈0.33，σ_f ≈ 0.72 MPa（Timco & Weeks 2010 量级），
#:   ρ≈910 kg/m³。
DEFAULT_MATERIALS: Dict[str, IceLayer] = {
    "snow": IceLayer(
        name="snow",
        thickness=0.0,
        young=1.0e7,
        poisson=0.10,
        flexural_strength=None,
        density=300.0,
    ),
    "snow-ice": IceLayer(
        name="snow-ice",
        thickness=0.0,
        young=4.0e9,
        poisson=0.31,
        flexural_strength=0.60e6,
        density=870.0,
    ),
    "ice": IceLayer(
        name="ice",
        thickness=0.0,
        young=9.5e9,
        poisson=0.33,
        flexural_strength=0.72e6,
        density=910.0,
    ),
}

#: Saloranta (2000) Baltic 海雪-冰层序的参考构型（量级，用于敏感性算例，不作实验基准）。
SALORANTA_2000: Dict[str, float] = {
    "h_snow": 0.10,
    "h_snow_ice": 0.05,
    "h_ice": 0.40,
}
SALORANTA_2000_CITATION: str = (
    "Saloranta, T. (2000) Modeling the evolution of snow, snow ice and ice in the "
    "Baltic Sea, Tellus A 52(1), doi:10.3402/tellusa.v52i1.12255"
)

#: 真实三点弯曲实验锚点：粒状雪冰 vs 柱状冰弯曲强度（−20 ℃，Water 2024, 16(23):3358）。
GRANULAR_SNOW_ICE_BENDING_TEST: Dict[str, float] = {
    "temperature_C": -20.0,
    "sigma_granular_snow_ice_Pa": 2.85e6,
    "sigma_columnar_ice_Pa": 1.93e6,
    "doi": "10.3390/w16233358",  # type: ignore[dict-item]
}


# ---------------------------------------------------------------------------
# 层叠体
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerStack:
    """自下而上排列的层叠体（``layers[0]`` 底面位于 ``z = 0``）。"""

    layers: Tuple[IceLayer, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        if not self.layers:
            raise ValueError("LayerStack requires at least one layer")
        for layer in self.layers:
            if layer.thickness <= 0.0:
                raise ValueError(
                    f"layer {layer.name!r} has non-positive thickness; "
                    "drop zero-thickness layers before building a LayerStack"
                )

    # -- 几何 -----------------------------------------------------------
    @property
    def total_thickness(self) -> float:
        """总厚度 h_tot (m)。"""
        return sum(layer.thickness for layer in self.layers)

    def z_bounds(self) -> List[Tuple[float, float]]:
        """各层 (z_bottom, z_top)，自下而上累加。"""
        bounds: List[Tuple[float, float]] = []
        z = 0.0
        for layer in self.layers:
            bounds.append((z, z + layer.thickness))
            z += layer.thickness
        return bounds

    def layer_centroids(self) -> List[float]:
        """各层形心高度 z_i (m)。"""
        return [0.5 * (zb + zt) for zb, zt in self.z_bounds()]

    # -- 中性轴 ---------------------------------------------------------
    def neutral_axis(self, *, plane_stress: bool = True) -> float:
        """弹性加权中性轴高度 z_na (m)，自底面积起算。

        ``z_na = Σ E'_i A_i z_i / Σ E'_i A_i``（单位宽度 A_i = t_i）。
        """
        num = 0.0
        den = 0.0
        for layer, z_i in zip(self.layers, self.layer_centroids()):
            e_mod = layer.plane_stress_modulus if plane_stress else layer.young
            num += e_mod * layer.thickness * z_i
            den += e_mod * layer.thickness
        if den <= 0.0:
            raise ValueError("degenerate stack: total E*A is zero")
        return num / den

    # -- 刚度 -----------------------------------------------------------
    def flexural_rigidity(self, *, plane_stress: bool = True) -> float:
        """单位宽度复合弯曲刚度 D_composite (Pa·m³)。

        ``D = Σ E'_i [ t_i³/12 + t_i (z_i - z_na)² ]``。
        """
        z_na = self.neutral_axis(plane_stress=plane_stress)
        d = 0.0
        for layer, z_i in zip(self.layers, self.layer_centroids()):
            e_mod = layer.plane_stress_modulus if plane_stress else layer.young
            t = layer.thickness
            d += e_mod * (t**3 / 12.0 + t * (z_i - z_na) ** 2)
        return d

    def axial_rigidity(self, *, plane_stress: bool = True) -> float:
        """单位宽度轴向刚度 Σ E'_i t_i (Pa·m)，用于一致性检查。"""
        return sum(
            (layer.plane_stress_modulus if plane_stress else layer.young)
            * layer.thickness
            for layer in self.layers
        )

    # -- 强度 -----------------------------------------------------------
    def first_crack_curvature(self, *, sagging: bool = True) -> Tuple[float, str]:
        """首裂曲率 κ_cr (1/m) 及控制层名。

        正弯 (sagging, ``sagging=True``)：下缘受拉，受拉最远纤维为各层底面；
        负弯：上缘受拉，受拉最远纤维为各层顶面。
        无受拉强度 (``flexural_strength=None``) 且落在受拉区的层 → κ_cr = 0。
        """
        z_na = self.neutral_axis()
        bounds = self.z_bounds()
        best = (math.inf, "none")
        for layer, (zb, zt) in zip(self.layers, bounds):
            fibre = zb if sagging else zt
            lever = (z_na - fibre) if sagging else (fibre - z_na)
            if lever <= 0.0:
                continue  # 该层整体位于受压侧，不受拉控
            if layer.flexural_strength is None:
                return 0.0, layer.name
            e_mod = layer.plane_stress_modulus
            if layer.flexural_strength <= 0.0:
                return 0.0, layer.name
            kappa = layer.flexural_strength / (e_mod * lever)
            if kappa < best[0]:
                best = (kappa, layer.name)
        if math.isinf(best[0]):
            raise ValueError("no tensile-governed fibre found (check stack geometry)")
        return best

    def first_crack_moment(self, *, sagging: bool = True) -> Tuple[float, float, str]:
        """首裂弯矩 M_cr (N·m/m)、首裂曲率 (1/m) 与控制层。"""
        kappa, layer_name = self.first_crack_curvature(sagging=sagging)
        return self.flexural_rigidity() * kappa, kappa, layer_name

    def apparent_flexural_strength(self, *, sagging: bool = True) -> float:
        """按同总厚度均质板折算的表观弯曲强度 σ_app (Pa)。

        ``σ_app = 6 M_cr / h_tot²``（平面应力模量 E' 已在 D 与 κ_cr 中自洽使用）。
        退化为纯冰板时 ``M_cr = σ_f h²/6``，故 ``σ_app ≡ σ_f,ice``（精确）。
        """
        m_cr, _, _ = self.first_crack_moment(sagging=sagging)
        h = self.total_thickness
        return 6.0 * m_cr / h**2

    def strength_reduction_factor(self, reference: "LayerStack") -> float:
        """折减系数 η = σ_app(self) / σ_app(reference)（无量纲）。"""
        return self.apparent_flexural_strength() / reference.apparent_flexural_strength()

    def stiffness_ratio(self, reference: "LayerStack") -> float:
        """刚度比 D(self) / D(reference)（无量纲）。"""
        return self.flexural_rigidity() / reference.flexural_rigidity()

    # -- 应力分布 -------------------------------------------------------
    def bending_stress(self, z: float, curvature: float, *, sagging: bool = True) -> float:
        """给定曲率下高度 z 处的弯曲正应力 (Pa)，受拉为正。"""
        z_na = self.neutral_axis()
        for layer, (zb, zt) in zip(self.layers, self.z_bounds()):
            if zb - 1e-15 <= z <= zt + 1e-15:
                lever = (z_na - z) if sagging else (z - z_na)
                return layer.plane_stress_modulus * curvature * lever
        raise ValueError(f"z={z} outside stack [0, {self.total_thickness}]")

    # -- 便捷构造 -------------------------------------------------------
    def substituting(self, name: str, **kwargs) -> "LayerStack":
        """按层名替换字段，返回新 LayerStack。"""
        new_layers = tuple(
            replace(layer, **kwargs) if layer.name == name else layer
            for layer in self.layers
        )
        return LayerStack(layers=new_layers)

    def total_weight_per_area(self) -> float:
        """单位面积自重 (N/m²)，缺密度时按 0 计。"""
        g = 9.81
        return sum(
            (layer.density or 0.0) * g * layer.thickness for layer in self.layers
        )


# ---------------------------------------------------------------------------
# 构型构造器
# ---------------------------------------------------------------------------

def default_layers(**thicknesses: float) -> LayerStack:
    """按 ``h_snow`` / ``h_snow_ice`` / ``h_ice`` 生成三层叠体（自下而上 ice→snow-ice→snow）。

    厚度为 0 的层自动丢弃。
    """
    order = (("ice", "h_ice"), ("snow-ice", "h_snow_ice"), ("snow", "h_snow"))
    allowed = {key for _, key in order}
    unknown = set(thicknesses) - allowed
    if unknown:
        raise ValueError(f"unknown thickness keys: {sorted(unknown)}")
    stack: List[IceLayer] = []
    for name, key in order:
        h = float(thicknesses.get(key, 0.0))
        if h < 0.0:
            raise ValueError(f"{key} must be >= 0")
        if h > 0.0:
            stack.append(replace(DEFAULT_MATERIALS[name], thickness=h))
    if not stack:
        raise ValueError("all layers are zero thickness")
    return LayerStack(layers=tuple(stack))


def snow_ice_composite(
    h_snow: float = 0.0, h_snow_ice: float = 0.0, h_ice: float = 0.5
) -> LayerStack:
    """三层复合叠体（雪叠在冰之上，总厚度 = 三层之和，即 "additive" 构型）。"""
    return default_layers(h_snow=h_snow, h_snow_ice=h_snow_ice, h_ice=h_ice)


def snow_on_ice_fixed_total(h_snow: float, h_total: float = 0.5) -> LayerStack:
    """**总厚度固定**构型：雪层占用总厚度的一部分，其余为冰。

    对应「雪覆盖后冰板总厚度不变、上部材料由冰换成雪（或无雪-冰）」的比较状态，
    用于量化雪层对同厚度冰板承载能力的折减。
    """
    if h_snow < 0.0:
        raise ValueError("h_snow must be >= 0")
    if h_total <= 0.0:
        raise ValueError("h_total must be > 0")
    if h_snow >= h_total:
        raise ValueError("h_snow must be smaller than h_total")
    return default_layers(h_snow=h_snow, h_ice=h_total - h_snow)


# ---------------------------------------------------------------------------
# Winkler 弹性地基板（简支矩形板，Navier 双三角级数）
# ---------------------------------------------------------------------------

def _navier_terms(n_terms: int) -> Iterable[Tuple[int, int]]:
    """生成 Navier 级数用的奇数模态对 (m, n)，每个方向保留 n_terms 个。"""
    ms = range(1, 2 * n_terms, 2)
    ns = range(1, 2 * n_terms, 2)
    for m in ms:
        for n in ns:
            yield m, n


def winkler_plate_center_deflection(
    flexural_rigidity: float,
    foundation_modulus: float,
    load: float,
    side: float,
    n_modes: int = 61,
) -> float:
    """简支矩形 Winkler 板在均布载荷下的中心挠度 (m)。

    方程 ``D ∇⁴w + k_f w = q0``，Navier 解

    ``w(x,y) = Σ_{m,n odd} (16 q0 / (π² m n)) sin(mπx/L) sin(nπy/L)
       / (D π⁴ (m²+n²)² / L⁴ + k_f)``。

    ``n_modes`` 为每个方向保留的奇数模态数（默认 61×61，中心挠度收敛到 1e-6 相对精度）。
    """
    if flexural_rigidity <= 0.0:
        raise ValueError("flexural_rigidity must be > 0")
    if foundation_modulus < 0.0:
        raise ValueError("foundation_modulus must be >= 0")
    if side <= 0.0:
        raise ValueError("side must be > 0")
    acc = 0.0
    for m, n in _navier_terms(n_modes):
        q_mn = 16.0 * load / (math.pi**2 * m * n)
        eig = flexural_rigidity * math.pi**4 * (m * m + n * n) ** 2 / side**4
        shape = ((-1.0) ** ((m - 1) // 2)) * ((-1.0) ** ((n - 1) // 2))
        acc += q_mn * shape / (eig + foundation_modulus)
    return acc


def winkler_plate_deflection_profile(
    flexural_rigidity: float,
    foundation_modulus: float,
    load: float,
    side: float,
    n_points: int = 81,
    n_modes: int = 41,
) -> Tuple[List[float], List[float]]:
    """沿 ``y = L/2`` 的挠度剖面，返回 (x 数组, w 数组)。"""
    xs = [side * i / (n_points - 1) for i in range(n_points)]
    ws: List[float] = []
    for x in xs:
        acc = 0.0
        for m, n in _navier_terms(n_modes):
            q_mn = 16.0 * load / (math.pi**2 * m * n)
            eig = flexural_rigidity * math.pi**4 * (m * m + n * n) ** 2 / side**4
            acc += (
                q_mn
                * math.sin(m * math.pi * x / side)
                * math.sin(n * math.pi * 0.5)
                / (eig + foundation_modulus)
            )
        ws.append(acc)
    return xs, ws


def winkler_plate_field(
    flexural_rigidity: float,
    foundation_modulus: float,
    load: float,
    side: float,
    n_points: int = 41,
    n_modes: int = 41,
) -> Tuple[List[float], List[float], List[List[float]]]:
    """简支 Winkler 板的全场挠度（Navier 级数），返回 (xs, ys, W[y][x])。"""
    xs = [side * i / (n_points - 1) for i in range(n_points)]
    terms = [
        (
            m,
            n,
            16.0 * load / (math.pi**2 * m * n)
            / (flexural_rigidity * math.pi**4 * (m * m + n * n) ** 2 / side**4 + foundation_modulus),
        )
        for m, n in _navier_terms(n_modes)
    ]
    W: List[List[float]] = []
    for y in xs:
        row = []
        for x in xs:
            acc = 0.0
            for m, n, amp in terms:
                acc += (
                    amp
                    * math.sin(m * math.pi * x / side)
                    * math.sin(n * math.pi * y / side)
                )
            row.append(acc)
        W.append(row)
    return xs, xs, W


# ---------------------------------------------------------------------------
# 与均质板 FEM 的耦合
# ---------------------------------------------------------------------------

def equivalent_homogeneous_modulus(
    flexural_rigidity: float, total_thickness: float, poisson: float
) -> float:
    """把复合刚度折算为同厚度均质板的等效杨氏模量 (Pa)。

    ``D = E h³ / (12 (1-ν²)) ⇒ E_eq = 12 (1-ν²) D / h³``。
    用于把 ``D_composite`` 注入既有的 Mindlin/Winkler 板有限元求解器
    （如 ``scenarios/suboff_icebreaking/si1_one_way/plate_response``）。
    """
    if total_thickness <= 0.0:
        raise ValueError("total_thickness must be > 0")
    return 12.0 * (1.0 - poisson**2) * flexural_rigidity / total_thickness**3