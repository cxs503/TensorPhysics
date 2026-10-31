"""雪-冰复合层弯曲力学模型验收测试 (pytest_test_snow_ice_composite)。

对应战役节点 W2-N03 的机器可算判据：

(a) ``neutral_axis_equal_stiffness`` : 三层等刚度特例 z_na = h_total/2，误差 < 1e-12 m
(b) ``stiffness_reduction_physical``: 雪层 E_s ≪ E_ice 时 D_composite/D_ice < 1
(d) ``zero_snow_recovery``          : h_snow = 0 时 D_composite = D_ice，误差 < 1e-12
(e) ``deflection_increase_with_snow``: 含雪 Winkler 板中心挠度 > 无雪挠度
(f) 本文件整体 PASS（``pytest_test_snow_ice_composite == PASS``）

另外包含实现正确性的独立复核（数值积分 / Timoshenko 板基准 / 等效模量往返 / 纯冰首裂弯矩解析值）。
"""

from __future__ import annotations

import math

import pytest

from tensorphysics.composite_mechanics import (
    DEFAULT_MATERIALS,
    IceLayer,
    LayerStack,
    default_layers,
    equivalent_homogeneous_modulus,
    snow_on_ice_fixed_total,
    winkler_plate_center_deflection,
    winkler_plate_deflection_profile,
)
from tensorphysics.composite_mechanics.snow_ice_composite import (
    GRANULAR_SNOW_ICE_BENDING_TEST,
    SALORANTA_2000,
)

# 文献实验锚点：Edington, Murdza & Schulson (2025), J. Glaciol. 71, e90,
# doi:10.1017/jog.2025.10077 —— 粒状冰层 ≈ 总厚 6%，−10 ℃，外层纤维应变率 1e-7 s^-1。
EDINGTON_2025 = {
    "layer_thickness_fraction": 1.0 / 17.0,
    "ductile_strain_rate": 1.0e-7,
    "brittle_strain_rate": 1.0e-4,
    "freshwater_columnar_MPa": 1.67,
    "freshwater_composite_MPa": 2.29,
    "saline_columnar_MPa": 0.96,
    "saline_composite_MPa": 1.13,
    "sea_ice_columnar_MPa": 1.40,
    "sea_ice_composite_MPa": 1.63,
}


# ---------------------------------------------------------------------------
# 判据 (a): 三层等刚度特例 → 中性轴位于总厚中面
# ---------------------------------------------------------------------------

def test_neutral_axis_equal_stiffness():
    """等刚度（等 E、等 ν）三层叠体的弹性加权中性轴 = h_total/2，误差 < 1e-12 m。"""
    stacks = [
        # 等厚
        (0.10, 0.10, 0.10),
        # 不等厚（连续的整块板，形心必在中面）
        (0.05, 0.30, 0.15),
        (0.31, 0.02, 0.47),
    ]
    for thicknesses in stacks:
        layers = tuple(
            IceLayer(name=f"L{i}", thickness=t, young=5.0e9, poisson=0.3)
            for i, t in enumerate(thicknesses)
        )
        stack = LayerStack(layers=layers)
        z_na = stack.neutral_axis()
        assert abs(z_na - 0.5 * stack.total_thickness) < 1e-12
        # 不启用平面应力修正时结论不变
        assert abs(stack.neutral_axis(plane_stress=False) - 0.5 * stack.total_thickness) < 1e-12


def test_neutral_axis_shifts_toward_stiffer_layer():
    """模量比 2:1 的两层叠体中性轴偏向高模量层（定性检查）。"""
    stack = LayerStack(
        layers=(
            IceLayer("soft", 0.10, 1.0e9, 0.3),
            IceLayer("stiff", 0.10, 9.0e9, 0.3),
        )
    )
    assert stack.neutral_axis() > 0.10  # 偏向位于上方的 stiff 层


# ---------------------------------------------------------------------------
# 判据 (b): 雪层 E_s ≪ E_ice 时刚度折减（同总厚度替代构型）
# ---------------------------------------------------------------------------

def test_stiffness_reduction_physical():
    """总厚度固定、顶层由雪替代时 D_composite/D_ice < 1，且随雪厚单调下降。"""
    h_total = 0.45
    ice = default_layers(h_ice=h_total)
    ratios = []
    for h_snow in (0.02, 0.05, 0.10, 0.15, 0.20):
        comp = snow_on_ice_fixed_total(h_snow, h_total)
        ratio = comp.stiffness_ratio(ice)
        assert ratio < 1.0
        ratios.append(ratio)
    assert ratios == sorted(ratios, reverse=True)  # 单调
    # 极限检查：E_snow -> 0 时顶层贡献消失，D 只剩冰层部分
    zero_stiff = snow_on_ice_fixed_total(0.10, h_total).substituting(
        "snow", young=1e-6
    )
    assert zero_stiff.stiffness_ratio(ice) < 0.6
    # 物理一致性：雪层模量越低，折减越强
    soft = snow_on_ice_fixed_total(0.10, h_total).substituting("snow", young=1.0e6)
    default = snow_on_ice_fixed_total(0.10, h_total)
    assert soft.stiffness_ratio(ice) < default.stiffness_ratio(ice)


def test_additive_stack_stiffness_never_below_ice():
    """叠加上构型（雪真正叠在冰上，总厚度增大）应满足平行轴定理：D ≥ D_ice。"""
    h_ice = 0.45
    ice = default_layers(h_ice=h_ice)
    for h_snow in (0.05, 0.10, 0.20):
        additive = default_layers(h_snow=h_snow, h_ice=h_ice)
        assert additive.stiffness_ratio(ice) >= 1.0 - 1e-15


# ---------------------------------------------------------------------------
# 判据 (d): h_snow = 0 退化为纯冰板
# ---------------------------------------------------------------------------

def test_zero_snow_recovery():
    """h_snow = 0 时 D_composite 与纯冰板刚度一致（相对误差 < 1e-12），η = 1。"""
    h_ice = 0.45
    ice = default_layers(h_ice=h_ice)
    recovered = default_layers(h_snow=0.0, h_ice=h_ice)
    rel = abs(recovered.flexural_rigidity() - ice.flexural_rigidity()) / ice.flexural_rigidity()
    assert rel < 1e-12

    analytic = (
        DEFAULT_MATERIALS["ice"].plane_stress_modulus * h_ice**3 / 12.0
    )
    assert abs(ice.flexural_rigidity() - analytic) / analytic < 1e-12
    assert abs(ice.strength_reduction_factor(ice) - 1.0) < 1e-12
    assert abs(ice.apparent_flexural_strength() - DEFAULT_MATERIALS["ice"].flexural_strength) / (
        DEFAULT_MATERIALS["ice"].flexural_strength
    ) < 1e-12


# ---------------------------------------------------------------------------
# 判据 (e): 含雪 Winkler 板中心挠度大于无雪
# ---------------------------------------------------------------------------

def test_deflection_increase_with_snow():
    """同一载荷/地基下，含雪板（刚度折减）中心挠度大于纯冰板。"""
    h_total = 0.45
    side = 10.0
    k_f = 5.0e6
    q = 2.0e3
    ice = default_layers(h_ice=h_total)
    comp = snow_on_ice_fixed_total(0.10, h_total)

    w_ice = winkler_plate_center_deflection(ice.flexural_rigidity(), k_f, q, side)
    w_comp = winkler_plate_center_deflection(comp.flexural_rigidity(), k_f, q, side)
    assert w_comp > w_ice

    # 无地基 (k_f = 0) 时挠度与 D 成反比，可精确校验刚度的传导
    w_ice_free = winkler_plate_center_deflection(ice.flexural_rigidity(), 0.0, q, side)
    w_comp_free = winkler_plate_center_deflection(comp.flexural_rigidity(), 0.0, q, side)
    assert w_comp_free / w_ice_free == pytest.approx(
        1.0 / comp.stiffness_ratio(ice), rel=1e-9
    )

    # 纯冰板基准：Timohsenko 无地基矩形板中心挠度 0.00406 q L^4 / D（误差 < 0.5%）
    timoshenko = 0.00406 * q * side**4 / ice.flexural_rigidity()
    assert abs(winkler_plate_center_deflection(ice.flexural_rigidity(), 0.0, q, side) / timoshenko - 1.0) < 5e-3


def test_snow_self_weight_also_increases_deflection():
    """叠加上构型：刚度略增，但雪自重使总载荷增大，挠度仍然增大（物理正确）。"""
    h_ice = 0.45
    ice = default_layers(h_ice=h_ice)
    snow_ice = default_layers(h_snow=0.10, h_ice=h_ice)
    q_ice = ice.total_weight_per_area()
    q_snow = snow_ice.total_weight_per_area()
    assert q_snow > q_ice
    w_ice = winkler_plate_center_deflection(ice.flexural_rigidity(), 5.0e6, q_ice, 10.0)
    w_snow = winkler_plate_center_deflection(snow_ice.flexural_rigidity(), 5.0e6, q_snow, 10.0)
    assert w_snow > w_ice


# ---------------------------------------------------------------------------
# 实现正确性的独立复核
# ---------------------------------------------------------------------------

def _numeric_rigidity(stack: LayerStack, n_slices_per_m: int = 400_000) -> float:
    """独立的数值积分实现：把每层切成薄片，用「片内精确二阶矩」求和。

    对第 k 片 ∫(z - z_na)^2 dz = dz * [(z_c - z_na)^2 + dz^2 / 12]，
    该式与解析式 Σ[t^3/12 + t d^2] 的差别仅来自浮点累加，故是一路真正独立的
    （逐片累加）实现，可用于检出解析式里的代数错误。
    """
    bounds = stack.z_bounds()
    z_na = stack.neutral_axis()
    total = 0.0
    for layer, (zb, zt) in zip(stack.layers, bounds):
        e_mod = layer.plane_stress_modulus
        n_i = max(1, int(round(layer.thickness * n_slices_per_m)))
        dz = layer.thickness / n_i
        for k in range(n_i):
            z_c = zb + (k + 0.5) * dz
            total += e_mod * dz * ((z_c - z_na) ** 2 + dz * dz / 12.0)
    return total


@pytest.mark.parametrize("h_snow,h_snow_ice", [(0.10, 0.05), (0.05, 0.0), (0.0, 0.10)])
def test_rigidity_matches_independent_quadrature(h_snow, h_snow_ice):
    """D_composite 与独立数值积分（20 万片）一致到 1e-6 相对误差。"""
    stack = default_layers(h_snow=h_snow, h_snow_ice=h_snow_ice, h_ice=0.40)
    analytic = stack.flexural_rigidity()
    numeric = _numeric_rigidity(stack)
    assert abs(analytic - numeric) / analytic < 1e-10


def test_equivalent_modulus_roundtrip():
    """D → E_eq → D 往返精确（同厚度均质板折算）。"""
    stack = default_layers(h_snow=0.10, h_snow_ice=0.05, h_ice=0.40)
    h = stack.total_thickness
    nu = 0.33
    d = stack.flexural_rigidity()
    e_eq = equivalent_homogeneous_modulus(d, h, nu)
    d_back = e_eq * h**3 / (12.0 * (1.0 - nu**2))
    assert abs(d_back - d) / d < 1e-12


def test_first_crack_moment_of_pure_ice_is_analytic():
    """纯冰板首裂弯矩 = σ_f h² / 6（平面应力模量自洽）。"""
    h = 0.45
    ice = default_layers(h_ice=h)
    sigma_f = DEFAULT_MATERIALS["ice"].flexural_strength
    m_cr, _, control = ice.first_crack_moment()
    assert control == "ice"
    assert abs(m_cr - sigma_f * h**2 / 6.0) / (sigma_f * h**2 / 6.0) < 1e-12


def test_saloranta_configuration_case():
    """Saloranta (2000) Baltic 层序参考构型：含雪-冰层时 D 与 η 均小于同厚度纯冰。"""
    h_total = SALORANTA_2000["h_snow"] + SALORANTA_2000["h_snow_ice"] + SALORANTA_2000["h_ice"]
    stack = default_layers(
        h_snow=SALORANTA_2000["h_snow"],
        h_snow_ice=SALORANTA_2000["h_snow_ice"],
        h_ice=SALORANTA_2000["h_ice"],
    )
    reference = default_layers(h_ice=h_total)  # 同总厚度纯冰板（替代构型）
    assert stack.total_thickness == pytest.approx(h_total)
    assert stack.stiffness_ratio(reference) < 1.0
    assert stack.strength_reduction_factor(reference) < 1.0
    # 中性轴被拉向高模量的冰层（下移，低于几何中面）
    assert stack.neutral_axis() < 0.5 * h_total


def test_strength_reference_ductile_regime_within_15pct():
    """与真实复合梁三点弯曲实验（Edington et al. 2025, 延性区）比较，误差 < 15%。

    实验：外层纤维应变率 1e-7 s^-1（延性），6% 厚粒状冰层对柱状冰弯曲强度
    "无可检测影响"（强度比 1.00）。模型：粒状层与柱状层模量接近（淡水冰），
    复合表观强度与纯柱状冰之差即模型误差。
    """
    f = EDINGTON_2025["layer_thickness_fraction"]
    h = 0.05
    # 淡水冰：粒状层 E ≈ 8.5 GPa、柱状层 E ≈ 9.5 GPa（论文只给强度，模量取文献量级）；
    # 强度取论文实测值：柱状 1.67 MPa，粒状层取同温文献量级 2.0 MPa。
    pure = LayerStack(layers=(IceLayer("columnar", h, 9.5e9, 0.33, 1.67e6),))
    layers = (
        IceLayer("columnar", h * (1.0 - f), 9.5e9, 0.33, 1.67e6),
        IceLayer("granular", h * f, 8.5e9, 0.33, 2.0e6),
    )
    stack = LayerStack(layers=layers)
    measured_ductile_ratio = 1.0  # 论文："no detectable effect"
    # 粒状层在上（受压侧）：柱状冰下缘控制首裂
    eta_sagging = stack.strength_reduction_factor(pure)
    # 粒状层在上（该层受拉，对应 coating 位于受拉面的试验构型）
    eta_top_tension = stack.apparent_flexural_strength(sagging=False) / pure.apparent_flexural_strength(
        sagging=False
    )
    for eta in (eta_sagging, eta_top_tension):
        assert abs(eta - measured_ductile_ratio) / measured_ductile_ratio < 0.15


def test_granular_vs_columnar_measured_strength_ratio_documented():
    """实验锚点数值自检（Water 2024, doi:10.3390/w16233358）：2.85 / 1.93 MPa。"""
    ratio = (
        GRANULAR_SNOW_ICE_BENDING_TEST["sigma_granular_snow_ice_Pa"]
        / GRANULAR_SNOW_ICE_BENDING_TEST["sigma_columnar_ice_Pa"]
    )
    assert ratio == pytest.approx(2.85 / 1.93)
    assert ratio > 1.0


def test_deflection_profile_shape():
    """挠度剖面：简支边界两端为零、中心最大。"""
    ice = default_layers(h_ice=0.45)
    xs, ws = winkler_plate_deflection_profile(ice.flexural_rigidity(), 5.0e6, 2.0e3, 10.0)
    assert abs(ws[0]) < 1e-12
    assert abs(ws[-1]) < 1e-12
    assert max(ws) == pytest.approx(ws[len(ws) // 2], rel=1e-3)


def test_invalid_inputs_raise():
    """输入校验：负厚度/非法泊松比/零厚度叠体应显式报错。"""
    with pytest.raises(ValueError):
        IceLayer("bad", -0.1, 1e9, 0.3)
    with pytest.raises(ValueError):
        IceLayer("bad", 0.1, 1e9, 0.7)
    with pytest.raises(ValueError):
        default_layers(h_snow=0.0, h_snow_ice=0.0, h_ice=0.0)
    with pytest.raises(ValueError):
        snow_on_ice_fixed_total(h_snow=0.5, h_total=0.4)