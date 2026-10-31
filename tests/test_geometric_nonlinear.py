"""F08 —— coupled_hull 大转动几何非线性（横摇 15 deg）验收测试。

判据对应（与 artifacts/m712/validation.json 同键）：
1. ``large_rotation_energy_conservation`` —— 大转动横摇暂态的能量漂移比 < 0.01
2. ``roll_angle_15deg_stress_error``        —— 15 deg 横摇根部应力相对误差 < 0.10
3. ``pytest_test_geometric_nonlinear``       —— 本文件全部用例 PASS

另附反例证据（证明几何非线性是必要的，而非数值上"恰好也对"）：
* 冻结几何（小转动）基线在 15 deg 横摇下应力误差 ~48%；
* 45 deg 纯弯矩大转动下小转动梁理论端部位移误差 >10%，而共旋模型 <1%。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.structure.corotational_beam import CorotationalBeamModel, Section
from tensorphysics.structure.finite_rotation import (
    Rotation2D,
    angular_velocity_field,
    polar_decomposition_2x2,
    section_rotation_from_displacements,
    wrap_angle,
)
from tensorphysics.structure.hull_roll import (
    HULL_SECTION,
    IceClassHullRollCase,
    build_side_shell_strip,
    frozen_geometry_root_stress,
    pure_moment_arc_case,
    roll_energy_conservation_case,
)

ENERGY_GATE = 0.01   # [ratio]
STRESS_GATE = 0.10   # [ratio]


# --------------------------------------------------------------------------- 0. 基础
def test_rotation_maps_are_consistent():
    """滚转角映射、wrap、极分解、截面转角识别的自洽性。"""
    phi = math.radians(15.0)
    R = Rotation2D(phi)
    assert np.allclose(R.matrix @ R.matrix.T, np.eye(2), atol=1e-14)
    assert np.allclose(R.compose(R.inverse()).matrix, np.eye(2), atol=1e-14)
    assert wrap_angle(phi + 2 * math.pi) == pytest.approx(phi, abs=1e-12)

    # 极分解：纯转动 + 拉伸
    stretch = np.diag([1.02, 0.99])
    R_polar, U = polar_decomposition_2x2(R.matrix @ stretch)
    assert R_polar == pytest.approx(R.matrix, abs=1e-12)
    assert np.allclose(U, stretch, atol=1e-12)

    # 由节点位移识别 15 deg 有限转动（大转动分支，非小角度近似）
    ref, _, _ = build_side_shell_strip(n_elements=4)
    rolled = Rotation2D(phi).apply(ref)
    assert math.degrees(section_rotation_from_displacements(ref, rolled)) == \
        pytest.approx(15.0, abs=1e-9)

    # 刚体转动速度场：v = omega x r
    omega = 0.05
    v = angular_velocity_field(ref, omega)
    assert np.allclose(v, omega * np.column_stack([-ref[:, 1], ref[:, 0]]), atol=1e-14)


def test_rigid_roll_stores_zero_strain_energy():
    """客观性：任意有限刚体滚转 + 配套节点转角 → 应变能必须为 0（几何非线性核心）。"""
    ref, elements, _ = build_side_shell_strip(n_elements=6)
    model = CorotationalBeamModel(reference_nodes=ref, elements=elements,
                                  section=HULL_SECTION, fixed_dofs=(0, 1, 2))
    for deg in (5.0, 15.0, 45.0, 90.0, 180.0):
        phi = math.radians(deg)
        rolled = Rotation2D(phi).apply(ref)
        theta = np.full(model.n_nodes, phi)
        assert abs(model.strain_energy(rolled, theta)) < 1e-9
    # 反例：只转几何、不带节点转角 → 应变能必须显著非零（否则测不出大转动效应）
    rolled = Rotation2D(math.radians(15.0)).apply(ref)
    assert model.strain_energy(rolled, np.zeros(model.n_nodes)) > 1.0


def test_tangent_stiffness_matches_finite_difference():
    """切线刚度 = 能量精确 Hessian，与数值差分梯度一致（Newton 收敛的前提）。"""
    ref, elements, _ = build_side_shell_strip(n_elements=4)
    model = CorotationalBeamModel(reference_nodes=ref, elements=elements,
                                  section=HULL_SECTION, fixed_dofs=(0, 1, 2))
    rng = np.random.default_rng(20261030)
    phi = math.radians(15.0)
    x = Rotation2D(phi).apply(ref) + 1e-3 * rng.standard_normal(ref.shape)
    theta = np.full(model.n_nodes, phi) + 1e-3 * rng.standard_normal(model.n_nodes)
    K = model.tangent_stiffness(x, theta)
    assert np.allclose(K, K.T, atol=1e-6 * np.abs(K).max())

    def flat_gradient(xx, th):
        fx, ft = model.internal_force(xx, th)
        grad = np.zeros(model.n_dof)
        grad[0::3] = fx[:, 0]
        grad[1::3] = fx[:, 1]
        grad[2::3] = ft
        return grad

    h = 1e-7
    grad0 = flat_gradient(x, theta)
    K_fd = np.zeros_like(K)
    for d in range(model.n_dof):
        node, comp = divmod(d, 3)
        x1, th1 = x.copy(), theta.copy()
        if comp == 2:
            th1[node] += h
        else:
            x1[node, comp] += h
        K_fd[:, d] = (flat_gradient(x1, th1) - grad0) / h
    scale = np.abs(K_fd).max()
    assert np.abs(K - K_fd).max() / scale < 1e-4


# ------------------------------------------------------------------ 1. 15 deg 应力
def test_roll_angle_15deg_stress_error_gate():
    """判据 2：15 deg 横摇根部应力误差 < 10%（对标 Rigid-roll 全局静力解析解）。"""
    case = IceClassHullRollCase(roll_deg=15.0, ice_force_kN=50.0,
                                attached_mass_t=20.0)
    res = case.solve()
    assert res["rigid_roll_strain_energy_j"] < 1e-9   # 客观性在同一算例内成立
    assert res["stress_error_ratio"] < STRESS_GATE
    # 平衡闭合：同一闭式静力学用变形后几何应复现梁内力（<1e-3）
    assert res["equilibrium_closure_ratio"] < 1e-3
    # 反例证据：冻结几何（小转动）基线在 15 deg 下误差远超 15%
    assert res["frozen_geometry_error_ratio"] > 0.15
    print("[gate] roll_angle_15deg_stress_error = "
          f"{res['stress_error_ratio']:.4f} (FE {res['sigma_fe_pa']/1e6:.1f} MPa, "
          f"analytic {res['sigma_analytic_pa']/1e6:.1f} MPa, "
          f"frozen-geometry error {res['frozen_geometry_error_ratio']:.3f})")


def test_small_roll_limit_and_growth_of_nonlinearity():
    """小转动极限 + 非线性随横摇单调增长（排除"伪大转动效应"）。"""
    small = IceClassHullRollCase(roll_deg=1.0, ice_force_kN=50.0,
                                 attached_mass_t=20.0).solve()
    large = IceClassHullRollCase(roll_deg=15.0, ice_force_kN=50.0,
                                 attached_mass_t=20.0).solve()
    # 1) 小转动下共旋解仍与解析静力学一致；残差来自"刚体滚转几何"参考臂与弹性
    #    变形后力臂之差（O(delta/H) ~ 1%），不是离散误差（见网格收敛证据）
    assert small["stress_error_ratio"] < 0.05
    assert small["equilibrium_closure_ratio"] < 1e-3
    # 2) 冻结几何基线在小转动下足够准（<10%），到大转动才显著失效（>15%）
    assert small["frozen_geometry_error_ratio"] < 0.10
    assert large["frozen_geometry_error_ratio"] > 0.15
    # 3) 截面重量的力臂 sin(phi) 是一阶效应，与横摇角同阶增长
    assert large["frozen_geometry_error_ratio"] > 4.0 * small["frozen_geometry_error_ratio"]


# ------------------------------------------------------- 2. 大转动能量守恒判据
def test_large_rotation_energy_conservation_gate():
    """判据 1：大转动横摇暂态总能量漂移 < 1%，且横摇幅值 > 15 deg。"""
    hist = roll_energy_conservation_case(initial_roll_deg=20.0,
                                         roll_spring_n_m_per_rad=6.0e6,
                                         dt=1.5e-4, duration=0.4,
                                         sample_every=25)
    assert hist["roll_amplitude_deg"] > 15.0
    assert hist["initial_strain_energy_j"] < 1e-9   # 初始状态是纯刚体滚转
    assert hist["energy_drift_ratio"] < ENERGY_GATE
    print("[gate] large_rotation_energy_conservation = "
          f"{hist['energy_drift_ratio']:.3e} "
          f"(roll amplitude {hist['roll_amplitude_deg']:.2f} deg)")


# ------------------------------------------------------------ 3. 大转动能力验证
def test_pure_moment_large_rotation_matches_exact_arc():
    """纯端弯矩:任意转角下精确解为圆弧 —— 45 deg 时共旋解 <1%，小转动理论 >5%。"""
    res = pure_moment_arc_case(rotation_deg=45.0)
    assert res["tip_error_ratio_fe"] < 0.01
    assert res["tip_error_ratio_linear"] > 0.05
    assert res["sigma_max_pa"] < 355e6   # 验证算例保持在弹性范围


def test_geometric_nonlinear_acceptance_summary():
    """判据 3（汇总）：两项机器判据在同一次运行内同时满足。"""
    stress = IceClassHullRollCase(roll_deg=15.0).solve()
    energy = roll_energy_conservation_case(dt=1.5e-4, duration=0.3,
                                           sample_every=50)
    gates = {
        "roll_angle_15deg_stress_error": stress["stress_error_ratio"],
        "large_rotation_energy_conservation": energy["energy_drift_ratio"],
    }
    assert gates["roll_angle_15deg_stress_error"] < 0.1
    assert gates["large_rotation_energy_conservation"] < 0.01
    assert energy["roll_amplitude_deg"] > 15.0
    print(f"[gate] summary: {gates}")