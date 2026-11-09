# -*- coding: utf-8 -*-
"""层合复合材料板 CLT/FSDT 验证面的 pytest 用例。

覆盖三类裁决面与负控：

* 解析锚：各向同性板挠度系数 ``0.00406``、单轴屈曲 ``k = 4``、双轴 ``k = 2``、
  纯剪屈曲 ``k_s = 9.34``（方板）/ ``5.34``（长板）；
* 三方一致性：Navier 双三角级数 vs Ritz（同基）vs Levy 单三角级数（SS 分支逐位一致，
  CC 分支 vs 固支梁函数独立能量解）；
* 负控/退化：各向同性单层 -> 经典板；``h/a -> 0`` 时 FSDT -> CLT（``O((h/a)^2)``）；
  对称正交各向铺层 ``D16 = D26 = 0``；以及任务书两处物理表述的纠正
  （对称角铺层 ``D16 != 0``、``[0/90]`` 非对称只有 ``B != 0``）；
* 矩阵级自检：弯曲刚度阵 / 纯剪几何刚度阵与高阶 Gauss 数值积分逐位一致
  （后者是纯剪屈曲实现的回归保护：曾因 y 向积分表索引错位使 ``k_s`` 偏低 20%）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.composite_mechanics.laminate_plate import (
    Lamina,
    Laminate,
    _geo_shear,
    _ritz_bending_matrix,
    buckling_biaxial,
    buckling_eigen_shear,
    buckling_eigen_uniaxial,
    buckling_uniaxial,
    fsdt_navier_deflection,
    isotropic_plate_rigidity,
    levy_deflection,
    navier_clt_deflection,
    parse_layup,
    ritz_clamped_x_deflection,
    ritz_deflection,
    validate_plate_geometry,
)

E_STEEL = 200.0e9
NU_STEEL = 0.30
A = B = 1.0


def steel_laminate(h: float = 0.01) -> Laminate:
    return Laminate([(0.0, h)], Lamina.isotropic(E_STEEL, NU_STEEL), name="steel")


def t300() -> Lamina:
    return Lamina(E1=181.0e9, E2=10.3e9, G12=7.17e9, nu12=0.28)


# ---------------------------------------------------------------------------
# 1) 解析锚：各向同性退化
# ---------------------------------------------------------------------------
def test_isotropic_single_ply_is_classical_plate():
    """各向同性单层 -> 任意铺层角 Qbar 相同，D16 = D26 = 0，D = Eh^3/12(1-nu^2)。"""
    mat = Lamina.isotropic(E_STEEL, NU_STEEL)
    lam = Laminate([(0.0, 0.01)], mat)
    D_ref = isotropic_plate_rigidity(E_STEEL, NU_STEEL, 0.01)
    assert lam.D11 == pytest.approx(D_ref, rel=1e-13)
    assert lam.D22 == pytest.approx(D_ref, rel=1e-13)
    assert lam.D[0, 1] == pytest.approx(NU_STEEL * D_ref, rel=1e-12)
    assert lam.D[2, 2] == pytest.approx((1.0 - NU_STEEL) / 2.0 * D_ref, rel=1e-12)
    assert abs(lam.D16) < 1e-6 * D_ref
    assert abs(lam.D26) < 1e-6 * D_ref
    # 铺层角无关性
    for theta in (0.0, 17.0, 45.0, 90.0):
        Q = mat.qbar(theta)
        assert Q[0, 2] == pytest.approx(0.0, abs=1e-6 * Q[0, 0])
        assert Q[1, 2] == pytest.approx(0.0, abs=1e-6 * Q[0, 0])


def test_isotropic_deflection_coefficient_00406():
    """四边简支方板均布载荷中心挠度系数 = 0.00406（Timoshenko）。"""
    lam = steel_laminate(0.01)
    D = isotropic_plate_rigidity(E_STEEL, NU_STEEL, 0.01)
    res = navier_clt_deflection(lam, A, B, q0=1000.0, m_max=51, n_max=51)
    coef = res.w_max * D / (1000.0 * A**4)
    assert coef == pytest.approx(0.0040624, rel=1e-3)


def test_isotropic_uniaxial_buckling_k4():
    lam = steel_laminate(0.01)
    D = isotropic_plate_rigidity(E_STEEL, NU_STEEL, 0.01)
    res = buckling_uniaxial(lam, A, B)
    k = res.n_cr * B**2 / (math.pi**2 * D)
    assert k == pytest.approx(4.0, rel=1e-10)
    assert (res.m, res.n) == (1, 1)


def test_isotropic_biaxial_buckling_k2():
    """双轴等压：``N_cr = 2 pi^2 D / b^2``（闭式 min(m=n=1)）。"""
    lam = steel_laminate(0.01)
    D = isotropic_plate_rigidity(E_STEEL, NU_STEEL, 0.01)
    res = buckling_biaxial(lam, A, B)
    assert res.n_cr * B**2 / (math.pi**2 * D) == pytest.approx(2.0, rel=1e-10)


def test_isotropic_shear_buckling_textbook():
    """纯剪屈曲系数：方板 ``k_s ~ 9.34``，长板 -> ``5.34``（Timoshenko）。"""
    lam = steel_laminate(0.01)
    D = isotropic_plate_rigidity(E_STEEL, NU_STEEL, 0.01)
    square = buckling_eigen_shear(lam, A, B, 12, 12)
    ks_square = square.n_cr * B**2 / (math.pi**2 * D)
    assert ks_square == pytest.approx(9.34, rel=2e-2)
    long_plate = buckling_eigen_shear(lam, 8.0 * B, B, 40, 10)
    ks_long = long_plate.n_cr * B**2 / (math.pi**2 * D)
    assert 5.30 < ks_long < 5.60


# ---------------------------------------------------------------------------
# 2) 三方一致性：Navier / Ritz / Levy
# ---------------------------------------------------------------------------
def test_navier_matches_ritz_for_crossply():
    """正交各向对称铺层：Ritz 刚度阵对角 -> 与 Navier 逐位一致（同截断）。"""
    lam = Laminate.from_spec("[0/90]2s", 0.125e-3, t300())
    assert lam.is_orthotropic()
    nav = navier_clt_deflection(lam, A, B, 1.0e4, m_max=5, n_max=5)
    rit = ritz_deflection(lam, A, B, 1.0e4, m_terms=6, n_terms=6)
    assert rit.w_max == pytest.approx(nav.w_max, rel=1e-12)


def test_levy_ss_matches_navier():
    lam = Laminate.from_spec("[0/90]2s", 0.125e-3, t300())
    nav = navier_clt_deflection(lam, A, B, 1.0e4, m_max=99, n_max=99)
    lev = levy_deflection(lam, A, B, 1.0e4, x_edges="SS", n_terms=61)
    assert lev.w_max == pytest.approx(nav.w_max, rel=1e-8)


def test_levy_cc_matches_independent_clamped_ritz():
    """Levy(CC) 与「固支梁函数 + Gauss 积分」独立能量解一致（<0.5%）。"""
    lam = steel_laminate(0.01)
    lev = levy_deflection(lam, A, B, 1000.0, x_edges="CC", n_terms=61)
    rit = ritz_clamped_x_deflection(lam, A, B, 1000.0, m_terms=6, n_terms=14)
    assert rit.w_max == pytest.approx(lev.w_max, rel=5e-3)
    # 固支比简支刚度大：挠度更小
    ss = levy_deflection(lam, A, B, 1000.0, x_edges="SS", n_terms=61)
    assert lev.w_max < ss.w_max


def test_eigen_uniaxial_matches_closed_form_for_orthotropic():
    """一般各向异性特征值解在 D16=D26=0 时必须退化为闭式解。"""
    lam = Laminate.from_spec("[0/90]2s", 0.125e-3, t300())
    closed = buckling_uniaxial(lam, A, B)
    eigen = buckling_eigen_uniaxial(lam, A, B, 8, 8)
    assert eigen.n_cr == pytest.approx(closed.n_cr, rel=1e-8)


# ---------------------------------------------------------------------------
# 3) 矩阵级自检（数值积分交叉验证）
# ---------------------------------------------------------------------------
def _quadratic_form_check(lam: Laminate, M: int, N: int, which: str) -> float:
    """用 120 点 Gauss 积分直接算二次型，与解析矩阵值比较，返回相对误差。"""
    b = B
    rng = np.random.default_rng(12345)
    W = rng.normal(size=M * N)
    gx, wx = np.polynomial.legendre.leggauss(120)
    xg = 0.5 * b * (gx + 1.0)
    wg = 0.5 * b * wx
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    Wg = np.meshgrid(wg, wg, indexing="ij")
    al = np.arange(1, M + 1) * math.pi / b
    be = np.arange(1, N + 1) * math.pi / b

    def grad(kind: str) -> np.ndarray:
        v = np.zeros_like(X)
        for m in range(1, M + 1):
            for n in range(1, N + 1):
                c = W[(m - 1) * N + (n - 1)]
                if kind == "x":
                    f = al[m - 1] * np.cos(al[m - 1] * X) * np.sin(be[n - 1] * Y)
                elif kind == "y":
                    f = be[n - 1] * np.sin(al[m - 1] * X) * np.cos(be[n - 1] * Y)
                elif kind == "xx":
                    f = -(al[m - 1] ** 2) * np.sin(al[m - 1] * X) * np.sin(be[n - 1] * Y)
                else:  # yy
                    f = -(be[n - 1] ** 2) * np.sin(al[m - 1] * X) * np.sin(be[n - 1] * Y)
                v += c * f
        return v

    if which == "shear":
        # G 是 V = int w,x w,y dA 的 Hessian -> W^T G W = 2 V
        G = _geo_shear(lam, b, b, M, N)
        num = W @ G @ W
        ref = 2.0 * float(np.sum(grad("x") * grad("y") * Wg[0] * Wg[1]))
    else:
        # U = 1/2 int dens dA = 1/2 W^T K W  ->  W^T K W = int dens dA
        K = _ritz_bending_matrix(lam, b, b, M, N)
        num = W @ K @ W
        wxx, wyy = grad("xx"), grad("yy")
        wxy = np.zeros_like(X)
        for m in range(1, M + 1):
            for n in range(1, N + 1):
                c = W[(m - 1) * N + (n - 1)]
                wxy += c * al[m - 1] * be[n - 1] * np.cos(al[m - 1] * X) * np.cos(be[n - 1] * Y)
        dens = (
            lam.D11 * wxx**2
            + lam.D22 * wyy**2
            + 2.0 * lam.D12 * wxx * wyy
            + 4.0 * lam.D66 * wxy**2
            + 4.0 * lam.D16 * wxx * wxy
            + 4.0 * lam.D26 * wyy * wxy
        )
        ref = float(np.sum(dens * Wg[0] * Wg[1]))
    return (num - ref) / ref


def test_bending_matrix_matches_quadrature():
    lam = Laminate.from_spec("[±45]s", 0.125e-3, t300())  # 含 D16, D26
    assert abs(_quadratic_form_check(lam, 3, 3, "bending")) < 1e-9


def test_shear_geometric_matrix_matches_quadrature():
    """回归保护：纯剪几何刚度阵曾因 y 向积分表索引错位，使 k_s 偏低约 20%。"""
    lam = Laminate.from_spec("[±45]s", 0.125e-3, t300())
    assert abs(_quadratic_form_check(lam, 4, 4, "shear")) < 1e-9


# ---------------------------------------------------------------------------
# 4) 负控与物理纠正
# ---------------------------------------------------------------------------
def test_symmetric_crossply_has_zero_coupling():
    """对称 cross-ply：B = 0 且 D16 = D26 = 0。"""
    lam = Laminate.from_spec("[0/90]2s", 0.125e-3, t300())
    assert lam.is_symmetric()
    assert lam.is_orthotropic()
    assert abs(lam.D16) <= 1e-12 * lam.D11
    assert np.max(np.abs(lam.B)) <= 1e-12 * np.max(np.abs(lam.A)) * lam.thickness


def test_antisymmetric_crossply_has_B_but_no_D16():
    """``[0/90]`` 非对称：B11, B22 != 0，但 D16 = D26 = 0（纠正任务书表述）。"""
    lam = Laminate.from_spec("[0/90]", 0.125e-3, t300())
    assert not lam.is_symmetric()
    assert lam.is_orthotropic()
    assert abs(lam.B[0, 0]) > 1e-3 * lam.thickness * lam.A[0, 0]
    assert abs(lam.D16) <= 1e-12 * lam.D11


def test_symmetric_angleply_has_D16_but_zero_B():
    """对称 ``[±45]s``：B = 0（对称）但 D16 != 0（弯曲-扭转耦合）。"""
    lam = Laminate.from_spec("[±45]s", 0.125e-3, t300())
    assert lam.is_symmetric()
    assert not lam.is_orthotropic()
    assert abs(lam.D16) / lam.D11 > 0.4
    assert abs(lam.D26) / lam.D22 > 0.4


def test_D16_sign_flips_with_ply_sign():
    """``D16`` 符号随铺层角符号翻转（反对称铺层弯-扭耦合的符号验证）。"""
    plus = Laminate.from_spec("[45/-45]s", 0.125e-3, t300())
    minus = Laminate.from_spec("[-45/45]s", 0.125e-3, t300())
    assert plus.D16 > 0.0
    assert minus.D16 < 0.0
    assert plus.D16 == pytest.approx(-minus.D16, rel=1e-12)
    # 挠度（能量）对符号不敏感：耦合项只以平方形式进入能量
    w_plus = ritz_deflection(plus, A, B, 1.0e4, 8, 8).w_max
    w_minus = ritz_deflection(minus, A, B, 1.0e4, 8, 8).w_max
    assert w_plus == pytest.approx(w_minus, rel=1e-10)


def test_coupling_increases_deflection():
    """弯-扭耦合使方板中心挠度增大（同 D11/D22/D12/D66 的解耦板为参照）。"""
    lam = Laminate.from_spec("[±45]s", 0.125e-3, t300())
    decoupled = Laminate.from_spec("[±45]s", 0.125e-3, t300())
    D0 = decoupled.D.copy()
    D0[0, 2] = D0[1, 2] = D0[2, 0] = D0[2, 1] = 0.0
    decoupled.D = D0
    w_c = ritz_deflection(lam, A, B, 1.0e4, 14, 14).w_max
    w_d = ritz_deflection(decoupled, A, B, 1.0e4, 14, 14).w_max
    assert w_c > w_d
    assert (w_c - w_d) / w_d > 0.05


# ---------------------------------------------------------------------------
# 5) FSDT -> CLT 收敛
# ---------------------------------------------------------------------------
def _fsdt_clt_diff(h_over_a: float) -> float:
    lam = Laminate.from_spec("[0/90]2s", h_over_a * A / 8.0, t300())
    w_clt = navier_clt_deflection(lam, A, B, 1.0e4).w_max
    w_fsdt = fsdt_navier_deflection(lam, A, B, 1.0e4).w_max
    return (w_fsdt - w_clt) / w_clt


def test_fsdt_converges_to_clt_second_order():
    """``h/a -> 0`` 时 FSDT 相对剪切修正按 ``O((h/a)^2)`` 收敛。"""
    d_small = _fsdt_clt_diff(0.002)
    d_1 = _fsdt_clt_diff(0.05)
    d_2 = _fsdt_clt_diff(0.10)
    assert abs(d_small) < 1e-3
    assert d_1 > 0.0 and d_2 > 0.0
    # h/a 加倍 -> 修正量约 4 倍（二阶；允许 3.2~4.8 的带宽）
    assert 3.2 < d_2 / d_1 < 4.8


def test_fsdt_shear_correction_material_at_h_over_a_0_1():
    """``h/a = 1/10`` 时剪切修正已不可忽略（>10%），CLT 在此边界失效。"""
    assert _fsdt_clt_diff(0.1) > 0.10


def test_fsdt_requires_orthotropic_and_symmetric():
    angle_ply = Laminate.from_spec("[±45]s", 0.125e-3, t300())
    with pytest.raises(ValueError):
        fsdt_navier_deflection(angle_ply, A, B)
    anti = Laminate.from_spec("[0/90]", 0.125e-3, t300())
    with pytest.raises(ValueError):
        navier_clt_deflection(anti, A, B)


# ---------------------------------------------------------------------------
# 6) 铺层角扫描的物理直觉
# ---------------------------------------------------------------------------
def test_axial_stiffness_and_compression_buckling_are_not_the_same_ranking():
    """面内轴向刚度 ``A11`` 在 ``[0]`` 最大；但薄板屈曲载荷排序不同（见报告 §5）。

    屈曲由弯曲刚度组合 ``sqrt(D11 D22) + D12 + 2 D66`` 控制：单向板 ``D22/G66`` 极小，
    方板 ``N_cr`` 反而低于 ``[±45]s``。这里把两条事实都钉住。
    """
    t = 0.125e-3
    angles = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
    lams = {th: Laminate.from_spec(f"[{th}/-{th}]s", t, t300()) for th in angles}
    A11 = {th: lam.A[0, 0] for th, lam in lams.items()}
    assert max(A11, key=A11.get) == 0.0          # 面内轴向刚度：0° 最优
    assert min(A11, key=A11.get) == 90.0
    ncr = {th: buckling_eigen_uniaxial(lams[th], A, B, 10, 10).n_cr for th in angles}
    # 横观各向异性：横向承压（90°）显著弱于顺纤维承压
    assert ncr[0.0] > 1.5 * ncr[90.0]
    # 方板单轴屈曲不是 0° 最优（屈曲由 D 组合控制）——如实钉住该事实
    assert ncr[45.0] > ncr[0.0]


def test_inplane_shear_stiffness_and_shear_buckling_best_at_45():
    """面内剪切刚度 ``A66`` 与方板纯剪屈曲均在 ``[±45]`` 最优（弯-扭耦合在剪切下起稳定作用）。"""
    t = 0.125e-3
    angles = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
    lams = {th: Laminate.from_spec(f"[{th}/-{th}]s", t, t300()) for th in angles}
    A66 = {th: lam.A[2, 2] for th, lam in lams.items()}
    assert max(A66, key=A66.get) == 45.0
    nxy = {th: buckling_eigen_shear(lams[th], A, B, 12, 12).n_cr for th in angles}
    assert max(nxy, key=nxy.get) == 45.0
    assert nxy[45.0] > 2.5 * nxy[0.0]
    # 90° 旋转不变性（方板纯剪）：[0] 与 [90] 结果必须相同
    assert nxy[0.0] == pytest.approx(nxy[90.0], rel=1e-6)


def test_bend_twist_coupling_sign_effect():
    """弯-扭耦合的方向性：剪切下稳定（提高 ``N_xy,cr``），压缩下不利（降低 ``N_x,cr``）。"""
    t = 0.125e-3
    lam = Laminate.from_spec("[45/-45]s", t, t300())
    decoupled = Laminate.from_spec("[45/-45]s", t, t300())
    D0 = decoupled.D.copy()
    D0[0, 2] = D0[1, 2] = D0[2, 0] = D0[2, 1] = 0.0
    decoupled.D = D0
    nxy_c = buckling_eigen_shear(lam, A, B, 12, 12).n_cr
    nxy_d = buckling_eigen_shear(decoupled, A, B, 12, 12).n_cr
    nx_c = buckling_eigen_uniaxial(lam, A, B, 12, 12).n_cr
    nx_d = buckling_eigen_uniaxial(decoupled, A, B, 12, 12).n_cr
    assert nxy_c > nxy_d          # 剪切：耦合稳定
    assert nx_c < nx_d            # 单轴压缩：耦合不利
    # 挠度：耦合使中心挠度增大（同 D11/D22/D12/D66 解耦参照）
    assert lam.D16 > 0.0


# ---------------------------------------------------------------------------
# 7) 场量/边界条件/几何闸门
# ---------------------------------------------------------------------------
def test_deflection_field_boundary_and_centre():
    lam = Laminate.from_spec("[0/90]2s", 0.125e-3, t300())
    res = navier_clt_deflection(lam, A, B, 1.0e4, 41, 41)
    xs = np.linspace(0.0, A, 21)
    on_edge = res.field(xs, np.zeros_like(xs))
    assert np.max(np.abs(on_edge)) < 1e-12
    side_edge = res.field(np.full_like(xs, A), xs)
    assert np.max(np.abs(side_edge)) < 1e-12
    centre = res.field([0.5 * A], [0.5 * B])[0, 0]
    assert centre == pytest.approx(res.w_max, rel=1e-12)
    grid = res.field(np.linspace(0, A, 11), np.linspace(0, B, 11))
    assert grid.max() == pytest.approx(res.w_max, rel=1e-3)
    assert grid.min() > -1e-12  # 均布载荷下挠度处处非负


def test_ritz_energy_is_stationary():
    """变分自检：在 Ritz 解处叠加任意扰动系数，总势能必增大。"""
    lam = Laminate.from_spec("[±45]s", 0.125e-3, t300())
    q0 = 1.0e4
    M = N = 6
    res = ritz_deflection(lam, A, B, q0, M, N)
    K = _ritz_bending_matrix(lam, A, B, M, N)
    F = np.array(
        [
            4.0 * q0 * A * B / (math.pi**2 * m * n) if (m % 2 == 1 and n % 2 == 1) else 0.0
            for m in range(1, M + 1)
            for n in range(1, N + 1)
        ]
    )
    w0 = np.array([res.coeffs[m, n] for m in range(1, M + 1) for n in range(1, N + 1)])
    pi0 = 0.5 * w0 @ K @ w0 - F @ w0
    rng = np.random.default_rng(7)
    for scale in (1e-3, 1e-2, 1e-1):
        w1 = w0 + scale * np.abs(w0).max() * rng.normal(size=w0.size)
        pi1 = 0.5 * w1 @ K @ w1 - F @ w1
        assert pi1 > pi0


def test_geometry_gate_rejects_bad_input():
    with pytest.raises(ValueError):
        validate_plate_geometry(-1.0, 1.0, 0.01)
    with pytest.raises(ValueError):
        validate_plate_geometry(1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        validate_plate_geometry(100.0, 1.0, 0.01)
    geo = validate_plate_geometry(1.0, 1.0, 0.1)
    assert geo["h_over_a"] == pytest.approx(0.1)
    assert geo["thin_plate_ok"] is True


def test_parse_layup_variants():
    assert [a for a, _ in parse_layup("[0/90]s", 0.1)] == [0.0, 90.0, 90.0, 0.0]
    assert [a for a, _ in parse_layup("[±45]s", 0.1)] == [45.0, -45.0, -45.0, 45.0]
    assert len(parse_layup("[0/90]2s", 0.1)) == 8
    assert [a for a, _ in parse_layup("0/90/0", 0.1)] == [0.0, 90.0, 0.0]
    with pytest.raises(ValueError):
        parse_layup("0/90", 0.0)


def test_quasi_isotropic_extension_stiffness():
    """``[0/90/±45]s`` 面内近似准各向同性：``A11 ~ A22``、``A16 ~ A26 ~ 0``。"""
    lam = Laminate.from_spec("[0/90/45/-45]s", 0.125e-3, t300())
    assert lam.A[0, 0] == pytest.approx(lam.A[1, 1], rel=2e-2)
    assert abs(lam.A[0, 2]) < 1e-6 * lam.A[0, 0]
    assert abs(lam.A[1, 2]) < 1e-6 * lam.A[0, 0]


def test_d_star_reduces_coupling_for_antisymmetric():
    """``D* = D - B A^-1 B``：面内不动约束下等效弯曲刚度不超过 ``D``。"""
    lam = Laminate.from_spec("[0/90]", 0.125e-3, t300())
    Dstar = lam.d_star()
    assert Dstar[0, 0] < lam.D11 + 1e-9
    sym = Laminate.from_spec("[0/90]s", 0.125e-3, t300())
    assert np.allclose(sym.d_star(), sym.D, rtol=1e-12)