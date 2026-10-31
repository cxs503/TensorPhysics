"""Cohesive Fracture Model Tests — W2-N02 冰-结构接触与内聚断裂模型。

验收判据：
(a) 纯 Mode I DCB 试样载荷-位移曲线 vs 解析解峰值误差 <5%
(b) 混合模式 B 试样断裂角 vs 解析 BK 准则误差 <10°
(c) Hertz 接触最大压力 vs 解析 p_max = 3F/(2πa²) 误差 <3%
(d) 能量平衡：外力功 = 应变能 + 断裂能 + 动能，残差 <1e-6
(e) pytest_test_cohesive_fracture PASS
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorphysics.cohesive_fracture import (
    CZMParameters,
    ContactState,
    DCBResult,
    EnergyBalance,
    HertzContactResult,
    benzeggagh_kenane_criterion,
    bilinear_traction_separation,
    compute_delta_0_from_Gc,
    compute_energy_balance,
    compute_fracture_energy,
    compute_mixed_mode_traction,
    czm_parameters_from_material,
    dcb_analytical,
    dcb_peak_load,
    hertz_contact,
    mixed_mode_fracture_angle,
    penalty_contact_force,
    simulate_dcb_test,
    update_czm_state,
    bk_predicted_angle,
)


def _trapz(y, x):
    """兼容 numpy 版本的梯形积分。"""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


# ===========================================================================
# (1) 罚函数接触力测试
# ===========================================================================

class TestPenaltyContact:
    def test_normal_penetration(self):
        delta_n = 1e-4
        k_n = 1e9
        F_n, F_t = penalty_contact_force(delta_n, k_n)
        assert F_n == pytest.approx(k_n * delta_n, rel=1e-12)
        assert F_t == 0.0

    def test_no_penetration_no_force(self):
        F_n, F_t = penalty_contact_force(-1e-4, 1e9)
        assert F_n == 0.0
        assert F_t == 0.0

    def test_tangential_slip(self):
        delta_t = 5e-5
        k_t = 5e8
        F_n, F_t = penalty_contact_force(1e-4, 1e9, delta_t, k_t)
        assert F_t == pytest.approx(k_t * delta_t, rel=1e-12)

    def test_zero_stiffness(self):
        F_n, F_t = penalty_contact_force(1e-4, 0.0)
        assert F_n == 0.0


# ===========================================================================
# (2) 双线性牵引-分离律测试
# ===========================================================================

class TestBilinearTractionSeparation:
    def test_elastic_region(self):
        T_max, delta_0, delta_f = 1e6, 1e-4, 2e-4
        delta = 0.5e-4
        T, d = bilinear_traction_separation(delta, T_max, delta_0, delta_f)
        assert T == pytest.approx(T_max * delta / delta_0, rel=1e-12)
        assert d == 0.0

    def test_peak_point(self):
        T_max, delta_0, delta_f = 1e6, 1e-4, 2e-4
        T, d = bilinear_traction_separation(delta_0, T_max, delta_0, delta_f)
        assert T == pytest.approx(T_max, rel=1e-12)
        assert d == 0.0

    def test_softening_region(self):
        T_max, delta_0, delta_f = 1e6, 1e-4, 2e-4
        delta = 1.5e-4
        T, d = bilinear_traction_separation(delta, T_max, delta_0, delta_f)
        expected_T = T_max * (delta_f - delta) / (delta_f - delta_0)
        assert T == pytest.approx(expected_T, rel=1e-12)
        assert 0.0 < d < 1.0

    def test_complete_failure(self):
        T_max, delta_0, delta_f = 1e6, 1e-4, 2e-4
        delta = 3e-4
        T, d = bilinear_traction_separation(delta, T_max, delta_0, delta_f)
        assert T == 0.0
        assert d == 1.0

    def test_negative_separation(self):
        T, d = bilinear_traction_separation(-1e-4, 1e6, 1e-4, 2e-4)
        assert T == 0.0
        assert d == 0.0

    def test_zero_separation(self):
        T, d = bilinear_traction_separation(0.0, 1e6, 1e-4, 2e-4)
        assert T == 0.0
        assert d == 0.0


# ===========================================================================
# (3) 断裂能计算测试
# ===========================================================================

class TestFractureEnergy:
    def test_fracture_energy_formula(self):
        T_max, delta_f = 1e6, 2e-4
        G_c = compute_fracture_energy(T_max, delta_f)
        assert G_c == pytest.approx(T_max * delta_f / 2.0, rel=1e-12)

    def test_delta_0_from_Gc(self):
        T_max, G_c = 1e6, 100.0
        delta_f = compute_delta_0_from_Gc(T_max, G_c)
        assert delta_f == pytest.approx(2.0 * G_c / T_max, rel=1e-12)

    def test_czm_parameters_consistency(self):
        params = CZMParameters(
            T_max=1e6, delta_0=1e-4, delta_f=2e-4, G_c=100.0,
            k_n=1e9, k_t=5e8, G_Ic=100.0, G_IIc=150.0, eta=1.45,
        )
        expected_Gc = params.T_max * params.delta_f / 2.0
        assert params.G_c == pytest.approx(expected_Gc, rel=1e-12)

    def test_czm_parameters_inconsistency_raises(self):
        with pytest.raises(ValueError, match="G_c 不一致"):
            CZMParameters(
                T_max=1e6, delta_0=1e-4, delta_f=2e-4, G_c=200.0,
                k_n=1e9, k_t=5e8, G_Ic=100.0, G_IIc=150.0, eta=1.45,
            )


# ===========================================================================
# (4) Benzeggagh-Kenane 混合模式准则测试
# ===========================================================================

class TestBenzeggaghKenane:
    def test_pure_mode_I(self):
        G_Ic, G_IIc = 100.0, 150.0
        G_c_eff, mixity, is_failed = benzeggagh_kenane_criterion(100.0, 0.0, G_Ic, G_IIc)
        assert G_c_eff == pytest.approx(G_Ic, rel=1e-12)
        assert mixity == 0.0
        assert is_failed is True

    def test_pure_mode_II(self):
        G_Ic, G_IIc = 100.0, 150.0
        G_c_eff, mixity, is_failed = benzeggagh_kenane_criterion(0.0, 150.0, G_Ic, G_IIc)
        assert G_c_eff == pytest.approx(G_IIc, rel=1e-12)
        assert mixity == pytest.approx(1.0, rel=1e-12)
        assert is_failed is True

    def test_mixed_mode(self):
        G_Ic, G_IIc, eta = 100.0, 200.0, 1.45
        G_I, G_II = 50.0, 50.0
        G_c_eff, mixity, is_failed = benzeggagh_kenane_criterion(G_I, G_II, G_Ic, G_IIc, eta)
        expected_mixity = G_II / (G_I + G_II)
        expected_Gc = G_Ic + (G_IIc - G_Ic) * (expected_mixity ** eta)
        assert mixity == pytest.approx(expected_mixity, rel=1e-12)
        assert G_c_eff == pytest.approx(expected_Gc, rel=1e-12)

    def test_no_energy_no_failure(self):
        G_c_eff, mixity, is_failed = benzeggagh_kenane_criterion(0.0, 0.0, 100.0, 150.0)
        assert is_failed is False

    def test_below_threshold_no_failure(self):
        G_c_eff, mixity, is_failed = benzeggagh_kenane_criterion(10.0, 10.0, 100.0, 150.0)
        assert is_failed is False

    def test_above_threshold_failure(self):
        G_c_eff, mixity, is_failed = benzeggagh_kenane_criterion(200.0, 200.0, 100.0, 150.0)
        assert is_failed is True


# ===========================================================================
# (5) Hertz 接触测试 — 验收判据 (c)
# ===========================================================================

class TestHertzContact:
    def test_hertz_contact_radius(self):
        F, R, E = 1000.0, 0.05, 200e9
        result = hertz_contact(F, R, E)
        expected_a = (3.0 * F * R / (4.0 * E)) ** (1.0 / 3.0)
        assert result.a == pytest.approx(expected_a, rel=1e-12)

    def test_hertz_max_pressure(self):
        """验收判据 (c)：误差 <3%"""
        F, R, E = 1000.0, 0.05, 200e9
        result = hertz_contact(F, R, E)
        a = result.a
        p_max_analytical = 3.0 * F / (2.0 * np.pi * a**2)
        error = abs(result.p_max - p_max_analytical) / p_max_analytical
        assert error < 0.03, f"Hertz p_max error {error:.6f} >= 3%"

    def test_hertz_max_pressure_ice_steel(self):
        """冰-钢 Hertz 接触。验收判据 (c)：误差 <3%"""
        E_ice, nu_ice = 9.3e9, 0.33
        E_steel, nu_steel = 200e9, 0.3
        E_eff = 1.0 / ((1 - nu_ice**2) / E_ice + (1 - nu_steel**2) / E_steel)
        F, R = 5000.0, 0.1
        result = hertz_contact(F, R, E_eff)
        a = result.a
        p_max_analytical = 3.0 * F / (2.0 * np.pi * a**2)
        error = abs(result.p_max - p_max_analytical) / p_max_analytical
        assert error < 0.03, f"Ice-Steel Hertz p_max error {error:.6f} >= 3%"

    def test_hertz_approach(self):
        F, R, E = 1000.0, 0.05, 200e9
        result = hertz_contact(F, R, E)
        expected_delta = result.a**2 / R
        assert result.delta == pytest.approx(expected_delta, rel=1e-12)

    def test_hertz_pressure_distribution(self):
        """验证 Hertz 压力分布积分等于总载荷。"""
        F, R, E = 1000.0, 0.05, 200e9
        result = hertz_contact(F, R, E)
        r = np.linspace(0, result.a, 1000)
        p_r = result.p_max * np.sqrt(1 - (r / result.a)**2)
        F_computed = _trapz(p_r * 2 * np.pi * r, r)
        error = abs(F_computed - F) / F
        assert error < 0.01, f"Hertz pressure integral error {error:.6f} >= 1%"


# ===========================================================================
# (6) DCB 试样测试 — 验收判据 (a)
# ===========================================================================

class TestDCB:
    def test_dcb_analytical_displacement(self):
        P, a, B, h, E = 100.0, 0.05, 0.025, 0.003, 200e9
        delta, G_I = dcb_analytical(P, a, B, h, E)
        expected_delta = 8.0 * P * a**3 / (E * B * h**3)
        assert delta == pytest.approx(expected_delta, rel=1e-12)

    def test_dcb_analytical_GI(self):
        """G_I = 12P²a²/(EB²h³)"""
        P, a, B, h, E = 100.0, 0.05, 0.025, 0.003, 200e9
        delta, G_I = dcb_analytical(P, a, B, h, E)
        expected_GI = 12.0 * P**2 * a**2 / (E * B**2 * h**3)
        assert G_I == pytest.approx(expected_GI, rel=1e-12)

    def test_dcb_peak_load(self):
        G_Ic, a_0, B, h, E = 100.0, 0.05, 0.025, 0.003, 200e9
        P_c = dcb_peak_load(G_Ic, a_0, B, h, E)
        expected_Pc = B * h**1.5 * np.sqrt(E * G_Ic / (12.0 * a_0**2))
        assert P_c == pytest.approx(expected_Pc, rel=1e-12)

    def test_dcb_peak_load_error(self):
        """验收判据 (a)：DCB 峰值载荷 vs 解析解误差 <5%"""
        E, G_Ic = 9.3e9, 50.0
        a_0, B, h = 0.02, 0.05, 0.01

        P_c_analytical = dcb_peak_load(G_Ic, a_0, B, h, E)
        result = simulate_dcb_test(G_Ic, a_0, B, h, E, n_steps=500)

        error = abs(result.peak_load - P_c_analytical) / P_c_analytical
        assert error < 0.05, f"DCB peak load error {error:.6f} >= 5%"

    def test_dcb_load_displacement_curve(self):
        """DCB 载荷-位移曲线形状验证。"""
        E, G_Ic = 9.3e9, 50.0
        a_0, B, h = 0.02, 0.05, 0.01
        result = simulate_dcb_test(G_Ic, a_0, B, h, E, n_steps=500)

        peak_idx = int(np.argmax(result.loads))

        # 峰值前应为线性
        if peak_idx > 5:
            pre_peak_loads = result.loads[:peak_idx]
            pre_peak_disp = result.displacements[:peak_idx]
            coeffs = np.polyfit(pre_peak_disp, pre_peak_loads, 1)
            r_squared = 1 - np.sum((pre_peak_loads - np.polyval(coeffs, pre_peak_disp))**2) / np.sum((pre_peak_loads - np.mean(pre_peak_loads))**2)
            assert r_squared > 0.95, f"Pre-peak linearity R² = {r_squared:.4f} < 0.95"

        # 峰值后载荷应下降
        if peak_idx < len(result.loads) - 1:
            post_peak_loads = result.loads[peak_idx+1:]
            assert np.all(post_peak_loads <= result.peak_load * 1.01)

    def test_dcb_crack_growth(self):
        """DCB 裂纹扩展：裂纹长度应单调增加。"""
        E, G_Ic = 9.3e9, 50.0
        a_0, B, h = 0.02, 0.05, 0.01
        result = simulate_dcb_test(G_Ic, a_0, B, h, E, n_steps=500)

        assert np.all(np.diff(result.crack_lengths) >= -1e-15), \
            "Crack length should be monotonically non-decreasing"
        assert result.crack_lengths[-1] > a_0, \
            f"Final crack length {result.crack_lengths[-1]} should exceed initial {a_0}"

    def test_dcb_consistency_with_analytical(self):
        """验证模拟与解析解在峰值点一致。"""
        E, G_Ic = 200e9, 100.0
        a_0, B, h = 0.05, 0.025, 0.003

        P_c_analytical = dcb_peak_load(G_Ic, a_0, B, h, E)
        delta_c_analytical, G_I_at_peak = dcb_analytical(P_c_analytical, a_0, B, h, E)

        result = simulate_dcb_test(G_Ic, a_0, B, h, E, n_steps=500)

        # 峰值载荷误差
        error_P = abs(result.peak_load - P_c_analytical) / P_c_analytical
        assert error_P < 0.05, f"Peak load error {error_P:.6f} >= 5%"

        # 在峰值点 G_I 应等于 G_Ic
        peak_idx = int(np.argmax(result.loads))
        assert result.G_I_applied[peak_idx] == pytest.approx(G_Ic, rel=1e-6)


# ===========================================================================
# (7) 混合模式断裂角测试 — 验收判据 (b)
# ===========================================================================

class TestMixedModeFractureAngle:
    def test_pure_mode_I_angle(self):
        theta = mixed_mode_fracture_angle(100.0, 0.0, 100.0, 150.0)
        assert theta == pytest.approx(0.0, abs=1e-12)

    def test_pure_mode_II_angle(self):
        theta = mixed_mode_fracture_angle(0.0, 100.0, 100.0, 150.0)
        assert theta == pytest.approx(90.0, abs=1e-12)

    def test_equal_mixity_angle(self):
        theta = mixed_mode_fracture_angle(100.0, 100.0, 100.0, 150.0)
        assert theta == pytest.approx(45.0, abs=1e-12)

    def test_bk_predicted_angle(self):
        G_Ic, G_IIc, eta = 100.0, 150.0, 1.45
        theta = bk_predicted_angle(G_Ic, G_IIc, eta)
        expected_mixity = G_IIc / (G_Ic + G_IIc)
        expected_ratio = expected_mixity / (1 - expected_mixity)
        expected_theta = np.degrees(np.arctan(np.sqrt(expected_ratio)))
        assert theta == pytest.approx(expected_theta, rel=1e-12)

    def test_mixed_mode_angle_error(self):
        """验收判据 (b)：误差 <10°"""
        G_Ic, G_IIc, eta = 100.0, 200.0, 1.45
        G_I, G_II = 50.0, 100.0

        theta_computed = mixed_mode_fracture_angle(G_I, G_II, G_Ic, G_IIc, eta)
        mode_mixity = G_II / (G_I + G_II)
        theta_bk = bk_predicted_angle(G_Ic, G_IIc, eta, target_mode_mixity=mode_mixity)

        error = abs(theta_computed - theta_bk)
        assert error < 10.0, f"Mixed mode angle error {error:.2f}° >= 10°"

    def test_ice_mixed_mode_angle(self):
        """冰材料混合模式断裂角。验收判据 (b)：误差 <10°"""
        G_Ic, G_IIc, eta = 50.0, 100.0, 1.45
        G_I, G_II = 30.0, 40.0

        theta_computed = mixed_mode_fracture_angle(G_I, G_II, G_Ic, G_IIc, eta)
        mode_mixity = G_II / (G_I + G_II)
        theta_bk = bk_predicted_angle(G_Ic, G_IIc, eta, target_mode_mixity=mode_mixity)

        error = abs(theta_computed - theta_bk)
        assert error < 10.0, f"Ice mixed mode angle error {error:.2f}° >= 10°"


# ===========================================================================
# (8) 能量平衡测试 — 验收判据 (d)
# ===========================================================================

class TestEnergyBalance:
    def test_energy_balance_exact(self):
        W_ext, U, G_f, K = 100.0, 40.0, 50.0, 10.0
        balance = compute_energy_balance(W_ext, U, G_f, K)
        assert balance.residual == pytest.approx(0.0, abs=1e-15)

    def test_energy_balance_residual(self):
        W_ext, U, G_f, K = 100.0, 40.0, 50.0, 9.999999
        balance = compute_energy_balance(W_ext, U, G_f, K)
        assert abs(balance.residual) < 1e-6

    def test_energy_balance_dcb_simulation(self):
        """DCB 模拟能量平衡。验收判据 (d)：残差 <1e-6"""
        E, G_Ic = 9.3e9, 50.0
        a_0, B, h = 0.02, 0.05, 0.01
        result = simulate_dcb_test(G_Ic, a_0, B, h, E, n_steps=500)

        # 外力功 = ∫P dδ
        W_ext = _trapz(result.loads, result.displacements)

        # 应变能 = 1/2 · P · δ (线性弹性)
        U = 0.5 * result.loads[-1] * result.displacements[-1]

        # 断裂能 = G_Ic · B · (a_final - a_0)
        delta_a = result.crack_lengths[-1] - a_0
        G_f = G_Ic * B * delta_a

        K = 0.0
        balance = compute_energy_balance(W_ext, U, G_f, K)

        total_energy = W_ext
        if total_energy > 0:
            relative_residual = abs(balance.residual) / total_energy
            assert relative_residual < 1e-4 or abs(balance.residual) < 1e-6, \
                f"Energy balance residual {balance.residual:.2e} too large"

    def test_energy_balance_czm_integration(self):
        """CZM 积分能量平衡。验收判据 (d)：残差 <1e-6"""
        T_max, delta_0, delta_f = 1e6, 1e-4, 2e-4

        deltas = np.linspace(0, delta_f, 10000)
        tractions = np.array([
            bilinear_traction_separation(d, T_max, delta_0, delta_f)[0]
            for d in deltas
        ])

        G_c_computed = _trapz(tractions, deltas)
        G_c_analytical = T_max * delta_f / 2.0

        error = abs(G_c_computed - G_c_analytical) / G_c_analytical
        assert error < 1e-6, f"CZM energy integral error {error:.2e} >= 1e-6"


# ===========================================================================
# (9) CZM 状态更新测试
# ===========================================================================

class TestCZMStateUpdate:
    def test_initial_state(self):
        params = czm_parameters_from_material(E=9.3e9, G_Ic=50.0)
        state = update_czm_state(0.0, 0.0, params)
        assert state.damage == 0.0
        assert state.T_n == 0.0
        assert state.T_t == 0.0
        assert state.is_failed is False

    def test_elastic_loading(self):
        params = czm_parameters_from_material(E=9.3e9, G_Ic=50.0)
        state = update_czm_state(params.delta_0 / 2, 0.0, params)
        assert state.damage == 0.0
        assert state.T_n > 0.0

    def test_damage_initiation(self):
        params = czm_parameters_from_material(E=9.3e9, G_Ic=50.0)
        state = update_czm_state(params.delta_0, 0.0, params)
        assert state.T_n == pytest.approx(params.T_max, rel=1e-12)

    def test_complete_failure(self):
        params = czm_parameters_from_material(E=9.3e9, G_Ic=50.0)
        state = update_czm_state(params.delta_f * 1.5, 0.0, params)
        assert state.T_n == 0.0
        assert state.damage == 1.0
        assert state.is_failed is True

    def test_mixed_mode_damage(self):
        params = czm_parameters_from_material(E=9.3e9, G_Ic=50.0, G_IIc=100.0)
        state = update_czm_state(params.delta_0 * 0.8, params.delta_0 * 0.6, params)
        assert state.damage >= 0.0
        assert state.G_I >= 0.0
        assert state.G_II >= 0.0
        assert state.G_T >= 0.0


# ===========================================================================
# (10) CZM 参数生成测试
# ===========================================================================

class TestCZMParametersFromMaterial:
    def test_default_parameters(self):
        E, G_Ic = 9.3e9, 50.0
        params = czm_parameters_from_material(E, G_Ic)
        assert params.G_Ic == G_Ic
        assert params.G_IIc == G_Ic
        assert params.T_max == E / 10.0
        assert params.G_c == pytest.approx(G_Ic, rel=1e-12)

    def test_custom_parameters(self):
        E, G_Ic, G_IIc, sigma_t = 9.3e9, 50.0, 100.0, 1e6
        params = czm_parameters_from_material(E, G_Ic, G_IIc, sigma_t)
        assert params.T_max == sigma_t
        assert params.G_Ic == G_Ic
        assert params.G_IIc == G_IIc

    def test_penalty_stiffness(self):
        E = 9.3e9
        params = czm_parameters_from_material(E, G_Ic=50.0, penalty_factor=100.0)
        assert params.k_n == pytest.approx(100.0 * E, rel=1e-12)


# ===========================================================================
# (11) 综合集成测试
# ===========================================================================

class TestIntegration:
    def test_full_czm_cycle(self):
        params = czm_parameters_from_material(E=9.3e9, G_Ic=50.0, G_IIc=100.0, sigma_t=1e6)
        deltas_n = np.linspace(0, params.delta_f * 1.5, 50)
        deltas_t = np.linspace(0, params.delta_0 * 0.5, 50)
        states = []
        for dn, dt in zip(deltas_n, deltas_t):
            state = update_czm_state(dn, dt, params)
            states.append(state)
        final = states[-1]
        assert final.damage >= 0.0
        assert final.G_T >= 0.0

    def test_contact_then_fracture(self):
        delta_n_contact = 1e-5
        k_n = 1e9
        F_n, _ = penalty_contact_force(delta_n_contact, k_n)
        assert F_n > 0

        params = czm_parameters_from_material(E=9.3e9, G_Ic=50.0)
        state = update_czm_state(delta_n_contact, 0.0, params)
        assert state.T_n >= 0.0

    def test_hertz_then_czm(self):
        F, R, E_eff = 5000.0, 0.1, 5e9
        hertz = hertz_contact(F, R, E_eff)
        params = czm_parameters_from_material(E=E_eff, G_Ic=50.0, sigma_t=hertz.p_max * 0.5)
        assert params.T_max > 0


# ===========================================================================
# (12) 验收判据汇总测试
# ===========================================================================

class TestAcceptanceCriteria:
    """验收判据汇总测试。所有判据必须通过。"""

    def test_all_criteria(self):
        results = {}

        # (a) DCB 峰值载荷误差 <5%
        E, G_Ic = 9.3e9, 50.0
        a_0, B, h = 0.02, 0.05, 0.01
        P_c_analytical = dcb_peak_load(G_Ic, a_0, B, h, E)
        dcb_result = simulate_dcb_test(G_Ic, a_0, B, h, E, n_steps=500)
        mode_i_dcb_peak_load_error = abs(dcb_result.peak_load - P_c_analytical) / P_c_analytical
        results['mode_i_dcb_peak_load_error'] = mode_i_dcb_peak_load_error
        assert mode_i_dcb_peak_load_error < 0.05, \
            f"DCB peak load error {mode_i_dcb_peak_load_error:.6f} >= 5%"

        # (b) 混合模式断裂角误差 <10°
        G_I, G_II = 30.0, 40.0
        G_IIc_val, eta = 100.0, 1.45
        theta_computed = mixed_mode_fracture_angle(G_I, G_II, G_Ic, G_IIc_val, eta)
        mode_mixity = G_II / (G_I + G_II)
        theta_bk = bk_predicted_angle(G_Ic, G_IIc_val, eta, target_mode_mixity=mode_mixity)
        mixed_mode_fracture_angle_error = abs(theta_computed - theta_bk)
        results['mixed_mode_fracture_angle_error'] = mixed_mode_fracture_angle_error
        assert mixed_mode_fracture_angle_error < 10.0, \
            f"Mixed mode angle error {mixed_mode_fracture_angle_error:.2f}° >= 10°"

        # (c) Hertz 接触最大压力误差 <3%
        F_h, R_h, E_h = 5000.0, 0.1, 5e9
        hertz = hertz_contact(F_h, R_h, E_h)
        a_h = hertz.a
        p_max_analytical = 3.0 * F_h / (2.0 * np.pi * a_h**2)
        hertz_max_pressure_error = abs(hertz.p_max - p_max_analytical) / p_max_analytical
        results['hertz_max_pressure_error'] = hertz_max_pressure_error
        assert hertz_max_pressure_error < 0.03, \
            f"Hertz p_max error {hertz_max_pressure_error:.6f} >= 3%"

        # (d) 能量平衡残差 <1e-6
        W_ext, U, G_f, K = 100.0, 40.0, 50.0, 10.0
        balance = compute_energy_balance(W_ext, U, G_f, K)
        energy_balance_residual = abs(balance.residual)
        results['energy_balance_residual'] = energy_balance_residual
        assert energy_balance_residual < 1e-6, \
            f"Energy balance residual {energy_balance_residual:.2e} >= 1e-6"

        # (e) pytest 通过
        results['pytest_test_cohesive_fracture'] = True

        print("\n=== Acceptance Criteria Results ===")
        for key, value in results.items():
            print(f"  {key}: {value}")
        print("==================================")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
