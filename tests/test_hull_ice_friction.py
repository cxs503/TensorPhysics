"""Hull-ice friction 温度依赖模块测试（物理正确性 + 数值收敛性 + 端到端判据）。

覆盖：
1. 冰材料关联式（E/σ_c/τ_s/H 的温度相关性、熔点盐度修正）
2. Blok 界面热分配
3. 闪温解析解的量纲/标度关系
4. 一维有限差分对解析解的验证与网格收敛阶
5. 混合润滑接触律的自洽性（μ=(1-λ)μ_dry+λμ_wet 恒等式、区间约束、单调性）
6. 不动点求解器收敛性（二分无条件收敛；松弛 Picard 在冷端不收敛）
7. 数值摩擦试验：10 重复的确定性、温度单调性、统计离散度
8. 幂律拟合：参数回收、R²、相对误差
9. 参考数据表（5 点、标定/验证划分）
10. 端到端 validation 流程：判据 a/b/c 全通过 + JSON 产物字段
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorphysics.hull_ice_friction import (
    DEFAULT_ICE,
    ANALYTIC_SURFACE_COEFFICIENT,
    IceSteelContactParameters,
    PowerLawFit,
    fit_power_law,
    flash_temperature,
    fd_analytic_relative_error,
    fd_grid_convergence,
    fd_surface_temperature,
    interface_heat_partition,
    power_law_mu,
    r_squared,
    reference_table,
    reference_temperatures,
    role_targets,
    run_temperature_scan,
    solve_contact_ensemble,
)
from tensorphysics.hull_ice_friction.calibration import (
    identify_real_contact_pressure,
    relative_errors,
)
from tensorphysics.hull_ice_friction.contact_law import (
    dry_friction_coefficient,
    melt_fraction,
)
from tensorphysics.hull_ice_friction.experiment import (
    AsperityEnsembleParameters,
    FrictionTestConfig,
    build_ensemble_bank,
    evaluate_ensemble_bank,
)
from tensorphysics.hull_ice_friction.ice_materials import (
    MELTING_DEPRESSION_C_PER_PSU,
    IceMaterialCorrelations,
)
from tensorphysics.hull_ice_friction.reference_data import PROVENANCE
from tensorphysics.hull_ice_friction.thermal_contact import STEEL_THERMAL
from tensorphysics.hull_ice_friction.validation import CRITERIA, run_validation


# ---------------------------------------------------------------------------
# 1. 冰材料关联式
# ---------------------------------------------------------------------------


class TestIceMaterialCorrelations:
    def test_undercooling_and_salinity_melting_point(self):
        """过冷度定义与盐度导致的熔点降低。"""
        material = IceMaterialCorrelations()
        assert material.melting_temperature(0.0) == 0.0
        assert material.melting_temperature(35.0) == pytest.approx(
            -MELTING_DEPRESSION_C_PER_PSU * 35.0
        )
        assert float(material.undercooling(-15.0)) == pytest.approx(15.0)
        # 高于熔点时过冷度截断为 0（不允许负过冷度）
        assert float(material.undercooling(2.0)) == 0.0
        with pytest.raises(ValueError):
            material.melting_temperature(-1.0)

    def test_modulus_and_strength_increase_with_cold(self):
        """E、σ_c、τ_s、H 均随过冷度单调增加，且数值落在文献量级区间。"""
        material = IceMaterialCorrelations()
        theta = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        e = material.elastic_modulus_GPa(theta)
        sigma = material.compressive_strength_MPa(theta)
        tau = material.shear_strength_Pa(theta)
        hardness = material.hardness_Pa(theta)
        for series in (e, sigma, tau, hardness):
            assert np.all(np.diff(series) > 0.0)
        # 量级检查（文献区间）
        assert 8.0 < e[0] < 12.0
        assert 1.0 < sigma[0] < 5.0
        assert 1.0 < sigma[-1] < 8.0
        assert 0.2e6 < tau[0] < 2.0e6
        assert 10e6 < hardness[0] < 60e6
        # 剪切强度 = χ·σ_c
        assert np.allclose(tau, material.tau_over_sigma * material.compressive_strength_Pa(theta))

    def test_thermal_properties_consistency(self):
        """热扩散率与热渗透系数满足定义式。"""
        assert DEFAULT_ICE.thermal_diffusivity_m2_s == pytest.approx(
            DEFAULT_ICE.conductivity_W_mK
            / (DEFAULT_ICE.density_kg_m3 * DEFAULT_ICE.specific_heat_J_kgK)
        )
        assert DEFAULT_ICE.effusivity == pytest.approx(
            math.sqrt(
                DEFAULT_ICE.conductivity_W_mK
                * DEFAULT_ICE.density_kg_m3
                * DEFAULT_ICE.specific_heat_J_kgK
            )
        )


# ---------------------------------------------------------------------------
# 2. 界面热分配
# ---------------------------------------------------------------------------


class TestHeatPartition:
    def test_partition_sums_to_one_and_ice_is_minority_share(self):
        """钢的热渗透系数远大于冰，故摩擦热大部分进入钢侧（~87%）。"""
        partition = interface_heat_partition()
        assert partition.zeta_ice + partition.zeta_steel == pytest.approx(1.0)
        assert 0.10 < partition.zeta_ice < 0.15
        assert partition.effusivity_sum == pytest.approx(
            DEFAULT_ICE.effusivity + STEEL_THERMAL.effusivity
        )


# ---------------------------------------------------------------------------
# 3. 解析闪温
# ---------------------------------------------------------------------------


class TestFlashTemperature:
    def test_linear_in_flux_and_sqrt_in_time(self):
        """ΔT ∝ q、∝ sqrt(t_c)（半无限体恒定热流解的标度关系）。"""
        base = float(flash_temperature(1.0e6, 1.0e-4))
        assert float(flash_temperature(2.0e6, 1.0e-4)) == pytest.approx(2.0 * base)
        assert float(flash_temperature(1.0e6, 4.0e-4)) == pytest.approx(2.0 * base)

    def test_matches_closed_form_expression(self):
        """与 2q/S*sqrt(t/π) 的显式闭式表达式一致。"""
        partition = interface_heat_partition()
        q, t_c = 3.0e6, 2.5e-4
        expected = 2.0 * q / partition.effusivity_sum * math.sqrt(t_c / math.pi)
        assert float(flash_temperature(q, t_c, partition=partition)) == pytest.approx(expected)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            flash_temperature(1.0e6, -1.0)
        with pytest.raises(ValueError):
            flash_temperature(1.0e6, 1.0e-4, effusivity_sum=-1.0)


# ---------------------------------------------------------------------------
# 4. 有限差分验证与网格收敛
# ---------------------------------------------------------------------------


class TestFiniteDifferenceVerification:
    def test_fd_matches_analytic_surface_temperature(self):
        """隐式 Euler + 追赶法的无量纲表面温度与 2√(Fo/π) 一致（相对误差 < 1%）。"""
        for fo in (0.05, 0.25):
            fd = fd_surface_temperature(fo, n_nodes=321)
            analytic = ANALYTIC_SURFACE_COEFFICIENT * math.sqrt(fo)
            assert abs(fd - analytic) / analytic < 1.0e-2
        assert ANALYTIC_SURFACE_COEFFICIENT == pytest.approx(2.0 / math.sqrt(math.pi))

    def test_fd_grid_convergence_second_order(self):
        """网格加密后误差单调下降，相邻网格观测阶接近 2。"""
        table = fd_grid_convergence(0.25, node_counts=(41, 81, 161))
        errors = [row["abs_rel_error"] for row in table]
        assert errors[0] > errors[1] > errors[2]
        assert table[1]["observed_order"] == pytest.approx(2.0, abs=0.25)
        assert errors[-1] < 1.0e-3
        assert abs(fd_analytic_relative_error(0.25, 161)) < 1.0e-3

    def test_fd_invalid_inputs(self):
        with pytest.raises(ValueError):
            fd_surface_temperature(0.0)
        with pytest.raises(ValueError):
            fd_surface_temperature(0.1, n_nodes=3)


# ---------------------------------------------------------------------------
# 5. 接触律自洽性
# ---------------------------------------------------------------------------


def _ensemble(theta=15.0, n=64, seed=7, p_median=1.0e7, a_median=4.0e-4):
    rng = np.random.default_rng(seed)
    radius = rng.lognormal(mean=np.log(a_median), sigma=0.6, size=n)
    pressure = p_median * np.exp(0.5 * rng.standard_normal(n) - 0.125)
    return pressure, radius


class TestContactLaw:
    def test_dry_branch_equals_shear_strength_over_pressure(self):
        """μ_dry = τ_s(θ)/p_real（解析恒等式）。"""
        material = IceMaterialCorrelations()
        p_real = 3.0e6
        for theta in (5.0, 15.0, 25.0):
            expected = material.shear_strength_Pa(theta) / p_real
            assert dry_friction_coefficient(theta, p_real, material) == pytest.approx(expected)

    def test_melt_fraction_is_monotone_and_bounded(self):
        """融膜覆盖度对闪温单调增、对过冷度单调减，取值在 (0,1)。"""
        s_hot = melt_fraction(20.0, 5.0, 2.0)
        s_mid = melt_fraction(10.0, 5.0, 2.0)
        s_cold = melt_fraction(2.0, 5.0, 2.0)
        assert s_cold < s_mid < s_hot
        assert 0.0 < s_cold and s_hot < 1.0
        assert float(melt_fraction(5.0, 5.0, 2.0)) == pytest.approx(0.5)
        with pytest.raises(ValueError):
            melt_fraction(5.0, 5.0, 0.0)

    def test_mixed_law_identity_and_bounds(self):
        """面积加权结果满足 μ=(1-λ)μ_dry+λμ_wet，且 μ_wet ≤ μ ≤ μ_dry。"""
        p, a = _ensemble()
        params = IceSteelContactParameters()
        solution = solve_contact_ensemble(p, a, 15.0, params)
        assert solution.converged
        expected = (1.0 - solution.lambda_wet) * solution.dry_friction + (
            solution.lambda_wet * solution.wet_friction
        )
        assert solution.mu == pytest.approx(expected, rel=1e-9)
        assert solution.wet_friction <= solution.mu <= solution.dry_friction
        assert np.all(solution.local_mu >= solution.wet_friction - 1e-12)
        assert np.all(solution.local_mu <= solution.dry_friction + 1e-12)

    def test_friction_decreases_with_undercooling(self):
        """同一微凸体集合下，温度越低摩擦越大（干分支 τ_s(θ) 主导）。"""
        p, a = _ensemble()
        params = IceSteelContactParameters()
        mus = [
            solve_contact_ensemble(p, a, theta, params).mu for theta in (5.0, 15.0, 25.0)
        ]
        assert mus[0] < mus[1] < mus[2]

    def test_input_validation(self):
        p, a = _ensemble(n=8)
        params = IceSteelContactParameters()
        with pytest.raises(ValueError):
            solve_contact_ensemble(p, a[:-1], 15.0, params)
        with pytest.raises(ValueError):
            solve_contact_ensemble(-p, a, 15.0, params)
        with pytest.raises(ValueError):
            solve_contact_ensemble(p, a, -1.0, params)
        with pytest.raises(ValueError):
            IceSteelContactParameters(velocity_m_s=0.0)
        # μ_wet ≥ μ_dry 时物理不成立
        bad = IceSteelContactParameters(wet_friction_coefficient=0.8)
        with pytest.raises(ValueError):
            solve_contact_ensemble(p, a, 15.0, bad)


# ---------------------------------------------------------------------------
# 6. 不动点求解器收敛性
# ---------------------------------------------------------------------------


class TestSolverConvergence:
    def test_bisection_converges_and_residual_halves(self):
        """二分求解器区间宽度每次迭代减半，最终残差低于容差。"""
        p, a = _ensemble()
        params = IceSteelContactParameters()
        solution = solve_contact_ensemble(p, a, 25.0, params, solver="bisection")
        assert solution.converged
        assert solution.solver == "bisection"
        history = np.array(solution.residual_history)
        ratios = history[1:] / history[:-1]
        assert np.allclose(ratios, 0.5, atol=5e-3)  # 取 max 使个别步略偏离 0.5
        assert history[-1] < params.fixed_point_tolerance * 1.001

    def test_bisection_and_picard_agree_when_picard_converges(self):
        """暖端（-5 °C）松弛 Picard 亦收敛，且与二分结果一致。"""
        p, a = _ensemble()
        params = IceSteelContactParameters()
        bis = solve_contact_ensemble(p, a, 5.0, params, solver="bisection")
        pic = solve_contact_ensemble(p, a, 5.0, params, solver="picard")
        assert pic.converged
        assert bis.mu == pytest.approx(pic.mu, rel=1e-6)

    def test_picard_limit_cycle_for_hot_contact_spots_bisection_robust(self):
        """高闪温接触斑点（大斑 radius ~2 mm、高压 ~60 MPa）下松弛 Picard 出现极限环；
        二分求解器在宽参数范围内无条件收敛（生产求解器选择依据）。"""
        params = IceSteelContactParameters()
        hot_p, hot_a = np.array([6.0e7]), np.array([2.0e-3])
        pic = solve_contact_ensemble(hot_p, hot_a, 15.0, params, solver="picard")
        assert not pic.converged
        assert pic.iterations == params.max_iterations
        bis = solve_contact_ensemble(hot_p, hot_a, 15.0, params, solver="bisection")
        assert bis.converged and bis.residual_max < params.fixed_point_tolerance

        # 温和接触斑点（0.2 mm、2 MPa）下 Picard 收敛，且与二分一致
        mild_p, mild_a = np.array([2.0e6]), np.array([2.0e-4])
        pic_mild = solve_contact_ensemble(mild_p, mild_a, 15.0, params, solver="picard")
        bis_mild = solve_contact_ensemble(mild_p, mild_a, 15.0, params, solver="bisection")
        assert pic_mild.converged
        assert bis_mild.mu == pytest.approx(pic_mild.mu, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. 数值摩擦试验
# ---------------------------------------------------------------------------


class TestNumericalFrictionExperiment:
    def test_ten_repetitions_are_reproducible(self):
        """相同配置两次运行结果逐位一致（种子确定，无隐式随机）。"""
        config = FrictionTestConfig(n_repetitions=10)
        first = run_temperature_scan([-5.0, -25.0], config)
        second = run_temperature_scan([-5.0, -25.0], config)
        assert first["points"] == second["points"]
        for point in first["points"]:
            assert len(point["mu_repetitions"]) == 10
            assert point["mu_mean"] == pytest.approx(np.mean(point["mu_repetitions"]))
            assert point["mu_stderr"] < 0.01  # 蒙特卡洛离散度应远小于 15% 判据

    def test_temperature_trend_and_material_coupling(self):
        """μ 随温度降低单调上升；E/σ_c 随温度降低单调上升（材料耦合生效）。"""
        scan = run_temperature_scan([-5.0, -15.0, -25.0], FrictionTestConfig())
        points = scan["points"]
        mus = [p["mu_mean"] for p in points]
        assert mus[0] < mus[1] < mus[2]
        assert all(p["converged_all"] for p in points)
        assert points[0]["elastic_modulus_GPa"] < points[2]["elastic_modulus_GPa"]
        assert points[0]["compressive_strength_MPa"] < points[2]["compressive_strength_MPa"]
        # 融膜润滑在冷端应基本消失（λ 单调下降）
        lambdas = [p["lambda_wet_mean"] for p in points]
        assert lambdas[0] > lambdas[1] > lambdas[2]

    def test_ensemble_bank_reuse_is_equivalent_to_direct_scan(self):
        """样本库复用（标定用）与直接扫描数值一致。"""
        config = FrictionTestConfig()
        bank = build_ensemble_bank([-10.0], config)
        reused = evaluate_ensemble_bank(bank, config)[0]["mu_mean"]
        direct = run_temperature_scan([-10.0], config)["points"][0]["mu_mean"]
        assert reused == pytest.approx(direct, rel=1e-12)

    def test_ensemble_parameters_validation(self):
        with pytest.raises(ValueError):
            AsperityEnsembleParameters(n_asperities=4)
        with pytest.raises(ValueError):
            FrictionTestConfig(n_repetitions=0)


# ---------------------------------------------------------------------------
# 8. 幂律拟合
# ---------------------------------------------------------------------------


class TestPowerLawCalibration:
    def test_recovers_synthetic_parameters(self):
        """对合成幂律数据应回收参数并给出 R²=1。"""
        theta = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        target = (0.02, 0.0135, 1.0)
        mu = power_law_mu(theta, *target)
        fit = fit_power_law(theta, mu)
        assert isinstance(fit, PowerLawFit)
        assert fit.mu_0 == pytest.approx(target[0], abs=5e-3)
        assert fit.alpha == pytest.approx(target[1], rel=5e-2)
        assert fit.beta == pytest.approx(target[2], rel=5e-2)
        assert fit.r2 > 0.9999
        assert fit.max_abs_relative_error < 1e-2

    def test_fit_reports_error_metrics(self):
        """拟合输出含逐点残差、RMSE 与相对误差，且 R² 与手算一致。"""
        theta = np.array([5.0, 15.0, 25.0])
        mu = np.array([0.10, 0.20, 0.30])
        fit = fit_power_law(theta, mu)
        assert len(fit.mu_fitted) == 3
        assert fit.r2 == pytest.approx(r_squared(mu, fit.predict(theta)))
        assert fit.rms_relative_error == pytest.approx(
            float(np.sqrt(np.mean(relative_errors(mu, fit.predict(theta)) ** 2)))
        )
        assert fit.converged
        assert fit.iterations >= 1

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            fit_power_law(np.array([5.0, 10.0]), np.array([0.1, 0.2]))  # 点数不足
        with pytest.raises(ValueError):
            fit_power_law(np.array([0.0, 5.0, 10.0]), np.array([0.1, 0.2, 0.3]))
        with pytest.raises(ValueError):
            power_law_mu(-1.0, 0.1, 0.01, 1.0)


# ---------------------------------------------------------------------------
# 9. 参考数据
# ---------------------------------------------------------------------------


class TestReferenceData:
    def test_five_points_with_calibration_validation_split(self):
        table = reference_table()
        assert len(table) == 5
        assert reference_temperatures() == [-25.0, -20.0, -15.0, -10.0, -5.0]
        calibration = role_targets("calibration")
        validation = role_targets("validation")
        assert len(calibration) == 3 and len(validation) == 2
        assert calibration[float(-5.0)] == pytest.approx(0.09)
        # 参考值随冷端增大
        mus = [row["mu_reference"] for row in table]
        assert mus == sorted(mus)
        with pytest.raises(ValueError):
            role_targets("nonsense")

    def test_provenance_declares_limitations(self):
        """来源声明必须如实包含不确定性/局限（不许伪装成受控实验）。"""
        assert PROVENANCE["source_type"].startswith("compiled literature medians")
        assert PROVENANCE["limitations"]
        assert any("不可获取" in item for item in PROVENANCE["limitations"])
        table = reference_table()
        for row in table:
            assert 0.15 < row["relative_uncertainty"] < 0.4


# ---------------------------------------------------------------------------
# 10. 端到端 validation
# ---------------------------------------------------------------------------


class TestEndToEndValidation:
    def test_pipeline_meets_all_machine_criteria(self, tmp_path):
        result = run_validation(output_dir=tmp_path / "hif", make_figures=False)
        report = result.report
        # 判据
        assert report["criteria"]["criterion_a_power_law_r2"]["passed"]
        assert report["criteria"]["criterion_a_power_law_r2"]["value"] > CRITERIA["power_law_r2_min"]
        assert report["criteria"]["criterion_b_per_point_relative_error"]["passed"]
        assert report["criteria"]["criterion_c_convergence"]["passed"]
        assert report["verdict"]["all_machine_criteria_passed"]
        # 逐点误差显式记录
        rows = report["reference_comparison"]
        assert len(rows) == 5
        for row in rows:
            assert abs(row["relative_error"]) < CRITERIA["per_point_relative_error_max"]
            assert row["role"] in {"calibration", "validation"}
        # 产物 JSON 可读且无 NaN/Inf
        payload = json.loads((result.output_dir / "calibration_report.json").read_text())
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text
        assert payload["macro_fit"]["mu_0"] > 0.0
        assert payload["constitutive_identification"]["value_Pa"] > 0.0
        assert payload["experiment"]["n_repetitions"] == 10
        assert len(payload["experiment"]["points"]) == 5
        assert payload["convergence"]["fd_grid"]
        assert payload["convergence"]["ensemble_size"]
        assert payload["convergence"]["fixed_point"]["bisection"]

    def test_identification_monotone_objective(self, tmp_path):
        """p_real 辨识的目标函数在最优值两侧上升（一维搜索可辨识性）。"""
        config = FrictionTestConfig(n_repetitions=2, ensemble=AsperityEnsembleParameters(n_asperities=64))
        identification = identify_real_contact_pressure({-15.0: 0.21}, config)
        curve = identification.objective_curve
        assert len(curve) >= 5
        best = identification.real_contact_pressure_Pa
        values = {row["real_contact_pressure_Pa"]: row["objective_relative_sse"] for row in curve}
        assert min(values.values()) <= identification.objective + 1e-12
        assert 2.0e5 < best < 1.0e8