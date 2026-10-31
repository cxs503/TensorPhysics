"""标定：幂律 μ(T) 拟合 + 唯一本构参数（真实接触压力）辨识。

两级标定
--------
1. **本构参数辨识**（1 参数）：数值实验的干摩擦分支
   ``μ_dry = τ_s(θ)/p_real`` 中唯一不可由文献直接给定的量是真实接触压力
   ``p_real``（真实接触面积分数 A_r/A_n 由载荷与冰的蠕变决定）。用标定子集
   （-5 / -15 / -25 °C）上的相对残差平方和做目标函数，在 log p_real 上做
   黄金分割搜索。
2. **宏观拟合**（3 参数）：对标定+验证全部 5 个温度点的数值实验结果拟合

   .. math:: \\mu(T) = \\mu_0 + \\alpha\\,|T - T_{melt}|^{\\beta}

   采用自实现 Levenberg–Marquardt（不依赖 scipy），输出 R²、RMSE、逐点相对误差、
   迭代收敛历史与 JᵀJ 条件数。

所有参数化都把待定量映射到有界区间（sigmoid），因此 LM 迭代天然满足物理约束
（μ_0 ≥ 0、α > 0、β > 0）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .experiment import (
    FrictionTestConfig,
    build_ensemble_bank,
    evaluate_ensemble_bank,
)
from .ice_materials import DEFAULT_ICE, IceMaterialCorrelations

#: 参数边界：(下界, 上界)
_MU0_BOUNDS = (0.0, 0.5)
_ALPHA_BOUNDS = (1e-6, 0.5)
_BETA_BOUNDS = (0.05, 4.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def _unpack(x) -> tuple[float, float, float]:
    mu0 = _MU0_BOUNDS[0] + (_MU0_BOUNDS[1] - _MU0_BOUNDS[0]) * _sigmoid(x[0])
    alpha = _ALPHA_BOUNDS[0] + (_ALPHA_BOUNDS[1] - _ALPHA_BOUNDS[0]) * _sigmoid(x[1])
    beta = _BETA_BOUNDS[0] + (_BETA_BOUNDS[1] - _BETA_BOUNDS[0]) * _sigmoid(x[2])
    return float(mu0), float(alpha), float(beta)


def _pack(mu0: float, alpha: float, beta: float) -> np.ndarray:
    def inv(value, bounds):
        s = (value - bounds[0]) / (bounds[1] - bounds[0])
        return _logit(s)

    return np.array([inv(mu0, _MU0_BOUNDS), inv(alpha, _ALPHA_BOUNDS), inv(beta, _BETA_BOUNDS)])


def power_law_mu(theta, mu_0: float, alpha: float, beta: float):
    """幂律摩擦系数模型 μ(θ) = μ_0 + α θ^β（θ = |T - T_melt|）。"""
    theta = np.asarray(theta, dtype=float)
    if np.any(theta < 0.0):
        raise ValueError("undercooling must be non-negative")
    return mu_0 + alpha * np.power(theta, beta)


def r_squared(y_true, y_pred) -> float:
    """决定系数 R²（对 ``y_true`` 的均值取总平方和）。"""
    y = np.asarray(y_true, dtype=float)
    f = np.asarray(y_pred, dtype=float)
    if y.shape != f.shape:
        raise ValueError("shape mismatch")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if np.allclose(y, f) else 0.0
    ss_res = float(np.sum((y - f) ** 2))
    return 1.0 - ss_res / ss_tot


def relative_errors(y_true, y_pred):
    """逐点相对误差 ``(pred-true)/true``（无量纲，带符号）。"""
    y = np.asarray(y_true, dtype=float)
    f = np.asarray(y_pred, dtype=float)
    if np.any(y == 0.0):
        raise ValueError("reference values must be non-zero")
    return (f - y) / y


def rms_relative_error(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean(relative_errors(y_true, y_pred) ** 2)))


@dataclass
class PowerLawFit:
    """幂律拟合结果。"""

    mu_0: float
    alpha: float
    beta: float
    r2: float
    rmse: float
    rms_relative_error: float
    max_abs_relative_error: float
    undercooling_K: list[float] = field(default_factory=list)
    mu_measured: list[float] = field(default_factory=list)
    mu_fitted: list[float] = field(default_factory=list)
    relative_errors: list[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    objective_history: list[float] = field(default_factory=list)
    gradient_norm: float = 0.0
    jtj_condition_number: float = 0.0

    def predict(self, theta):
        return power_law_mu(theta, self.mu_0, self.alpha, self.beta)

    def as_dict(self) -> dict:
        return {
            "model": "mu(theta) = mu_0 + alpha * theta**beta",
            "mu_0": self.mu_0,
            "alpha": self.alpha,
            "beta": self.beta,
            "r2": self.r2,
            "rmse": self.rmse,
            "rms_relative_error": self.rms_relative_error,
            "max_abs_relative_error": self.max_abs_relative_error,
            "undercooling_K": self.undercooling_K,
            "mu_measured": self.mu_measured,
            "mu_fitted": self.mu_fitted,
            "relative_errors": self.relative_errors,
            "iterations": self.iterations,
            "converged": self.converged,
            "gradient_norm": self.gradient_norm,
            "jtj_condition_number": self.jtj_condition_number,
            "objective_history": self.objective_history,
        }


def fit_power_law(
    undercooling_K,
    mu_values,
    initial: tuple[float, float, float] | None = None,
    max_iterations: int = 300,
    tolerance: float = 1e-14,
) -> PowerLawFit:
    """Levenberg–Marquardt 拟合 μ(θ)=μ_0+αθ^β（自实现，仅依赖 numpy）。"""
    theta = np.asarray(undercooling_K, dtype=float)
    mu = np.asarray(mu_values, dtype=float)
    if theta.shape != mu.shape:
        raise ValueError("shape mismatch")
    if theta.size < 3:
        raise ValueError("need at least 3 points to fit 3 parameters")
    if np.any(theta <= 0.0):
        raise ValueError("undercooling must be positive for a power law")

    if initial is None:
        # 线性形式的初值：μ ≈ μ_0 + α θ（β=1）
        slope, intercept = np.polyfit(theta, mu, 1)
        mu0 = max(intercept, 1e-6)
        alpha = max(slope, 1e-6)
        beta = 1.0
    else:
        mu0, alpha, beta = initial
    x = _pack(mu0, alpha, beta)

    def residuals(xv):
        p0, pa, pb = _unpack(xv)
        return power_law_mu(theta, p0, pa, pb) - mu

    def jacobian(xv):
        jac = np.zeros((theta.size, xv.size))
        for k in range(xv.size):
            step = 1e-6 * max(1.0, abs(xv[k]))
            xp = xv.copy()
            xp[k] += step
            jac[:, k] = (residuals(xp) - residuals(xv)) / step
        return jac

    lam = 1e-3
    objective = float(np.sum(residuals(x) ** 2))
    history = [objective]
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        res = residuals(x)
        jac = jacobian(x)
        grad = jac.T @ res
        jtj = jac.T @ jac
        try:
            cond = float(np.linalg.cond(jtj))
        except np.linalg.LinAlgError:  # pragma: no cover - 极端退化矩阵
            cond = float("inf")
        step_matrix = jtj + lam * np.diag(np.maximum(np.diag(jtj), 1e-12))
        try:
            delta = np.linalg.solve(step_matrix, -grad)
        except np.linalg.LinAlgError:  # pragma: no cover
            lam *= 10.0
            continue
        x_new = x + delta
        res_new = residuals(x_new)
        objective_new = float(np.sum(res_new**2))
        if objective_new < objective:
            improvement = objective - objective_new
            x = x_new
            objective = objective_new
            history.append(objective)
            lam = max(lam * 0.3, 1e-12)
            if improvement <= tolerance * max(objective, 1e-30) or np.linalg.norm(delta) < 1e-12:
                converged = True
                break
        else:
            lam *= 10.0
            history.append(objective)
            if lam > 1e12:
                break

    mu0, alpha, beta = _unpack(x)
    fitted = power_law_mu(theta, mu0, alpha, beta)
    rel = relative_errors(mu, fitted)
    res = residuals(x)
    jac = jacobian(x)
    try:
        cond = float(np.linalg.cond(jac.T @ jac))
    except np.linalg.LinAlgError:  # pragma: no cover
        cond = float("inf")
    gradient_norm = float(np.linalg.norm(jac.T @ res))
    # 收敛判据：目标函数的相对改进已耗尽（数值 Jacobian 噪声量级），或梯度已足够小
    if not converged:
        scaled_gradient = gradient_norm / (1.0 + objective)
        if scaled_gradient < 1e-10 or objective <= tolerance:
            converged = True
    return PowerLawFit(
        mu_0=mu0,
        alpha=alpha,
        beta=beta,
        r2=r_squared(mu, fitted),
        rmse=float(np.sqrt(np.mean(res**2))),
        rms_relative_error=float(np.sqrt(np.mean(rel**2))),
        max_abs_relative_error=float(np.max(np.abs(rel))),
        undercooling_K=[float(v) for v in theta],
        mu_measured=[float(v) for v in mu],
        mu_fitted=[float(v) for v in fitted],
        relative_errors=[float(v) for v in rel],
        iterations=iterations,
        converged=converged,
        objective_history=history,
        gradient_norm=gradient_norm,
        jtj_condition_number=cond,
    )


def _golden_section_minimise(func, lower: float, upper: float, tol: float = 1e-4, max_iter: int = 200):
    """黄金分割一维极小化（返回最优点、最优值、迭代次数）。"""
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lower), float(upper)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = func(c), func(d)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        if abs(b - a) < tol * max(1.0, abs(a) + abs(b)):
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = func(d)
    x_best = 0.5 * (a + b)
    return x_best, float(func(x_best)), iterations


@dataclass
class ConstitutiveIdentification:
    """真实接触压力 p_real 的辨识结果。"""

    real_contact_pressure_Pa: float
    objective: float
    iterations: int
    calibration_temperatures_C: list[float]
    nominal_pressure_Pa: float = 1.0e6
    objective_curve: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "parameter": "real_contact_pressure_Pa",
            "value_Pa": self.real_contact_pressure_Pa,
            "real_area_fraction": self.nominal_pressure_Pa / self.real_contact_pressure_Pa,
            "objective_relative_sse": self.objective,
            "iterations": self.iterations,
            "calibration_temperatures_C": self.calibration_temperatures_C,
            "objective_curve": self.objective_curve,
        }


def identify_real_contact_pressure(
    calibration_targets_C: dict[float, float],
    config: FrictionTestConfig | None = None,
    bounds_Pa: tuple[float, float] = (2.0e5, 1.0e8),
    tolerance_log10: float = 1e-3,
) -> ConstitutiveIdentification:
    """在标定温度子集上辨识真实接触压力 ``p_real``。

    ``calibration_targets_C``: ``{temperature_C: mu_reference}``。目标函数为相对残差
    平方和；微凸体样本库只抽取一次（同一批样本），避免蒙特卡洛噪声干扰一维搜索。
    """
    config = config or FrictionTestConfig()
    temperatures = sorted(calibration_targets_C)
    bank = build_ensemble_bank(temperatures, config)
    targets = np.array([calibration_targets_C[t] for t in temperatures], dtype=float)
    curve: list[dict] = []

    def objective(log_p: float) -> float:
        p = float(np.exp(log_p))
        sub = config.replace(contact=config.contact.replace(real_contact_pressure_Pa=p))
        points = evaluate_ensemble_bank(bank, sub)
        mu = np.array([pt["mu_mean"] for pt in points], dtype=float)
        value = float(np.sum(((mu - targets) / targets) ** 2))
        curve.append({"real_contact_pressure_Pa": p, "objective_relative_sse": value})
        return value

    lo, hi = np.log(bounds_Pa[0]), np.log(bounds_Pa[1])
    best_log, best_value, iterations = _golden_section_minimise(
        objective, lo, hi, tol=tolerance_log10, max_iter=200
    )
    # 曲线去掉重复评估并排序，便于画图
    seen: dict[float, float] = {}
    for row in curve:
        seen.setdefault(round(row["real_contact_pressure_Pa"], 3), row["objective_relative_sse"])
    curve_sorted = [
        {"real_contact_pressure_Pa": p, "objective_relative_sse": v}
        for p, v in sorted(seen.items())
    ]
    return ConstitutiveIdentification(
        real_contact_pressure_Pa=float(np.exp(best_log)),
        objective=best_value,
        iterations=iterations,
        calibration_temperatures_C=[float(t) for t in temperatures],
        nominal_pressure_Pa=config.contact.nominal_pressure_Pa,
        objective_curve=curve_sorted,
    )


def compare_with_reference(
    points: list[dict],
    reference_mu: dict[float, float],
    salinity_psu: float = 0.0,
    material: IceMaterialCorrelations = DEFAULT_ICE,
) -> list[dict]:
    """数值实验 vs 参考数据逐点比对。"""
    rows: list[dict] = []
    for point in points:
        temperature = point["temperature_C"]
        if temperature not in reference_mu:
            continue
        ref = float(reference_mu[temperature])
        sim = float(point["mu_mean"])
        theta = float(material.undercooling(temperature, salinity_psu))
        rows.append(
            {
                "temperature_C": temperature,
                "undercooling_K": theta,
                "mu_simulation": sim,
                "mu_stderr": point.get("mu_stderr", 0.0),
                "mu_reference": ref,
                "relative_error": (sim - ref) / ref,
                "abs_relative_error": abs((sim - ref) / ref),
            }
        )
    return rows


__all__ = [
    "ConstitutiveIdentification",
    "PowerLawFit",
    "compare_with_reference",
    "fit_power_law",
    "identify_real_contact_pressure",
    "power_law_mu",
    "r_squared",
    "relative_errors",
    "rms_relative_error",
]