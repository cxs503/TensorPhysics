"""守恒账本模块 —— 双向耦合的动量/能量双账本（移植自 wedge S4）。

SI-3 双向耦合管线第四步：
- 输入：每耦合步的流体载荷、结构响应、反力
- 输出：守恒账本（动量残差 + 能量残差）

判据（S4 同标尺）：
- 动量守恒：|流体冲量 − 结构动量变化 − 反力冲量| / scale ≤ 1e-12
- 能量守恒：|流体做功 − 结构应变能 − 结构动能| / scale ≤ 1e-12

成熟度声明：Implemented 级，移植自 wedge S4 范式。
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ConservationLedger:
    """守恒账本。"""
    coupling_step: int
    momentum_residual: float
    energy_residual: float
    passed: bool
    timestamp: str


def compute_momentum_balance(
    fluid_impulse: np.ndarray,
    struct_momentum_change: np.ndarray,
    reaction_impulse: np.ndarray,
) -> Dict[str, Any]:
    """计算动量守恒残差。

    动量守恒：流体冲量 = 结构动量变化 + 反力冲量
    残差 = |流体冲量 − 结构动量变化 − 反力冲量| / scale

    Parameters
    ----------
    fluid_impulse : ndarray (3,)
        流体给结构的冲量 ∫F_fluid dt。
    struct_momentum_change : ndarray (3,)
        结构动量变化 Δp = M * Δv。
    reaction_impulse : ndarray (3,)
        固定端反力冲量 ∫R dt。

    Returns
    -------
    dict
        动量账本数据。
    """
    residual_vector = fluid_impulse - struct_momentum_change - reaction_impulse
    residual_magnitude = float(np.linalg.norm(residual_vector))
    
    # 归一化尺度
    scale = float(max(
        np.linalg.norm(fluid_impulse),
        np.linalg.norm(struct_momentum_change),
        np.linalg.norm(reaction_impulse),
        1e-12
    ))
    
    relative_residual = float(residual_magnitude / scale)
    
    return {
        "fluid_impulse": fluid_impulse.tolist(),
        "struct_momentum_change": struct_momentum_change.tolist(),
        "reaction_impulse": reaction_impulse.tolist(),
        "residual_vector": residual_vector.tolist(),
        "residual_magnitude": residual_magnitude,
        "scale": scale,
        "relative_residual": relative_residual,
    }


def compute_energy_balance(
    fluid_work: float,
    struct_strain_energy_change: float,
    struct_kinetic_energy_change: float,
) -> Dict[str, Any]:
    """计算能量守恒残差。

    能量守恒：流体做功 = 结构应变能变化 + 结构动能变化
    残差 = |流体做功 − 结构应变能变化 − 结构动能变化| / scale

    Parameters
    ----------
    fluid_work : float
        流体对结构做功 ∫F_fluid · v_struct dt。
    struct_strain_energy_change : float
        结构应变能变化 ΔU。
    struct_kinetic_energy_change : float
        结构动能变化 ΔK。

    Returns
    -------
    dict
        能量账本数据。
    """
    residual = fluid_work - struct_strain_energy_change - struct_kinetic_energy_change
    residual_abs = abs(residual)
    
    # 归一化尺度
    scale = max(
        abs(fluid_work),
        abs(struct_strain_energy_change),
        abs(struct_kinetic_energy_change),
        1e-12
    )
    
    relative_residual = residual_abs / scale
    
    return {
        "fluid_work_J": float(fluid_work),
        "struct_strain_energy_change_J": float(struct_strain_energy_change),
        "struct_kinetic_energy_change_J": float(struct_kinetic_energy_change),
        "residual_J": float(residual),
        "residual_abs_J": float(residual_abs),
        "scale_J": float(scale),
        "relative_residual": float(relative_residual),
    }


def compute_conservation_ledger(
    coupling_step: int,
    fluid_force: np.ndarray,
    struct_displacement: np.ndarray,
    struct_strain_energy: float,
    reaction_force: np.ndarray,
    dt: float,
    fluid_force_prev: np.ndarray = None,
    struct_displacement_prev: np.ndarray = None,
    struct_strain_energy_prev: float = 0.0,
) -> Dict[str, Any]:
    """计算单步守恒账本（准静态力平衡 + 能量增量一致性）。

    Parameters
    ----------
    coupling_step : int
        耦合步编号。
    fluid_force : ndarray (3,)
        当前步流体载荷（总力）。
    struct_displacement : ndarray (3,)
        当前步结构位移（参考点）。
    struct_strain_energy : float
        当前步结构应变能。
    reaction_force : ndarray (3,)
        当前步固定端反力。
    dt : float
        时间步长。
    fluid_force_prev : ndarray (3,), optional
        上一步流体载荷。
    struct_displacement_prev : ndarray (3,), optional
        上一步结构位移。
    struct_strain_energy_prev : float
        上一步结构应变能。

    Returns
    -------
    dict
        守恒账本数据。
    """
    # === 准静态力平衡：F_fluid + R = 0 ===
    force_balance = fluid_force + reaction_force
    force_residual = float(np.linalg.norm(force_balance))
    force_scale = float(max(
        np.linalg.norm(fluid_force),
        np.linalg.norm(reaction_force),
        1e-12
    ))
    momentum_residual = force_residual / force_scale
    
    # === 能量增量一致性 ===
    # 对于准静态线性弹性系统：ΔU_strain = W_ext_incremental
    # 增量外力功用梯形法则：W = 0.5 * (F_k + F_{k-1}) · (u_k - u_{k-1})
    if fluid_force_prev is not None and struct_displacement_prev is not None:
        du = struct_displacement - struct_displacement_prev
        f_avg = 0.5 * (fluid_force + fluid_force_prev)
        incremental_work = float(np.dot(f_avg, du))
        delta_strain_energy = struct_strain_energy - struct_strain_energy_prev
        energy_residual_abs = abs(delta_strain_energy - incremental_work)
        energy_scale = float(max(
            abs(delta_strain_energy),
            abs(incremental_work),
            1e-12
        ))
        energy_residual = energy_residual_abs / energy_scale
    else:
        # 第一步：无增量信息，跳过能量检查
        energy_residual = 0.0
    
    # 判据（S4 同标尺：1e-12）
    momentum_passed = bool(momentum_residual <= 1e-12)
    energy_passed = bool(energy_residual <= 1e-12)
    passed = momentum_passed and energy_passed
    
    return {
        "ledger_type": "conservation",
        "description": "双向耦合守恒账本（力平衡 + 能量增量一致性）",
        "schema_version": 1,
        "coupling_step": int(coupling_step),
        "momentum_balance": {
            "F_fluid": fluid_force.tolist(),
            "R": reaction_force.tolist(),
            "force_balance": force_balance.tolist(),
            "residual": float(force_residual),
            "scale": float(force_scale),
            "relative_residual": float(momentum_residual),
        },
        "energy_balance": {
            "U_strain_J": float(struct_strain_energy),
            "delta_U_strain_J": float(struct_strain_energy - struct_strain_energy_prev) if fluid_force_prev is not None else 0.0,
            "incremental_work_J": float(incremental_work) if fluid_force_prev is not None else 0.0,
            "residual_J": float(energy_residual_abs) if fluid_force_prev is not None else 0.0,
            "scale_J": float(energy_scale) if fluid_force_prev is not None else 1.0,
            "relative_residual": float(energy_residual),
        },
        "momentum_passed": momentum_passed,
        "energy_passed": energy_passed,
        "passed": passed,
        "timestamp": _utcnow_z(),
    }


def save_conservation_ledger(ledger: Dict, output_path: str) -> None:
    """保存守恒账本到 JSON。"""
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)


def load_conservation_ledger(input_path: str) -> Dict:
    """从 JSON 加载守恒账本。"""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_conservation_ledgers(ledgers: List[Dict]) -> Dict[str, Any]:
    """聚合多步守恒账本。

    Parameters
    ----------
    ledgers : list of dict
        多步守恒账本列表。

    Returns
    -------
    dict
        聚合统计。
    """
    if not ledgers:
        return {"n_steps": 0, "all_passed": False}
    
    momentum_residuals = [l["momentum_balance"]["relative_residual"] for l in ledgers]
    energy_residuals = [l["energy_balance"]["relative_residual"] for l in ledgers]
    
    max_momentum = max(momentum_residuals)
    max_energy = max(energy_residuals)
    mean_momentum = np.mean(momentum_residuals)
    mean_energy = np.mean(energy_residuals)
    
    all_passed = all(l["passed"] for l in ledgers)
    
    return {
        "n_steps": len(ledgers),
        "all_passed": all_passed,
        "momentum_max_residual": max_momentum,
        "momentum_mean_residual": float(mean_momentum),
        "energy_max_residual": max_energy,
        "energy_mean_residual": float(mean_energy),
        "criterion_momentum": 1e-12,
        "criterion_energy": 1e-12,
    }
