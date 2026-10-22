"""浸入边界模块 —— 板挠度→LBM 流场边界修正（结构→流体反馈）。

SI-3 双向耦合管线第三步：
- 输入：板挠度场 (从 FEM 求解器获得)
- 输出：LBM 流场边界修正量（浸入边界力或速度修正）

物理模型：
- 冰板挠度改变流场边界几何
- 采用浸入边界法 (IBM) 或移动边界法处理流固交界面
- 板底面法向速度 = dw/dt（挠度时间导数）
- 简化模型：准静态假设，每耦合步更新边界位置

成熟度声明：Implemented 级，移植自 wedge S4 范式。
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Tuple


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ImmersedBoundaryData:
    """浸入边界数据。"""
    boundary_nodes: np.ndarray  # (N_boundary, 3) 边界节点位置
    boundary_velocity: np.ndarray  # (N_boundary, 3) 边界法向速度
    boundary_displacement: np.ndarray  # (N_boundary, 3) 边界位移
    timestamp: str


def compute_immersed_boundary_from_plate_deflection(
    plate_nodes: np.ndarray,
    plate_deflection: np.ndarray,
    plate_velocity: np.ndarray,
    fluid_nodes: np.ndarray,
    coupling_step: int,
    dt: float,
) -> Dict[str, Any]:
    """从板挠度计算浸入边界修正。

    Parameters
    ----------
    plate_nodes : ndarray (N_plate, 2)
        板节点 (x, y) 位置。
    plate_deflection : ndarray (N_plate,)
        板法向挠度 w（z 方向）。
    plate_velocity : ndarray (N_plate,)
        板法向速度 dw/dt。
    fluid_nodes : ndarray (N_fluid, 3)
        流体节点位置 (x, y, z)。
    coupling_step : int
        当前耦合步编号。
    dt : float
        耦合时间步长。

    Returns
    -------
    dict
        浸入边界数据（JSON 可序列化）。
    """
    n_plate = len(plate_nodes)
    
    # 构造 3D 板节点位置（z = 0 为初始位置）
    plate_nodes_3d = np.zeros((n_plate, 3), dtype=np.float64)
    plate_nodes_3d[:, 0] = plate_nodes[:, 0]  # x
    plate_nodes_3d[:, 1] = plate_nodes[:, 1]  # y
    plate_nodes_3d[:, 2] = plate_deflection  # z = w（挠度）
    
    # 边界速度（法向，z 方向）
    boundary_velocity = np.zeros((n_plate, 3), dtype=np.float64)
    boundary_velocity[:, 2] = plate_velocity  # v_z = dw/dt
    
    # 边界位移
    boundary_displacement = np.zeros((n_plate, 3), dtype=np.float64)
    boundary_displacement[:, 2] = plate_deflection  # u_z = w
    
    # 找流体节点中靠近板面的节点（用于施加边界条件）
    # 简化模型：找 z 坐标最接近板面的流体节点
    z_plate_mean = np.mean(plate_deflection)
    z_tolerance = 0.1 * (np.max(plate_deflection) - np.min(plate_deflection) + 1e-12)
    
    # 标记靠近板面的流体节点
    near_plate_mask = np.abs(fluid_nodes[:, 2] - z_plate_mean) < z_tolerance
    n_near_plate = int(np.sum(near_plate_mask))
    
    return {
        "ledger_type": "immersed_boundary",
        "description": "结构→流体反馈：板挠度→浸入边界修正",
        "schema_version": 1,
        "coupling_step": int(coupling_step),
        "n_boundary_nodes": int(n_plate),
        "boundary_nodes": plate_nodes_3d.tolist(),
        "boundary_velocity": boundary_velocity.tolist(),
        "boundary_displacement": boundary_displacement.tolist(),
        "n_fluid_nodes_near_plate": n_near_plate,
        "z_plate_mean": float(z_plate_mean),
        "z_tolerance": float(z_tolerance),
        "dt": float(dt),
        "timestamp": _utcnow_z(),
    }


def apply_immersed_boundary_to_lbm_snapshot(
    snapshot_path: str,
    immersed_boundary_data: Dict,
    output_path: str,
) -> str:
    """将浸入边界修正应用到 LBM 快照（生成修正后的快照）。

    Parameters
    ----------
    snapshot_path : str
        原始 LBM 快照路径。
    immersed_boundary_data : dict
        浸入边界数据。
    output_path : str
        输出修正后快照路径。

    Returns
    -------
    str
        输出快照路径。
    """
    data = np.load(snapshot_path)
    
    # 提取原始数据
    pressure = data["pressure"].copy()
    shear_x = data["shear_x"].copy()
    shear_y = data["shear_y"].copy()
    fluid_nodes = data["fluid_nodes"].copy()
    
    # 应用浸入边界修正
    # 简化模型：在靠近板面的流体节点上施加速度修正
    boundary_velocity = np.array(immersed_boundary_data["boundary_velocity"])
    z_plate_mean = immersed_boundary_data["z_plate_mean"]
    z_tolerance = immersed_boundary_data["z_tolerance"]
    
    # 找靠近板面的流体节点
    near_plate_mask = np.abs(fluid_nodes[:, 2] - z_plate_mean) < z_tolerance
    
    # 在这些节点上施加法向速度修正（影响压力和剪切）
    # 简化：压力修正 ∝ -ρ * v_boundary（伯努利近似）
    rho = 1000.0  # 水密度 (kg/m³)
    v_boundary_mean = np.mean(np.abs(boundary_velocity[:, 2]))
    
    # 压力修正：Δp = -ρ * v_z
    pressure_correction = -rho * boundary_velocity[:, 2]
    pressure[near_plate_mask] += np.mean(pressure_correction) * 0.1  # 松弛因子 0.1
    
    # 保存修正后快照
    np.savez(
        output_path,
        pressure=pressure,
        shear_x=shear_x,
        shear_y=shear_y,
        fluid_nodes=fluid_nodes,
    )
    
    return output_path


def save_immersed_boundary_data(data: Dict, output_path: str) -> None:
    """保存浸入边界数据到 JSON。"""
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_immersed_boundary_data(input_path: str) -> Dict:
    """从 JSON 加载浸入边界数据。"""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
