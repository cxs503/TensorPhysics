"""LBM 流载提取模块 —— 从 TensorLBM 流场快照提取冰板底面节点载荷。

SI-1 单向耦合管线第一步：
- 输入：TensorLBM 流场快照 (npz 格式，含压力/剪切/节点位置)
- 输出：ledger_a.json (账本 A：LBM 侧发出的节点力)

物理模型：
- 从流场压力场和剪切应力场积分得到作用在冰板底面的节点力
- 力向量按节点组织，仅保留法向分量 (fz)——冰板为薄板，
  面内力 (fx, fy) 对弯曲响应贡献可忽略，但账本仍记录全向量
- 节点力 = 压力 × 节点代表面积 (集中力模型)
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 核心提取逻辑
# ---------------------------------------------------------------------------

def extract_nodal_forces_from_lbm_snapshot(
    snapshot_path: str,
    plate_nodes: np.ndarray,
    plate_area_per_node: np.ndarray,
) -> Dict[str, Any]:
    """从 LBM 流场快照提取冰板底面节点力。

    Parameters
    ----------
    snapshot_path : str
        LBM 流场快照文件路径 (npz 格式)，应包含：
        - 'pressure': 压力场 (N_fluid,)
        - 'shear_x', 'shear_y': 剪切应力场 (N_fluid,)
        - 'fluid_nodes': 流体节点位置 (N_fluid, 3)
    plate_nodes : ndarray (N_plate, 3)
        冰板底面节点位置。
    plate_area_per_node : ndarray (N_plate,)
        每个节点代表的 tributary 面积。

    Returns
    -------
    dict
        账本 A 数据结构。
    """
    data = np.load(snapshot_path)
    pressure = data["pressure"]
    shear_x = data["shear_x"]
    shear_y = data["shear_y"]
    fluid_nodes = data["fluid_nodes"]

    n_plate = len(plate_nodes)
    forces = np.zeros((n_plate, 3), dtype=np.float64)

    # 对每个板节点，找最近流体节点并提取应力
    for i in range(n_plate):
        distances = np.linalg.norm(fluid_nodes - plate_nodes[i], axis=1)
        nearest = int(np.argmin(distances))
        area = float(plate_area_per_node[i])
        forces[i, 0] = shear_x[nearest] * area   # fx: 面内剪切
        forces[i, 1] = shear_y[nearest] * area   # fy: 面内剪切
        forces[i, 2] = -pressure[nearest] * area  # fz: 法向压力 (向下为负)

    total_force = forces.sum(axis=0).tolist()

    return {
        "ledger_type": "A",
        "description": "LBM 侧发出的节点力",
        "schema_version": 1,
        "n_nodes": int(n_plate),
        "nodes": plate_nodes.tolist(),
        "forces": forces.tolist(),
        "total_force": total_force,
        "source_file": str(snapshot_path),
        "source_sha256": _file_sha256(snapshot_path),
        "timestamp": _utcnow_z(),
    }


# ---------------------------------------------------------------------------
# 合成快照生成 (测试/验证用)
# ---------------------------------------------------------------------------

def generate_synthetic_lbm_snapshot(
    output_path: str,
    plate_nodes: np.ndarray,
    load_case: str = "uniform",
    *,
    pressure_magnitude: float = 1000.0,
    seed: int = 42,
) -> str:
    """生成合成 LBM 流场快照用于测试。

    Parameters
    ----------
    output_path : str
        输出 npz 路径。
    plate_nodes : ndarray (N, 3)
        板节点位置。
    load_case : str
        'uniform' | 'gaussian' | 'point' | 'sinusoidal'
    pressure_magnitude : float
        特征压力幅值 (Pa)。
    seed : int
        随机种子 (用于流体节点微扰)。

    Returns
    -------
    str
        生成的快照路径。
    """
    rng = np.random.default_rng(seed)
    n_plate = len(plate_nodes)
    n_fluid = n_plate * 2

    # 流体节点：板节点上方微扰
    fluid_nodes = np.tile(plate_nodes, (2, 1))
    fluid_nodes[:, 2] += rng.uniform(0.01, 0.05, n_fluid)

    p0 = pressure_magnitude
    if load_case == "uniform":
        pressure = np.full(n_fluid, p0)
        shear_x = np.full(n_fluid, p0 * 0.01)
        shear_y = np.full(n_fluid, p0 * 0.005)
    elif load_case == "gaussian":
        center = plate_nodes.mean(axis=0)
        r2 = np.sum((fluid_nodes[:, :2] - center[:2]) ** 2, axis=1)
        envelope = np.exp(-r2 / 0.1)
        pressure = p0 * 2.0 * envelope
        shear_x = p0 * 0.02 * envelope
        shear_y = p0 * 0.01 * envelope
    elif load_case == "point":
        pressure = np.zeros(n_fluid)
        pressure[n_fluid // 2] = p0 * 10.0
        shear_x = np.zeros(n_fluid)
        shear_y = np.zeros(n_fluid)
    elif load_case == "sinusoidal":
        # q = q0 * sin(pi*x/a) * sin(pi*y/b) — 用于与 Navier 精确解对比
        xmax = plate_nodes[:, 0].max()
        ymax = plate_nodes[:, 1].max()
        pressure = p0 * np.sin(np.pi * fluid_nodes[:, 0] / xmax) * \
                   np.sin(np.pi * fluid_nodes[:, 1] / ymax)
        shear_x = np.zeros(n_fluid)
        shear_y = np.zeros(n_fluid)
    else:
        raise ValueError(f"Unknown load case: {load_case}")

    np.savez(output_path,
             pressure=pressure, shear_x=shear_x, shear_y=shear_y,
             fluid_nodes=fluid_nodes)
    return output_path


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

def save_ledger_a(ledger_a: Dict, output_path: str) -> None:
    """保存账本 A 到 JSON 文件。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ledger_a, f, indent=2, ensure_ascii=False)


def load_ledger_a(input_path: str) -> Dict:
    """加载账本 A。"""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
