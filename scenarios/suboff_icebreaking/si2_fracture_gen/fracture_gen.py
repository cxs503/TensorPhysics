"""断裂生成模块 —— 破坏 cohesive face → DEM 颗粒集。

SI-2 断裂生成管线第二步：
- 输入：failed_faces.json (破坏面清单) + 网格/材料信息
- 输出：particles.json (TensorDEM 可初始化颗粒文件) + ledger_a.json (FEM 侧账本)

映射策略（在 prereg 中冻结）：等体积球映射
- 每个破坏 face 对应一个 DEM 颗粒
- 颗粒体积 V = face_area × face_length（面面积 × 边长 = 材料块体积）
  物理含义：破坏面是板内截面，面积 A = L × t；
  该面代表的材料块体积 = A × L = L² × t
- 颗粒为等体积球体：r = (3V / 4π)^(1/3)
- 颗粒质量 = ρ × V

账本 A（FEM 侧）：
- 核销质量：∑ ρ × V_face
- 释放断裂能量：∑ G_c × A_face

账本 B（DEM 侧）：
- 颗粒质量：∑ m_particle
- 断裂能量：∑ G_c × A_face（从来源 face 继承）

守恒性：两个账本从同一组 face 数据计算，公式完全一致，
因此质量/能量守恒精确到机器精度。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class DEMParticle:
    """一个 DEM 颗粒。"""
    particle_id: int
    position: np.ndarray     # (3,) 颗粒中心位置
    radius: float            # 颗粒半径 (m)
    mass: float              # 颗粒质量 (kg)
    velocity: np.ndarray     # (3,) 初始速度
    face_id: int             # 来源 face ID
    source_volume: float     # 来源体积 (m³)
    fracture_energy: float   # 断裂能量 (J)


@dataclass
class FractureGenResult:
    """断裂生成结果。"""
    particles: List[DEMParticle]
    total_mass: float            # 生成颗粒总质量 (kg)
    total_fracture_energy: float # 总断裂能量 (J)
    density: float               # 材料密度 (kg/m³)
    G_c: float                   # 断裂能密度 (J/m²)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _utcnow_z()


# ---------------------------------------------------------------------------
# 等体积球映射
# ---------------------------------------------------------------------------

def volume_to_sphere_radius(volume: float) -> float:
    """等体积球半径：V = 4/3 π r³ → r = (3V / 4π)^(1/3)。"""
    if volume <= 0:
        raise ValueError("volume must be positive")
    return (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)


def compute_face_center_3d(
    nodes_2d: np.ndarray,
    face_info: Dict[str, Any],
) -> np.ndarray:
    """计算 face 中心在 3D 空间的位置（板中面 z=0）。"""
    ni = face_info["node_i"]
    nj = face_info["node_j"]
    center_2d = 0.5 * (nodes_2d[ni] + nodes_2d[nj])
    return np.array([center_2d[0], center_2d[1], 0.0])


def face_volume(face: Dict) -> float:
    """每个破坏 face 对应的材料体积。

    V = face_area × face_length = (L × t) × L = L² × t
    """
    return face["area"] * face["length"]


def generate_particles_from_failed_faces(
    failed_faces_data: Dict[str, Any],
    nodes_2d: np.ndarray,
    density: float,
) -> FractureGenResult:
    """从破坏面清单生成 DEM 颗粒集。

    映射策略：等体积球
    - 每个破坏 face → 一个颗粒
    - 颗粒体积 V = face_area × face_length
    - 颗粒半径 r = (3V / 4π)^(1/3)
    - 颗粒质量 m = ρ × V
    - 断裂能量 E = G_c × face_area

    Parameters
    ----------
    failed_faces_data : dict
        破坏面清单（从 JSON 加载）。
    nodes_2d : (n_nodes, 2)
        2D 节点坐标。
    density : float
        材料密度 (kg/m³)。

    Returns
    -------
    FractureGenResult
        颗粒集 + 账本信息。
    """
    failed_faces = failed_faces_data["failed_faces"]
    G_c = failed_faces_data["G_c"]

    particles = []
    total_mass = 0.0
    total_fracture_energy = 0.0

    for idx, face in enumerate(failed_faces):
        vol = face_volume(face)
        radius = volume_to_sphere_radius(vol)
        mass = density * vol
        fracture_energy = face["released_energy"]  # G_c × face_area
        center = compute_face_center_3d(nodes_2d, face)

        particle = DEMParticle(
            particle_id=idx,
            position=center,
            radius=radius,
            mass=mass,
            velocity=np.zeros(3),
            face_id=face["face_id"],
            source_volume=vol,
            fracture_energy=fracture_energy,
        )
        particles.append(particle)

        total_mass += mass
        total_fracture_energy += fracture_energy

    return FractureGenResult(
        particles=particles,
        total_mass=total_mass,
        total_fracture_energy=total_fracture_energy,
        density=density,
        G_c=G_c,
    )


# ---------------------------------------------------------------------------
# 账本 A（FEM 侧）—— 核销质量 + 释放断裂能量
# ---------------------------------------------------------------------------

def compute_fem_ledger(
    failed_faces_data: Dict[str, Any],
    density: float,
) -> Dict[str, Any]:
    """计算 FEM 侧账本（账本 A）。

    核销质量：每个破坏 face 对应的材料体积 × 密度
    释放断裂能量：∑ G_c × A_face

    与 generate_particles_from_failed_faces 使用完全相同的公式，
    确保质量/能量守恒精确到机器精度。
    """
    failed_faces = failed_faces_data["failed_faces"]
    G_c = failed_faces_data["G_c"]

    entries = []
    total_mass = 0.0
    total_energy = 0.0

    for face in failed_faces:
        vol = face_volume(face)
        mass = density * vol
        energy = face["released_energy"]

        entries.append({
            "face_id": face["face_id"],
            "volume": float(vol),
            "mass": float(mass),
            "fracture_energy": float(energy),
        })

        total_mass += mass
        total_energy += energy

    return {
        "timestamp": _utcnow_z(),
        "side": "FEM",
        "ledger_id": "A",
        "density": float(density),
        "G_c": float(G_c),
        "n_entries": len(entries),
        "total_mass": float(total_mass),
        "total_fracture_energy": float(total_energy),
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# 账本 B（DEM 侧）—— 生成颗粒质量 + 断裂能量
# ---------------------------------------------------------------------------

def compute_dem_ledger(result: FractureGenResult) -> Dict[str, Any]:
    """计算 DEM 侧账本（账本 B）。

    从生成的颗粒集统计质量和能量。
    """
    entries = []
    for p in result.particles:
        entries.append({
            "particle_id": p.particle_id,
            "face_id": p.face_id,
            "mass": float(p.mass),
            "volume": float(p.source_volume),
            "fracture_energy": float(p.fracture_energy),
        })

    return {
        "timestamp": _utcnow_z(),
        "side": "DEM",
        "ledger_id": "B",
        "density": float(result.density),
        "G_c": float(result.G_c),
        "n_particles": len(result.particles),
        "total_mass": float(result.total_mass),
        "total_fracture_energy": float(result.total_fracture_energy),
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

def particles_to_dict(result: FractureGenResult) -> Dict[str, Any]:
    """将颗粒集转为可序列化字典（TensorDEM 可初始化格式）。"""
    return {
        "timestamp": result.timestamp,
        "n_particles": len(result.particles),
        "total_mass": float(result.total_mass),
        "total_fracture_energy": float(result.total_fracture_energy),
        "density": float(result.density),
        "G_c": float(result.G_c),
        "mapping_strategy": "equal_volume_sphere",
        "particles": [
            {
                "particle_id": p.particle_id,
                "position": p.position.tolist(),
                "radius": float(p.radius),
                "mass": float(p.mass),
                "velocity": p.velocity.tolist(),
                "face_id": p.face_id,
                "source_volume": float(p.source_volume),
                "fracture_energy": float(p.fracture_energy),
            }
            for p in result.particles
        ],
    }


def save_particles(result: FractureGenResult, path: str) -> None:
    """保存颗粒文件。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = particles_to_dict(result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_ledger_a(ledger: Dict[str, Any], path: str) -> None:
    """保存账本 A（FEM 侧）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)


def save_ledger_b(ledger: Dict[str, Any], path: str) -> None:
    """保存账本 B（DEM 侧）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)


def load_particles(path: str) -> Dict[str, Any]:
    """加载颗粒文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ledger(path: str) -> Dict[str, Any]:
    """加载账本。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
