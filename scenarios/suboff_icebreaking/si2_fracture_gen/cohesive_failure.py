"""Cohesive 破坏判据模块 —— 从 FEM 板单元应力场判定内部 face 破坏。

SI-2 断裂生成管线第一步：
- 输入：SI-1 PlateResult (板位移/网格/材料)
- 输出：破坏面清单 (failed_faces)

物理模型：
- 对每个内部 face（两个相邻 Q4 单元的共享边），计算法向牵引力
- 若法向牵引力超过 cohesive 强度 σ_c，则该 face 破坏
- 破坏 face 释放能量 G_c × 面积（G_c 为断裂能密度）

判据：
- 最大主应力准则（简化）：σ_n > σ_c 即破坏
- 对解析锚：均匀应力场下可手算破坏 face 集合
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FailedFace:
    """一个破坏的 cohesive face。"""
    face_id: int               # 全局面编号
    node_i: int                # 边的起点节点
    node_j: int                # 边的终点节点
    element_left: int          # 左侧单元编号
    element_right: int         # 右侧单元编号
    length: float              # 边长度 (m)
    normal: np.ndarray         # 法向量 (2,)
    sigma_n: float             # 法向应力 (Pa)
    sigma_c: float             # cohesive 强度 (Pa)
    area: float                # 面面积 = length × thickness (m²)
    G: float                   # 释放能量密度 (J/m²) = G_c
    released_energy: float     # 释放能量 = G × area (J)


@dataclass
class CohesiveFailureResult:
    """Cohesive 破坏分析结果。"""
    failed_faces: List[FailedFace]
    n_total_faces: int          # 总内部 face 数
    n_failed: int               # 破坏 face 数
    sigma_c: float              # cohesive 强度
    G_c: float                  # 断裂能密度
    thickness: float            # 板厚度
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _utcnow_z()


# ---------------------------------------------------------------------------
# 网格拓扑：提取内部 face
# ---------------------------------------------------------------------------

def _element_edges(elements: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """提取每个单元的边。

    Returns
    -------
    list of (elem_id, node_i, node_j, edge_local_id)
        每个单元的 4 条边。
    """
    edges = []
    n_elem = len(elements)
    for e in range(n_elem):
        nodes = elements[e]
        # Q4 单元的 4 条边 (按节点顺序)
        for k in range(4):
            ni = int(nodes[k])
            nj = int(nodes[(k + 1) % 4])
            edges.append((e, ni, nj, k))
    return edges


def extract_internal_faces(
    elements: np.ndarray,
) -> List[Dict[str, Any]]:
    """提取网格的内部 face（两个单元共享的边）。

    Returns
    -------
    list of dict
        每个内部 face 的信息：
        - face_id: 全局编号
        - node_i, node_j: 边的两个端点（小节点号在前）
        - element_left, element_right: 左右单元编号
        - edge_length: 边长度（需后续用节点坐标计算）
    """
    # 建立边→单元的映射
    edge_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    n_elem = len(elements)

    for e in range(n_elem):
        nodes = elements[e]
        for k in range(4):
            ni = int(nodes[k])
            nj = int(nodes[(k + 1) % 4])
            # 规范化：小节点号在前
            edge_key = (min(ni, nj), max(ni, nj))
            if edge_key not in edge_map:
                edge_map[edge_key] = []
            edge_map[edge_key].append((e, k))

    # 提取内部 face（恰好被两个单元共享的边）
    faces = []
    face_id = 0
    for (ni, nj), elem_list in sorted(edge_map.items()):
        if len(elem_list) == 2:
            faces.append({
                "face_id": face_id,
                "node_i": ni,
                "node_j": nj,
                "element_left": elem_list[0][0],
                "edge_local_left": elem_list[0][1],
                "element_right": elem_list[1][0],
                "edge_local_right": elem_list[1][1],
            })
            face_id += 1

    return faces


# ---------------------------------------------------------------------------
# 应力计算
# ---------------------------------------------------------------------------

def compute_element_stress(
    nodes: np.ndarray,
    elements: np.ndarray,
    displacement: np.ndarray,
    young: float,
    poisson: float,
    thickness: float,
) -> np.ndarray:
    """计算每个 Q4 板单元的中心应力（弯曲 + 剪切）。

    对于 Mindlin 板，弯曲应力在单元中心（z = ±t/2）为：
    σ_x = -E/(1-ν²) * (t/2) * (κ_x + ν*κ_y)
    σ_y = -E/(1-ν²) * (t/2) * (κ_y + ν*κ_x)
    τ_xy = -E/(2(1+ν)) * (t/2) * κ_xy * 2 = -G*t*κ_xy

    简化：返回单元中心的等效膜应力（弯曲应力在表面最大值）。

    Parameters
    ----------
    nodes : (n_nodes, 2) 节点坐标
    elements : (n_elem, 4) 单元连接
    displacement : (n_dof,) 位移向量 [w, θx, θy] per node
    young, poisson, thickness : 材料参数

    Returns
    -------
    stress : (n_elem, 3) 单元中心应力 [σ_x, σ_y, τ_xy]
    """
    from scenarios.suboff_icebreaking.si1_one_way.plate_response import _shape

    E, nu, t = young, poisson, thickness
    n_elem = len(elements)
    stress = np.zeros((n_elem, 3))

    # 弯曲刚度矩阵
    Db = E * t**3 / (12 * (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])

    for e in range(n_elem):
        elem_nodes = elements[e]
        xy = nodes[elem_nodes]

        # 单元中心 (ξ=0, η=0)
        _, dN_nat = _shape(0.0, 0.0)
        J = dN_nat.T @ xy
        detJ = np.linalg.det(J)
        dN = dN_nat @ np.linalg.inv(J).T

        # 弯曲应变（曲率）
        kappa = np.zeros(3)
        for i in range(4):
            node_idx = elem_nodes[i]
            theta_x = displacement[3 * node_idx + 1]
            theta_y = displacement[3 * node_idx + 2]
            kappa[0] += dN[i, 0] * theta_x  # κ_x = dθx/dx
            kappa[1] += dN[i, 1] * theta_y  # κ_y = dθy/dy
            kappa[2] += dN[i, 1] * theta_x + dN[i, 0] * theta_y  # κ_xy

        # 弯矩
        M = Db @ kappa  # [Mx, My, Mxy]

        # 表面应力（z = t/2）：σ = M * z / (t³/12) = 6M/t²
        # 等效膜应力
        stress[e, 0] = 6 * M[0] / t**2   # σ_x
        stress[e, 1] = 6 * M[1] / t**2   # σ_y
        stress[e, 2] = 6 * M[2] / t**2   # τ_xy

    return stress


def compute_face_traction(
    face: Dict[str, Any],
    nodes: np.ndarray,
    element_stress: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """计算 face 上的法向牵引力。

    简化模型：取 face 两侧单元应力的平均值，投影到 face 法向。

    Parameters
    ----------
    face : dict
        内部 face 信息。
    nodes : (n_nodes, 2)
        节点坐标。
    element_stress : (n_elem, 3)
        单元应力 [σ_x, σ_y, τ_xy]。

    Returns
    -------
    sigma_n : float
        法向应力（正值=拉伸）。
    normal : (2,) ndarray
        单位法向量。
    """
    ni, nj = face["node_i"], face["node_j"]
    pi = nodes[ni]
    pj = nodes[nj]

    # 边方向
    edge_vec = pj - pi
    edge_len = np.linalg.norm(edge_vec)
    tangent = edge_vec / edge_len

    # 法向量（逆时针旋转 90°）
    normal = np.array([-tangent[1], tangent[0]])

    # 两侧单元应力平均
    e_left = face["element_left"]
    e_right = face["element_right"]
    stress_avg = 0.5 * (element_stress[e_left] + element_stress[e_right])

    sigma_x, sigma_y, tau_xy = stress_avg

    # 法向应力：σ_n = n · σ · n = σx*nx² + σy*ny² + 2*τxy*nx*ny
    nx, ny = normal
    sigma_n = sigma_x * nx**2 + sigma_y * ny**2 + 2 * tau_xy * nx * ny

    return sigma_n, normal


# ---------------------------------------------------------------------------
# 破坏判据
# ---------------------------------------------------------------------------

def identify_failed_faces(
    nodes: np.ndarray,
    elements: np.ndarray,
    displacement: np.ndarray,
    young: float,
    poisson: float,
    thickness: float,
    sigma_c: float,
    G_c: float,
) -> CohesiveFailureResult:
    """识别所有破坏的 cohesive face。

    Parameters
    ----------
    nodes : (n_nodes, 2)
        节点坐标。
    elements : (n_elem, 4)
        单元连接。
    displacement : (n_dof,)
        位移向量。
    young, poisson, thickness : float
        材料参数。
    sigma_c : float
        Cohesive 强度 (Pa)。
    G_c : float
        断裂能密度 (J/m²)。

    Returns
    -------
    CohesiveFailureResult
        破坏分析结果。
    """
    # 计算单元应力
    element_stress = compute_element_stress(
        nodes, elements, displacement, young, poisson, thickness
    )

    # 提取内部 face
    internal_faces = extract_internal_faces(elements)

    # 逐个判定
    failed_faces = []
    for face_info in internal_faces:
        sigma_n, normal = compute_face_traction(face_info, nodes, element_stress)

        # 判据：法向应力绝对值超过 cohesive 强度（拉伸或压缩均可破坏）
        if abs(sigma_n) > sigma_c:
            ni, nj = face_info["node_i"], face_info["node_j"]
            edge_len = np.linalg.norm(nodes[nj] - nodes[ni])
            face_area = edge_len * thickness  # 面面积 (m²)

            failed = FailedFace(
                face_id=face_info["face_id"],
                node_i=ni,
                node_j=nj,
                element_left=face_info["element_left"],
                element_right=face_info["element_right"],
                length=edge_len,
                normal=normal,
                sigma_n=sigma_n,
                sigma_c=sigma_c,
                area=face_area,
                G=G_c,
                released_energy=G_c * face_area,
            )
            failed_faces.append(failed)

    return CohesiveFailureResult(
        failed_faces=failed_faces,
        n_total_faces=len(internal_faces),
        n_failed=len(failed_faces),
        sigma_c=sigma_c,
        G_c=G_c,
        thickness=thickness,
    )


# ---------------------------------------------------------------------------
# 解析应力场下的破坏判定（用于锚3）
# ---------------------------------------------------------------------------

def identify_failed_faces_uniform_stress(
    nodes: np.ndarray,
    elements: np.ndarray,
    uniform_stress: np.ndarray,
    thickness: float,
    sigma_c: float,
    G_c: float,
) -> CohesiveFailureResult:
    """在均匀应力场下识别破坏 face（解析锚专用）。

    给定均匀应力 [σ_x, σ_y, τ_xy]，直接计算每个 face 的法向应力。
    无 FEM 位移→应力转换，消除数值误差。

    Parameters
    ----------
    nodes : (n_nodes, 2)
        节点坐标。
    elements : (n_elem, 4)
        单元连接。
    uniform_stress : (3,)
        均匀应力 [σ_x, σ_y, τ_xy]。
    thickness : float
        板厚度。
    sigma_c : float
        Cohesive 强度。
    G_c : float
        断裂能密度。

    Returns
    -------
    CohesiveFailureResult
        破坏分析结果。
    """
    # 构造均匀应力场（所有单元相同）
    n_elem = len(elements)
    element_stress = np.tile(uniform_stress, (n_elem, 1))

    # 提取内部 face
    internal_faces = extract_internal_faces(elements)

    # 逐个判定
    failed_faces = []
    for face_info in internal_faces:
        sigma_n, normal = compute_face_traction(face_info, nodes, element_stress)

        # 判据：法向应力绝对值超过 cohesive 强度（拉伸或压缩均可破坏）
        if abs(sigma_n) > sigma_c:
            ni, nj = face_info["node_i"], face_info["node_j"]
            edge_len = np.linalg.norm(nodes[nj] - nodes[ni])
            face_area = edge_len * thickness

            failed = FailedFace(
                face_id=face_info["face_id"],
                node_i=ni,
                node_j=nj,
                element_left=face_info["element_left"],
                element_right=face_info["element_right"],
                length=edge_len,
                normal=normal,
                sigma_n=sigma_n,
                sigma_c=sigma_c,
                area=face_area,
                G=G_c,
                released_energy=G_c * face_area,
            )
            failed_faces.append(failed)

    return CohesiveFailureResult(
        failed_faces=failed_faces,
        n_total_faces=len(internal_faces),
        n_failed=len(failed_faces),
        sigma_c=sigma_c,
        G_c=G_c,
        thickness=thickness,
    )


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

def failed_faces_to_dict(result: CohesiveFailureResult) -> Dict[str, Any]:
    """将破坏结果转为可序列化字典。"""
    return {
        "timestamp": result.timestamp,
        "n_total_faces": result.n_total_faces,
        "n_failed": result.n_failed,
        "sigma_c": result.sigma_c,
        "G_c": result.G_c,
        "thickness": result.thickness,
        "failed_faces": [
            {
                "face_id": f.face_id,
                "node_i": f.node_i,
                "node_j": f.node_j,
                "element_left": f.element_left,
                "element_right": f.element_right,
                "length": float(f.length),
                "normal": f.normal.tolist(),
                "sigma_n": float(f.sigma_n),
                "sigma_c": float(f.sigma_c),
                "area": float(f.area),
                "G": float(f.G),
                "released_energy": float(f.released_energy),
            }
            for f in result.failed_faces
        ],
    }


def save_failed_faces(result: CohesiveFailureResult, path: str) -> None:
    """保存破坏面清单到 JSON 文件。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = failed_faces_to_dict(result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_failed_faces(path: str) -> Dict[str, Any]:
    """加载破坏面清单。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
