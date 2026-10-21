"""FEM 冰板响应求解器 —— 接收节点载荷驱动 TensorFEM 弹性地基板求解。

SI-1 单向耦合管线第二步：
- 输入：ledger_a.json (节点载荷)
- 输出：ledger_b.json (账本 B：FEM 收到的节点力) + 板挠度场

物理模型：
- Mindlin-Reissner Q4 板单元 (选择性减缩积分 SRI)
- Winkler 弹性地基 (k_f 地基系数)
- 简支边界条件
- 线弹性小变形假设
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Mindlin-Reissner Q4 板单元 (从 TensorFEM plate.py 移植)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlateResult:
    """板求解结果。"""
    displacement: np.ndarray  # (ndof,)
    reaction: np.ndarray      # (ndof,)
    stiffness: np.ndarray     # (ndof, ndof)
    force: np.ndarray         # (ndof,)
    nodes: np.ndarray         # (n_nodes, 2)
    n_elements: int


def _shape(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Q4 形函数及其自然坐标导数。"""
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])
    dN = 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],
        [(1 - eta), -(1 + xi)],
        [(1 + eta), (1 + xi)],
        [-(1 + eta), (1 - xi)]
    ])
    return N, dN


def q4_mindlin_stiffness(
    xy: np.ndarray,
    young: float,
    poisson: float,
    thickness: float,
    shear_correction: float = 5.0 / 6.0,
) -> np.ndarray:
    """Q4 Mindlin 板单元 12x12 刚度矩阵 (SRI)。"""
    if xy.shape != (4, 2):
        raise ValueError("xy must have shape (4, 2)")
    
    E, nu, t = young, poisson, thickness
    if t <= 0 or nu <= -1 or nu >= 0.5:
        raise ValueError("invalid plate material or thickness")
    
    # 弯曲刚度矩阵 Db
    Db = E * t**3 / (12 * (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    
    # 剪切刚度矩阵 Ds
    G = E / (2 * (1 + nu))
    Ds = shear_correction * G * t * np.eye(2)
    
    ke = np.zeros((12, 12))
    gp = 1.0 / 3.0**0.5
    
    # 2x2 Gauss 积分 (弯曲)
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            _, dN_nat = _shape(xif, etaf)
            J = dN_nat.T @ xy
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError("plate element has non-positive Jacobian")
            dN = dN_nat @ np.linalg.inv(J).T
            
            # 弯曲应变-位移矩阵 Bb
            Bb = np.zeros((3, 12))
            for i in range(4):
                Bb[0, 3*i + 1] = dN[i, 0]  # kappa_x = d(theta_x)/dx
                Bb[1, 3*i + 2] = dN[i, 1]  # kappa_y = d(theta_y)/dy
                Bb[2, 3*i + 1] = dN[i, 1]  # kappa_xy
                Bb[2, 3*i + 2] = dN[i, 0]
            
            ke += Bb.T @ Db @ Bb * detJ
    
    # 1 点 Gauss 积分 (剪切，选择性减缩)
    N, dN_nat = _shape(0.0, 0.0)
    J = dN_nat.T @ xy
    detJ = np.linalg.det(J)
    dN = dN_nat @ np.linalg.inv(J).T
    
    Bs = np.zeros((2, 12))
    for i in range(4):
        Bs[0, 3*i] = dN[i, 0]      # gamma_xz = dw/dx + theta_y
        Bs[0, 3*i + 1] = N[i]
        Bs[1, 3*i] = dN[i, 1]      # gamma_yz = dw/dy + theta_x
        Bs[1, 3*i + 2] = N[i]
    
    ke += Bs.T @ Ds @ Bs * detJ * 4.0  # 权重=4
    
    return ke


def q4_winkler_foundation_stiffness(
    xy: np.ndarray,
    foundation_coefficient: float,
) -> np.ndarray:
    """Q4 Winkler 地基 12x12 刚度矩阵。"""
    if xy.shape != (4, 2):
        raise ValueError("xy must have shape (4, 2)")
    
    k = foundation_coefficient
    kw = np.zeros((12, 12))
    if k == 0.0:
        return kw
    
    gp = 1.0 / 3.0**0.5
    kw_4x4 = np.zeros((4, 4))
    
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            N, dN_nat = _shape(xif, etaf)
            J = dN_nat.T @ xy
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError("plate element has non-positive Jacobian")
            kw_4x4 += np.outer(N, N) * detJ
    
    kw_4x4 *= k
    
    # 仅填充 w-DOF (0, 3, 6, 9)
    w_dofs = [0, 3, 6, 9]
    for i, di in enumerate(w_dofs):
        for j, dj in enumerate(w_dofs):
            kw[di, dj] = kw_4x4[i, j]
    
    return kw


def structured_square_mesh(n: int, side: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """生成 n x n 结构化四边形网格。"""
    x = np.linspace(0, side, n + 1)
    y = np.linspace(0, side, n + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.flatten(), yy.flatten()])
    
    elements = []
    for j in range(n):
        for i in range(n):
            n0 = j * (n + 1) + i
            elements.append([n0, n0 + 1, n0 + n + 2, n0 + n + 1])
    
    return nodes, np.array(elements, dtype=np.int64)


# ---------------------------------------------------------------------------
# Hetényi 解析参考解
# ---------------------------------------------------------------------------

def hetenyi_infinite_beam_deflection(
    x: np.ndarray,
    P: float,
    lam: float,
    k_b: float,
) -> np.ndarray:
    """无限梁集中载荷 P 的 Hetényi 解。
    
    Parameters
    ----------
    x : ndarray
        评估位置。
    P : float
        集中载荷 (N)。
    lam : float
        λ = (k_b / 4EI)^(1/4)。
    k_b : float
        地基系数 (N/m²)。
    
    Returns
    -------
    w : ndarray
        挠度。
    """
    xi = lam * np.abs(x)
    A = np.exp(-xi) * (np.cos(xi) + np.sin(xi))
    return P * lam / (2 * k_b) * A


def hetenyi_plate_strip_parameters(
    E: float, nu: float, h: float, k_f: float, width: float = 1.0
) -> Tuple[float, float, float]:
    """板条参数映射 (用于 Hetényi 梁解近似板)。
    
    Returns
    -------
    D : float
        板弯曲刚度。
    EI : float
        等效梁刚度 (D * width)。
    k_b : float
        梁地基系数 (k_f * width)。
    """
    D = E * h**3 / (12 * (1 - nu**2))
    EI = D * width
    k_b = k_f * width
    return D, EI, k_b


def hetenyi_characteristic_wavenumber(EI: float, k_b: float) -> float:
    """Hetényi 特征波数 λ = (k_b / 4EI)^(1/4)。"""
    return (k_b / (4 * EI))**0.25


# ---------------------------------------------------------------------------
# 板求解器
# ---------------------------------------------------------------------------

def solve_plate_on_foundation(
    nodes: np.ndarray,
    elements: np.ndarray,
    young: float,
    poisson: float,
    thickness: float,
    foundation_coefficient: float,
    nodal_forces: np.ndarray,
    boundary_conditions: str = "simply_supported",
) -> PlateResult:
    """求解弹性地基板。
    
    Parameters
    ----------
    nodes : ndarray (N, 2)
        节点坐标。
    elements : ndarray (E, 4)
        单元连接。
    young, poisson, thickness : float
        材料参数。
    foundation_coefficient : float
        Winkler 地基系数 k_f (N/m³)。
    nodal_forces : ndarray (N, 3)
        节点力 (fx, fy, fz)。
    boundary_conditions : str
        'simply_supported' | 'clamped' | 'free'
    
    Returns
    -------
    PlateResult
    """
    n_nodes = len(nodes)
    ndof = 3 * n_nodes
    
    # 组装全局刚度矩阵
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    
    for elem_nodes in elements:
        xy = nodes[elem_nodes]
        ke_bending = q4_mindlin_stiffness(xy, young, poisson, thickness)
        ke_foundation = q4_winkler_foundation_stiffness(xy, foundation_coefficient)
        ke = ke_bending + ke_foundation
        
        # 组装
        dofs = []
        for n in elem_nodes:
            dofs.extend([3*n, 3*n+1, 3*n+2])
        dofs = np.array(dofs)
        
        K[np.ix_(dofs, dofs)] += ke
        
        # 一致载荷向量：F_i = ∫∫ N_i * q(x,y) dA
        # 在高斯点评估压力（从节点值插值）
        gp = 1.0 / 3.0**0.5
        for xif in (-gp, gp):
            for etaf in (-gp, gp):
                N, dN_nat = _shape(xif, etaf)
                J = dN_nat.T @ xy
                detJ = np.linalg.det(J)
                
                # 在高斯点插值压力值
                q_gauss = 0.0
                for i_local, n_global in enumerate(elem_nodes):
                    q_gauss += N[i_local] * nodal_forces[n_global, 2]
                
                # 分配到单元节点的w自由度（索引0, 3, 6, 9...）
                for i_local, n_global in enumerate(elem_nodes):
                    F[3*n_global] += N[i_local] * q_gauss * detJ
    
    # 施加边界条件
    if boundary_conditions == "simply_supported":
        # 简支：边缘 w=0，切向转角=0
        tol = nodes[:, 0].max() * 1e-12
        x_edge = (np.abs(nodes[:, 0]) < tol) | (np.abs(nodes[:, 0] - nodes[:, 0].max()) < tol)
        y_edge = (np.abs(nodes[:, 1]) < tol) | (np.abs(nodes[:, 1] - nodes[:, 1].max()) < tol)
        edge = x_edge | y_edge
        
        fixed = []
        for i in range(n_nodes):
            if edge[i]:
                fixed.append(3*i)      # w=0
            if y_edge[i]:
                fixed.append(3*i + 1)  # theta_x=0
            if x_edge[i]:
                fixed.append(3*i + 2)  # theta_y=0
        
        fixed = np.unique(fixed)
        free = np.setdiff1d(np.arange(ndof), fixed)
        
        # 求解
        u = np.zeros(ndof)
        u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
        
    elif boundary_conditions == "clamped":
        # 固支：边缘 w=0, theta_x=0, theta_y=0
        tol = nodes[:, 0].max() * 1e-12
        x_edge = (np.abs(nodes[:, 0]) < tol) | (np.abs(nodes[:, 0] - nodes[:, 0].max()) < tol)
        y_edge = (np.abs(nodes[:, 1]) < tol) | (np.abs(nodes[:, 1] - nodes[:, 1].max()) < tol)
        edge = x_edge | y_edge
        
        fixed = []
        for i in range(n_nodes):
            if edge[i]:
                fixed.extend([3*i, 3*i+1, 3*i+2])
        
        fixed = np.unique(fixed)
        free = np.setdiff1d(np.arange(ndof), fixed)
        
        u = np.zeros(ndof)
        u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
        
    else:
        raise ValueError(f"Unknown boundary condition: {boundary_conditions}")
    
    reaction = K @ u - F
    
    return PlateResult(
        displacement=u,
        reaction=reaction,
        stiffness=K,
        force=F,
        nodes=nodes,
        n_elements=len(elements)
    )


# ---------------------------------------------------------------------------
# 账本 B 生成
# ---------------------------------------------------------------------------

def generate_ledger_b(
    ledger_a: Dict,
    plate_result: PlateResult,
    material_params: Dict[str, float],
) -> Dict[str, Any]:
    """生成账本 B (FEM 侧收到的节点力)。
    
    Parameters
    ----------
    ledger_a : dict
        账本 A (LBM 发出的力)。
    plate_result : PlateResult
        板求解结果。
    material_params : dict
        材料参数 (young, poisson, thickness, foundation_coefficient)。
    
    Returns
    -------
    dict
        账本 B。
    """
    # 账本 B 的力应与账本 A 完全相同 (单向纯传递)
    forces_a = np.array(ledger_a["forces"])
    total_force_b = forces_a.sum(axis=0).tolist()
    
    # 提取挠度场
    n_nodes = len(plate_result.nodes)
    deflections = []
    for i in range(n_nodes):
        w = plate_result.displacement[3*i + 2]  # z 方向位移
        deflections.append(float(w))
    
    return {
        "ledger_type": "B",
        "description": "FEM 侧收到的节点力",
        "schema_version": 1,
        "n_nodes": int(n_nodes),
        "nodes": plate_result.nodes.tolist() if plate_result.nodes.shape[1] == 2 else 
                 np.column_stack([plate_result.nodes, np.zeros(n_nodes)]).tolist(),
        "forces": forces_a.tolist(),  # 与 A 相同
        "total_force": total_force_b,
        "deflections": deflections,
        "material_params": material_params,
        "source_ledger_a": ledger_a.get("source_file", "unknown"),
        "timestamp": _utcnow_z(),
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_plate_response(
    ledger_a_path: str,
    output_dir: str,
    material_params: Optional[Dict[str, float]] = None,
    mesh_resolution: int = 8,
    boundary_conditions: str = "simply_supported",
) -> Tuple[Dict, PlateResult]:
    """运行板响应求解。
    
    Parameters
    ----------
    ledger_a_path : str
        账本 A 路径。
    output_dir : str
        输出目录。
    material_params : dict, optional
        材料参数，默认冰材料。
    mesh_resolution : int
        网格分辨率 (n x n)。
    boundary_conditions : str
        边界条件。
    
    Returns
    -------
    ledger_b : dict
        账本 B。
    plate_result : PlateResult
        板求解结果。
    """
    # 默认冰材料参数
    if material_params is None:
        material_params = {
            "young": 1.0e9,       # Pa (冰弹性模量)
            "poisson": 0.3,
            "thickness": 0.1,     # m (冰板厚度)
            "foundation_coefficient": 1.0e5,  # N/m³ (Winkler 地基)
        }
    
    # 加载账本 A
    with open(ledger_a_path, "r") as f:
        ledger_a = json.load(f)
    
    # 生成板网格
    nodes_2d, elements = structured_square_mesh(mesh_resolution, side=1.0)
    n_nodes = len(nodes_2d)
    
    # 从账本 A 提取节点力
    forces_a = np.array(ledger_a["forces"])
    
    # 如果账本 A 节点数与网格节点数不匹配，插值
    if len(forces_a) != n_nodes:
        # 简单最近邻插值
        nodes_a = np.array(ledger_a["nodes"])
        forces_interp = np.zeros((n_nodes, 3))
        for i, node in enumerate(nodes_2d):
            node_3d = np.array([node[0], node[1], 0.0])
            dists = np.linalg.norm(nodes_a - node_3d, axis=1)
            nearest = np.argmin(dists)
            forces_interp[i] = forces_a[nearest]
        forces_a = forces_interp
    
    # 求解板
    plate_result = solve_plate_on_foundation(
        nodes_2d, elements,
        material_params["young"],
        material_params["poisson"],
        material_params["thickness"],
        material_params["foundation_coefficient"],
        forces_a,
        boundary_conditions
    )
    
    # 生成账本 B
    ledger_b = generate_ledger_b(ledger_a, plate_result, material_params)
    
    # 保存
    output_path = Path(output_dir) / "ledger_b.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ledger_b, f, indent=2, ensure_ascii=False)
    
    return ledger_b, plate_result


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

def save_ledger_b(ledger_b: Dict, output_path: str) -> None:
    """保存账本 B。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ledger_b, f, indent=2, ensure_ascii=False)


def load_ledger_b(input_path: str) -> Dict:
    """加载账本 B。"""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
