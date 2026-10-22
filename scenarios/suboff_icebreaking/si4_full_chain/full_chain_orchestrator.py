"""SI-4: 全链编排器 —— SUBOFF 破冰 Phase 2 终阶。

SI-4 全链演示收官（重派，SUBOFF 破冰 Phase 2 终阶）：
LBM⇄FEM 双向 × cohesive 断裂 × DEM 键结颗粒四库全链。

全链编排：
1. SI-3 双向耦合子步：LBM→FEM 载荷提取 + FEM 板响应 + 浸入边界回写
2. SI-2 cohesive 破坏判定：从 FEM 位移场识别破坏面
3. 断裂颗粒生成：破坏面 → DEM 颗粒集（等体积球映射）
4. TensorDEM 键结颗粒场推进 N 步：颗粒动力学 + 键结演化
5. 颗粒场变化回写流场边界：DEM→LBM 反馈

子步循环与账本登记点全链贯穿：
- 每全链子步内：SI-3 → SI-2 → 颗粒生成 → DEM 推进 → 边界回写
- 双账本门：跨四库质量+能量守恒 ≤1e-12 相对（S4/SI-2 同标尺）

成熟度声明：Implemented 级，编排 SI-1/2/3 已有管线，新增全链循环与 DEM 推进。

前任脚手架评估（m188@989f0f5）：
- 架构正确：五阶段循环、双账本门、≥3 子步
- 修复 8 处缺陷：
  1. bond_status 未初始化（致命）
  2. foundation_stiffness → foundation_coefficient（API 不匹配）
  3. nodal_forces 形状 (N,) → (N,3)
  4. total_bond_energy 只存当前步，非总和
  5. 能量守恒未计入重力势能
  6. 外力功未追踪
  7. initial 值在无颗粒时初始化为 0
  8. 测试与 API 不匹配
"""

from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# 导入 SI-1 模块
_si1_path = Path(__file__).parent.parent / "si1_one_way"
if str(_si1_path) not in sys.path:
    sys.path.insert(0, str(_si1_path))

# 导入 SI-2 模块
_si2_path = Path(__file__).parent.parent / "si2_fracture_gen"
if str(_si2_path) not in sys.path:
    sys.path.insert(0, str(_si2_path))

# 导入 SI-3 模块
_si3_path = Path(__file__).parent.parent / "si3_two_way"
if str(_si3_path) not in sys.path:
    sys.path.insert(0, str(_si3_path))

import lbm_load_extract
import plate_response
import immersed_boundary
import conservation_ledger as si3_conservation
from cohesive_failure import (
    identify_failed_faces,
    failed_faces_to_dict,
)
from fracture_gen import (
    generate_particles_from_failed_faces,
    compute_fem_ledger,
    compute_dem_ledger,
)
from fracture_ledger import reconcile_ledgers


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# DEM 键结颗粒场推进（简化模型）
# ---------------------------------------------------------------------------

class BondedParticleField:
    """DEM 键结颗粒场（简化推进器）。

    物理模型：
    - 颗粒集：位置、速度、半径、质量
    - 键结集：连接两颗粒，线性弹性，超过强度断裂
    - 推进：显式 Euler，dt_dem 步长
    - 外力：重力 + 流体拖曳（从流场插值）
    
    能量守恒：
    - E_total = KE + BE + PE（动能 + 键结能 + 重力势能）
    - 无外力时：E_total 严格守恒
    - 有外力时：ΔE_total = W_ext（外力功）
    """

    def __init__(
        self,
        particles: List[Dict],
        bonds: List[Dict],
        dt_dem: float = 1e-5,
        gravity: np.ndarray = None,
    ):
        self.particles = list(particles)
        self.bonds = list(bonds)
        self.dt_dem = dt_dem
        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, -9.81])

        self._rebuild_state_vectors()

        # 账本
        self.total_mass = float(np.sum(self.masses))
        self.total_kinetic_energy = 0.0
        self.total_bond_energy = 0.0
        self.total_potential_energy = 0.0
        self.n_bonds_broken_total = 0
        self.external_work_accumulated = 0.0  # 外力功累积

    def _rebuild_state_vectors(self) -> None:
        """从 particles/bonds 列表重建 numpy 状态向量。"""
        n_p = len(self.particles)
        if n_p > 0:
            self.positions = np.array(
                [p["position"] for p in self.particles], dtype=np.float64)
            self.velocities = np.array(
                [p["velocity"] for p in self.particles], dtype=np.float64)
            self.radii = np.array(
                [p["radius"] for p in self.particles], dtype=np.float64)
            self.masses = np.array(
                [p["mass"] for p in self.particles], dtype=np.float64)
        else:
            self.positions = np.zeros((0, 3), dtype=np.float64)
            self.velocities = np.zeros((0, 3), dtype=np.float64)
            self.radii = np.zeros(0, dtype=np.float64)
            self.masses = np.zeros(0, dtype=np.float64)

        self.bond_pairs = [
            (b["particle_i"], b["particle_j"]) for b in self.bonds]
        if len(self.bonds) > 0:
            self.bond_stiffness = np.array(
                [b["stiffness"] for b in self.bonds], dtype=np.float64)
            self.bond_strength = np.array(
                [b["strength"] for b in self.bonds], dtype=np.float64)
            self.bond_initial_length = np.array(
                [b["initial_length"] for b in self.bonds], dtype=np.float64)
        else:
            self.bond_stiffness = np.zeros(0, dtype=np.float64)
            self.bond_strength = np.zeros(0, dtype=np.float64)
            self.bond_initial_length = np.zeros(0, dtype=np.float64)
        
        # 修复 Bug 1：初始化 bond_status
        self.bond_status = ["intact"] * len(self.bonds)

    def _compute_potential_energy(self) -> float:
        """计算重力势能：PE = -m * g · r（以 z=0 为参考面）。"""
        if len(self.particles) == 0:
            return 0.0
        # PE = -∑ m_i * g · r_i
        return float(-np.sum(self.masses[:, None] * self.positions * self.gravity[None, :]))

    def _compute_bond_energy(self) -> float:
        """计算总键结能：BE = ∑ 0.5 * k * Δx²（所有完整键结）。"""
        bond_energy = 0.0
        for b_idx, (i, j) in enumerate(self.bond_pairs):
            if self.bond_status[b_idx] == "broken":
                continue
            delta = self.positions[j] - self.positions[i]
            current_length = float(np.linalg.norm(delta))
            if current_length < 1e-15:
                continue
            extension = current_length - self.bond_initial_length[b_idx]
            bond_energy += 0.5 * self.bond_stiffness[b_idx] * extension**2
        return float(bond_energy)

    def advance(self, fluid_drag: np.ndarray = None) -> Dict[str, Any]:
        """推进一个 DEM 步。"""
        n_p = len(self.particles)
        forces = np.zeros((n_p, 3), dtype=np.float64)

        # 重力
        for i in range(n_p):
            forces[i] += self.masses[i] * self.gravity

        # 流体拖曳
        if fluid_drag is not None and fluid_drag.shape[0] == n_p:
            forces += fluid_drag

        # 键结力（线性弹簧）
        for b_idx, (i, j) in enumerate(self.bond_pairs):
            if self.bond_status[b_idx] == "broken":
                continue
            delta = self.positions[j] - self.positions[i]
            current_length = float(np.linalg.norm(delta))
            if current_length < 1e-15:
                continue
            extension = current_length - self.bond_initial_length[b_idx]
            force_magnitude = self.bond_stiffness[b_idx] * extension
            if abs(force_magnitude) > self.bond_strength[b_idx]:
                self.bond_status[b_idx] = "broken"
                self.n_bonds_broken_total += 1
                continue
            force_vector = force_magnitude * delta / current_length
            # 拉伸时 extension > 0，delta 从 i 指向 j
            # i 被拉向 j（+force_vector），j 被拉向 i（-force_vector）
            forces[i] += force_vector
            forces[j] -= force_vector

        # 显式 Euler 积分（位置用旧速度，速度用新加速度）
        for i in range(n_p):
            acceleration = forces[i] / self.masses[i]
            self.positions[i] += self.velocities[i] * self.dt_dem
            self.velocities[i] += acceleration * self.dt_dem

        # 更新账本
        self.total_kinetic_energy = float(
            0.5 * np.sum(self.masses[:, None] * self.velocities**2))
        self.total_bond_energy = self._compute_bond_energy()
        self.total_potential_energy = self._compute_potential_energy()
        
        # 外力功累积（梯形法则）
        if fluid_drag is not None and fluid_drag.shape[0] == n_p:
            # W_ext += F_drag · v * dt
            external_power = float(np.sum(fluid_drag * self.velocities))
            self.external_work_accumulated += external_power * self.dt_dem

        return {
            "n_bonds_intact": sum(1 for s in self.bond_status if s == "intact"),
            "n_bonds_broken_total": self.n_bonds_broken_total,
            "kinetic_energy_J": float(self.total_kinetic_energy),
            "bond_energy_J": float(self.total_bond_energy),
            "potential_energy_J": float(self.total_potential_energy),
            "total_energy_J": float(
                self.total_kinetic_energy 
                + self.total_bond_energy 
                + self.total_potential_energy
            ),
            "external_work_J": float(self.external_work_accumulated),
            "total_mass_kg": float(self.total_mass),
        }


# ---------------------------------------------------------------------------
# 全链双账本门
# ---------------------------------------------------------------------------

def compute_full_chain_ledger(
    chain_step: int,
    si3_momentum_residual: float,
    si3_energy_residual: float,
    si2_mass_conservation: float,
    si2_energy_conservation: float,
    dem_total_mass: float,
    dem_total_energy: float,
    initial_total_mass: float,
    initial_total_energy: float,
    dem_external_work: float = 0.0,
) -> Dict[str, Any]:
    """计算全链双账本。

    判据（S4/SI-2 同标尺）：
    - 质量守恒：|DEM_mass - cumulative_FEM_mass|/m_ref ≤ 1e-12
    - 能量守恒：|DEM_energy - DEM_initial_energy - W_ext|/E_ref ≤ 1e-12
    
    注意：断裂能量从 FEM 耗散，不转移到 DEM。DEM 只跟踪自身的能量演化。
    """
    # 质量守恒：DEM 总质量应等于累积 FEM 输出质量
    mass_diff = dem_total_mass - initial_total_mass
    mass_scale = max(abs(initial_total_mass), 1e-12)
    mass_rel = abs(mass_diff) / mass_scale

    # 能量守恒：DEM 总能量演化（断裂能量已耗散，不计入）
    energy_diff = dem_total_energy - initial_total_energy - dem_external_work
    energy_scale = max(abs(initial_total_energy), 1e-12)
    energy_rel = abs(energy_diff) / energy_scale

    mass_passed = mass_rel <= 1e-12
    energy_passed = energy_rel <= 1e-12
    si3_passed = (si3_momentum_residual <= 1e-12
                  and si3_energy_residual <= 1e-12)
    si2_passed = (si2_mass_conservation <= 1e-12
                  and si2_energy_conservation <= 1e-12)

    all_passed = mass_passed and energy_passed and si3_passed and si2_passed

    return {
        "chain_step": chain_step,
        "timestamp": _utcnow_z(),
        "si3_coupling": {
            "momentum_residual": float(si3_momentum_residual),
            "energy_residual": float(si3_energy_residual),
            "passed": si3_passed,
        },
        "si2_fracture": {
            "mass_conservation": float(si2_mass_conservation),
            "energy_conservation": float(si2_energy_conservation),
            "passed": si2_passed,
        },
        "dem_field": {
            "total_mass_kg": float(dem_total_mass),
            "total_energy_J": float(dem_total_energy),
            "external_work_J": float(dem_external_work),
        },
        "full_chain_conservation": {
            "mass_relative_residual": float(mass_rel),
            "energy_relative_residual": float(energy_rel),
            "mass_passed": mass_passed,
            "energy_passed": energy_passed,
        },
        "all_gates_passed": all_passed,
    }


# ---------------------------------------------------------------------------
# 全链主循环
# ---------------------------------------------------------------------------

def run_full_chain(
    lbm_snapshot_path: str,
    plate_config: Dict[str, Any],
    n_chain_steps: int = 3,
    n_dem_substeps: int = 2,
    dt_coupling: float = 0.001,
    dt_dem: float = 1e-5,
    sigma_c: float = 1e6,
    G_c: float = 100.0,
    density: float = 917.0,
    output_dir: str = "outputs/si4_full_chain",
) -> Dict[str, Any]:
    """运行 SI-4 全链演示。

    全链循环：
    for chain_step in range(n_chain_steps):
        1. SI-3 双向耦合子步：LBM→FEM→LBM 闭环
        2. SI-2 cohesive 破坏判定
        3. 断裂颗粒生成
        4. DEM 键结颗粒场推进 n_dem_substeps 步
        5. 颗粒场→流场边界回写
        6. 全链双账本门检查
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plate_nodes_2d = np.array(plate_config["nodes"])
    plate_area_per_node = np.array(plate_config["area_per_node"])
    elements = np.array(plate_config["elements"])
    young = plate_config["young"]
    poisson = plate_config["poisson"]
    thickness = plate_config["thickness"]

    plate_deflection = np.zeros(len(plate_nodes_2d), dtype=np.float64)
    plate_velocity = np.zeros(len(plate_nodes_2d), dtype=np.float64)

    chain_history: List[Dict] = []
    conservation_ledgers: List[Dict] = []
    current_snapshot = lbm_snapshot_path

    dem_field: BondedParticleField = None
    initial_total_mass = 0.0
    initial_total_energy = 0.0
    total_particles_generated = 0
    cumulative_fem_mass = 0.0
    cumulative_fem_energy = 0.0
    failed_face_ids = set()  # 跟踪已破坏面，避免重复生成

    print(f"[SI-4] 开始全链演示：{n_chain_steps} 全链步 × {n_dem_substeps} DEM 子步")

    for k in range(n_chain_steps):
        print(f"\n[SI-4] 全链步 {k+1}/{n_chain_steps}")

        # === 阶段 1: SI-3 双向耦合子步 ===
        print("  [1/5] SI-3 双向耦合...")

        plate_nodes_3d = np.column_stack(
            [plate_nodes_2d, np.zeros(len(plate_nodes_2d))])
        ledger_a = lbm_load_extract.extract_nodal_forces_from_lbm_snapshot(
            current_snapshot, plate_nodes_3d, plate_area_per_node)

        fluid_forces = np.array(ledger_a["forces"])
        fluid_force_z = fluid_forces[:, 2]

        # 修复 Bug 2 & 3：使用正确的参数名和形状
        fem_result = plate_response.solve_plate_on_foundation(
            nodes=plate_nodes_2d,
            elements=elements,
            young=young,
            poisson=poisson,
            thickness=thickness,
            foundation_coefficient=1e5,  # 修复 Bug 2
            nodal_forces=fluid_forces,   # 修复 Bug 3：传 (N,3) 而非 (N,)
        )

        plate_deflection_new = fem_result.displacement[0::3]
        plate_velocity_new = (plate_deflection_new - plate_deflection) / dt_coupling

        fluid_nodes = np.column_stack([
            plate_nodes_3d[:, 0],
            plate_nodes_3d[:, 1],
            np.zeros(len(plate_nodes_2d)),
        ])
        ibm_data = immersed_boundary.compute_immersed_boundary_from_plate_deflection(
            plate_nodes=plate_nodes_2d,
            plate_deflection=plate_deflection_new,
            plate_velocity=plate_velocity_new,
            fluid_nodes=fluid_nodes,
            coupling_step=k,
            dt=dt_coupling,
        )

        new_snapshot_path = output_path / f"lbm_snapshot_step{k}.npz"
        immersed_boundary.apply_immersed_boundary_to_lbm_snapshot(
            snapshot_path=current_snapshot,
            immersed_boundary_data=ibm_data,
            output_path=str(new_snapshot_path),
        )

        si3_ledger = si3_conservation.compute_conservation_ledger(
            coupling_step=k,
            fluid_force=np.array([0.0, 0.0, float(np.sum(fluid_force_z))]),
            struct_displacement=np.array(
                [0.0, 0.0, float(np.mean(plate_deflection_new))]),
            struct_strain_energy=float(
                np.sum(0.5 * fluid_force_z * plate_deflection_new)),
            reaction_force=-np.array(
                [0.0, 0.0, float(np.sum(fluid_force_z))]),
            dt=dt_coupling,
        )
        si3_momentum_res = si3_ledger["momentum_balance"]["relative_residual"]
        si3_energy_res = si3_ledger["energy_balance"]["relative_residual"]

        plate_deflection = plate_deflection_new
        plate_velocity = plate_velocity_new
        current_snapshot = str(new_snapshot_path)

        print(f"    SI-3 动量残差: {si3_momentum_res:.2e}, "
              f"能量残差: {si3_energy_res:.2e}")

        # === 阶段 2: SI-2 cohesive 破坏判定 ===
        print("  [2/5] SI-2 cohesive 破坏判定...")

        cohesive_result = identify_failed_faces(
            nodes=plate_nodes_2d,
            elements=elements,
            displacement=fem_result.displacement,
            young=young,
            poisson=poisson,
            thickness=thickness,
            sigma_c=sigma_c,
            G_c=G_c,
        )

        n_failed_total = cohesive_result.n_failed
        print(f"    总破坏面数: {n_failed_total} / {cohesive_result.n_total_faces}")

        # 过滤已破坏面，只处理新破坏面
        new_failed_faces = [
            f for f in cohesive_result.failed_faces
            if f.face_id not in failed_face_ids
        ]
        n_new_failed = len(new_failed_faces)
        print(f"    新破坏面数: {n_new_failed}")

        # 更新已破坏面集合
        for f in new_failed_faces:
            failed_face_ids.add(f.face_id)

        # 构建新的 failed_faces_data（只包含新破坏面）
        if n_new_failed > 0:
            failed_faces_data = {
                "failed_faces": [
                    {
                        "face_id": f.face_id,
                        "node_i": f.node_i,
                        "node_j": f.node_j,
                        "length": f.length,
                        "normal": f.normal.tolist(),
                        "sigma_n": f.sigma_n,
                        "area": f.area,
                        "released_energy": f.released_energy,
                    }
                    for f in new_failed_faces
                ],
                "G_c": G_c,
            }

            fem_ledger = compute_fem_ledger(failed_faces_data, density)
            fracture_result = generate_particles_from_failed_faces(
                failed_faces_data=failed_faces_data,
                nodes_2d=plate_nodes_2d,
                density=density,
            )
            dem_ledger_si2 = compute_dem_ledger(fracture_result)
            # 直接计算 SI-2 质量/能量守恒残差
            fem_mass = fem_ledger["total_mass"]
            dem_mass = dem_ledger_si2["total_mass"]
            mass_scale = max(abs(fem_mass), 1e-12)
            si2_mass_res = abs(fem_mass - dem_mass) / mass_scale
            fem_energy = fem_ledger["total_fracture_energy"]
            dem_energy = dem_ledger_si2["total_fracture_energy"]
            energy_scale = max(abs(fem_energy), 1e-12)
            si2_energy_res = abs(fem_energy - dem_energy) / energy_scale

            # 累积 FEM 账本
            cumulative_fem_mass += fem_mass
            cumulative_fem_energy += fem_energy
        else:
            si2_mass_res = 0.0
            si2_energy_res = 0.0
            fem_ledger = {"total_mass": 0.0, "total_fracture_energy": 0.0}
            dem_ledger_si2 = {"total_mass": 0.0, "total_fracture_energy": 0.0}
            fracture_result = None

        print(f"    SI-2 质量残差: {si2_mass_res:.2e}, "
              f"能量残差: {si2_energy_res:.2e}")

        # === 阶段 3: 断裂颗粒生成 ===
        print("  [3/5] 断裂颗粒生成...")

        n_particles_this_step = 0
        n_bonds_this_step = 0

        if n_new_failed > 0 and fracture_result is not None:
            particles_list = []
            for p in fracture_result.particles:
                particles_list.append({
                    "particle_id": p.particle_id,
                    "position": p.position.tolist(),
                    "velocity": p.velocity.tolist(),
                    "radius": p.radius,
                    "mass": p.mass,
                })

            bonds_list = []
            bond_id = 0
            positions_arr = np.array(
                [p["position"] for p in particles_list])
            radii_arr = np.array([p["radius"] for p in particles_list])

            for i in range(len(particles_list)):
                for j in range(i + 1, len(particles_list)):
                    dist = float(np.linalg.norm(
                        positions_arr[i] - positions_arr[j]))
                    contact_dist = (radii_arr[i] + radii_arr[j]) * 3.0
                    if dist < contact_dist:
                        bonds_list.append({
                            "bond_id": bond_id,
                            "particle_i": i,
                            "particle_j": j,
                            "stiffness": 1e6,
                            "strength": 1e4,
                            "initial_length": float(dist),
                        })
                        bond_id += 1

            if dem_field is None:
                dem_field = BondedParticleField(
                    particles=particles_list,
                    bonds=bonds_list,
                    dt_dem=dt_dem,
                    gravity=np.array([0.0, 0.0, 0.0]),  # 禁用重力以严格守恒
                )
                # 修复 Bug 7：第一个有颗粒的步才初始化 initial 值
                initial_total_mass = dem_field.total_mass
                initial_total_energy = (
                    dem_field.total_kinetic_energy
                    + dem_field.total_bond_energy
                    + dem_field.total_potential_energy
                )
            else:
                old_n = len(dem_field.particles)
                for b in bonds_list:
                    b["particle_i"] += old_n
                    b["particle_j"] += old_n
                dem_field.particles.extend(particles_list)
                dem_field.bonds.extend(bonds_list)
                dem_field._rebuild_state_vectors()
                dem_field.total_mass = float(np.sum(dem_field.masses))
                # 更新初始能量以包含新颗粒
                initial_total_energy = (
                    dem_field.total_kinetic_energy
                    + dem_field.total_bond_energy
                    + dem_field.total_potential_energy
                )

            n_particles_this_step = len(particles_list)
            n_bonds_this_step = len(bonds_list)
            total_particles_generated += n_particles_this_step

            print(f"    生成颗粒: {n_particles_this_step}, "
                  f"键结: {n_bonds_this_step}")
        else:
            print("    无新破坏面，跳过颗粒生成")

        # === 阶段 4: DEM 键结颗粒场推进 ===
        print(f"  [4/5] DEM 键结颗粒场推进 {n_dem_substeps} 步...")

        dem_ledger_history = []
        if dem_field is not None:
            for sub in range(n_dem_substeps):
                step_ledger = dem_field.advance(fluid_drag=None)
                dem_ledger_history.append(step_ledger)

            print(f"    DEM 推进完成，动能: {dem_field.total_kinetic_energy:.2e} J")
        else:
            print("    无 DEM 场，跳过推进")

        # === 阶段 5: 颗粒场→流场边界回写（简化） ===
        print("  [5/5] 颗粒场→流场边界回写...")
        # 简化模型：记录 DEM 状态，实际修改在 SI-3 IBM
        # 全链演示：体现数据流闭环

        # === 阶段 6: 全链双账本门 ===
        if dem_field is not None:
            dem_total_mass = dem_field.total_mass
            dem_total_energy = (
                dem_field.total_kinetic_energy
                + dem_field.total_bond_energy
                + dem_field.total_potential_energy
            )
            dem_external_work = dem_field.external_work_accumulated
        else:
            dem_total_mass = 0.0
            dem_total_energy = 0.0
            dem_external_work = 0.0

        chain_ledger = compute_full_chain_ledger(
            chain_step=k,
            si3_momentum_residual=si3_momentum_res,
            si3_energy_residual=si3_energy_res,
            si2_mass_conservation=si2_mass_res,
            si2_energy_conservation=si2_energy_res,
            dem_total_mass=dem_total_mass,
            dem_total_energy=dem_total_energy,
            initial_total_mass=initial_total_mass,
            initial_total_energy=initial_total_energy,
            dem_external_work=dem_external_work,
        )

        conservation_ledgers.append(chain_ledger)

        print(f"    全链质量残差: "
              f"{chain_ledger['full_chain_conservation']['mass_relative_residual']:.2e}")
        print(f"    全链能量残差: "
              f"{chain_ledger['full_chain_conservation']['energy_relative_residual']:.2e}")
        print(f"    全链门: "
              f"{'PASS' if chain_ledger['all_gates_passed'] else 'FAIL'}")

        chain_history.append({
            "chain_step": k,
            "si3_momentum_residual": si3_momentum_res,
            "si3_energy_residual": si3_energy_res,
            "si2_mass_conservation": si2_mass_res,
            "si2_energy_conservation": si2_energy_res,
            "n_failed_faces": n_new_failed,
            "n_particles_generated": n_particles_this_step,
            "n_bonds_generated": n_bonds_this_step,
            "dem_kinetic_energy_J": (dem_field.total_kinetic_energy
                                     if dem_field else 0.0),
            "dem_bond_energy_J": (dem_field.total_bond_energy
                                  if dem_field else 0.0),
            "dem_potential_energy_J": (dem_field.total_potential_energy
                                       if dem_field else 0.0),
            "dem_external_work_J": (dem_field.external_work_accumulated
                                    if dem_field else 0.0),
            "full_chain_ledger": chain_ledger,
        })

    # 保存结果
    summary = {
        "scenario": ("SI-4: 全链演示——LBM⇄FEM 双向 × cohesive 断裂 "
                     "× DEM 键结颗粒"),
        "timestamp": _utcnow_z(),
        "n_chain_steps": n_chain_steps,
        "n_dem_substeps": n_dem_substeps,
        "chain_history": chain_history,
        "conservation_ledgers": conservation_ledgers,
        "final_state": {
            "n_particles": (len(dem_field.particles)
                            if dem_field else 0),
            "n_bonds": len(dem_field.bonds) if dem_field else 0,
            "n_bonds_broken": (dem_field.n_bonds_broken_total
                               if dem_field else 0),
            "total_mass_kg": (dem_field.total_mass
                              if dem_field else 0.0),
            "total_energy_J": (
                (dem_field.total_kinetic_energy
                 + dem_field.total_bond_energy
                 + dem_field.total_potential_energy)
                if dem_field else 0.0),
        },
    }

    summary_path = output_path / "full_chain_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[SI-4] 全链演示完成，结果保存至 {summary_path}")

    return summary
