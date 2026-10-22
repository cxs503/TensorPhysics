"""SI-4 全链演示测试。

测试内容：
1. DEM 键结颗粒场推进：显式 Euler 积分正确性
2. 全链双账本门：质量/能量守恒 ≤1e-12
3. 全链编排：≥3 个子步循环，四库协同
4. 子步账本登记：每步均有完整记录
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scenarios.suboff_icebreaking.si1_one_way.plate_response import (
    structured_square_mesh,
)
from scenarios.suboff_icebreaking.si1_one_way.lbm_load_extract import (
    generate_synthetic_lbm_snapshot,
)
from scenarios.suboff_icebreaking.si4_full_chain.full_chain_orchestrator import (
    BondedParticleField,
    compute_full_chain_ledger,
    run_full_chain,
)


# ---------------------------------------------------------------------------
# 测试 1：DEM 键结颗粒场推进
# ---------------------------------------------------------------------------

class TestBondedParticleField:
    """测试 DEM 键结颗粒场推进。"""

    def test_single_particle_free_fall(self):
        """单颗粒自由落体：验证重力加速度。"""
        particles = [{
            "particle_id": 0,
            "position": [0.0, 0.0, 1.0],
            "velocity": [0.0, 0.0, 0.0],
            "radius": 0.01,
            "mass": 1.0,
        }]
        bonds = []
        
        field = BondedParticleField(particles, bonds, dt_dem=0.01)
        
        # 推进 10 步
        for _ in range(10):
            field.advance()
        
        # 验证：z 方向速度 = -g * t = -9.81 * 0.1 = -0.981 m/s
        expected_vz = -9.81 * 0.1
        assert abs(field.velocities[0, 2] - expected_vz) < 1e-10
        
        # 验证：z 方向位置 = z0 + 0.5 * g * t^2 = 1.0 - 0.5 * 9.81 * 0.01 = 0.95095
        expected_z = 1.0 - 0.5 * 9.81 * 0.01
        assert abs(field.positions[0, 2] - expected_z) < 1e-10

    def test_two_particles_with_bond(self):
        """两颗粒键结：验证弹簧力。"""
        particles = [
            {
                "particle_id": 0,
                "position": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "radius": 0.01,
                "mass": 1.0,
            },
            {
                "particle_id": 1,
                "position": [0.1, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "radius": 0.01,
                "mass": 1.0,
            },
        ]
        bonds = [{
            "bond_id": 0,
            "particle_i": 0,
            "particle_j": 1,
            "stiffness": 1000.0,
            "strength": 100.0,
            "initial_length": 0.1,
        }]
        
        field = BondedParticleField(particles, bonds, dt_dem=0.001, 
                                    gravity=np.array([0.0, 0.0, 0.0]))
        
        # 初始状态：键结未伸长
        assert field.bond_status[0] == "intact"
        assert field.total_bond_energy == 0.0
        
        # 手动拉伸键结
        field.positions[1, 0] = 0.11  # 伸长 0.01 m
        
        # 推进 1 步
        ledger = field.advance()
        
        # 验证：键结力 = k * Δx = 1000 * 0.01 = 10 N
        # 颗粒 0 受正 x 方向力，颗粒 1 受负 x 方向力
        # 加速度 a = F/m = 10 / 1 = 10 m/s^2
        # 速度变化 Δv = a * dt = 10 * 0.001 = 0.01 m/s
        assert abs(field.velocities[0, 0] - 0.01) < 1e-10
        assert abs(field.velocities[1, 0] + 0.01) < 1e-10
        
        # 验证：键结能 = 0.5 * k * Δx^2 = 0.5 * 1000 * 0.01^2 = 0.05 J
        assert abs(field.total_bond_energy - 0.05) < 1e-10

    def test_bond_breaking(self):
        """键结断裂：验证强度判据。"""
        particles = [
            {
                "particle_id": 0,
                "position": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "radius": 0.01,
                "mass": 1.0,
            },
            {
                "particle_id": 1,
                "position": [0.1, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "radius": 0.01,
                "mass": 1.0,
            },
        ]
        bonds = [{
            "bond_id": 0,
            "particle_i": 0,
            "particle_j": 1,
            "stiffness": 1000.0,
            "strength": 5.0,  # 低强度
            "initial_length": 0.1,
        }]
        
        field = BondedParticleField(particles, bonds, dt_dem=0.001,
                                    gravity=np.array([0.0, 0.0, 0.0]))
        
        # 手动拉伸键结超过强度
        # 断裂判据：F = k * Δx > strength → Δx > 5/1000 = 0.005 m
        field.positions[1, 0] = 0.106  # 伸长 0.006 m > 0.005 m
        
        # 推进 1 步
        ledger = field.advance()
        
        # 验证：键结断裂
        assert field.bond_status[0] == "broken"
        assert field.n_bonds_broken_total == 1
        assert ledger["n_bonds_intact"] == 0


# ---------------------------------------------------------------------------
# 测试 2：全链双账本门
# ---------------------------------------------------------------------------

class TestFullChainLedger:
    """测试全链双账本门。"""

    def test_mass_conservation_pass(self):
        """质量守恒通过：|Δm|/m_0 ≤ 1e-12。"""
        ledger = compute_full_chain_ledger(
            chain_step=0,
            si3_momentum_residual=1e-14,
            si3_energy_residual=1e-14,
            si2_mass_conservation=1e-14,
            si2_energy_conservation=1e-14,
            dem_total_mass=100.0,
            dem_total_energy=50.0,
            initial_total_mass=100.0,
            initial_total_energy=50.0,
        )
        
        assert ledger["full_chain_conservation"]["mass_passed"] is True
        assert ledger["full_chain_conservation"]["mass_relative_residual"] < 1e-12
        assert ledger["all_gates_passed"] is True

    def test_energy_conservation_pass(self):
        """能量守恒通过：|ΔE|/E_0 ≤ 1e-12。"""
        ledger = compute_full_chain_ledger(
            chain_step=0,
            si3_momentum_residual=1e-14,
            si3_energy_residual=1e-14,
            si2_mass_conservation=1e-14,
            si2_energy_conservation=1e-14,
            dem_total_mass=100.0,
            dem_total_energy=50.0,
            initial_total_mass=100.0,
            initial_total_energy=50.0,
        )
        
        assert ledger["full_chain_conservation"]["energy_passed"] is True
        assert ledger["full_chain_conservation"]["energy_relative_residual"] < 1e-12
        assert ledger["all_gates_passed"] is True

    def test_mass_conservation_fail(self):
        """质量守恒失败：|Δm|/m_0 > 1e-12。"""
        ledger = compute_full_chain_ledger(
            chain_step=0,
            si3_momentum_residual=1e-14,
            si3_energy_residual=1e-14,
            si2_mass_conservation=1e-14,
            si2_energy_conservation=1e-14,
            dem_total_mass=100.0,
            dem_total_energy=50.0,
            initial_total_mass=99.0,  # 质量不守恒
            initial_total_energy=50.0,
        )
        
        assert ledger["full_chain_conservation"]["mass_passed"] is False
        assert ledger["all_gates_passed"] is False

    def test_si3_gate_fail(self):
        """SI-3 账本失败：动量残差 > 1e-12。"""
        ledger = compute_full_chain_ledger(
            chain_step=0,
            si3_momentum_residual=1e-10,  # 超过阈值
            si3_energy_residual=1e-14,
            si2_mass_conservation=1e-14,
            si2_energy_conservation=1e-14,
            dem_total_mass=100.0,
            dem_total_energy=50.0,
            initial_total_mass=100.0,
            initial_total_energy=50.0,
        )
        
        assert ledger["si3_coupling"]["passed"] is False
        assert ledger["all_gates_passed"] is False


# ---------------------------------------------------------------------------
# 测试 3：全链编排
# ---------------------------------------------------------------------------

class TestFullChainOrchestration:
    """测试全链编排。"""

    def test_full_chain_runs_three_steps(self):
        """全链运行 3 步：验证子步循环。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 准备输入
            nodes, elements = structured_square_mesh(n=3, side=1.0)
            nodes_3d = np.column_stack([nodes, np.zeros(len(nodes))])
            
            snapshot_path = Path(tmpdir) / "lbm_snapshot.npz"
            generate_synthetic_lbm_snapshot(
                output_path=str(snapshot_path),
                plate_nodes=nodes_3d,
                pressure_magnitude=1000.0,
                seed=42,
            )
            
            # 计算节点面积
            side = 1.0
            n = 3
            dx = side / n
            area_per_node = np.full(len(nodes), dx**2)
            
            plate_config = {
                "nodes": nodes.tolist(),
                "elements": elements.tolist(),
                "area_per_node": area_per_node.tolist(),
                "young": 2.1e11,
                "poisson": 0.3,
                "thickness": 0.01,
            }
            
            # 运行全链
            output_dir = Path(tmpdir) / "outputs"
            summary = run_full_chain(
                lbm_snapshot_path=str(snapshot_path),
                plate_config=plate_config,
                n_chain_steps=3,
                n_dem_substeps=2,
                dt_coupling=0.001,
                dt_dem=1e-5,
                sigma_c=1e6,
                G_c=100.0,
                density=917.0,
                output_dir=str(output_dir),
            )
            
            # 验证：3 个全链步
            assert len(summary["chain_history"]) == 3
            assert len(summary["conservation_ledgers"]) == 3
            
            # 验证：每步均有完整记录
            for k, step in enumerate(summary["chain_history"]):
                assert step["chain_step"] == k
                assert "si3_momentum_residual" in step
                assert "si3_energy_residual" in step
                assert "si2_mass_conservation" in step
                assert "si2_energy_conservation" in step
                assert "n_failed_faces" in step
                assert "full_chain_ledger" in step

    def test_full_chain_dual_ledger_gates(self):
        """全链双账本门：质量/能量守恒 ≤1e-12。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes, elements = structured_square_mesh(n=3, side=1.0)
            nodes_3d = np.column_stack([nodes, np.zeros(len(nodes))])
            
            snapshot_path = Path(tmpdir) / "lbm_snapshot.npz"
            generate_synthetic_lbm_snapshot(
                output_path=str(snapshot_path),
                plate_nodes=nodes_3d,
                pressure_magnitude=1000.0,
                seed=42,
            )
            
            side = 1.0
            n = 3
            dx = side / n
            area_per_node = np.full(len(nodes), dx**2)
            
            plate_config = {
                "nodes": nodes.tolist(),
                "elements": elements.tolist(),
                "area_per_node": area_per_node.tolist(),
                "young": 2.1e11,
                "poisson": 0.3,
                "thickness": 0.01,
            }
            
            output_dir = Path(tmpdir) / "outputs"
            summary = run_full_chain(
                lbm_snapshot_path=str(snapshot_path),
                plate_config=plate_config,
                n_chain_steps=3,
                n_dem_substeps=2,
                output_dir=str(output_dir),
            )
            
            # 验证：所有账本门通过
            for ledger in summary["conservation_ledgers"]:
                assert ledger["all_gates_passed"] is True
                assert ledger["full_chain_conservation"]["mass_passed"] is True
                assert ledger["full_chain_conservation"]["energy_passed"] is True
                assert ledger["si3_coupling"]["passed"] is True
                assert ledger["si2_fracture"]["passed"] is True

    def test_full_chain_four_libraries_participate(self):
        """四库协同：LBM、FEM、DEM、cohesive 均参与。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes, elements = structured_square_mesh(n=3, side=1.0)
            nodes_3d = np.column_stack([nodes, np.zeros(len(nodes))])
            
            snapshot_path = Path(tmpdir) / "lbm_snapshot.npz"
            generate_synthetic_lbm_snapshot(
                output_path=str(snapshot_path),
                plate_nodes=nodes_3d,
                pressure_magnitude=1000.0,
                seed=42,
            )
            
            side = 1.0
            n = 3
            dx = side / n
            area_per_node = np.full(len(nodes), dx**2)
            
            plate_config = {
                "nodes": nodes.tolist(),
                "elements": elements.tolist(),
                "area_per_node": area_per_node.tolist(),
                "young": 2.1e11,
                "poisson": 0.3,
                "thickness": 0.01,
            }
            
            output_dir = Path(tmpdir) / "outputs"
            summary = run_full_chain(
                lbm_snapshot_path=str(snapshot_path),
                plate_config=plate_config,
                n_chain_steps=3,
                output_dir=str(output_dir),
            )
            
            # 验证：LBM 快照生成
            lbm_snapshots = list(output_dir.glob("lbm_snapshot_step*.npz"))
            assert len(lbm_snapshots) >= 3
            
            # 验证：FEM 板响应（通过 SI-3 账本体现）
            for step in summary["chain_history"]:
                assert step["si3_momentum_residual"] < 1e-12
                assert step["si3_energy_residual"] < 1e-12
            
            # 验证：cohesive 破坏判定（通过 SI-2 账本体现）
            for step in summary["chain_history"]:
                assert step["si2_mass_conservation"] < 1e-12
                assert step["si2_energy_conservation"] < 1e-12
            
            # 验证：DEM 颗粒场（通过 final_state 体现）
            assert "final_state" in summary
            assert "n_particles" in summary["final_state"]
            assert "n_bonds" in summary["final_state"]
