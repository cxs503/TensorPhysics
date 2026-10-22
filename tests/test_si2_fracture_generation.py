"""SI-2 断裂生成管线测试。

测试内容：
1. 内部 face 提取：结构化网格的拓扑正确性
2. 均匀应力场下的破坏判面：解析手算验证
3. 质量/能量账本守恒：FEM 侧 vs DEM 侧
4. 等体积球映射：颗粒半径/质量正确性
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from scenarios.suboff_icebreaking.si1_one_way.plate_response import (
    structured_square_mesh,
)
from scenarios.suboff_icebreaking.si2_fracture_gen.cohesive_failure import (
    extract_internal_faces,
    identify_failed_faces_uniform_stress,
    identify_failed_faces,
    compute_element_stress,
)
from scenarios.suboff_icebreaking.si2_fracture_gen.fracture_gen import (
    generate_particles_from_failed_faces,
    compute_fem_ledger,
    compute_dem_ledger,
    volume_to_sphere_radius,
    face_volume,
)
from scenarios.suboff_icebreaking.si2_fracture_gen.fracture_ledger import (
    reconcile_ledgers,
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_failed_faces_data(faces_list, G_c=100.0):
    """构造 failed_faces_data 字典。"""
    return {
        "G_c": G_c,
        "failed_faces": faces_list,
    }


# ---------------------------------------------------------------------------
# 测试 1：内部 face 提取
# ---------------------------------------------------------------------------

class TestInternalFaceExtraction:
    """测试内部 face 提取。"""

    def test_single_element_no_internal_faces(self):
        """单单元无内部 face。"""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        elements = np.array([[0, 1, 2, 3]])
        faces = extract_internal_faces(elements)
        assert len(faces) == 0

    def test_two_elements_one_internal_face(self):
        """两个相邻单元有一个共享边。"""
        nodes = np.array([
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1, 1], [2, 1],
        ], dtype=float)
        elements = np.array([
            [0, 1, 4, 3],
            [1, 2, 5, 4],
        ])
        faces = extract_internal_faces(elements)
        assert len(faces) == 1
        # 共享边应该是 1-4
        face = faces[0]
        assert set([face["node_i"], face["node_j"]]) == {1, 4}

    def test_2x2_mesh_internal_faces(self):
        """2×2 网格有 4 个内部 face。"""
        nodes_2d, elements = structured_square_mesh(2, side=1.0)
        faces = extract_internal_faces(elements)
        # 2×2 网格：4 个单元，内部边 = 4 (2 水平 + 2 垂直)
        assert len(faces) == 4

    def test_nxN_mesh_internal_face_count(self):
        """N×N 网格内部 face 数 = 2*N*(N-1)。"""
        for N in [2, 3, 4, 5]:
            nodes_2d, elements = structured_square_mesh(N, side=1.0)
            faces = extract_internal_faces(elements)
            expected = 2 * N * (N - 1)
            assert len(faces) == expected, f"N={N}: expected {expected}, got {len(faces)}"


# ---------------------------------------------------------------------------
# 测试 2：均匀应力场下的破坏判面（解析锚）
# ---------------------------------------------------------------------------

class TestUniformStressFailure:
    """测试均匀应力场下的破坏判面。"""

    def test_uniaxial_tension_x(self):
        """单轴拉伸 σ_x > σ_c：只有垂直边破坏。"""
        N = 3
        side = 1.0
        t = 0.01
        sigma_c = 1e5
        G_c = 100.0
        sigma0 = 2e5  # > σ_c

        nodes_2d, elements = structured_square_mesh(N, side=side)
        uniform_stress = np.array([sigma0, 0.0, 0.0])

        result = identify_failed_faces_uniform_stress(
            nodes_2d, elements, uniform_stress, t, sigma_c, G_c
        )

        # 垂直内部边数 = (N-1) * N
        expected_failed = (N - 1) * N
        assert result.n_failed == expected_failed

        # 每个破坏 face 的法向应该沿 x 方向
        for face in result.failed_faces:
            nx = abs(face.normal[0])
            ny = abs(face.normal[1])
            # 法向应该主要是 x 方向
            assert nx > 0.9 or ny > 0.9  # 法向沿某个轴

    def test_uniaxial_compression_x(self):
        """单轴压缩 |σ_x| > σ_c：也应该破坏。"""
        N = 3
        side = 1.0
        t = 0.01
        sigma_c = 1e5
        G_c = 100.0
        sigma0 = -2e5  # 压缩

        nodes_2d, elements = structured_square_mesh(N, side=side)
        uniform_stress = np.array([sigma0, 0.0, 0.0])

        result = identify_failed_faces_uniform_stress(
            nodes_2d, elements, uniform_stress, t, sigma_c, G_c
        )

        expected_failed = (N - 1) * N
        assert result.n_failed == expected_failed

    def test_below_threshold_no_failure(self):
        """应力低于阈值：无破坏。"""
        N = 3
        side = 1.0
        t = 0.01
        sigma_c = 1e5
        G_c = 100.0
        sigma0 = 0.5e5  # < σ_c

        nodes_2d, elements = structured_square_mesh(N, side=side)
        uniform_stress = np.array([sigma0, 0.0, 0.0])

        result = identify_failed_faces_uniform_stress(
            nodes_2d, elements, uniform_stress, t, sigma_c, G_c
        )

        assert result.n_failed == 0

    def test_total_area_and_energy(self):
        """验证总面积和总释放能的解析值。"""
        N = 4
        side = 1.0
        t = 0.01
        sigma_c = 1e5
        G_c = 50.0
        sigma0 = 2e5

        nodes_2d, elements = structured_square_mesh(N, side=side)
        uniform_stress = np.array([sigma0, 0.0, 0.0])

        result = identify_failed_faces_uniform_stress(
            nodes_2d, elements, uniform_stress, t, sigma_c, G_c
        )

        # 解析值
        edge_length = side / N
        n_failed = (N - 1) * N
        expected_area = n_failed * edge_length * t
        expected_energy = n_failed * G_c * edge_length * t

        actual_area = sum(f.area for f in result.failed_faces)
        actual_energy = sum(f.released_energy for f in result.failed_faces)

        assert abs(actual_area - expected_area) / expected_area < 1e-12
        assert abs(actual_energy - expected_energy) / expected_energy < 1e-12


# ---------------------------------------------------------------------------
# 测试 3：质量/能量账本守恒
# ---------------------------------------------------------------------------

class TestLedgerConservation:
    """测试双账本守恒。"""

    def test_mass_conservation(self):
        """质量守恒：FEM 核销 = DEM 颗粒。"""
        N = 3
        side = 1.0
        t = 0.01
        rho = 917.0
        sigma_c = 1e5
        G_c = 100.0
        sigma0 = 2e5

        nodes_2d, elements = structured_square_mesh(N, side=side)
        uniform_stress = np.array([sigma0, 0.0, 0.0])

        result = identify_failed_faces_uniform_stress(
            nodes_2d, elements, uniform_stress, t, sigma_c, G_c
        )

        # 保存并加载（模拟文件传递）
        from scenarios.suboff_icebreaking.si2_fracture_gen.cohesive_failure import (
            failed_faces_to_dict,
        )
        failed_faces_data = failed_faces_to_dict(result)

        # 生成颗粒
        gen_result = generate_particles_from_failed_faces(
            failed_faces_data, nodes_2d, rho
        )

        # 计算账本
        ledger_a = compute_fem_ledger(failed_faces_data, rho)
        ledger_b = compute_dem_ledger(gen_result)

        # 对账
        report = reconcile_ledgers(ledger_a, ledger_b)

        assert report["status"] == "PASS"
        assert report["mass_conservation"]["relative_error"] <= 1e-12
        assert report["energy_conservation"]["relative_error"] <= 1e-12

    def test_energy_conservation(self):
        """断裂能量守恒。"""
        N = 4
        side = 1.0
        t = 0.01
        rho = 917.0
        sigma_c = 1e5
        G_c = 200.0
        sigma0 = 3e5

        nodes_2d, elements = structured_square_mesh(N, side=side)
        uniform_stress = np.array([sigma0, 0.0, 0.0])

        result = identify_failed_faces_uniform_stress(
            nodes_2d, elements, uniform_stress, t, sigma_c, G_c
        )

        from scenarios.suboff_icebreaking.si2_fracture_gen.cohesive_failure import (
            failed_faces_to_dict,
        )
        failed_faces_data = failed_faces_to_dict(result)

        gen_result = generate_particles_from_failed_faces(
            failed_faces_data, nodes_2d, rho
        )

        ledger_a = compute_fem_ledger(failed_faces_data, rho)
        ledger_b = compute_dem_ledger(gen_result)

        report = reconcile_ledgers(ledger_a, ledger_b)

        assert report["status"] == "PASS"
        assert report["energy_conservation"]["relative_error"] <= 1e-12

    def test_empty_failed_faces(self):
        """无破坏面时账本为零。"""
        failed_faces_data = {"G_c": 100.0, "failed_faces": []}
        nodes_2d = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

        gen_result = generate_particles_from_failed_faces(
            failed_faces_data, nodes_2d, 917.0
        )
        ledger_a = compute_fem_ledger(failed_faces_data, 917.0)
        ledger_b = compute_dem_ledger(gen_result)

        assert gen_result.total_mass == 0.0
        assert gen_result.total_fracture_energy == 0.0
        assert ledger_a["total_mass"] == 0.0
        assert ledger_b["total_mass"] == 0.0


# ---------------------------------------------------------------------------
# 测试 4：等体积球映射
# ---------------------------------------------------------------------------

class TestEqualVolumeSphereMapping:
    """测试等体积球映射。"""

    def test_volume_to_radius(self):
        """体积→半径转换。"""
        V = 1.0
        r = volume_to_sphere_radius(V)
        # V = 4/3 π r³ → r = (3/(4π))^(1/3)
        expected_r = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        assert abs(r - expected_r) < 1e-12

    def test_volume_to_radius_scaling(self):
        """体积缩放→半径缩放。"""
        V1 = 1.0
        V2 = 8.0
        r1 = volume_to_sphere_radius(V1)
        r2 = volume_to_sphere_radius(V2)
        # V2/V1 = 8 → r2/r1 = 2
        assert abs(r2 / r1 - 2.0) < 1e-12

    def test_particle_mass_equals_density_times_volume(self):
        """颗粒质量 = 密度 × 体积。"""
        rho = 917.0
        face_data = {
            "face_id": 0,
            "node_i": 0,
            "node_j": 1,
            "element_left": 0,
            "element_right": 1,
            "length": 0.5,
            "normal": [1.0, 0.0],
            "sigma_n": 2e5,
            "sigma_c": 1e5,
            "area": 0.005,  # 0.5 × 0.01
            "G": 100.0,
            "released_energy": 0.5,  # 100 × 0.005
        }
        failed_faces_data = {"G_c": 100.0, "failed_faces": [face_data]}
        nodes_2d = np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]], dtype=float)

        gen_result = generate_particles_from_failed_faces(
            failed_faces_data, nodes_2d, rho
        )

        assert len(gen_result.particles) == 1
        p = gen_result.particles[0]
        expected_volume = face_data["area"] * face_data["length"]
        expected_mass = rho * expected_volume
        assert abs(p.mass - expected_mass) < 1e-12
        assert abs(p.source_volume - expected_volume) < 1e-12

    def test_particle_radius_from_volume(self):
        """颗粒半径从体积正确计算。"""
        face_data = {
            "face_id": 0,
            "node_i": 0,
            "node_j": 1,
            "element_left": 0,
            "element_right": 1,
            "length": 1.0,
            "normal": [1.0, 0.0],
            "sigma_n": 2e5,
            "sigma_c": 1e5,
            "area": 0.01,
            "G": 100.0,
            "released_energy": 1.0,
        }
        failed_faces_data = {"G_c": 100.0, "failed_faces": [face_data]}
        nodes_2d = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

        gen_result = generate_particles_from_failed_faces(
            failed_faces_data, nodes_2d, 917.0
        )

        p = gen_result.particles[0]
        expected_volume = 0.01 * 1.0  # area × length
        expected_radius = volume_to_sphere_radius(expected_volume)
        assert abs(p.radius - expected_radius) < 1e-12


# ---------------------------------------------------------------------------
# 测试 5：序列化/反序列化
# ---------------------------------------------------------------------------

class TestSerialization:
    """测试文件 I/O。"""

    def test_save_and_load_particles(self):
        """保存和加载颗粒文件。"""
        from scenarios.suboff_icebreaking.si2_fracture_gen.fracture_gen import (
            save_particles,
            load_particles,
        )

        face_data = {
            "face_id": 0,
            "node_i": 0,
            "node_j": 1,
            "element_left": 0,
            "element_right": 1,
            "length": 0.5,
            "normal": [1.0, 0.0],
            "sigma_n": 2e5,
            "sigma_c": 1e5,
            "area": 0.005,
            "G": 100.0,
            "released_energy": 0.5,
        }
        failed_faces_data = {"G_c": 100.0, "failed_faces": [face_data]}
        nodes_2d = np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]], dtype=float)

        gen_result = generate_particles_from_failed_faces(
            failed_faces_data, nodes_2d, 917.0
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "particles.json"
            save_particles(gen_result, str(path))
            loaded = load_particles(str(path))

            assert loaded["n_particles"] == 1
            assert loaded["mapping_strategy"] == "equal_volume_sphere"
            assert abs(loaded["total_mass"] - gen_result.total_mass) < 1e-12

    def test_save_and_load_ledgers(self):
        """保存和加载账本文件。"""
        from scenarios.suboff_icebreaking.si2_fracture_gen.fracture_gen import (
            save_ledger_a,
            save_ledger_b,
            load_ledger,
        )

        ledger_a = {"total_mass": 1.0, "total_fracture_energy": 2.0}
        ledger_b = {"total_mass": 1.0, "total_fracture_energy": 2.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "ledger_a.json"
            path_b = Path(tmpdir) / "ledger_b.json"
            save_ledger_a(ledger_a, str(path_a))
            save_ledger_b(ledger_b, str(path_b))

            loaded_a = load_ledger(str(path_a))
            loaded_b = load_ledger(str(path_b))

            assert loaded_a["total_mass"] == 1.0
            assert loaded_b["total_fracture_energy"] == 2.0
