"""SIR-3 契约测试：G3 重力/浮力/自由面入链

测试内容：
1. 验证 run_validation.py 生成的工件完整性
2. 验证三元组可复算
3. 验证语义判据与契约一致
"""

import json
import pytest
from pathlib import Path


SIR3_DIR = Path(__file__).parent.parent / "scenarios" / "suboff_icebreaking" / "sir3_buoyancy_freesurface"
PREREG_DIR = Path(__file__).parent.parent / "scenarios" / "suboff_icebreaking" / "prereg"


def test_sir3_validation_report_exists():
    """验证报告文件存在"""
    report_path = SIR3_DIR / "validation_report.json"
    assert report_path.exists(), f"验证报告不存在: {report_path}"


def test_sir3_gates_json_exists():
    """三元组文件存在"""
    gates_path = SIR3_DIR / "gates.json"
    assert gates_path.exists(), f"三元组文件不存在: {gates_path}"


def test_sir3_validation_report_schema():
    """验证报告 schema 完整性"""
    report_path = SIR3_DIR / "validation_report.json"
    with open(report_path) as f:
        report = json.load(f)
    
    assert report["schema"] == "tensorphysics-suboff-sir3-validation-v1"
    assert report["mission"] == "m308"
    assert report["leg"] == "SIR-3"
    assert report["gap"] == "G3"
    assert report["prereg_md5"] == "47c72ea0ec404a7f3cb31ad8ba13891d"
    assert "tests" in report
    assert "gates" in report
    assert "semantic_criteria" in report


def test_sir3_gates_triplets_recomputable():
    """三元组可复算：值-文件-键直读验证"""
    report_path = SIR3_DIR / "validation_report.json"
    gates_path = SIR3_DIR / "gates.json"
    
    with open(report_path) as f:
        report = json.load(f)
    
    with open(gates_path) as f:
        gates = json.load(f)
    
    # 验证每个三元组可复算
    for triplet in gates["triplets"]:
        gate_id = triplet["id"]
        value = triplet["value"]
        file_ref = triplet["file"]
        key_path = triplet["key"]
        
        # 从报告中复算值
        keys = key_path.split(".")
        recomputed = report
        for k in keys:
            recomputed = recomputed[k]
        
        assert recomputed == value, f"三元组 {gate_id} 不可复算: {recomputed} != {value}"


def test_sir3_machine_gates_pass():
    """机器门全部通过"""
    report_path = SIR3_DIR / "validation_report.json"
    with open(report_path) as f:
        report = json.load(f)
    
    gates = report["gates"]
    assert gates["M3.1_gravity_enabled"] is True
    assert gates["M3.2_buoyancy_nonzero"] is True
    assert gates["M3.3_free_surface_present"] is True
    assert gates["M3.4_mass_ledger"] is True
    assert gates["M3.5_archimedes"] is True
    
    assert report["gates_passed"] == 5
    assert report["gates_total"] == 5
    assert report["overall_pass"] is True


def test_sir3_semantic_criteria_alignment():
    """语义判据与契约一致"""
    report_path = SIR3_DIR / "validation_report.json"
    criteria_path = PREREG_DIR / "semantic_criteria.json"
    
    with open(report_path) as f:
        report = json.load(f)
    
    with open(criteria_path) as f:
        criteria = json.load(f)
    
    # 契约定义的 SIR-3 语义判据
    sir3_criteria = criteria["gates"]["SIR-3"]
    expected_ids = {c["id"] for c in sir3_criteria}
    
    # 报告中实现的语义判据
    actual_ids = set(report["semantic_criteria"].keys())
    
    assert expected_ids == actual_ids, f"语义判据不一致: {expected_ids} != {actual_ids}"
    
    # 全部通过
    for cid in expected_ids:
        assert report["semantic_criteria"][cid] is True, f"语义判据 {cid} 未通过"


def test_sir3_prereg_frozen():
    """prereg 已冻结（MD5 匹配）"""
    prereg_path = PREREG_DIR / "sir3_prereg.md"
    assert prereg_path.exists()
    
    import hashlib
    with open(prereg_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    
    assert md5 == "47c72ea0ec404a7f3cb31ad8ba13891d", "prereg MD5 不匹配"


def test_sir3_gravity_enabled_value():
    """M3.1 重力开关值验证"""
    report_path = SIR3_DIR / "validation_report.json"
    with open(report_path) as f:
        report = json.load(f)
    
    gravity_test = report["tests"]["gravity_enabled"]
    assert gravity_test["gravity_enabled"] is True
    assert gravity_test["gravity_magnitude"] == 9.81
    assert gravity_test["hydrostatic_buoyancy"] is True
    assert gravity_test["gravity_vector"] == [0.0, 0.0, -9.81]


def test_sir3_buoyancy_archimedes_accuracy():
    """M3.5 浮力对阿基米德解析解精度"""
    report_path = SIR3_DIR / "validation_report.json"
    with open(report_path) as f:
        report = json.load(f)
    
    buoyancy_test = report["tests"]["buoyancy"]
    assert buoyancy_test["buoyancy_net_force_N"] >= 1e-6
    assert buoyancy_test["relative_error"] < 0.01
    assert buoyancy_test["archimedes_force_N"] > 0


def test_sir3_free_surface_present():
    """M3.3 自由面在场"""
    report_path = SIR3_DIR / "validation_report.json"
    with open(report_path) as f:
        report = json.load(f)
    
    fs_test = report["tests"]["free_surface"]
    assert fs_test["n_interface_cells"] > 0
    assert fs_test["vertical_cells"] >= 32


def test_sir3_mass_ledger_precision():
    """M3.4 质量双账本精度"""
    report_path = SIR3_DIR / "validation_report.json"
    with open(report_path) as f:
        report = json.load(f)
    
    fs_test = report["tests"]["free_surface"]
    assert fs_test["mass_relative_error"] <= 1e-12


def test_sir3_scale_criteria_documented():
    """规模判据已记录"""
    report_path = SIR3_DIR / "validation_report.json"
    with open(report_path) as f:
        report = json.load(f)
    
    assert "scale_criteria" in report
    assert "scale_criteria_passed" in report
    assert "scale_criteria_total" in report
    
    # 至少记录了 3 个规模判据
    assert report["scale_criteria_total"] >= 3


def test_sir3_no_solver_code_in_scenario():
    """场景目录无求解器代码（薄编排原则）"""
    scenario_dir = SIR3_DIR
    
    # 检查是否有 .py 文件包含求解器实现
    for py_file in scenario_dir.glob("*.py"):
        if py_file.name == "__init__.py" or py_file.name == "run_validation.py":
            continue
        
        with open(py_file) as f:
            content = f.read()
        
        # 不应该包含求解器核心实现
        assert "def free_surface_step" not in content
        assert "def buoyancy_response" not in content
        assert "class IceSimulationConfig" not in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
