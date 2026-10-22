"""双账本对账模块 —— 验证 FEM→DEM 断裂生成的质量/能量守恒。

SI-2 断裂生成管线第三步：
- 输入：ledger_a.json (FEM 侧) + ledger_b.json (DEM 侧)
- 输出：对账报告 (consistency_report.json)

判据：
- 锚 1：质量守恒 |Δm|/Σm ≤ 1e-12
- 锚 2：断裂能量守恒 |ΔE|/ΣE ≤ 1e-12
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def reconcile_ledgers(
    ledger_a: Dict,
    ledger_b: Dict,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """对账：比较 FEM 侧账本与 DEM 侧账本。

    Parameters
    ----------
    ledger_a : dict
        账本 A (FEM 侧：核销质量 + 释放断裂能量)。
    ledger_b : dict
        账本 B (DEM 侧：生成颗粒质量 + 断裂能量)。
    tolerance : float
        容差 (默认 1e-12)。

    Returns
    -------
    dict
        对账报告。
    """
    # 质量对账
    mass_a = ledger_a["total_mass"]
    mass_b = ledger_b["total_mass"]
    mass_diff = mass_a - mass_b
    mass_sum = mass_a + mass_b
    mass_rel_error = abs(mass_diff) / mass_sum if mass_sum > 0 else 0.0
    mass_passed = mass_rel_error <= tolerance

    # 断裂能量对账
    energy_a = ledger_a["total_fracture_energy"]
    energy_b = ledger_b["total_fracture_energy"]
    energy_diff = energy_a - energy_b
    energy_sum = energy_a + energy_b
    energy_rel_error = abs(energy_diff) / energy_sum if energy_sum > 0 else 0.0
    energy_passed = energy_rel_error <= tolerance

    # 综合判定
    all_passed = mass_passed and energy_passed

    return {
        "status": "PASS" if all_passed else "FAIL",
        "timestamp": _utcnow_z(),
        "mass_conservation": {
            "ledger_a_mass": mass_a,
            "ledger_b_mass": mass_b,
            "absolute_diff": mass_diff,
            "relative_error": mass_rel_error,
            "criterion": f"|Δm|/Σm ≤ {tolerance}",
            "passed": mass_passed,
        },
        "energy_conservation": {
            "ledger_a_energy": energy_a,
            "ledger_b_energy": energy_b,
            "absolute_diff": energy_diff,
            "relative_error": energy_rel_error,
            "criterion": f"|ΔE|/ΣE ≤ {tolerance}",
            "passed": energy_passed,
        },
    }


def reconcile_ledger_files(
    ledger_a_path: str,
    ledger_b_path: str,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """从文件加载账本并对账。"""
    with open(ledger_a_path, "r", encoding="utf-8") as f:
        ledger_a = json.load(f)
    with open(ledger_b_path, "r", encoding="utf-8") as f:
        ledger_b = json.load(f)

    return reconcile_ledgers(ledger_a, ledger_b, tolerance)


def save_reconciliation_report(report: Dict, output_path: str) -> None:
    """保存对账报告。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def load_reconciliation_report(input_path: str) -> Dict:
    """加载对账报告。"""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
