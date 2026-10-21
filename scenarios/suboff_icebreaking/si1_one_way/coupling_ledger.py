"""双账本对账模块 —— 验证 LBM→FEM 单向耦合的力传递守恒性。

SI-1 单向耦合管线第三步：
- 输入：ledger_a.json + ledger_b.json
- 输出：对账报告 (consistency_report.json)

判据：
- 锚 1：max|A−B| ≤ 1e-12 (机器精度级守恒)
- 总力守恒：|sum(A) - sum(B)| ≤ 1e-12
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def reconcile_ledgers(
    ledger_a: Dict,
    ledger_b: Dict,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """对账：比较账本 A 与账本 B 的节点力。

    Parameters
    ----------
    ledger_a : dict
        账本 A (LBM 发出的力)。
    ledger_b : dict
        账本 B (FEM 收到的力)。
    tolerance : float
        容差 (默认 1e-12)。

    Returns
    -------
    dict
        对账报告。
    """
    forces_a = np.array(ledger_a["forces"])
    forces_b = np.array(ledger_b["forces"])

    if forces_a.shape != forces_b.shape:
        return {
            "status": "FAIL",
            "reason": f"shape mismatch: A={forces_a.shape}, B={forces_b.shape}",
            "timestamp": _utcnow_z(),
        }

    # 逐节点力对比
    diff = forces_a - forces_b
    max_abs_diff = float(np.max(np.abs(diff)))
    total_force_a = forces_a.sum(axis=0)
    total_force_b = forces_b.sum(axis=0)
    total_force_diff = total_force_a - total_force_b

    passed = max_abs_diff <= tolerance

    return {
        "status": "PASS" if passed else "FAIL",
        "anchor": "anchor_1: 传递守恒账本",
        "criterion": f"max|A−B| ≤ {tolerance}",
        "max_abs_diff": max_abs_diff,
        "total_force_A": total_force_a.tolist(),
        "total_force_B": total_force_b.tolist(),
        "total_force_diff": total_force_diff.tolist(),
        "n_nodes": int(forces_a.shape[0]),
        "timestamp": _utcnow_z(),
    }


def reconcile_ledger_files(
    ledger_a_path: str,
    ledger_b_path: str,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """从文件加载账本并对账。"""
    with open(ledger_a_path, "r") as f:
        ledger_a = json.load(f)
    with open(ledger_b_path, "r") as f:
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
