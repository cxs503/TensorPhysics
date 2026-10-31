#!/usr/bin/env python3
"""Hull-ice friction 温度依赖：标定 + 验证一键运行。

用法::

    python examples/run_hull_ice_friction_calibration.py \
        --output artifacts/hull_ice_friction

产物：
* ``artifacts/hull_ice_friction/calibration_report.json``（全部判据数字，程序化生成）
* ``artifacts/hull_ice_friction/figures/*.png``（μ(T) 曲线、机理分解、收敛性、速度敏感性）

流程：文献锚定参考表 → 标定子集辨识 p_real → 5 温度 × 10 重复数值摩擦试验
→ 幂律 μ(T)=μ_0+α|T-T_melt|^β 拟合 → 参考数据逐点比对 → 收敛性研究 → 判决。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorphysics.hull_ice_friction.experiment import FrictionTestConfig  # noqa: E402
from tensorphysics.hull_ice_friction.validation import run_validation  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/hull_ice_friction",
        help="产物目录（默认 artifacts/hull_ice_friction）",
    )
    parser.add_argument("--no-figures", action="store_true", help="跳过 matplotlib 出图")
    parser.add_argument("--repetitions", type=int, default=10, help="每个温度的重复次数")
    args = parser.parse_args(argv)

    config = FrictionTestConfig(n_repetitions=args.repetitions)
    result = run_validation(
        output_dir=args.output, base_config=config, make_figures=not args.no_figures
    )
    report = result.report

    print("=" * 78)
    print("identified real contact pressure p_real = "
          f"{report['constitutive_identification']['value_Pa']:.4e} Pa")
    fit = report["macro_fit"]
    print(f"power law: mu(T) = {fit['mu_0']:.5f} + {fit['alpha']:.6f} * theta^{fit['beta']:.4f}"
          f"   (R^2 = {fit['r2']:.6f})")
    print("-" * 78)
    print(f"{'T (deg C)':>10} {'mu_sim':>9} {'mu_ref':>8} {'rel.err':>9}  role")
    for row in report["reference_comparison"]:
        print(
            f"{row['temperature_C']:>10.1f} {row['mu_simulation']:>9.4f} "
            f"{row['mu_reference']:>8.3f} {row['relative_error'] * 100:>8.2f}%  {row['role']}"
        )
    print("-" * 78)
    for key, crit in report["criteria"].items():
        if key == "criterion_d_pytest":
            continue
        print(f"{key}: passed={crit.get('passed')}")
    print(f"all_machine_criteria_passed = {report['verdict']['all_machine_criteria_passed']}")
    print("=" * 78)
    print(json.dumps({"report": str(result.output_dir / "calibration_report.json")}, indent=2))
    return 0 if report["verdict"]["all_machine_criteria_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())