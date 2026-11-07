#!/usr/bin/env python3
"""Cross-architecture verification of the IN16-K8 clock semantics (stdlib only).

The loong64 compute nodes ship a bare CPython (no numpy/scipy), so importing
``tensorphysics`` normally fails in ``tensorphysics/__init__`` (which pulls in the
ice-sheet FEM stack).  The clock semantics themselves are pure stdlib, so this
harness loads the two relevant subpackages by path -- bypassing the optional
dependencies -- and re-runs the exactness/order/band checks.

On any machine:  python3 scripts/verify_m5044_crossarch.py [src_dir]
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

_SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "src")


def _load_by_path() -> None:
    """Register namespace stubs so the submodules import without the heavy __init__."""
    for name, rel in (("tensorphysics", "tensorphysics"), ("tensorphysics.campaign", "tensorphysics/campaign")):
        module = types.ModuleType(name)
        module.__path__ = [str(_SRC / rel)]  # type: ignore[attr-defined]
        sys.modules[name] = module


def main() -> int:
    _load_by_path()
    from tensorphysics.campaign.clock_rational import RationalClockSpec, verify_gcd_cadence
    from tensorphysics.campaign.multirate import (
        BAND_ZOH_COEF,
        ORDER_WINDOW,
        TwoIntegratorModel,
        band_scan,
        interpolation_scan,
        refinement_scan,
    )

    verify: dict[str, object] = {}
    for key, dts, periods in (
        ("ratio_4_1", (0.004, 0.001), 1),
        ("ratio_3_2", (0.003, 0.002), 1),
        ("ratio_4_3_x5", (0.004, 0.003), 5),
        ("ratio_8_3", (0.02, 0.0075), 1),
    ):
        spec = RationalClockSpec("master", "sub", *dts, exchange_periods=periods)
        report = verify_gcd_cadence(spec, master_ticks=spec.master_ticks_per_exchange * 12)
        verify[key] = {
            "ratio_exact": f"{spec.ratio.numerator}/{spec.ratio.denominator}",
            "atom_exact": f"{spec.atom_s.numerator}/{spec.atom_s.denominator}",
            "exchange_interval_exact": f"{spec.exchange_interval_s.numerator}/{spec.exchange_interval_s.denominator}",
            "verified": report["verified"],
            "max_abs_residual_s": report["max_abs_residual_s"],
        }

    model = TwoIntegratorModel()
    refine = refinement_scan(model, levels=4)
    bands = band_scan(model, periods=(1, 2, 4, 8, 16), base_intervals=256)
    interp = interpolation_scan(model)

    payload = {
        "python": sys.version.split()[0],
        "platform_machine": __import__("platform").machine(),
        "gcd_cadence": verify,
        "all_gcd_verified": all(bool(row["verified"]) for row in verify.values()),
        "multirate_order": refine["order"],
        "multirate_order_window": list(ORDER_WINDOW),
        "multirate_order_ok": refine["order_ok"],
        "band_zoh_coef": BAND_ZOH_COEF,
        "band_all_within_refinement": refine["all_within_band"],
        "band_worst_ratio": refine["worst_band_ratio"],
        "band_all_within_decade": bands["all_within_band"],
        "interp_zoh_order": interp["order_zoh"],
        "interp_linear_order": interp["order_linear"],
        "interp_linear_order_ok": interp["linear_order_ok"],
        "interp_linear_rejected_non_causal": interp["linear_rejected_as_non_causal"],
    }
    payload["verdict"] = (
        "PASS"
        if payload["all_gcd_verified"]
        and payload["multirate_order_ok"]
        and payload["band_all_within_refinement"]
        and payload["band_all_within_decade"]
        and payload["interp_linear_order_ok"]
        else "FAIL"
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())