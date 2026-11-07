"""Command line front end of the V&V machine gate.

Examples
--------
``python -m tensorphysics.vv.gci --obs obs.json``
``python -m tensorphysics.vv.gci --benchmark evidence.json --repo tensorfem \
      --study manufactured --expected-order 2 --json``
``python -m tensorphysics.vv.gci --table vv20_table.json --json``

Exit codes: ``0`` PASS, ``1`` WARN, ``2`` FAIL, ``64`` usage / input error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import adapters
from .gci import FAIL, PASS, WARN, GatePolicy, Observation, evaluate
from .validation import evaluate_validation_table

EXIT_CODES = {PASS: 0, WARN: 1, FAIL: 2}
EXIT_USAGE = 64


def _load_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    return adapters.load_payload(path)


def _observations_from_payload(payload: Any, args: argparse.Namespace) -> tuple[list[Observation], str, dict[str, Any]]:
    """Build observations from either a gate payload or a native benchmark file."""
    if isinstance(payload, dict) and "levels" in payload and not args.benchmark:
        levels = [Observation.from_mapping(item) for item in payload["levels"]]
        quantity = payload.get("quantity") or args.quantity or "observable"
        extra = {k: payload[k] for k in ("expected_order", "validation", "source") if k in payload}
        return levels, quantity, extra
    series = adapters.extract_series(
        payload,
        repo=args.repo,
        quantity=args.quantity,
        study=args.study,
        value_key=args.value_key,
        source_path=args.benchmark or args.obs or "",
    )
    return list(series.levels), series.quantity, {"source": series.as_dict()}


def _policy_from_args(args: argparse.Namespace, expected_order: float | None = None) -> GatePolicy:
    policy = GatePolicy()
    overrides: dict[str, Any] = {
        "p_min": args.p_min,
        "p_max": args.p_max,
        "expected_order": expected_order if expected_order is not None else args.expected_order,
        "order_tolerance": args.order_tolerance,
        "assumed_order": args.assumed_order,
        "strict": True if args.strict else None,
    }
    return policy.merged(**overrides)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _render(verdict, policy: GatePolicy) -> str:
    lines: list[str] = []
    study = verdict.study
    lines.append(f"V&V gate -- quantity: {verdict.quantity}")
    if study is not None:
        lines.append(
            f"  levels={study.n_levels}  uniform_refinement={study.uniform_refinement}  "
            f"refinement_ratios={[round(r, 6) for r in study.refinement_ratios]}"
        )
        rows = [
            ("observed order p", study.order_p, study.order_source),
            ("extrapolated f_exact", study.f_extrapolated, "Richardson"),
            ("finest-grid value", study.f_fine, study.levels[-1].label),
            (f"GCI_fine (Fs={study.safety_factor})", study.gci_fine,
             "relative" if study.gci_is_relative else "absolute"),
            ("convergence ratio R", study.convergence_ratio, study.convergence_type),
            ("power-law residual", study.fit_residual_rel, "relative"),
            ("triplet order spread", study.order_spread, "order consistency"),
            ("noise/signal", study.noise_ratio, "declared sigma"),
        ]
        for name, value, note in rows:
            lines.append(f"  {name:<28} {_fmt(value):>22}   {note}")
        if study.notes:
            lines.append("  notes:")
            lines.extend(f"    - {n}" for n in study.notes)
    if verdict.validation is not None:
        v = verdict.validation
        lines.append(
            f"  V&V20: E={v.error:.6g}  U_val={v.u_val:.6g}  |E|/U_val={v.ratio:.4f}  -> {v.status}"
        )
    lines.append("  checks:")
    for check in verdict.checks:
        lines.append(f"    [{check['status']:<4}] {check['check']}: {check['detail']}")
    lines.append(f"VERDICT: {verdict.status}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tensorphysics.vv.gci",
        description="Richardson/GCI + ASME V&V 20 machine gate for convergence studies.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--obs", help="gate observation JSON (schema tensorphysics.vv.observations/1) or '-'")
    source.add_argument("--benchmark", help="native benchmark JSON from a solver repository")
    source.add_argument("--table", help="ASME V&V 20 table JSON: {\"rows\": [...]}")
    parser.add_argument("--repo", default="auto",
                        help="benchmark dialect: auto|tensorlbm|tensorfvm|tensorfem|tensordem|tensorbem")
    parser.add_argument("--study", default=None, help="sub-study of an evidence document")
    parser.add_argument("--quantity", default=None, help="observable name / value field")
    parser.add_argument("--value-key", default=None, help="explicit observable field name")
    parser.add_argument("--exp", type=float, default=None, help="reference (experimental) value f_exp")
    parser.add_argument("--u-input", type=float, default=None, help="input uncertainty U_input")
    parser.add_argument("--u-num", type=float, default=None, help="override the numerical uncertainty U_num")
    parser.add_argument("--f-sim", type=float, default=None, help="simulation value to validate (default: finest grid)")
    parser.add_argument("--expected-order", type=float, default=None, help="theoretical order of accuracy")
    parser.add_argument("--order-tolerance", type=float, default=None)
    parser.add_argument("--assumed-order", type=float, default=None, help="order for a two-grid study")
    parser.add_argument("--p-min", type=float, default=None)
    parser.add_argument("--p-max", type=float, default=None)
    parser.add_argument("--strict", action="store_true",
                        help="a three-grid certificate cannot PASS (needs >=4 grids or per-level sigma)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--out", default=None, help="write the JSON verdict to this path")
    return parser


def _run_table(path: str, as_json: bool) -> int:
    payload = _load_json(path)
    rows = payload["rows"] if isinstance(payload, dict) else payload
    results = evaluate_validation_table(rows)
    record = {
        "schema": "tensorphysics.vv.vv20-table/1",
        "source": path,
        "rows": [r.as_dict() for r in results],
        "n_pass": sum(1 for r in results if r.passed),
        "n_fail": sum(1 for r in results if not r.passed),
    }
    if as_json:
        print(json.dumps(record, indent=2))
    else:
        print(f"ASME V&V 20 table: {path}")
        print(f"{'label':<16}{'f_sim':>12}{'f_exp':>12}{'U_num':>12}{'U_input':>12}{'U_val':>12}{'|E|':>12}{'ratio':>9}  status")
        for r in results:
            print(
                f"{r.label or '-':<16}{r.f_sim:>12.6g}{r.f_exp:>12.6g}{r.u_num:>12.6g}"
                f"{r.u_input:>12.6g}{r.u_val:>12.6g}{abs(r.error):>12.6g}{r.ratio:>9.4f}  {r.status}"
            )
        print(f"PASS={record['n_pass']} FAIL={record['n_fail']}")
    return EXIT_CODES[FAIL] if record["n_fail"] else EXIT_CODES[PASS]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.table:
            return _run_table(args.table, args.json)
        payload = _load_json(args.obs or args.benchmark)
        levels, quantity, extra = _observations_from_payload(payload, args)
        validation_block = extra.get("validation") or {}
        f_exp = args.exp if args.exp is not None else validation_block.get("f_exp")
        u_input = args.u_input if args.u_input is not None else validation_block.get("u_input")
        u_num = args.u_num if args.u_num is not None else validation_block.get("u_num")
        f_sim = args.f_sim if args.f_sim is not None else validation_block.get("f_sim")
        expected = args.expected_order if args.expected_order is not None else extra.get("expected_order")
        policy = _policy_from_args(args, expected)
        verdict = evaluate(
            levels,
            quantity=quantity,
            policy=policy,
            f_exp=f_exp,
            u_input=u_input,
            u_num=u_num,
            f_sim=f_sim,
        )
    except (adapters.AdapterError, ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    record = verdict.as_dict()
    if extra.get("source"):
        record["adapter"] = extra["source"]
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(record, indent=2))
    else:
        print(_render(verdict, policy))
    return EXIT_CODES[verdict.status]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())