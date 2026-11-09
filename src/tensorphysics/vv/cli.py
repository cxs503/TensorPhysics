"""Command line front end of the V&V machine gate.

Three output formats share one computation:

* ``--format text``  (default) human-readable report;
* ``--format json``  machine-readable verdict (the ``--json`` alias is kept);
* ``--format table`` fixed-width ASCII tables (metrics + checks) for reports;
* ``--format plot``  PNG figure (``--plot-out``) of the convergence study.

``--table path.json`` is a *data source* (an ASME V&V 20 table); ``--format
table`` is an *output format*.  They are independent and can be combined.

Examples
--------
``python -m tensorphysics.vv.gci --obs obs.json``
``python -m tensorphysics.vv.gci --benchmark evidence.json --repo tensorfem \\
      --study manufactured --expected-order 2 --format json``
``python -m tensorphysics.vv.gci --obs obs.json --format table``
``python -m tensorphysics.vv.gci --obs obs.json --format plot --plot-out gci.png``
``python -m tensorphysics.vv.gci --table vv20_table.json --format plot \\
      --plot-out vv20.png``

Exit codes: ``0`` PASS, ``1`` WARN, ``2`` FAIL, ``3`` WITHHELD, ``4``
UNRESOLVED, ``64`` usage / input error.  ``WITHHELD`` / ``UNRESOLVED`` are the
resolvability statements of the ``R = delta / dx`` law: the discretisation
cannot decide the question, so the number must not be consumed as converged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import adapters
from .gci import FAIL, PASS, UNRESOLVED, WARN, WITHHELD, GatePolicy, Observation, evaluate
from .validation import evaluate_validation_table

EXIT_CODES = {PASS: 0, WARN: 1, WITHHELD: 3, UNRESOLVED: 4, FAIL: 2}
EXIT_USAGE = 64
FORMATS = ("text", "json", "table", "plot")


def _load_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    return adapters.load_payload(path)


def _observations_from_payload(payload: Any, args: argparse.Namespace) -> tuple[list[Observation], str, dict[str, Any]]:
    """Build observations from either a gate payload or a native benchmark file."""
    _extra_keys = ("expected_order", "validation", "source", "features", "dx", "feature_scale")
    if isinstance(payload, dict) and "levels" in payload and not args.benchmark:
        levels = [Observation.from_mapping(item) for item in payload["levels"]]
        quantity = payload.get("quantity") or args.quantity or "observable"
        extra = {k: payload[k] for k in _extra_keys if k in payload}
        return levels, quantity, extra
    series = adapters.extract_series(
        payload,
        repo=args.repo,
        quantity=args.quantity,
        study=args.study,
        value_key=args.value_key,
        source_path=args.benchmark or args.obs or "",
    )
    # A benchmark document may carry its own feature scales and validation block
    # (the cross-repository contract of tensorphysics.vv.schema): forward them so
    # the CLI reproduces the document's V&V statement without extra flags.
    extra: dict[str, Any] = {"source": series.as_dict()}
    if isinstance(payload, dict):
        for key in _extra_keys:
            if key != "source" and key in payload:
                extra[key] = payload[key]
        if "feature_scale" in payload and "features" not in payload:
            extra["features"] = payload["feature_scale"]
    return list(series.levels), series.quantity, extra


def _policy_from_args(args: argparse.Namespace, expected_order: float | None = None) -> GatePolicy:
    policy = GatePolicy()
    overrides: dict[str, Any] = {
        "p_min": args.p_min,
        "p_max": args.p_max,
        "expected_order": expected_order if expected_order is not None else args.expected_order,
        "order_tolerance": args.order_tolerance,
        "assumed_order": args.assumed_order,
        "strict": True if args.strict else None,
        "gci_absolute": True if getattr(args, "gci_absolute", False) else None,
    }
    if getattr(args, "allow_undeclared_noise", False):
        overrides["pass_requires_noise_certificate"] = False
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
    if verdict.resolvability is not None:
        res = verdict.resolvability
        lines.append(
            f"  resolvability: R = delta/dx  min R={_fmt(res.ratio)}  dx={_fmt(res.dx)}  -> {res.status}"
        )
        for featspec in res.features:
            lines.append(f"    - {featspec.detail}")
    if verdict.uncertainty is not None:
        led = verdict.uncertainty
        lines.append(
            f"  uncertainty ledger: U_num={led.u_num:.6g}  U_input={led.u_input:.6g}  "
            f"U_val={led.u_val:.6g}  U_ref={led.u_ref:.6g}  U_ref/U_val={_fmt(led.reference_ratio)}  "
            f"dominant={led.dominant_term()}"
        )
    lines.append("  checks:")
    for check in verdict.checks:
        lines.append(f"    [{check['status']:<4}] {check['check']}: {check['detail']}")
    lines.append(f"VERDICT: {verdict.status}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# table renderer
# ---------------------------------------------------------------------------


def _table(title: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a fixed-width ASCII table (values formatted by ``_fmt``)."""
    cells = [[_fmt(c) for c in row] for row in rows]
    widths = [len(h) for h in header]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [title, sep, "| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |", sep]
    for row in cells:
        out.append("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")
    out.append(sep)
    return "\n".join(out)


def _render_table(verdict, policy: GatePolicy) -> str:
    """ASCII tables: study metrics, then the individual gate checks."""
    study = verdict.study
    parts: list[str] = []
    if study is not None:
        metric_rows = [
            ("levels", study.n_levels, ""),
            ("uniform refinement", study.uniform_refinement,
             ", ".join(f"{r:.6g}" for r in study.refinement_ratios)),
            ("observed order p", study.order_p, study.order_source),
            ("Richardson f_exact", study.f_extrapolated, "extrapolated"),
            ("finest-grid value", study.f_fine, study.levels[-1].label),
            ("GCI_fine", study.gci_fine,
             f"Fs={study.safety_factor}, " + ("relative" if study.gci_is_relative else "absolute")),
            ("convergence ratio R", study.convergence_ratio, study.convergence_type),
            ("power-law residual", study.fit_residual_rel, "relative"),
            ("triplet order spread", study.order_spread, "order consistency"),
            ("noise/signal", study.noise_ratio, "declared sigma"),
            ("noise certified", study.noise_certified, "certificate"),
            ("GCI monotone", study.gci_monotone, "GCI vs refinement"),
        ]
        parts.append(_table(f"V&V gate - {verdict.quantity} - study", ("metric", "value", "note"), metric_rows))
    check_rows = [(c["status"], c["check"], c["detail"]) for c in verdict.checks]
    parts.append(_table("gate checks", ("status", "check", "detail"), check_rows or [("-", "-", "-")]))
    if verdict.validation is not None:
        v = verdict.validation
        parts.append(
            _table(
                "ASME V&V 20 validation",
                ("f_sim", "f_exp", "U_num", "U_input", "U_val", "E", "|E|/U_val", "status"),
                [(v.f_sim, v.f_exp, v.u_num, v.u_input, v.u_val, v.error, v.ratio, v.status)],
            )
        )
    parts.append(f"VERDICT: {verdict.status}")
    return "\n".join(parts)


def _render_vv20_table(record: dict[str, Any]) -> str:
    rows = [
        (r["label"] or "-", r["f_sim"], r["f_exp"], r["u_num"], r["u_input"], r["u_val"],
         r["error"], r["ratio"], r["status"])
        for r in record["rows"]
    ]
    body = _table(
        f"ASME V&V 20 table: {record['source']}",
        ("label", "f_sim", "f_exp", "U_num", "U_input", "U_val", "E", "|E|/U_val", "status"),
        rows,
    )
    return f"{body}\nPASS={record['n_pass']} FAIL={record['n_fail']}"


# ---------------------------------------------------------------------------
# plot renderer
# ---------------------------------------------------------------------------


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "plot output needs matplotlib (pip install 'tensorphysics[plots]')"
        ) from exc
    return plt


def _render_plot(verdict, path: str) -> str:
    """PNG figure: convergence of the observable, Richardson limit and GCI trend."""
    plt = _pyplot()
    study = verdict.study
    if study is None:  # pragma: no cover - evaluate always builds a study
        raise RuntimeError("no study to plot")
    levels = study.levels
    h = [lvl.h for lvl in levels]
    f = [lvl.f for lvl in levels]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.plot(h, f, "o-", color="#1f4e79", lw=1.6, ms=7, label="discrete levels")
    if study.f_extrapolated is not None:
        ax.axhline(study.f_extrapolated, color="#c00000", ls="--", lw=1.4,
                   label=f"Richardson f_exact={study.f_extrapolated:.10g}")
        ax.plot([min(h)], [study.f_extrapolated], "s", color="#c00000", ms=7)
    ax.set_xscale("log")
    ax.set_xlabel("grid length h")
    ax.set_ylabel("observable f")
    order = "-" if study.order_p is None else f"{study.order_p:.4f}"
    ax.set_title(f"observed order p = {order} ({study.order_source})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    if study.gci_pairs:
        idx = list(range(1, len(study.gci_pairs) + 1))
        ax.bar(idx, study.gci_pairs, color="#2e75b6", alpha=0.85)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"pair {i}-{i + 1}" for i in idx], rotation=0, fontsize=8)
        ax.set_ylabel("GCI")
        ax.set_yscale("log")
        ax.set_title(
            f"GCI per grid pair (Fs={study.safety_factor}), "
            f"monotone={study.gci_monotone}"
        )
    else:  # pragma: no cover - a two-level study still has one pair
        ax.text(0.5, 0.5, "no GCI pairs", ha="center", va="center")
    for a in axes:
        a.tick_params(labelsize=8)
    fig.suptitle(f"V&V gate: {verdict.quantity} -> {verdict.status}")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return f"V&V gate plot written to {path} (verdict {verdict.status})"


def _render_vv20_plot(record: dict[str, Any], path: str) -> str:
    """PNG figure of |E|/U_val per table row with the acceptance line at 1."""
    plt = _pyplot()
    rows = record["rows"]
    labels = [r["label"] or f"row{i + 1}" for i, r in enumerate(rows)]
    ratios = [r["ratio"] for r in rows]
    colors = ["#2e7d32" if r["status"] == PASS else "#c62828" for r in rows]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(rows) + 2.0), 3.8))
    ax.bar(range(len(rows)), ratios, color=colors)
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="|E|/U_val = 1 (acceptance)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|E| / U_val")
    ax.set_title("ASME V&V 20 table: comparison error vs validation uncertainty")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return f"ASME V&V 20 plot written to {path} (PASS={record['n_pass']} FAIL={record['n_fail']})"


# ---------------------------------------------------------------------------
# argument parsing / dispatch
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tensorphysics.vv.gci",
        description="Richardson/GCI + ASME V&V 20 machine gate for convergence studies.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--obs", help="gate observation JSON (schema tensorphysics.vv.observations/1) or '-'")
    source.add_argument("--benchmark", help="native benchmark JSON from a solver repository")
    source.add_argument("--table", "--vv20-table", dest="table",
                        help="input ASME V&V 20 table JSON: {\"rows\": [...]}")
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
    parser.add_argument("--gci-absolute", action="store_true",
                        help="report the absolute GCI band (correct for error-norm "
                             "observables, whose relative band is unbounded)")
    parser.add_argument("--strict", action="store_true",
                        help="a three-grid certificate cannot PASS (needs >=4 grids or per-level sigma)")
    parser.add_argument("--allow-undeclared-noise", action="store_true",
                        help="restore an unqualified PASS for a study whose noise floor is not "
                             "certified (default: such a study is capped at WARN)")
    parser.add_argument("--format", choices=FORMATS, default=None,
                        help="output format: text (default), json, table, plot")
    parser.add_argument("--json", action="store_true", help="alias of --format json")
    parser.add_argument(
        "--feature-scale",
        action="append",
        default=None,
        metavar="NAME=DELTA",
        help="physical feature scale delta (repeatable), e.g. --feature-scale boundary_layer=0.004",
    )
    parser.add_argument("--dx", type=float, default=None,
                        help="finest grid spacing for the R=delta/dx law (default: finest level h)")
    parser.add_argument("--min-cells-per-feature", type=float, default=None,
                        help="resolution floor of the R law (default 4.0)")
    parser.add_argument("--require-resolvability", action="store_true",
                        help="withhold the verdict unless a feature scale is declared")
    parser.add_argument("--u-ref", type=float, default=None,
                        help="reference uncertainty U_ref of the comparison benchmark")
    parser.add_argument("--plot-out", default=None,
                        help="PNG path for --format plot (default: vv_gate.png)")
    parser.add_argument("--out", default=None, help="write the JSON verdict to this path")
    return parser


def _resolve_format(args: argparse.Namespace) -> str:
    if args.format:
        return args.format
    if args.json:
        return "json"
    return "text"


def _run_table(path: str, fmt: str, plot_out: str | None) -> int:
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
    try:
        if fmt == "json":
            print(json.dumps(record, indent=2))
        elif fmt == "plot":
            print(_render_vv20_plot(record, plot_out or "vv20_table.png"))
        else:
            print(_render_vv20_table(record))
    except RuntimeError as exc:  # plotting dependency missing
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_CODES[FAIL] if record["n_fail"] else EXIT_CODES[PASS]


def _features_from_args(args: argparse.Namespace, extra: dict[str, Any]):
    """Feature specs from ``--feature-scale NAME=DELTA`` and/or the payload."""
    from .resolvability import FeatureSpec

    specs = []
    for item in extra.get("features") or ():
        specs.append(FeatureSpec(**item) if isinstance(item, dict) else item)
    for token in args.feature_scale or ():
        if "=" not in token:
            raise ValueError(f"--feature-scale expects NAME=DELTA, got {token!r}")
        name, delta = token.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--feature-scale needs a feature name, got {token!r}")
        specs.append(FeatureSpec(name=name, delta=float(delta)))
    return specs


def _resolvability_policy(args: argparse.Namespace):
    from .resolvability import DiscriminabilityPolicy

    if args.min_cells_per_feature is None:
        return None
    return DiscriminabilityPolicy(min_cells_per_feature=args.min_cells_per_feature)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    fmt = _resolve_format(args)
    try:
        if args.table:
            return _run_table(args.table, fmt, args.plot_out)
        payload = _load_json(args.obs or args.benchmark)
        levels, quantity, extra = _observations_from_payload(payload, args)
        validation_block = extra.get("validation") or {}
        f_exp = args.exp if args.exp is not None else validation_block.get("f_exp")
        u_input = args.u_input if args.u_input is not None else validation_block.get("u_input")
        u_num = args.u_num if args.u_num is not None else validation_block.get("u_num")
        f_sim = args.f_sim if args.f_sim is not None else validation_block.get("f_sim")
        expected = args.expected_order if args.expected_order is not None else extra.get("expected_order")
        policy = _policy_from_args(args, expected)
        features = _features_from_args(args, extra)
        u_ref = args.u_ref if args.u_ref is not None else validation_block.get("u_ref")
        verdict = evaluate(
            levels,
            quantity=quantity,
            policy=policy,
            f_exp=f_exp,
            u_input=u_input,
            u_num=u_num,
            f_sim=f_sim,
            features=features or None,
            dx=args.dx if args.dx is not None else extra.get("dx"),
            resolvability_policy=_resolvability_policy(args),
            require_resolvability=bool(args.require_resolvability),
            u_ref=u_ref,
        )
    except (adapters.AdapterError, ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except RuntimeError as exc:  # plotting dependency missing
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    record = verdict.as_dict()
    if extra.get("source"):
        record["adapter"] = extra["source"]
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    try:
        if fmt == "json":
            print(json.dumps(record, indent=2))
        elif fmt == "table":
            print(_render_table(verdict, policy))
        elif fmt == "plot":
            print(_render_plot(verdict, args.plot_out or "vv_gate.png"))
            print(_render(verdict, policy))
        else:
            print(_render(verdict, policy))
    except RuntimeError as exc:  # plotting dependency missing
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_CODES[verdict.status]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())