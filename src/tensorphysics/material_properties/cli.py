"""Command-line interface for MPDB v1.

Usage examples
--------------
::

    python -m tensorphysics.material_properties.cli list
    python -m tensorphysics.material_properties.cli describe --substance seawater
    python -m tensorphysics.material_properties.cli query --substance seawater \
        --property density --T-C 15 --salinity 35
    python -m tensorphysics.material_properties.cli query --substance seawater \
        --property kinematic_viscosity --T-C 15 --salinity 35 --json
    python -m tensorphysics.material_properties.cli table --substance air --property density
    python -m tensorphysics.material_properties.cli validate
    python -m tensorphysics.material_properties.cli build --check
    python -m tensorphysics.material_properties.cli inventory --repo /path/to/repo --out DIR

Exit codes: 0 success, 1 validation/anchor failure, 2 query/lookup error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import adapters
from .build import build, build_records, check_committed
from .inventory import scan_repo, summarise, to_csv, to_markdown
from .query import (
    AmbiguousValueError,
    MaterialDB,
    MPDBError,
    MissingConditionError,
    OutOfDomainError,
    UnknownPropertyError,
    UnknownSubstanceError,
)
from .schema import SCHEMA_VERSION, validate_records
from .sources import ANCHORS, SOURCE_REGISTRY


def _db() -> MaterialDB:
    return MaterialDB.load()


def _print(obj: Any) -> None:
    if isinstance(obj, str):
        print(obj)
    else:
        print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_list(args: argparse.Namespace) -> int:
    db = _db()
    rows = []
    for name, meta in db.substances().items():
        if args.substance_class and meta.get("class") != args.substance_class:
            continue
        rows.append({
            "substance": name,
            "class": meta.get("class"),
            "label": meta.get("label"),
            "rows": len(db.table(name, db.props_for(name)[0])) if db.props_for(name) else 0,
            "properties": db.props_for(name),
        })
    if args.json:
        _print(rows)
    else:
        print(f"MPDB {SCHEMA_VERSION}: {len(rows)} substances shown, "
              f"{sum(r['rows'] for r in rows)} rows in the first property of each")
        for r in rows:
            print(f"  {r['substance']:<16} [{r['class']:<12}] {r['label']}")
            print(f"    properties: {', '.join(r['properties'])}")
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    db = _db()
    substances = db.substances()
    if args.substance not in substances:
        print(f"unknown substance {args.substance!r}; known: {', '.join(sorted(substances))}",
              file=sys.stderr)
        return 2
    meta = substances[args.substance]
    out: Dict[str, Any] = {
        "substance": args.substance,
        "class": meta.get("class"),
        "label": meta.get("label"),
        "use_cases": meta.get("use_cases", []),
        "properties": {},
    }
    for prop in db.props_for(args.substance):
        rows = db.table(args.substance, prop)
        out["properties"][prop] = {
            "rows": len(rows),
            "unit": rows[0]["unit"],
            "variants": sorted({r.get("variant") for r in rows if r.get("variant")}),
            "domain": db.domain(args.substance, prop),
            "provenance_sources": sorted({r["provenance"]["source_id"] for r in rows}),
            "evidence_classes": sorted({r["provenance"]["evidence_class"] for r in rows}),
            "uncertainty_rel": sorted({r["uncertainty"]["rel"] for r in rows if r["uncertainty"]["rel"]}),
        }
    _print(out)
    return 0


def cmd_props(args: argparse.Namespace) -> int:
    db = _db()
    try:
        props = db.props_for(args.substance)
    except UnknownSubstanceError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        _print(props)
    else:
        for p in props:
            print(p)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    db = _db()
    props: List[str] = list(args.property or [])
    if not props and _has_substance(db, args.substance):
        props = db.props_for(args.substance)
    if not props:
        print("no property selected", file=sys.stderr)
        return 2
    results: Dict[str, Any] = {}
    rc = 0
    for prop in props:
        try:
            pv = db.get(args.substance, prop, T=args.t, p=args.p, salinity=args.salinity,
                        T_C=args.t_celsius, variant=args.variant, clamp=args.clamp)
            results[prop] = pv.to_dict()
            if not args.json:
                cond = ", ".join(f"{k}={v:g}" for k, v in pv.conditions.items() if v is not None)
                print(f"{args.substance}.{prop} @ {cond or 'standard'} = "
                      f"{pv.value:.6g} {pv.unit}")
                ev = pv.provenance.get("evidence_class")
                print(f"    [{ev}] {pv.provenance.get('source_id')} / {pv.provenance.get('locator')}")
                if pv.uncertainty_rel is not None:
                    print(f"    uncertainty: {pv.uncertainty_rel * 100:.3g}% "
                          f"({pv.uncertainty_basis})")
                if pv.interpolated:
                    print(f"    interpolation: {pv.interpolated}")
        except (OutOfDomainError, MissingConditionError, AmbiguousValueError,
                UnknownSubstanceError, UnknownPropertyError) as exc:
            print(f"{prop}: {type(exc).__name__}: {exc}", file=sys.stderr)
            rc = 2
    if args.json:
        _print(results)
    return rc


def _has_substance(db: MaterialDB, substance: str) -> bool:
    return substance in db.substances()


def cmd_table(args: argparse.Namespace) -> int:
    db = _db()
    try:
        rows = db.table(args.substance, args.property, variant=args.variant)
    except (UnknownSubstanceError, UnknownPropertyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not rows:
        print("no rows", file=sys.stderr)
        return 2
    if args.csv:
        cols = ["id", "value", "unit", "T_K", "p_Pa", "salinity_g_per_kg", "variant",
                "source_id", "locator", "evidence_class", "uncertainty_rel"]
        print(",".join(cols))
        for r in rows:
            c = r["conditions"]
            vals = [
                r["id"], r["value"], r["unit"],
                "" if c.get("T_K") is None else c["T_K"],
                "" if c.get("p_Pa") is None else c["p_Pa"],
                "" if c.get("salinity_g_per_kg") is None else c["salinity_g_per_kg"],
                r.get("variant") or "",
                r["provenance"]["source_id"], r["provenance"]["locator"],
                r["provenance"]["evidence_class"],
                "" if r["uncertainty"]["rel"] is None else r["uncertainty"]["rel"],
            ]
            print(",".join(str(v).replace(",", ";") for v in vals))
        return 0
    if args.json:
        _print(rows)
        return 0
    print(f"{len(rows)} rows for {args.substance}.{args.property} [{rows[0]['unit']}]")
    for r in rows:
        c = {k: v for k, v in r["conditions"].items() if v is not None}
        cond = ", ".join(f"{k}={v:g}" for k, v in c.items())
        v = r.get("variant")
        print(f"  {r['value']:>14.6g}  {cond}{' ' + v if v else ''}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    records = build_records()
    errors = validate_records(records, SOURCE_REGISTRY)
    anchors_env = build(write=False)["anchors"]
    failed = [a for a in anchors_env if not a["passed"]]
    consistency = [adapters.seawater_consistency_check(T_C=t, salinity=35.0)
                   for t in (0.0, 15.0, 30.0)]
    report = {
        "schema_version": SCHEMA_VERSION,
        "rows": len(records),
        "schema_errors": errors,
        "classes": sorted({r["substance_class"] for r in records}),
        "substances": sorted({r["substance"] for r in records}),
        "sources": len(SOURCE_REGISTRY),
        "anchors_total": len(anchors_env),
        "anchors_failed": [a["id"] for a in failed],
        "anchor_max_rel_error": max(a["rel_error"] or 0.0 for a in anchors_env),
        "seawater_row_vs_correlation_max_dev": max(
            max(c["rho_rel_dev"], c["mu_rel_dev"]) for c in consistency),
        "ok": not errors and not failed,
    }
    _print(report)
    return 0 if report["ok"] else 1


def cmd_build(args: argparse.Namespace) -> int:
    if args.check:
        res = check_committed()
        _print(res)
        return 0 if res["ok"] else 1
    out = build()
    meta = out["meta"]
    _print({k: meta[k] for k in ("db", "schema_version", "rows", "substances",
                                 "substance_classes", "sources", "rows_with_provenance",
                                 "anchors_total", "anchors_passed", "jsonl_sha256",
                                 "repro_cmd") if k in meta})
    return 0


def cmd_inventory(args: argparse.Namespace) -> int:
    if not os.path.isdir(args.repo):
        print(f"--repo {args.repo!r} is not a directory", file=sys.stderr)
        return 2
    rows = scan_repo(args.repo)
    summary = summarise(rows, os.path.abspath(args.repo))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "property_carriers.csv"), "w", encoding="utf-8") as fh:
            fh.write(to_csv(rows))
        with open(os.path.join(args.out, "property_carriers.md"), "w", encoding="utf-8") as fh:
            fh.write(to_markdown(rows, summary))
        with open(os.path.join(args.out, "property_carriers.json"), "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    if args.json:
        _print({"summary": summary, "rows": rows})
    else:
        _print(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tensorphysics.material_properties.cli",
                                description="MPDB v1 - material & fluid property database")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list", help="list substances (and their properties)")
    s.add_argument("--class", dest="substance_class", default=None,
                   help="filter by substance class (liquid/gas/ice/sediment/metal/composite/cementitious)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("describe", help="describe one substance (domains, sources, uncertainties)")
    s.add_argument("--substance", required=True)
    s.set_defaults(func=cmd_describe)

    s = sub.add_parser("props", help="list properties of a substance")
    s.add_argument("--substance", required=True)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_props)

    s = sub.add_parser("query", help="query one or more properties")
    s.add_argument("--substance", required=True)
    s.add_argument("--property", action="append", default=None,
                   help="repeatable; default = every property of the substance")
    s.add_argument("--T", dest="t", type=float, default=None, help="absolute temperature [K]")
    s.add_argument("--T-C", dest="t_celsius", type=float, default=None,
                   help="temperature in degC (access unit, converted to K)")
    s.add_argument("--p", dest="p", type=float, default=None, help="pressure [Pa]")
    s.add_argument("--salinity", type=float, default=None, help="salinity [g/kg]")
    s.add_argument("--variant", default=None, help="state label (e.g. first_year, effective)")
    s.add_argument("--clamp", action="store_true",
                   help="saturate at the nearest sampled state instead of raising")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_query)

    s = sub.add_parser("table", help="dump the rows of one property")
    s.add_argument("--substance", required=True)
    s.add_argument("--property", required=True)
    s.add_argument("--variant", default=None)
    s.add_argument("--csv", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_table)

    s = sub.add_parser("validate", help="validate the built database (schema + anchors + consistency)")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("build", help="rebuild the packaged database")
    s.add_argument("--check", action="store_true", help="verify the committed artefacts instead")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("inventory", help="stocktake scattered property carriers in a consumer repo")
    s.add_argument("--repo", required=True)
    s.add_argument("--out", default=None, help="write csv/md/json artefacts here")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_inventory)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except MPDBError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())