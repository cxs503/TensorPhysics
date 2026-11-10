"""Command line entry point: ``python -m tensorphysics.ice_encounter_stats``.

Runs AC1..AC6 of node P233W2-N05 against the pre-registered manifest, writes
``acceptance.json`` / ``verdict.json`` / ``report.md`` / ``figures/`` under
``--outdir`` and exits non-zero when the verdict is not ``PASS``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from .acceptance import run_acceptance
from .config import load_manifest
from .prereg import summarize


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tensorphysics.ice_encounter_stats")
    parser.add_argument("--manifest", default=None,
                        help="pre-registration manifest (default: packaged one)")
    parser.add_argument("--outdir", default="artifacts/p233w2_n05",
                        help="artifact directory for results, figures and report")
    parser.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    parser.add_argument("--no-write", action="store_true", help="do not touch the filesystem")
    parser.add_argument("--quiet", action="store_true", help="suppress the verdict summary")
    parser.add_argument("--docs-copy", default=None,
                        help="also write the report markdown to this path "
                             "(figure links are rewritten relative to it)")
    args = parser.parse_args(argv)

    manifest = load_manifest(pathlib.Path(args.manifest) if args.manifest else None)
    result = run_acceptance(manifest, outdir=args.outdir,
                            make_figures=not args.no_figures,
                            write=not args.no_write)
    if args.docs_copy:
        import pathlib as _pathlib

        report_path = _pathlib.Path(args.outdir) / "report.md"
        target = _pathlib.Path(args.docs_copy)
        target.parent.mkdir(parents=True, exist_ok=True)
        rel = _pathlib.Path(*([".."] * max(len(target.parent.parts), 1))) / args.outdir / "figures"
        markdown = report_path.read_text(encoding="utf-8")
        markdown = markdown.replace("](figures/", f"]({rel.as_posix()}/")
        target.write_text(markdown, encoding="utf-8")
        if not args.quiet:
            print(f"docs copy -> {target}")
    if not args.quiet:
        print(summarize(result.verdict))
        print(f"elapsed {result.elapsed_s:.1f} s -> {result.outdir}")
    return 0 if result.verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())