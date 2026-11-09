#!/usr/bin/env python3
"""Shard worker for a cluster-dispatched UQ evaluation.

Two modes, both driven by the *pinned* design ``(spec, rule, seed, n)`` so that
a shard recomputes exactly the rows it is asked for instead of shipping the
sample matrix around:

.. code-block:: shell

    # evaluate shard 0 of 2 on one node (the {-i}/{n} placeholders of run_cluster)
    python scripts/uq_shard_worker.py --benchmark holtrop --n 256 --seed 20240 \\
        --shard 0/2 --out shard_0.npz
    # merge the shards, refusing any shard whose sample-set hash differs
    python scripts/uq_shard_worker.py --merge shard_0.npz shard_1.npz \\
        --out merged.npz

The rendered ``run_cluster`` template is produced by
:func:`tensorphysics.uq.adapter.cluster_shard_command`, e.g.::

    python scripts/uq_shard_worker.py --benchmark holtrop --n 256 --seed 20240 \
        --shard {i}/{n} --out shard_{i}.npz

Every shard file carries ``values`` (float64, NaN for failed rows),
``indices`` (global row numbers), ``sample_hash`` and ``spec_hash``; the merge
therefore reconstructs the full result vector in the original order and can
only succeed if all shards were produced from the same design.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.uq.adapter import SampleHashMismatch  # noqa: E402
from tensorphysics.uq.benchmarks import (  # noqa: E402
    holtrop_type_benchmark,
    ishigami,
    ishigami_spec,
    sobol_g,
    sobol_g_spec,
)
from tensorphysics.uq.sampler import sample  # noqa: E402

BENCHMARKS = ("holtrop", "ishigami", "gfunction")


def build_benchmark(name: str, *, rel_uncertainty: float = 0.01):
    """Return ``(spec, evaluate)`` for a named benchmark."""
    if name == "holtrop":
        bench = holtrop_type_benchmark(rel_uncertainty=rel_uncertainty)
        return bench.spec, bench.resistance_kn
    if name == "ishigami":
        return ishigami_spec(), ishigami
    if name == "gfunction":
        a = np.array([0.5, 1.0, 1.5, 2.0])
        return sobol_g_spec(4), (lambda points: sobol_g(points, a))
    raise SystemExit(f"unknown benchmark {name!r}; expected one of {BENCHMARKS}")


def _parse_shard(text: str) -> tuple[int, int]:
    try:
        i, n = text.split("/")
        index, count = int(i), int(n)
    except ValueError as exc:  # pragma: no cover - CLI guard
        raise SystemExit(f"--shard must look like 0/4, got {text!r}") from exc
    if not (0 <= index < count):
        raise SystemExit(f"shard index {index} outside [0, {count})")
    return index, count


def _shard_bounds(n: int, index: int, count: int) -> tuple[int, int]:
    edges = np.linspace(0, n, count + 1).astype(int)
    return int(edges[index]), int(edges[index + 1])


def evaluate_shard(args: argparse.Namespace) -> int:
    spec, evaluate = build_benchmark(args.benchmark, rel_uncertainty=args.rel_uncertainty)
    design = sample(spec, args.n, rule=args.rule, seed=args.seed)
    index, count = _parse_shard(args.shard)
    lo, hi = _shard_bounds(design.n, index, count)
    rows = np.arange(lo, hi)
    points = design.points[rows]
    values = np.asarray(evaluate(points), dtype=float).reshape(-1)
    if values.size != rows.size:
        raise SystemExit(
            f"benchmark returned {values.size} values for {rows.size} rows"
        )
    out = pathlib.Path(args.out)
    np.savez(
        out,
        values=values,
        indices=rows,
        sample_hash=np.array(design.hash),
        spec_hash=np.array(spec.fingerprint()),
        benchmark=np.array(args.benchmark),
        rule=np.array(args.rule),
        seed=np.array(args.seed),
        n=np.array(design.n),
        shard=np.array(args.shard),
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "benchmark": args.benchmark,
                "shard": args.shard,
                "rows": [lo, hi],
                "sample_hash": design.hash,
                "n_nonfinite": int(np.count_nonzero(~np.isfinite(values))),
            }
        )
    )
    return 0


def merge_shards(paths: Sequence[str], out_path: str) -> int:
    payloads = [np.load(p, allow_pickle=False) for p in paths]
    hashes = {str(p["sample_hash"]) for p in payloads}
    if len(hashes) != 1:
        raise SystemExit(
            "sample-set hash mismatch across shards: "
            + ", ".join(sorted(h[:12] + "..." for h in hashes))
        )
    n = int(payloads[0]["n"])
    values = np.full(n, np.nan)
    for payload in payloads:
        values[payload["indices"]] = payload["values"]
    np.savez(out_path, values=values, sample_hash=np.array(payloads[0]["sample_hash"]),
             spec_hash=np.array(payloads[0]["spec_hash"]), n=np.array(n))
    print(
        json.dumps(
            {
                "out": out_path,
                "n": n,
                "shards": len(payloads),
                "sample_hash": str(payloads[0]["sample_hash"]),
                "n_valid": int(np.count_nonzero(np.isfinite(values))),
            }
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="holtrop")
    parser.add_argument("--n", type=int, default=256, help="total samples in the design")
    parser.add_argument("--seed", type=int, default=20240)
    parser.add_argument("--rule", default="sobol")
    parser.add_argument("--rel-uncertainty", type=float, default=0.01)
    parser.add_argument("--shard", default="0/1", help="i/n of the shard to evaluate")
    parser.add_argument("--out", default="shard.npz")
    parser.add_argument("--merge", nargs="+", metavar="SHARD", help="merge shard files")
    args = parser.parse_args(argv)
    try:
        if args.merge:
            return merge_shards(args.merge, args.out)
        return evaluate_shard(args)
    except SampleHashMismatch as exc:  # pragma: no cover - CLI guard
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())