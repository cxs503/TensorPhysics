#!/usr/bin/env python
"""Standalone cluster shard worker for the P96W3-N11 ``{i}/{n}`` channel.

Why a wrapper instead of ``python -m tensorphysics.sharding_cluster``: the
package ``__init__`` eagerly imports numpy/scipy (~10 s on a cold cluster node),
while a shard worker only needs the stdlib (the shard/channel modules use numpy
lazily).  This wrapper therefore installs a *namespace-only* ``tensorphysics``
package object pointing at ``src/tensorphysics`` so the modules load without
executing ``__init__.py``.

Relative ``--plan`` / ``--out`` paths are resolved against the repository root,
so the same command template works regardless of the node's working directory.

Usage (identical CLI to the module):
    python scripts/p96w3_n11_cluster_shard.py run-shard \
        --plan artifacts/p96w3_n11/cluster/plan.json --i 3 --n 8 \
        --out artifacts/p96w3_n11/cluster/out
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def _bootstrap() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if "tensorphysics" not in sys.modules:
        pkg = types.ModuleType("tensorphysics")
        pkg.__path__ = [str(SRC / "tensorphysics")]  # namespace only: no __init__
        sys.modules["tensorphysics"] = pkg


def _resolve_paths(argv: list[str]) -> list[str]:
    out: list[str] = []
    path_flags = {"--plan", "--out", "--json-out"}
    resolve_next = False
    for arg in argv:
        if resolve_next and not os.path.isabs(arg):
            out.append(str((REPO / arg).resolve()))
            resolve_next = False
            continue
        out.append(arg)
        resolve_next = arg in path_flags
    return out


def main() -> int:
    _bootstrap()
    from tensorphysics.sharding_cluster import main as channel_main

    return channel_main(_resolve_paths(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())