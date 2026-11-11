"""Capability reuse graph: does a delivered capability ever enter a later task's
implementation path, or is it shelved the moment it is accepted?

The campaign has thousands of ``done`` nodes.  Nothing answered "how many of
them are actually *reused*".  This package turns that question into a
recomputable graph audit:

``indexer``
    repository manifest + module index (imports roots per repository), python
    ``ast`` dependency extraction and a mechanical lexical (grep) extraction
    over exactly the same universe of files.
``extract``
    capability -> code path mapping from the node/report evidence corpus, then
    module-level dependencies projected onto capability level.
``aggregate``
    dual-caliber cross-validation (Jaccard, per-item disagreement list),
    in-degree / PageRank / connected components / cross-repository edge share,
    and the four opinion buckets: reused / orphan / hub / gap.
``hints``
    seeding hints (orphan build|test|doc classification, hub redundancy list,
    high-rank-but-unverified gap list) with explicit ranking reasons.
``gold``
    hand-labelled capability-level edges (positives + verified negatives) and
    the extractor precision/recall against them.
``audit``
    the machine-checkable acceptance criteria of the campaign node, written as
    one JSON artefact.

Design rules
------------
* **no repository names, domains or thresholds in the logic** -- repositories,
  capability vocabularies, thresholds and the gold labels are *data*
  (``configs/*.json`` / snapshot / gold file); ``selfcheck`` proves it.
* **determinism** -- canonical JSON, sorted collections, no clock/network in
  the graph itself: the same snapshot replays to the same bytes and hash.
* **honest accounting** -- capabilities whose evidence yields no resolvable
  code path go to the ``undetermined`` bucket and are *excluded* from the
  orphan rate denominator instead of being silently counted as orphans; a
  caliber disagreement is listed, never hidden behind a silent union.
"""
from __future__ import annotations

from .aggregate import (build_graph, capability_graph, components, degrees,
                        dual_caliber_compare, pagerank)
from .audit import run_audit, write_json
from .extract import (evidence_edges, extract_code_edges, map_capabilities)
from .gold import load_gold, score_extraction
from .hints import build_hints
from .indexer import (RepoIndex, build_index, load_snapshot)
from .model import (CAP_EDGE_SCHEMA, Capability, Edge, canonical_json,
                    canonical_hash, sha256_text)

__all__ = [
    "CAP_EDGE_SCHEMA", "Capability", "Edge", "RepoIndex", "build_graph",
    "build_hints", "build_index", "canonical_hash", "canonical_json",
    "capability_graph", "components", "degrees", "dual_caliber_compare",
    "evidence_edges", "extract_code_edges", "load_gold", "load_snapshot", "map_capabilities", "pagerank",
    "run_audit", "score_extraction", "sha256_text",
    "write_json",
]