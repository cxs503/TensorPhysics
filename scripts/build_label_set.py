#!/usr/bin/env python3
"""Build/re-derive the 40-row human-labelled set (self-contained, corpus-independent).

Why self-contained: the frozen corpus is regenerated whenever the extractor is refined, which
shifts line numbers.  A label file that indexes into it would silently rot.  So each label
record stores the *parsed criterion* plus the raw cells it came from; the agreement test
re-parses nothing and needs only this file.

Selection (seeded, disclosed): 12 rows flagged degenerate by the current classifier + 28 rows
drawn uniformly from the whole corpus.  The corpus strict degeneracy rate is low (~2-7 %), so
balancing keeps the agreement statistic informative.  Agreement reported on a class-balanced
draw is a *balanced* accuracy, not a prevalence-weighted one.

Usage::

    python3 scripts/build_label_set.py --show    # print the rows that need labelling
    python3 scripts/build_label_set.py --write   # (re)write the skeleton
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorphysics.metrology.criteria_semantics import corpus as C  # noqa: E402
from tensorphysics.metrology.criteria_semantics import replay as R  # noqa: E402
from tensorphysics.metrology.criteria_semantics.model import Degeneracy  # noqa: E402

CORPUS = "artifacts/criteria_metrology/historical_criteria.jsonl"
OUT = "artifacts/criteria_metrology/label_set.json"
SEED_DEG = 7241
SEED_RAND = 20261111
N_DEG = 12
N_RAND = 28


def build(corpus: str = CORPUS):
    rows = C.load_corpus(corpus)
    ledger = R.replay_corpus(rows)
    # "degenerate" for sampling purposes = strict T1/T2/T3 or a self-referential reference
    deg_idx = [i for i, e in enumerate(ledger["entries"])
               if e.degeneracy is not Degeneracy.NONE or e.self_referential]
    rnd = random.Random(SEED_DEG)
    deg_pick = sorted(rnd.sample(deg_idx, min(N_DEG, len(deg_idx))))
    rnd2 = random.Random(SEED_RAND)
    rest = [i for i in range(len(rows)) if i not in set(deg_pick)]
    rand_pick = sorted(rnd2.sample(rest, min(N_RAND, len(rest))))
    picked = [("D", i) for i in deg_pick] + [("U", i) for i in rand_pick]
    records = []
    for stratum, i in picked:
        row = rows[i]
        records.append({"stratum": stratum, "criterion": row.as_dict(),
                        "label": None, "note": ""})
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    records = build(args.corpus)
    if args.show:
        for k, rec in enumerate(records):
            c, p = rec["criterion"], rec["criterion"]["provenance"]
            print(f"[{k:02d}] {rec['stratum']} q={str(p.get('quantity_cell'))[:40]!r}")
            print(f"      ref={str(p.get('reference_cell'))[:56]!r}")
            print(f"      err={str(p.get('error_cell'))[:46]!r} ver={str(p.get('verdict_cell'))[:34]!r}")
        return 0
    if args.write:
        art = {
            "schema_version": "criteria-labels/1.1",
            "self_contained": True,
            "corpus_of_origin": args.corpus,
            "selection": {"seed_degenerate": SEED_DEG, "seed_uniform": SEED_RAND,
                          "n_degenerate_stratum": sum(1 for r in records if r["stratum"] == "D"),
                          "n_uniform_stratum": sum(1 for r in records if r["stratum"] == "U"),
                          "note": "class-balanced draw (disclosed); balanced accuracy, not prevalence"},
            "annotator": {
                "A": ("agent annotator: labels assigned from the raw report cells applying the "
                      "published T1/T2/T3 definitions"),
                "B": ("independent text-only heuristic "
                      "(labeling.label_text_only): string markers only, no numeric witnesses; "
                      "a disclosed proxy for a second annotator, not a human"),
            },
            "records": records,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(art, fh, ensure_ascii=False, indent=1)
        print("wrote", args.out, len(records), "records")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())