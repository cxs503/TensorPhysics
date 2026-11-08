"""CLI entry: python -m tensorphysics.campaign.p27w2_gate.

Runs the P27-W2 integration gate against the live campaign state and
writes the artifact bundle into the output directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import live as L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P27-W2 integration gate")
    ap.add_argument("--api-base", default=L.DEFAULT_API_BASE)
    ap.add_argument("--plan-id", type=int, default=L.P27W2_PLAN_ID)
    ap.add_argument("--out", default="runs/p27w2_n10")
    ap.add_argument("--contracts", default=None,
                    help="JSON file with {rao, granular, gci_uq} payloads")
    ap.add_argument("--capabilities", default=None,
                    help="JSON file with the capability ledger list")
    ap.add_argument("--pytest-summary", default=None,
                    help="JSON file with {passed, failed, errors}")
    ap.add_argument("--commits", default=None,
                    help="JSON file mapping node id -> commit hash")
    ap.add_argument("--maturity-artifacts", default=None,
                    help="JSON file mapping node id -> result artifact")
    args = ap.parse_args(argv)

    def load_json(p: str | None) -> object:
        if not p:
            return None
        return json.loads(Path(p).read_text())

    token = L.login(args.api_base)
    verdict, state = L.run_live_gate(
        api_base=args.api_base,
        token=token,
        plan_id=args.plan_id,
        contract_payloads=load_json(args.contracts) or {},
        capabilities=load_json(args.capabilities) or [],
        test_summary=load_json(args.pytest_summary) or {},
        commits={int(k): v for k, v in (load_json(args.commits) or {}).items()},
        maturity_artifacts=(
            {int(k): v for k, v in (load_json(args.maturity_artifacts) or {}).items()}
        ),
    )
    ledger = L.build_ledger_from_live(
        state, {int(k): v for k, v in (load_json(args.commits) or {}).items()}
    )
    maturity = L.build_maturity_from_live(
        state,
        {int(k): v for k, v in (load_json(args.maturity_artifacts) or {}).items()},
    )
    from . import contracts as C
    contract_results = C.validate_all(load_json(args.contracts) or {})
    paths = L.write_all_outputs(
        args.out, verdict, state, ledger, maturity, contract_results,
        load_json(args.capabilities) or [], load_json(args.pytest_summary) or {},
    )
    print(json.dumps({"overall": verdict.overall, "paths": paths},
                     ensure_ascii=False, indent=2))
    return 0 if verdict.overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
