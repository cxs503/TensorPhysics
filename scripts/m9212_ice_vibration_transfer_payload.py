"""Generate the p228w1-N08 payload: ice-induced vibration transfer chain.

Runs :func:`tensorphysics.ice_vibration_transfer.run_chain` on the reference hull
(KCS) and writes the complete JSON payload (modal table, both load paths,
transmissibility, pipe stress spectra, rainflow/Miner fatigue, mount tuning map,
criteria and disclosures) to ``artifacts/m9212/ice_vibration_transfer_n08.json``.

Every number in the report is read back from this file - nothing is transcribed
by hand.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.ice_vibration_transfer import ChainConfig, run_chain  # noqa: E402

OUT = ROOT / "artifacts" / "m9212"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = ChainConfig()
    t0 = time.time()
    result = run_chain(config)
    payload = result.to_json_dict()
    payload["runtime_s"] = round(time.time() - t0, 2)
    payload["artifacts"] = {
        "payload": "artifacts/m9212/ice_vibration_transfer_n08.json",
        "figures_dir": "docs/ice_vibration_transfer_n08/",
        "report": "docs/ICE_VIBRATION_TRANSFER_N08.md",
        "module": "src/tensorphysics/ice_vibration_transfer/",
        "tests": "tests/test_ice_vibration_transfer.py",
    }
    path = OUT / "ice_vibration_transfer_n08.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {path} ({path.stat().st_size/1024:.0f} kB) in {payload['runtime_s']} s")
    print(result.summary())
    print("criteria:", {c["id"]: c["passed"] for c in payload["criteria"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())