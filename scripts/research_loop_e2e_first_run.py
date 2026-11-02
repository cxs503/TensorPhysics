#!/usr/bin/env python3
"""Research-loop E2E first run: execute -> independent validation -> TP write-back.

Live evidence for the 2026-11 boundary wiring (R3): one SUCCESS leg and one
honest-FAIL leg, both terminal-registered to TensorPlatform's run ledger and
read back through the same client.

Real physics throughout, no mocks:
- execution backend = the R1 audited kernel ``shared_core_resistance`` at the
  SUBOFF-01 canonical operating point (tests/test_physics_contracts.py inputs);
- validator = closed-form re-derivation written INLINE in this script (does
  not import the kernel under test) + Schoenherr-1932 historical cross-check
  + the frozen DERIVED contract anchor 0.003461889103158906 (rel 1e-9);
- fail leg = a deliberately mis-scoped validator (EXPERIMENTAL band 0.0038
  with a 1e-9 tolerance that belongs to the DERIVED band) -- the honest-FAIL
  branch must register ``failed`` to the platform, not be papered over.

Credentials: TP_LOGIN_USER / TP_LOGIN_PASSWORD env vars (never printed).
Evidence: SQLite task store + append-only event log at E2E_EVIDENCE_DIR.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorphysics.physics_kernels_resistance import (  # noqa: E402
    shared_core_resistance, schoenherr_1932_cf,
)
from tensorphysics.research_runtime import (  # noqa: E402
    AgentRequest, AgentResult, ResearchTaskStore, TaskStatus,
)
from tensorphysics.research_runner import ResearchTaskRunner  # noqa: E402

TP_URL = os.environ.get("TENSORPLATFORM_URL", "http://127.0.0.1:8010")
EVIDENCE_DIR = Path(os.environ.get("E2E_EVIDENCE_DIR", "e2e_evidence"))

SUBOFF01 = dict(length_m=4.356, beam_m=0.508, cb=0.542, speed_ms=3.0,
                rho_kg_m3=998.2, nu_m2_s=1.004e-06,
                wetted_surface_m2=4.417911491407555)
ANCHOR_DERIVED_CT = 0.003461889103158906   # frozen contract anchor, rel 1e-9
ANCHOR_EXPERIMENTAL_CT = 0.0038             # tol 0.15, provenance EXPERIMENTAL
RUN_DATE = os.environ.get("RUN_TAG", "2026-11-03")


def tp_login() -> tuple[str, int]:
    user = os.environ["TP_LOGIN_USER"]
    password = os.environ["TP_LOGIN_PASSWORD"]
    for path in ("/api/v1/auth/login", "/auth/login"):
        try:
            req = Request(f"{TP_URL}{path}",
                          data=json.dumps({"username": user, "password": password}).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            token = data.get("access_token") or data.get("token")
            if token:
                break
        except Exception:
            continue
    else:
        raise SystemExit("TP login failed on all known routes")
    orgs = None
    try:
        req = Request(f"{TP_URL}/api/v1/users/me",
                      headers={"Authorization": f"Bearer {token}"}, method="GET")
        with urlopen(req, timeout=10) as r:
            me = json.loads(r.read())
        orgs = me.get("organizations") or me.get("organization_ids")
        print(f"login ok: user={me.get('username')} role={me.get('role')} orgs={orgs}")
    except Exception as exc:
        print(f"users/me unavailable ({exc}); org falls back to 1")
    org_id = int(orgs[0]) if isinstance(orgs, list) and orgs else 1
    return token, org_id


class ContractKernelAgent:
    """Deterministic execution backend: runs the R1 audited physics kernel."""

    @property
    def name(self) -> str:
        return "tensorphysics.physics_kernels_resistance.shared_core_resistance"

    def execute(self, request: AgentRequest) -> AgentResult:
        r = shared_core_resistance(**SUBOFF01)
        return AgentResult(
            outcome="completed",
            summary=(f"SUBOFF-01 shared-core composition: ct={r.ct!r} "
                     f"(cf={r.cf!r} + cr={r.cr!r}) at Re={r.reynolds:.1f}, Fn={r.froude:.4f}"),
            artifact_uris=(f"sqlite://{EVIDENCE_DIR}/research_tasks.sqlite#events",),
            metadata={"ct": r.ct, "cf": r.cf, "cr": r.cr, "re": r.reynolds,
                      "fn": r.froude, "rt_n": r.rt_n},
        )


def _inline_rederive() -> dict[str, float]:
    """Closed-form re-derivation, independent of the kernel under test."""
    re = SUBOFF01["speed_ms"] * SUBOFF01["length_m"] / SUBOFF01["nu_m2_s"]
    fn = SUBOFF01["speed_ms"] / math.sqrt(9.81 * SUBOFF01["length_m"])
    cf = 0.075 / (math.log10(re) - 2.0) ** 2
    cr = 0.0 if fn <= 0.05 else 0.002 * (1.0 + 2.0 * SUBOFF01["cb"]) * fn ** 2.5
    return {"re": re, "fn": fn, "cf": cf, "cr": cr, "ct": cf + cr}


class IndependentCrossCheckValidator:
    """Independent validation: inline re-derivation + Schoenherr cross-check
    + frozen DERIVED anchor. Imports NOTHING from the kernel under test."""

    def validate(self, request: AgentRequest, result: AgentResult) -> dict:
        got = dict(result.metadata or {})
        ref = _inline_rederive()
        checks = {
            "inline_ct_matches_agent": math.isclose(got["ct"], ref["ct"], rel_tol=1e-12),
            "frozen_derived_anchor_1e-9": math.isclose(
                got["ct"], ANCHOR_DERIVED_CT, rel_tol=1e-9),
            "schoenherr_cf_within_5pct": abs(
                got["cf"] / schoenherr_1932_cf(ref["re"]) - 1.0) < 0.05,
            "reynolds_consistent": math.isclose(got["re"], ref["re"], rel_tol=1e-9),
        }
        status = "passed" if all(checks.values()) else "failed"
        return {"status": status, "checks": checks,
                "reference": {"method": "inline closed-form + Schoenherr 1932 + "
                             "frozen DERIVED anchor (physics_contracts SUBOFF-01)",
                              "inline": {k: repr(v) for k, v in ref.items()},
                              "anchor_derived_ct": repr(ANCHOR_DERIVED_CT),
                              "anchor_provenance": "DERIVED"},
                "independent_of_agent_code": True}


class MisScopedAnchorValidator:
    """Deliberately WRONG validator for the honest-FAIL leg: applies the
    DERIVED band's 1e-9 tolerance to the EXPERIMENTAL band value 0.0038.
    The task MUST fail and the failure MUST be registered -- not overridden."""

    def validate(self, request: AgentRequest, result: AgentResult) -> dict:
        got = dict(result.metadata or {})
        passed = math.isclose(got["ct"], ANCHOR_EXPERIMENTAL_CT, rel_tol=1e-9)
        return {"status": "passed" if passed else "failed",
                "checks": {"experimental_anchor_with_derived_tolerance": passed},
                "defect": "provenance misuse: EXPERIMENTAL band (tol 0.15) judged "
                          "with the DERIVED band tolerance (1e-9)"}


GOAL = {
    "objective": "SUBOFF-01 shared-core resistance composition, "
                 "E2E first run of the research loop",
    "kernel": "tensorphysics.physics_kernels_resistance.shared_core_resistance",
    "inputs": SUBOFF01,
    "acceptance": {"anchor_derived_ct": ANCHOR_DERIVED_CT, "rel_tol": 1e-9,
                   "independent_validator": "inline closed-form + Schoenherr 1932"},
    "e2e_wiring": "R3 research_runner terminal registration -> TP run ledger",
}


def dump_events(store: ResearchTaskStore, task_id: str) -> None:
    print(f"--- events {task_id}")
    for ev in store.list_events(task_id):
        extra = ""
        if ev.event_type == "platform_registration":
            extra = f" registration={dict(ev.payload).get('outcome')}"
        fs = ev.from_status.value if ev.from_status else "-"
        print(f"  {ev.event_id:3d} {fs:>9} -> {ev.to_status.value:<9} "
              f"{ev.event_type}{extra}")


def main() -> None:
    token, org_id = tp_login()
    os.environ["TENSORPLATFORM_URL"] = TP_URL
    os.environ["TENSORPLATFORM_API_TOKEN"] = token
    os.environ["TENSORPLATFORM_ORGANIZATION_ID"] = str(org_id)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    store = ResearchTaskStore(EVIDENCE_DIR / "research_tasks.sqlite")

    from tensorphysics.platform_registry import TensorPlatformRunRegistry
    registry = TensorPlatformRunRegistry.from_env()

    legs = [
        (f"e2e-first-run-{RUN_DATE}-success", IndependentCrossCheckValidator(), 3,
         ContractKernelAgent()),
        (f"e2e-first-run-{RUN_DATE}-fail", MisScopedAnchorValidator(), 1,
         ContractKernelAgent()),
    ]
    for task_id, validator, max_attempts, agent in legs:
        store.create_task(task_id, goal_id=f"e2e-first-run-{RUN_DATE}", plan=GOAL,
                          max_attempts=max_attempts)
        runner = ResearchTaskRunner(store, agent, validator)
        task = runner.run_once(task_id, goal=GOAL, workspace=str(EVIDENCE_DIR),
                               timeout_seconds=600)
        print(f"=== leg {task_id}: final={task.status.value} attempt={task.attempt}")
        dump_events(store, task_id)
        # read-back through the same client the runner used
        back = registry.get(task_id)
        print(f"    TP ledger read-back: status={back.get('status')} "
              f"validation={back.get('validation_status')} "
              f"manifest_keys={sorted((back.get('manifest') or {}).keys())[:6]}")
        assert task.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED)

    store.close()
    print("E2E FIRST RUN COMPLETE (both legs terminal + registered + read back)")


if __name__ == "__main__":
    main()
