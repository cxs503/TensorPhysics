"""Best-effort TensorPlatform run registration for the research loop.

Wiring point of the 2026-11-02 boundary ruling: research runs finalized by
``research_runner`` are registered to TensorPlatform's evidence plane
(``POST /api/v1/tensorphysics/runs``) so the platform's run ledger reflects
the physics layer's research state machine.  Registration is *reporting*,
never gating: every outcome -- registered / skipped / failed -- is returned
as a record for the append-only event stream, and this module never raises.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from .platform_registry import PlatformRegistryError, TensorPlatformRunRegistry

logger = logging.getLogger(__name__)


def register_run_outcome(
    *,
    run_id: str,
    status: str,
    validation_status: str,
    manifest: Mapping[str, Any],
    registry: TensorPlatformRunRegistry | None = None,
) -> dict[str, Any]:
    """Register one terminal run outcome. Returns an event-stream record.

    Outcomes:
      {"outcome": "registered", "response": {...}}   platform accepted the run
      {"outcome": "skipped", "reason": "..."}        registry env not configured
      {"outcome": "failed", "reason": "..."}         transport or payload error

    ``skipped`` is an honest disclosure (the platform ledger simply has no
    entry for this run), not an error state.
    """
    if registry is None:
        try:
            registry = TensorPlatformRunRegistry.from_env()
        except ValueError as exc:
            return {"outcome": "skipped", "reason": str(exc)}
    try:
        response = registry.register({
            "run_id": run_id,
            "status": status,
            "validation_status": validation_status,
            **dict(manifest),
        })
    except (PlatformRegistryError, ValueError) as exc:
        logger.warning("platform run registration failed for %s: %s", run_id, exc)
        return {"outcome": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    return {"outcome": "registered", "response": response}
