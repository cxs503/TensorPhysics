"""P84-W2C campaign integration gate (plan 385, DAG node N11).

Public surface:

``run_gate(...)``
    Collect node evidence, judge C1..C4, write ``campaign_gate_report.json``,
    the campaign markdown and the figure; return the verdict dict.
``spec.NODES`` / ``spec.CONTRACTS``
    The ten declared nodes and the three declared cross-domain interfaces.

The package is imported lazily so that ``campaign.schema_defs`` can import
:mod:`tensorphysics.campaign.p84w2c_gate.schemas` without an import cycle.
"""
from __future__ import annotations

__all__ = ["run_gate", "summarise", "render_markdown", "spec", "collect", "contracts",
           "criteria", "lineage", "schemas", "GATE_ID"]

GATE_ID = "P84W2C-N11"


def __getattr__(name: str):
    if name in {"run_gate", "summarise"}:
        from . import gate as _gate
        return getattr(_gate, name)
    if name == "render_markdown":
        from .report import render_markdown
        return render_markdown
    if name in {"spec", "collect", "contracts", "criteria", "lineage", "schemas"}:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)