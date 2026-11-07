"""Benchmark-JSON adapters feeding the V&V gate from the Tensor* solver repos.

Every solver repository publishes its benchmark output in its own dialect.  The
gate only needs one shape -- a list of ``(h, f)`` levels of a single
observable -- so the adapters normalise the dialects instead of asking the
solver repositories to change: the gate stays *zero-intrusion*.

Supported dialects (all read-only):

``tensorlbm``
    ``{"quantity": ..., "levels": [{"resolution": 32, "value": 1.2e-4}, ...]}``
``tensorfem``
    h-convergence/GCI evidence, schema ``tensorfem.h-convergence-gci/1.0``;
    the study (``bending``, ``manufactured``, ...) is selected by ``study``.
``tensorfvm``
    ``{"grids": [{"h": 0.02, "l2_error": ...}, ...]}``
``tensordem``
    time-step refinement, ``{"timesteps": [{"dt": 1e-3, "value": ...}, ...]}``
``tensorbem``
    panel refinement, ``{"meshes": [{"n_panels": 512, "value": ...}, ...]}``
``tensorphysics``
    this repository's own benchmark JSON ledgers.

If the dialect is unknown the generic profile is used: the adapters walk the
document and take the first list of at least two mappings that carries both a
grid-length key (``h``, ``mesh_size``, ``dx``, ``dt``, ``panel_size``, ...) or a
grid-count key (``resolution``, ``n``, ``nx``, ``n_panels``, ...) and an
observable key (``value``, ``error``, ``l2_error``, ...).  A count is converted
to a length with ``h = 1 / count`` because only ratios between levels matter.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .gci import Observation

__all__ = [
    "AdapterError",
    "AdapterSeries",
    "REPO_PROFILES",
    "detect_repo",
    "extract_series",
    "load_payload",
    "to_gate_payload",
    "write_gate_payload",
]

LENGTH_KEYS = ("h", "mesh_size", "dx", "dt", "cell_size", "spacing", "panel_size", "element_size")
COUNT_KEYS = (
    "resolution",
    "n",
    "nx",
    "ny",
    "nz",
    "n_cells",
    "n_panels",
    "panels",
    "divisions",
    "nodes",
    "elements",
    "dofs",
)
VALUE_KEYS = (
    "value",
    "f",
    "observable",
    "metric",
    "error",
    "l2_error",
    "rel_l2_error",
    "relative_l2_error",
    "relative_error",
    "rao_error",
    "added_mass_error",
    "relative_l2_displacement_error",
    "relative_strain_energy_error",
    "strain_energy",
    "max_error",
    "position_error",
)
LEVEL_CONTAINERS = (
    "levels",
    "grids",
    "meshes",
    "runs",
    "resolutions",
    "timesteps",
    "time_steps",
    "studies",
    "results",
    "rows",
    "samples",
)

REPO_PROFILES: dict[str, dict[str, Any]] = {
    "tensorphysics": {
        "length_keys": LENGTH_KEYS,
        "count_keys": COUNT_KEYS,
        "value_keys": VALUE_KEYS,
        "containers": LEVEL_CONTAINERS,
    },
    "tensorlbm": {
        "length_keys": ("h", "dx", "cell_size", "dt", "lattice_spacing"),
        "count_keys": ("resolution", "n", "nx", "ny", "nz", "n_cells", "lattice"),
        "value_keys": (
            "value",
            "f",
            "error",
            "l2_error",
            "rel_l2_error",
            "relative_l2_error",
            "kinetic_energy",
            "momentum_flux",
            "observed",
        ),
        "containers": ("levels", "grids", "runs", "resolutions", "cases"),
    },
    "tensorfvm": {
        "length_keys": ("h", "dx", "cell_size", "mesh_size"),
        "count_keys": ("n", "nx", "ny", "n_cells", "cells", "resolution"),
        "value_keys": ("value", "f", "error", "l1_error", "l2_error", "linf_error", "residual"),
        "containers": ("levels", "grids", "runs", "resolutions", "cases"),
    },
    "tensorfem": {
        "length_keys": ("mesh_size", "h", "element_size", "dx"),
        "count_keys": ("divisions", "n", "nodes", "elements", "dofs", "n_free_dofs"),
        "value_keys": (
            "relative_strain_energy_error",
            "relative_l2_displacement_error",
            "strain_energy",
            "value",
            "error",
            "l2_error",
        ),
        "containers": ("levels", "meshes", "runs", "studies"),
    },
    "tensordem": {
        "length_keys": ("dt", "h", "time_step", "spacing"),
        "count_keys": ("n_particles", "steps", "n", "resolution"),
        "value_keys": ("value", "f", "error", "position_error", "l2_error", "max_error"),
        "containers": ("levels", "timesteps", "time_steps", "runs", "cases"),
    },
    "tensorbem": {
        "length_keys": ("h", "panel_size", "mesh_size", "dx"),
        "count_keys": ("n_panels", "panels", "n", "nodes", "resolution"),
        "value_keys": ("value", "f", "error", "l2_error", "relative_error", "rao_error", "added_mass_error"),
        "containers": ("levels", "meshes", "panels", "runs", "cases"),
    },
}
REPO_PROFILES["generic"] = REPO_PROFILES["tensorphysics"]


class AdapterError(RuntimeError):
    """The benchmark document cannot be mapped onto an observation series."""


@dataclass(frozen=True)
class AdapterSeries:
    """A benchmark output normalised into gate input."""

    repo: str
    quantity: str
    levels: tuple[Observation, ...]
    source_path: str = ""
    study: str = ""
    metadata: Mapping[str, Any] | None = None
    raw_levels: tuple[Mapping[str, Any], ...] = ()

    def to_gate_payload(self, *, expected_order: float | None = None,
                        validation: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return to_gate_payload(self, expected_order=expected_order, validation=validation)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "repo": self.repo,
            "quantity": self.quantity,
            "source_path": self.source_path,
            "study": self.study,
            "levels": [lvl.as_dict() for lvl in self.levels],
        }
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


def load_payload(path: str | Path) -> Any:
    """Read a benchmark JSON document."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise AdapterError(f"{path}: not valid JSON ({exc})") from exc


def detect_repo(payload: Any) -> str:
    """Repository detection from the document's ``schema`` (or its declared repo).

    Dialects are recognised by the ``schema`` string they publish, e.g.
    ``"tensorfem.h-convergence-gci/1.0"``.  A document without a schema is read
    with the generic profile; pass ``repo=`` explicitly when a schema-less
    document carries ambiguous field names.
    """
    if isinstance(payload, Mapping):
        schema = payload.get("schema")
        if isinstance(schema, str):
            for repo in REPO_PROFILES:
                if repo != "generic" and schema.lower().startswith(repo):
                    return repo
        declared = payload.get("repo") or payload.get("solver")
        if isinstance(declared, str):
            key = declared.strip().lower()
            if key in REPO_PROFILES:
                return key
    # No declared dialect: the generic profile reads the document (it needs only
    # an ``h``/count key and a value key), and ``repo=`` on the CLI can be used to
    # force a dialect when a document carries ambiguous field names.
    return "generic"


def _qualifies(node: Any) -> bool:
    if not isinstance(node, list) or len(node) < 2:
        return False
    if not all(isinstance(item, Mapping) for item in node):
        return False
    keys = set().union(*(item.keys() for item in node))
    return bool((keys & set(LENGTH_KEYS)) or (keys & set(COUNT_KEYS))) and bool(keys & set(VALUE_KEYS))


def _find_level_list(node: Any, profile: Mapping[str, Any], depth: int = 0) -> tuple[list, str]:
    """Find the first list of level mappings in ``node`` (preferred containers first)."""
    if depth > 6:
        raise AdapterError("benchmark document is too deeply nested to locate the level list")
    if isinstance(node, list):
        if _qualifies(node):
            return node, ""
        for item in node:
            try:
                return _find_level_list(item, profile, depth + 1)
            except AdapterError:
                continue
        raise AdapterError("no list of level mappings found")
    if not isinstance(node, Mapping):
        raise AdapterError("no list of level mappings found")
    containers = tuple(profile.get("containers", LEVEL_CONTAINERS))
    for key in containers:
        if key in node and _qualifies(node[key]):
            return node[key], key
    for key in sorted(node, key=lambda k: (k not in containers, str(k))):
        value = node[key]
        if isinstance(value, Mapping):
            try:
                return _find_level_list(value, profile, depth + 1)
            except AdapterError:
                continue
        elif isinstance(value, list) and _qualifies(value):
            return value, key
    raise AdapterError("no list of level mappings found")


def _pick(mapping: Mapping[str, Any], keys: Iterable[str]) -> tuple[str, Any] | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return key, mapping[key]
    return None


def _level_observation(item: Mapping[str, Any], profile: Mapping[str, Any],
                       value_key: str | None) -> Observation:
    length = _pick(item, profile.get("length_keys", LENGTH_KEYS))
    count = None if length else _pick(item, profile.get("count_keys", COUNT_KEYS))
    if length is not None:
        h = float(length[1])
        h_label = length[0]
    elif count is not None:
        try:
            h = 1.0 / float(count[1])
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise AdapterError(f"cannot turn {count[0]}={count[1]!r} into a grid length") from exc
        h_label = f"1/{count[0]}"
    else:
        raise AdapterError(f"level {dict(item)!r} carries no grid length or grid count")
    if value_key is not None:
        if value_key not in item:
            raise AdapterError(f"level {dict(item)!r} has no {value_key!r} field")
        selected = (value_key, item[value_key])
    else:
        selected = _pick(item, profile.get("value_keys", VALUE_KEYS))
    if selected is None:
        raise AdapterError(f"level {dict(item)!r} carries no observable value")
    sigma_pick = _pick(item, ("sigma", "std", "uncertainty", "stderr"))
    sigma = sigma_pick[1] if sigma_pick else None
    label = str(item.get("label") or item.get("name") or item.get("case") or h_label)
    return Observation(h=h, f=float(selected[1]), label=label,
                       sigma=None if sigma is None else float(sigma))


def _select_study(payload: Any, study: str | None, value_key: str | None) -> tuple[Any, str]:
    """Pick the sub-study of an evidence document (e.g. TensorFEM ``bending``)."""
    if not isinstance(payload, Mapping):
        return payload, ""
    candidates = {k: v for k, v in payload.items()
                  if isinstance(v, Mapping) and any(_qualifies(v.get(c)) for c in LEVEL_CONTAINERS)}
    if not candidates:
        return payload, ""
    if study:
        if study not in candidates:
            raise AdapterError(
                f"study {study!r} not present; available: {sorted(candidates)}"
            )
        return candidates[study], study
    if value_key:
        for name, node in candidates.items():
            try:
                level_list, _ = _find_level_list(node, REPO_PROFILES["generic"])
            except AdapterError:
                continue
            if any(value_key in item for item in level_list):
                return node, name
    for name in candidates:
        if "manufactured" in name.lower() or "mms" in name.lower():
            return candidates[name], name
    name = sorted(candidates)[0]
    return candidates[name], name


def extract_series(
    payload: Any,
    *,
    repo: str = "auto",
    quantity: str | None = None,
    study: str | None = None,
    value_key: str | None = None,
    source_path: str = "",
) -> AdapterSeries:
    """Normalise one benchmark document into an :class:`AdapterSeries`."""
    detected = detect_repo(payload) if repo in ("auto", "", None) else repo
    if detected not in REPO_PROFILES:
        raise AdapterError(f"unknown repo profile {detected!r}; known: {sorted(REPO_PROFILES)}")
    profile = REPO_PROFILES[detected]
    node, chosen_study = _select_study(payload, study, value_key)
    level_list, container = _find_level_list(node, profile)
    explicit_key = value_key
    if explicit_key is None and quantity:
        if any(quantity in item for item in level_list):
            explicit_key = quantity
    observations = tuple(_level_observation(item, profile, explicit_key) for item in level_list)
    ordered = tuple(sorted(observations, key=lambda lvl: -lvl.h))
    used_key = explicit_key
    if used_key is None:
        for item in level_list:
            found = _pick(item, profile.get("value_keys", VALUE_KEYS))
            if found:
                used_key = found[0]
                break
    declared_q = None
    for node_like in (node, payload):
        if isinstance(node_like, Mapping) and isinstance(node_like.get("quantity"), str):
            declared_q = node_like["quantity"]
            break
    label_q = quantity or declared_q or chosen_study or used_key or "observable"
    metadata = {}
    if isinstance(payload, Mapping):
        for key in ("schema", "case", "quantity", "note", "oracle"):
            if key in payload and isinstance(payload[key], (str, int, float)):
                metadata[key] = payload[key]
        if isinstance(node, Mapping) and isinstance(node.get("oracle"), str):
            metadata.setdefault("oracle", node["oracle"])
    return AdapterSeries(
        repo=detected,
        quantity=label_q,
        levels=ordered,
        source_path=source_path,
        study=chosen_study,
        metadata={**metadata, "container": container, "value_key": used_key},
        raw_levels=tuple(dict(item) for item in level_list),
    )


def to_gate_payload(
    series: AdapterSeries,
    *,
    expected_order: float | None = None,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialise an adapted series into the gate's observation JSON."""
    payload: dict[str, Any] = {
        "schema": "tensorphysics.vv.observations/1",
        "quantity": series.quantity,
        "levels": [lvl.as_dict() for lvl in series.levels],
        "source": {
            "repo": series.repo,
            "path": series.source_path,
            "study": series.study,
            "metadata": dict(series.metadata or {}),
        },
    }
    if expected_order is not None:
        payload["expected_order"] = expected_order
    if validation:
        payload["validation"] = dict(validation)
    return payload


def write_gate_payload(series: AdapterSeries, path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Write the gate payload (used by the cross-repo runners)."""
    payload = to_gate_payload(series, **kwargs)
    text = json.dumps(payload, indent=2, sort_keys=False)
    Path(path).write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    return payload