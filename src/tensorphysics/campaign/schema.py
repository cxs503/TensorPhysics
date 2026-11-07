"""Dependency-free artifact schemas for campaign orchestration contracts.

The campaign layer exchanges *artifacts*, never solver internals.  Every artifact
that crosses a library boundary (coupling clock, dual reconciliation ledger,
geometry contract, failure-face packet) carries an explicit schema name and
version so that the producer and the consumer can be locked against the same
definition.  The schemas implement a small, closed JSON-Schema subset:
``additionalProperties: false`` on every object means a contract change is always
a deliberate, reviewable edit instead of a silent extra field.

``schema_fingerprint`` hashes the canonical JSON form of a schema.  Pinning that
fingerprint in tests is the machine-side lock: producer tests and consumer tests
compare against the same constant, so neither side can drift away from the other
without breaking the build.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping


class SchemaError(ValueError):
    """An artifact payload does not satisfy its registered schema."""


_JSON_TYPES: dict[str, Any] = {
    "object": lambda value: isinstance(value, Mapping),
    "array": lambda value: isinstance(value, (list, tuple)),
    "string": lambda value: isinstance(value, str),
    "boolean": lambda value: isinstance(value, bool),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ),
    "null": lambda value: value is None,
}


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _fail(path: str, message: str) -> None:
    raise SchemaError(f"{path}: {message}")


def canonical_json(payload: Any) -> str:
    """Return the canonical JSON text used for hashing and byte-diffing."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def payload_digest(payload: Any) -> str:
    """SHA-256 of the canonical JSON form (one link of the audit chain)."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate(schema: Mapping[str, Any], payload: Any, path: str = "$") -> None:
    """Validate ``payload`` against the supported JSON-Schema subset."""
    if isinstance(schema, bool):
        if not schema:
            _fail(path, "schema forbids every value")
        return
    if not isinstance(schema, Mapping):
        raise SchemaError("schema node must be a mapping or boolean")

    if "const" in schema and payload != schema["const"]:
        _fail(path, f"expected constant {schema['const']!r}, got {payload!r}")
    if "enum" in schema and payload not in schema["enum"]:
        _fail(path, f"{payload!r} is not one of {schema['enum']!r}")

    declared = schema.get("type")
    if declared is not None:
        names = [declared] if isinstance(declared, str) else list(declared)
        for name in names:
            if name not in _JSON_TYPES:
                raise SchemaError(f"unsupported schema type {name!r}")
        if not any(_JSON_TYPES[name](payload) for name in names):
            _fail(path, f"expected type {names}, got {type(payload).__name__}")

    if _is_number(payload):
        if "minimum" in schema and payload < schema["minimum"]:
            _fail(path, f"{payload} < minimum {schema['minimum']}")
        if "maximum" in schema and payload > schema["maximum"]:
            _fail(path, f"{payload} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and payload <= schema["exclusiveMinimum"]:
            _fail(path, f"{payload} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and payload >= schema["exclusiveMaximum"]:
            _fail(path, f"{payload} >= exclusiveMaximum {schema['exclusiveMaximum']}")

    if isinstance(payload, str):
        if "minLength" in schema and len(payload) < schema["minLength"]:
            _fail(path, "string shorter than minLength")
        if "pattern" in schema and not re.search(schema["pattern"], payload):
            _fail(path, f"{payload!r} does not match {schema['pattern']!r}")

    if isinstance(payload, Mapping):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise SchemaError("'properties' must be a mapping")
        for name in schema.get("required", ()):
            if name not in payload:
                _fail(f"{path}.{name}", "required property is missing")
        for name, subschema in properties.items():
            if name in payload:
                validate(subschema, payload[name], f"{path}.{name}")
        extra = sorted(set(payload) - set(properties))
        allowance = schema.get("additionalProperties", True)
        if allowance is False and extra:
            _fail(path, f"additional properties not allowed: {extra}")
        elif isinstance(allowance, Mapping):
            for name in extra:
                validate(allowance, payload[name], f"{path}.{name}")

    if isinstance(payload, (list, tuple)):
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(payload):
                validate(items, item, f"{path}[{index}]")
        if "minItems" in schema and len(payload) < schema["minItems"]:
            _fail(path, f"array shorter than minItems {schema['minItems']}")
        if "maxItems" in schema and len(payload) > schema["maxItems"]:
            _fail(path, f"array longer than maxItems {schema['maxItems']}")
        if schema.get("uniqueItems"):
            seen = {canonical_json(item) for item in payload}
            if len(seen) != len(payload):
                _fail(path, "array items are not unique")