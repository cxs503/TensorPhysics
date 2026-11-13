"""Stocktake of scattered material/fluid property carriers in a consumer repo.

The deliverable is the migration *mapping table* required by the asset spec: every
existing hard-coded property carrier in the consumer repo (first run:
``ship-performance-platform``) gets a bucket and an action, so the migration is
auditable instead of a vibe.

Buckets
-------
``MIGRATE_TO_MPDB``
    The payload *is* a reusable property catalogue (a dict of materials with
    density/modulus/strength fields) or a block of inline property constants.
    Replace it with an MPDB lookup; the row names the MPDB substances that cover it.
``REFERENCE_MPDB``
    A case/vessel/trial/benchmark data file that embeds property values as
    *inputs*.  Keep the file (it is case evidence), resolve the literals from MPDB.
``KEEP_DOMAIN``
    Domain data with no reusable property payload (RAO cases, anchors, catalogues,
    ORC working-fluid design spaces -- the latter is a *v2 candidate*, see docs).
``KEEP_GENERATED``
    Build artefacts, reports, ledgers, backups -- never a property source.

Detection is deliberately conservative (structure first, then regex): a file is a
property carrier only if it contains at least two *distinct property keys* with
numeric values, or at least two inline property-constant assignments.

The scan is read-only and deterministic: same commit in, same table out.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

BUCKETS = ("MIGRATE_TO_MPDB", "REFERENCE_MPDB", "KEEP_DOMAIN", "KEEP_GENERATED")

#: strict property-key vocabulary (dict keys / variable-name tokens)
PROP_KEY = re.compile(
    r"^(density|rho|rho_air|rho_water|rho_kgm3|bulk_density|dry_bulk_density|grain_density|"
    r"viscosity|dynamic_viscosity|kinematic_viscosity|nu|mu|youngs_modulus|elastic_modulus|e_modulus|"
    r"poisson|poisson_ratio|yield_strength|yield_strength_mpa|sigma_0_2|sigma_u|fy|fu|"
    r"tensile_strength|compressive_strength|flexural_strength|shear_strength|"
    r"thermal_conductivity|thermal_expansion|specific_heat|cp|salinity|friction_angle|cohesion|"
    r"haz_strength_ratio|haz_rho_typical|damping_loss_factor)$",
    re.I,
)

#: property tokens that may appear as a ``_``-separated part of a variable name
#: (``RHO_SEA``, ``nu_water``, ``E_STEEL``); generic tokens such as ``k``/``e`` are
#: deliberately excluded to keep the scanner from flagging unrelated constants.
PROP_TOKENS = {
    "rho", "density", "dens", "mu", "nu", "viscosity", "viscos", "young", "youngs",
    "modulus", "poisson", "yield", "strength", "flexural", "tensile", "compressive",
    "conductivity", "specificheat", "cp", "salinity", "friction", "cohesion",
    "haz", "sigma", "eta",
}
_NAME_SPLIT = re.compile(r"[^a-z0-9]+")

MATERIAL_WORDS = re.compile(
    r"(steel|alumin|5083|6082|6061|haz|concrete|gfrp|cfrp|laminate|sandwich|"
    r"seawater|sea_?water|fresh_?water|water|air|sea_?ice|\bice\b|silt|clay|gravel|sediment)",
    re.I)

GENERATED_PATH = re.compile(
    r"(?:^|/)(artifacts?|artefacts?|docs|runs?|exports?|backups?|reports?|samples)(?:/|$)|"
    r"/c\d{6,}_\d+/|(?:^|/)knowledge-center/|(?:^|/)node_modules/|(?:^|/)frontend/|"
    r"(?:^|/)__pycache__/|(?:^|/)\.git/")

CASE_WORDS = re.compile(
    r"(case|cases|rao|anchor|benchmark|trial|catalog|catalogue|prereg|ledger|"
    r"mmg|kvlcc2|kcs|verification|validation|report|scenario)", re.I)

CATALOGUE_HINTS = re.compile(
    r"(material|materials|alloy|alloys|haz|property|properties|thermophys|plate_materials)", re.I)

#: module-level constant assignment, e.g. ``rho_water = 1025`` / ``"E": 70e9,``
CONST_ASSIGN = re.compile(
    r"^\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,?\s*(?:#.*)?$",
    re.M)

DICT_PROP_KEY = re.compile(r"[\"'](%s)[\"']\s*:" % PROP_KEY.pattern[1:-1], re.I)

#: path/material -> MPDB substance mapping (declared here, not guessed at query time)
SUBSTANCE_HINTS = [
    (re.compile(r"alumin|5083|6082|6061", re.I), ["alu_5083_H116", "alu_5083_O", "alu_6082_T6"]),
    (re.compile(r"haz|alloy", re.I), ["alu_5083_H116", "alu_6082_T6", "steel_AH36"]),
    (re.compile(r"steel|hull_?struct|ice_?belt", re.I),
     ["steel_S235", "steel_S355", "steel_AH36", "steel_EH36"]),
    (re.compile(r"concrete|caisson", re.I), ["concrete_C30"]),
    (re.compile(r"gfrp|cfrp|laminate|sandwich|composite", re.I), ["gfrp_laminate", "gfrp_sandwich"]),
    (re.compile(r"sea_?water|seawater|ocean|condenser", re.I), ["seawater"]),
    (re.compile(r"fresh_?water|water", re.I), ["water", "seawater"]),
    (re.compile(r"\bair\b|wind|aero", re.I), ["air"]),
    (re.compile(r"sea_?ice|ice_?(mech|plate|strength|physics)|\bice\b", re.I), ["ice_sea", "ice_fresh"]),
    (re.compile(r"silt|clay|gravel|sand|sediment|soil|seabed", re.I), ["sand_quartz", "silt", "clay", "gravel"]),
]

SCAN_EXTS = (".json", ".jsonl", ".yaml", ".yml", ".csv", ".py")

#: loose in-text hint used only to decide whether a generated artefact mentions properties
RAW_PROP_HINT = re.compile(r"(density|viscos|rho|young|modulus|strength|conductivity)", re.I)


def _read_text(path: str, limit: int = 2_000_000) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return fh.read(limit)
    except OSError:
        return ""


def _walk_json(obj: Any, prop_keys: set, material_keys: set, depth: int = 0) -> None:
    """Collect property keys and material-looking keys from a parsed payload."""
    if depth > 8:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            if PROP_KEY.match(ks) and isinstance(v, (int, float)) and not isinstance(v, bool):
                prop_keys.add(ks.lower())
            if MATERIAL_WORDS.search(ks):
                material_keys.add(ks)
            _walk_json(v, prop_keys, material_keys, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:200]:
            _walk_json(v, prop_keys, material_keys, depth + 1)


def _json_structure(path: str) -> Optional[Dict[str, Any]]:
    text = _read_text(path)
    ext = os.path.splitext(path)[1].lower()
    if not text.strip():
        return None
    if ext in (".json", ".jsonl"):
        try:
            if ext == ".jsonl":
                obj = [json.loads(l) for l in text.splitlines() if l.strip()]
            else:
                obj = json.loads(text)
        except Exception:
            return None
    elif ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            obj = yaml.safe_load(text)
        except Exception:
            return None
    else:
        return None
    prop_keys: set = set()
    material_keys: set = set()
    _walk_json(obj, prop_keys, material_keys)
    return {"prop_keys": prop_keys, "material_keys": material_keys}


def _name_has_prop_token(name: str) -> bool:
    parts = [p for p in _NAME_SPLIT.split(name.lower()) if p]
    return any(p in PROP_TOKENS for p in parts)


def _py_payload(text: str) -> Dict[str, Any]:
    sites: List[str] = []
    for m in CONST_ASSIGN.finditer(text):
        if _name_has_prop_token(m.group(1)):
            sites.append(m.group(1))
    dict_keys = {m.group(1).lower() for m in DICT_PROP_KEY.finditer(text)}
    return {"const_sites": sites, "prop_keys": dict_keys}


def _targets(path: str, text: str) -> List[str]:
    blob = path + "\n" + text[:20000]
    out: List[str] = []
    for pattern, substances in SUBSTANCE_HINTS:
        if pattern.search(blob):
            for s in substances:
                if s not in out:
                    out.append(s)
    return out


def classify(path: str, text: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Classify one carrier; returns ``None`` when the file carries no property payload."""
    text = _read_text(path) if text is None else text
    ext = os.path.splitext(path)[1].lower()
    base = os.path.basename(path)

    if GENERATED_PATH.search(path):
        if not RAW_PROP_HINT.search(text):
            return None
        return {
            "path": path, "ext": ext, "bytes": os.path.getsize(path) if os.path.exists(path) else 0,
            "bucket": "KEEP_GENERATED", "property_keys": 0, "inline_sites": 0,
            "material_words": "", "mpdb_substances": "", "confidence": "n/a",
            "action": "exclude: generated artefact / report / backup, never a property source",
        }

    prop_keys: set = set()
    mat_keys: set = set()
    sites: List[str] = []
    if ext in (".json", ".jsonl", ".yaml", ".yml"):
        st = _json_structure(path)
        if st:
            prop_keys = st["prop_keys"]
            mat_keys = st["material_keys"]
        if not prop_keys:
            return None
    elif ext == ".py":
        payload = _py_payload(text)
        prop_keys = payload["prop_keys"]
        sites = payload["const_sites"]
        if len(prop_keys) < 2 and len(sites) < 2:
            return None
    elif ext == ".csv":
        header = text.splitlines()[0] if text else ""
        prop_keys = {h.strip().lower() for h in header.split(",") if PROP_KEY.match(h.strip())}
        if len(prop_keys) < 2:
            return None
    else:
        return None

    n_payload = len(prop_keys) + len(sites)
    catalogue = bool(CATALOGUE_HINTS.search(base) or CATALOGUE_HINTS.search(path))
    is_case = bool(CASE_WORDS.search(path))
    materials = sorted(mat_keys)[:8]

    confidence = "low"
    if ext == ".py":
        if len(sites) >= 2 or len(prop_keys) >= 2:
            bucket = "MIGRATE_TO_MPDB"
            confidence = "high" if len(prop_keys) >= 2 else "medium"
            action = "replace inline property constants with an MPDB lookup"
        else:
            bucket = "KEEP_DOMAIN"
            action = "no reusable property payload"
    elif catalogue and n_payload >= 2:
        bucket = "MIGRATE_TO_MPDB"
        confidence = "high"
        action = "replace payload with MPDB lookup (catalogue duplicated by MPDB)"
    elif is_case and n_payload >= 2:
        bucket = "REFERENCE_MPDB"
        confidence = "medium"
        action = "keep case data, resolve embedded property literals from MPDB"
    elif n_payload >= 3 and mat_keys:
        bucket = "MIGRATE_TO_MPDB"
        confidence = "medium"
        action = "material catalogue detected; verify then replace with MPDB lookup"
    else:
        bucket = "KEEP_DOMAIN"
        confidence = "low"
        action = "domain data, no reusable property payload"

    return {
        "path": path,
        "ext": ext,
        "bytes": os.path.getsize(path) if os.path.exists(path) else 0,
        "bucket": bucket,
        "property_keys": len(prop_keys),
        "inline_sites": len(sites),
        "material_words": ",".join(materials),
        "mpdb_substances": ",".join(_targets(path, text)),
        "confidence": confidence,
        "action": (action if _targets(path, text) else
                   action + " (no MPDB substance matched - manual triage)"),
    }


def scan_repo(root: str, limit_files: int = 200000) -> List[Dict[str, Any]]:
    """Scan a repository checkout; one mapping row per property carrier."""
    rows: List[Dict[str, Any]] = []
    n = 0
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__", ".venv", "dist", "build")]
        for f in sorted(fn):
            if not f.endswith(SCAN_EXTS):
                continue
            p = os.path.join(dp, f)
            n += 1
            if n > limit_files:
                break
            rel = os.path.relpath(p, root)
            try:
                row = classify(p)
            except OSError:
                continue
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda r: (BUCKETS.index(r["bucket"]), r["path"]))
    return rows


def summarise(rows: Iterable[Dict[str, Any]], root: str = "") -> Dict[str, Any]:
    rows = list(rows)
    counts: Dict[str, int] = {b: 0 for b in BUCKETS}
    by_ext: Dict[str, int] = {}
    substances: Dict[str, int] = {}
    sites = 0
    confidence: Dict[str, int] = {}
    for r in rows:
        confidence[r.get("confidence", "low")] = confidence.get(r.get("confidence", "low"), 0) + 1
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
        by_ext[r["ext"]] = by_ext.get(r["ext"], 0) + 1
        sites += int(r.get("inline_sites") or 0)
        for s in filter(None, r["mpdb_substances"].split(",")):
            substances[s] = substances.get(s, 0) + 1
    return {
        "root": root,
        "carriers_total": len(rows),
        "by_bucket": counts,
        "by_extension": dict(sorted(by_ext.items())),
        "inline_constant_sites": sites,
        "by_confidence": dict(sorted(confidence.items())),
        "mpdb_targets": dict(sorted(substances.items(), key=lambda kv: -kv[1])),
    }


def to_csv(rows: Iterable[Dict[str, Any]]) -> str:
    cols = ["path", "ext", "bytes", "bucket", "confidence", "property_keys", "inline_sites",
            "material_words", "mpdb_substances", "action"]
    out = [",".join(cols)]
    for r in rows:
        out.append(",".join('"' + str(r.get(c, "")).replace('"', "'") + '"' for c in cols))
    return "\n".join(out) + "\n"


def to_markdown(rows: Iterable[Dict[str, Any]], summary: Dict[str, Any], head: int = 0) -> str:
    rows = list(rows)
    lines = [
        f"# Property-carrier inventory: `{summary.get('root', '')}`",
        "",
        f"- carriers in the table: **{summary['carriers_total']}**",
        f"- inline property-constant sites: **{summary['inline_constant_sites']}**",
    ]
    for b in BUCKETS:
        lines.append(f"  - `{b}`: {summary['by_bucket'].get(b, 0)}")
    lines += ["", "| bucket | confidence | path | prop keys | inline sites | MPDB substances | action |",
              "|---|---|---|---|---|---|---|"]
    shown = rows if head in (0, None) else rows[:head]
    for r in shown:
        lines.append(
            f"| {r['bucket']} | {r.get('confidence', '-')} | `{r['path']}` | "
            f"{r['property_keys']} | {r['inline_sites']} | {r['mpdb_substances'] or '-'} | "
            f"{r['action']} |")
    if head and len(rows) > head:
        lines.append(f"| ... | ({len(rows) - head} more rows in the CSV) | | | | | |")
    return "\n".join(lines) + "\n"