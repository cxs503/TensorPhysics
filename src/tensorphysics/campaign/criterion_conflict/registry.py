"""Criterion registry: a reproducible, hash-addressed object.

``CriterionRegistry.build(entries, config)`` extracts all criteria from a
corpus and exposes:

* ``registry_hash()``  - sha256 over canonical criterion records (idempotent:
  two runs on the same corpus/config yield the same hash; no timestamps);
* ``conflicts()``      - cross-campaign conflicts (see ``conflict.py``);
* ``coverage()``       - extraction coverage against a reference annotator.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import ConflictConfig, load_config
from .conflict import Conflict, detect_conflicts
from .extract import _compile_alias_table, extract_criteria
from .model import Criterion, SourceRef, canonical_json

_NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
# Independent, deliberately loose reference annotator used only to *measure*
# extraction coverage (never to emit registry entries).
_REF_RE = re.compile(r"(?:<=|>=|≤|≥|<|>)\s*(" + _NUM + r")")


@dataclass
class CorpusEntry:
    mission_id: str
    campaign: str
    repo: str = ""
    goal: str = ""
    report: str = ""
    status: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusEntry":
        return cls(
            mission_id=str(data.get("mission_id", data.get("id", ""))),
            campaign=str(data.get("campaign") or data.get("task_key", "") or ""),
            repo=str(data.get("repo", "")),
            goal=str(data.get("goal", "") or ""),
            report=str(data.get("report", "") or ""),
            status=str(data.get("status", "") or ""),
        )


@dataclass
class CriterionRegistry:
    criteria: list[Criterion]
    config: ConflictConfig
    entries: list[CorpusEntry] = field(default_factory=list)
    residue: list[dict[str, Any]] = field(default_factory=list)
    _conflicts: Optional[list[Conflict]] = None

    # ---- construction ---------------------------------------------------
    @classmethod
    def build(cls, entries: Iterable[CorpusEntry | dict[str, Any]],
              config: "ConflictConfig | dict | str | Path") -> "CriterionRegistry":
        cfg = load_config(config)
        parsed: list[CorpusEntry] = []
        criteria: list[Criterion] = []
        residue: list[dict[str, Any]] = []
        for raw in entries:
            entry = raw if isinstance(raw, CorpusEntry) else CorpusEntry.from_dict(raw)
            if not entry.campaign:
                entry.campaign = entry.mission_id
            parsed.append(entry)
            for field_name in ("goal", "report"):
                text = getattr(entry, field_name)
                if not text:
                    continue
                source = SourceRef.make(entry.mission_id, entry.campaign, entry.repo,
                                        field_name, 0, 0, 0, "")
                partial = extract_criteria(text, source, cfg)
                criteria.extend(partial.criteria)
                residue.extend(partial.residue)
        criteria.sort(key=lambda c: c.criterion_id)
        # duplicate ids cannot happen for distinct spans; keep first anyway
        dedup: dict[str, Criterion] = {}
        for c in criteria:
            dedup.setdefault(c.criterion_id, c)
        return cls(criteria=sorted(dedup.values(), key=lambda c: c.criterion_id),
                   config=cfg, entries=parsed, residue=residue)

    # ---- views ----------------------------------------------------------
    def mapped(self) -> list[Criterion]:
        return [c for c in self.criteria if c.status == "mapped"]

    def criteria_by_mission(self) -> dict[str, list[Criterion]]:
        out: dict[str, list[Criterion]] = {}
        for c in self.criteria:
            out.setdefault(c.source.mission_id, []).append(c)
        return out

    def registry_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "criterion-registry/1.0",
            "config_sha256": self.config.sha256(),
            "criteria": [c.to_dict() for c in sorted(self.criteria, key=lambda c: c.criterion_id)],
        }

    def registry_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.registry_payload()).encode("utf-8")).hexdigest()

    def conflicts(self, same_campaign_ok: bool = False) -> list[Conflict]:
        if self._conflicts is None:
            self._conflicts = detect_conflicts(self.mapped(), self.config, same_campaign_ok)
        return self._conflicts

    def coverage(self) -> dict[str, Any]:
        """Instance-level extraction coverage vs an independent reference annotator.

        Reference instance = a comparator+number pair on a line that also names
        a configured physical quantity (alias).  This measures the parser, not
        the vocabulary: unknown quantities are reported separately as residue.
        """
        aliases = _compile_alias_table(self.config.alias_table())
        by_mission = self.criteria_by_mission()
        total_ref = 0
        recovered = 0
        misses: list[dict[str, Any]] = []
        per_mission: list[dict[str, Any]] = []
        # independent clause splitter (structural delimiters only); the
        # reference annotator is a loose "comparator + number next to a known
        # quantity" regex, deliberately sharing no code with the extractor.
        splitter = re.compile(r"[;；。|+]|(?<!\d)[,，](?!\d)")
        for entry in self.entries:
            instances: list[tuple[int, float]] = []
            for line_no, line in enumerate(entry.goal.splitlines(), start=1):
                for segment in splitter.split(line):
                    if not any(regex.search(segment) for regex, _, _ in aliases):
                        continue
                    for m in _REF_RE.finditer(segment):
                        try:
                            instances.append((line_no, float(m.group(1).replace(",", ""))))
                        except ValueError:
                            continue
            if not instances:
                continue
            own = by_mission.get(entry.mission_id, [])
            hit = 0
            for line_no, value in instances:
                total_ref += 1
                matched = any(
                    c.source.field == "goal" and c.source.line == line_no
                    and c.threshold is not None and abs(float(c.threshold) - value) <= 1e-9
                    for c in own
                )
                if matched:
                    hit += 1
                    recovered += 1
                else:
                    misses.append({"mission_id": entry.mission_id, "line": line_no,
                                   "value": value, "snippet": entry.goal.splitlines()[line_no - 1][:160]})
            per_mission.append({"mission_id": entry.mission_id, "reference": len(instances), "recovered": hit})
        rate = (recovered / total_ref) if total_ref else 0.0
        missions_recovered = sum(1 for p in per_mission if p["recovered"] > 0)
        return {
            "reference_instances": total_ref,
            "recovered_instances": recovered,
            "coverage_rate": rate,
            "missions_with_reference": len(per_mission),
            "missions_recovered": missions_recovered,
            "mission_rate": (missions_recovered / len(per_mission)) if per_mission else 0.0,
            "misses": misses[:50],
        }

    # ---- IO -------------------------------------------------------------
    def to_dict(self, include_conflicts: bool = True) -> dict[str, Any]:
        payload = self.registry_payload()
        payload["registry_hash"] = self.registry_hash()
        payload["criterion_count"] = len(self.criteria)
        payload["mapped_count"] = len(self.mapped())
        if include_conflicts:
            payload["conflicts"] = [c.to_dict() for c in self.conflicts()]
        return payload

    def to_json(self, path: str | Path, include_conflicts: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(include_conflicts), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path, config: "ConflictConfig | dict | str | Path") -> "CriterionRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = load_config(config)
        criteria = [Criterion.from_dict(d) for d in data.get("criteria", [])]
        return cls(criteria=criteria, config=cfg)
