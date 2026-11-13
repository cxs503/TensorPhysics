"""Query surface for MPDB v1: ``substance x conditions -> value + provenance``.

Design rules
------------
* **No silent extrapolation.**  Every property carries the validity box of its
  source; a request outside that box raises :class:`OutOfDomainError` instead of
  returning a number the source never supported (``clamp=True` opts into the
  nearest valid state and says so in ``PropertyValue.interpolated``).
* **No silent ambiguity.**  Materials with several states (first-year vs
  multi-year ice, initial vs effective modulus) require ``variant=``; the error
  message lists the available variants.
* **Provenance travels with the number.**  :class:`PropertyValue` carries the
  source citation, locator, evidence class and the row ids used, so a solver can
  persist the provenance into its own run ledger (see ``adapters``).

Example
-------
>>> from tensorphysics.material_properties import MaterialDB
>>> db = MaterialDB.load()
>>> db.get("seawater", "density", T_C=15.0, salinity=35.0).value
1025.75...
>>> db.get("water", "kinematic_viscosity", T_C=15.0).provenance["source_id"]
'SRC-IAPWS-WATER-VISCOSITY'
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .schema import SCHEMA_VERSION, celsius, kelvin

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_NAME = "mpdb_v1"


class MPDBError(Exception):
    """Base class for query-surface errors."""


class UnknownSubstanceError(MPDBError, KeyError):
    pass


class UnknownPropertyError(MPDBError, KeyError):
    pass


class OutOfDomainError(MPDBError, ValueError):
    pass


class AmbiguousValueError(MPDBError, ValueError):
    pass


class MissingConditionError(MPDBError, ValueError):
    pass


@dataclass(frozen=True)
class PropertyValue:
    """One queried property value together with its provenance."""

    substance: str
    prop: str
    value: float
    unit: str
    conditions: Mapping[str, Optional[float]]
    variant: Optional[str]
    provenance: Mapping[str, Any]
    uncertainty_rel: Optional[float]
    uncertainty_basis: Optional[str]
    interpolated: Mapping[str, str] = field(default_factory=dict)
    row_ids: Sequence[str] = field(default_factory=tuple)
    derived_from: Optional[str] = None

    #: ``property`` is also the attribute name used in the JSON payload
    @property
    def property(self) -> str:  # noqa: A003 - mirror the schema key
        return self.prop

    def to_dict(self) -> Dict[str, Any]:
        return {
            "substance": self.substance,
            "property": self.prop,
            "value": self.value,
            "unit": self.unit,
            "conditions": dict(self.conditions),
            "variant": self.variant,
            "provenance": dict(self.provenance),
            "uncertainty": {"rel": self.uncertainty_rel, "basis": self.uncertainty_basis},
            "interpolated": dict(self.interpolated),
            "row_ids": list(self.row_ids),
            "derived_from": self.derived_from,
        }

    def __float__(self) -> float:  # convenience: float(db.get(...))
        return float(self.value)


def _axis_values(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[float]]:
    axes: Dict[str, List[float]] = {}
    for r in rows:
        for ax, val in r["conditions"].items():
            if val is None:
                continue
            axes.setdefault(ax, [])
            if float(val) not in axes[ax]:
                axes[ax].append(float(val))
    return {ax: sorted(v) for ax, v in axes.items()}


class MaterialDB:
    """In-memory query surface over the MPDB rows."""

    def __init__(self, records: Sequence[Mapping[str, Any]],
                 substances: Optional[Mapping[str, Any]] = None,
                 meta: Optional[Mapping[str, Any]] = None):
        self._records: List[dict] = [dict(r) for r in records]
        self._by_sp: Dict[tuple, List[dict]] = {}
        for r in self._records:
            self._by_sp.setdefault((r["substance"], r["property"]), []).append(r)
        self._substances: Dict[str, dict] = dict(substances or {})
        if not self._substances:
            for r in self._records:
                self._substances.setdefault(r["substance"], {
                    "class": r["substance_class"],
                    "label": r["substance"],
                    "use_cases": [],
                })
        self.meta: Dict[str, Any] = dict(meta or {})

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: Optional[str] = None, data_dir: str = DATA_DIR) -> "MaterialDB":
        jsonl = path or os.path.join(data_dir, f"{DB_NAME}.jsonl")
        if not os.path.exists(jsonl):
            raise FileNotFoundError(
                f"{jsonl} not found -- rebuild it with: python scripts/build_material_properties_db.py")
        with open(jsonl, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        substances: Dict[str, dict] = {}
        sub_path = os.path.join(os.path.dirname(jsonl), f"{DB_NAME}.substances.json")
        if os.path.exists(sub_path):
            substances = json.load(open(sub_path, encoding="utf-8"))
        meta_path = os.path.join(os.path.dirname(jsonl), f"{DB_NAME}.meta.json")
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        return cls(records, substances, meta)

    # ------------------------------------------------------------- inventory
    @property
    def schema_version(self) -> str:
        return SCHEMA_VERSION

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> List[dict]:
        return [dict(r) for r in self._records]

    def substances(self) -> Dict[str, dict]:
        return {k: dict(v) for k, v in sorted(self._substances.items())}

    def classes(self) -> List[str]:
        return sorted({r["substance_class"] for r in self._records})

    def props_for(self, substance: str) -> List[str]:
        if substance not in self._substances:
            raise UnknownSubstanceError(
                f"unknown substance {substance!r}; known: {sorted(self._substances)}")
        return sorted(r["property"] for r in self._records if r["substance"] == substance)

    def table(self, substance: str, prop: str, variant: Optional[str] = None) -> List[dict]:
        rows = self._rows(substance, prop)
        if variant is not None:
            rows = [r for r in rows if r.get("variant") == variant]
        return [dict(r) for r in rows]

    def domain(self, substance: str, prop: str) -> Dict[str, Dict[str, float]]:
        rows = self._rows(substance, prop)
        boxes: Dict[str, Dict[str, float]] = {}
        for r in rows:
            for ax, (lo, hi) in r["domain"].items():
                b = boxes.setdefault(ax, {"min": float(lo), "max": float(hi)})
                b["min"] = max(b["min"], float(lo))
                b["max"] = min(b["max"], float(hi))
        return boxes

    # ------------------------------------------------------------------ query
    def _rows(self, substance: str, prop: str) -> List[dict]:
        if substance not in self._substances:
            raise UnknownSubstanceError(
                f"unknown substance {substance!r}; known: {sorted(self._substances)}")
        rows = self._by_sp.get((substance, prop))
        if not rows:
            raise UnknownPropertyError(
                f"property {prop!r} not in MPDB for {substance!r}; available: {self.props_for(substance)}")
        return rows

    def get(self, substance: str, prop: str, T: Optional[float] = None,
            p: Optional[float] = None, salinity: Optional[float] = None,
            T_C: Optional[float] = None, variant: Optional[str] = None,
            clamp: bool = False) -> PropertyValue:
        """Query one property.

        Parameters
        ----------
        T : absolute temperature [K]
        T_C : temperature in degrees Celsius (convenience access unit)
        p : absolute pressure [Pa]
        salinity : bulk salinity [g/kg]
        variant : state label for materials with several states
        clamp : if True, clamp to the nearest valid state instead of raising
        """
        if T is not None and T_C is not None:
            raise MPDBError("give either T [K] or T_C [degC], not both")
        if T_C is not None:
            T = kelvin(T_C)
        req = {"T_K": T, "p_Pa": p, "salinity_g_per_kg": salinity}
        rows = self._rows(substance, prop)

        variants = sorted({r.get("variant") for r in rows if r.get("variant")})
        if variant is None and len({r.get("variant") for r in rows}) > 1 and variants:
            raise AmbiguousValueError(
                f"{substance}.{prop} has several states; pass variant= one of {variants}")
        if variant is not None:
            rows = [r for r in rows if r.get("variant") == variant]
            if not rows:
                raise AmbiguousValueError(
                    f"variant {variant!r} not available for {substance}.{prop}; "
                    f"available: {variants or 'none'}")

        axes = _axis_values(rows)
        # No silent extrapolation: guard against the *sampled* box (what the rows
        # actually cover) and report the *source* box when they differ, because the
        # correlation may be valid beyond the sampled states but this database is not.
        dom = self.domain(substance, prop)
        used_clamp: Dict[str, str] = {}
        req_eff = dict(req)
        for ax, samples in axes.items():
            val = req.get(ax)
            if val is None and len(samples) == 1:
                # single-state axis (e.g. p = 1 atm in every row): adopt it, it is the
                # only state the source covers, and say so in ``interpolated``.
                req_eff[ax] = samples[0]
                used_clamp[ax] = f"single sampled state {samples[0]:g}"
                continue
            if val is None and ax == "p_Pa":
                # standard-atmosphere default, recorded so the caller can see it
                req_eff[ax] = 101325.0
                used_clamp[ax] = "default standard atmosphere 101325 Pa"
                continue
            if val is None:
                raise MissingConditionError(
                    f"{substance}.{prop} depends on {ax}; supply it (sampled "
                    f"{samples[0]:g}..{samples[-1]:g})")
            lo, hi = samples[0], samples[-1]
            if not (lo - 1e-12 <= float(val) <= hi + 1e-12):
                src = dom.get(ax)
                src_txt = (f"; source validity {src['min']:g}..{src['max']:g}" if src else "")
                if not clamp:
                    raise OutOfDomainError(
                        f"{ax} = {val:g} is outside the sampled range [{lo:g}, {hi:g}] for "
                        f"{substance}.{prop}{src_txt}; pass clamp=True to saturate at the "
                        f"nearest sampled state")
                req_eff[ax] = min(max(float(val), lo), hi)
                used_clamp[ax] = f"clamped to {req_eff[ax]:g} (sampled {lo:g}..{hi:g})"

        if not axes:
            unique = {(r["value"]) for r in rows}
            if len(rows) > 1 and len(unique) > 1:
                raise AmbiguousValueError(
                    f"{substance}.{prop} has {len(rows)} distinct rows and no condition axis; "
                    "select variant=")
            row = rows[0]
            return self._pv(row, req_eff, used_clamp)

        missing = [ax for ax in axes if req_eff.get(ax) is None]
        if missing:
            raise MissingConditionError(
                f"{substance}.{prop} depends on {missing}; valid ranges: "
                + ", ".join(f"{ax} = {axes[ax][0]:g}..{axes[ax][-1]:g}" for ax in missing))

        grid = list(itertools.product(*[axes[ax] for ax in axes]))
        value_at: Dict[tuple, float] = {}
        row_at: Dict[tuple, dict] = {}
        for key in grid:
            cond = dict(zip(axes, key))
            for r in rows:
                ok = True
                for ax, v in cond.items():
                    rv = r["conditions"].get(ax)
                    if rv is None or abs(float(rv) - v) > 1e-9 * max(1.0, abs(v)):
                        ok = False
                        break
                if ok:
                    value_at[key] = float(r["value"])
                    row_at[key] = r
                    break

        # multilinear interpolation over the available axes
        weights = {}
        for ax in axes:
            x = float(req_eff[ax])
            xs = axes[ax]
            if len(xs) == 1:
                weights[ax] = [(xs[0], 1.0)]
                continue
            if x <= xs[0]:
                weights[ax] = [(xs[0], 1.0)]
            elif x >= xs[-1]:
                weights[ax] = [(xs[-1], 1.0)]
            else:
                for lo, hi in zip(xs, xs[1:]):
                    if lo <= x <= hi:
                        w = (x - lo) / (hi - lo)
                        weights[ax] = [(lo, 1.0 - w), (hi, w)]
                        break
        total = 0.0
        used_rows: List[str] = []
        interp: Dict[str, str] = dict(used_clamp)
        for ax in axes:
            if len(axes[ax]) == 1:
                continue
            x = float(req_eff[ax])
            if x in axes[ax]:
                continue
            interp[ax] = "linear" if x < axes[ax][-1] else "endpoint"
        for combo in itertools.product(*[weights[ax] for ax in axes]):
            key = tuple(c[0] for c in combo)
            w = 1.0
            for c in combo:
                w *= c[1]
            if w == 0.0 or key not in value_at:
                continue
            total += w * value_at[key]
            used_rows.append(row_at[key]["id"])
        if not used_rows:
            raise MPDBError(
                f"could not form an interpolation cell for {substance}.{prop} at {req_eff}")
        # provenance: the row whose evidence dominates (exact match, else first contributor)
        prov_row = rows[0]
        for r in rows:
            if r["id"] == used_rows[0]:
                prov_row = r
                break
        pv = self._pv(prov_row, req_eff, used_clamp)
        return PropertyValue(
            substance=pv.substance, prop=pv.prop, value=total, unit=pv.unit,
            conditions=pv.conditions, variant=pv.variant, provenance=pv.provenance,
            uncertainty_rel=pv.uncertainty_rel, uncertainty_basis=pv.uncertainty_basis,
            interpolated=interp, row_ids=tuple(sorted(set(used_rows))),
            derived_from=pv.derived_from,
        )

    def _pv(self, row: Mapping[str, Any], req: Mapping[str, Optional[float]],
            interp: Mapping[str, str]) -> PropertyValue:
        cond = dict(row["conditions"])
        for ax, val in req.items():
            if val is not None:
                cond[ax] = float(val)
        return PropertyValue(
            substance=row["substance"], prop=row["property"], value=float(row["value"]),
            unit=row["unit"], conditions=cond, variant=row.get("variant"),
            provenance=dict(row["provenance"]),
            uncertainty_rel=row["uncertainty"]["rel"],
            uncertainty_basis=row["uncertainty"]["basis"],
            interpolated=dict(interp), row_ids=(row["id"],),
            derived_from=row.get("derived_from"),
        )

    def query(self, substance: str, properties: Sequence[str], **conditions: Any) -> Dict[str, PropertyValue]:
        """Query several properties of one substance at the same conditions."""
        return {p: self.get(substance, p, **conditions) for p in properties}

    # ---------------------------------------------------------------- summary
    def summary(self) -> Dict[str, Any]:
        by_class: Dict[str, int] = {}
        by_substance: Dict[str, int] = {}
        for r in self._records:
            by_class[r["substance_class"]] = by_class.get(r["substance_class"], 0) + 1
            by_substance[r["substance"]] = by_substance.get(r["substance"], 0) + 1
        return {
            "schema_version": SCHEMA_VERSION,
            "rows": len(self._records),
            "substances": len(self._substances),
            "classes": dict(sorted(by_class.items())),
            "rows_by_substance": dict(sorted(by_substance.items())),
            "properties": len({r["property"] for r in self._records}),
            "sources": len({r["provenance"]["source_id"] for r in self._records}),
            "jsonl_sha256": self.meta.get("jsonl_sha256"),
        }


def load_default() -> MaterialDB:
    """Load the packaged database (raises if the artefacts were not built)."""
    return MaterialDB.load()


__all__ = [
    "MaterialDB",
    "PropertyValue",
    "load_default",
    "MPDBError",
    "UnknownSubstanceError",
    "UnknownPropertyError",
    "OutOfDomainError",
    "AmbiguousValueError",
    "MissingConditionError",
    "celsius",
    "kelvin",
]