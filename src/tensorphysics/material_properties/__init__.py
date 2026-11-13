"""Material & fluid property database (MPDB v1).

A single, provenance-carrying asset for the properties every solver keeps
re-declaring: water and seawater (density/viscosity/cp/k/sound speed vs
temperature and salinity), air, fresh-water and sea ice mechanics, seabed
sediment and ship structural materials.

Public surface
--------------
``MaterialDB.load()``            load the packaged database
``db.get(substance, prop, ...)`` one value + provenance (SI, no silent
                                 extrapolation, no silent ambiguity)
``db.query(substance, props)``   several properties at the same conditions
``db.summary()``                 row/class/source counts
``build.build_records()``        deterministic rebuild of every row
``adapters.*``                   solver-side integration samples (LBM/FVM/FEM/
                                 ship resistance)
``inventory.scan_repo()``        stocktake of scattered carriers in a consumer repo

Rebuild::

    python scripts/build_material_properties_db.py

Docs: ``docs/MATERIAL_PROPERTIES_DB_V1.md``.
"""

from __future__ import annotations

from .adapters import (
    fem_ice_material,
    fvm_air_state,
    ittc57_cf,
    lbm_scaling,
    plate_rigidity,
    provenance_ledger,
    seawater_consistency_check,
    ship_resistance_env,
)
from .build import build, build_records, check_committed
from .inventory import scan_repo, summarise, to_csv, to_markdown
from .query import (
    AmbiguousValueError,
    MaterialDB,
    MissingConditionError,
    MPDBError,
    OutOfDomainError,
    PropertyValue,
    UnknownPropertyError,
    UnknownSubstanceError,
    load_default,
)
from .schema import (
    CONDITION_AXES,
    NON_SI_ACCESS_UNITS,
    PROPERTY_UNITS,
    SCHEMA_VERSION,
    SUBSTANCE_CLASSES,
    celsius,
    kelvin,
    to_json_schema,
    validate_record,
    validate_records,
)
from .sources import ANCHORS, SOURCE_REGISTRY

__all__ = [
    # database / query
    "MaterialDB",
    "PropertyValue",
    "load_default",
    "MPDBError",
    "UnknownSubstanceError",
    "UnknownPropertyError",
    "OutOfDomainError",
    "AmbiguousValueError",
    "MissingConditionError",
    # schema
    "SCHEMA_VERSION",
    "PROPERTY_UNITS",
    "SUBSTANCE_CLASSES",
    "CONDITION_AXES",
    "NON_SI_ACCESS_UNITS",
    "celsius",
    "kelvin",
    "validate_record",
    "validate_records",
    "to_json_schema",
    # build
    "build",
    "build_records",
    "check_committed",
    # adapters
    "lbm_scaling",
    "fvm_air_state",
    "fem_ice_material",
    "ship_resistance_env",
    "plate_rigidity",
    "ittc57_cf",
    "provenance_ledger",
    "seawater_consistency_check",
    # inventory
    "scan_repo",
    "summarise",
    "to_csv",
    "to_markdown",
    # sources
    "SOURCE_REGISTRY",
    "ANCHORS",
]