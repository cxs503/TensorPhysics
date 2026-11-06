"""Ice-navigation reference-value corpus (sha256-anchored validation data).

The corpus is a set of frozen, content-addressed reference values for the
ice-navigation performance chain (level-ice resistance components, brash-ice
resistance coefficients, ice pressure-area relations, Polar Class local panel
pressure and Polar Code requirements).  It exists so that campaign nodes can
assert *reference validation* -- "does my computed number stay inside the
literature/rule tolerance?" -- without re-typing literature numbers into tests,
and so that a change of a stored number is detectable through its sha256 anchor.

Provenance discipline
=====================
Every asset carries ``source_citation``, ``values``, ``tolerance``,
``applicable_regime`` plus ``evidence_tier``, ``role`` and
``primary_source_verified``.  ``role`` is the load-bearing field:

* ``accuracy_reference``  -- the number may be used to judge a computation.
* ``regression_anchor``   -- the number may only be used to detect drift of this
  repository's own implementation; it must not be quoted as external truth.

Assets whose value could not be checked against the primary document inside the
build environment are marked ``primary_source_verified=False``; the reason is
recorded verbatim in ``verification_note``.  See
``docs/ICE_NAV_REFS_CORPUS.md`` for the verification matrix.

Examples
--------
>>> from tensorphysics.ice_nav_refs import load_corpus
>>> corpus = load_corpus()                       # doctest: +SKIP
>>> corpus.check("RA_LOCAL_PRESSURE_TABLE", {"p_A1.0_MPa": 3.0})["pass"]  # doctest: +SKIP
True
"""

from .corpus import (
    EVIDENCE_TIERS,
    REGIME_REQUIRED_KEYS,
    REQUIRED_FIELDS,
    ROLES,
    SCHEMA,
    TOLERANCE_MODES,
    CorpusError,
    CorpusIntegrityError,
    MissingCitationError,
    MissingFieldError,
    ReferenceAsset,
    ReferenceCorpus,
    Tolerance,
    ToleranceError,
    asset_from_dict,
    canonical_json,
    check_against,
    default_root,
    describe_corpus,
    load_corpus,
    sha256_digest,
    value_digest,
    verify_manifest,
)

__all__ = [
    "SCHEMA",
    "REQUIRED_FIELDS",
    "EVIDENCE_TIERS",
    "ROLES",
    "TOLERANCE_MODES",
    "REGIME_REQUIRED_KEYS",
    "CorpusError",
    "CorpusIntegrityError",
    "MissingCitationError",
    "MissingFieldError",
    "ToleranceError",
    "Tolerance",
    "ReferenceAsset",
    "ReferenceCorpus",
    "asset_from_dict",
    "canonical_json",
    "check_against",
    "default_root",
    "describe_corpus",
    "load_corpus",
    "sha256_digest",
    "value_digest",
    "verify_manifest",
]