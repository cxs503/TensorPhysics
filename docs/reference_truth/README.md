# Reference-Truth Certificate System

## Overview

The Reference-Truth Certificate system provides a convergence-certification framework for reference values used as acceptance criteria in TensorPhysics campaign nodes. The core principle is: **un-certified references are barred from serving as downstream verification baselines**.

This system enables the critical distinction between **"model failure"** (the simulation is wrong) and **"ruler failure"** (the reference value we're comparing against is unreliable).

## Design Principles

1. **Four evidence classes**: analytical, self-convergence, experimental, statistical
2. **Type-specific validation**: each class has minimum evidence requirements
3. **Content-addressable**: SHA-256 digest of evidence payload; data change = digest change = invalidation
4. **Lifecycle management**: draft → valid → expired/revoked
5. **Gate enforcement**: reference gate blocks citation of un-certified references
6. **Zero domain semantics**: the system is domain-agnostic; domain content arrives through evidence payloads

## Schema (v1.0.0)

### Certificate Structure

```python
ReferenceTruthCertificate:
    # Identity
    reference_id: str           # e.g. "IR-L01"
    reference_name: str         # human-readable
    schema_version: str         # "1.0.0"

    # Classification
    certificate_type: CertificateType  # analytical|self_convergence|experimental|statistical
    status: CertificateStatus          # draft|valid|expired|revoked|insufficient

    # Core evidence
    reference_value: float      # the number being vouched
    unit: str                   # physical unit
    quantity_description: str   # what physical quantity

    # Type-specific evidence (only one populated)
    analytical_evidence: dict   # exact_formula, symbolic_equivalence, source_equation
    refinement_sequence: list   # [{level_index, resolution_parameter, quantity_of_interest}]
    richardson: dict            # {observed_order_p, extrapolated_value, gci_fine, gci_coarse}
    statistical: dict           # {n_samples, mean, standard_error, ci_lower, ci_upper, ...}
    experimental: dict          # {source_reference, measurement_method, stated_uncertainty, ...}

    # Caliber & boundaries
    caliber_declaration: str    # what this reference IS valid for
    disclaimer_boundaries: list[str]  # where it must NOT be used

    # Provenance
    source_citation: str        # literature source
    version_sha256: str         # content hash of evidence payload
    certified_by: str           # mission/agent that issued this
    certification_timestamp: str  # ISO-8601

    # Validation outcome
    validation_notes: list[str]
    minimum_requirements_met: bool
```

### Certificate Types

| Type | Evidence Required | Minimum Requirements |
|------|------------------|---------------------|
| **analytical** | `exact_formula` or `symbolic_equivalence=True` | Finite reference value; symbolic check passes |
| **self_convergence** | ≥3 refinement levels + Richardson extrapolation | p ≥ 0.5; GCI computed; monotonic refinement |
| **experimental** | Source reference, method, uncertainty | stated_uncertainty > 0; confidence ≥ 90% |
| **statistical** | n_samples, mean, SEM, CI | n ≥ 30; CI brackets mean; confidence ≥ 90% |

## Validation Rules

### Analytical
- Must provide `exact_formula`, `identity_check`, or `symbolic_equivalence`
- If `symbolic_equivalence` is provided, it must be `True`
- Reference value must be finite

### Self-Convergence
- ≥3 refinement levels required
- Each level must have `resolution_parameter` and `quantity_of_interest`
- Richardson extrapolation must provide `observed_order_p` ≥ 0.5
- GCI (Grid Convergence Index) must be computed for finest pair
- Resolution parameters should be monotonically decreasing

### Experimental
- Must cite `source_reference` (paper/report)
- Must state `measurement_method`
- Must provide `stated_uncertainty` > 0
- `confidence_level` ≥ 0.90

### Statistical
- `n_samples` ≥ 30
- Must provide `mean`, `standard_error`, `ci_lower`, `ci_upper`
- CI must bracket the mean
- `confidence_level` ≥ 0.90
- Power analysis recommended (warning if absent)

## Reference Gate

The gate enforces certificate requirements before any criterion can cite a reference:

```python
gate = ReferenceGate(store)
result = gate.check("IR-L01")

if result.decision == GateDecision.ALLOWED:
    # Safe to use as acceptance criterion
    ...
elif result.decision == GateDecision.BLOCKED_NO_CERT:
    # No certificate exists — cannot use
    ...
elif result.decision == GateDecision.BLOCKED_REVOKED:
    # Certificate was revoked — cannot use
    ...
```

### Gate Decisions

| Decision | Meaning |
|----------|---------|
| `ALLOWED` | Certificate is VALID; reference may be cited |
| `BLOCKED_NO_CERT` | No certificate found for this reference_id |
| `BLOCKED_INVALID` | Certificate exists but status ≠ VALID |
| `BLOCKED_REVOKED` | Certificate was explicitly revoked |
| `BLOCKED_EXPIRED` | Certificate has expired |

## Store & Persistence

Certificates are stored in a `CertificateStore` with JSON persistence:

```python
store = CertificateStore()
store.add(certificate)
store.save_to_file("certificates.json")

# Later:
store2 = CertificateStore()
store2.load_from_file("certificates.json")
```

## Backfill: Gold-Standard Certificates

The first wave of gold-standard certifications covers 26 references from existing benchmarks:

| Family | Count | Type | Source |
|--------|-------|------|--------|
| Level-ice (Lindqvist) | 8 | analytical | L2 anchor corpus |
| Brash-ice (Keinonen) | 6 | analytical | L2 anchor corpus |
| Ridge resistance | 7 | analytical | L2 anchor corpus |
| Pressure-area (Riska-Kujala) | 3 | analytical | L2 anchor corpus |
| Hetényi beam | 1 | analytical | Ice-beam FSI |
| Lindqvist components | 1 | analytical | ice_nav_refs |

All 26 certificates validated as VALID with 100% valid_rate.

## Usage in Campaign Nodes

### Before (uncontrolled):
```python
# Criterion cites a reference without checking its validity
assert computed_value == reference_value  # Is the reference trustworthy?
```

### After (certificate-gated):
```python
from tensorphysics.reference_truth import ReferenceGate, CertificateStore

store = CertificateStore()
store.load_from_file("artifacts/reference_truth_certificates.json")
gate = ReferenceGate(store)

# Check before citing
result = gate.check("IR-L01")
assert result.decision == "allowed", f"Cannot cite IR-L01: {result.reason}"

# Now safe to use as acceptance criterion
assert abs(computed_value - reference_value) < tolerance
```

## Failure Attribution

This system enables precise failure attribution:

- **Model failure**: Certificate is VALID, but simulation doesn't match → the model is wrong
- **Ruler failure**: Certificate is INSUFFICIENT/REVOKED → the reference is unreliable; cannot conclude anything about the model

## Limitations & Future Work

1. **Symbolic equivalence**: Currently relies on manual declaration; future integration with SymPy for automated symbolic checking
2. **Richardson extrapolation**: Assumes monotonic convergence; non-smooth convergence patterns need specialized treatment
3. **Experimental uncertainty**: Currently accepts stated uncertainty at face value; future work could cross-validate against multiple sources
4. **Certificate chaining**: A criterion may depend on multiple references; transitive validity checking is not yet implemented
5. **Temporal expiry**: Certificates currently don't auto-expire; manual lifecycle management required

## Files

```
src/tensorphysics/reference_truth/
    __init__.py          # Public API
    schema.py            # Certificate data model & serialization
    validator.py         # Type-specific validation rules
    gate.py              # Reference gate enforcement
    store.py             # JSON-backed certificate store
    backfill.py          # Gold-standard certificate generation

tests/
    test_reference_truth_certificate.py  # 27 tests (all passing)

artifacts/
    reference_truth_certificates.json    # 26 gold-standard certificates
```
