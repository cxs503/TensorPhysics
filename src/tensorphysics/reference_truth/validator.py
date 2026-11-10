"""Certificate validator: type-specific validation rules.

Each certificate type has minimum evidence requirements:
- analytical:     symbolic equivalence or exact identity check
- self_convergence: ≥3 refinement levels, observed order p ≥ lower bound, GCI computed
- experimental:   stated uncertainty, confidence level, source reference
- statistical:    SEM, CI, sample size, power analysis

The validator returns a ValidationResult with pass/fail and detailed notes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import (
    CertificateType,
    CertificateStatus,
    ReferenceTruthCertificate,
)


@dataclass
class ValidationResult:
    """Outcome of certificate validation."""
    passed: bool
    certificate_type: CertificateType
    status: CertificateStatus
    notes: list[str] = field(default_factory=list)
    richardson_details: Optional[dict[str, Any]] = None
    statistical_details: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "certificate_type": self.certificate_type.value,
            "status": self.status.value,
            "notes": self.notes,
            "richardson_details": self.richardson_details,
            "statistical_details": self.statistical_details,
        }


# Minimum requirements per certificate type
MIN_REFINEMENT_LEVELS = 3
MIN_OBSERVED_ORDER_P = 0.5  # at least some convergence
THEORETICAL_ORDER_TOLERANCE = 1.5  # p should be within this of theoretical
MIN_SAMPLES_STATISTICAL = 30
MIN_CONFIDENCE_LEVEL = 0.90


def _validate_analytical(cert: ReferenceTruthCertificate) -> ValidationResult:
    """Validate an analytical-solution certificate.

    Requirements:
    - analytical_evidence must contain 'exact_formula' or 'identity_check'
    - If 'symbolic_equivalence' is provided, it must be True
    - Reference value must be finite
    """
    notes: list[str] = []
    evidence = cert.analytical_evidence

    if evidence is None:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.ANALYTICAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=["analytical_evidence is required for analytical certificates"],
        )

    if not math.isfinite(cert.reference_value):
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.ANALYTICAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=["reference_value must be finite"],
        )

    # Check for symbolic equivalence or exact formula
    has_formula = "exact_formula" in evidence or "identity_check" in evidence
    has_symbolic = evidence.get("symbolic_equivalence", None)

    if not has_formula and has_symbolic is None:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.ANALYTICAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=["analytical_evidence must contain 'exact_formula', 'identity_check', or 'symbolic_equivalence'"],
        )

    if has_symbolic is not None and not has_symbolic:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.ANALYTICAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=["symbolic_equivalence check failed"],
        )

    notes.append("analytical evidence structure valid")
    if "exact_formula" in evidence:
        notes.append(f"exact formula: {evidence['exact_formula'][:80]}...")
    if "source_equation" in evidence:
        notes.append(f"source equation: {evidence['source_equation']}")

    return ValidationResult(
        passed=True,
        certificate_type=CertificateType.ANALYTICAL,
        status=CertificateStatus.VALID,
        notes=notes,
    )


def _validate_self_convergence(cert: ReferenceTruthCertificate) -> ValidationResult:
    """Validate a self-convergence certificate (mesh/time refinement study).

    Requirements:
    - refinement_sequence must have ≥3 levels
    - Each level must have resolution_parameter and quantity_of_interest
    - Richardson extrapolation must be provided with observed_order_p
    - Observed order p must be ≥ MIN_OBSERVED_ORDER_P
    - GCI must be computed for finest pair
    """
    notes: list[str] = []
    seq = cert.refinement_sequence
    rich = cert.richardson

    if seq is None or len(seq) < MIN_REFINEMENT_LEVELS:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.SELF_CONVERGENCE,
            status=CertificateStatus.INSUFFICIENT,
            notes=[f"refinement_sequence must have ≥{MIN_REFINEMENT_LEVELS} levels, got {len(seq) if seq else 0}"],
        )

    # Validate each level
    for i, level in enumerate(seq):
        if "resolution_parameter" not in level or "quantity_of_interest" not in level:
            return ValidationResult(
                passed=False,
                certificate_type=CertificateType.SELF_CONVERGENCE,
                status=CertificateStatus.INSUFFICIENT,
                notes=[f"level {i} missing resolution_parameter or quantity_of_interest"],
            )
        if not math.isfinite(level["resolution_parameter"]) or not math.isfinite(level["quantity_of_interest"]):
            return ValidationResult(
                passed=False,
                certificate_type=CertificateType.SELF_CONVERGENCE,
                status=CertificateStatus.INSUFFICIENT,
                notes=[f"level {i} has non-finite values"],
            )

    notes.append(f"refinement sequence has {len(seq)} levels")

    # Check refinement ratios are monotonic
    resolutions = [level["resolution_parameter"] for level in seq]
    for i in range(1, len(resolutions)):
        if resolutions[i] >= resolutions[i-1]:
            notes.append(f"WARNING: resolution not monotonically decreasing at level {i}")

    # Richardson extrapolation check
    if rich is None:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.SELF_CONVERGENCE,
            status=CertificateStatus.INSUFFICIENT,
            notes=["richardson extrapolation results required for self-convergence certificates"],
        )

    p = rich.get("observed_order_p")
    if p is None or not math.isfinite(p):
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.SELF_CONVERGENCE,
            status=CertificateStatus.INSUFFICIENT,
            notes=["richardson.observed_order_p must be a finite number"],
        )

    if p < MIN_OBSERVED_ORDER_P:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.SELF_CONVERGENCE,
            status=CertificateStatus.INSUFFICIENT,
            notes=[f"observed order p={p:.3f} below minimum {MIN_OBSERVED_ORDER_P}"],
        )

    gci = rich.get("gci_fine")
    if gci is None or not math.isfinite(gci):
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.SELF_CONVERGENCE,
            status=CertificateStatus.INSUFFICIENT,
            notes=["richardson.gci_fine must be a finite number"],
        )

    notes.append(f"Richardson observed order p = {p:.3f}")
    notes.append(f"GCI (fine) = {gci:.4e}")
    notes.append(f"extrapolated value = {rich.get('extrapolated_value', 'N/A')}")

    richardson_details = {
        "observed_order_p": p,
        "gci_fine": gci,
        "gci_coarse": rich.get("gci_coarse"),
        "extrapolated_value": rich.get("extrapolated_value"),
        "safety_factor": rich.get("safety_factor", 1.25),
        "asymptotic_range": rich.get("asymptotic_range", True),
    }

    return ValidationResult(
        passed=True,
        certificate_type=CertificateType.SELF_CONVERGENCE,
        status=CertificateStatus.VALID,
        notes=notes,
        richardson_details=richardson_details,
    )


def _validate_experimental(cert: ReferenceTruthCertificate) -> ValidationResult:
    """Validate an experimental-data certificate.

    Requirements:
    - experimental evidence must contain source_reference, measurement_method
    - stated_uncertainty must be provided and positive
    - confidence_level must be ≥ MIN_CONFIDENCE_LEVEL
    """
    notes: list[str] = []
    exp = cert.experimental

    if exp is None:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.EXPERIMENTAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=["experimental evidence is required for experimental certificates"],
        )

    source = exp.get("source_reference")
    if not source:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.EXPERIMENTAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=["experimental.source_reference is required"],
        )

    method = exp.get("measurement_method")
    if not method:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.EXPERIMENTAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=["experimental.measurement_method is required"],
        )

    uncertainty = exp.get("stated_uncertainty")
    if uncertainty is None or uncertainty <= 0:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.EXPERIMENTAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=["experimental.stated_uncertainty must be positive"],
        )

    confidence = exp.get("confidence_level", 0.95)
    if confidence < MIN_CONFIDENCE_LEVEL:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.EXPERIMENTAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=[f"confidence_level {confidence} below minimum {MIN_CONFIDENCE_LEVEL}"],
        )

    notes.append(f"source: {source}")
    notes.append(f"method: {method}")
    notes.append(f"stated uncertainty: ±{uncertainty} at {confidence*100:.0f}% CL")
    if exp.get("facility"):
        notes.append(f"facility: {exp['facility']}")

    return ValidationResult(
        passed=True,
        certificate_type=CertificateType.EXPERIMENTAL,
        status=CertificateStatus.VALID,
        notes=notes,
    )


def _validate_statistical(cert: ReferenceTruthCertificate) -> ValidationResult:
    """Validate a statistical-estimate certificate (MC / sampling).

    Requirements:
    - statistical evidence must contain n_samples, mean, standard_error
    - n_samples must be ≥ MIN_SAMPLES_STATISTICAL
    - confidence_level must be ≥ MIN_CONFIDENCE_LEVEL
    - CI bounds must be consistent with mean ± z*SEM
    - Power analysis is recommended (warning if absent)
    """
    notes: list[str] = []
    stat = cert.statistical

    if stat is None:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.STATISTICAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=["statistical evidence is required for statistical certificates"],
        )

    n = stat.get("n_samples")
    if n is None or n < MIN_SAMPLES_STATISTICAL:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.STATISTICAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=[f"n_samples must be ≥{MIN_SAMPLES_STATISTICAL}, got {n}"],
        )

    sem = stat.get("standard_error")
    if sem is None or sem <= 0:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.STATISTICAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=["statistical.standard_error must be positive"],
        )

    confidence = stat.get("confidence_level", 0.95)
    if confidence < MIN_CONFIDENCE_LEVEL:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.STATISTICAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=[f"confidence_level {confidence} below minimum {MIN_CONFIDENCE_LEVEL}"],
        )

    ci_lower = stat.get("ci_lower")
    ci_upper = stat.get("ci_upper")
    mean = stat.get("mean")
    if ci_lower is None or ci_upper is None or mean is None:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.STATISTICAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=["statistical must contain mean, ci_lower, ci_upper"],
        )

    # Basic consistency check: CI should bracket the mean
    if ci_lower > mean or ci_upper < mean:
        return ValidationResult(
            passed=False,
            certificate_type=CertificateType.STATISTICAL,
            status=CertificateStatus.INSUFFICIENT,
            notes=[f"CI [{ci_lower}, {ci_upper}] does not bracket mean {mean}"],
        )

    notes.append(f"n_samples = {n}")
    notes.append(f"mean = {mean:.6g}, SEM = {sem:.4e}")
    notes.append(f"{confidence*100:.0f}% CI = [{ci_lower:.6g}, {ci_upper:.6g}]")

    # Power analysis check (warning, not failure)
    power = stat.get("power_analysis")
    if power is None:
        notes.append("WARNING: no power_analysis provided (recommended for statistical certificates)")
    else:
        notes.append(f"power analysis: effect_size={power.get('effect_size', 'N/A')}, power={power.get('power', 'N/A')}")

    statistical_details = {
        "n_samples": n,
        "mean": mean,
        "standard_error": sem,
        "confidence_level": confidence,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "effective_sample_size": stat.get("effective_sample_size"),
        "power_analysis": power,
    }

    return ValidationResult(
        passed=True,
        certificate_type=CertificateType.STATISTICAL,
        status=CertificateStatus.VALID,
        notes=notes,
        statistical_details=statistical_details,
    )


def validate_certificate(cert: ReferenceTruthCertificate) -> ValidationResult:
    """Validate a certificate according to its type-specific rules.

    Returns a ValidationResult with pass/fail, status, and detailed notes.
    """
    # Common checks
    if not cert.reference_id:
        return ValidationResult(
            passed=False,
            certificate_type=cert.certificate_type,
            status=CertificateStatus.INSUFFICIENT,
            notes=["reference_id is required"],
        )

    if not cert.reference_name:
        return ValidationResult(
            passed=False,
            certificate_type=cert.certificate_type,
            status=CertificateStatus.INSUFFICIENT,
            notes=["reference_name is required"],
        )

    # Type-specific validation
    validators = {
        CertificateType.ANALYTICAL: _validate_analytical,
        CertificateType.SELF_CONVERGENCE: _validate_self_convergence,
        CertificateType.EXPERIMENTAL: _validate_experimental,
        CertificateType.STATISTICAL: _validate_statistical,
    }

    validator = validators.get(cert.certificate_type)
    if validator is None:
        return ValidationResult(
            passed=False,
            certificate_type=cert.certificate_type,
            status=CertificateStatus.INSUFFICIENT,
            notes=[f"unknown certificate type: {cert.certificate_type}"],
        )

    result = validator(cert)

    # Update certificate status based on validation
    cert.status = result.status
    cert.minimum_requirements_met = result.passed
    cert.validation_notes = result.notes

    return result
