"""High-level ASME V&V 20 validation uncertainty engine.

This module provides a class-based, user-friendly wrapper around the
solver-neutral validation gate in :mod:`tensorphysics.vv.validation`.  It
implements the ASME V&V 20-2009 uncertainty framework:

    U_val = sqrt(U_num**2 + U_input**2 + U_model**2)

where ``U_num`` is the numerical uncertainty (normally the finest-grid GCI),
``U_input`` the input-parameter uncertainty (propagated through sensitivities),
and ``U_model`` the model-form uncertainty (the residual after the other two
are accounted for).  A prediction is *validated* when the comparison error
``|E|`` lies inside ``U_val``.

The class API is deliberately thin: all numerical work is delegated to the
functional gate so that the two layers cannot drift apart.  A caller that
needs the full check ledger should use :func:`tensorphysics.vv.validation.validate`
directly; this module is for the common "one quantity, one reference" case.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .validation import (
    ValidationResult,
    validate,
    validation_uncertainty,
)

__all__ = [
    "ASMEVV20",
    "UncertaintyBudget",
]


@dataclass
class UncertaintyBudget:
    """Complete ASME V&V 20 uncertainty budget (high-level API).

    ``u_num`` / ``u_input`` / ``u_model`` / ``u_val`` are **percentages** of
    the quantity being validated.  ``error`` is the comparison error
    ``|computed - experimental| / |experimental| * 100`` (percent).
    """

    u_num: float
    u_input: float
    u_model: float
    u_val: float
    experimental_value: Optional[float] = None
    computed_value: Optional[float] = None
    error: Optional[float] = None
    error_within_uncertainty: Optional[bool] = None
    quantity_name: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary."""
        d: dict[str, Any] = {
            "u_num": self.u_num,
            "u_input": self.u_input,
            "u_model": self.u_model,
            "u_val": self.u_val,
            "quantity_name": self.quantity_name,
            "notes": self.notes,
        }
        if self.experimental_value is not None:
            d["experimental_value"] = self.experimental_value
        if self.computed_value is not None:
            d["computed_value"] = self.computed_value
        if self.error is not None:
            d["error"] = self.error
        if self.error_within_uncertainty is not None:
            d["error_within_uncertainty"] = self.error_within_uncertainty
        return d

    def is_validated(self) -> bool:
        """True when the comparison error lies inside the uncertainty band."""
        if self.error_within_uncertainty is None:
            return False
        return self.error_within_uncertainty


class ASMEVV20:
    """ASME V&V 20 validation uncertainty engine (high-level API).

    Parameters
    ----------
    max_acceptable_uncertainty : float
        Maximum acceptable total validation uncertainty (percent).  Default 10.
    """

    def __init__(self, max_acceptable_uncertainty: float = 10.0):
        if not math.isfinite(max_acceptable_uncertainty) or max_acceptable_uncertainty <= 0.0:
            raise ValueError(
                f"max_acceptable_uncertainty must be finite and positive, "
                f"got {max_acceptable_uncertainty!r}"
            )
        self.max_acceptable_uncertainty = max_acceptable_uncertainty

    def compute_numerical_uncertainty(
        self,
        gci_fine: float,
        f_ext: Optional[float] = None,
        f_fine: Optional[float] = None,
    ) -> float:
        """Numerical uncertainty ``U_num`` (percent) from the finest-grid GCI.

        When ``f_ext`` and ``f_fine`` are supplied the extrapolation error
        ``|f_fine - f_ext| / |f_fine| * 100`` is also computed and the *larger*
        of the two is returned (conservative).
        """
        if not math.isfinite(gci_fine) or gci_fine < 0.0:
            raise ValueError(f"gci_fine must be finite and non-negative, got {gci_fine!r}")
        u_num = gci_fine
        if f_ext is not None and f_fine is not None:
            if not math.isfinite(f_ext) or not math.isfinite(f_fine):
                raise ValueError("f_ext and f_fine must be finite")
            if abs(f_fine) > 1e-14:
                u_alt = 100.0 * abs(f_fine - f_ext) / abs(f_fine)
                u_num = max(u_num, u_alt)
        return u_num

    def compute_input_uncertainty(
        self,
        input_params: Dict[str, dict],
        quantity_sensitivities: Optional[Dict[str, float]] = None,
    ) -> float:
        """Input-parameter uncertainty ``U_input`` (percent), RSS-combined.

        Each entry of ``input_params`` is ``{name: {value, uncertainty,
        is_relative?}}``.  ``uncertainty`` is an absolute value (same units as
        ``value``) unless ``is_relative`` is True, in which case it is a
        percent of ``value``.  When ``quantity_sensitivities`` is supplied the
        per-parameter contribution is ``|sensitivity| * relative_uncertainty``;
        otherwise the relative uncertainty is used directly.  Contributions
        are combined by root-sum-square.
        """
        if not input_params:
            return 0.0
        contributions: list[float] = []
        for name, spec in input_params.items():
            value = spec.get("value")
            unc = spec.get("uncertainty")
            if value is None or unc is None:
                raise ValueError(f"parameter {name!r} needs 'value' and 'uncertainty'")
            if not math.isfinite(value) or not math.isfinite(unc) or unc < 0.0:
                raise ValueError(
                    f"parameter {name!r}: value and uncertainty must be finite, "
                    f"uncertainty non-negative"
                )
            is_relative = spec.get("is_relative", False)
            if is_relative:
                rel = unc
            else:
                if abs(value) < 1e-14:
                    raise ValueError(f"parameter {name!r}: cannot compute relative uncertainty of zero")
                rel = 100.0 * unc / abs(value)
            if quantity_sensitivities and name in quantity_sensitivities:
                sens = quantity_sensitivities[name]
                if not math.isfinite(sens):
                    raise ValueError(f"sensitivity for {name!r} must be finite")
                rel = abs(sens) * rel
            contributions.append(rel)
        return math.sqrt(math.fsum(c * c for c in contributions))

    def compute_model_uncertainty(
        self,
        computed_value: float,
        experimental_value: float,
        u_num: float,
        u_input: float = 0.0,
    ) -> float:
        """Model-form uncertainty ``U_model`` (percent) from the residual.

        ``U_model = sqrt(max(0, E**2 - U_num**2 - U_input**2))`` where ``E``
        is the comparison error (percent).  A negative radicand (the declared
        uncertainties already cover the error) is clamped to zero.
        """
        if not math.isfinite(computed_value) or not math.isfinite(experimental_value):
            raise ValueError("computed_value and experimental_value must be finite")
        if abs(experimental_value) < 1e-14:
            raise ValueError("experimental_value must be non-zero")
        for name, value in (("u_num", u_num), ("u_input", u_input)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
        e = 100.0 * abs(computed_value - experimental_value) / abs(experimental_value)
        radicand = e * e - u_num * u_num - u_input * u_input
        return math.sqrt(max(0.0, radicand))

    def validate(
        self,
        computed_value: float,
        experimental_value: float,
        u_num: float,
        u_input: float = 0.0,
        u_model: float = 0.0,
        quantity_name: str = "",
        notes: str = "",
    ) -> UncertaintyBudget:
        """Run the full ASME V&V 20 validation for one quantity.

        Returns an :class:`UncertaintyBudget` with ``u_val``, the comparison
        error and the ``error_within_uncertainty`` flag.
        """
        if not math.isfinite(computed_value) or not math.isfinite(experimental_value):
            raise ValueError("computed_value and experimental_value must be finite")
        for name, value in (("u_num", u_num), ("u_input", u_input), ("u_model", u_model)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")

        # Total validation uncertainty (RSS).
        u_val = validation_uncertainty(
            math.sqrt(u_num * u_num + u_model * u_model), u_input
        )

        # Comparison error (percent).
        if abs(experimental_value) < 1e-14:
            error = float("inf")
        else:
            error = 100.0 * abs(computed_value - experimental_value) / abs(experimental_value)

        # Validation assessment.
        error_within = error <= u_val

        return UncertaintyBudget(
            u_num=u_num,
            u_input=u_input,
            u_model=u_model,
            u_val=u_val,
            experimental_value=experimental_value,
            computed_value=computed_value,
            error=error,
            error_within_uncertainty=error_within,
            quantity_name=quantity_name,
            notes=notes,
        )
