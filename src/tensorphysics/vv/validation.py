"""
ASME V&V 20 validation uncertainty quantification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ValidationResult:
    """Result of a validation assessment."""
    label: str
    f_sim: float
    f_exp: float
    u_num: float
    u_input: float
    u_val: float
    error: float
    relative_error: float
    ratio: float
    status: str
    passed: bool


def validation_uncertainty(u_num: float, u_input: float, u_model: float = 0.0) -> float:
    """
    Compute total validation uncertainty (RSS combination).
    
    U_val = sqrt(U_num^2 + U_input^2 + U_model^2)
    
    Parameters
    ----------
    u_num : float
        Numerical uncertainty
    u_input : float
        Input parameter uncertainty
    u_model : float
        Model-form uncertainty (default 0)
    
    Returns
    -------
    float
        Total validation uncertainty
    """
    if u_num < 0 or u_input < 0 or u_model < 0:
        raise ValueError("Uncertainties must be non-negative")
    if math.isnan(u_num) or math.isnan(u_input) or math.isnan(u_model):
        raise ValueError("Uncertainties must not be NaN")
    
    return math.sqrt(u_num**2 + u_input**2 + u_model**2)


def validate(
    f_sim: float,
    f_exp: float,
    u_num: float,
    u_input: float,
    u_model: float = 0.0,
    label: str = "",
) -> ValidationResult:
    """
    Validate a simulation result against experimental/reference data.
    
    Parameters
    ----------
    f_sim : float
        Simulated/computed value
    f_exp : float
        Experimental/reference value
    u_num : float
        Numerical uncertainty
    u_input : float
        Input parameter uncertainty
    u_model : float
        Model-form uncertainty (default 0)
    label : str
        Label for this validation case
    
    Returns
    -------
    ValidationResult
        Complete validation assessment
    """
    u_val = validation_uncertainty(u_num, u_input, u_model)
    error = f_sim - f_exp
    
    if abs(f_exp) < 1e-14:
        relative_error = error
    else:
        relative_error = error / f_exp
    
    abs_error = abs(error)
    
    if u_val < 1e-14:
        # Zero uncertainty band
        if abs_error < 1e-14:
            ratio = 0.0
            passed = True
        else:
            ratio = float('inf')
            passed = False
    else:
        ratio = abs_error / u_val
        passed = abs_error <= u_val
    
    status = "PASS" if passed else "FAIL"
    
    return ValidationResult(
        label=label,
        f_sim=f_sim,
        f_exp=f_exp,
        u_num=u_num,
        u_input=u_input,
        u_val=u_val,
        error=error,
        relative_error=relative_error,
        ratio=ratio,
        status=status,
        passed=passed,
    )


def evaluate_validation_table(rows: List[Dict[str, Any]]) -> List[ValidationResult]:
    """
    Evaluate a table of validation cases.
    
    Parameters
    ----------
    rows : list of dict
        Each dict must have keys: label, f_sim, f_exp, u_num, u_input
    
    Returns
    -------
    list of ValidationResult
    """
    results = []
    
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Row must be a dict, got {type(row)}")
        
        required_keys = {'label', 'f_sim', 'f_exp', 'u_num', 'u_input'}
        if not required_keys.issubset(row.keys()):
            missing = required_keys - set(row.keys())
            raise ValueError(f"Row missing required keys: {missing}")
        
        result = validate(
            f_sim=row['f_sim'],
            f_exp=row['f_exp'],
            u_num=row['u_num'],
            u_input=row['u_input'],
            label=row['label'],
        )
        results.append(result)
    
    return results
