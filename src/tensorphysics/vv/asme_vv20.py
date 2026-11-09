"""
ASME V&V 20 Validation Uncertainty Quantification.

Implements the verification and validation uncertainty framework per
ASME V&V 20 standard. Total validation uncertainty:
    U_val = sqrt(U_num^2 + U_input^2 + U_model^2)

where:
    U_num   = numerical uncertainty (from GCI/Richardson)
    U_input = input parameter uncertainty
    U_model = model-form uncertainty (comparison with experimental data)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class UncertaintyBudget:
    """Complete uncertainty budget per ASME V&V 20."""
    # Component uncertainties (as percentages of the quantity)
    u_num: float          # Numerical uncertainty (%)
    u_input: float        # Input parameter uncertainty (%)
    u_model: float        # Model-form uncertainty (%)
    
    # Total validation uncertainty
    u_val: float          # Total validation uncertainty (%)
    
    # Comparison data
    experimental_value: Optional[float] = None
    computed_value: Optional[float] = None
    error: Optional[float] = None       # |computed - experimental| / |experimental| * 100
    error_within_uncertainty: Optional[bool] = None
    
    # Metadata
    quantity_name: str = ""
    notes: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            'u_num': self.u_num,
            'u_input': self.u_input,
            'u_model': self.u_model,
            'u_val': self.u_val,
            'quantity_name': self.quantity_name,
            'notes': self.notes,
        }
        if self.experimental_value is not None:
            d['experimental_value'] = self.experimental_value
        if self.computed_value is not None:
            d['computed_value'] = self.computed_value
        if self.error is not None:
            d['error'] = self.error
        if self.error_within_uncertainty is not None:
            d['error_within_uncertainty'] = self.error_within_uncertainty
        return d
    
    def is_validated(self) -> bool:
        """Check if the computation is validated (error within uncertainty)."""
        if self.error_within_uncertainty is None:
            return False
        return self.error_within_uncertainty


class ASMEVV20:
    """
    ASME V&V 20 validation uncertainty quantification engine.
    
    Computes total validation uncertainty U_val from component uncertainties
    and compares with experimental data for validation assessment.
    
    Parameters
    ----------
    max_acceptable_uncertainty : float
        Maximum acceptable total validation uncertainty (%). Default 10%.
    """
    
    def __init__(self, max_acceptable_uncertainty: float = 10.0):
        self.max_acceptable_uncertainty = max_acceptable_uncertainty
    
    def compute_numerical_uncertainty(
        self,
        gci_fine: float,
        f_ext: Optional[float] = None,
        f_fine: Optional[float] = None,
    ) -> float:
        """
        Compute numerical uncertainty from GCI.
        
        Parameters
        ----------
        gci_fine : float
            Grid Convergence Index for fine grid (%)
        f_ext : float, optional
            Extrapolated value (for alternative calculation)
        f_fine : float, optional
            Fine grid solution (for alternative calculation)
            
        Returns
        -------
        float
            Numerical uncertainty (%)
        """
        # Primary: use GCI directly
        u_num = gci_fine
        
        # Alternative: if extrapolated value available, use it
        if f_ext is not None and f_fine is not None and abs(f_fine) > 1e-14:
            u_alt = 100.0 * abs(f_fine - f_ext) / abs(f_fine)
            # Use the larger of the two for conservatism
            u_num = max(u_num, u_alt)
        
        return u_num
    
    def compute_input_uncertainty(
        self,
        input_params: Dict[str, dict],
        quantity_sensitivities: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute input parameter uncertainty via error propagation.
        
        Parameters
        ----------
        input_params : dict
            Dictionary of input parameters with their uncertainties.
            Format: {'param_name': {'value': float, 'uncertainty': float}}
            where uncertainty is absolute or relative (%).
        quantity_sensitivities : dict, optional
            Sensitivity of output to each input: dQ/dparam.
            If not provided, assumes unit sensitivity.
            
        Returns
        -------
        float
            Input uncertainty (%)
        """
        if not input_params:
            return 0.0
        
        u_input_sq = 0.0
        
        for param_name, param_data in input_params.items():
            value = param_data.get('value', 0.0)
            uncertainty = param_data.get('uncertainty', 0.0)
            
            # Determine if uncertainty is absolute or relative
            if param_data.get('is_relative', False):
                # Already in percentage
                u_param = uncertainty
            else:
                # Absolute uncertainty -> convert to relative
                if abs(value) > 1e-14:
                    u_param = 100.0 * abs(uncertainty) / abs(value)
                else:
                    u_param = 100.0 * abs(uncertainty)
            
            # Apply sensitivity if available
            sensitivity = 1.0
            if quantity_sensitivities and param_name in quantity_sensitivities:
                sensitivity = abs(quantity_sensitivities[param_name])
            
            u_input_sq += (sensitivity * u_param) ** 2
        
        return np.sqrt(u_input_sq)
    
    def compute_model_uncertainty(
        self,
        computed_value: float,
        experimental_value: float,
        u_num: float,
        u_input: float,
    ) -> float:
        """
        Compute model-form uncertainty from comparison with experimental data.
        
        The model uncertainty is the residual after accounting for numerical
        and input uncertainties:
            delta = |computed - experimental| / |experimental| * 100
            U_model = sqrt(max(0, delta^2 - U_num^2 - U_input^2))
        
        Parameters
        ----------
        computed_value : float
            Computed/simulated value
        experimental_value : float
            Experimental/reference value
        u_num : float
            Numerical uncertainty (%)
        u_input : float
            Input uncertainty (%)
            
        Returns
        -------
        float
            Model-form uncertainty (%)
        """
        if abs(experimental_value) < 1e-14:
            # Avoid division by zero
            delta = abs(computed_value - experimental_value)
        else:
            delta = 100.0 * abs(computed_value - experimental_value) / abs(experimental_value)
        
        # Model uncertainty is the residual
        residual_sq = delta**2 - u_num**2 - u_input**2
        if residual_sq > 0:
            u_model = np.sqrt(residual_sq)
        else:
            # Error is fully explained by numerical + input uncertainty
            u_model = 0.0
        
        return u_model
    
    def compute_total_uncertainty(
        self,
        u_num: float,
        u_input: float,
        u_model: float,
    ) -> float:
        """
        Compute total validation uncertainty (RSS combination).
        
        U_val = sqrt(U_num^2 + U_input^2 + U_model^2)
        
        Parameters
        ----------
        u_num : float
            Numerical uncertainty (%)
        u_input : float
            Input parameter uncertainty (%)
        u_model : float
            Model-form uncertainty (%)
            
        Returns
        -------
        float
            Total validation uncertainty (%)
        """
        return np.sqrt(u_num**2 + u_input**2 + u_model**2)
    
    def validate(
        self,
        computed_value: float,
        experimental_value: float,
        u_num: float,
        u_input: float,
        u_model: Optional[float] = None,
        quantity_name: str = "",
    ) -> UncertaintyBudget:
        """
        Perform complete validation assessment.
        
        Parameters
        ----------
        computed_value : float
            Computed/simulated value
        experimental_value : float
            Experimental/reference value
        u_num : float
            Numerical uncertainty (%)
        u_input : float
            Input parameter uncertainty (%)
        u_model : float, optional
            Model-form uncertainty (%). If None, computed from residual.
        quantity_name : str
            Name of the quantity being validated
            
        Returns
        -------
        UncertaintyBudget
            Complete uncertainty budget with validation assessment
        """
        # Compute error
        if abs(experimental_value) > 1e-14:
            error = 100.0 * abs(computed_value - experimental_value) / abs(experimental_value)
        else:
            error = abs(computed_value - experimental_value)
        
        # Compute model uncertainty if not provided
        if u_model is None:
            u_model = self.compute_model_uncertainty(
                computed_value, experimental_value, u_num, u_input
            )
        
        # Total uncertainty
        u_val = self.compute_total_uncertainty(u_num, u_input, u_model)
        
        # Check if error is within uncertainty band
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
            notes=f"Error {error:.2f}% {'within' if error_within else 'exceeds'} uncertainty band {u_val:.2f}%",
        )
    
    def assess_validation(
        self,
        budgets: List[UncertaintyBudget],
    ) -> dict:
        """
        Assess overall validation status from multiple quantity budgets.
        
        Parameters
        ----------
        budgets : list of UncertaintyBudget
            Uncertainty budgets for multiple quantities
            
        Returns
        -------
        dict
            Overall validation assessment
        """
        if not budgets:
            return {'status': 'NO_DATA', 'message': 'No uncertainty budgets provided'}
        
        n_validated = sum(1 for b in budgets if b.is_validated())
        n_total = len(budgets)
        
        # Check if all uncertainties are acceptable
        max_u_val = max(b.u_val for b in budgets)
        all_acceptable = max_u_val <= self.max_acceptable_uncertainty
        
        # Overall status
        if n_validated == n_total and all_acceptable:
            status = 'VALIDATED'
        elif n_validated > 0:
            status = 'PARTIALLY_VALIDATED'
        else:
            status = 'NOT_VALIDATED'
        
        return {
            'status': status,
            'n_validated': n_validated,
            'n_total': n_total,
            'max_u_val': max_u_val,
            'max_acceptable': self.max_acceptable_uncertainty,
            'all_acceptable': all_acceptable,
            'details': [b.to_dict() for b in budgets],
        }
