import numpy as np
from typing import Dict, Any
import warnings

class FunctionalForms:
    @staticmethod
    def ces(price_ratio: float, initial_value: float, elasticity: float) -> float:
        """Constant Elasticity of Substitution form."""
        return initial_value * (price_ratio ** elasticity)
    
    @staticmethod
    def log_linear(price_ratio: float, initial_value: float, elasticity: float) -> float:
        """
        Log-linear (Cobb-Douglas) form.
        Correctly represents elasticities for all price changes.
        
        Args:
            price_ratio: Current price / Initial price
            initial_value: Initial quantity
            elasticity: Elasticity of response
            
        Returns:
            New quantity after price change
        """
        return initial_value * (price_ratio ** elasticity)
    
    @staticmethod
    def linear(price_ratio: float, initial_value: float, elasticity: float) -> float:
        """
        Linear form.
        WARNING: Only accurate for small price changes (less than 10%).
        For larger changes, use log_linear form instead.
        
        Args:
            price_ratio: Current price / Initial price
            initial_value: Initial quantity
            elasticity: Elasticity of response
            
        Returns:
            New quantity after price change
        """
        warnings.warn("Linear form is only accurate for small price changes (<10%). "
                     "For larger changes, use log_linear form instead.",
                     RuntimeWarning)
        return initial_value * (1 + elasticity * (price_ratio - 1))
    
    @staticmethod
    def translog(price_ratio: float, initial_value: float, params: Dict[str, float]) -> float:
        """
        Translog form with quadratic terms.
        WARNING: Only accurate for small price changes (less than 10%).
        For larger changes, use log_linear form instead.
        
        Args:
            price_ratio: Current price / Initial price
            initial_value: Initial quantity
            params: Dictionary containing 'elasticity' and 'quadratic_term'
            
        Returns:
            New quantity after price change
        """
        warnings.warn("Translog form is only accurate for small price changes (<10%). "
                     "For larger changes, use log_linear form instead.",
                     RuntimeWarning)
        elasticity = params['elasticity']
        quadratic_term = params['quadratic_term']
        return initial_value * np.exp(elasticity * np.log(price_ratio) + 
                                    0.5 * quadratic_term * np.log(price_ratio)**2)
    
    @staticmethod
    def quadratic(price_ratio: float, initial_value: float, params: Dict[str, float]) -> float:
        """
        Quadratic form.
        WARNING: Only accurate for small price changes (less than 10%).
        For larger changes, use log_linear form instead.
        
        Args:
            price_ratio: Current price / Initial price
            initial_value: Initial quantity
            params: Dictionary containing 'elasticity' and 'quadratic_term'
            
        Returns:
            New quantity after price change
        """
        warnings.warn("Quadratic form is only accurate for small price changes (<10%). "
                     "For larger changes, use log_linear form instead.",
                     RuntimeWarning)
        elasticity = params['elasticity']
        quadratic_term = params['quadratic_term']
        return initial_value * (1 + elasticity * (price_ratio - 1) + 
                              0.5 * quadratic_term * (price_ratio - 1)**2)
    
    @staticmethod
    def piecewise(price_ratio: float, initial_value: float, params: Dict[str, float]) -> float:
        """
        Piecewise linear form with different elasticities above/below threshold.
        WARNING: Only accurate for small price changes (less than 10%).
        For larger changes, use log_linear form instead.
        
        Args:
            price_ratio: Current price / Initial price
            initial_value: Initial quantity
            params: Dictionary containing 'elasticity_low', 'elasticity_high', and 'price_threshold'
            
        Returns:
            New quantity after price change
        """
        warnings.warn("Piecewise linear form is only accurate for small price changes (<10%). "
                     "For larger changes, use log_linear form instead.",
                     RuntimeWarning)
        threshold = params['price_threshold']
        elasticity_low = params['elasticity_low']
        elasticity_high = params['elasticity_high']
        
        if price_ratio <= threshold:
            return initial_value * (1 + elasticity_low * (price_ratio - 1))
        else:
            return initial_value * (1 + elasticity_low * (threshold - 1) + 
                                  elasticity_high * (price_ratio - threshold))
    
    @classmethod
    def get_function(cls, form: str) -> callable:
        """
        Get the appropriate function based on the form name.
        Defaults to log_linear if form not found.
        """
        functions = {
            'ces': cls.ces,
            'log_linear': cls.log_linear,
            'linear': cls.linear,
            'translog': cls.translog,
            'quadratic': cls.quadratic,
            'piecewise': cls.piecewise
        }
        if form not in functions:
            warnings.warn(f"Unknown functional form '{form}'. Using log_linear instead.",
                         RuntimeWarning)
        return functions.get(form, cls.log_linear) 