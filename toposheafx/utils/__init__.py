# toposheafx/utils/__init__.py
"""
Utility functions for toposheafx.
"""

from .diagnostics import (
    analyze_smoothing_rate,
    check_oversmoothing_condition,
    compute_dirichlet_energy,
    extract_dimension_specific_weight_norms,
    extract_model_weight_norms,
    validate_time_parameters,
)
from .simplicial import (
    adjust_model_time_parameters,
    compute_adjacency_matrices,
    compute_betti_numbers,
    compute_hodge_laplacians,
    compute_laplacians,  # Alias for backward compatibility
    compute_lower_laplacians,
    compute_upper_laplacians,
    validate_boundary_operators,
)

__all__ = [
    # Diagnostics
    "check_oversmoothing_condition",
    "compute_dirichlet_energy",
    "analyze_smoothing_rate",
    "extract_model_weight_norms",
    "extract_dimension_specific_weight_norms",
    "validate_time_parameters",
    # Simplicial operations
    "compute_hodge_laplacians",
    "compute_laplacians",
    "compute_lower_laplacians",
    "compute_upper_laplacians",
    "compute_adjacency_matrices",
    "validate_boundary_operators",
    "compute_betti_numbers",
    "adjust_model_time_parameters",
]
