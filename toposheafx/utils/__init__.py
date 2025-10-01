# toposheafx/utils/__init__.py
"""
Utility functions for toposheafx.
"""

from .diagnostics import (
    check_oversmoothing_condition,
    compute_dirichlet_energy,
    analyze_smoothing_rate,
    extract_model_weight_norms,
    extract_dimension_specific_weight_norms,
    validate_time_parameters,
)

from .simplicial import (
    compute_hodge_laplacians,
    compute_laplacians,  # Alias for backward compatibility
    compute_lower_laplacians,
    compute_upper_laplacians,
    compute_adjacency_matrices,
    validate_boundary_operators,
    compute_betti_numbers,
    adjust_model_time_parameters,
)

__all__ = [
    # Diagnostics
    'check_oversmoothing_condition',
    'compute_dirichlet_energy',
    'analyze_smoothing_rate',
    'extract_model_weight_norms',
    'extract_dimension_specific_weight_norms',
    'validate_time_parameters',

    # Simplicial operations
    'compute_hodge_laplacians',
    'compute_laplacians',
    'compute_lower_laplacians',
    'compute_upper_laplacians',
    'compute_adjacency_matrices',
    'validate_boundary_operators',
    'compute_betti_numbers',
    'adjust_model_time_parameters',
]