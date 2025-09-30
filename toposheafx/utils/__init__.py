# toposheafx/utils/__init__.py
from .guards import _check_index, _same_device
from .diagnostics import (
    check_oversmoothing_condition,
    compute_dirichlet_energy,
    analyze_smoothing_rate,
    extract_model_weight_norms
)

__all__ = [
    '_check_index',
    '_same_device',
    'check_oversmoothing_condition',
    'compute_dirichlet_energy',
    'analyze_smoothing_rate',
    'extract_model_weight_norms'
]