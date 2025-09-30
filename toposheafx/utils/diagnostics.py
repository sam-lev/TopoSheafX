# toposheafx/utils/diagnostics.py
"""
Diagnostic utilities for analyzing topological neural networks.
"""

import torch
from typing import Dict, List, Any, Optional
import warnings


def check_oversmoothing_condition(
        laplacians: Dict[int, torch.Tensor],
        time_params: Dict[str, float],
        weight_norms: Optional[Dict[str, float]] = None,
        warn: bool = True
) -> Dict[str, Any]:
    """
    Check over-smoothing condition from COSIMO Proposition 5.5:
    t < ln(s) / (λ_min(L_k) + κ_f(L_k))

    Args:
        laplacians: Dict mapping simplex dimension to Laplacian matrices
        time_params: Dict with 't_d' and 't_u' time parameters
        weight_norms: Optional dict of weight norms, otherwise computed as 1.0
        warn: Whether to emit warnings for violations

    Returns:
        Dictionary with diagnostics for each simplex dimension
    """
    diagnostics = {
        'satisfies_condition': True,
        'dimensions': []
    }

    for k, L_k in laplacians.items():
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(L_k)
        lambda_min = eigenvalues[eigenvalues > 1e-6].min()
        lambda_max = eigenvalues.max()
        kappa_f = lambda_max / lambda_min  # Condition number

        # Weight norm (use provided or default to 1.0)
        s_norm = weight_norms.get(f'dim_{k}', 1.0) if weight_norms else 1.0

        # Maximum safe time parameter
        max_t = (torch.log(torch.tensor(s_norm)) / (lambda_min + kappa_f)).item()

        # Check if current time params satisfy condition
        current_t = max(time_params.get('t_d', 0), time_params.get('t_u', 0))
        satisfies = current_t < max_t

        dim_info = {
            'simplex_dim': k,
            'lambda_min': lambda_min.item(),
            'lambda_max': lambda_max.item(),
            'condition_number': kappa_f.item(),
            'max_safe_t': max_t,
            'current_t': current_t,
            'satisfies_condition': satisfies,
            'weight_norm': s_norm
        }

        diagnostics['dimensions'].append(dim_info)

        if not satisfies:
            diagnostics['satisfies_condition'] = False
            if warn:
                warnings.warn(
                    f"Over-smoothing risk at dimension {k}: "
                    f"t={current_t:.4f} exceeds safe threshold {max_t:.4f}. "
                    f"Consider reducing time parameters or network depth."
                )

    return diagnostics


def compute_dirichlet_energy(
        features: torch.Tensor,
        laplacian: torch.Tensor
) -> float:
    """
    Compute Dirichlet energy E(x) = x^T L x (Equation 1 from COSIMO).

    Lower energy indicates smoother signals.

    Args:
        features: Feature matrix [num_simplices, feature_dim]
        laplacian: Laplacian matrix [num_simplices, num_simplices]

    Returns:
        Dirichlet energy scalar
    """
    energy = torch.trace(features.T @ laplacian @ features)
    return energy.item()


def analyze_smoothing_rate(
        model,
        features: torch.Tensor,
        laplacians: Dict[int, torch.Tensor],
        num_layers: int = 10
) -> Dict[str, List[float]]:
    """
    Analyze how quickly features smooth across layers.

    Useful for validating Theorems 5.1 and 5.3.

    Args:
        model: Neural network model
        features: Initial features
        laplacians: Laplacian matrices per dimension
        num_layers: Number of layers to simulate

    Returns:
        Dict with energy evolution per dimension
    """
    from torch_geometric.utils import scatter

    energy_evolution = {k: [] for k in laplacians.keys()}

    x = features
    for layer_idx in range(num_layers):
        # Compute energy at each dimension
        for k, L_k in laplacians.items():
            if x.shape[0] == L_k.shape[0]:
                energy = compute_dirichlet_energy(x, L_k)
                energy_evolution[k].append(energy)

        # Forward pass (assuming model has single forward method)
        # This is simplified - adjust based on actual model interface
        try:
            x = model.layers[layer_idx % len(model.layers)](x)
        except:
            break

    return energy_evolution


def extract_model_weight_norms(model) -> Dict[str, float]:
    """
    Extract weight norms from model for over-smoothing analysis.

    Args:
        model: PyTorch model

    Returns:
        Dict mapping layer names to weight norms
    """
    weight_norms = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            norm = torch.norm(module.weight).item()
            weight_norms[name] = norm

    # Also compute global statistics
    all_norms = list(weight_norms.values())
    if all_norms:
        weight_norms['max'] = max(all_norms)
        weight_norms['mean'] = sum(all_norms) / len(all_norms)

    return weight_norms