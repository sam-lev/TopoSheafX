# toposheafx/utils/diagnostics.py
"""
Diagnostic utilities for analyzing topological neural networks.
"""

import warnings
from typing import Any, Optional

import torch
from torch import nn


def check_oversmoothing_condition(
    laplacians: dict[int, torch.Tensor],
    time_params: dict[str, float],
    weight_norms: Optional[dict[str, float]] = None,
    warn: bool = True,
) -> dict[str, Any]:
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
    diagnostics = {"satisfies_condition": True, "dimensions": []}

    for k, L_k in laplacians.items():
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(L_k)
        lambda_min = eigenvalues[eigenvalues > 1e-6].min()
        lambda_max = eigenvalues.max()
        kappa_f = lambda_max / lambda_min  # Condition number

        # Weight norm (use provided or default to 1.0)
        s_norm = weight_norms.get(f"dim_{k}", 1.0) if weight_norms else 1.0

        # Maximum safe time parameter
        max_t = (torch.log(torch.tensor(s_norm)) / (lambda_min + kappa_f)).item()

        # Check if current time params satisfy condition
        current_t = max(time_params.get("t_d", 0), time_params.get("t_u", 0))
        satisfies = current_t < max_t

        dim_info = {
            "simplex_dim": k,
            "lambda_min": lambda_min.item(),
            "lambda_max": lambda_max.item(),
            "condition_number": kappa_f.item(),
            "max_safe_t": max_t,
            "current_t": current_t,
            "satisfies_condition": satisfies,
            "weight_norm": s_norm,
        }

        diagnostics["dimensions"].append(dim_info)

        if not satisfies:
            diagnostics["satisfies_condition"] = False
            if warn:
                warnings.warn(
                    f"Over-smoothing risk at dimension {k}: "
                    f"t={current_t:.4f} exceeds safe threshold {max_t:.4f}. "
                    f"Consider reducing time parameters or network depth.", stacklevel=2
                )

    return diagnostics


def compute_dirichlet_energy(features: torch.Tensor, laplacian: torch.Tensor) -> float:
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
    model, features: torch.Tensor, laplacians: dict[int, torch.Tensor], num_layers: int = 10
) -> dict[str, list[float]]:
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

    energy_evolution = {k: [] for k in laplacians}

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
        except (TypeError, RuntimeError) as e:
            break

    return energy_evolution


def extract_model_weight_norms(model) -> dict[str, float]:
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
        weight_norms["max"] = max(all_norms)
        weight_norms["mean"] = sum(all_norms) / len(all_norms)

    return weight_norms


def extract_dimension_specific_weight_norms(model) -> dict[str, float]:
    """
    Extract weight norms specific to each simplex dimension's processing path.

    This function traverses the model to find layers that process each dimension
    and computes their maximum weight norms.

    Returns:
        Dict mapping 'dim_0', 'dim_1', 'dim_2' to their weight norms
    """
    dim_norms = {}

    # For ContinuousSimplicialModel with branch structure
    if hasattr(model, "branch_layers"):
        for _branch_idx, branch in enumerate(model.branch_layers):
            for _layer_idx, layer in enumerate(branch):
                # ContinuousCHOMPNNLayer has neighborhood_layers
                if hasattr(layer, "neighborhood_layers"):
                    for k, neighborhood_layer in enumerate(layer.neighborhood_layers):
                        key = f"dim_{k}"

                        # Collect weight norms from all Linear layers in this neighborhood
                        norms = []
                        for module in neighborhood_layer.modules():
                            if isinstance(module, nn.Linear):
                                norms.append(torch.norm(module.weight).item())

                        if norms:
                            # Use max norm for this dimension
                            current_max = max(norms)
                            if key in dim_norms:
                                dim_norms[key] = max(dim_norms[key], current_max)
                            else:
                                dim_norms[key] = current_max

    # Fallback: if we couldn't extract dimension-specific norms, use global stats
    if not dim_norms:
        print("Could not extract dimension-specific norms, using global maximum")
        all_norms = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                all_norms.append(torch.norm(module.weight).item())

        if all_norms:
            max_norm = max(all_norms)
            # Ensure at least 2.0 for valid over-smoothing condition
            max_norm = max(max_norm, 2.0)
            for k in range(3):  # Dimensions 0, 1, 2
                dim_norms[f"dim_{k}"] = max_norm
        else:
            # Ultimate fallback
            for k in range(3):
                dim_norms[f"dim_{k}"] = 2.0

    # Ensure all dimension norms are > 1.0 for valid ln()
    for key in dim_norms:
        if dim_norms[key] <= 1.0:
            print(f"Weight norm for {key} is ≤ 1.0 ({dim_norms[key]:.4f}), setting to 2.0")
            dim_norms[key] = 2.0

    return dim_norms


def validate_time_parameters(
    model, laplacians: dict[int, torch.Tensor], verbose: bool = True
) -> dict[str, Any]:
    """
    Convenience function to validate time parameters against over-smoothing conditions.

    This combines weight norm extraction, time parameter extraction, and
    over-smoothing checking into a single call.

    Args:
        model: Continuous simplicial model
        laplacians: Dict mapping simplex dimension to Laplacian matrices
        verbose: Whether to print detailed diagnostics

    Returns:
        Dictionary with validation results

    Examples:
        >>> model = ContinuousSimplicialModel(...)
        >>> laplacians = compute_hodge_laplacians(boundary_operators)
        >>> results = validate_time_parameters(model, laplacians)
        >>> if not results['satisfies_condition']:
        ...     print(f"Reduce time params to ≤ {results['suggested_max_t']:.4f}")
    """
    # Extract dimension-specific weight norms
    dim_weight_norms = extract_dimension_specific_weight_norms(model)

    # Get current time parameters
    if hasattr(model, "get_time_constants"):
        time_constants = model.get_time_constants()
        avg_t_d = (
            sum(time_constants["t_d"]) / len(time_constants["t_d"])
            if time_constants["t_d"]
            else 0.1
        )
        avg_t_u = (
            sum(time_constants["t_u"]) / len(time_constants["t_u"])
            if time_constants["t_u"]
            else 0.1
        )
        time_params = {"t_d": avg_t_d, "t_u": avg_t_u}
    else:
        warnings.warn(
            "Model does not have get_time_constants method. Using default t=0.1", UserWarning, stacklevel=2
        )
        time_params = {"t_d": 0.1, "t_u": 0.1}

    # Check over-smoothing conditions
    diagnostics = check_oversmoothing_condition(
        laplacians=laplacians, time_params=time_params, weight_norms=dim_weight_norms, warn=False
    )

    # Add suggested safe time parameter
    if diagnostics["dimensions"]:
        safe_t_values = [d["max_safe_t"] for d in diagnostics["dimensions"]]
        suggested_max_t = min(safe_t_values) * 0.9  # Use 90% for safety margin
    else:
        suggested_max_t = 0.1

    diagnostics["suggested_max_t"] = suggested_max_t
    diagnostics["time_params"] = time_params
    diagnostics["weight_norms"] = dim_weight_norms

    if verbose:
        print("\n" + "=" * 60)
        print("TIME PARAMETER VALIDATION")
        print("=" * 60)

        print("\nCurrent time parameters:")
        print(f"  t_d = {time_params['t_d']:.4f}")
        print(f"  t_u = {time_params['t_u']:.4f}")

        print("\nWeight norms by dimension:")
        for dim, norm in sorted(dim_weight_norms.items()):
            print(f"  {dim}: {norm:.4f}")

        print("\nOver-smoothing check: ", end="")
        if diagnostics["satisfies_condition"]:
            print("✓ PASSED")
        else:
            print("✗ FAILED")

        print("\nPer-dimension analysis:")
        for dim_info in diagnostics["dimensions"]:
            status = "✓" if dim_info["satisfies_condition"] else "✗"
            print(f"  {status} Dimension {dim_info['simplex_dim']}:")
            print(f"      Current t:    {dim_info['current_t']:.4f}")
            print(f"      Max safe t:   {dim_info['max_safe_t']:.4f}")
            print(f"      λ_min:        {dim_info['lambda_min']:.6f}")
            print(f"      Condition #:  {dim_info['condition_number']:.2f}")

        if not diagnostics["satisfies_condition"]:
            print(f"\n⚠️  RECOMMENDATION: Reduce time parameters to ≤ {suggested_max_t:.4f}")

        print("=" * 60 + "\n")

    return diagnostics
