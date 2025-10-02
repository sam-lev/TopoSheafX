# toposheafx/utils/simplicial.py
"""
Utilities for working with simplicial complexes.

This module provides functions for computing topological operators
like Hodge Laplacians, boundary operators, and adjacency matrices.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


def compute_hodge_laplacians(
        boundary_operators: Dict[int, torch.Tensor],
        normalize: bool = False
) -> Dict[int, torch.Tensor]:
    """
    Compute Hodge Laplacians for each dimension of a simplicial complex.

    The Hodge Laplacian for k-simplices is defined as:
        L_k = L_k_down + L_k_up
    where:
        L_k_down = B_k^T @ B_k (lower Laplacian)
        L_k_up = B_{k+1} @ B_{k+1}^T (upper Laplacian)

    For dimension 0 (vertices): L_0 = B_1 @ B_1^T (graph Laplacian)
    For highest dimension: L_max = B_max^T @ B_max

    Args:
        boundary_operators: Dict mapping dimension k to boundary operator B_k
                           B_k maps k-simplices to (k-1)-simplices
                           Shape: [num_(k-1)_simplices, num_k_simplices]
        normalize: Whether to apply symmetric normalization (default: False)

    Returns:
        Dict mapping dimension k to Hodge Laplacian L_k

    Examples:
        >>> boundary_ops = {1: B1, 2: B2}  # B1: edges->vertices, B2: triangles->edges
        >>> laplacians = compute_hodge_laplacians(boundary_ops)
        >>> L_0 = laplacians[0]  # Graph Laplacian (vertex-level)
        >>> L_1 = laplacians[1]  # Edge Hodge Laplacian
        >>> L_2 = laplacians[2]  # Triangle Laplacian

    References:
        - Schaub et al. "Random Walks on Simplicial Complexes" (2020)
        - Ebli et al. "Simplicial Neural Networks" (2020)
    """
    if not boundary_operators:
        raise ValueError("boundary_operators dict cannot be empty")

    laplacians = {}
    dimensions = sorted(boundary_operators.keys())

    # Dimension 0 (vertices): L_0 = B_1 @ B_1^T (graph Laplacian)
    if 1 in boundary_operators:
        B_1 = boundary_operators[1]
        L_0 = B_1 @ B_1.T

        if normalize:
            L_0 = _symmetric_normalize_laplacian(L_0)

        laplacians[0] = L_0

    # Intermediate dimensions: L_k = B_k^T @ B_k + B_{k+1} @ B_{k+1}^T
    for i, k in enumerate(dimensions):
        B_k = boundary_operators[k]

        # Lower Laplacian: B_k^T @ B_k
        L_k_down = B_k.T @ B_k

        # Upper Laplacian: B_{k+1} @ B_{k+1}^T (if exists)
        if i + 1 < len(dimensions):
            k_plus_1 = dimensions[i + 1]
            B_k_plus_1 = boundary_operators[k_plus_1]
            L_k_up = B_k_plus_1 @ B_k_plus_1.T
        else:
            # Highest dimension: no upper Laplacian
            L_k_up = torch.zeros_like(L_k_down)

        # Combined Hodge Laplacian
        L_k = L_k_down + L_k_up

        if normalize:
            L_k = _symmetric_normalize_laplacian(L_k)

        laplacians[k] = L_k

    return laplacians


def compute_lower_laplacians(
        boundary_operators: Dict[int, torch.Tensor]
) -> Dict[int, torch.Tensor]:
    """
    Compute lower Hodge Laplacians: L_k_down = B_k^T @ B_k

    The lower Laplacian captures connections between k-simplices
    through their common (k-1)-dimensional faces.

    Args:
        boundary_operators: Dict mapping dimension k to boundary operator B_k

    Returns:
        Dict mapping dimension k to lower Laplacian L_k_down

    Examples:
        >>> lower_laplacians = compute_lower_laplacians(boundary_ops)
        >>> L_1_down = lower_laplacians[1]  # Edge-edge adjacency through shared vertices
    """
    lower_laplacians = {}

    for k, B_k in boundary_operators.items():
        lower_laplacians[k] = B_k.T @ B_k

    return lower_laplacians


def compute_upper_laplacians(
        boundary_operators: Dict[int, torch.Tensor]
) -> Dict[int, torch.Tensor]:
    """
    Compute upper Hodge Laplacians: L_k_up = B_{k+1} @ B_{k+1}^T

    The upper Laplacian captures connections between k-simplices
    through their common (k+1)-dimensional cofaces.

    Args:
        boundary_operators: Dict mapping dimension k to boundary operator B_k

    Returns:
        Dict mapping dimension k to upper Laplacian L_k_up

    Examples:
        >>> upper_laplacians = compute_upper_laplacians(boundary_ops)
        >>> L_1_up = upper_laplacians[1]  # Edge-edge adjacency through shared triangles
    """
    upper_laplacians = {}
    dimensions = sorted(boundary_operators.keys())

    for i, k in enumerate(dimensions[:-1]):
        k_plus_1 = dimensions[i + 1]
        B_k_plus_1 = boundary_operators[k_plus_1]
        upper_laplacians[k] = B_k_plus_1 @ B_k_plus_1.T

    return upper_laplacians


def _symmetric_normalize_laplacian(L: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Apply symmetric normalization to a Laplacian matrix.

    Normalized Laplacian: L_norm = D^{-1/2} @ L @ D^{-1/2}
    where D is the degree matrix (diagonal of L).

    Args:
        L: Laplacian matrix [n, n]
        eps: Small constant for numerical stability

    Returns:
        Normalized Laplacian
    """
    # Extract diagonal (degrees)
    deg = torch.diag(L)

    # Compute D^{-1/2}
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Create diagonal matrix
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    # Symmetric normalization
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    return L_norm


def compute_adjacency_matrices(
        boundary_operators: Dict[int, torch.Tensor],
        include_self_loops: bool = False
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Compute adjacency matrices between different simplex dimensions.

    This computes various types of adjacencies:
    - (k, k): k-simplices adjacent through shared (k-1)-faces (lower adjacency)
    - (k, k'): k-simplices adjacent to k'-simplices through incidence

    Args:
        boundary_operators: Dict mapping dimension k to boundary operator B_k
        include_self_loops: Whether to include diagonal entries (default: False)

    Returns:
        Dict mapping (dim_from, dim_to) to adjacency matrix

    Examples:
        >>> adj_matrices = compute_adjacency_matrices(boundary_ops)
        >>> A_01 = adj_matrices[(0, 1)]  # Vertex-edge incidence
        >>> A_11 = adj_matrices[(1, 1)]  # Edge-edge adjacency
    """
    adjacencies = {}
    dimensions = sorted(boundary_operators.keys())

    # Same-dimension adjacencies (lower Laplacian -> adjacency)
    for k, B_k in boundary_operators.items():
        L_k_down = B_k.T @ B_k

        # Convert Laplacian to adjacency (remove diagonal, negate off-diagonal)
        A_kk = -L_k_down.clone()

        if not include_self_loops:
            A_kk.fill_diagonal_(0)
        else:
            # Keep degree on diagonal
            pass

        # Make non-negative (for some applications)
        A_kk = torch.clamp(A_kk, min=0)

        adjacencies[(k, k)] = A_kk

    # Cross-dimension adjacencies (incidence relations)
    for k, B_k in boundary_operators.items():
        # (k-1) -> k incidence
        adjacencies[(k - 1, k)] = B_k.T
        # k -> (k-1) incidence
        adjacencies[(k, k - 1)] = B_k

    return adjacencies


def validate_boundary_operators(
        boundary_operators: Dict[int, torch.Tensor],
        verbose: bool = True
) -> bool:
    """
    Validate that boundary operators satisfy the chain complex property: B_k @ B_{k+1} = 0

    This is a fundamental property of simplicial complexes that must hold.

    Args:
        boundary_operators: Dict mapping dimension k to boundary operator B_k
        verbose: Whether to print detailed validation results

    Returns:
        True if all boundary operators are valid, False otherwise

    Examples:
        >>> is_valid = validate_boundary_operators(boundary_ops, verbose=True)
        >>> if not is_valid:
        ...     print("Invalid simplicial complex!")
    """
    dimensions = sorted(boundary_operators.keys())
    all_valid = True

    if verbose:
        print("Validating boundary operators...")

    for i in range(len(dimensions) - 1):
        k = dimensions[i]
        k_plus_1 = dimensions[i + 1]

        B_k = boundary_operators[k]
        B_k_plus_1 = boundary_operators[k_plus_1]

        # Check: B_k @ B_{k+1} should be zero matrix
        product = B_k @ B_k_plus_1
        max_error = torch.abs(product).max().item()

        is_valid = max_error < 1e-6
        all_valid = all_valid and is_valid

        if verbose:
            status = "✓" if is_valid else "✗"
            print(f"  {status} B_{k} @ B_{k_plus_1}: max|error| = {max_error:.2e}")

    if verbose:
        if all_valid:
            print("✓ All boundary operators are valid!")
        else:
            print("✗ Some boundary operators violate chain complex property!")

    return all_valid


def compute_betti_numbers(
        boundary_operators: Dict[int, torch.Tensor],
        tolerance: float = 1e-6
) -> Dict[int, int]:
    """
    Compute Betti numbers (topological invariants) of the simplicial complex.

    Betti numbers count the number of k-dimensional holes:
    - β_0: number of connected components
    - β_1: number of 1-dimensional holes (loops)
    - β_2: number of 2-dimensional voids

    Args:
        boundary_operators: Dict mapping dimension k to boundary operator B_k
        tolerance: Tolerance for determining rank via SVD

    Returns:
        Dict mapping dimension k to Betti number β_k

    Examples:
        >>> betti = compute_betti_numbers(boundary_ops)
        >>> print(f"Connected components: {betti[0]}")
        >>> print(f"Loops: {betti[1]}")
    """
    betti_numbers = {}
    dimensions = sorted(boundary_operators.keys())

    def matrix_rank(M: torch.Tensor) -> int:
        """Compute rank of matrix using SVD."""
        if M.numel() == 0:
            return 0
        s = torch.linalg.svdvals(M.float())
        return int((s > tolerance).sum().item())

    for i, k in enumerate(dimensions):
        B_k = boundary_operators[k]

        # Get dimensions
        if i > 0:
            B_k_minus_1 = boundary_operators[dimensions[i - 1]]
            rank_B_k_minus_1 = matrix_rank(B_k_minus_1)
        else:
            rank_B_k_minus_1 = 0

        rank_B_k = matrix_rank(B_k)
        num_k_simplices = B_k.shape[1]

        # Betti number: β_k = dim(ker(B_k)) - dim(im(B_{k+1}))
        # Simplified: β_k = (num k-simplices) - rank(B_k) - rank(B_{k-1})
        betti_k = num_k_simplices - rank_B_k - rank_B_k_minus_1
        betti_numbers[k] = max(0, betti_k)  # Betti numbers are non-negative

    return betti_numbers


def adjust_model_time_parameters(
        model,
        boundary_operators: Dict[int, torch.Tensor],
        safety_margin: float = 0.9,
        verbose: bool = True
) -> Dict[str, float]:
    """
    Automatically adjust model's time parameters to satisfy over-smoothing conditions.

    This function:
    1. Computes Hodge Laplacians from boundary operators
    2. Validates current time parameters
    3. Adjusts them if they violate over-smoothing conditions

    Args:
        model: Continuous simplicial model with learnable time parameters
        boundary_operators: Dict mapping dimension k to boundary operator B_k
        safety_margin: Use this fraction of max safe value (default: 0.9 = 90%)
        verbose: Print adjustment details

    Returns:
        Dict with 'old_t_d', 'old_t_u', 'new_t_d', 'new_t_u', 'adjusted'

    Examples:
        >>> from toposheafx.utils import adjust_model_time_parameters
        >>> result = adjust_model_time_parameters(model, boundary_ops, verbose=True)
        >>> if result['adjusted']:
        ...     print(f"Time parameters adjusted to {result['new_t_d']:.4f}")
    """
    from .diagnostics import validate_time_parameters

    # Compute Laplacians
    laplacians = compute_hodge_laplacians(boundary_operators)

    # Get current state
    if hasattr(model, 'get_time_constants'):
        old_time_params = model.get_time_constants()
        old_t_d = np.mean(old_time_params['t_d']) if old_time_params['t_d'] else 0.1
        old_t_u = np.mean(old_time_params['t_u']) if old_time_params['t_u'] else 0.1
    else:
        old_t_d = old_t_u = 0.1

    # Validate
    results = validate_time_parameters(model, laplacians, verbose=verbose)

    if results['satisfies_condition']:
        return {
            'old_t_d': old_t_d,
            'old_t_u': old_t_u,
            'new_t_d': old_t_d,
            'new_t_u': old_t_u,
            'adjusted': False,
            'max_safe_t': results['suggested_max_t']
        }

    # Calculate safe value
    new_t = results['suggested_max_t'] * safety_margin

    # Apply adjustment
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, 't_d'):
                module.t_d.data = torch.tensor(new_t, dtype=torch.float32, device=module.t_d.device)
            if hasattr(module, 't_u'):
                module.t_u.data = torch.tensor(new_t, dtype=torch.float32, device=module.t_u.device)

    return {
        'old_t_d': old_t_d,
        'old_t_u': old_t_u,
        'new_t_d': new_t,
        'new_t_u': new_t,
        'adjusted': True,
        'max_safe_t': results['suggested_max_t']
    }



# Backward compatibility alias
compute_laplacians = compute_hodge_laplacians