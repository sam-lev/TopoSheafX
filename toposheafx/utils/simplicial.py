# toposheafx/utils/simplicial.py
"""
Utilities for working with simplicial complexes.

This module provides functions for computing topological operators
like Hodge Laplacians, boundary operators, and adjacency matrices.
"""

from collections import defaultdict
from typing import Optional, Union

import numpy as np
import torch


def compute_hodge_laplacians(
    boundary_operators: dict[int, torch.Tensor], normalize: bool = False
) -> dict[int, torch.Tensor]:
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
    boundary_operators: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
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
    boundary_operators: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
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


def compute_hodge_decomposition(boundary_ops):
    """
    Compute Hodge Laplacian decomposition: L = L_down + L_up

    Returns:
        Dict mapping dimension to (L_down, L_up, L_hodge) tuples
    """

    hodge = compute_hodge_laplacians(boundary_ops)
    lower = compute_lower_laplacians(boundary_ops)
    upper = compute_upper_laplacians(boundary_ops)

    result = {}
    for dim in hodge:
        L_hodge = hodge[dim]
        L_down = lower.get(dim)
        L_up = upper.get(dim)
        result[dim] = (L_down, L_up, L_hodge)

    return result


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
    boundary_operators: dict[int, torch.Tensor], include_self_loops: bool = False
) -> dict[tuple[int, int], torch.Tensor]:
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
    sorted(boundary_operators.keys())

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
    boundary_operators: dict[int, torch.Tensor], verbose: bool = True
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
    boundary_operators: dict[int, torch.Tensor], tolerance: float = 1e-6
) -> dict[int, int]:
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
    boundary_operators: dict[int, torch.Tensor],
    safety_margin: float = 0.9,
    verbose: bool = True,
) -> dict[str, float]:
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
    if hasattr(model, "get_time_constants"):
        old_time_params = model.get_time_constants()
        old_t_d = np.mean(old_time_params["t_d"]) if old_time_params["t_d"] else 0.1
        old_t_u = np.mean(old_time_params["t_u"]) if old_time_params["t_u"] else 0.1
    else:
        old_t_d = old_t_u = 0.1

    # Validate
    results = validate_time_parameters(model, laplacians, verbose=verbose)

    if results["satisfies_condition"]:
        return {
            "old_t_d": old_t_d,
            "old_t_u": old_t_u,
            "new_t_d": old_t_d,
            "new_t_u": old_t_u,
            "adjusted": False,
            "max_safe_t": results["suggested_max_t"],
        }

    # Calculate safe value
    new_t = results["suggested_max_t"] * safety_margin

    # Apply adjustment
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "t_d"):
                module.t_d.data = torch.tensor(new_t, dtype=torch.float32, device=module.t_d.device)
            if hasattr(module, "t_u"):
                module.t_u.data = torch.tensor(new_t, dtype=torch.float32, device=module.t_u.device)

    return {
        "old_t_d": old_t_d,
        "old_t_u": old_t_u,
        "new_t_d": new_t,
        "new_t_u": new_t,
        "adjusted": True,
        "max_safe_t": results["suggested_max_t"],
    }


"""
Boundary Operator Construction Utilities for Simplicial Complexes

This module provides functions to construct boundary operators from various
input formats (edge lists, triangle lists, etc.) for use in simplicial neural networks.
"""


def compute_boundary_operators(
    simplices: dict[int, Union[list[tuple], list[list], torch.Tensor, np.ndarray]],
    oriented: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    validate: bool = True,
) -> dict[int, torch.Tensor]:
    """
    Compute boundary operators for a simplicial complex from simplex lists.

    This function constructs the boundary operator matrices B_k for each dimension k,
    where B_k maps k-simplices to (k-1)-simplices. The boundary operators encode
    the incidence structure of the simplicial complex and satisfy the fundamental
    chain complex property: B_k @ B_{k+1} = 0.

    MATHEMATICAL DEFINITION:
    =======================
    For a simplicial complex with k-simplices and (k-1)-simplices:

    B_k: C_k → C_{k-1}

    where C_k is the chain group of k-simplices. The matrix B_k has:
    - Rows: indexed by (k-1)-simplices
    - Columns: indexed by k-simplices
    - Entries: B_k[i, j] indicates if (k-1)-simplex i is a face of k-simplex j

    For oriented complexes:
        B_k[i, j] = +1 if face i appears with positive orientation in simplex j
        B_k[i, j] = -1 if face i appears with negative orientation in simplex j
        B_k[i, j] = 0  if face i is not a face of simplex j

    For unoriented complexes:
        B_k[i, j] ∈ {0, 1} (only incidence, no orientation)

    CHAIN COMPLEX PROPERTY:
    ======================
    The fundamental property that must hold:
        B_k @ B_{k+1} = 0  (zero matrix)

    This reflects that "the boundary of a boundary is zero" - a cornerstone
    of algebraic topology.

    EXAMPLES:
    =========

    Example 1: Simple triangle (vertices, edges, one triangle)
    ----------------------------------------------------------
    >>> vertices = [(0,), (1,), (2,)]  # Can be omitted, inferred from edges
    >>> edges = [(0, 1), (1, 2), (2, 0)]
    >>> triangles = [(0, 1, 2)]
    >>> simplices = {1: edges, 2: triangles}
    >>> boundary_ops = compute_boundary_operators(simplices)
    >>> B_1 = boundary_ops[1]  # Shape: [3 vertices, 3 edges]
    >>> B_2 = boundary_ops[2]  # Shape: [3 edges, 1 triangle]
    >>> # Verify chain complex property
    >>> assert torch.allclose(B_1 @ B_2, torch.zeros_like(B_1 @ B_2))

    Example 2: Multiple triangles forming a mesh
    --------------------------------------------
    >>> edges = [(0,1), (1,2), (2,3), (3,0), (0,2)]  # 5 edges
    >>> triangles = [(0,1,2), (0,2,3)]  # 2 triangles sharing edge (0,2)
    >>> simplices = {1: edges, 2: triangles}
    >>> boundary_ops = compute_boundary_operators(simplices)
    >>> B_1 = boundary_ops[1]  # [4 vertices, 5 edges]
    >>> B_2 = boundary_ops[2]  # [5 edges, 2 triangles]

    Example 3: 3D tetrahedral mesh
    -----------------------------
    >>> triangles = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]  # 4 faces
    >>> tetrahedra = [(0,1,2,3)]  # 1 tetrahedron
    >>> simplices = {2: triangles, 3: tetrahedra}
    >>> boundary_ops = compute_boundary_operators(simplices)
    >>> B_2 = boundary_ops[2]  # [6 edges, 4 triangles]
    >>> B_3 = boundary_ops[3]  # [4 triangles, 1 tetrahedron]

    Example 4: Using numpy arrays or tensors
    ----------------------------------------
    >>> edges_np = np.array([[0, 1], [1, 2], [2, 0]])
    >>> triangles_tensor = torch.tensor([[0, 1, 2]])
    >>> simplices = {1: edges_np, 2: triangles_tensor}
    >>> boundary_ops = compute_boundary_operators(simplices)

    Args:
        simplices: Dictionary mapping dimension k to list of k-simplices.
                  Each k-simplex is a tuple/list of vertex indices.
                  Format: {k: [(v0, v1, ..., v_k), ...]}
                  Examples:
                    - Edges (1-simplices): {1: [(0,1), (1,2), (2,0)]}
                    - Triangles (2-simplices): {2: [(0,1,2), (1,2,3)]}
                    - Tetrahedra (3-simplices): {3: [(0,1,2,3)]}

                  Can accept:
                    - List of tuples: [(0,1), (1,2)]
                    - List of lists: [[0,1], [1,2]]
                    - Numpy array: np.array([[0,1], [1,2]])
                    - PyTorch tensor: torch.tensor([[0,1], [1,2]])

        oriented: If True, compute oriented boundary operators with +1/-1 entries.
                 If False, use only 0/1 (incidence only, no orientation).
                 Default: True

        dtype: PyTorch data type for output tensors.
              Default: torch.float32

        device: PyTorch device for output tensors (CPU, CUDA, etc.).
               If None, uses default device.
               Default: None

        validate: If True, validate the chain complex property after construction.
                 This checks that B_k @ B_{k+1} = 0 for all k.
                 Recommended to keep True for debugging.
                 Default: True

    Returns:
        Dictionary mapping dimension k to boundary operator B_k.
        - B_k: torch.Tensor of shape [num_(k-1)_simplices, num_k_simplices]
        - Keys start from 1 (edges) since B_0 (vertex to nothing) doesn't exist

        Example return:
        {
            1: B_1,  # [num_vertices, num_edges]
            2: B_2,  # [num_edges, num_triangles]
            3: B_3,  # [num_triangles, num_tetrahedra]
        }

    Raises:
        ValueError: If input format is invalid or chain complex property fails
        TypeError: If simplex format is not recognized

    Notes:
        - Vertex indices should be consecutive integers starting from 0
        - Vertices (0-simplices) are automatically inferred from higher simplices
        - Orientation convention: alternating sign based on position in simplex
        - For efficiency, uses sparse representations internally but returns dense tensors

    Mathematical Background:
        See "Computational Topology" by Edelsbrunner & Harer (2010)
        and "Discrete Morse Theory" by Forman (1998)

    Implementation Details:
        1. Extract all vertices from higher-dimensional simplices
        2. Build mappings from simplices to indices
        3. For each dimension k:
           a. Enumerate all k-simplices and (k-1)-simplices
           b. For each k-simplex, find its (k-1)-dimensional faces
           c. Populate B_k matrix with appropriate signs
        4. Validate chain complex property if requested
    """
    if not simplices:
        raise ValueError("simplices dictionary cannot be empty")

    # Determine device
    if device is None:
        device = torch.device("cpu")

    # Step 1: Normalize input format and extract dimensions
    normalized_simplices = _normalize_simplex_input(simplices)
    dimensions = sorted(normalized_simplices.keys())

    if not dimensions:
        raise ValueError("No simplices provided")

    # Step 2: Build vertex set and lower-dimensional simplices
    all_simplices = _build_complete_simplex_hierarchy(normalized_simplices)

    # Step 3: Create index mappings for each dimension
    simplex_to_index = {}
    for dim in range(max(dimensions) + 1):
        if dim in all_simplices:
            simplex_to_index[dim] = {
                simplex: idx for idx, simplex in enumerate(sorted(all_simplices[dim]))
            }

    # Step 4: Compute boundary operators
    boundary_operators = {}

    for k in dimensions:
        if k == 0:
            continue  # B_0 doesn't exist (no -1 dimensional simplices)

        k_simplices = sorted(all_simplices[k])
        k_minus_1_simplices = sorted(all_simplices[k - 1])

        num_k_minus_1 = len(k_minus_1_simplices)
        num_k = len(k_simplices)

        # Initialize boundary operator matrix
        B_k = torch.zeros(num_k_minus_1, num_k, dtype=dtype, device=device)

        # Populate the boundary operator
        for j, k_simplex in enumerate(k_simplices):
            # Get all (k-1)-dimensional faces of this k-simplex
            faces = _get_faces(k_simplex, oriented=oriented)

            for face, sign in faces:
                # Get index of this face
                i = simplex_to_index[k - 1][face]

                # Set the boundary operator entry
                if oriented:
                    B_k[i, j] = sign
                else:
                    B_k[i, j] = 1.0

        boundary_operators[k] = B_k

    # Step 5: Validate if requested
    if validate:
        _validate_chain_complex(boundary_operators, verbose=False)

    return boundary_operators


def _normalize_simplex_input(
    simplices: dict[int, Union[list, torch.Tensor, np.ndarray]],
) -> dict[int, list[tuple]]:
    """
    Normalize various input formats to standard list of tuples.

    Converts:
    - List of lists [[0,1], [1,2]] → [(0,1), (1,2)]
    - Numpy arrays → list of tuples
    - PyTorch tensors → list of tuples
    - Already tuples → pass through
    """
    normalized = {}

    for dim, simplex_list in simplices.items():
        if not isinstance(dim, int) or dim < 0:
            raise ValueError(f"Dimension must be non-negative integer, got {dim}")

        # Convert to list if tensor/array
        if isinstance(simplex_list, torch.Tensor):
            simplex_list = simplex_list.cpu().numpy().tolist()
        elif isinstance(simplex_list, np.ndarray):
            simplex_list = simplex_list.tolist()

        # Convert each simplex to sorted tuple
        normalized_dim = []
        for simplex in simplex_list:
            if isinstance(simplex, (list, tuple)):
                # Sort vertices for canonical representation
                simplex_tuple = tuple(sorted(simplex))

                # Validate dimension
                if len(simplex_tuple) != dim + 1:
                    raise ValueError(
                        f"Simplex {simplex} has {len(simplex_tuple)} vertices "
                        f"but dimension {dim} requires {dim + 1} vertices"
                    )

                normalized_dim.append(simplex_tuple)
            else:
                raise TypeError(f"Invalid simplex format: {simplex}")

        normalized[dim] = normalized_dim

    return normalized


def _build_complete_simplex_hierarchy(simplices: dict[int, list[tuple]]) -> dict[int, set[tuple]]:
    """
    Build complete hierarchy including all lower-dimensional faces.

    Given higher-dimensional simplices, extract all lower-dimensional
    faces (vertices, edges, etc.).

    Example:
        Input: {2: [(0,1,2)]}  # One triangle
        Output: {
            0: {(0,), (1,), (2,)},        # 3 vertices
            1: {(0,1), (0,2), (1,2)},      # 3 edges
            2: {(0,1,2)}                    # 1 triangle
        }
    """
    all_simplices = defaultdict(set)

    # Add provided simplices
    for dim, simplex_list in simplices.items():
        all_simplices[dim].update(simplex_list)

    # Extract all faces recursively
    max_dim = max(simplices.keys())

    for dim in range(max_dim, 0, -1):
        if dim not in all_simplices:
            continue

        for simplex in all_simplices[dim]:
            # Get all (dim-1)-dimensional faces
            faces = _get_all_faces_recursive(simplex, target_dim=dim - 1)
            all_simplices[dim - 1].update(faces)

    # Ensure vertices (0-simplices) exist extract verts from dimension 1
    if 0 not in all_simplices and len(all_simplices) > 0 and 1 in all_simplices:
        for edge in all_simplices[1]:
            for vertex in edge:
                all_simplices[0].add((vertex,))

    return dict(all_simplices)


def _get_all_faces_recursive(simplex: tuple, target_dim: int) -> set[tuple]:
    """
    Get all faces of a simplex at a specific dimension.

    Example:
        simplex = (0,1,2,3)  # tetrahedron
        target_dim = 2       # want triangular faces
        Returns: {(0,1,2), (0,1,3), (0,2,3), (1,2,3)}
    """
    from itertools import combinations

    current_dim = len(simplex) - 1

    if current_dim < target_dim:
        return set()

    if current_dim == target_dim:
        return {simplex}

    # Get all (target_dim + 1)-sized subsets
    faces = set()
    for face_vertices in combinations(simplex, target_dim + 1):
        faces.add(tuple(sorted(face_vertices)))

    return faces


def _get_faces(simplex: tuple, oriented: bool = True) -> list[tuple[tuple, int]]:
    """
    Get all codimension-1 faces of a simplex with orientation.

    For a k-simplex (v_0, v_1, ..., v_k), the faces are obtained by
    removing one vertex at a time. The orientation alternates based on
    which vertex is removed.

    Args:
        simplex: Tuple of vertex indices (sorted)
        oriented: If True, return signed faces; if False, all signs are +1

    Returns:
        List of (face, sign) tuples where:
        - face: tuple of vertices (sorted)
        - sign: +1 or -1 (orientation)

    Example:
        simplex = (0, 1, 2)  # triangle
        Returns: [
            ((0, 1), +1),  # Remove vertex 2
            ((0, 2), -1),  # Remove vertex 1
            ((1, 2), +1),  # Remove vertex 0
        ]

    Mathematical convention:
        Face obtained by removing vertex at position i has sign (-1)^i
    """
    faces = []
    vertices = list(simplex)

    for _i, vertex_to_remove in enumerate(vertices):
        # Create face by removing this vertex
        face_vertices = [v for v in vertices if v != vertex_to_remove]
        face = tuple(sorted(face_vertices))

        # Determine sign based on position
        # Convention: removing vertex at position i gives sign (-1)^i
        if oriented:
            # Need to track position in ORIGINAL simplex
            original_position = vertices.index(vertex_to_remove)
            sign = (-1) ** original_position
        else:
            sign = 1

        faces.append((face, sign))

    return faces


def _validate_chain_complex(
    boundary_operators: dict[int, torch.Tensor], tolerance: float = 1e-6, verbose: bool = True
) -> None:
    """
    Validate that boundary operators satisfy B_k @ B_{k+1} = 0.

    Raises:
        ValueError: If chain complex property is violated
    """
    dimensions = sorted(boundary_operators.keys())

    for i in range(len(dimensions) - 1):
        k = dimensions[i]
        k_plus_1 = dimensions[i + 1]

        B_k = boundary_operators[k]
        B_k_plus_1 = boundary_operators[k_plus_1]

        # Compute product
        product = B_k @ B_k_plus_1
        max_error = torch.abs(product).max().item()

        if max_error > tolerance:
            raise ValueError(
                f"Chain complex property violated: B_{k} @ B_{k_plus_1} has "
                f"max absolute value {max_error:.2e} > tolerance {tolerance:.2e}\n"
                f"This indicates an invalid simplicial complex structure."
            )

        if verbose:
            print(f"✓ B_{k} @ B_{k_plus_1} = 0 (max error: {max_error:.2e})")


# ============================================================================
# Additional Utility Functions
# ============================================================================


def compute_boundary_operators_from_edge_index(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> dict[int, torch.Tensor]:
    """
    Compute B_1 (vertex-to-edge boundary operator) from PyTorch Geometric edge_index.

    This is a convenience function for graph neural network users who have
    edge data in the standard PyG format.

    Args:
        edge_index: PyTorch Geometric edge index [2, num_edges]
                   Format: [[source_nodes], [target_nodes]]
        num_nodes: Total number of nodes (if None, inferred from edge_index)
        dtype: Data type for output tensor
        device: Device for output tensor

    Returns:
        Dictionary with single key {1: B_1} where B_1 is [num_nodes, num_edges]

    Example:
        >>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Triangle
        >>> boundary_ops = compute_boundary_operators_from_edge_index(edge_index)
        >>> B_1 = boundary_ops[1]  # Shape: [3, 3]
    """
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges], got {edge_index.shape}")

    # Determine number of nodes
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    num_edges = edge_index.shape[1]

    # Determine device
    if device is None:
        device = edge_index.device

    # Create B_1 matrix
    B_1 = torch.zeros(num_nodes, num_edges, dtype=dtype, device=device)

    # Fill in boundary operator
    # For directed edge (src, tgt), we have:
    # B_1[src, edge_idx] = -1  (source/tail)
    # B_1[tgt, edge_idx] = +1  (target/head)
    for edge_idx in range(num_edges):
        src = edge_index[0, edge_idx].item()
        tgt = edge_index[1, edge_idx].item()

        B_1[src, edge_idx] = -1.0
        B_1[tgt, edge_idx] = +1.0

    return {1: B_1}


def infer_simplicial_complex(
    edge_index: torch.Tensor,
    find_triangles: bool = True,
    find_tetrahedra: bool = False,
    num_nodes: Optional[int] = None,
) -> dict[int, list[tuple]]:
    """
    Infer higher-order simplices from graph edges (clique detection).

    Given a graph, find all cliques which form simplices:
    - Triangles (3-cliques) → 2-simplices
    - Tetrahedra (4-cliques) → 3-simplices

    This is useful when you have a graph but want to leverage its
    higher-order structure for simplicial neural networks.

    Args:
        edge_index: PyTorch Geometric edge index [2, num_edges]
        find_triangles: If True, detect all triangles
        find_tetrahedra: If True, detect all tetrahedra (computationally expensive!)
        num_nodes: Total number of nodes

    Returns:
        Dictionary mapping dimensions to simplex lists:
        {1: edges, 2: triangles, 3: tetrahedra}

    Example:
        >>> # Triangle graph
        >>> edge_index = torch.tensor([[0,1,2,1,2,0], [1,2,0,0,1,2]])
        >>> simplices = infer_simplicial_complex(edge_index)
        >>> print(simplices[2])  # [(0, 1, 2)]

    Warning:
        Finding tetrahedra is O(n^4) and can be very slow for large graphs!
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    # Build adjacency information
    adjacency = defaultdict(set)
    edges = []

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        tgt = edge_index[1, i].item()

        if src != tgt:  # Ignore self-loops
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)

            edge = tuple(sorted([src, tgt]))
            if edge not in edges:
                edges.append(edge)

    simplices = {1: edges}

    # Find triangles
    if find_triangles:
        triangles = set()

        for edge in edges:
            u, v = edge
            # Find common neighbors
            common_neighbors = adjacency[u] & adjacency[v]

            for w in common_neighbors:
                if w > v:  # Avoid duplicates (canonical ordering)
                    triangle = tuple(sorted([u, v, w]))
                    triangles.add(triangle)

        if triangles:
            simplices[2] = list(triangles)

    # Find tetrahedra
    if find_tetrahedra and find_triangles and 2 in simplices:
        tetrahedra = set()
        triangles = simplices[2]

        for tri in triangles:
            u, v, w = tri
            # Find nodes connected to all three vertices
            common = adjacency[u] & adjacency[v] & adjacency[w]

            for x in common:
                if x > w:  # Canonical ordering
                    tet = tuple(sorted([u, v, w, x]))
                    tetrahedra.add(tet)

        if tetrahedra:
            simplices[3] = list(tetrahedra)

    return simplices


# Backward compatibility alias
compute_laplacians = compute_hodge_laplacians
