# conv.py
"""
conv.py

Copresheaf-based message passing components for graph and cell complex neural networks.
Implements local copresheaf morphisms and aggregation rules as described in
"Copresheaf Topological Neural Networks" Hajij et. al.

Usage:
-------
>>> import torch
>>> x = torch.randn(4, 8)  # 4 nodes, 8-dimensional features
>>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
>>> model = Conv(8, 16)
>>> h_src, h_tgt = x[edge_index[0]], x[edge_index[1]]
>>> messages = model(h_src, h_tgt)
>>> agg = model.aggregate_messages(messages, edge_index[1], num_nodes=4)

>>> # Higher-order message passing
>>> model_ho = HigherOrderConv(8, 16, num_neighborhoods=1)
>>> out = model_ho(x, [edge_index])
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from toposheafx.base.message_passing import MessagePassing, HigherOrderMessagePassing



class Conv(MessagePassing):
    """
    Copresheaf morphism for message passing using the custom MessagePassing framework.

    This class uses the custom MessagePassing base class from toposheafx.base.message_passing
    and provides the specific copresheaf morphism factories for the mathematical components.

    Args:
        in_dim (int): Dimensionality of node input features F(x).
        out_dim (int): Dimensionality of the transformed message output.
        edge_attr_dim (int): Optional dimension of edge attributes.
        aggr (str): Aggregation method ('sum', 'mean', 'max').
        bias (bool): Whether to add learnable bias.
    """

    def __init__(self, in_dim, out_dim, edge_attr_dim=0, aggr="sum", bias=True, **kwargs):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            edge_attr_dim=edge_attr_dim,
            aggr=aggr,
            copresheaf_map_factory=self._conv_copresheaf_factory,
            message_factory=self._conv_message_factory,
            update_factory=self._conv_update_factory,
            **kwargs,
        )

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def _conv_copresheaf_factory(self, in_dim: int, out_dim: int, edge_attr_dim: int):
        """
        Factory for the copresheaf morphism ρ : F(y) → F(x).
        This matches the rho_map from your original Conv class.
        """
        rho_input_dim = 2 * in_dim + edge_attr_dim
        return nn.Sequential(nn.Linear(rho_input_dim, out_dim), nn.Tanh())

    def _conv_message_factory(self, in_dim: int, out_dim: int):
        """
        Factory for the message function α: combines h_x and ρ(h_y).
        This matches the alpha from your original Conv class.
        """
        return nn.Sequential(nn.Linear(in_dim + out_dim, out_dim), nn.ReLU())

    def _conv_update_factory(self, in_dim: int, out_dim: int):
        """
        Factory for the update function β: takes h_x and aggregated messages.
        This matches the Beta from your original Conv class.
        """
        return nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        """
        Forward pass implementing copresheaf message passing.

        Args:
            x (Tensor): Node features [num_nodes, in_dim]
            edge_index (Tensor): Edge indices [2, num_edges]
            edge_attr (Tensor, optional): Edge attributes [num_edges, edge_attr_dim]
            size (tuple, optional): Size of source and target nodes

        Returns:
            Tensor: Updated node features [num_nodes, out_dim]
        """
        # Use the parent MessagePassing forward method
        out = super().forward(x, edge_index, edge_attr, size)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias

        return out

    # Maintain backward compatibility with existing interface
    def aggregate_messages(self, messages, index, num_nodes, reduce="sum"):
        """
        Legacy method for backward compatibility with existing code.
        Delegates to the parent class's aggregate method.
        """
        return self.aggregate(messages, index, dim_size=num_nodes)

    def update_node(self, h_x, aggr_msg):
        """
        Legacy method for backward compatibility with existing code.
        Delegates to the parent class's update method.
        """
        return self.update(aggr_msg, h_x)



"""
Continuous simplicial convolution operations based on COSIMO methodology.
Integrates continuous Hodge-Laplacian dynamics with copresheaf message passing.
"""
def continuous_simplicial_conv(
        x_k: torch.Tensor,
        laplacian_k: torch.Tensor,
        t: torch.Tensor,
        K: Optional[int] = None,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    Continuous simplicial convolution using matrix exponential: e^(-t*L) @ x

    Implements the continuous filter from COSIMO paper:
    h(t) = e^(-tL) x where L is the Hodge Laplacian

    Args:
        x_k: Features on k-simplices [num_simplices, features]
        laplacian_k: Hodge Laplacian for k-simplices [num_simplices, num_simplices]
        t: Continuous time parameter (scalar or learnable parameter)
        K: Number of dominant eigenvectors to use (None = all)

    Returns:
        Filtered features [num_simplices, features]
    """
    # Validate inputs
    if torch.isnan(x_k).any() or torch.isnan(laplacian_k).any():
        raise ValueError("Input contains NaN!")

    # Clamp time parameter to safe range
    # t_safe = torch.clamp(t, min=eps, max=100.0)

    if K is None or K >= laplacian_k.size(0):
        # Full matrix exponential
        # Add small epsilon to diagonal for numerical stability
        L_stable = laplacian_k + eps * torch.eye(laplacian_k.size(0), device=laplacian_k.device)
        exp_L = torch.matrix_exp(-t_safe * L_stable)

        # Check for NaN in result
        if torch.isnan(exp_L).any():
            print(f"WARNING: matrix_exp produced NaN with t={t_safe.item()}")
            return x_k  # Return input unchanged as fallback

        return exp_L @ x_k
    else:
        # Efficient approximation using top-K eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_k)

        # Filter out negative eigenvalues (shouldn't happen)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)

        # Take K largest eigenvalues (they're sorted ascending)
        top_K_indices = torch.argsort(eigenvalues, descending=True)[:K]
        V_K = eigenvectors[:, top_K_indices]  # [num_simplices, K]
        lambda_K = eigenvalues[top_K_indices]  # [K]

        # Compute e^(-t*λ) for top-K eigenvalues
        exp_lambda = torch.exp(-t * lambda_K)  # [K]
        # clamp to safe range
        exp_lambda = torch.clamp(exp_lambda, min=eps, max=1.0)

        # Reconstruct: V * diag(e^(-tλ)) * V^T * x
        result = V_K @ (exp_lambda.unsqueeze(-1) * (V_K.T @ x_k))

        if torch.isnan(result).any():
            print(f"WARNING: Eigenvalue approximation produced NaN")
            return x_k

        return result


class ContinuousConv(MessagePassing):
    """
    Continuous copresheaf morphism using Hodge-Laplacian dynamics.

    Extends the discrete Conv class to use continuous diffusion instead of
    discrete message passing. Maintains copresheaf structure while enabling
    continuous receptive fields.

    Mathematical Framework:
    - Lower dynamics: e^(-t_d * L_k,d) @ x_k,d
    - Upper dynamics: e^(-t_u * L_k,u) @ x_k,u
    - Joint dynamics: (e^(-t_d * L_k,d) + e^(-t_u * L_k,u)) @ x_k

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        t_d_init: Initial value for lower time parameter
        t_u_init: Initial value for upper time parameter
        learnable_t: Whether time parameters are learnable
        K: Number of eigenvectors for approximation
        edge_attr_dim: Optional edge attribute dimension
        aggr: Aggregation method
        bias: Whether to include bias
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            t_d_init: float = 0.1,
            t_u_init: float = 0.1,
            learnable_t: bool = True,
            K: Optional[int] = None,
            edge_attr_dim: int = 0,
            aggr: str = "sum",
            bias: bool = True,
            **kwargs
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            edge_attr_dim=edge_attr_dim,
            aggr=aggr,
            copresheaf_map_factory=self._continuous_copresheaf_factory,
            message_factory=self._continuous_message_factory,
            update_factory=self._continuous_update_factory,
            **kwargs
        )

        # Continuous time parameters (learnable by default)
        if learnable_t:
            self.t_d = nn.Parameter(torch.tensor(t_d_init))
            self.t_u = nn.Parameter(torch.tensor(t_u_init))
        else:
            self.register_buffer('t_d', torch.tensor(t_d_init))
            self.register_buffer('t_u', torch.tensor(t_u_init))

        self.K = K  # Number of eigenvectors for approximation

        # Projection layers for continuous features
        self.proj_down = nn.Linear(in_dim, out_dim)
        self.proj_up = nn.Linear(in_dim, out_dim)
        self.proj_joint = nn.Linear(in_dim, out_dim)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def _continuous_copresheaf_factory(self, in_dim: int, out_dim: int, edge_attr_dim: int):
        """Factory for continuous copresheaf morphism."""
        # Simpler than discrete version since continuous diffusion handles transport
        return nn.Sequential(
            nn.Linear(in_dim + edge_attr_dim, out_dim),
            nn.Tanh()
        )

    def _continuous_message_factory(self, in_dim: int, out_dim: int):
        """Factory for message combination."""
        return nn.Sequential(
            nn.Linear(out_dim + out_dim, out_dim),
            nn.ReLU()
        )

    def _continuous_update_factory(self, in_dim: int, out_dim: int):
        """Factory for final update."""
        return nn.Sequential(
            nn.Linear(out_dim + out_dim, out_dim),
            nn.ReLU()
        )

    def reset_parameters(self):
        """Initialize parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def compute_hodge_laplacians(
            self,
            B_k: torch.Tensor,
            B_k_plus_1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute lower and upper Hodge Laplacians.

        Args:
            B_k: Boundary operator k [num_k-1_simplices, num_k_simplices]
            B_k_plus_1: Boundary operator k+1 [num_k_simplices, num_k+1_simplices]

        Returns:
            (L_k_down, L_k_up) where:
            - L_k_down = B_k^T @ B_k (lower Laplacian)
            - L_k_up = B_k+1 @ B_k+1^T (upper Laplacian)
        """
        L_k_down = B_k.T @ B_k
        L_k_up = B_k_plus_1 @ B_k_plus_1.T if B_k_plus_1 is not None else None
        return L_k_down, L_k_up

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            B_k: torch.Tensor,
            B_k_plus_1: Optional[torch.Tensor] = None,
            edge_attr: Optional[torch.Tensor] = None,
            size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Forward pass with continuous Hodge-Laplacian dynamics.

        Args:
            x: Node/simplex features [num_simplices, in_dim]
            edge_index: Edge connectivity [2, num_edges]
            B_k: Boundary operator for lower Laplacian
            B_k_plus_1: Boundary operator for upper Laplacian (optional)
            edge_attr: Edge attributes (optional)
            size: Size tuple (optional)

        Returns:
            Updated features [num_simplices, out_dim]
        """
        # Compute Hodge Laplacians
        L_k_down, L_k_up = self.compute_hodge_laplacians(B_k, B_k_plus_1)

        # Apply continuous diffusion on lower space
        x_down = continuous_simplicial_conv(x, L_k_down, self.t_d, self.K)
        x_down = self.proj_down(x_down)

        # Apply continuous diffusion on upper space (if exists)
        if L_k_up is not None:
            x_up = continuous_simplicial_conv(x, L_k_up, self.t_u, self.K)
            x_up = self.proj_up(x_up)
        else:
            x_up = torch.zeros_like(x_down)

        # Joint dynamics
        x_joint = continuous_simplicial_conv(x, L_k_down, self.t_d, self.K)
        if L_k_up is not None:
            x_joint = x_joint + continuous_simplicial_conv(x, L_k_up, self.t_u, self.K)
        x_joint = self.proj_joint(x_joint)

        # Combine all components
        out = x_down + x_up + x_joint

        if self.bias is not None:
            out = out + self.bias

        return out

    def check_oversmoothing_condition(self, laplacian: torch.Tensor) -> torch.Tensor:
        """
        Check over-smoothing condition from COSIMO Proposition 5.5:
        t < ln(s) / (λ_min(L) + κ_f(L))

        Args:
            laplacian: Hodge Laplacian matrix

        Returns:
            Maximum allowed time parameter
        """
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        lambda_min = eigenvalues[eigenvalues > 1e-6].min()
        lambda_max = eigenvalues.max()
        kappa_f = lambda_max / lambda_min  # Condition number

        # Assume s_norm from weight norms
        s_norm = max(
            torch.norm(self.proj_down.weight).item(),
            torch.norm(self.proj_up.weight).item()
        )

        max_t = torch.log(torch.tensor(s_norm)) / (lambda_min + kappa_f)
        return max_t
#
# class Conv(nn.Module):
#     """
#     Implements a copresheaf morphism for message passing on directed complexes.
#
#     Implements components described in Proposition 1:
#     - rho: F(y) → F(x), a learnable transport map between feature spaces of incident cells
#     - alpha: combines h_x and transported h_y (i.e., alpha(h_x, rho(h_y)))
#     - beta: updates h_x using the cumulative sum ⊕_y alpha(h_x, rho(h_y))
#
#     Args:
#         in_dim (int): Dimensionality of node input features F(x).
#         out_dim (int): Dimensionality of the transformed message output.
#         edge_attr_dim (int): Optional dimension of edge attributes.
#     """
#     def __init__(self, in_dim, out_dim, edge_attr_dim=0):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.edge_attr_dim = edge_attr_dim
#
#         # Learnable copresheaf transport morphism: ρ : F(y) → F(x)
#         # Takes h_y, h_x, and optional edge_attr and produces ρ(h_y; h_x)
#         rho_input_dim = 2 * in_dim + edge_attr_dim
#         self.rho_map = nn.Sequential(
#             nn.Linear(rho_input_dim, out_dim),
#             nn.Tanh()
#         )
#
#         # Message function α: combines h_x and ρ(h_y)
#         self.alpha = nn.Sequential(
#             nn.Linear(in_dim + out_dim, out_dim),
#             nn.ReLU()
#         )
#
#         # Update function β: takes h_x and the aggregated α-messages
#         self.Beta = nn.Sequential(
#             nn.Linear(in_dim + out_dim, out_dim),
#             nn.ReLU(),
#             nn.Linear(out_dim, out_dim)
#         )
#
#     def forward(self, h_src, h_tgt, edge_attr=None):
#         """
#         Computes directional messages α(h_x, ρ(h_y)) per edge.
#
#         Args:
#             h_src (Tensor): Source node features h_y [num_edges, in_dim]
#             h_tgt (Tensor): Target node features h_x [num_edges, in_dim]
#             edge_attr (Tensor or None): Optional edge attributes [num_edges, edge_attr_dim]
#
#         Returns:
#             Tensor: Output messages [num_edges, out_dim]
#         """
#         # if edge_attr is not None:
#         #     if not isinstance(edge_attr, torch.Tensor):
#         #         raise TypeError(f"Expected edge_attr to be a Tensor or None, but got {type(edge_attr)}.")
#         #     rho_input = torch.cat([h_src, h_tgt, edge_attr], dim=-1)
#         # else:
#         #     rho_input = torch.cat([h_src, h_tgt], dim=-1)
#         #
#         # rho_output = self.rho_map(rho_input)
#         # return self.alpha(torch.cat([h_tgt, rho_output], dim=-1))
#
#         _same_device(h_src, h_tgt, edge_attr if edge_attr is not None else h_src)
#         if h_src.dim() != 2 or h_tgt.dim() != 2:
#             raise ValueError("h_src and h_tgt must be [num_edges, in_dim].")
#         if h_src.size(0) != h_tgt.size(0):
#             raise ValueError("h_src and h_tgt must have the same num_edges.")
#         if h_src.size(1) != self.in_dim or h_tgt.size(1) != self.in_dim:
#             raise ValueError(f"Expected in_dim={self.in_dim} for h_src/h_tgt.")
#
#         if edge_attr is not None:
#             if edge_attr.dim() != 2 or edge_attr.size(0) != h_src.size(0):
#                 raise ValueError("edge_attr must be [num_edges, edge_attr_dim].")
#             if edge_attr.size(1) != self.edge_attr_dim:
#                 raise ValueError(f"edge_attr_dim mismatch (got {edge_attr.size(1)}, "
#                                  f"expected {self.edge_attr_dim}).")
#             rho_input = torch.cat([h_src, h_tgt, edge_attr], dim=-1)
#         else:
#             rho_input = torch.cat([h_src, h_tgt], dim=-1)
#
#         rho_output = self.rho_map(rho_input)
#         return self.alpha(torch.cat([h_tgt, rho_output], dim=-1))
#
#     # old implementation
#     # def aggregate_messages(self, messages, index, num_nodes):
#     #     """
#     #     Aggregates messages per node using cumulative sum ⊕.
#     #
#     #     Args:
#     #         messages (Tensor): Message values [num_edges, out_dim]
#     #         index (Tensor): Target indices (node IDs) for each message [num_edges]
#     #         num_nodes (int): Total number of nodes in the graph
#     #
#     #     Returns:
#     #         Tensor: Aggregated messages [num_nodes, out_dim]
#     #     """
#     #     return torch.zeros(num_nodes, self.out_dim, device=messages.device).scatter_add_(
#     #         dim=0,
#     #         index=index.unsqueeze(-1).expand(-1, messages.size(-1)),
#     #         src=messages
#     #     )
#
#     def aggregate_messages(self, messages, index, num_nodes, reduce='sum'): #reduce='sum', 'mean', 'max'
#         """
#         Aggregates messages per node using torch_scatter.
#
#         Args:
#             messages (Tensor): [num_edges, out_dim]
#             index (Tensor): [num_edges]
#             num_nodes (int): number of nodes
#             reduce (str): aggregation type: 'sum', 'mean', or 'max'
#
#         Returns:
#             Tensor: [num_nodes, out_dim]
#         """
#         _check_index(index, num_nodes, name="target index (col)")
#         _same_device(messages, index)
#         if reduce not in {"sum","mean","max"}:
#             raise ValueError("reduce must be 'sum', 'mean', or 'max'.")
#         return scatter(messages, index, dim=0, dim_size=num_nodes, reduce=reduce)
#
#     def update_node(self, h_x, aggr_msg):
#         """
#         Updates node representation via β(h_x, ⊕_y α(h_x, ρ(h_y))).
#
#         Args:
#             h_x (Tensor): Previous node features [num_nodes, in_dim]
#             aggr_msg (Tensor): Aggregated messages [num_nodes, out_dim]
#
#         Returns:
#             Tensor: Updated node features [num_nodes, out_dim]
#         """
#         return self.Beta(torch.cat([h_x, aggr_msg], dim=-1))


class HigherOrderConv(HigherOrderMessagePassing):
    """
    Higher-order copresheaf-based message passing using the custom HigherOrderMessagePassing.

    This class uses the HigherOrderMessagePassing base class and provides Conv instances
    for each neighborhood, maintaining the same mathematical framework as your original code.

    Args:
        in_dim (int): Input feature dimension F(x)
        out_dim (int): Output feature dimension F'(x)
        edge_attr_dim (int): Dim of edge attributes per neighborhood
        num_neighborhoods (int): Number of neighborhood types
        edge_map (class): Convolution class to use (default: Conv)
        combine_method (str): How to combine neighborhoods ('concat', 'sum', 'mean')
        aggr (str): Aggregation method for each neighborhood
        bias (bool): Whether to add learnable bias
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        edge_attr_dim=0,
        num_neighborhoods=1,
        edge_map=None,
        combine_method="concat",
        aggr="sum",
        bias=True,
        **kwargs,
    ):

        if edge_map is None:
            edge_map = Conv

        # Create factory function for the neighborhood layers
        def neighborhood_factory():
            return edge_map(in_dim, out_dim, edge_attr_dim, aggr=aggr, bias=False)

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            edge_attr_dim=edge_attr_dim,
            num_neighborhoods=num_neighborhoods,
            aggr=aggr,
            combine_method=combine_method,
            **kwargs,
        )

        # Replace the default neighborhood layers with our custom Conv layers
        self.neighborhood_layers = nn.ModuleList(
            [neighborhood_factory() for _ in range(num_neighborhoods)]
        )

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index_list, edge_attr_list=None):
        """
        Forward pass implementing higher-order copresheaf message passing (Proposition 3).

        MATHEMATICAL EXECUTION:
        ======================
        This method executes the multi-step process described in Proposition 3:

        Step 1: For each neighborhood k ∈ {1, ..., n}:
                Compute ⊕_{y∈N_k(x)} α_k(h_x^l, ρ_k_{y→x}(h_y^l))

        Step 2: Combine across neighborhoods:
                ⋀_{k=1}^n (⊕ from Step 1)

        Step 3: Final update:
                h_x^{l+1} = β(h_x^l, result from Step 2)

        ALGORITHMIC FLOW:
        ================
        1. PARALLEL PROCESSING: Each neighborhood is processed independently by
           its corresponding Conv layer in self.neighborhood_layers

        2. NEIGHBORHOOD AGGREGATION: Within each neighborhood k:
           - Extract edges: edge_index_list[k]
           - Extract attributes: edge_attr_list[k]
           - Apply copresheaf message passing: Conv.forward()
           - Result: ⊕_{y∈N_k(x)} α_k(h_x, ρ_k(h_y))

        3. INTER-NEIGHBORHOOD COMBINATION: Merge results using ⋀:
           - 'concat': [⊕_1, ⊕_2, ..., ⊕_n] (preserves all information)
           - 'sum': ⊕_1 + ⊕_2 + ... + ⊕_n (equal weighting)
           - 'mean': (⊕_1 + ⊕_2 + ... + ⊕_n) / n (normalized)

        4. FINAL UPDATE: Apply β to get h_x^{l+1}

        TOPOLOGICAL INTERPRETATION:
        ==========================
        Each neighborhood k represents a different topological relation:
        - k=0: Adjacency (vertex-vertex connections)
        - k=1: Incidence (vertex-edge connections)
        - k=2: Co-boundary (edge-face connections)
        - etc.

        The parallel processing respects the independence of these relations
        while the combination step ⋀ enables cross-scale information transfer.

        ERROR HANDLING:
        ==============
        - Validates neighborhood count matches edge_index_list length
        - Ensures proper tensor shapes and devices
        - Handles optional edge attributes gracefully

        Args:
            x (Tensor): Node features [num_nodes, in_dim]
            edge_index_list (List[Tensor]): List of edge indices for each neighborhood
            edge_attr_list (List[Tensor], optional): List of edge attributes per neighborhood

        Returns:
            Tensor: Updated node features [num_nodes, out_dim]
        """
        # Use the parent HigherOrderMessagePassing forward method
        out = super().forward(x, edge_index_list, edge_attr_list)

        # Add bias if present (learnable per-node bias)
        if self.bias is not None:
            out = out + self.bias

        return out


# class HigherOrderConv(nn.Module):
#     """
#     Implements higher-order copresheaf-based message passing (Proposition 3).
#
#     For each vertex x, messages are computed for each neighborhood k = 1..n
#     (e.g., adjacency, incidence, co-boundary). Then, these are aggregated using:
#
#         ⓧ({⊕_y α_k(h_x, ρ_k(h_y))}) for k = 1 to n
#
#     Followed by an update function β that combines the result of ⓧ with h_x^l.
#
#     Args:
#         in_dim (int): Input feature dimension F(x)
#         out_dim (int): Output feature dimension F'(x)
#         edge_attr_dim (int): Dim of edge attributes per neighborhood
#         num_neighborhoods (int): Number of neighborhood types
#     """
#     def __init__(self, in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=1, edge_map=Conv):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.num_neighborhoods = num_neighborhoods
#
#         # One copresheaf map per neighborhood (e.g., adjacency, higher-order face maps)
#         self.maps = nn.ModuleList([
#             edge_map(in_dim, out_dim, edge_attr_dim) for _ in range(num_neighborhoods)
#         ])
#
#         # Combination function ⓧ: merges [⊕_1, ⊕_2, ..., ⊕_n] into single vector
#         self.combine = nn.Sequential(
#             nn.Linear(num_neighborhoods * out_dim, out_dim),
#             nn.ReLU()
#         )
#
#         # Final update function β(h_x, ⓧ(...))
#         self.Beta = nn.Sequential(
#             nn.Linear(in_dim + out_dim, out_dim),
#             nn.ReLU()
#         )
#
#     def forward(self, x, edge_index_list, edge_attr_list=None):
#         """
#         Performs higher-order copresheaf message passing for each neighborhood.
#
#         Args:
#             x (Tensor): Node features h_x [num_nodes, in_dim]
#             edge_index_list (List[Tensor]): List of edge_index tensors for n neighborhoods
#             edge_attr_list (List[Tensor or None], optional): List of edge attributes per neighborhood
#
#         Returns:
#             Tensor: Updated node features h_x^{l+1} [num_nodes, out_dim]
#         """
#         # if edge_attr_list is None:
#         #     edge_attr_list = [None] * len(edge_index_list)
#         #
#         # aggregated = []
#         # for i, (edge_index, edge_attr) in enumerate(zip(edge_index_list, edge_attr_list)):
#         #     row, col = edge_index  # edge y → x: src = y, tgt = x
#         #     h_src, h_tgt = x[row], x[col]
#         #     messages = self.maps[i](h_src, h_tgt, edge_attr)
#         #     agg = self.maps[i].aggregate_messages(messages, col, x.size(0))
#         #     aggregated.append(agg)
#         #
#         # combined = torch.cat(aggregated, dim=-1)  # Stack [⊕_1, ..., ⊕_n]
#         # fused = self.combine(combined)            # ⓧ(⊕_1, ..., ⊕_n)
#         # return self.Beta(torch.cat([x, fused], dim=-1))  # β(h_x, ⓧ(...))
#
#
#         if edge_attr_list is None:
#             edge_attr_list = [None] * len(edge_index_list)
#         if len(edge_index_list) != len(self.maps):
#             raise ValueError("num_neighborhoods mismatch with edge_index_list length.")
#
#         aggregated = []
#         for i, (edge_index, edge_attr) in enumerate(zip(edge_index_list, edge_attr_list)):
#             if edge_index.dtype != torch.long:
#                 raise TypeError("edge_index must be torch.long.")
#             if edge_index.size(0) != 2:
#                 raise ValueError("edge_index must be shape [2, num_edges].")
#
#             row, col = edge_index                              # edge y → x: src = y, tgt = x
#             _check_index(row, x.size(0), "row (src)")
#             _check_index(col, x.size(0), "col (tgt)")
#             _same_device(x, edge_index, edge_attr if edge_attr is not None else x)
#
#             h_src, h_tgt = x[row], x[col]
#             messages = self.maps[i](h_src, h_tgt, edge_attr)
#             agg = self.maps[i].aggregate_messages(messages, col, x.size(0))
#             aggregated.append(agg)
#
#         combined = torch.cat(aggregated, dim=-1)    # Stack [⊕_1, ..., ⊕_n]
#         fused = self.combine(combined)              # ⓧ(⊕_1, ..., ⊕_n)
#         return self.Beta(torch.cat([x, fused], dim=-1))  # β(h_x, ⓧ(...))
