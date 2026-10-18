# layer.py
"""

Core architecture classes for CMPNN and CHOMPNN models using Copresheaf-based message passing.

Usage:
-------
>>> import torch
>>> from toposheafx.nn.model import CMPNN, CHOMPNN
>>> x = torch.randn(6, 8)
>>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
>>> batch = torch.tensor([0, 0, 0, 1, 1, 1])
>>> model = CMPNN(in_dim=8, hidden_dim=16, out_dim=3)
>>> out = model(x, [edge_index], batch)

>>> chom = CHOMPNN(in_dim=8, hidden_dim=16, out_dim=3)
>>> out2 = chom(x, [edge_index], batch)


Copresheaf-based message passing components for graph and cell complex neural networks.
Implements local copresheaf morphisms and aggregation rules as described in
"Copresheaf Topological Neural Networks" Hajij et. al.

Usage:
-------
>>> import torch
>>> from toposheafx.base.conv import Conv, HigherOrderConv
>>> x = torch.randn(4, 8)  # 4 nodes, 8-dimensional features
>>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
>>> model = Conv(8, 16)
>>> h_src, h_tgt = x[edge_index[0]], x[edge_index[1]]
>>> messages = model(h_src, h_tgt)
>>> agg = model.aggregate_messages(messages, edge_index[1], num_nodes=4)

>>> # Higher-order message passing
>>> model_ho = HigherOrderConv(8, 16, num_neighborhoods=1)
>>> out = model_ho(x, [edge_index])


Core architecture classes for CMPNN and CHOMPNN models using Copresheaf-based message passing.

These implementations follow the mathematical framework from "Copresheaf Topological Neural Networks"
by Hajij et al., specifically implementing Propositions 1 and 3.
"""

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as segment_softmax

from toposheafx.base.conv import Conv, HigherOrderConv, continuous_simplicial_conv
from toposheafx.base.message_passing import (
    CrossDimensionalMessagePassing,
    HigherOrderMessagePassing,
)
from toposheafx.data.simplicial import SimplicialFeatures
from toposheafx.utils.guards import _check_index, _same_device


class CopresheafLayer(nn.Module):
    """
    Edge-specific Copresheaf Layer (Section 3.2 of the paper).

    This implements a special case where each edge has its own learnable copresheaf morphism,
    allowing for maximum expressiveness at the cost of more parameters.

    Mathematical Framework:
    - For each edge (y→x), we have a unique morphism ρ_{y→x}: F(y) → F(x)
    - This generalizes the shared morphism approach in standard copresheaf networks

    Args:
        in_channels (int): Dimension of input stalks F(y)
        out_channels (int): Dimension of output stalks F(x)
        edge_index (LongTensor): [2, num_edges] tensor of edge indices
        edge_map_factory (Callable): Factory function for edge-specific morphisms
    """

    def __init__(self, in_channels, out_channels, edge_index, edge_map_factory):
        super().__init__()
        if edge_index.dtype != torch.long or edge_index.size(0) != 2:
            raise ValueError("edge_index must be [2, num_edges] with dtype torch.long.")
        self.edge_index = edge_index
        num_edges = edge_index.size(1)

        # Create edge-specific copresheaf morphisms (Equation 3 in paper)
        # Each morphism ρ_e: F(σ(e)) → F(τ(e)) where σ is source, τ is target
        self.edge_mlps = nn.ModuleList(
            [edge_map_factory(in_channels, out_channels) for _ in range(num_edges)]
        )

    def forward(self, x):
        """
        Forward pass implementing edge-specific copresheaf message passing.

        For each edge e = (y→x):
        1. Apply edge-specific morphism: ρ_e(h_y)
        2. Aggregate at target: ⊕_{y∈N(x)} ρ_{(y,x)}(h_y)

        Args:
            x (Tensor): Node features [num_nodes, in_channels]

        Returns:
            Tensor: Updated features [num_nodes, out_channels]
        """
        row, col = self.edge_index  # source, target
        _check_index(row, x.size(0), "row (src)")
        _check_index(col, x.size(0), "col (tgt)")
        _same_device(x, self.edge_index)

        # Apply edge-specific copresheaf morphisms
        messages = []
        for k in range(self.edge_index.size(1)):
            source_node = row[k]
            mlp = self.edge_mlps[k]
            if next(mlp.parameters()).device != x.device:
                mlp.to(x.device)
            # Apply morphism ρ_{y→x}: F(y) → F(x)
            message = mlp(x[source_node].unsqueeze(0))
            messages.append(message)

        messages = torch.cat(messages, dim=0)

        # Aggregate using ⊕ operator (Proposition 1)
        out = scatter(messages, col, dim=0, dim_size=x.size(0), reduce="sum")
        return out


class CMPNNLayer(nn.Module):
    """
    CMPNN Layer: Implements Proposition 1 from the copresheaf paper.

    Mathematical Framework (Proposition 1):
    h_x^{l+1} = β(h_x^l, ⊕_{(y→x)∈E} α(h_x^l, ρ_{y→x}(h_y^l)))

    Where:
    - ρ_{y→x}: Copresheaf morphism (transport map) from stalk F(y) to F(x)
    - α: Message function combining target features with transported source
    - ⊕: Permutation-invariant aggregation operator
    - β: Update function integrating old and new features

    This layer extends to multiple neighborhoods for higher-order complexes.
    """

    def __init__(self, in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_neighborhoods = num_neighborhoods

        # Create copresheaf maps for each neighborhood type
        # Each implements the full message passing from Proposition 1
        self.copresheaf_maps = nn.ModuleList(
            [Conv(in_dim, out_dim, edge_attr_dim, bias=False) for _ in range(num_neighborhoods)]
        )

    def forward(self, x, edge_index_list, edge_attr_list=None):
        """
        Forward pass implementing copresheaf message passing over multiple neighborhoods.

        For each neighborhood k:
        1. Apply copresheaf message passing via Conv layer
        2. Conv internally handles: ρ, α, ⊕, and β operations

        The results are averaged across neighborhoods.

        Args:
            x (Tensor): Node features [num_nodes, in_dim]
            edge_index_list (list): List of edge indices per neighborhood
            edge_attr_list (list, optional): Edge attributes per neighborhood

        Returns:
            Tensor: Updated features [num_nodes, out_dim]
        """
        if edge_attr_list is None:
            edge_attr_list = [None] * self.num_neighborhoods

        all_updates = []
        for i in range(self.num_neighborhoods):
            edge_index = edge_index_list[i]
            edge_attr = edge_attr_list[i]

            # Conv layer handles the complete copresheaf message passing:
            # h_x^{l+1} = β(h_x^l, ⊕_{(y→x)∈E} α(h_x^l, ρ_{y→x}(h_y^l)))
            updated = self.copresheaf_maps[i](x, edge_index, edge_attr)
            all_updates.append(updated)

        # Average across neighborhoods (can be replaced with other combination strategies)
        return sum(all_updates) / len(all_updates)


class CHOMPNNLayer(nn.Module):
    """
    CHOMPNN Layer: Implements Proposition 3 from the copresheaf paper.

    Mathematical Framework (Proposition 3):
    h_x^{l+1} = β(h_x^l, ⋀_{k=1}^n ⊕_{y∈N_k(x)} α_k(h_x^l, ρ_k_{y→x}(h_y^l)))

    Where:
    - Multiple neighborhoods N_k(x) capture different topological relations
    - Each neighborhood has its own ρ_k, α_k operators
    - ⋀ combines information across neighborhoods
    - β performs final update

    This enables learning on higher-order topological structures like simplicial
    complexes and cell complexes.
    """

    def __init__(self, in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=1):
        super().__init__()
        # HigherOrderConv implements the full Proposition 3
        self.model = HigherOrderConv(
            in_dim=in_dim,
            out_dim=out_dim,
            edge_attr_dim=edge_attr_dim,
            num_neighborhoods=num_neighborhoods,
            combine_method="concat",  # ⋀ operator implementation
            bias=True,
        )

    def forward(self, x, edge_index_list, edge_attr_list=None):
        """
        Forward pass for higher-order copresheaf message passing.

        Delegates to HigherOrderConv which implements:
        1. Parallel processing of each neighborhood with ρ_k, α_k, ⊕_k
        2. Inter-neighborhood combination via ⋀
        3. Final update via β

        Args:
            x (Tensor): Node features [num_nodes, in_dim]
            edge_index_list (List[Tensor]): Edge indices per neighborhood
            edge_attr_list (List[Tensor], optional): Edge attributes

        Returns:
            Tensor: Updated features [num_nodes, out_dim]
        """
        return self.model(x, edge_index_list, edge_attr_list)


class CopresheafTransformerLayer(nn.Module):
    """
    Attention-based Copresheaf Transformer Layer (Algorithm 1, Appendix E).

    This layer combines attention mechanisms with copresheaf message passing,
    allowing the network to learn importance weights for different messages.

    Mathematical Framework:
    - Attention weights: α_{xy} = softmax(Q_x · K_y / √d)
    - Weighted message passing: h_x^{l+1} = β(h_x^l, ⊕_{y∈N(x)} α_{xy} · ρ_{y→x}(h_y^l))

    The attention mechanism provides a learnable, context-dependent aggregation
    that generalizes the fixed aggregation schemes.
    """

    def __init__(self, in_dim, out_dim, edge_attr_dim=0, heads=1, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Copresheaf morphism for message computation
        self.copresheaf_map = Conv(in_dim, out_dim, edge_attr_dim=edge_attr_dim, bias=False)

        # Attention components (Q, K, V transformations)
        self.key = nn.Linear(in_dim, out_dim * heads)
        self.query = nn.Linear(in_dim, out_dim * heads)
        self.value = nn.Linear(in_dim, out_dim * heads)

        # Output projection to combine heads
        self.out_proj = nn.Linear(out_dim * heads, out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with attention-weighted copresheaf message passing.

        Steps:
        1. Compute attention scores using Q, K projections
        2. Apply copresheaf morphism to get messages
        3. Weight messages by attention scores
        4. Aggregate and project to output dimension

        Args:
            x (Tensor): Node features [num_nodes, in_dim]
            edge_index (LongTensor): Edge indices [2, num_edges]
            edge_attr (Tensor, optional): Edge attributes

        Returns:
            Tensor: Updated features [num_nodes, out_dim]
        """
        row, col = edge_index  # source, target
        h_src, h_tgt = x[row], x[col]

        # Compute multi-head attention scores
        k = self.key(h_src).view(-1, self.heads, self.out_dim)
        q = self.query(h_tgt).view(-1, self.heads, self.out_dim)

        # Scaled dot-product attention (Equation 7 in Appendix E)
        attn_logits = (q * k).sum(dim=-1) / (self.out_dim**0.5)

        # Normalize attention per target node
        logits_flat = attn_logits.reshape(-1)
        index_flat = col.repeat_interleave(self.heads)
        alpha = segment_softmax(logits_flat, index_flat, num_nodes=x.size(0))
        alpha = alpha.view(-1, self.heads)
        alpha = self.dropout(alpha)

        # Apply copresheaf morphism to get base messages
        # This uses Conv which implements: ρ_{y→x}(h_y) with context h_x
        messages_base = self.copresheaf_map(x, edge_index, edge_attr)

        # Transform source features through value projection
        v = self.value(h_src).view(-1, self.heads, self.out_dim)

        # Combine copresheaf messages with attention-weighted values
        # This creates a richer message that incorporates both structural
        # transport (via copresheaf) and learned importance (via attention)
        messages_attended = v * alpha.unsqueeze(-1)

        # Aggregate messages to target nodes
        messages_flat = messages_attended.reshape(-1, self.heads * self.out_dim)
        aggregated = scatter(messages_flat, col, dim=0, dim_size=x.size(0), reduce="sum")

        # Add copresheaf structural bias
        aggregated = aggregated + messages_base.repeat(1, self.heads)

        # Final projection
        return self.out_proj(aggregated)


class SimplicialComplexCopresheafLayer(nn.Module):
    """
    Copresheaf message passing layer specialized for simplicial complexes.

    This layer implements Proposition 3 with neighborhoods specific to
    simplicial topology:
    - Adjacency (vertex-vertex)
    - Incidence (vertex-edge)
    - Co-boundary (edge-triangle)
    """

    def __init__(
        self, in_dim, out_dim, num_neighborhoods=3, aggregation="sum", combine_method="concat"
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_neighborhoods = num_neighborhoods

        # Use HigherOrderMessagePassing for multi-neighborhood processing
        self.copresheaf_layer = HigherOrderMessagePassing(
            in_dim=in_dim,
            out_dim=out_dim,
            num_neighborhoods=num_neighborhoods,
            aggr=aggregation,
            combine_method=combine_method,
        )

        # Neighborhood-specific aggregation strategies
        self.neighborhood_configs = {
            0: {"name": "adjacency", "aggr": "mean"},  # Normalize by degree
            1: {"name": "incidence", "aggr": "sum"},  # Accumulate from edges
            2: {"name": "coboundary", "aggr": "max"},  # Dominant triangle feature
        }

    def forward(self, x, edge_index_list, simplex_features=None):
        """
        Forward pass with copresheaf message passing.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index_list: List of edge indices for each neighborhood
            simplex_features: Optional higher-order simplex features

        Returns:
            Updated node features [num_nodes, out_dim]
        """
        # Apply copresheaf message passing across all neighborhoods
        out = self.copresheaf_layer(x, edge_index_list)

        return out

    def get_aggregation_info(self):
        """Get information about neighborhood-specific aggregations."""
        return self.copresheaf_layer.get_aggregation_summary()


class MultiDimensionalSimplicialCopresheafLayer(nn.Module):
    """
    Copresheaf layer supporting message passing across simplex dimensions.
    """

    def __init__(self, feature_dims, out_dims, neighborhoods_config):
        """
        Args:
            feature_dims: dict {dim: feature_size} for each dimension
            out_dims: dict {dim: output_size} for each dimension
            neighborhoods_config: list of (name, source_dim, target_dim) tuples
        """
        super().__init__()

        self.feature_dims = feature_dims
        self.out_dims = out_dims

        # Create cross-dimensional message passing modules
        self.message_passers = nn.ModuleDict()

        for name, source_dim, target_dim in neighborhoods_config:
            self.message_passers[name] = CrossDimensionalMessagePassing(
                source_dim=feature_dims[source_dim],
                target_dim=feature_dims[target_dim],
                out_dim=out_dims[target_dim],
            )

    def forward(self, features, neighborhoods):
        """
        Args:
            features: SimplicialFeatures object
            neighborhoods: dict from extract_cross_dimensional_neighborhoods

        Returns:
            SimplicialFeatures with updated features
        """
        # Initialize output for each dimension
        updated_features = {}

        # Group neighborhoods by target dimension
        by_target = {}
        for name, config in neighborhoods.items():
            target = config["target_dim"]
            if target not in by_target:
                by_target[target] = []
            by_target[target].append((name, config))

        # Update each dimension
        for target_dim, dim_neighborhoods in by_target.items():
            messages_list = []

            for name, config in dim_neighborhoods:
                source_dim = config["source_dim"]
                edge_index = config["edge_index"]

                if edge_index.shape[1] == 0:  # Skip empty neighborhoods
                    continue

                # Apply cross-dimensional message passing
                x_source = features[source_dim]
                x_target = features[target_dim]

                messages = self.message_passers[name](x_source, x_target, edge_index)
                messages_list.append(messages)

            # Combine messages from all neighborhoods targeting this dimension
            if messages_list:
                updated_features[target_dim] = sum(messages_list) / len(messages_list)
            else:
                updated_features[target_dim] = features[target_dim]

        return SimplicialFeatures.from_dict(updated_features)


# Continuous Hodge Laplacians for Copresheaves working with
# restriction maps between different levels of a simplicial complex
# class ContinuousCopresheafLayer:
#     def __init__(self, k_simplex_dim, t_d, t_u):
#         self.t_d = nn.Parameter(torch.tensor(t_d))  # Learnable time params
#         self.t_u = nn.Parameter(torch.tensor(t_u))
#         self.L_k_down = None  # Lower Hodge Laplacian
#         self.L_k_up = None  # Upper Hodge Laplacian
#
#     def forward(self, X_k, X_k_down, X_k_up, B_k, B_k_plus_1):
#         # Compute Hodge Laplacians
#         self.L_k_down = B_k.T @ B_k
#         self.L_k_up = B_k_plus_1 @ B_k_plus_1.T
#
#         # Continuous diffusion (COSIMO-style)
#         # Use matrix exponential: e^(-t*L)
#         diff_down = torch.matrix_exp(-self.t_d * self.L_k_down) @ X_k_down
#         diff_up = torch.matrix_exp(-self.t_u * self.L_k_up) @ X_k_up
#         diff_joint = (torch.matrix_exp(-self.t_d * self.L_k_down) +
#                       torch.matrix_exp(-self.t_u * self.L_k_up)) @ X_k
#
#         return diff_down, diff_joint, diff_up


#
#    !! Improperly handles boundary operator for 0 cells
#    !! Improperly handles transport maps fro copresheafs
#
# class ContinuousCopresheafLayer(nn.Module):
#     """
#     Continuous copresheaf layer integrating COSIMO methodology with copresheaf theory.
#
#     This layer combines:
#     1. Copresheaf morphisms (restriction maps between stalks)
#     2. Continuous Hodge-Laplacian dynamics (COSIMO approach)
#     3. Multi-scale information flow (lower/upper/joint dynamics)
#
#     Mathematical Framework:
#     For each k-simplex x with neighborhoods:
#     - Lower: (k-1)-simplices → k-simplices (faces)
#     - Upper: (k+1)-simplices → k-simplices (cofaces)
#
#     Dynamics:
#     - Independent diffusion: e^(-t_d * L_k,d) @ x_k,d and e^(-t_u * L_k,u) @ x_k,u
#     - Joint diffusion: (e^(-t_d * L_k,d) + e^(-t_u * L_k,u)) @ x_k
#     - Final: x_k' = φ(x_k,d' + x_k' + x_k,u')
#
#     Args:
#         in_dim: Input feature dimension
#         out_dim: Output feature dimension
#         k_simplex_dim: Dimension of simplices (0=nodes, 1=edges, 2=triangles)
#         t_d_init: Initial lower time parameter
#         t_u_init: Initial upper time parameter
#         learnable_t: Whether time parameters are learnable
#         K: Number of eigenvectors for approximation
#         combine_method: How to combine dynamics ('sum', 'concat', 'attention')
#     """
#
#     def __init__(
#             self,
#             in_dim: int,
#             out_dim: int,
#             k_simplex_dim: int = 1,
#             t_d_init: float = 0.1,
#             t_u_init: float = 0.1,
#             learnable_t: bool = True,
#             K: Optional[int] = None,
#             combine_method: str = 'sum'
#     ):
#         super().__init__()
#
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.k_simplex_dim = k_simplex_dim
#         self.K = K
#         self.combine_method = combine_method
#
#         # Learnable time parameters
#         if learnable_t:
#             self.t_d = nn.Parameter(torch.tensor(t_d_init, dtype=torch.float32))
#             self.t_u = nn.Parameter(torch.tensor(t_u_init, dtype=torch.float32))
#         else:
#             self.register_buffer('t_d', torch.tensor(t_d_init, dtype=torch.float32))
#             self.register_buffer('t_u', torch.tensor(t_u_init, dtype=torch.float32))
#
#         # Projection layers for each dynamics component
#         self.proj_down = nn.Linear(in_dim, out_dim)
#         self.proj_up = nn.Linear(in_dim, out_dim)
#         self.proj_joint = nn.Linear(in_dim, out_dim)
#
#         # Combination layer
#         if combine_method == 'concat':
#             self.combine = nn.Sequential(
#                 nn.Linear(3 * out_dim, out_dim),
#                 nn.ReLU()
#             )
#         elif combine_method == 'attention':
#             self.attention = nn.MultiheadAttention(out_dim, num_heads=4, batch_first=True)
#         elif combine_method == 'sum':
#             self.combine = None
#         else:
#             raise ValueError(f"Unknown combine_method: {combine_method}")
#
#         # Final update function (copresheaf Beta operator)
#         self.update_fn = nn.Sequential(
#             nn.Linear(in_dim + out_dim, out_dim),
#             nn.ReLU(),
#             nn.Linear(out_dim, out_dim)
#         )
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         """Initialize parameters."""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#
#         """Initialize all learnable parameters."""
#         # Initialize projection layers
#         nn.init.xavier_uniform_(self.proj_down.weight)
#         nn.init.zeros_(self.proj_down.bias)
#
#         nn.init.xavier_uniform_(self.proj_up.weight)
#         nn.init.zeros_(self.proj_up.bias)
#
#         nn.init.xavier_uniform_(self.proj_joint.weight)
#         nn.init.zeros_(self.proj_joint.bias)
#
#         # Initialize combination layer if it exists
#         if self.combine is not None:
#             for module in self.combine.modules():
#                 if isinstance(module, nn.Linear):
#                     nn.init.xavier_uniform_(module.weight)
#                     if module.bias is not None:
#                         nn.init.zeros_(module.bias)
#
#         # Initialize attention if used
#         if hasattr(self, 'attention'):
#             # MultiheadAttention initializes itself, but we can reset if needed
#             pass
#
#         # Initialize update function
#         for module in self.update_fn.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#
#         # Ensure time parameters are valid (should already be from __init__)
#         if hasattr(self, 't_d') and (torch.isnan(self.t_d).any() or torch.isinf(self.t_d).any()):
#             print(f"⚠️ WARNING: t_d is NaN/Inf in {self.__class__.__name__}. Resetting.")
#             self.t_d.data = torch.tensor(0.1, dtype=torch.float32)
#
#         if hasattr(self, 't_u') and (torch.isnan(self.t_u).any() or torch.isinf(self.t_u).any()):
#             print(f"⚠️ WARNING: t_u is NaN/Inf in {self.__class__.__name__}. Resetting.")
#             self.t_u.data = torch.tensor(0.1, dtype=torch.float32)
#
#     def forward(
#             self,
#             X_k: torch.Tensor,
#             X_k_down: torch.Tensor,
#             X_k_up: torch.Tensor,
#             B_k: torch.Tensor,
#             B_k_plus_1: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         """
#         Forward pass implementing continuous copresheaf dynamics.
#
#         Args:
#             X_k: Features on k-simplices [num_k_simplices, in_dim]
#             X_k_down: Features projected from (k-1)-simplices [num_k_simplices, in_dim]
#             X_k_up: Features projected from (k+1)-simplices [num_k_simplices, in_dim]
#             B_k: Boundary operator k [num_k-1_simplices, num_k_simplices]
#             B_k_plus_1: Boundary operator k+1 (optional)
#
#         Returns:
#             Updated features [num_k_simplices, out_dim]
#         """
#         # Compute Hodge Laplacians
#         #  CHECK FOR NONE BEFORE COMPUTING LAPLACIANS
#         if B_k is not None:
#             L_k_down = B_k.T @ B_k
#         else:
#             # Dimension 0: no lower Laplacian, use identity
#             L_k_down = torch.eye(X_k.size(0), device=X_k.device, dtype=X_k.dtype)
#
#         if B_k_plus_1 is not None:
#             L_k_up = B_k_plus_1 @ B_k_plus_1.T
#         else:
#             # Top dimension: no upper Laplacian, use identity
#             L_k_up = torch.eye(X_k.size(0), device=X_k.device, dtype=X_k.dtype)
#
#         # Independent lower dynamics: e^(-t_d * L_k,d) @ X_k,d
#         if B_k is not None:
#             diff_down = continuous_simplicial_conv(X_k_down, L_k_down, self.t_d, self.K)
#             diff_down = self.proj_down(diff_down)
#         else:
#             # Dimension 0: no lower neighbors, use self-diffusion
#             diff_down = continuous_simplicial_conv(X_k, L_k_down, self.t_d, self.K)
#             diff_down = self.proj_down(diff_down)
#
#         # Independent upper dynamics
#         if B_k_plus_1 is not None:
#             diff_up = continuous_simplicial_conv(X_k_up, L_k_up, self.t_u, self.K)
#             diff_up = self.proj_up(diff_up)
#         else:
#             # Top dimension: no upper neighbors
#             diff_up = torch.zeros_like(diff_down)
#
#         # Joint dynamics: combines both spaces
#         diff_joint = continuous_simplicial_conv(X_k, L_k_down, self.t_d, self.K)
#         if B_k_plus_1 is not None:
#             diff_joint = diff_joint + continuous_simplicial_conv(X_k, L_k_up, self.t_u, self.K)
#         diff_joint = self.proj_joint(diff_joint)
#
#         # Combine dynamics components
#         if self.combine_method == 'sum':
#             combined = diff_down + diff_joint + diff_up
#         elif self.combine_method == 'concat':
#             combined = torch.cat([diff_down, diff_joint, diff_up], dim=-1)
#             combined = self.combine(combined)
#         elif self.combine_method == 'attention':
#             stacked = torch.stack([diff_down, diff_joint, diff_up], dim=1)
#             combined, _ = self.attention(stacked, stacked, stacked)
#             combined = combined.mean(dim=1)
#
#         # Final update (Beta operator from copresheaf)
#         if X_k.size(-1) == combined.size(-1):
#             out = self.update_fn(torch.cat([X_k, combined], dim=-1))
#         else:
#             X_k_proj = nn.Linear(X_k.size(-1), combined.size(-1), device=X_k.device)(X_k)
#             out = self.update_fn(torch.cat([X_k_proj, combined], dim=-1))
#
#         return out
#
#     def get_effective_time_constants(self) -> dict:
#         """Return current time constants for monitoring."""
#         return {
#             't_d': self.t_d.item(),
#             't_u': self.t_u.item()
#         }
#
#     def clamp_time_parameters(self, min_val: float = 1e-6, max_val: float = 10.0):
#         """
#         Clamp time parameters to safe range.
#
#         This should be called after optimizer.step() to prevent:
#         - Negative values
#         - Values too close to zero (numerical instability)
#         - Values too large (over-smoothing)
#
#         Args:
#             min_val: Minimum allowed value (default: 1e-6)
#             max_val: Maximum allowed value (default: 10.0)
#         """
#         with torch.no_grad():
#             if hasattr(self, 't_d'):
#                 self.t_d.data = self.t_d.data.clamp(min=min_val, max=max_val)
#                 if torch.isnan(self.t_d).any():
#                     self.t_d.data = torch.tensor(self._t_d_init, dtype=torch.float32, device=self.t_d.device)
#
#             if hasattr(self, 't_u'):
#                 self.t_u.data = self.t_u.data.clamp(min=min_val, max=max_val)
#                 if torch.isnan(self.t_u).any():
#                     self.t_u.data = torch.tensor(self._t_u_init, dtype=torch.float32, device=self.t_u.device)
#


class ContinuousCopresheafLayer(nn.Module):
    """
    Integrates copresheaf morphisms with continuous Hodge-Laplacian dynamics.

    Combines:
    1. Copresheaf transport maps ρ_y→x: F(y) → F(x) (Proposition 1)
    2. COSIMO continuous dynamics (Proposition 4.1, Equation 10)

    Architecture per layer:
        1. Apply ρ morphisms to transported features
        2. Compute continuous diffusion via Hodge Laplacians
        3. Message function α combines current + transported
        4. Update function β produces new representations
        5. Copresheaf morphisms ρ (transport maps)
        6. Learnable Θ projections (COSIMO Eq. 10)
        7. Learnable Ψ projections (COSIMO Eq. 10)
        8. Message function α (copresheaf paper)
    -
    Mathematically:
        1. Apply copresheaf morphisms first: ρ_down_input and X_k_down_transported
        2. Compute Laplacians: L_k_down = B_k.T @ B_k
        3. Apply continuous diffusion to transported features: lower_diffusion = e^(-t_d * L_k_down) @ X_k_down_transported
        4. Project features Use Θ and Ψ projections (COSIMO Eq. 10):
            a. lower_features = Θ_down(lower_diffusion)
            b. joint_features = Ψ_down(joint_diffusion)
        5. Combine: sum or concat or attention
        6. Message function α: messages = α(X_k, combined_features)
        7. Update: β(X_k, combined)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k_simplex_dim: int,
        t_d_init: float = 0.1,
        t_u_init: float = 0.1,
        learnable_t: bool = True,
        K: Optional[int] = None,
        use_morphisms: bool = True,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k_simplex_dim = k_simplex_dim
        self.use_morphisms = use_morphisms
        self.K = K

        # COSIMO time parameters
        self._create_time_parameters(t_d_init, t_u_init, learnable_t)

        # COPRESHEAF MORPHISMS (Proposition 1)
        if use_morphisms:
            # ρ_down: Transport from (k-1)-simplices to k-simplices
            self.rho_down = nn.Sequential(
                nn.Linear(2 * in_dim, out_dim),  # [h_k, h_{k-1}]
                nn.Tanh(),  # Smooth transport
            )

            # ρ_up: Transport from (k+1)-simplices to k-simplices
            self.rho_up = nn.Sequential(
                nn.Linear(2 * in_dim, out_dim),  # [h_k, h_{k+1}]
                nn.Tanh(),
            )

        # COSIMO projections (Equation 10: Θ and Ψ)
        self.theta_down = nn.Linear(in_dim, out_dim)  # Θ^l_{k,d}
        self.theta_up = nn.Linear(in_dim, out_dim)  # Θ^l_{k,u}
        self.psi_down = nn.Linear(in_dim, out_dim)  # Ψ^l_{k,d}
        self.psi_up = nn.Linear(in_dim, out_dim)  # Ψ^l_{k,u}

        # MESSAGE AND UPDATE FUNCTIONS (Copresheaf α and β)
        self.message_fn = nn.Sequential(
            nn.Linear(out_dim + out_dim, out_dim),  # [h_k, aggregated_messages]
            nn.ReLU(),
        )

        self.update_fn = nn.Sequential(
            nn.Linear(out_dim + out_dim, out_dim),  # [h_k, combined_features]
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

        self.reset_parameters()

    def _create_time_parameters(self, t_d_init, t_u_init, learnable_t):
        """Create time parameters with validation."""
        if learnable_t:
            self.t_d = nn.Parameter(torch.tensor(t_d_init, dtype=torch.float32))
            self.t_u = nn.Parameter(torch.tensor(t_u_init, dtype=torch.float32))
        else:
            self.register_buffer("t_d", torch.tensor(t_d_init, dtype=torch.float32))
            self.register_buffer("t_u", torch.tensor(t_u_init, dtype=torch.float32))

    def forward(
        self,
        X_k: torch.Tensor,
        X_k_down: torch.Tensor,
        X_k_up: torch.Tensor,
        B_k: Optional[torch.Tensor] = None,
        B_k_plus_1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass combining copresheaf morphisms with continuous dynamics.

        Process:
        1. Apply copresheaf morphisms ρ to transported features
        2. Compute continuous diffusion using Hodge Laplacians
        3. Combine via message function α
        4. Update via β
        """

        # STEP 1: COPRESHEAF MORPHISMS
        if self.use_morphisms and B_k is not None:
            # Transport features with context-dependent morphisms
            # ρ_down(h_k, h_{k-1}): considers target context
            rho_down_input = torch.cat([X_k, X_k_down], dim=-1)
            X_k_down_transported = self.rho_down(rho_down_input)
        else:
            X_k_down_transported = X_k_down

        if self.use_morphisms and B_k_plus_1 is not None:
            rho_up_input = torch.cat([X_k, X_k_up], dim=-1)
            X_k_up_transported = self.rho_up(rho_up_input)
        else:
            X_k_up_transported = X_k_up

        # STEP 2: COMPUTE HODGE LAPLACIANS (Handle None case)
        if B_k is not None:
            L_k_down = B_k.T @ B_k
        else:
            L_k_down = torch.eye(X_k.size(0), device=X_k.device, dtype=X_k.dtype)

        if B_k_plus_1 is not None:
            L_k_up = B_k_plus_1 @ B_k_plus_1.T
        else:
            L_k_up = torch.eye(X_k.size(0), device=X_k.device, dtype=X_k.dtype)

        # STEP 3: CONTINUOUS DYNAMICS (COSIMO Equation 10)
        # X^l_k = σ(e^{-t_d L_{k,d}} X^{l-1}_{k,d} Θ^l_{k,d} +
        #          e^{-t_u L_{k,u}} X^{l-1}_{k,u} Θ^l_{k,u} +
        #          e^{-t_d L_{k,d}} X^{l-1}_k Ψ^l_{k,d} +
        #          e^{-t_u L_{k,u}} X^{l-1}_k Ψ^l_{k,u})

        # Independent lower dynamics
        if B_k is not None:
            lower_diffusion = continuous_simplicial_conv(
                X_k_down_transported, L_k_down, self.t_d, self.K
            )
            lower_features = self.theta_down(lower_diffusion)
        else:
            lower_features = torch.zeros(X_k.size(0), self.out_dim, device=X_k.device)

        # Independent upper dynamics
        if B_k_plus_1 is not None:
            upper_diffusion = continuous_simplicial_conv(
                X_k_up_transported, L_k_up, self.t_u, self.K
            )
            upper_features = self.theta_up(upper_diffusion)
        else:
            upper_features = torch.zeros(X_k.size(0), self.out_dim, device=X_k.device)

        # Joint dynamics on current features
        joint_lower = continuous_simplicial_conv(X_k, L_k_down, self.t_d, self.K)
        joint_lower_features = self.psi_down(joint_lower)

        joint_upper = continuous_simplicial_conv(X_k, L_k_up, self.t_u, self.K)
        joint_upper_features = self.psi_up(joint_upper)

        # Combine all components
        combined = lower_features + upper_features + joint_lower_features + joint_upper_features

        # STEP 4: MESSAGE FUNCTION (α)
        # Combines current state with aggregated diffused features
        messages = self.message_fn(torch.cat([X_k, combined], dim=-1))

        # STEP 5: UPDATE FUNCTION (β)
        # Final update with residual connection
        output = self.update_fn(torch.cat([X_k, messages], dim=-1))

        return output

    def clamp_time_parameters(self, min_val=1e-4, max_val=10.0):
        """Clamp time parameters to valid range."""
        if isinstance(self.t_d, nn.Parameter):
            self.t_d.data = torch.clamp(self.t_d.data, min_val, max_val)
            self.t_u.data = torch.clamp(self.t_u.data, min_val, max_val)

    def reset_parameters(self):
        """Initialize parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class ContinuousCHOMPNNLayer(nn.Module):
    """
    Continuous Higher-Order Message Passing layer combining:
    - Multiple neighborhoods (Proposition 3)
    - Continuous Hodge-Laplacian dynamics (COSIMO)
    - Copresheaf morphisms

    This represents the full integration of all three papers' methodologies.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_neighborhoods: int = 3,
        t_d_init: float = 0.1,
        t_u_init: float = 0.1,
        learnable_t: bool = True,
        K: Optional[int] = None,
        combine_method: str = "concat",
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_neighborhoods = num_neighborhoods
        self.K = K
        self.combine_method = combine_method

        # Store initialization values for potential reset
        self._t_d_init = t_d_init
        self._t_u_init = t_u_init
        self._learnable_t = learnable_t

        # Create time parameters with explicit dtype and validation
        self._create_time_parameters(t_d_init, t_u_init, learnable_t)

        # One continuous copresheaf layer per neighborhood
        self.neighborhood_layers = nn.ModuleList(
            [
                ContinuousCopresheafLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    k_simplex_dim=k,
                    t_d_init=t_d_init,
                    t_u_init=t_u_init,
                    learnable_t=learnable_t,
                    K=K,
                    combine_method="sum",  # Inner combination
                )
                for k in range(num_neighborhoods)
            ]
        )

        # Cross-neighborhood combination
        if combine_method == "concat":
            self.combine = nn.Sequential(nn.Linear(num_neighborhoods * out_dim, out_dim), nn.ReLU())
        elif combine_method == "sum":
            self.combine = None
        elif combine_method == "attention":
            self.combine = nn.MultiheadAttention(embed_dim=out_dim, num_heads=4, batch_first=True)
        else:
            raise ValueError(
                f"Unknown combine_method: {combine_method}. Must be 'concat', 'sum', or 'attention'"
            )

        # Initialize all parameters at the end
        self.reset_parameters()

    def _create_time_parameters(self, t_d_init: float, t_u_init: float, learnable_t: bool):
        """
        Safely create time parameters with validation.

        This method ensures:
        1. Explicit float32 dtype
        2. Values are not NaN or Inf
        3. Values are positive
        4. Proper parameter/buffer registration
        """
        # Validate input values
        if not isinstance(t_d_init, (int, float)) or np.isnan(t_d_init) or np.isinf(t_d_init):
            print(f"⚠️ WARNING: Invalid t_d_init={t_d_init}, using default 0.1")
            t_d_init = 0.1

        if not isinstance(t_u_init, (int, float)) or np.isnan(t_u_init) or np.isinf(t_u_init):
            print(f"⚠️ WARNING: Invalid t_u_init={t_u_init}, using default 0.1")
            t_u_init = 0.1

        # Ensure positive values
        t_d_init = max(float(t_d_init), 1e-6)
        t_u_init = max(float(t_u_init), 1e-6)

        # Create parameters with explicit dtype
        if learnable_t:
            self.t_d = nn.Parameter(torch.tensor(t_d_init, dtype=torch.float32), requires_grad=True)
            self.t_u = nn.Parameter(torch.tensor(t_u_init, dtype=torch.float32), requires_grad=True)
        else:
            # Use buffers for non-learnable parameters
            self.register_buffer("t_d", torch.tensor(t_d_init, dtype=torch.float32))
            self.register_buffer("t_u", torch.tensor(t_u_init, dtype=torch.float32))

        # Immediate validation
        self._validate_time_parameters()

    def _validate_time_parameters(self):
        """
        Validate time parameters and fix if necessary.

        This catches issues from:
        - Improper initialization
        - Gradient updates that produce NaN
        - Numerical instability
        """
        if hasattr(self, "t_d"):
            if torch.isnan(self.t_d).any():
                print(
                    f"⚠️ CRITICAL: t_d is NaN in {self.__class__.__name__}! Resetting to {self._t_d_init}"
                )
                self.t_d.data = torch.tensor(
                    self._t_d_init, dtype=torch.float32, device=self.t_d.device
                )
            elif torch.isinf(self.t_d).any():
                print(
                    f"⚠️ CRITICAL: t_d is Inf in {self.__class__.__name__}! Resetting to {self._t_d_init}"
                )
                self.t_d.data = torch.tensor(
                    self._t_d_init, dtype=torch.float32, device=self.t_d.device
                )
            elif self.t_d.item() <= 0:
                print(
                    f"⚠️ WARNING: t_d is non-positive ({self.t_d.item()})! Resetting to {self._t_d_init}"
                )
                self.t_d.data = torch.tensor(
                    self._t_d_init, dtype=torch.float32, device=self.t_d.device
                )

        if hasattr(self, "t_u"):
            if torch.isnan(self.t_u).any():
                print(
                    f"⚠️ CRITICAL: t_u is NaN in {self.__class__.__name__}! Resetting to {self._t_u_init}"
                )
                self.t_u.data = torch.tensor(
                    self._t_u_init, dtype=torch.float32, device=self.t_u.device
                )
            elif torch.isinf(self.t_u).any():
                print(
                    f"⚠️ CRITICAL: t_u is Inf in {self.__class__.__name__}! Resetting to {self._t_u_init}"
                )
                self.t_u.data = torch.tensor(
                    self._t_u_init, dtype=torch.float32, device=self.t_u.device
                )
            elif self.t_u.item() <= 0:
                print(
                    f"⚠️ WARNING: t_u is non-positive ({self.t_u.item()})! Resetting to {self._t_u_init}"
                )
                self.t_u.data = torch.tensor(
                    self._t_u_init, dtype=torch.float32, device=self.t_u.device
                )

    def reset_parameters(self):
        """
        Initialize all learnable parameters.

        Initialization strategy:
        1. Time parameters: Already initialized in _create_time_parameters()
        2. Neighborhood layers: Recursively call their reset_parameters()
        3. Combination layers: Xavier uniform for weights, zeros for biases

        This ensures a consistent initialization across the entire layer.
        """
        # Reset neighborhood layers (will initialize their time parameters)
        for layer in self.neighborhood_layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        # Initialize combination layers
        if self.combine is not None and self.combine_method == "concat":
            # Sequential with Linear layers
            for module in self.combine.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                # MultiheadAttention initializes itself properly
                # But we can ensure it's reset if needed
                elif self.combine_method == "attention" and hasattr(self.combine, "_reset_parameters"):
                    self.combine._reset_parameters()

        # Final validation of time parameters
        self._validate_time_parameters()

    def forward(
        self, features_dict: dict[int, torch.Tensor], boundary_operators: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass across multiple simplex dimensions.

        Args:
            features_dict: {0: X_0, 1: X_1, 2: X_2, ...} features per dimension
            boundary_operators: {1: B_1, 2: B_2, ...} boundary operators

        Returns:
            Updated features for dimension 0 (vertices)
        """
        # ✅ Pre-forward validation to catch NaN early
        if self.training:
            self._validate_time_parameters()

        # Process each neighborhood and collect dimension 0 outputs
        vertex_outputs = []

        for k, layer in enumerate(self.neighborhood_layers):
            if k not in features_dict:
                continue

            X_k = features_dict[k]

            # Get features from neighboring dimensions
            X_k_down = (
                features_dict.get(k - 1, torch.zeros_like(X_k)) if k > 0 else torch.zeros_like(X_k)
            )
            X_k_up = features_dict.get(k + 1, torch.zeros_like(X_k))

            # Get boundary operators
            B_k = boundary_operators.get(k)
            B_k_plus_1 = boundary_operators.get(k + 1)

            if B_k is None:
                continue

            try:
                # Apply continuous copresheaf layer
                output = layer(X_k, X_k_down, X_k_up, B_k, B_k_plus_1)

                # Check for NaN
                if torch.isnan(output).any():
                    print(f"⚠️ WARNING: NaN detected in neighborhood {k} output!")
                    output = X_k

                # ✅ Project to vertex space (dimension 0)
                if k == 0:
                    # Already in vertex space
                    vertex_outputs.append(output)
                elif k == 1 and B_k is not None:
                    # Project edges to vertices: B_1^T @ output
                    vertex_proj = B_k.T @ output
                    vertex_outputs.append(vertex_proj)
                elif k == 2 and k in boundary_operators and (k - 1) in boundary_operators:
                    # Project triangles to edges to vertices: B_1^T @ B_2^T @ output
                    B_2 = boundary_operators[k]
                    B_1 = boundary_operators[k - 1]
                    edge_proj = B_2.T @ output
                    vertex_proj = B_1.T @ edge_proj
                    vertex_outputs.append(vertex_proj)
                else:
                    # Skip this dimension if we can't project
                    continue

            except RuntimeError as e:
                print(f"⚠️ ERROR in neighborhood {k}: {e}")
                continue

        if not vertex_outputs:
            # Return zeros if no valid outputs
            return torch.zeros_like(features_dict[0])

        # Combine outputs in vertex space (all same dimension now)
        if self.combine_method == "sum":
            combined = torch.stack(vertex_outputs).sum(dim=0)

        elif self.combine_method == "concat":
            # Pad if needed
            while len(vertex_outputs) < self.num_neighborhoods:
                vertex_outputs.append(torch.zeros_like(vertex_outputs[0]))

            concatenated = torch.cat(vertex_outputs, dim=-1)
            combined = self.combine(concatenated)

        elif self.combine_method == "attention":
            # Pad if needed
            while len(vertex_outputs) < self.num_neighborhoods:
                vertex_outputs.append(torch.zeros_like(vertex_outputs[0]))

            stacked = torch.stack(
                vertex_outputs, dim=1
            )  # [num_vertices, num_neighborhoods, out_dim]
            combined, _ = self.combine(stacked, stacked, stacked)
            combined = combined.mean(dim=1)

        # Final NaN check
        if torch.isnan(combined).any():
            print("⚠️ WARNING: NaN in final combined output! Using input as fallback.")
            combined = features_dict[0]

        return combined

    def get_time_constants(self) -> dict[str, float]:
        """
        Get current time parameters.

        Returns:
            Dictionary with 't_d' and 't_u' values
        """
        return {
            "t_d": self.t_d.item() if hasattr(self, "t_d") else 0.0,
            "t_u": self.t_u.item() if hasattr(self, "t_u") else 0.0,
        }

    def clamp_time_parameters(self, min_val: float = 1e-6, max_val: float = 10.0):
        """
        Clamp time parameters to safe range.

        This should be called after optimizer.step() to prevent:
        - Negative values
        - Values too close to zero (numerical instability)
        - Values too large (over-smoothing)

        Args:
            min_val: Minimum allowed value (default: 1e-6)
            max_val: Maximum allowed value (default: 10.0)
        """
        with torch.no_grad():
            if hasattr(self, "t_d"):
                self.t_d.data = self.t_d.data.clamp(min=min_val, max=max_val)
                if torch.isnan(self.t_d).any():
                    self.t_d.data = torch.tensor(
                        self._t_d_init, dtype=torch.float32, device=self.t_d.device
                    )

            if hasattr(self, "t_u"):
                self.t_u.data = self.t_u.data.clamp(min=min_val, max=max_val)
                if torch.isnan(self.t_u).any():
                    self.t_u.data = torch.tensor(
                        self._t_u_init, dtype=torch.float32, device=self.t_u.device
                    )

    def extra_repr(self) -> str:
        """String representation showing key parameters."""
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"num_neighborhoods={self.num_neighborhoods}, "
            f"t_d={self.t_d.item():.4f}, t_u={self.t_u.item():.4f}, "
            f"K={self.K}, combine_method={self.combine_method}"
        )


#
# from torch import nn as nn
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import scatter, softmax as segment_softmax
# from toposheafx.base.conv import Conv, HigherOrderConv
# from toposheafx.base.conv import HigherOrderConv
#
# import torch
#
# from toposheafx.utils.guards import _same_device
# from toposheafx.utils.guards import _check_index
# from toposheafx.utils.mlp    import mlp_factory

# class CopresheafLayer(nn.Module):
#     """
#     Message passing layer where each edge has its own local transformation (copresheaf morphism).
#
#     Args:
#         in_channels (int): Dim of input node features.
#         out_channels (int): Dim of output node features.
#         edge_index (LongTensor): [2, num_edges] tensor of edge indices.
#         edge_mlp_factory (Callable): function that returns a new MLP for each edge.
#     """
#     def __init__(self, in_channels, out_channels, edge_index, edge_map_factory):
#         super().__init__()
#         if edge_index.dtype != torch.long or edge_index.size(0) != 2:
#             raise ValueError("edge_index must be [2, num_edges] with dtype torch.long.")
#         self.edge_index = edge_index
#         num_edges = edge_index.size(1)
#
#         # Create a module list of local maps (MLPs), one per edge
#         self.edge_mlps = nn.ModuleList(
#             [edge_map_factory(in_channels, out_channels) for _ in range(num_edges)]
#         )
#
#     def forward(self, x):
#         """
#         Forward pass.
#
#         Args:
#             x (Tensor): Node features, shape [num_nodes, in_channels].
#
#         Returns:
#             Tensor: Updated node features, shape [num_nodes, out_channels].
#         """
#         row, col = self.edge_index
#         _check_index(row, x.size(0), "row (src)")
#         _check_index(col, x.size(0), "col (tgt)")
#         _same_device(x, self.edge_index)
#
#         # Apply per-edge local maps
#         messages = []
#         for k in range(self.edge_index.size(1)):
#             source_node = row[k]
#             mlp = self.edge_mlps[k]
#             if next(mlp.parameters()).device != x.device:
#                 mlp.to(x.device)
#             message = mlp(x[source_node].unsqueeze(0)) # shape [1, out_channels]
#             messages.append(message)
#
#         messages = torch.cat(messages, dim=0) # shape [num_edges, out_channels]
#
#         # Aggregate messages at each target node
#         # old
#         # out = scatter_add(messages, col, dim=0, dim_size=x.size(0)) # [num_nodes, out_channels]
#
#         # use of more general scatter to allow for alternate aggregation approaches (chagne reduce: reduce='sum', 'mean', 'max')
#         out = scatter(messages, col, dim=0, dim_size=x.size(0), reduce='sum')
#         return out
#
#
# # Setup
# edge_index = torch.tensor([
#     [0, 1, 2, 3, 1, 2], # source
#     [1, 2, 3, 0, 0, 1] # target
# ])
#
# x = torch.randn(4, 8) # 4 nodes, 8-dim features
#
# copresheaf_l = copresheaf_layer(
#     in_channels=8,
#     out_channels=16,
#     edge_index=edge_index,
#     edge_map_factory=mlp_factory,
# )
#
# out = copresheaf_l(x)
# print(out.shape) # [4, 16]
#
#
# class CMPNNLayer(MessagePassing):
#     """
#     CMPNNLayer: Copresheaf Message Passing Layer (Proposition 1).
#
#     A single layer of Copresheaf Message Passing Neural Network (CMPNN),
#     compatible with PyTorch Geometric's MessagePassing interface.
#     Supports edge attributes and higher-dimensional message passing across
#     multiple neighborhood functions for higher-order cell complexes.
#
#     This layer applies directional message passing over a single neighborhood (e.g. 1-skeleton)
#     using a learned copresheaf morphism ρ, message function α, and update function β.
#     For each edge y → x, we compute:
#         - Transport map: ρ(h_y; h_x)
#         - Message: m_yx = α(h_x, ρ(h_y))
#         - Aggregate: ⊕_{y ∈ N(x)} m_yx
#         - Update: h_x^{l+1} = β(h_x, ⊕ messages)
#
#     Args:
#         in_dim (int): Dimensionality of input node features F(x)
#         out_dim (int): Dimensionality of output node features F'(x)
#         edge_attr_dim (int): Dimensionality of edge attributes
#     """
#     def __init__(self, in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=1):
#         super().__init__()  # Disable built-in aggregation
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.edge_attr_dim = edge_attr_dim
#         self.num_neighborhoods = num_neighborhoods
#
#         # Multiple copresheaf maps for different neighborhoods (e.g., adjacency, incidence)
#         self.copresheaf_maps = nn.ModuleList([
#             Conv(in_dim, out_dim, edge_attr_dim) for _ in range(num_neighborhoods)
#         ])
#
#     def forward(self, x, edge_index_list, edge_attr_list=None):
#         """
#         Args:
#             x (Tensor): Node feature matrix of shape [num_nodes, in_dim].
#             edge_index_list (list of LongTensor): List of edge index tensors [2, num_edges] per neighborhood.
#             edge_attr_list (list of Tensor, optional): List of edge attribute tensors [num_edges, edge_attr_dim] per neighborhood.
#
#         Returns:
#             Tensor: Updated node feature matrix of shape [num_nodes, out_dim].
#         """
#         if edge_attr_list is None:
#             edge_attr_list = [None] * self.num_neighborhoods
#
#         all_updates = []
#         for i in range(self.num_neighborhoods):
#             edge_index = edge_index_list[i]
#             edge_attr = edge_attr_list[i]
#             row, col = edge_index  # source: row (y), target: col (x)
#
#             h_src = x[row]
#             h_tgt = x[col]
#             messages = self.copresheaf_maps[i](h_src, h_tgt, edge_attr)
#             # old implementation
#             # aggr = self.copresheaf_maps[i].aggregate_messages(messages, col, x.size(0))
#
#             # aggregate messages with scatter, allows for other aggregation approaches (change reduce)
#             aggr = self.copresheaf_maps[i].aggregate_messages(messages, col, x.size(0), reduce='sum')
#             updated = self.copresheaf_maps[i].update_node(x, aggr)
#             all_updates.append(updated)
#
#         return sum(all_updates) / len(all_updates)  # average across neighborhoods
#
#
# class CHOMPNNLayer(cmpnn_layer):
#     """
#     CHOMPNNLayer: Copresheaf-based Higher-Order Message Passing Layer (Proposition 3).
#
#     This module applies higher-order copresheaf-based message passing over multiple
#     neighborhoods of each node (e.g., edges, triangles, k-cells). It uses:
#         - One CopresheafMap per neighborhood to compute directional messages.
#         - A shared fusion function ⓧ to combine the ⊕ of α(h_x, ρ(h_y)) across neighborhoods.
#         - An update function β(h_x, ⓧ(...)) to produce h_x^{l+1}.
#
#     Args:
#         in_dim (int): Input feature dimension for each node.
#         out_dim (int): Output feature dimension for each node.
#         edge_attr_dim (int): Dimension of edge attributes.
#         num_neighborhoods (int): Number of different neighborhood structures (e.g. 1-skeleton, 2-cells, etc).
#     """
#     def __init__(self, in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=1):
#         super().__init__(in_dim, out_dim, edge_attr_dim, num_neighborhoods)  # custom aggregation logic handled by ⓧ and β
#         self.model = HigherOrderConv(
#             in_dim=in_dim,
#             out_dim=out_dim,
#             edge_attr_dim=edge_attr_dim,
#             num_neighborhoods=num_neighborhoods
#         )
#
#     def forward(self, x, edge_index_list, edge_attr_list=None):
#         """
#         Applies one CHOMPNNLayer forward pass.
#
#         Args:
#             x (Tensor): Node features [num_nodes, in_dim]
#             edge_index_list (List[Tensor]): List of edge index tensors for each neighborhood [2, num_edges_k]
#             edge_attr_list (List[Tensor or None], optional): Edge attributes per neighborhood [num_edges_k, edge_attr_dim]
#
#         Returns:
#             Tensor: Updated node features [num_nodes, out_dim]
#         """
#         return self.model(x, edge_index_list, edge_attr_list)
#
#
# class CopresheafTransformerLayer(cmpnn_layer):
#     """
#     SheafTransformerLayer: Attention-based Copresheaf Transformer Layer
#
#     Implements Algorithm 1 from Appendix E of the paper. This layer computes attention-weighted
#     directional messages using learned copresheaf transport maps ρ and attention scores derived from
#     learned queries, keys, and values for each vertex.
#
#     Args:
#         in_dim (int): Input dimension of node features
#         out_dim (int): Output dimension after transformer update
#         edge_attr_dim (int): Edge feature dimensionality
#         heads (int): Number of attention heads (default: 1)
#     """
#     def __init__(self, in_dim, out_dim, edge_attr_dim=0, heads=1, dropout=0.0):
#         super().__init__(aggr=None)
#         self.heads = heads
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#
#         self.map = Conv(in_dim, out_dim, edge_attr_dim=edge_attr_dim)
#
#         self.key = nn.Linear(in_dim, out_dim * heads)
#         self.query = nn.Linear(in_dim, out_dim * heads)
#         self.value = nn.Linear(in_dim, out_dim * heads)
#
#         self.out_proj = nn.Linear(out_dim * heads, out_dim)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x, edge_index, edge_attr=None):
#         """
#         Args:
#             x (Tensor): Node features [N, in_dim]
#             edge_index (LongTensor): Directed edges [2, E] (row=src, col=dst)
#             edge_attr (Tensor, optional): Edge features [E, edge_attr_dim]
#         Returns:
#             Tensor: Updated node features [N, out_dim]
#         """
#         row, col = edge_index  # row: src indices, col: dst indices
#         h_src, h_tgt = x[row], x[col]  # [E, in_dim] each
#
#         # Linear projections per head
#         k = self.key(h_src).view(-1, self.heads, self.out_dim)  # [E, H, d]
#         q = self.query(h_tgt).view(-1, self.heads, self.out_dim)  # [E, H, d]
#
#         # Scaled dot-product attention logits per edge & head
#         attn_logits = (Q * K).sum(dim=-1) / (self.out_dim ** 0.5)  # [E, H]
#
#         # Segment softmax over incoming edges of each *target node* for each head.
#         # Flatten heads, repeat dst index accordingly, softmax by col.
#         logits_flat = attn_logits.reshape(-1)  # [E*H]
#         index_flat = col.repeat_interleave(self.heads)  # [E*H]
#         alpha = segment_softmax(logits_flat, index_flat, num_nodes=x.size(0)).view(-1, self.heads)  # [E, H]
#         alpha = self.dropout(alpha)  # dropout on attention probs
#
#         # Copresheaf transport (edge-wise message in value space)
#         messages = self.map(h_src, h_tgt, edge_attr)  # [E, d]
#
#         # Broadcast messages over heads and weight by attention per head
#         M = messages.unsqueeze(1).expand(-1, self.heads, -1)  # [E, H, d]
#         M = M * alpha.unsqueeze(-1)  # [E, H, d]
#
#         # Aggregate head-weighted messages to target nodes
#         # -> [N, H*d], using your sheaf aggregator (sum by default here)
#         out = self.map.aggregate_messages(
#             M.reshape(-1, self.heads * self.out_dim), col, x.size(0), reduce="sum"
#         )
#
#         # Final projection mixes heads -> [N, out_dim]
#         return self.out_proj(out)
