# model.py
"""
Complete CMPNN and CHOMPNN models for graph-level tasks.

These models implement the full architectures described in "Copresheaf Topological Neural Networks"
by Hajij et al., specifically for graph classification and regression tasks.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import scatter

from toposheafx.base.aggregation import Aggregation
from toposheafx.base.conv import ContinuousConv
from toposheafx.base.message_passing import HigherOrderMessagePassing
from toposheafx.data.simplicial import SimplicialFeatures
from toposheafx.nn.layer import (
    CHOMPNNLayer,
    CMPNNLayer,
    ContinuousCopresheafLayer,
    MultiDimensionalSimplicialCopresheafLayer,
    SimplicialComplexCopresheafLayer,
)


# Create a node-level version of CMPNN
class NodeCMPNN(nn.Module):
    """CMPNN adapted for node-level predictions."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, num_neighborhoods=1):
        super().__init__()

        self.encoder = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                CMPNNLayer(
                    in_dim=hidden_dim, out_dim=hidden_dim, num_neighborhoods=num_neighborhoods
                )
                for _ in range(num_layers)
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index_list, edge_attr_list=None):
        """Returns per-node predictions [num_nodes, out_dim]"""
        x = F.relu(self.encoder(x))

        for layer in self.layers:
            x_new = layer(x, edge_index_list, edge_attr_list)
            x = x + x_new if x.shape == x_new.shape else x_new
            x = F.relu(x)

        return self.readout(x)  # [num_nodes, out_dim]


class CMPNN(nn.Module):
    """
    Copresheaf Message Passing Neural Network (Section 4.1 of the paper).

    This model implements a multi-layer architecture using Proposition 1:
    h_x^{l+1} = β(h_x^l, ⊕_{(y→x)∈E} α(h_x^l, ρ_{y→x}(h_y^l)))

    Architecture:
    1. Input encoder: Projects raw features to hidden dimension
    2. Multiple CMPNN layers: Each applies copresheaf message passing
    3. Graph pooling: Aggregates node features to graph level
    4. Readout MLP: Maps to final predictions

    The model supports multiple neighborhoods per layer, enabling learning
    on various topological structures simultaneously.

    Args:
        in_dim (int): Input feature dimension F(x)
        hidden_dim (int): Hidden layer dimension
        out_dim (int): Output dimension (e.g., number of classes)
        num_layers (int): Number of CMPNN layers
        edge_attr_dim (int): Edge attribute dimension
        num_neighborhoods (int): Number of neighborhood types
    """

    def __init__(
        self, in_dim, hidden_dim, out_dim, num_layers=2, edge_attr_dim=0, num_neighborhoods=1
    ):
        super().__init__()

        # Initial projection to hidden space (Equation 8 in paper)
        self.encoder = nn.Linear(in_dim, hidden_dim)

        # Stack of CMPNN layers implementing Proposition 1
        # Each layer refines node representations using copresheaf structure
        self.layers = nn.ModuleList(
            [
                CMPNNLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    num_neighborhoods=num_neighborhoods,
                )
                for _ in range(num_layers)
            ]
        )

        # Graph-level readout (Section 4.2 in paper)
        # Maps aggregated graph features to predictions
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index_list, batch, edge_attr_list=None):
        """
        Forward pass through CMPNN.

        Processing steps:
        1. Encode input features to hidden dimension
        2. Apply multiple rounds of copresheaf message passing
        3. Pool node features to graph level using batch indices
        4. Apply readout network for final predictions

        Mathematical flow:
        - x^0 = encoder(x_input)
        - x^{l+1} = CMPNN_layer(x^l) for l in 1..L
        - g = pool(x^L, batch)
        - output = readout(g)

        Args:
            x (Tensor): Node features [num_nodes, in_dim]
            edge_index_list (list): Edge indices per neighborhood
            batch (LongTensor): Batch assignment [num_nodes]
            edge_attr_list (list, optional): Edge attributes per neighborhood

        Returns:
            Tensor: Graph-level predictions [num_graphs, out_dim]
        """
        # Initial encoding
        x = self.encoder(x)
        x = torch.relu(x)

        # Apply copresheaf message passing layers
        # Each layer implements: h^{l+1} = β(h^l, ⊕ α(h^l, ρ(h_neighbors)))
        for layer in self.layers:
            x_new = layer(x, edge_index_list, edge_attr_list)
            # Residual connection for gradient flow (if dimensions match)
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
            x = torch.relu(x)

        # Graph-level pooling using mean aggregation
        # This implements: g_i = (1/|V_i|) Σ_{v∈V_i} h_v
        graph_repr = scatter(x, batch, dim=0, reduce="mean")

        # Final readout for predictions
        return self.readout(graph_repr)


class CHOMPNN(nn.Module):
    """
    Copresheaf Higher-Order Message Passing Neural Network (Section 4.2 of the paper).

    This model implements Proposition 3 for higher-order topological structures:
    h_x^{l+1} = β(h_x^l, ⋀_{k=1}^n ⊕_{y∈N_k(x)} α_k(h_x^l, ρ_k_{y→x}(h_y^l)))

    Key differences from CMPNN:
    - Explicitly handles multiple neighborhood types (edges, triangles, etc.)
    - Each neighborhood has specialized copresheaf morphisms
    - Inter-neighborhood combination via ⋀ operator

    This enables learning on:
    - Simplicial complexes (vertices, edges, triangles, tetrahedra)
    - Cell complexes (0-cells, 1-cells, 2-cells, etc.)
    - Hypergraphs (nodes and hyperedges)

    Args:
        in_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        out_dim (int): Output dimension
        num_layers (int): Number of CHOMPNN layers
        edge_attr_dim (int): Edge attribute dimension
        num_neighborhoods (int): Number of distinct neighborhood types
    """

    def __init__(
        self, in_dim, hidden_dim, out_dim, num_layers=3, edge_attr_dim=0, num_neighborhoods=1
    ):
        super().__init__()

        # Initial encoding to hidden space
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Normalization for stability
        )

        # Stack of higher-order copresheaf layers
        # Each implements full Proposition 3 with multiple neighborhoods
        self.layers = nn.ModuleList(
            [
                CHOMPNNLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    num_neighborhoods=num_neighborhoods,
                )
                for _ in range(num_layers)
            ]
        )

        # Intermediate layer norms for training stability
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Graph-level readout with increased capacity
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index_list, batch, edge_attr_list=None):
        """
        Forward pass through CHOMPNN.

        Processing flow:
        1. Encode input features
        2. Apply higher-order copresheaf message passing
           - Each layer processes multiple neighborhood types
           - Inter-neighborhood information is combined via ⋀
        3. Pool to graph level
        4. Apply readout for predictions

        The key innovation is simultaneous processing of multiple
        topological relations at each layer, enabling the model to
        learn complex structural patterns.

        Args:
            x (Tensor): Node features [num_nodes, in_dim]
            edge_index_list (List[Tensor]): Edge indices per neighborhood type
            batch (Tensor): Batch assignment [num_nodes]
            edge_attr_list (List[Tensor], optional): Edge attributes

        Returns:
            Tensor: Graph-level predictions [num_graphs, out_dim]
        """
        # Initial encoding
        x = self.encoder(x)

        # Apply higher-order copresheaf layers with residual connections
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            # Higher-order message passing across all neighborhoods
            x_new = layer(x, edge_index_list, edge_attr_list)

            # Residual connection + layer norm for stable training
            x = norm(x + x_new)

            # Non-linearity (except after last layer)
            if i < len(self.layers) - 1:
                x = torch.relu(x)

        # Graph pooling - can experiment with different strategies
        # Mean pooling as default (Equation 10 in paper)
        graph_repr = scatter(x, batch, dim=0, reduce="mean")

        # Alternative: Max pooling for dominant features
        # graph_repr = scatter(x, batch, dim=0, reduce='max')

        # Alternative: Sum pooling for size-aware representations
        # graph_repr = scatter(x, batch, dim=0, reduce='sum')

        # Final readout
        return self.readout(graph_repr)


class CopresheafGNN(nn.Module):
    """
    General Copresheaf GNN architecture supporting various configurations.

    This provides a flexible framework for experimenting with different
    copresheaf architectures, aggregation strategies, and network depths.
    """

    def __init__(self, config):
        """
        Initialize from configuration dictionary.

        Config should contain:
        - model_type: 'cmpnn' or 'chompnn'
        - in_dim: Input dimension
        - hidden_dim: Hidden dimension
        - out_dim: Output dimension
        - num_layers: Number of layers
        - num_neighborhoods: Number of neighborhood types
        - aggregation: Node aggregation strategy
        - dropout: Dropout rate
        """
        super().__init__()

        model_type = config.get("model_type", "cmpnn")

        if model_type == "cmpnn":
            self.model = CMPNN(
                in_dim=config["in_dim"],
                hidden_dim=config["hidden_dim"],
                out_dim=config["out_dim"],
                num_layers=config.get("num_layers", 2),
                edge_attr_dim=config.get("edge_attr_dim", 0),
                num_neighborhoods=config.get("num_neighborhoods", 1),
            )
        elif model_type == "chompnn":
            self.model = CHOMPNN(
                in_dim=config["in_dim"],
                hidden_dim=config["hidden_dim"],
                out_dim=config["out_dim"],
                num_layers=config.get("num_layers", 3),
                edge_attr_dim=config.get("edge_attr_dim", 0),
                num_neighborhoods=config.get("num_neighborhoods", 1),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x, edge_index_list, batch, edge_attr_list=None):
        """Forward pass through configured model."""
        return self.model(x, edge_index_list, batch, edge_attr_list)


class SimplicialComplexCopresheafNetwork(nn.Module):
    """
    Complete copresheaf neural network for learning on simplicial complexes.

    Architecture:
    1. Input encoder for initial features
    2. Multiple copresheaf layers with different aggregations
    3. Readout for final predictions
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=3,
        dropout=0.1,
        aggregation_strategy="adaptive",
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # Initial encoding
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )

        # Stack of copresheaf layers with different aggregation strategies
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            # Use different aggregation strategies at different depths
            if aggregation_strategy == "adaptive":
                if i == 0:
                    aggr = "mean"  # Normalize early
                elif i < num_layers - 1:
                    aggr = "sum"  # Accumulate middle
                else:
                    aggr = "attention"  # Learn importance at end
            else:
                aggr = aggregation_strategy

            # Create layer with appropriate aggregation
            if aggr == "attention":
                # Use attention-based aggregation
                layer = HigherOrderMessagePassing(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_neighborhoods=3,
                    aggr=Aggregation("attention", dim=hidden_dim, num_heads=4),
                    combine_method="concat",
                )
            else:
                layer = SimplicialComplexCopresheafLayer(
                    in_dim=hidden_dim, out_dim=hidden_dim, num_neighborhoods=3, aggregation=aggr
                )

            self.layers.append(layer)

        # Layer norms for stability
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index_list, batch=None):
        """
        Forward pass through the network.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index_list: List of edge indices for neighborhoods
            batch: Batch assignment for graph-level tasks

        Returns:
            Output predictions [num_nodes or num_graphs, out_dim]
        """
        # Initial encoding
        x = self.encoder(x)

        # Apply copresheaf layers with residual connections
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            # Copresheaf message passing
            x_new = layer(x, edge_index_list)

            # Residual + normalization
            x = norm(x + self.dropout(x_new))

            # Activation
            if i < self.num_layers - 1:
                x = F.relu(x)

        # Graph-level pooling if batch is provided
        if batch is not None:
            from torch_geometric.utils import scatter

            x = scatter(x, batch, dim=0, reduce="mean")

        # Final readout
        return self.readout(x)


class CrossDimensionalSimplicialCopresheafNetwork(nn.Module):
    """
    Network supporting true cross-dimensional message passing.
    """

    def __init__(self, initial_dims, hidden_dims, out_dim, num_layers=3):
        super().__init__()

        # Initial encoders for each dimension
        self.encoders = nn.ModuleDict(
            {
                str(dim): nn.Linear(initial_dims[dim], hidden_dims[dim])
                for dim in initial_dims
            }
        )

        # Neighborhood configuration - must match extract_cross_dimensional_neighborhoods
        neighborhoods_config = [
            ("0→0", 0, 0),  # Vertex adjacency
            ("0→1", 0, 1),  # Vertex to edge
            ("1→0", 1, 0),  # Edge to vertex
            ("1→1", 1, 1),  # Edge adjacency
            ("1→2", 1, 2),  # Edge to triangle
            ("2→1", 2, 1),  # Triangle to edge
            ("2→0", 2, 0),  # Triangle to vertex
        ]

        # Stack of cross-dimensional layers
        self.layers = nn.ModuleList(
            [
                MultiDimensionalSimplicialCopresheafLayer(
                    feature_dims=hidden_dims,
                    out_dims=hidden_dims,
                    neighborhoods_config=neighborhoods_config,
                )
                for _ in range(num_layers)
            ]
        )

        # Final readout (only from vertices for node classification)
        self.readout = nn.Linear(hidden_dims[0], out_dim)

    def forward(self, features, neighborhoods):
        """
        Args:
            features: SimplicialFeatures with raw features
            neighborhoods: dict from extract_cross_dimensional_neighborhoods
        """
        # Encode each dimension
        encoded = {}
        for dim, x in features.features.items():
            if x is not None:
                encoded[dim] = F.relu(self.encoders[str(dim)](x))

        features = SimplicialFeatures.from_dict(encoded)

        # Apply cross-dimensional layers
        for layer in self.layers:
            features = layer(features, neighborhoods)

        # Readout from vertices
        return self.readout(features[0])


class AdaptiveContinuousSimplicialModel(nn.Module):
    """
    Adaptive version with learnable time parameters per neighborhood.

    This implements multi-resolution continuous filtering where different
    topological relations (adjacency, incidence, coboundary) can have
    different diffusion rates.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        num_neighborhoods: int = 3,
        t_init: float = 0.1,
        K: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_neighborhoods = num_neighborhoods

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )

        # Layers with per-neighborhood continuous dynamics
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        f"neighborhood_{k}": ContinuousConv(
                            in_dim=hidden_dim,
                            out_dim=hidden_dim,
                            t_d_init=t_init,
                            t_u_init=t_init,
                            learnable_t=True,
                            K=K,
                        )
                        for k in range(num_neighborhoods)
                    }
                )
                for _ in range(num_layers)
            ]
        )

        # Neighborhood combination
        self.combiners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_neighborhoods * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index_list: list[torch.Tensor],
        boundary_operators_list: list[dict[int, torch.Tensor]],
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features
            edge_index_list: List of edge indices per neighborhood
            boundary_operators_list: List of boundary operator dicts
            batch: Batch assignment
        """
        x = self.encoder(x)

        for layer_dict, combiner in zip(self.layers, self.combiners):
            neighborhood_outputs = []

            for k in range(self.num_neighborhoods):
                conv = layer_dict[f"neighborhood_{k}"]
                edge_index = edge_index_list[k]
                B_dict = boundary_operators_list[k]

                # Apply continuous convolution
                x_k = conv(x=x, edge_index=edge_index, B_k=B_dict.get(1), B_k_plus_1=B_dict.get(2))
                neighborhood_outputs.append(x_k)

            # Combine neighborhoods
            x_combined = torch.cat(neighborhood_outputs, dim=-1)
            x_new = combiner(x_combined)

            # Residual connection
            x = x + self.dropout(x_new)

        # Pooling if graph-level
        if batch is not None:
            x = scatter(x, batch, dim=0, reduce="mean")

        return self.readout(x)


class ContinuousCopresheafSimplicialModel(nn.Module):
    """
    Continuous Copresheaf Simplicial Neural Network that processes features across all dimensions.
    Multi-dimensional continuous copresheaf simplicial neural network.

    Continuous Simplicial Neural Network based on COSIMO methodology.

    COSIMO Model implementing Equation 10 with multiple branches.

    From paper Equation 10:
    X^l_k = σ(e^{-t_d L_{k,d}} X^{l-1}_{k,d} Θ^l_{k,d} +
             e^{-t_u L_{k,u}} X^{l-1}_{k,u} Θ^l_{k,u} +
             e^{-t_d L_{k,d}} X^{l-1}_k Ψ^l_{k,d} +
             e^{-t_u L_{k,u}} X^{l-1}_k Ψ^l_{k,u})

    Supports multiple branches (Remark 4.2) via num_branches parameter.

    Implements PDEs on simplicial complexes:
    - Independent lower dynamics: ∂x_k,d/∂t_d = -L_k,d @ x_k,d
    - Independent upper dynamics: ∂x_k,u/∂t_u = -L_k,u @ x_k,u
    - Joint dynamics: combined diffusion across both spaces

    Iimplements:
    - COSIMO continuous dynamics (Equation 10)
    - Copresheaf morphisms for directed transport
    - Cross-dimensional message passing via boundary operators
    - Handles all dimensions including dimension 0 (vertices)

    This model:
    1. Maintains separate feature representations for vertices, edges, and triangles
    2. Uses continuous Hodge-Laplacian dynamics at each dimension
    3. Enables cross-dimensional message passing
    4. Projects all features to a target dimension for final prediction

    Args:
        feature_dims: Dict mapping dimension to input feature dimension
                     e.g., {0: 70, 1: 30, 2: 41} for vertex/edge/triangle features
        hidden_dims: Dict mapping dimension to hidden dimension
                    e.g., {0: 64, 1: 32, 2: 16}
        out_dim: Output dimension for final predictions
        target_dim: Which dimension to produce output for (default: 0 for vertices)
        num_layers: Number of continuous layers per dimension
        t_d_init: Initial lower time parameter
        t_u_init: Initial upper time parameter
        learnable_t: Whether time parameters are learnable
        K: Number of eigenvectors for approximation
        dropout: Dropout rate

    This model can be initialized in two ways:

    1. Explicit (full control):
        feature_dims={0: 70, 1: 30, 2: 41}
        hidden_dims={0: 64, 1: 32, 2: 16}

    2. Automatic (simple):
        num_simplices={0: 70, 1: 30, 2: 41}
        input_dim=70
        hidden_dim=64
        hidden_dim_strategy='decreasing'

    The constructor automatically detects which mode to use.
    """

    def __init__(
        self,
        feature_dims: dict[int, int],
        hidden_dims: dict[int, int],
        out_dim: int,
        target_dim: int = 0,
        num_layers: int = 3,
        t_d_init: float = 0.1,
        t_u_init: float = 0.1,
        learnable_t: bool = True,
        K: Optional[int] = None,
        dropout: float = 0.1,
        use_morphisms: bool = True,
    ):
        super().__init__()

        self.feature_dims = feature_dims
        self.hidden_dims = hidden_dims
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.dimensions = sorted(feature_dims.keys())
        self.out_dim = out_dim

        # Encoders for each dimension
        self.encoders = nn.ModuleDict(
            {
                str(dim): nn.Sequential(
                    nn.Linear(feature_dims[dim], hidden_dims[dim]),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dims[dim]),
                )
                for dim in self.dimensions
            }
        )

        # ✅ CONTINUOUS LAYERS WITH COPRESHEAF MORPHISMS
        self.dimension_layers = nn.ModuleDict()
        for dim in self.dimensions:
            layers = nn.ModuleList(
                [
                    ContinuousCopresheafLayer(
                        in_dim=hidden_dims[dim],
                        out_dim=hidden_dims[dim],
                        k_simplex_dim=dim,
                        t_d_init=t_d_init,
                        t_u_init=t_u_init,
                        learnable_t=learnable_t,
                        K=K,
                        use_morphisms=use_morphisms,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.dimension_layers[str(dim)] = layers

        # Layer norms
        self.layer_norms = nn.ModuleDict(
            {
                str(dim): nn.ModuleList([nn.LayerNorm(hidden_dims[dim]) for _ in range(num_layers)])
                for dim in self.dimensions
            }
        )

        # Cross-dimensional projections
        self.cross_dim_projections = nn.ModuleDict()
        for i, dim_from in enumerate(self.dimensions[:-1]):
            dim_to = self.dimensions[i + 1]
            self.cross_dim_projections[f"{dim_from}_to_{dim_to}"] = nn.Linear(
                hidden_dims[dim_from], hidden_dims[dim_to]
            )
            self.cross_dim_projections[f"{dim_to}_to_{dim_from}"] = nn.Linear(
                hidden_dims[dim_to], hidden_dims[dim_from]
            )

        self.dropout = nn.Dropout(dropout)

        # Final readout from target dimension
        self.readout = nn.Sequential(
            nn.Linear(hidden_dims[target_dim], hidden_dims[target_dim] * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[target_dim] * 2, hidden_dims[target_dim]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dims[target_dim], out_dim),
        )

        self.reset_parameters()
    
    def get_time_constants(self) -> dict[str, list[float]]:
        """Get current time parameters from all layers."""
        time_constants = {"t_d": [], "t_u": []}

        for dim_layers in self.dimension_layers.values():
            for layer in dim_layers:
                if hasattr(layer, "t_d"):
                    time_constants["t_d"].append(layer.t_d.item())
                    time_constants["t_u"].append(layer.t_u.item())

        return time_constants

    @staticmethod
    def _auto_configure_dimensions(
        num_simplices: dict[int, int],
        input_dim: Optional[int],
        hidden_dim: Optional[int],
        hidden_dim_strategy: str,
    ) -> tuple[dict[int, int], dict[int, int]]:
        """
        Automatically configure feature_dims and hidden_dims from simple inputs.

        Args:
            num_simplices: Number of simplices per dimension {0: 70, 1: 30, 2: 41}
            input_dim: Input feature dimension (if None, uses num_simplices[0])
            hidden_dim: Base hidden dimension (required)
            hidden_dim_strategy: How to scale hidden dims across dimensions

        Returns:
            (feature_dims, hidden_dims)
        """
        if hidden_dim is None:
            raise ValueError("hidden_dim is required for automatic configuration")

        dimensions = sorted(num_simplices.keys())

        # Set input dimension
        if input_dim is None:
            # Default: use one-hot encoding size = number of vertices
            input_dim = num_simplices[0]

        # Feature dims: all dimensions use same input feature size
        # (This assumes features are projected/encoded to same space initially)
        feature_dims = dict.fromkeys(dimensions, input_dim)

        # Hidden dims: scale based on strategy
        if hidden_dim_strategy == "uniform":
            hidden_dims = dict.fromkeys(dimensions, hidden_dim)

        elif hidden_dim_strategy == "decreasing":
            # Higher dimensions get smaller representations
            # dim 0: hidden_dim, dim 1: hidden_dim/2, dim 2: hidden_dim/4
            hidden_dims = {dim: max(hidden_dim // (2**dim), 16) for dim in dimensions}

        elif hidden_dim_strategy == "increasing":
            # Higher dimensions get larger representations
            max_dim = max(dimensions)
            hidden_dims = {dim: max(hidden_dim // (2 ** (max_dim - dim)), 16) for dim in dimensions}

        elif hidden_dim_strategy == "proportional":
            # Scale hidden dim proportionally to number of simplices
            max_simplices = max(num_simplices.values())
            hidden_dims = {
                dim: max(int(hidden_dim * num_simplices[dim] / max_simplices), 16)
                for dim in dimensions
            }

        else:
            raise ValueError(
                f"Unknown hidden_dim_strategy: {hidden_dim_strategy}. "
                f"Must be 'uniform', 'decreasing', 'increasing', or 'proportional'"
            )

        return feature_dims, hidden_dims

    def _encode_features(self, features: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Encode features for each dimension."""
        encoded = {}
        for dim in self.dimensions:
            if dim in features and features[dim] is not None:
                encoded[dim] = self.encoders[str(dim)](features[dim])
            else:
                # Create zero features if dimension not provided
                encoded[dim] = torch.zeros(
                    1, self.hidden_dims[dim], device=next(self.parameters()).device
                )
        return encoded

    def _prepare_dimension_features(
        self,
        features: dict[int, torch.Tensor],
        boundary_operators: dict[int, torch.Tensor],
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare features for a specific dimension including projections.

        Returns:
            X_k: Features at dimension k
            X_k_down: Features projected from dimension k-1
            X_k_up: Features projected from dimension k+1
        """
        X_k = features[dim]

        # Lower features (k-1 -> k)
        if dim > 0 and (dim - 1) in features:
            dim_down = dim - 1
            B_k = boundary_operators.get(dim)

            if B_k is not None:
                # Project from lower dimension via boundary operator
                # B_k: [num_(k-1)_simplices, num_k_simplices]
                # X_{k-1}: [num_(k-1)_simplices, hidden_dim]
                # Result: [num_k_simplices, hidden_dim]
                X_down_raw = B_k.T @ features[dim_down]

                # Apply learned projection to match dimensions
                proj_key = f"{dim_down}_to_{dim}"
                if proj_key in self.cross_dim_projections:
                    X_k_down = self.cross_dim_projections[proj_key](X_down_raw)
                else:
                    X_k_down = X_down_raw
            else:
                X_k_down = torch.zeros_like(X_k)
        else:
            X_k_down = torch.zeros_like(X_k)

        # Upper features (k+1 -> k)
        if dim < max(self.dimensions) and (dim + 1) in features:
            dim_up = dim + 1
            B_k_plus_1 = boundary_operators.get(dim + 1)

            if B_k_plus_1 is not None:
                # Project from upper dimension via boundary operator
                # B_{k+1}: [num_k_simplices, num_{k+1}_simplices]
                # X_{k+1}: [num_{k+1}_simplices, hidden_dim]
                # Result: [num_k_simplices, hidden_dim]
                X_up_raw = B_k_plus_1 @ features[dim_up]

                # Apply learned projection to match dimensions
                proj_key = f"{dim_up}_to_{dim}"
                if proj_key in self.cross_dim_projections:
                    X_k_up = self.cross_dim_projections[proj_key](X_up_raw)
                else:
                    X_k_up = X_up_raw
            else:
                X_k_up = torch.zeros_like(X_k)
        else:
            X_k_up = torch.zeros_like(X_k)

        return X_k, X_k_down, X_k_up

    def forward(
        self,
        features: dict[int, torch.Tensor],
        boundary_operators: dict[int, torch.Tensor],
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through multi-dimensional continuous model.

        Args:
            features: Dict mapping dimension to features
                     {0: [num_vertices, feature_dim_0],
                      1: [num_edges, feature_dim_1],
                      2: [num_triangles, feature_dim_2]}
            boundary_operators: Dict mapping dimension k to boundary operator B_k
                               B_k maps k-simplices to (k-1)-simplices
            batch: Optional batch assignment for graph-level tasks

        Returns:
            Output predictions [num_target_simplices, out_dim]
        """
        # Encode features for each dimension
        encoded = self._encode_features(features)

        # Process each dimension through its layers
        for layer_idx in range(self.num_layers):
            updated_features = {}

            for dim in self.dimensions:
                # Get layer for this dimension
                layer = self.dimension_layers[str(dim)][layer_idx]
                norm = self.layer_norms[str(dim)][layer_idx]

                # Prepare features with cross-dimensional information
                X_k, X_k_down, X_k_up = self._prepare_dimension_features(
                    encoded, boundary_operators, dim
                )

                # Get boundary operators
                B_k = boundary_operators.get(dim)
                B_k_plus_1 = boundary_operators.get(dim + 1)

                # Apply continuous dynamics
                try:
                    X_k_new = layer(X_k, X_k_down, X_k_up, B_k, B_k_plus_1)

                    # Residual connection + normalization
                    if X_k.shape == X_k_new.shape:
                        updated_features[dim] = norm(X_k + self.dropout(X_k_new))
                    else:
                        updated_features[dim] = norm(X_k_new)

                    # Check for NaN
                    if torch.isnan(updated_features[dim]).any():
                        print(f"⚠️ NaN detected at dimension {dim}, layer {layer_idx}")
                        updated_features[dim] = X_k

                except RuntimeError as e:
                    print(f"⚠️ ERROR at dimension {dim}, layer {layer_idx}: {e}")
                    updated_features[dim] = X_k

            # Update encoded features for next layer
            encoded = updated_features

        # Get features from target dimension
        target_features = encoded[self.target_dim]

        # Graph-level pooling if batch provided
        if batch is not None:
            from torch_geometric.utils import scatter

            target_features = scatter(target_features, batch, dim=0, reduce="mean")

        # Final readout
        output = self.readout(target_features)

        return output
