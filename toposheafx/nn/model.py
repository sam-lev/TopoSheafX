# model.py
"""
Complete CMPNN and CHOMPNN models for graph-level tasks.

These models implement the full architectures described in "Copresheaf Topological Neural Networks"
by Hajij et al., specifically for graph classification and regression tasks.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from toposheafx.nn.layer import CHOMPNNLayer, CMPNNLayer, SimplicialComplexCopresheafLayer
from toposheafx.base.message_passing import HigherOrderMessagePassing
from toposheafx.base.aggregation import Aggregation

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

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3,
                 dropout=0.1, aggregation_strategy='adaptive'):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # Initial encoding
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Stack of copresheaf layers with different aggregation strategies
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            # Use different aggregation strategies at different depths
            if aggregation_strategy == 'adaptive':
                if i == 0:
                    aggr = 'mean'  # Normalize early
                elif i < num_layers - 1:
                    aggr = 'sum'  # Accumulate middle
                else:
                    aggr = 'attention'  # Learn importance at end
            else:
                aggr = aggregation_strategy

            # Create layer with appropriate aggregation
            if aggr == 'attention':
                # Use attention-based aggregation
                layer = HigherOrderMessagePassing(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_neighborhoods=3,
                    aggr=Aggregation('attention', dim=hidden_dim, num_heads=4),
                    combine_method='concat'
                )
            else:
                layer = SimplicialComplexCopresheafLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_neighborhoods=3,
                    aggregation=aggr
                )

            self.layers.append(layer)

        # Layer norms for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
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
            x = scatter(x, batch, dim=0, reduce='mean')

        # Final readout
        return self.readout(x)