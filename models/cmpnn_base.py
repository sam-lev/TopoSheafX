# cmpnn_base.py
"""
cmpnn_base.py

Core architecture classes for CMPNN and CHOMPNN models using Copresheaf-based message passing.

Usage:
-------
>>> import torch
>>> from models.cmpnn_base import CMPNN, CHOMPNN
>>> x = torch.randn(6, 8)
>>> edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
>>> batch = torch.tensor([0, 0, 0, 1, 1, 1])
>>> model = CMPNN(in_dim=8, hidden_dim=16, out_dim=3)
>>> out = model(x, [edge_index], batch)

>>> chom = CHOMPNN(in_dim=8, hidden_dim=16, out_dim=3)
>>> out2 = chom(x, [edge_index], batch)
"""

import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from models.copresheaf import CopresheafMap
from models.copresheaf import CopresheafHigherOrderMap

class CMPNNLayer(MessagePassing):
    """
    CMPNNLayer: Copresheaf Message Passing Layer (Proposition 1).

    A single layer of Copresheaf Message Passing Neural Network (CMPNN),
    compatible with PyTorch Geometric's MessagePassing interface.
    Supports edge attributes and higher-dimensional message passing across
    multiple neighborhood functions for higher-order cell complexes.

    This layer applies directional message passing over a single neighborhood (e.g. 1-skeleton)
    using a learned copresheaf morphism ρ, message function α, and update function β.
    For each edge y → x, we compute:
        - Transport map: ρ(h_y; h_x)
        - Message: m_yx = α(h_x, ρ(h_y))
        - Aggregate: ⊕_{y ∈ N(x)} m_yx
        - Update: h_x^{l+1} = β(h_x, ⊕ messages)

    Args:
        in_dim (int): Dimensionality of input node features F(x)
        out_dim (int): Dimensionality of output node features F'(x)
        edge_attr_dim (int): Dimensionality of edge attributes
    """
    def __init__(self, in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=1):
        super().__init__(aggr=None)  # Disable built-in aggregation
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_neighborhoods = num_neighborhoods

        # Multiple copresheaf maps for different neighborhoods (e.g., adjacency, incidence)
        self.copresheaf_maps = nn.ModuleList([
            CopresheafMap(in_dim, out_dim, edge_attr_dim) for _ in range(num_neighborhoods)
        ])

    def forward(self, x, edge_index_list, edge_attr_list=None):
        """
        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_dim].
            edge_index_list (list of LongTensor): List of edge index tensors [2, num_edges] per neighborhood.
            edge_attr_list (list of Tensor, optional): List of edge attribute tensors [num_edges, edge_attr_dim] per neighborhood.

        Returns:
            Tensor: Updated node feature matrix of shape [num_nodes, out_dim].
        """
        if edge_attr_list is None:
            edge_attr_list = [None] * self.num_neighborhoods

        all_updates = []
        for i in range(self.num_neighborhoods):
            edge_index = edge_index_list[i]
            edge_attr = edge_attr_list[i]
            row, col = edge_index  # source: row (y), target: col (x)

            h_src = x[row]
            h_tgt = x[col]
            messages = self.copresheaf_maps[i](h_src, h_tgt, edge_attr)
            # old implementation
            # aggr = self.copresheaf_maps[i].aggregate_messages(messages, col, x.size(0))

            # aggregate messages with scatter, allows for other aggregation approaches (change reduce)
            aggr = self.copresheaf_maps[i].aggregate_messages(messages, col, x.size(0), reduce='sum')
            updated = self.copresheaf_maps[i].update_node(x, aggr)
            all_updates.append(updated)

        return sum(all_updates) / len(all_updates)  # average across neighborhoods


class CHOMPNNLayer(MessagePassing):
    """
    CHOMPNNLayer: Copresheaf-based Higher-Order Message Passing Layer (Proposition 3).

    This module applies higher-order copresheaf-based message passing over multiple
    neighborhoods of each node (e.g., edges, triangles, k-cells). It uses:
        - One CopresheafMap per neighborhood to compute directional messages.
        - A shared fusion function ⓧ to combine the ⊕ of α(h_x, ρ(h_y)) across neighborhoods.
        - An update function β(h_x, ⓧ(...)) to produce h_x^{l+1}.

    Args:
        in_dim (int): Input feature dimension for each node.
        out_dim (int): Output feature dimension for each node.
        edge_attr_dim (int): Dimension of edge attributes.
        num_neighborhoods (int): Number of different neighborhood structures (e.g. 1-skeleton, 2-cells, etc).
    """
    def __init__(self, in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=1):
        super().__init__(aggr=None)  # custom aggregation logic handled by ⓧ and β
        self.model = CopresheafHigherOrderMap(
            in_dim=in_dim,
            out_dim=out_dim,
            edge_attr_dim=edge_attr_dim,
            num_neighborhoods=num_neighborhoods
        )

    def forward(self, x, edge_index_list, edge_attr_list=None):
        """
        Applies one CHOMPNNLayer forward pass.

        Args:
            x (Tensor): Node features [num_nodes, in_dim]
            edge_index_list (List[Tensor]): List of edge index tensors for each neighborhood [2, num_edges_k]
            edge_attr_list (List[Tensor or None], optional): Edge attributes per neighborhood [num_edges_k, edge_attr_dim]

        Returns:
            Tensor: Updated node features [num_nodes, out_dim]
        """
        return self.model(x, edge_index_list, edge_attr_list)


class CMPNN(nn.Module):
    """
    Multi-layer Copresheaf Message Passing Neural Network for graph-level tasks,
    extended to support higher-dimensional cell complexes.

    This model uses:
        - An encoder to project input features into hidden space.
        - Multiple CMPNNLayers with directional message passing.
        - Graph-level mean pooling followed by MLP readout.

    Implements the forward update:
        h_x^{l+1} = β(h_x^l, ⊕_{y ∈ N(x)} α(h_x^l, ρ(h_y^l)))

    Args:
        in_dim (int): Input dimension F(x)
        hidden_dim (int): Hidden layer dimension
        out_dim (int): Output prediction dimension (e.g., # of classes)
        num_layers (int): Number of stacked CMPNN layers
        edge_attr_dim (int): Edge feature dimensionality
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, edge_attr_dim=0, num_neighborhoods=1):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            CMPNNLayer(hidden_dim, hidden_dim, edge_attr_dim=edge_attr_dim, num_neighborhoods=num_neighborhoods)
            for _ in range(num_layers)
        ])
        self.readout = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index_list, batch, edge_attr_list=None):
        """
        Args:
            x (Tensor): Node feature matrix [num_nodes, in_dim].
            edge_index_list (list of LongTensor): List of edge index tensors per neighborhood.
            batch (LongTensor): Batch vector assigning each node to a graph [num_nodes].
            edge_attr_list (list of Tensor, optional): List of edge attributes per neighborhood.

        Returns:
            Tensor: Graph-level output vector [num_graphs, out_dim].
        """
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index_list, edge_attr_list=edge_attr_list)
        graph_repr = scatter(x, batch, dim=0, reduce='mean')
        return self.readout(graph_repr)


class CHOMPNN(nn.Module):
    """
    CHOMPNN: Multi-layer Copresheaf Higher-Order Message Passing Neural Network.

    This network stacks multiple CHOMPNNLayers for deep representation learning on
    combinatorial complexes. Includes:
        - Initial encoder to map input node features into hidden space
        - Multiple CHOMPNNLayer blocks
        - Readout MLP for graph-level classification

    Args:
        in_dim (int): Dimension of input node features
        hidden_dim (int): Hidden feature size
        out_dim (int): Output dimension for prediction (e.g., number of classes)
        num_layers (int): Number of CHOMPNN layers
        edge_attr_dim (int): Dimension of edge attributes per neighborhood
        num_neighborhoods (int): Number of neighborhood functions per graph
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, edge_attr_dim=0, num_neighborhoods=1):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            CHOMPNNLayer(hidden_dim, hidden_dim, edge_attr_dim, num_neighborhoods)
            for _ in range(num_layers)
        ])
        self.readout = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index_list, batch, edge_attr_list=None):
        """
        Forward pass through the full CHOMPNN model.

        Args:
            x (Tensor): Node features [num_nodes, in_dim]
            edge_index_list (List[Tensor]): List of edge index tensors for each neighborhood
            batch (Tensor): Batch vector mapping each node to a graph [num_nodes]
            edge_attr_list (List[Tensor], optional): Edge attribute list per neighborhood

        Returns:
            Tensor: Graph-level outputs [num_graphs, out_dim]
        """
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index_list, edge_attr_list)
        graph_repr = scatter(x, batch, dim=0, reduce='mean')
        return self.readout(graph_repr)



class SheafTransformerLayer(MessagePassing):
    """
    SheafTransformerLayer: Attention-based Copresheaf Transformer Layer

    Implements Algorithm 1 from Appendix E of the paper. This layer computes attention-weighted
    directional messages using learned copresheaf transport maps ρ and attention scores derived from
    learned queries, keys, and values for each vertex.

    Args:
        in_dim (int): Input dimension of node features
        out_dim (int): Output dimension after transformer update
        edge_attr_dim (int): Edge feature dimensionality
        heads (int): Number of attention heads (default: 1)
    """
    def __init__(self, in_dim, out_dim, edge_attr_dim=0, heads=1, dropout=0.0):
        super().__init__(aggr=None)
        self.heads = heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.map = CopresheafMap(in_dim, out_dim, edge_attr_dim=edge_attr_dim)

        self.key = nn.Linear(in_dim, out_dim * heads)
        self.query = nn.Linear(in_dim, out_dim * heads)
        self.value = nn.Linear(in_dim, out_dim * heads)

        self.out_proj = nn.Linear(out_dim * heads, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x (Tensor): Node features [num_nodes, in_dim]
            edge_index (Tensor): Directed edges [2, num_edges]
            edge_attr (Tensor): Optional edge attributes [num_edges, edge_attr_dim]

        Returns:
            Tensor: Updated node features [num_nodes, out_dim]
        """
        row, col = edge_index  # y, x
        h_src, h_tgt = x[row], x[col]

        # Linear projections for attention
        K = self.key(h_src).view(-1, self.heads, self.out_dim)
        Q = self.query(h_tgt).view(-1, self.heads, self.out_dim)
        V = self.value(h_src).view(-1, self.heads, self.out_dim)

        attn_logits = (Q * K).sum(dim=-1) / (self.out_dim ** 0.5)  # scaled dot-product
        attn_weights = self.softmax(attn_logits, dim=1)  # attention weights per head [num_edges, heads] [E, H]
        attn_weights = (attn_weights.unsqueeze(-1) * V).sum(dim=1)  # [E, out_dim]
        attn_weights = self.dropout(attn_weights)  # apply dropout to attention

        # Transport messages using copresheaf map ρ then apply attention
        messages = self.map(h_src, h_tgt, edge_attr)  # [num_edges, out_dim]
        M = messages.view(-1, 1, self.out_dim).expand(-1, self.heads, -1) * attn_weights.unsqueeze(-1)

        # old implementation
        # out = self.map.aggregate_messages(
        #     M.view(-1, self.heads * self.out_dim), col, x.size(0)
        # )

        # aggregate per edge, use of scatter for efficiency and allows for other aggregation
        # approaches (change reduce)
        out = self.map.aggregate_messages(
            M.view(-1, self.heads * self.out_dim), col, x.size(0), reduce='sum'
        )
        return self.out_proj(out)