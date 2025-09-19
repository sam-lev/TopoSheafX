# copresheaf.py
"""
copresheaf.py

Copresheaf-based message passing components for graph and cell complex neural networks.
Implements local copresheaf morphisms and aggregation rules as described in
"Copresheaf Topological Neural Networks" Hajij et. al.

Usage:
-------
>>> import torch
>>> from models.copresheaf import Conv, HigherOrderConv
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
from torch_scatter import scatter
from toposheafx.utils.guards import _same_device
from toposheafx.utils.guards import _check_index



class Conv(nn.Module):
    """
    Implements a copresheaf morphism for message passing on directed complexes.

    Implements components described in Proposition 1:
    - rho: F(y) → F(x), a learnable transport map between feature spaces of incident cells
    - alpha: combines h_x and transported h_y (i.e., alpha(h_x, rho(h_y)))
    - beta: updates h_x using the cumulative sum ⊕_y alpha(h_x, rho(h_y))

    Args:
        in_dim (int): Dimensionality of node input features F(x).
        out_dim (int): Dimensionality of the transformed message output.
        edge_attr_dim (int): Optional dimension of edge attributes.
    """
    def __init__(self, in_dim, out_dim, edge_attr_dim=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_attr_dim = edge_attr_dim

        # Learnable copresheaf transport morphism: ρ : F(y) → F(x)
        # Takes h_y, h_x, and optional edge_attr and produces ρ(h_y; h_x)
        rho_input_dim = 2 * in_dim + edge_attr_dim
        self.rho_map = nn.Sequential(
            nn.Linear(rho_input_dim, out_dim),
            nn.Tanh()
        )

        # Message function α: combines h_x and ρ(h_y)
        self.alpha = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU()
        )

        # Update function β: takes h_x and the aggregated α-messages
        self.Beta = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, h_src, h_tgt, edge_attr=None):
        """
        Computes directional messages α(h_x, ρ(h_y)) per edge.

        Args:
            h_src (Tensor): Source node features h_y [num_edges, in_dim]
            h_tgt (Tensor): Target node features h_x [num_edges, in_dim]
            edge_attr (Tensor or None): Optional edge attributes [num_edges, edge_attr_dim]

        Returns:
            Tensor: Output messages [num_edges, out_dim]
        """
        # if edge_attr is not None:
        #     if not isinstance(edge_attr, torch.Tensor):
        #         raise TypeError(f"Expected edge_attr to be a Tensor or None, but got {type(edge_attr)}.")
        #     rho_input = torch.cat([h_src, h_tgt, edge_attr], dim=-1)
        # else:
        #     rho_input = torch.cat([h_src, h_tgt], dim=-1)
        #
        # rho_output = self.rho_map(rho_input)
        # return self.alpha(torch.cat([h_tgt, rho_output], dim=-1))

        _same_device(h_src, h_tgt, edge_attr if edge_attr is not None else h_src)
        if h_src.dim() != 2 or h_tgt.dim() != 2:
            raise ValueError("h_src and h_tgt must be [num_edges, in_dim].")
        if h_src.size(0) != h_tgt.size(0):
            raise ValueError("h_src and h_tgt must have the same num_edges.")
        if h_src.size(1) != self.in_dim or h_tgt.size(1) != self.in_dim:
            raise ValueError(f"Expected in_dim={self.in_dim} for h_src/h_tgt.")

        if edge_attr is not None:
            if edge_attr.dim() != 2 or edge_attr.size(0) != h_src.size(0):
                raise ValueError("edge_attr must be [num_edges, edge_attr_dim].")
            if edge_attr.size(1) != self.edge_attr_dim:
                raise ValueError(f"edge_attr_dim mismatch (got {edge_attr.size(1)}, "
                                 f"expected {self.edge_attr_dim}).")
            rho_input = torch.cat([h_src, h_tgt, edge_attr], dim=-1)
        else:
            rho_input = torch.cat([h_src, h_tgt], dim=-1)

        rho_output = self.rho_map(rho_input)
        return self.alpha(torch.cat([h_tgt, rho_output], dim=-1))

    # old implementation
    # def aggregate_messages(self, messages, index, num_nodes):
    #     """
    #     Aggregates messages per node using cumulative sum ⊕.
    #
    #     Args:
    #         messages (Tensor): Message values [num_edges, out_dim]
    #         index (Tensor): Target indices (node IDs) for each message [num_edges]
    #         num_nodes (int): Total number of nodes in the graph
    #
    #     Returns:
    #         Tensor: Aggregated messages [num_nodes, out_dim]
    #     """
    #     return torch.zeros(num_nodes, self.out_dim, device=messages.device).scatter_add_(
    #         dim=0,
    #         index=index.unsqueeze(-1).expand(-1, messages.size(-1)),
    #         src=messages
    #     )

    def aggregate_messages(self, messages, index, num_nodes, reduce='sum'): #reduce='sum', 'mean', 'max'
        """
        Aggregates messages per node using torch_scatter.

        Args:
            messages (Tensor): [num_edges, out_dim]
            index (Tensor): [num_edges]
            num_nodes (int): number of nodes
            reduce (str): aggregation type: 'sum', 'mean', or 'max'

        Returns:
            Tensor: [num_nodes, out_dim]
        """
        _check_index(index, num_nodes, name="target index (col)")
        _same_device(messages, index)
        if reduce not in {"sum","mean","max"}:
            raise ValueError("reduce must be 'sum', 'mean', or 'max'.")
        return scatter(messages, index, dim=0, dim_size=num_nodes, reduce=reduce)

    def update_node(self, h_x, aggr_msg):
        """
        Updates node representation via β(h_x, ⊕_y α(h_x, ρ(h_y))).

        Args:
            h_x (Tensor): Previous node features [num_nodes, in_dim]
            aggr_msg (Tensor): Aggregated messages [num_nodes, out_dim]

        Returns:
            Tensor: Updated node features [num_nodes, out_dim]
        """
        return self.Beta(torch.cat([h_x, aggr_msg], dim=-1))


class HigherOrderConv(nn.Module):
    """
    Implements higher-order copresheaf-based message passing (Proposition 3).

    For each vertex x, messages are computed for each neighborhood k = 1..n
    (e.g., adjacency, incidence, co-boundary). Then, these are aggregated using:

        ⓧ({⊕_y α_k(h_x, ρ_k(h_y))}) for k = 1 to n

    Followed by an update function β that combines the result of ⓧ with h_x^l.

    Args:
        in_dim (int): Input feature dimension F(x)
        out_dim (int): Output feature dimension F'(x)
        edge_attr_dim (int): Dim of edge attributes per neighborhood
        num_neighborhoods (int): Number of neighborhood types
    """
    def __init__(self, in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=1, Edge_Map=Conv):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_neighborhoods = num_neighborhoods

        # One copresheaf map per neighborhood (e.g., adjacency, higher-order face maps)
        self.maps = nn.ModuleList([
            Edge_Map(in_dim, out_dim, edge_attr_dim) for _ in range(num_neighborhoods)
        ])

        # Combination function ⓧ: merges [⊕_1, ⊕_2, ..., ⊕_n] into single vector
        self.combine = nn.Sequential(
            nn.Linear(num_neighborhoods * out_dim, out_dim),
            nn.ReLU()
        )

        # Final update function β(h_x, ⓧ(...))
        self.Beta = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index_list, edge_attr_list=None):
        """
        Performs higher-order copresheaf message passing for each neighborhood.

        Args:
            x (Tensor): Node features h_x [num_nodes, in_dim]
            edge_index_list (List[Tensor]): List of edge_index tensors for n neighborhoods
            edge_attr_list (List[Tensor or None], optional): List of edge attributes per neighborhood

        Returns:
            Tensor: Updated node features h_x^{l+1} [num_nodes, out_dim]
        """
        # if edge_attr_list is None:
        #     edge_attr_list = [None] * len(edge_index_list)
        #
        # aggregated = []
        # for i, (edge_index, edge_attr) in enumerate(zip(edge_index_list, edge_attr_list)):
        #     row, col = edge_index  # edge y → x: src = y, tgt = x
        #     h_src, h_tgt = x[row], x[col]
        #     messages = self.maps[i](h_src, h_tgt, edge_attr)
        #     agg = self.maps[i].aggregate_messages(messages, col, x.size(0))
        #     aggregated.append(agg)
        #
        # combined = torch.cat(aggregated, dim=-1)  # Stack [⊕_1, ..., ⊕_n]
        # fused = self.combine(combined)            # ⓧ(⊕_1, ..., ⊕_n)
        # return self.Beta(torch.cat([x, fused], dim=-1))  # β(h_x, ⓧ(...))


        if edge_attr_list is None:
            edge_attr_list = [None] * len(edge_index_list)
        if len(edge_index_list) != len(self.maps):
            raise ValueError("num_neighborhoods mismatch with edge_index_list length.")

        aggregated = []
        for i, (edge_index, edge_attr) in enumerate(zip(edge_index_list, edge_attr_list)):
            if edge_index.dtype != torch.long:
                raise TypeError("edge_index must be torch.long.")
            if edge_index.size(0) != 2:
                raise ValueError("edge_index must be shape [2, num_edges].")

            row, col = edge_index                              # edge y → x: src = y, tgt = x
            _check_index(row, x.size(0), "row (src)")
            _check_index(col, x.size(0), "col (tgt)")
            _same_device(x, edge_index, edge_attr if edge_attr is not None else x)

            h_src, h_tgt = x[row], x[col]
            messages = self.maps[i](h_src, h_tgt, edge_attr)
            agg = self.maps[i].aggregate_messages(messages, col, x.size(0))
            aggregated.append(agg)

        combined = torch.cat(aggregated, dim=-1)    # Stack [⊕_1, ..., ⊕_n]
        fused = self.combine(combined)              # ⓧ(⊕_1, ..., ⊕_n)
        return self.Beta(torch.cat([x, fused], dim=-1))  # β(h_x, ⓧ(...))



