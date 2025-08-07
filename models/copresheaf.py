# copresheaf.py
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_scatter import scatter

def mlp_factory(dim_in: int, dim_out: int, mlp_type='lin_lrelu_lin'):
    if mlp_type == 'lin':
        return nn.Linear(dim_in, dim_out)

    elif mlp_type == 'lin_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out)
        )

    elif mlp_type == 'lin_bn_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out)
        )
    else:
        raise ValueError("Unknown gin_mlp_type!")



class CopresheafMap(nn.Module):
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
        if edge_attr is not None:
            if not isinstance(edge_attr, torch.Tensor):
                raise TypeError(f"Expected edge_attr to be a Tensor or None, but got {type(edge_attr)}.")
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


class CopresheafHigherOrderMap(nn.Module):
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
    def __init__(self, in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=1, Edge_Map=CopresheafMap):
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
        if edge_attr_list is None:
            edge_attr_list = [None] * len(edge_index_list)

        aggregated = []
        for i, (edge_index, edge_attr) in enumerate(zip(edge_index_list, edge_attr_list)):
            row, col = edge_index  # edge y → x: src = y, tgt = x
            h_src, h_tgt = x[row], x[col]
            messages = self.maps[i](h_src, h_tgt, edge_attr)
            agg = self.maps[i].aggregate_messages(messages, col, x.size(0))
            aggregated.append(agg)

        combined = torch.cat(aggregated, dim=-1)  # Stack [⊕_1, ..., ⊕_n]
        fused = self.combine(combined)            # ⓧ(⊕_1, ..., ⊕_n)
        return self.Beta(torch.cat([x, fused], dim=-1))  # β(h_x, ⓧ(...))




class CopresheafMessagePassing(nn.Module):
    """
    Message passing layer where each edge has its own local transformation (copresheaf morphism).

    Args:
        in_channels (int): Dim of input node features.
        out_channels (int): Dim of output node features.
        edge_index (LongTensor): [2, num_edges] tensor of edge indices.
        edge_mlp_factory (Callable): function that returns a new MLP for each edge.
    """
    def __init__(self, in_channels, out_channels, edge_index, edge_map_factory):
        super().__init__()
        self.edge_index = edge_index
        num_edges = edge_index.size(1)

        # Create a module list of local maps (MLPs), one per edge
        self.edge_mlps = nn.ModuleList([
            edge_map_factory(in_channels, out_channels) for _ in range(num_edges)
        ])

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Node features, shape [num_nodes, in_channels].

        Returns:
            Tensor: Updated node features, shape [num_nodes, out_channels].
        """
        row, col = self.edge_index # source, target
        messages = []

        for k in range(self.edge_index.size(1)):
            source_node = row[k]
            mlp = self.edge_mlps[k]
            message = mlp(x[source_node].unsqueeze(0)) # shape [1, out_channels]
            messages.append(message)

        messages = torch.cat(messages, dim=0) # shape [num_edges, out_channels]

        # Aggregate messages at each target node
        # old
        # out = scatter_add(messages, col, dim=0, dim_size=x.size(0)) # [num_nodes, out_channels]

        # use of more general scatter to allow for alternate aggregation approaches (chagne reduce: reduce='sum', 'mean', 'max')
        out = scatter(messages, col, dim=0, dim_size=x.size(0), reduce='sum')
        return out







# Setup
edge_index = torch.tensor([
    [0, 1, 2, 3, 1, 2], # source
    [1, 2, 3, 0, 0, 1] # target
])

x = torch.randn(4, 8) # 4 nodes, 8-dim features

copresheaf_layer = CopresheafMessagePassing(
    in_channels=8,
    out_channels=16,
    edge_index=edge_index,
    edge_map_factory=mlp_factory,
)

out = copresheaf_layer(x)
print(out.shape) # [4, 16]