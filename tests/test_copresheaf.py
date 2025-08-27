
import torch

# These modules live in models/ per your structure
from models.copresheaf import CopresheafMap, CopresheafHigherOrderMap

torch.manual_seed(0)

def test_copresheaf_map_forward_and_aggregate_sum_mean_max():
    in_dim, out_dim = 8, 5
    layer = CopresheafMap(in_dim, out_dim)

    # Simple 4-node, 5-edge toy graph
    edge_index = torch.tensor([[0, 1, 2, 2, 3],
                               [1, 2, 3, 1, 0]])
    row, col = edge_index
    x = torch.randn(4, in_dim)

    messages = layer(x[row], x[col], edge_attr=None)
    assert messages.shape == (edge_index.size(1), out_dim)

    agg_sum  = layer.aggregate_messages(messages, col, num_nodes=x.size(0), reduce="sum")
    agg_mean = layer.aggregate_messages(messages, col, num_nodes=x.size(0), reduce="mean")
    agg_max  = layer.aggregate_messages(messages, col, num_nodes=x.size(0), reduce="max")

    for agg in (agg_sum, agg_mean, agg_max):
        assert agg.shape == (x.size(0), out_dim)

def test_copresheaf_map_update_shapes_and_grad():
    in_dim, out_dim = 7, 11
    layer = CopresheafMap(in_dim, out_dim)

    num_nodes = 6
    x = torch.randn(num_nodes, in_dim, requires_grad=True)
    # fake aggregated messages
    aggr = torch.randn(num_nodes, out_dim)

    out = layer.update_node(x, aggr)
    assert out.shape == (num_nodes, out_dim)

    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None

def test_higher_order_map_combines_neighborhoods():
    in_dim, out_dim = 6, 10
    # two neighborhoods (e.g., adjacency + incidence)
    ho = CopresheafHigherOrderMap(in_dim, out_dim, num_neighborhoods=2)

    x = torch.randn(5, in_dim)

    # neighborhood 1: chain
    e1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    # neighborhood 2: star to node 0
    e2 = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]])

    out = ho(x, [e1, e2], edge_attr_list=None)
    assert out.shape == (x.size(0), out_dim)
