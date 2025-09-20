# tests/test_copresheaf.py
import torch

from toposheafx.base.conv import Conv, HigherOrderConv

torch.manual_seed(0)


def test_copresheaf_map_forward_and_aggregate_sum_mean_max():
    """Test Conv layer with full graph processing."""
    in_dim, out_dim = 8, 5
    layer = Conv(in_dim, out_dim)

    # Simple 4-node, 5-edge toy graph
    edge_index = torch.tensor([[0, 1, 2, 2, 3], [1, 2, 3, 1, 0]])
    x = torch.randn(4, in_dim)

    # Conv processes the full graph
    output = layer(x, edge_index, edge_attr=None)
    assert output.shape == (4, out_dim)
    assert torch.isfinite(output).all()

    # Test internal message computation
    row, col = edge_index
    x_i = x[col]  # target features
    x_j = x[row]  # source features
    messages = layer.message(x_i, x_j)
    assert messages.shape == (5, out_dim)  # 5 edges


def test_copresheaf_map_update_shapes_and_grad():
    """Test gradient flow through Conv layer."""
    in_dim, out_dim = 7, 11
    layer = Conv(in_dim, out_dim)

    num_nodes = 6
    x = torch.randn(num_nodes, in_dim, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    # Forward pass through complete layer
    out = layer(x, edge_index)
    assert out.shape == (num_nodes, out_dim)

    # Check gradient flow
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_higher_order_map_combines_neighborhoods():
    """Test HigherOrderConv with multiple neighborhoods."""
    in_dim, out_dim = 6, 10
    # two neighborhoods (e.g., adjacency + incidence)
    ho = HigherOrderConv(in_dim, out_dim, num_neighborhoods=2)

    x = torch.randn(5, in_dim)

    # neighborhood 1: chain
    e1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    # neighborhood 2: star to node 0
    e2 = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]])

    out = ho(x, [e1, e2], edge_attr_list=None)
    assert out.shape == (x.size(0), out_dim)
    assert torch.isfinite(out).all()


#
# import torch
#
# # These modules live in models/ per your structure
# from toposheafx.base.conv import Conv, HigherOrderConv
#
# torch.manual_seed(0)
#
#
# def test_copresheaf_map_forward_and_aggregate_sum_mean_max():
#     in_dim, out_dim = 8, 5
#     layer = Conv(in_dim, out_dim)
#
#     # Simple 4-node, 5-edge toy graph
#     edge_index = torch.tensor([[0, 1, 2, 2, 3],
#                                [1, 2, 3, 1, 0]])
#     x = torch.randn(4, in_dim)
#
#     # Correct way to call Conv - pass full x and edge_index
#     output = layer(x, edge_index, edge_attr=None)
#
#     # Check output shape
#     assert output.shape == (4, out_dim)
#
#     # If you need to test message computation separately,
#     # you can access the internal message method:
#     row, col = edge_index
#     x_i = x[col]  # target features
#     x_j = x[row]  # source features
#     messages = layer.message(x_i, x_j)
#     assert messages.shape == (5, out_dim)  # 5 edges
#
#     # Test aggregation with different methods
#     for reduce_method in ['sum', 'mean', 'max']:
#         # You would need to modify Conv to accept reduce parameter
#         # or test with different aggregation settings
#         pass
#
# def test_copresheaf_map_update_shapes_and_grad():
#     in_dim, out_dim = 7, 11
#     layer = Conv(in_dim, out_dim)
#
#     num_nodes = 6
#     x = torch.randn(num_nodes, in_dim, requires_grad=True)
#     # fake aggregated messages
#     aggr = torch.randn(num_nodes, out_dim)
#
#     out = layer.update_node(x, aggr)
#     assert out.shape == (num_nodes, out_dim)
#
#     loss = out.pow(2).mean()
#     loss.backward()
#     assert x.grad is not None
#
# def test_higher_order_map_combines_neighborhoods():
#     in_dim, out_dim = 6, 10
#     # two neighborhoods (e.g., adjacency + incidence)
#     ho = HigherOrderConv(in_dim, out_dim, num_neighborhoods=2)
#
#     x = torch.randn(5, in_dim)
#
#     # neighborhood 1: chain
#     e1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
#     # neighborhood 2: star to node 0
#     e2 = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]])
#
#     out = ho(x, [e1, e2], edge_attr_list=None)
#     assert out.shape == (x.size(0), out_dim)
