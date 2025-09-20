# tests/test_copresheaf_ext.py
import pytest
import torch

from toposheafx.base.conv import Conv, HigherOrderConv


@pytest.fixture
def toy_edges_1():
    """Simple edge index for testing."""
    return torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])


@pytest.fixture
def toy_edges_2():
    """Two edge indices for multi-neighborhood testing."""
    e1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    e2 = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]])
    return [e1, e2]


def test_copresheaf_map_all_aggregators(toy_edges_1):
    """Test Conv with different aggregation methods."""
    in_dim, out_dim = 8, 5
    x = torch.randn(6, in_dim)

    # Test with different aggregation methods
    for aggr_method in ["sum", "mean", "max"]:
        layer = Conv(in_dim, out_dim, aggr=aggr_method)
        out = layer(x, toy_edges_1)
        assert out.shape == (x.size(0), out_dim)
        assert torch.isfinite(out).all()


def test_copresheaf_map_grad_flow(toy_edges_1):
    """Test gradient flow through Conv layer."""
    in_dim, out_dim = 6, 6
    x = torch.randn(6, in_dim, requires_grad=True)
    layer = Conv(in_dim, out_dim)

    # Full forward pass
    out = layer(x, toy_edges_1)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.all(x.grad == 0)


def test_copresheaf_higher_order_two_neighborhoods(toy_edges_2):
    """Test HigherOrderConv with two neighborhoods."""
    in_dim, out_dim = 7, 9
    x = torch.randn(6, in_dim)
    ho = HigherOrderConv(in_dim, out_dim, num_neighborhoods=2)
    out = ho(x, toy_edges_2, edge_attr_list=None)
    assert out.shape == (x.size(0), out_dim)
    assert torch.isfinite(out).all()


# # tests/test_copresheaf_ext.py
# import torch
# from toposheafx.base.conv import Conv, HigherOrderConv
#
# def test_copresheaf_map_all_aggregators(toy_edges_1):
#     in_dim, out_dim = 8, 5
#     x = torch.randn(6, in_dim)
#     row, col = toy_edges_1
#     layer = Conv(in_dim, out_dim)
#
#     msgs = layer(x[row], x[col], edge_attr=None)
#     assert msgs.shape == (toy_edges_1.size(1), out_dim)
#
#     for reduce in ("sum", "mean", "max"):
#         aggr = layer.aggregate_messages(msgs, col, num_nodes=x.size(0), reduce=reduce)
#         assert aggr.shape == (x.size(0), out_dim)
#
#     # Update should preserve node count and out_dim
#     out = layer.update_node(x, aggr)
#     assert out.shape == (x.size(0), out_dim)
#
# def test_copresheaf_map_grad_flow(toy_edges_1):
#     in_dim, out_dim = 6, 6
#     x = torch.randn(6, in_dim, requires_grad=True)
#     row, col = toy_edges_1
#     layer = Conv(in_dim, out_dim)
#
#     msgs = layer(x[row], x[col], edge_attr=None)
#     aggr = layer.aggregate_messages(msgs, col, num_nodes=x.size(0), reduce="sum")
#     out = layer.update_node(x, aggr)
#     loss = out.pow(2).mean()
#     loss.backward()
#     assert x.grad is not None and torch.isfinite(x.grad).all()
#
# def test_copresheaf_higher_order_two_neighborhoods(toy_edges_2):
#     in_dim, out_dim = 7, 9
#     x = torch.randn(6, in_dim)
#     ho = HigherOrderConv(in_dim, out_dim, num_neighborhoods=2)
#     out = ho(x, toy_edges_2, edge_attr_list=None)
#     assert out.shape == (x.size(0), out_dim)
