# tests/test_layers_parametrized.py
import torch
import pytest
from toposheafx.nn.layer import cmpnn_layer, chompnn_layer


@pytest.fixture
def toy_edges_1():
    """Simple edge index for testing."""
    return torch.tensor([[0, 1, 2, 3, 4],
                         [1, 2, 3, 4, 5]])


@pytest.fixture
def toy_edges_2():
    """Two edge indices for multi-neighborhood testing."""
    e1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    e2 = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]])
    return [e1, e2]


@pytest.mark.parametrize("num_neighborhoods", [1, 2])
@pytest.mark.parametrize("dims", [(8, 8), (8, 16)])
def test_cmpnn_layer_param_grid(num_neighborhoods, dims, toy_edges_1, toy_edges_2):
    """Test cmpnn_layer with different configurations."""
    in_dim, out_dim = dims
    x = torch.randn(6, in_dim)
    edge_lists = [toy_edges_1] if num_neighborhoods == 1 else toy_edges_2

    layer = cmpnn_layer(in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=num_neighborhoods)
    out = layer(x, edge_lists, edge_attr_list=None)

    assert out.shape == (x.size(0), out_dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("num_neighborhoods", [1, 2])
def test_chompnn_layer_param_grid(num_neighborhoods, toy_edges_1, toy_edges_2):
    """Test chompnn_layer with different configurations."""
    in_dim, out_dim = 8, 8
    x = torch.randn(6, in_dim)
    edge_lists = [toy_edges_1] if num_neighborhoods == 1 else toy_edges_2

    layer = chompnn_layer(in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=num_neighborhoods)
    out = layer(x, edge_lists, edge_attr_list=None)

    assert out.shape == (x.size(0), out_dim)
    assert torch.isfinite(out).all()

# # tests/test_layers_parametrized.py
# import torch
# import pytest
# from toposheafx.nn.layer import cmpnn_layer, chompnn_layer
#
# @pytest.mark.parametrize("num_neighborhoods", [1, 2])
# @pytest.mark.parametrize("dims", [(8, 8), (8, 16)])
# def test_cmpnn_layer_param_grid(num_neighborhoods, dims, toy_edges_1, toy_edges_2):
#     in_dim, out_dim = dims
#     x = torch.randn(6, in_dim)
#     edge_lists = [toy_edges_1] if num_neighborhoods == 1 else toy_edges_2
#     layer = cmpnn_layer(in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=num_neighborhoods)
#     out = layer(x, edge_lists, edge_attr_list=None)
#     assert out.shape == (x.size(0), out_dim)
#
# @pytest.mark.parametrize("num_neighborhoods", [1, 2])
# def test_chompnn_layer_param_grid(num_neighborhoods, toy_edges_1, toy_edges_2):
#     in_dim, out_dim = 8, 8
#     x = torch.randn(6, in_dim)
#     edge_lists = [toy_edges_1] if num_neighborhoods == 1 else toy_edges_2
#     layer = chompnn_layer(in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=num_neighborhoods)
#     out = layer(x, edge_lists, edge_attr_list=None)
#     assert out.shape == (x.size(0), out_dim)
