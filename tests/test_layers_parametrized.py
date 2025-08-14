# tests/test_layers_parametrized.py
import torch
import pytest
from models.cmpnn_base import CMPNNLayer, CHOMPNNLayer

@pytest.mark.parametrize("num_neighborhoods", [1, 2])
@pytest.mark.parametrize("dims", [(8, 8), (8, 16)])
def test_cmpnn_layer_param_grid(num_neighborhoods, dims, toy_edges_1, toy_edges_2):
    in_dim, out_dim = dims
    x = torch.randn(6, in_dim)
    edge_lists = [toy_edges_1] if num_neighborhoods == 1 else toy_edges_2
    layer = CMPNNLayer(in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=num_neighborhoods)
    out = layer(x, edge_lists, edge_attr_list=None)
    assert out.shape == (x.size(0), out_dim)

@pytest.mark.parametrize("num_neighborhoods", [1, 2])
def test_chompnn_layer_param_grid(num_neighborhoods, toy_edges_1, toy_edges_2):
    in_dim, out_dim = 8, 8
    x = torch.randn(6, in_dim)
    edge_lists = [toy_edges_1] if num_neighborhoods == 1 else toy_edges_2
    layer = CHOMPNNLayer(in_dim, out_dim, edge_attr_dim=0, num_neighborhoods=num_neighborhoods)
    out = layer(x, edge_lists, edge_attr_list=None)
    assert out.shape == (x.size(0), out_dim)
