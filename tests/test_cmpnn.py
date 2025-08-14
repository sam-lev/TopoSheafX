import torch
import pytest

from models.cmpnn_base import CMPNN, CHOMPNN

torch.manual_seed(0)

def _toy_edge_lists(num_nodes=6):
    # Two simple neighborhoods
    e1 = torch.tensor([[0, 1, 2, 3, 4],
                       [1, 2, 3, 4, 5]])  # chain
    e2 = torch.tensor([[1, 2, 3, 4, 5],
                       [0, 0, 0, 0, 0]])  # star to node 0
    return [e1, e2]

def _toy_batch(n):
    # group nodes into two graphs: first half -> graph 0, rest -> graph 1
    b = torch.zeros(n, dtype=torch.long)
    b[n//2:] = 1
    return b

@pytest.mark.parametrize("Model", [CMPNN, CHOMPNN])
def test_models_forward_graph_level_output(Model):
    in_dim, hidden, out_dim = 8, 16, 3
    num_layers = 2
    model = Model(in_dim, hidden, out_dim, num_layers=num_layers, edge_attr_dim=0, num_neighborhoods=2)

    num_nodes = 6
    x = torch.randn(num_nodes, in_dim)
    edge_index_list = _toy_edge_lists(num_nodes)
    batch = _toy_batch(num_nodes)

    out = model(x=x, edge_index_list=edge_index_list, batch=batch, edge_attr_list=None)
    # two graphs in batch, so output should be [2, out_dim]
    assert out.shape == (2, out_dim)

def test_models_backward_pass():
    in_dim, hidden, out_dim = 4, 8, 2
    model = CMPNN(in_dim, hidden, out_dim, num_layers=2, edge_attr_dim=0, num_neighborhoods=2)

    num_nodes = 8
    x = torch.randn(num_nodes, in_dim, requires_grad=True)
    edge_index_list = _toy_edge_lists(num_nodes)
    batch = _toy_batch(num_nodes)

    y_hat = model(x=x, edge_index_list=edge_index_list, batch=batch, edge_attr_list=None)
    loss = y_hat.pow(2).mean()
    loss.backward()

    # Check at least something in the graph received gradient
    assert x.grad is not None
