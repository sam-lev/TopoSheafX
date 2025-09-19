# tests/test_models_coverage.py
import torch
import pytest
from toposheafx.nn.model import CMPNN, CHOMPNN

def _edge_lists_pair():
    e1 = torch.tensor([[0,1,2,3,4],[1,2,3,4,5]])
    e2 = torch.tensor([[1,2,3,4,5],[0,0,0,0,0]])
    return [e1, e2]

def _batch(n):
    b = torch.zeros(n, dtype=torch.long)
    b[n//2:] = 1
    return b

@pytest.mark.parametrize("Model", [CMPNN, CHOMPNN])
def test_models_graph_level_and_backward(Model):
    in_dim, hidden, out_dim = 8, 16, 3
    model = Model(in_dim, hidden, out_dim, num_layers=2, edge_attr_dim=0, num_neighborhoods=2)
    x = torch.randn(6, in_dim, requires_grad=True)
    edge_lists = _edge_lists_pair()
    batch = _batch(6)

    y_hat = model(x=x, edge_index_list=edge_lists, batch=batch, edge_attr_list=None)
    assert y_hat.shape == (2, out_dim)

    loss = y_hat.pow(2).mean()
    loss.backward()
    assert x.grad is not None

def test_model_to_cpu_roundtrip():
    model = CMPNN(in_dim=4, hidden_dim=8, out_dim=2, num_layers=1, num_neighborhoods=1)
    model = model.to("cpu")
    x = torch.randn(4, 4)
    e = torch.tensor([[0,1,2,3],[1,2,3,0]])
    out = model(x=x, edge_index_list=[e], batch=torch.zeros(4, dtype=torch.long))
    assert out.shape == (1, 2)
