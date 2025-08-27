# tests/test_copresheaf_ext.py
import torch
from models.copresheaf import CopresheafMap, CopresheafHigherOrderMap

def test_copresheaf_map_all_aggregators(toy_edges_1):
    in_dim, out_dim = 8, 5
    x = torch.randn(6, in_dim)
    row, col = toy_edges_1
    layer = CopresheafMap(in_dim, out_dim)

    msgs = layer(x[row], x[col], edge_attr=None)
    assert msgs.shape == (toy_edges_1.size(1), out_dim)

    for reduce in ("sum", "mean", "max"):
        aggr = layer.aggregate_messages(msgs, col, num_nodes=x.size(0), reduce=reduce)
        assert aggr.shape == (x.size(0), out_dim)

    # Update should preserve node count and out_dim
    out = layer.update_node(x, aggr)
    assert out.shape == (x.size(0), out_dim)

def test_copresheaf_map_grad_flow(toy_edges_1):
    in_dim, out_dim = 6, 6
    x = torch.randn(6, in_dim, requires_grad=True)
    row, col = toy_edges_1
    layer = CopresheafMap(in_dim, out_dim)

    msgs = layer(x[row], x[col], edge_attr=None)
    aggr = layer.aggregate_messages(msgs, col, num_nodes=x.size(0), reduce="sum")
    out = layer.update_node(x, aggr)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_copresheaf_higher_order_two_neighborhoods(toy_edges_2):
    in_dim, out_dim = 7, 9
    x = torch.randn(6, in_dim)
    ho = CopresheafHigherOrderMap(in_dim, out_dim, num_neighborhoods=2)
    out = ho(x, toy_edges_2, edge_attr_list=None)
    assert out.shape == (x.size(0), out_dim)
