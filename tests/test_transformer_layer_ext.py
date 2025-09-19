# tests/test_transformer_layer_ext.py
import torch
import pytest

try:
    from toposheafx.nn.layer import copresheaf_transformer_layer

    HAS_LAYER = True
except Exception:
    HAS_LAYER = False


@pytest.mark.skipif(not HAS_LAYER, reason="copresheaf_transformer_layer not available")
def test_sheaf_transformer_basic():
    """Test basic forward pass of copresheaf_transformer_layer."""
    n, d_in, d_out = 7, 8, 10
    x = torch.randn(n, d_in)
    e = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])

    layer = copresheaf_transformer_layer(
        in_dim=d_in,
        out_dim=d_out,
        edge_attr_dim=0,
        heads=2,
        dropout=0.0
    )

    y = layer(x, e, edge_attr=None)
    assert y.shape == (n, d_out)
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not HAS_LAYER, reason="copresheaf_transformer_layer not available")
def test_sheaf_transformer_grad():
    """Test gradient flow through copresheaf_transformer_layer."""
    n, d_in, d_out = 6, 4, 6
    x = torch.randn(n, d_in, requires_grad=True)
    e = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    layer = copresheaf_transformer_layer(
        in_dim=d_in,
        out_dim=d_out,
        edge_attr_dim=0,
        heads=1
    )

    y = layer(x, e, edge_attr=None)
    loss = y.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.all(x.grad == 0)

# # tests/test_transformer_layer_ext.py
# import torch
# import pytest
#
# try:
#     from toposheafx.nn.layer import transformer_layer
#     HAS_LAYER = True
# except Exception:
#     HAS_LAYER = False
#
# @pytest.mark.skipif(not HAS_LAYER, reason="SheafTransformerLayer not available")
# def test_sheaf_transformer_basic():
#     n, d_in, d_out = 7, 8, 10
#     x = torch.randn(n, d_in)
#     e = torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,6]])
#     layer = transformer_layer(in_dim=d_in, out_dim=d_out, edge_attr_dim=0, heads=2, dropout=0.0)
#     y = layer(x, e, edge_attr=None)
#     assert y.shape == (n, d_out)
#
# @pytest.mark.skipif(not HAS_LAYER, reason="SheafTransformerLayer not available")
# def test_sheaf_transformer_grad():
#     n, d_in, d_out = 6, 4, 6
#     x = torch.randn(n, d_in, requires_grad=True)
#     e = torch.tensor([[0,1,2,3,4],[1,2,3,4,5]])
#     layer = transformer_layer(in_dim=d_in, out_dim=d_out, edge_attr_dim=0, heads=1)
#     y = layer(x, e, edge_attr=None)
#     loss = y.pow(2).mean()
#     loss.backward()
#     assert x.grad is not None
