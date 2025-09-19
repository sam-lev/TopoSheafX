# tests/test_cmpnn.py
"""
Test file for CMPNN and CHOMPNN models.
Tests basic forward pass and gradient flow.
"""

import torch
import pytest

# Import from the new structure
from toposheafx.nn.model import CMPNN, CHOMPNN

torch.manual_seed(0)


def _toy_edge_lists(num_nodes=6):
    """Create simple test neighborhoods."""
    # Two simple neighborhoods
    e1 = torch.tensor([[0, 1, 2, 3, 4],
                       [1, 2, 3, 4, 5]])  # chain
    e2 = torch.tensor([[1, 2, 3, 4, 5],
                       [0, 0, 0, 0, 0]])  # star to node 0
    return [e1, e2]


def _toy_batch(n):
    """Group nodes into two graphs: first half -> graph 0, rest -> graph 1."""
    b = torch.zeros(n, dtype=torch.long)
    b[n // 2:] = 1
    return b


@pytest.mark.parametrize("Model", [CMPNN, CHOMPNN])
def test_models_forward_graph_level_output(Model):
    """Test that models produce correct output shape for graph-level tasks."""
    in_dim, hidden, out_dim = 8, 16, 3
    num_layers = 2
    model = Model(
        in_dim=in_dim,
        hidden_dim=hidden,
        out_dim=out_dim,
        num_layers=num_layers,
        edge_attr_dim=0,
        num_neighborhoods=2
    )

    num_nodes = 6
    x = torch.randn(num_nodes, in_dim)
    edge_index_list = _toy_edge_lists(num_nodes)
    batch = _toy_batch(num_nodes)

    out = model(x=x, edge_index_list=edge_index_list, batch=batch, edge_attr_list=None)
    # two graphs in batch, so output should be [2, out_dim]
    assert out.shape == (2, out_dim)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf values"


def test_models_backward_pass():
    """Test that gradients flow through the model correctly."""
    in_dim, hidden, out_dim = 4, 8, 2
    model = CMPNN(
        in_dim=in_dim,
        hidden_dim=hidden,
        out_dim=out_dim,
        num_layers=2,
        edge_attr_dim=0,
        num_neighborhoods=2
    )

    num_nodes = 8
    x = torch.randn(num_nodes, in_dim, requires_grad=True)
    edge_index_list = _toy_edge_lists(num_nodes)
    batch = _toy_batch(num_nodes)

    y_hat = model(x=x, edge_index_list=edge_index_list, batch=batch, edge_attr_list=None)
    loss = y_hat.pow(2).mean()
    loss.backward()

    # Check that gradients flow to input
    assert x.grad is not None, "No gradient computed for input x"
    assert not torch.all(x.grad == 0), "All gradients are zero"

    # Check that model parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter {name}"


def test_cmpnn_with_edge_attributes():
    """Test CMPNN with edge attributes."""
    in_dim, hidden, out_dim = 8, 16, 3
    edge_attr_dim = 4

    model = CMPNN(
        in_dim=in_dim,
        hidden_dim=hidden,
        out_dim=out_dim,
        num_layers=2,
        edge_attr_dim=edge_attr_dim,
        num_neighborhoods=2
    )

    num_nodes = 6
    x = torch.randn(num_nodes, in_dim)
    edge_index_list = _toy_edge_lists(num_nodes)

    # Create edge attributes for each neighborhood
    edge_attr_list = [
        torch.randn(e.size(1), edge_attr_dim) for e in edge_index_list
    ]

    batch = _toy_batch(num_nodes)

    out = model(x=x, edge_index_list=edge_index_list, batch=batch, edge_attr_list=edge_attr_list)
    assert out.shape == (2, out_dim)


def test_chompnn_higher_order():
    """Test CHOMPNN with multiple neighborhoods."""
    in_dim, hidden, out_dim = 8, 16, 4
    num_neighborhoods = 3

    model = CHOMPNN(
        in_dim=in_dim,
        hidden_dim=hidden,
        out_dim=out_dim,
        num_layers=3,
        num_neighborhoods=num_neighborhoods
    )

    num_nodes = 8
    x = torch.randn(num_nodes, in_dim)

    # Create three different neighborhoods
    edge_index_list = [
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),  # chain
        torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]]),  # bipartite-like
        torch.tensor([[0, 0, 0], [5, 6, 7]]),  # star from 0
    ]

    batch = _toy_batch(num_nodes)

    out = model(x=x, edge_index_list=edge_index_list, batch=batch)
    assert out.shape == (2, out_dim)


def test_single_graph_batch():
    """Test models with a single graph (no batching)."""
    in_dim, hidden, out_dim = 8, 16, 3

    model = CMPNN(
        in_dim=in_dim,
        hidden_dim=hidden,
        out_dim=out_dim,
        num_layers=2,
        num_neighborhoods=2
    )

    num_nodes = 10
    x = torch.randn(num_nodes, in_dim)
    edge_index_list = _toy_edge_lists(num_nodes)

    # All nodes belong to the same graph
    batch = torch.zeros(num_nodes, dtype=torch.long)

    out = model(x=x, edge_index_list=edge_index_list, batch=batch)
    assert out.shape == (1, out_dim), "Single graph should produce [1, out_dim] output"


def test_empty_neighborhoods():
    """Test handling of empty neighborhoods."""
    in_dim, hidden, out_dim = 8, 16, 3

    model = CMPNN(
        in_dim=in_dim,
        hidden_dim=hidden,
        out_dim=out_dim,
        num_layers=1,
        num_neighborhoods=2
    )

    num_nodes = 4
    x = torch.randn(num_nodes, in_dim)

    # One normal neighborhood, one empty
    edge_index_list = [
        torch.tensor([[0, 1], [1, 2]]),  # normal edges
        torch.tensor([[], []]).long(),  # empty neighborhood
    ]

    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Should handle empty neighborhoods gracefully
    out = model(x=x, edge_index_list=edge_index_list, batch=batch)
    assert out.shape == (1, out_dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("num_layers", [1, 2, 5])
def test_variable_depth(num_layers):
    """Test models with different depths."""
    model = CMPNN(
        in_dim=8,
        hidden_dim=16,
        out_dim=3,
        num_layers=num_layers,
        num_neighborhoods=1
    )

    x = torch.randn(10, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    batch = torch.zeros(10, dtype=torch.long)

    out = model(x=x, edge_index_list=[edge_index], batch=batch)
    assert out.shape == (1, 3)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])