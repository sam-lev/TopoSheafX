# tests/test_conv.py
"""
Comprehensive test suite for toposheafx/base/conv.py
Tests Conv and HigherOrderConv classes.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from toposheafx.base.conv import Conv, HigherOrderConv
from toposheafx.base.aggregation import Aggregation


class TestConv:
    """Test suite for Conv class."""

    @pytest.fixture
    def setup_conv(self):
        """Setup Conv test configuration."""
        torch.manual_seed(42)
        in_dim = 8
        out_dim = 16
        num_nodes = 4
        num_edges = 5

        x = torch.randn(num_nodes, in_dim)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 1],  # source
            [1, 2, 3, 0, 0]  # target
        ])

        return {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'x': x,
            'edge_index': edge_index,
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }

    def test_initialization(self, setup_conv):
        """Test Conv initialization."""
        conv = Conv(
            in_dim=setup_conv['in_dim'],
            out_dim=setup_conv['out_dim']
        )

        assert conv.in_dim == setup_conv['in_dim']
        assert conv.out_dim == setup_conv['out_dim']
        assert hasattr(conv, 'bias')
        assert conv.bias.shape == (setup_conv['out_dim'],)

    def test_initialization_no_bias(self, setup_conv):
        """Test Conv initialization without bias."""
        conv = Conv(
            in_dim=setup_conv['in_dim'],
            out_dim=setup_conv['out_dim'],
            bias=False
        )

        assert conv.bias is None

    def test_forward_pass(self, setup_conv):
        """Test forward pass of Conv."""
        conv = Conv(
            in_dim=setup_conv['in_dim'],
            out_dim=setup_conv['out_dim']
        )

        output = conv(setup_conv['x'], setup_conv['edge_index'])
        assert output.shape == (setup_conv['num_nodes'], setup_conv['out_dim'])
        assert torch.isfinite(output).all()

    def test_forward_with_edge_attributes(self, setup_conv):
        """Test forward pass with edge attributes."""
        edge_attr_dim = 3
        conv = Conv(
            in_dim=setup_conv['in_dim'],
            out_dim=setup_conv['out_dim'],
            edge_attr_dim=edge_attr_dim
        )

        edge_attr = torch.randn(setup_conv['num_edges'], edge_attr_dim)
        output = conv(setup_conv['x'], setup_conv['edge_index'], edge_attr)

        assert output.shape == (setup_conv['num_nodes'], setup_conv['out_dim'])
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize('aggr', ['sum', 'mean', 'max'])
    def test_aggregation_methods(self, setup_conv, aggr):
        """Test different aggregation methods in Conv."""
        conv = Conv(
            in_dim=setup_conv['in_dim'],
            out_dim=setup_conv['out_dim'],
            aggr=aggr
        )

        output = conv(setup_conv['x'], setup_conv['edge_index'])
        assert output.shape == (setup_conv['num_nodes'], setup_conv['out_dim'])

    def test_reset_parameters(self, setup_conv):
        """Test parameter reset."""
        conv = Conv(
            in_dim=setup_conv['in_dim'],
            out_dim=setup_conv['out_dim']
        )

        # Store initial parameters
        initial_bias = conv.bias.clone() if conv.bias is not None else None

        # Reset parameters
        conv.reset_parameters()

        # Check bias is reset to zeros
        if conv.bias is not None:
            assert torch.allclose(conv.bias, torch.zeros_like(conv.bias))

    def test_backward_compatibility_aggregate(self, setup_conv):
        """Test backward compatibility of aggregate_messages method."""
        conv = Conv(
            in_dim=setup_conv['in_dim'],
            out_dim=setup_conv['out_dim']
        )

        messages = torch.randn(setup_conv['num_edges'], setup_conv['out_dim'])
        index = setup_conv['edge_index'][1]  # target indices

        aggregated = conv.aggregate_messages(
            messages,
            index,
            setup_conv['num_nodes']
        )

        assert aggregated.shape == (setup_conv['num_nodes'], setup_conv['out_dim'])

    def test_backward_compatibility_update(self, setup_conv):
        """Test backward compatibility of update_node method."""
        conv = Conv(
            in_dim=setup_conv['in_dim'],
            out_dim=setup_conv['out_dim']
        )

        aggr_msg = torch.randn(setup_conv['num_nodes'], setup_conv['out_dim'])

        updated = conv.update_node(setup_conv['x'], aggr_msg)
        assert updated.shape == (setup_conv['num_nodes'], setup_conv['out_dim'])

    def test_gradient_flow(self, setup_conv):
        """Test that gradients flow through Conv."""
        conv = Conv(
            in_dim=setup_conv['in_dim'],
            out_dim=setup_conv['out_dim']
        )

        x = setup_conv['x'].requires_grad_(True)
        output = conv(x, setup_conv['edge_index'])
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        # Check that model parameters have gradients
        for param in conv.parameters():
            assert param.grad is not None

    def test_custom_factories(self, setup_conv):
        """Test Conv with custom factory functions."""

        class CustomConv(Conv):
            def _conv_copresheaf_factory(self, in_dim, out_dim, edge_attr_dim):
                return nn.Sequential(
                    nn.Linear(2 * in_dim + edge_attr_dim, out_dim * 2),
                    nn.ReLU(),
                    nn.Linear(out_dim * 2, out_dim)
                )

        conv = CustomConv(
            in_dim=setup_conv['in_dim'],
            out_dim=setup_conv['out_dim']
        )

        output = conv(setup_conv['x'], setup_conv['edge_index'])
        assert output.shape == (setup_conv['num_nodes'], setup_conv['out_dim'])


class TestHigherOrderConv:
    """Test suite for HigherOrderConv class."""

    @pytest.fixture
    def setup_higher_order(self):
        """Setup HigherOrderConv test configuration."""
        torch.manual_seed(42)
        in_dim = 8
        out_dim = 16
        num_nodes = 4
        num_neighborhoods = 3

        x = torch.randn(num_nodes, in_dim)

        edge_index_list = [
            torch.tensor([[0, 1, 2], [1, 2, 3]]),  # Adjacency
            torch.tensor([[0, 2], [2, 0]]),  # Incidence
            torch.tensor([[1, 3], [3, 1]])  # Co-boundary
        ]

        return {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'x': x,
            'edge_index_list': edge_index_list,
            'num_nodes': num_nodes,
            'num_neighborhoods': num_neighborhoods
        }

    def test_initialization(self, setup_higher_order):
        """Test HigherOrderConv initialization."""
        ho_conv = HigherOrderConv(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods']
        )

        assert ho_conv.in_dim == setup_higher_order['in_dim']
        assert ho_conv.out_dim == setup_higher_order['out_dim']
        assert ho_conv.num_neighborhoods == setup_higher_order['num_neighborhoods']
        assert len(ho_conv.neighborhood_layers) == setup_higher_order['num_neighborhoods']
        assert hasattr(ho_conv, 'bias')

    def test_forward_pass(self, setup_higher_order):
        """Test forward pass of HigherOrderConv."""
        ho_conv = HigherOrderConv(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods']
        )

        output = ho_conv(
            setup_higher_order['x'],
            setup_higher_order['edge_index_list']
        )

        assert output.shape == (setup_higher_order['num_nodes'], setup_higher_order['out_dim'])
        assert torch.isfinite(output).all()

    def test_forward_with_edge_attributes(self, setup_higher_order):
        """Test forward pass with edge attributes."""
        edge_attr_dim = 3
        ho_conv = HigherOrderConv(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods'],
            edge_attr_dim=edge_attr_dim
        )

        edge_attr_list = [
            torch.randn(edge_index.size(1), edge_attr_dim)
            for edge_index in setup_higher_order['edge_index_list']
        ]

        output = ho_conv(
            setup_higher_order['x'],
            setup_higher_order['edge_index_list'],
            edge_attr_list
        )

        assert output.shape == (setup_higher_order['num_nodes'], setup_higher_order['out_dim'])

    @pytest.mark.parametrize('combine_method', ['concat', 'sum', 'mean'])
    def test_combine_methods(self, setup_higher_order, combine_method):
        """Test different combination methods."""
        ho_conv = HigherOrderConv(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods'],
            combine_method=combine_method
        )

        output = ho_conv(
            setup_higher_order['x'],
            setup_higher_order['edge_index_list']
        )

        assert output.shape == (setup_higher_order['num_nodes'], setup_higher_order['out_dim'])

    def test_custom_edge_map(self, setup_higher_order):
        """Test with custom Edge_Map class."""

        class CustomEdgeMap(Conv):
            def __init__(self, in_dim, out_dim, edge_attr_dim=0, aggr='sum', bias=False):
                super().__init__(in_dim, out_dim, edge_attr_dim, aggr, bias)
                self.custom_layer = nn.Linear(out_dim, out_dim)

            def forward(self, x, edge_index, edge_attr=None, size=None):
                out = super().forward(x, edge_index, edge_attr, size)
                return self.custom_layer(out)

        ho_conv = HigherOrderConv(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods'],
            Edge_Map=CustomEdgeMap
        )

        output = ho_conv(
            setup_higher_order['x'],
            setup_higher_order['edge_index_list']
        )

        assert output.shape == (setup_higher_order['num_nodes'], setup_higher_order['out_dim'])

    def test_reset_parameters(self, setup_higher_order):
        """Test parameter reset."""
        ho_conv = HigherOrderConv(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods']
        )

        ho_conv.reset_parameters()

        # Check bias is reset to zeros
        if ho_conv.bias is not None:
            assert torch.allclose(ho_conv.bias, torch.zeros_like(ho_conv.bias))

    def test_gradient_flow(self, setup_higher_order):
        """Test gradient flow through HigherOrderConv."""
        ho_conv = HigherOrderConv(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods']
        )

        x = setup_higher_order['x'].requires_grad_(True)
        output = ho_conv(x, setup_higher_order['edge_index_list'])
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        # Check that all neighborhood layers have gradients
        for layer in ho_conv.neighborhood_layers:
            for param in layer.parameters():
                assert param.grad is not None

    def test_different_aggregations_per_neighborhood(self, setup_higher_order):
        """Test different aggregation methods per neighborhood."""
        ho_conv = HigherOrderConv(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods'],
            aggr='sum'  # Can be overridden per neighborhood in base class
        )

        output = ho_conv(
            setup_higher_order['x'],
            setup_higher_order['edge_index_list']
        )

        assert output.shape == (setup_higher_order['num_nodes'], setup_higher_order['out_dim'])


class TestConvIntegration:
    """Integration tests for Conv components."""

    def test_multi_layer_conv(self):
        """Test stacking multiple Conv layers."""
        torch.manual_seed(42)

        in_dim, hidden_dim, out_dim = 8, 16, 4
        num_nodes = 10
        edge_index = torch.randint(0, num_nodes, (2, 20))
        x = torch.randn(num_nodes, in_dim)

        # Build multi-layer model
        conv1 = Conv(in_dim, hidden_dim)
        conv2 = Conv(hidden_dim, hidden_dim)
        conv3 = Conv(hidden_dim, out_dim)

        # Forward pass
        h = conv1(x, edge_index)
        h = torch.relu(h)
        h = conv2(h, edge_index)
        h = torch.relu(h)
        h = conv3(h, edge_index)

        assert h.shape == (num_nodes, out_dim)
        assert torch.isfinite(h).all()

    def test_large_graph_conv(self):
        """Test Conv on large graphs."""
        torch.manual_seed(42)

        in_dim, out_dim = 32, 64
        num_nodes = 1000
        num_edges = 5000

        x = torch.randn(num_nodes, in_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        conv = Conv(in_dim, out_dim)
        output = conv(x, edge_index)

        assert output.shape == (num_nodes, out_dim)
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])