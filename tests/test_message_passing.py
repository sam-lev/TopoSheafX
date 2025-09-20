"""
Comprehensive test suite for toposheafx/base/message_passing.py
Tests MessagePassing and HigherOrderMessagePassing classes with various aggregation strategies.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from toposheafx.base.aggregation import Aggregation, AggregationFactory

# Import the classes to test
from toposheafx.base.message_passing import HigherOrderMessagePassing, MessagePassing


class TestMessagePassing:
    """Test suite for MessagePassing class."""

    @pytest.fixture
    def setup_basic(self):
        """Setup basic test configuration."""
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

    def test_initialization_default(self, setup_basic):
        """Test default initialization of MessagePassing."""
        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim']
        )

        assert mp.in_dim == setup_basic['in_dim']
        assert mp.out_dim == setup_basic['out_dim']
        assert mp.edge_attr_dim == 0
        assert hasattr(mp, 'copresheaf_map')
        assert hasattr(mp, 'message_fn')
        assert hasattr(mp, 'update_fn')
        assert hasattr(mp, 'aggregation')

    def test_initialization_with_edge_attrs(self, setup_basic):
        """Test initialization with edge attributes."""
        edge_attr_dim = 3
        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim'],
            edge_attr_dim=edge_attr_dim
        )

        assert mp.edge_attr_dim == edge_attr_dim

        # Test that copresheaf map accepts correct input dimension
        rho_input_dim = 2 * setup_basic['in_dim'] + edge_attr_dim
        test_input = torch.randn(1, rho_input_dim)
        output = mp.copresheaf_map(test_input)
        assert output.shape == (1, setup_basic['out_dim'])

    @pytest.mark.parametrize('aggr_method', ['sum', 'mean', 'max', 'min', 'std', 'var'])
    def test_aggregation_methods(self, setup_basic, aggr_method):
        """Test different aggregation methods."""
        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim'],
            aggr=aggr_method
        )

        assert mp.aggregation.method == aggr_method

        # Test forward pass
        output = mp(setup_basic['x'], setup_basic['edge_index'])
        assert output.shape == (setup_basic['num_nodes'], setup_basic['out_dim'])

    def test_attention_aggregation(self, setup_basic):
        """Test attention-based aggregation."""
        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim'],
            aggr='attention',
            aggregation_config={'num_heads': 4, 'dropout': 0.1}
        )

        assert mp.aggregation.method == 'attention'
        assert mp.aggregation.num_heads == 4

        output = mp(setup_basic['x'], setup_basic['edge_index'])
        assert output.shape == (setup_basic['num_nodes'], setup_basic['out_dim'])

    def test_multi_aggregation(self, setup_basic):
        """Test multi-aggregation combining multiple methods."""
        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim'],
            aggr='multi',
            aggregation_config={'methods': ['sum', 'mean', 'max']}
        )

        assert mp.aggregation.method == 'multi'
        output = mp(setup_basic['x'], setup_basic['edge_index'])
        assert output.shape == (setup_basic['num_nodes'], setup_basic['out_dim'])

    def test_copresheaf_adaptive_aggregation(self, setup_basic):
        """Test copresheaf adaptive aggregation."""
        for complexity in ['simple', 'complex', 'heterogeneous']:
            mp = MessagePassing(
                in_dim=setup_basic['in_dim'],
                out_dim=setup_basic['out_dim'],
                aggr='copresheaf_adaptive',
                aggregation_config={'complexity': complexity}
            )

            output = mp(setup_basic['x'], setup_basic['edge_index'])
            assert output.shape == (setup_basic['num_nodes'], setup_basic['out_dim'])

    def test_forward_with_edge_attributes(self, setup_basic):
        """Test forward pass with edge attributes."""
        edge_attr_dim = 3
        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim'],
            edge_attr_dim=edge_attr_dim
        )

        edge_attr = torch.randn(setup_basic['num_edges'], edge_attr_dim)
        output = mp(setup_basic['x'], setup_basic['edge_index'], edge_attr)
        assert output.shape == (setup_basic['num_nodes'], setup_basic['out_dim'])

    def test_message_computation(self, setup_basic):
        """Test message computation."""
        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim']
        )

        row, col = setup_basic['edge_index']
        x_i = setup_basic['x'][col]  # target
        x_j = setup_basic['x'][row]  # source

        messages = mp.message(x_i, x_j)
        assert messages.shape == (setup_basic['num_edges'], setup_basic['out_dim'])

    def test_aggregate_function(self, setup_basic):
        """Test aggregate function."""
        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim']
        )

        messages = torch.randn(setup_basic['num_edges'], setup_basic['out_dim'])
        index = setup_basic['edge_index'][1]  # target indices

        aggregated = mp.aggregate(messages, index, dim_size=setup_basic['num_nodes'])
        assert aggregated.shape == (setup_basic['num_nodes'], setup_basic['out_dim'])

    def test_update_function(self, setup_basic):
        """Test update function."""
        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim']
        )

        aggregated = torch.randn(setup_basic['num_nodes'], setup_basic['out_dim'])
        updated = mp.update(aggregated, setup_basic['x'])
        assert updated.shape == (setup_basic['num_nodes'], setup_basic['out_dim'])

    def test_set_aggregation(self, setup_basic):
        """Test dynamic aggregation change."""
        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim'],
            aggr='sum'
        )

        assert mp.aggregation.method == 'sum'

        # Change aggregation
        mp.set_aggregation('mean')
        assert mp.aggregation.method == 'mean'

        # Change to attention with config
        mp.set_aggregation('attention', {'num_heads': 2})
        assert mp.aggregation.method == 'attention'
        assert mp.aggregation.num_heads == 2

    def test_get_aggregation_info(self, setup_basic):
        """Test aggregation info retrieval."""
        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim'],
            aggr='attention',
            aggregation_config={'num_heads': 4}
        )

        info = mp.get_aggregation_info()
        assert info['method'] == 'attention'
        assert info['dim'] == setup_basic['out_dim']
        assert info['num_heads'] == 4
        assert 'parameters' in info

    def test_custom_factories(self, setup_basic):
        """Test custom factory functions."""

        def custom_copresheaf_factory(in_dim, out_dim, edge_attr_dim):
            return nn.Linear(2 * in_dim + edge_attr_dim, out_dim)

        def custom_message_factory(in_dim, out_dim):
            return nn.Linear(in_dim + out_dim, out_dim)

        def custom_update_factory(in_dim, out_dim):
            return nn.Linear(in_dim + out_dim, out_dim)

        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim'],
            copresheaf_map_factory=custom_copresheaf_factory,
            message_factory=custom_message_factory,
            update_factory=custom_update_factory
        )

        output = mp(setup_basic['x'], setup_basic['edge_index'])
        assert output.shape == (setup_basic['num_nodes'], setup_basic['out_dim'])

    def test_invalid_inputs(self, setup_basic):
        """Test error handling for invalid inputs."""
        mp = MessagePassing(
            in_dim=setup_basic['in_dim'],
            out_dim=setup_basic['out_dim']
        )

        # Test invalid edge_index shape
        with pytest.raises(ValueError):
            bad_edge_index = torch.randn(3, 5)  # Should be 2xN
            mp(setup_basic['x'], bad_edge_index)

        # Test invalid aggregation method
        with pytest.raises(ValueError):
            MessagePassing(
                in_dim=setup_basic['in_dim'],
                out_dim=setup_basic['out_dim'],
                aggr='invalid_method'
            )

        # Test mismatched dimensions in message
        with pytest.raises(ValueError):
            x_i = torch.randn(5, setup_basic['in_dim'] + 1)  # Wrong dim
            x_j = torch.randn(5, setup_basic['in_dim'])
            mp.message(x_i, x_j)


class TestHigherOrderMessagePassing:
    """Test suite for HigherOrderMessagePassing class."""

    @pytest.fixture
    def setup_higher_order(self):
        """Setup higher-order test configuration."""
        torch.manual_seed(42)
        in_dim = 8
        out_dim = 16
        num_nodes = 4
        num_neighborhoods = 3

        x = torch.randn(num_nodes, in_dim)

        # Multiple edge index tensors for different neighborhoods
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
        """Test initialization of HigherOrderMessagePassing."""
        ho_mp = HigherOrderMessagePassing(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods']
        )

        assert ho_mp.num_neighborhoods == setup_higher_order['num_neighborhoods']
        assert len(ho_mp.neighborhood_layers) == setup_higher_order['num_neighborhoods']
        assert ho_mp.combine_method == 'concat'  # default

    @pytest.mark.parametrize('combine_method', ['concat', 'sum', 'mean'])
    def test_combine_methods(self, setup_higher_order, combine_method):
        """Test different neighborhood combination methods."""
        ho_mp = HigherOrderMessagePassing(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods'],
            combine_method=combine_method
        )

        output = ho_mp(
            setup_higher_order['x'],
            setup_higher_order['edge_index_list']
        )
        assert output.shape == (setup_higher_order['num_nodes'], setup_higher_order['out_dim'])

    def test_per_neighborhood_aggregation(self, setup_higher_order):
        """Test different aggregation per neighborhood."""
        aggr_list = ['sum', 'mean', 'max']

        ho_mp = HigherOrderMessagePassing(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods'],
            aggr=aggr_list
        )

        for i, expected_aggr in enumerate(aggr_list):
            assert ho_mp.neighborhood_layers[i].aggregation.method == expected_aggr

        output = ho_mp(
            setup_higher_order['x'],
            setup_higher_order['edge_index_list']
        )
        assert output.shape == (setup_higher_order['num_nodes'], setup_higher_order['out_dim'])

    def test_with_edge_attributes(self, setup_higher_order):
        """Test with edge attributes for each neighborhood."""
        edge_attr_dim = 3
        ho_mp = HigherOrderMessagePassing(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods'],
            edge_attr_dim=edge_attr_dim
        )

        edge_attr_list = [
            torch.randn(edge_index.size(1), edge_attr_dim)
            for edge_index in setup_higher_order['edge_index_list']
        ]

        output = ho_mp(
            setup_higher_order['x'],
            setup_higher_order['edge_index_list'],
            edge_attr_list
        )
        assert output.shape == (setup_higher_order['num_nodes'], setup_higher_order['out_dim'])

    def test_set_neighborhood_aggregation(self, setup_higher_order):
        """Test dynamic aggregation change for specific neighborhood."""
        ho_mp = HigherOrderMessagePassing(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods'],
            aggr='sum'
        )

        # Change aggregation for neighborhood 1
        ho_mp.set_neighborhood_aggregation(1, 'mean')
        assert ho_mp.neighborhood_layers[1].aggregation.method == 'mean'

        # Test invalid index
        with pytest.raises(IndexError):
            ho_mp.set_neighborhood_aggregation(10, 'mean')

    def test_aggregation_summary(self, setup_higher_order):
        """Test aggregation summary retrieval."""
        ho_mp = HigherOrderMessagePassing(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods'],
            aggr=['sum', 'mean', 'max']
        )

        summary = ho_mp.get_aggregation_summary()
        assert summary['total_neighborhoods'] == setup_higher_order['num_neighborhoods']
        assert summary['inter_neighborhood_combination'] == 'concat'
        assert len(summary['neighborhoods']) == setup_higher_order['num_neighborhoods']
        assert 'total_parameters' in summary

    def test_neighborhood_types_info(self, setup_higher_order):
        """Test neighborhood types information."""
        ho_mp = HigherOrderMessagePassing(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=4
        )

        info = ho_mp.get_neighborhood_types_info()
        assert len(info) == 4
        assert 'neighborhood_0' in info

    def test_complex_aggregation_config(self, setup_higher_order):
        """Test complex aggregation configuration."""
        config = {
            'neighborhood_0': {'dropout': 0.1},
            'neighborhood_1': {'num_heads': 2},
            'complexity_per_neighborhood': ['simple', 'complex', 'heterogeneous']
        }

        ho_mp = HigherOrderMessagePassing(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods'],
            aggr='copresheaf_adaptive',
            aggregation_config=config
        )

        output = ho_mp(
            setup_higher_order['x'],
            setup_higher_order['edge_index_list']
        )
        assert output.shape == (setup_higher_order['num_nodes'], setup_higher_order['out_dim'])

    def test_invalid_neighborhood_count(self, setup_higher_order):
        """Test error handling for mismatched neighborhood count."""
        ho_mp = HigherOrderMessagePassing(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=2  # Expecting 2 neighborhoods
        )

        with pytest.raises(ValueError):
            # Providing 3 neighborhoods
            ho_mp(
                setup_higher_order['x'],
                setup_higher_order['edge_index_list']
            )

    def test_combine_neighborhoods(self, setup_higher_order):
        """Test internal _combine_neighborhoods method."""
        ho_mp = HigherOrderMessagePassing(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=3,
            combine_method='concat'
        )

        # Create dummy neighborhood outputs
        outputs = [
            torch.randn(setup_higher_order['num_nodes'], setup_higher_order['out_dim'])
            for _ in range(3)
        ]

        combined = ho_mp._combine_neighborhoods(outputs)
        assert combined.shape == (setup_higher_order['num_nodes'], setup_higher_order['out_dim'])

    def test_gradients_flow(self, setup_higher_order):
        """Test that gradients flow through the model."""
        ho_mp = HigherOrderMessagePassing(
            in_dim=setup_higher_order['in_dim'],
            out_dim=setup_higher_order['out_dim'],
            num_neighborhoods=setup_higher_order['num_neighborhoods']
        )

        x = setup_higher_order['x'].requires_grad_(True)
        output = ho_mp(x, setup_higher_order['edge_index_list'])
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestIntegration:
    """Integration tests for message passing components."""

    def test_end_to_end_pipeline(self):
        """Test complete message passing pipeline."""
        torch.manual_seed(42)

        # Setup
        in_dim, hidden_dim, out_dim = 8, 16, 4
        num_nodes = 10
        num_layers = 3

        x = torch.randn(num_nodes, in_dim)
        edge_index = torch.randint(0, num_nodes, (2, 20))

        # Build multi-layer model
        layers = []
        current_dim = in_dim

        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else out_dim
            layer = MessagePassing(
                in_dim=current_dim,
                out_dim=next_dim,
                aggr='sum'
            )
            layers.append(layer)
            current_dim = next_dim

        # Forward pass through all layers
        h = x
        for layer in layers:
            h = layer(h, edge_index)
            h = torch.relu(h)  # Add non-linearity

        assert h.shape == (num_nodes, out_dim)

    def test_higher_order_with_varying_neighborhoods(self):
        """Test higher-order MP with varying neighborhood sizes."""
        torch.manual_seed(42)

        in_dim, out_dim = 8, 16
        num_nodes = 6

        x = torch.randn(num_nodes, in_dim)

        # Different sized neighborhoods
        edge_index_list = [
            torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),  # 5 edges
            torch.tensor([[0, 2], [2, 0]]),  # 2 edges
            torch.tensor([[1], [3]])  # 1 edge
        ]

        ho_mp = HigherOrderMessagePassing(
            in_dim=in_dim,
            out_dim=out_dim,
            num_neighborhoods=3,
            aggr=['sum', 'mean', 'max']
        )

        output = ho_mp(x, edge_index_list)
        assert output.shape == (num_nodes, out_dim)

    def test_memory_efficiency(self):
        """Test memory efficiency with large graphs."""
        torch.manual_seed(42)

        # Large graph setup
        in_dim, out_dim = 32, 64
        num_nodes = 1000
        num_edges = 5000

        x = torch.randn(num_nodes, in_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        mp = MessagePassing(
            in_dim=in_dim,
            out_dim=out_dim,
            aggr='mean'
        )

        # Should handle large graphs without issues
        output = mp(x, edge_index)
        assert output.shape == (num_nodes, out_dim)

        # Check that output values are reasonable (not NaN or Inf)
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
