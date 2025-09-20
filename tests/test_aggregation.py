# tests/test_aggregation.py
"""
Comprehensive test suite for toposheafx/base/aggregation.py
Tests Aggregation class and AggregationFactory with all aggregation methods.
"""

import pytest
import torch

from toposheafx.base.aggregation import Aggregation, AggregationFactory


class TestAggregation:
    """Test suite for Aggregation class."""

    @pytest.fixture
    def setup_basic(self):
        """Setup basic test configuration."""
        torch.manual_seed(42)
        num_edges = 10
        num_nodes = 5
        dim = 16

        messages = torch.randn(num_edges, dim)
        index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 4, 4])  # Target nodes

        return {
            'messages': messages,
            'index': index,
            'dim': dim,
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }

    @pytest.mark.parametrize('method', ['sum', 'mean', 'max', 'min'])
    def test_basic_aggregation(self, setup_basic, method):
        """Test basic aggregation methods."""
        agg = Aggregation(method=method, dim=setup_basic['dim'])

        result = agg(
            setup_basic['messages'],
            setup_basic['index'],
            dim_size=setup_basic['num_nodes']
        )

        assert result.shape == (setup_basic['num_nodes'], setup_basic['dim'])
        assert torch.isfinite(result).all()

        # Verify aggregation semantics
        if method == 'sum':
            # Node 0 should have sum of first 2 messages
            expected = setup_basic['messages'][:2].sum(dim=0)
            torch.testing.assert_close(result[0], expected, rtol=1e-5, atol=1e-5)
        elif method == 'mean':
            # Node 0 should have mean of first 2 messages
            expected = setup_basic['messages'][:2].mean(dim=0)
            torch.testing.assert_close(result[0], expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('method', ['std', 'var'])
    def test_statistical_aggregation(self, setup_basic, method):
        """Test statistical aggregation methods."""
        agg = Aggregation(method=method, dim=setup_basic['dim'])

        result = agg(
            setup_basic['messages'],
            setup_basic['index'],
            dim_size=setup_basic['num_nodes']
        )

        assert result.shape == (setup_basic['num_nodes'], setup_basic['dim'])
        assert torch.isfinite(result).all()

        # Verify non-negative for variance/std
        assert (result >= 0).all()

    def test_attention_aggregation(self, setup_basic):
        """Test attention-based aggregation."""
        agg = Aggregation(
            method='attention',
            dim=setup_basic['dim'],
            num_heads=4,
            dropout=0.1
        )

        result = agg(
            setup_basic['messages'],
            setup_basic['index'],
            dim_size=setup_basic['num_nodes']
        )

        assert result.shape == (setup_basic['num_nodes'], setup_basic['dim'])
        assert torch.isfinite(result).all()

    def test_lstm_aggregation(self, setup_basic):
        """Test LSTM-based aggregation."""
        agg = Aggregation(
            method='lstm',
            dim=setup_basic['dim'],
            dropout=0.0
        )

        result = agg(
            setup_basic['messages'],
            setup_basic['index'],
            dim_size=setup_basic['num_nodes']
        )

        assert result.shape == (setup_basic['num_nodes'], setup_basic['dim'])
        assert torch.isfinite(result).all()

    def test_set2set_aggregation(self, setup_basic):
        """Test Set2Set aggregation."""
        agg = Aggregation(
            method='set2set',
            dim=setup_basic['dim'],
            processing_steps=3
        )

        result = agg(
            setup_basic['messages'],
            setup_basic['index'],
            dim_size=setup_basic['num_nodes']
        )

        assert result.shape == (setup_basic['num_nodes'], setup_basic['dim'])
        assert torch.isfinite(result).all()

    def test_multi_aggregation(self, setup_basic):
        """Test multi-aggregation combining multiple methods."""
        methods = ['sum', 'mean', 'max']
        agg = Aggregation(
            method='multi',
            dim=setup_basic['dim'],
            methods=methods
        )

        result = agg(
            setup_basic['messages'],
            setup_basic['index'],
            dim_size=setup_basic['num_nodes']
        )

        # Multi-aggregation concatenates results
        expected_dim = setup_basic['dim'] * len(methods)
        assert result.shape == (setup_basic['num_nodes'], expected_dim)
        assert torch.isfinite(result).all()

    def test_empty_neighborhood(self, setup_basic):
        """Test aggregation with nodes having no incoming messages."""
        # Create index that skips node 2
        index = torch.tensor([0, 0, 1, 1, 3, 4])
        messages = torch.randn(6, setup_basic['dim'])

        agg = Aggregation(method='sum', dim=setup_basic['dim'])
        result = agg(messages, index, dim_size=5)

        assert result.shape == (5, setup_basic['dim'])
        # Node 2 should have zeros (no incoming messages)
        assert torch.allclose(result[2], torch.zeros(setup_basic['dim']))

    def test_single_message_per_node(self, setup_basic):
        """Test aggregation when each node has exactly one message."""
        index = torch.arange(setup_basic['num_nodes'])
        messages = torch.randn(setup_basic['num_nodes'], setup_basic['dim'])

        agg = Aggregation(method='mean', dim=setup_basic['dim'])
        result = agg(messages, index, dim_size=setup_basic['num_nodes'])

        # Each node's result should equal its single message
        torch.testing.assert_close(result, messages, rtol=1e-5, atol=1e-5)

    def test_invalid_method(self):
        """Test invalid aggregation method."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            Aggregation(method='invalid_method')

    def test_missing_dim_for_learnable(self):
        """Test that learnable methods require dim."""
        for method in ['attention', 'lstm', 'set2set']:
            with pytest.raises(ValueError, match="dim must be specified"):
                Aggregation(method=method, dim=None)

    def test_input_validation(self, setup_basic):
        """Test input validation."""
        agg = Aggregation(method='sum', dim=setup_basic['dim'])

        # Test wrong dimension for messages
        with pytest.raises(ValueError, match="messages must be 2D tensor"):
            bad_messages = torch.randn(10)  # 1D tensor
            agg(bad_messages, setup_basic['index'], dim_size=setup_basic['num_nodes'])

        # Test wrong dimension for index
        with pytest.raises(ValueError, match="index must be 1D tensor"):
            bad_index = torch.randn(10, 2)  # 2D tensor
            agg(setup_basic['messages'], bad_index, dim_size=setup_basic['num_nodes'])

        # Test mismatched sizes
        with pytest.raises(ValueError, match="same first dimension"):
            bad_index = torch.zeros(5, dtype=torch.long)  # Wrong size
            agg(setup_basic['messages'], bad_index, dim_size=setup_basic['num_nodes'])

    def test_attention_multi_head(self, setup_basic):
        """Test multi-head attention aggregation."""
        for num_heads in [1, 2, 4, 8]:
            if setup_basic['dim'] % num_heads != 0:
                continue  # Skip if dimension not divisible

            agg = Aggregation(
                method='attention',
                dim=setup_basic['dim'],
                num_heads=num_heads,
                dropout=0.0
            )

            result = agg(
                setup_basic['messages'],
                setup_basic['index'],
                dim_size=setup_basic['num_nodes']
            )

            assert result.shape == (setup_basic['num_nodes'], setup_basic['dim'])

    def test_dropout_effect(self, setup_basic):
        """Test that dropout has effect in training mode."""
        agg = Aggregation(
            method='attention',
            dim=setup_basic['dim'],
            num_heads=1,
            dropout=0.5
        )

        agg.train()  # Enable training mode for dropout

        # Run multiple times and check for different outputs
        results = []
        for _ in range(5):
            result = agg(
                setup_basic['messages'],
                setup_basic['index'],
                dim_size=setup_basic['num_nodes']
            )
            results.append(result)

        # At least some results should be different due to dropout
        all_same = all(torch.allclose(results[0], r) for r in results[1:])
        assert not all_same, "Dropout should cause variation in training mode"

    def test_eval_mode_consistency(self, setup_basic):
        """Test that eval mode produces consistent results."""
        agg = Aggregation(
            method='attention',
            dim=setup_basic['dim'],
            num_heads=1,
            dropout=0.5
        )

        agg.eval()  # Enable eval mode

        # Run multiple times and check for same outputs
        results = []
        for _ in range(3):
            result = agg(
                setup_basic['messages'],
                setup_basic['index'],
                dim_size=setup_basic['num_nodes']
            )
            results.append(result)

        # All results should be the same in eval mode
        for r in results[1:]:
            torch.testing.assert_close(results[0], r, rtol=1e-5, atol=1e-5)

    def test_gradient_flow(self, setup_basic):
        """Test that gradients flow through aggregation."""
        agg = Aggregation(
            method='attention',
            dim=setup_basic['dim'],
            num_heads=2
        )

        messages = setup_basic['messages'].requires_grad_(True)
        result = agg(messages, setup_basic['index'], dim_size=setup_basic['num_nodes'])

        loss = result.sum()
        loss.backward()

        assert messages.grad is not None
        assert not torch.all(messages.grad == 0)

    def test_set2set_processing_steps(self, setup_basic):
        """Test Set2Set with different processing steps."""
        for steps in [1, 3, 5]:
            agg = Aggregation(
                method='set2set',
                dim=setup_basic['dim'],
                processing_steps=steps
            )

            result = agg(
                setup_basic['messages'],
                setup_basic['index'],
                dim_size=setup_basic['num_nodes']
            )

            assert result.shape == (setup_basic['num_nodes'], setup_basic['dim'])

    def test_extra_repr(self, setup_basic):
        """Test string representation."""
        agg = Aggregation(
            method='attention',
            dim=setup_basic['dim'],
            num_heads=4
        )

        repr_str = agg.extra_repr()
        assert 'method=attention' in repr_str
        assert f'dim={setup_basic["dim"]}' in repr_str
        assert 'num_heads=4' in repr_str


class TestAggregationFactory:
    """Test suite for AggregationFactory class."""

    def test_create_basic(self):
        """Test basic aggregation creation."""
        for method in ['sum', 'mean', 'max', 'min']:
            agg = AggregationFactory.create_basic(method)
            assert isinstance(agg, Aggregation)
            assert agg.method == method

    def test_create_statistical(self):
        """Test statistical aggregation creation."""
        for method in ['std', 'var']:
            agg = AggregationFactory.create_statistical(method, dim=16)
            assert isinstance(agg, Aggregation)
            assert agg.method == method
            assert agg.dim == 16

    def test_create_attention(self):
        """Test attention aggregation creation."""
        agg = AggregationFactory.create_attention(
            dim=32,
            num_heads=4,
            dropout=0.1
        )
        assert isinstance(agg, Aggregation)
        assert agg.method == 'attention'
        assert agg.dim == 32
        assert agg.num_heads == 4
        assert agg.dropout == 0.1

    def test_create_set2set(self):
        """Test Set2Set aggregation creation."""
        agg = AggregationFactory.create_set2set(
            dim=16,
            processing_steps=5
        )
        assert isinstance(agg, Aggregation)
        assert agg.method == 'set2set'
        assert agg.dim == 16
        assert agg.processing_steps == 5

    def test_create_multi(self):
        """Test multi-aggregation creation."""
        methods = ['sum', 'mean', 'max']
        agg = AggregationFactory.create_multi(
            methods=methods,
            dim=16
        )
        assert isinstance(agg, Aggregation)
        assert agg.method == 'multi'
        assert agg.dim == 16
        assert len(agg.sub_aggregators) == len(methods)

    @pytest.mark.parametrize('complexity', ['simple', 'complex', 'heterogeneous'])
    def test_create_for_copresheaf(self, complexity):
        """Test copresheaf-specific aggregation creation."""
        agg = AggregationFactory.create_for_copresheaf(
            stalk_dim=32,
            neighborhood_complexity=complexity,
            dropout=0.1
        )
        assert isinstance(agg, Aggregation)

        # Check appropriate method for complexity
        if complexity == 'simple':
            assert agg.method == 'sum'
        elif complexity == 'complex':
            assert agg.method == 'multi'
        elif complexity == 'heterogeneous':
            assert agg.method == 'attention'

    def test_create_for_copresheaf_invalid(self):
        """Test invalid complexity level."""
        with pytest.raises(ValueError, match="Unknown complexity level"):
            AggregationFactory.create_for_copresheaf(
                stalk_dim=32,
                neighborhood_complexity='invalid'
            )


class TestAggregationIntegration:
    """Integration tests for aggregation functionality."""

    def test_large_graph_aggregation(self):
        """Test aggregation on large graphs."""
        torch.manual_seed(42)

        num_nodes = 1000
        num_edges = 10000
        dim = 64

        messages = torch.randn(num_edges, dim)
        index = torch.randint(0, num_nodes, (num_edges,))

        # Test multiple methods
        for method in ['sum', 'mean', 'max']:
            agg = Aggregation(method=method, dim=dim)
            result = agg(messages, index, dim_size=num_nodes)

            assert result.shape == (num_nodes, dim)
            assert torch.isfinite(result).all()

    def test_aggregation_with_batching(self):
        """Test aggregation with batched graphs."""
        torch.manual_seed(42)

        batch_size = 4
        nodes_per_graph = 25
        total_nodes = batch_size * nodes_per_graph
        dim = 32

        # Create batched edge index
        edge_indices = []
        for b in range(batch_size):
            offset = b * nodes_per_graph
            # Random edges within this graph
            edges = torch.randint(0, nodes_per_graph, (2, 50))
            edges += offset
            edge_indices.append(edges)

        edge_index = torch.cat(edge_indices, dim=1)
        num_edges = edge_index.size(1)

        messages = torch.randn(num_edges, dim)
        index = edge_index[1]  # Target nodes

        agg = Aggregation(method='mean', dim=dim)
        result = agg(messages, index, dim_size=total_nodes)

        assert result.shape == (total_nodes, dim)
        assert torch.isfinite(result).all()

    def test_aggregation_preserves_device(self):
        """Test that aggregation preserves tensor device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dim = 16
        messages = torch.randn(10, dim).cuda()
        index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).cuda()

        agg = Aggregation(method='sum', dim=dim).cuda()
        result = agg(messages, index, dim_size=5)

        assert result.device == messages.device
        assert result.is_cuda


if __name__ == "__main__":
    pytest.main([__file__, "-v"])