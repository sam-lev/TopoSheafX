# tests/test_model.py
"""
Comprehensive test suite for toposheafx/nn/model.py
Tests CMPNN and CHOMPNN model classes.
"""

import pytest
import torch
import torch.nn as nn

from toposheafx.nn.model import CHOMPNN, CMPNN


class TestCMPNN:
    """Test suite for CMPNN model class."""

    @pytest.fixture
    def setup_cmpnn(self):
        """Setup CMPNN test configuration."""
        torch.manual_seed(42)
        in_dim = 8
        hidden_dim = 16
        out_dim = 3  # For classification
        num_layers = 2
        batch_size = 2
        nodes_per_graph = 5
        total_nodes = batch_size * nodes_per_graph

        # Create batched data
        x = torch.randn(total_nodes, in_dim)
        batch = torch.cat([torch.full((nodes_per_graph,), i, dtype=torch.long)
                           for i in range(batch_size)])

        # Create edge indices for multiple neighborhoods
        edge_index_list = []
        for _ in range(2):  # 2 neighborhoods
            edges = []
            for b in range(batch_size):
                offset = b * nodes_per_graph
                # Random edges within each graph
                src = torch.randint(0, nodes_per_graph, (10,)) + offset
                dst = torch.randint(0, nodes_per_graph, (10,)) + offset
                edges.append(torch.stack([src, dst]))
            edge_index_list.append(torch.cat(edges, dim=1))

        return {
            'in_dim': in_dim,
            'hidden_dim': hidden_dim,
            'out_dim': out_dim,
            'num_layers': num_layers,
            'x': x,
            'edge_index_list': edge_index_list,
            'batch': batch,
            'batch_size': batch_size,
            'total_nodes': total_nodes
        }

    def test_initialization(self, setup_cmpnn):
        """Test CMPNN initialization."""
        model = CMPNN(
            in_dim=setup_cmpnn['in_dim'],
            hidden_dim=setup_cmpnn['hidden_dim'],
            out_dim=setup_cmpnn['out_dim'],
            num_layers=setup_cmpnn['num_layers']
        )

        assert hasattr(model, 'encoder')
        assert hasattr(model, 'layers')
        assert hasattr(model, 'readout')
        assert len(model.layers) == setup_cmpnn['num_layers']

    def test_forward_pass(self, setup_cmpnn):
        """Test forward pass of CMPNN."""
        model = CMPNN(
            in_dim=setup_cmpnn['in_dim'],
            hidden_dim=setup_cmpnn['hidden_dim'],
            out_dim=setup_cmpnn['out_dim'],
            num_layers=setup_cmpnn['num_layers'],
            num_neighborhoods=2
        )

        output = model(
            setup_cmpnn['x'],
            setup_cmpnn['edge_index_list'],
            setup_cmpnn['batch']
        )

        # Output should be graph-level predictions
        assert output.shape == (setup_cmpnn['batch_size'], setup_cmpnn['out_dim'])
        assert torch.isfinite(output).all()

    def test_with_edge_attributes(self, setup_cmpnn):
        """Test CMPNN with edge attributes."""
        edge_attr_dim = 4
        model = CMPNN(
            in_dim=setup_cmpnn['in_dim'],
            hidden_dim=setup_cmpnn['hidden_dim'],
            out_dim=setup_cmpnn['out_dim'],
            num_layers=setup_cmpnn['num_layers'],
            edge_attr_dim=edge_attr_dim,
            num_neighborhoods=2
        )

        edge_attr_list = [
            torch.randn(edge_index.size(1), edge_attr_dim)
            for edge_index in setup_cmpnn['edge_index_list']
        ]

        output = model(
            setup_cmpnn['x'],
            setup_cmpnn['edge_index_list'],
            setup_cmpnn['batch'],
            edge_attr_list=edge_attr_list
        )

        assert output.shape == (setup_cmpnn['batch_size'], setup_cmpnn['out_dim'])

    def test_single_neighborhood(self, setup_cmpnn):
        """Test CMPNN with single neighborhood."""
        model = CMPNN(
            in_dim=setup_cmpnn['in_dim'],
            hidden_dim=setup_cmpnn['hidden_dim'],
            out_dim=setup_cmpnn['out_dim'],
            num_layers=setup_cmpnn['num_layers'],
            num_neighborhoods=1
        )

        output = model(
            setup_cmpnn['x'],
            [setup_cmpnn['edge_index_list'][0]],
            setup_cmpnn['batch']
        )

        assert output.shape == (setup_cmpnn['batch_size'], setup_cmpnn['out_dim'])

    def test_gradient_flow(self, setup_cmpnn):
        """Test gradient flow through CMPNN."""
        model = CMPNN(
            in_dim=setup_cmpnn['in_dim'],
            hidden_dim=setup_cmpnn['hidden_dim'],
            out_dim=setup_cmpnn['out_dim'],
            num_layers=setup_cmpnn['num_layers'],
            num_neighborhoods=2
        )

        x = setup_cmpnn['x'].requires_grad_(True)
        output = model(x, setup_cmpnn['edge_index_list'], setup_cmpnn['batch'])

        # Create target and loss
        target = torch.randint(0, setup_cmpnn['out_dim'], (setup_cmpnn['batch_size'],))
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        # Check that all model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None

    def test_deep_model(self):
        """Test CMPNN with many layers."""
        model = CMPNN(
            in_dim=8,
            hidden_dim=16,
            out_dim=3,
            num_layers=5  # Deep model
        )

        x = torch.randn(20, 8)
        edge_index_list = [torch.randint(0, 20, (2, 30))]
        batch = torch.zeros(20, dtype=torch.long)

        output = model(x, edge_index_list, batch)
        assert output.shape == (1, 3)
        assert torch.isfinite(output).all()


class TestCHOMPNN:
    """Test suite for CHOMPNN model class."""

    @pytest.fixture
    def setup_chompnn(self):
        """Setup CHOMPNN test configuration."""
        torch.manual_seed(42)
        in_dim = 8
        hidden_dim = 16
        out_dim = 4
        num_layers = 3
        num_neighborhoods = 3
        batch_size = 2
        nodes_per_graph = 6
        total_nodes = batch_size * nodes_per_graph

        x = torch.randn(total_nodes, in_dim)
        batch = torch.cat([torch.full((nodes_per_graph,), i, dtype=torch.long)
                           for i in range(batch_size)])

        # Create multiple neighborhoods
        edge_index_list = []
        for _ in range(num_neighborhoods):
            edges = []
            for b in range(batch_size):
                offset = b * nodes_per_graph
                src = torch.randint(0, nodes_per_graph, (8,)) + offset
                dst = torch.randint(0, nodes_per_graph, (8,)) + offset
                edges.append(torch.stack([src, dst]))
            edge_index_list.append(torch.cat(edges, dim=1))

        return {
            'in_dim': in_dim,
            'hidden_dim': hidden_dim,
            'out_dim': out_dim,
            'num_layers': num_layers,
            'num_neighborhoods': num_neighborhoods,
            'x': x,
            'edge_index_list': edge_index_list,
            'batch': batch,
            'batch_size': batch_size,
            'total_nodes': total_nodes
        }

    def test_initialization(self, setup_chompnn):
        """Test CHOMPNN initialization."""
        model = CHOMPNN(
            in_dim=setup_chompnn['in_dim'],
            hidden_dim=setup_chompnn['hidden_dim'],
            out_dim=setup_chompnn['out_dim'],
            num_layers=setup_chompnn['num_layers'],
            num_neighborhoods=setup_chompnn['num_neighborhoods']
        )

        assert hasattr(model, 'encoder')
        assert hasattr(model, 'layers')
        assert hasattr(model, 'readout')
        assert len(model.layers) == setup_chompnn['num_layers']

    def test_forward_pass(self, setup_chompnn):
        """Test forward pass of CHOMPNN."""
        model = CHOMPNN(
            in_dim=setup_chompnn['in_dim'],
            hidden_dim=setup_chompnn['hidden_dim'],
            out_dim=setup_chompnn['out_dim'],
            num_layers=setup_chompnn['num_layers'],
            num_neighborhoods=setup_chompnn['num_neighborhoods']
        )

        output = model(
            setup_chompnn['x'],
            setup_chompnn['edge_index_list'],
            setup_chompnn['batch']
        )

        assert output.shape == (setup_chompnn['batch_size'], setup_chompnn['out_dim'])
        assert torch.isfinite(output).all()

    def test_with_edge_attributes(self, setup_chompnn):
        """Test CHOMPNN with edge attributes."""
        edge_attr_dim = 5
        model = CHOMPNN(
            in_dim=setup_chompnn['in_dim'],
            hidden_dim=setup_chompnn['hidden_dim'],
            out_dim=setup_chompnn['out_dim'],
            num_layers=setup_chompnn['num_layers'],
            edge_attr_dim=edge_attr_dim,
            num_neighborhoods=setup_chompnn['num_neighborhoods']
        )

        edge_attr_list = [
            torch.randn(edge_index.size(1), edge_attr_dim)
            for edge_index in setup_chompnn['edge_index_list']
        ]

        output = model(
            setup_chompnn['x'],
            setup_chompnn['edge_index_list'],
            setup_chompnn['batch'],
            edge_attr_list=edge_attr_list
        )

        assert output.shape == (setup_chompnn['batch_size'], setup_chompnn['out_dim'])

    def test_gradient_flow(self, setup_chompnn):
        """Test gradient flow through CHOMPNN."""
        model = CHOMPNN(
            in_dim=setup_chompnn['in_dim'],
            hidden_dim=setup_chompnn['hidden_dim'],
            out_dim=setup_chompnn['out_dim'],
            num_layers=setup_chompnn['num_layers'],
            num_neighborhoods=setup_chompnn['num_neighborhoods']
        )

        x = setup_chompnn['x'].requires_grad_(True)
        output = model(x, setup_chompnn['edge_index_list'], setup_chompnn['batch'])

        target = torch.randn(setup_chompnn['batch_size'], setup_chompnn['out_dim'])
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        for param in model.parameters():
            assert param.grad is not None

    def test_variable_neighborhoods(self, setup_chompnn):
        """Test CHOMPNN with different number of neighborhoods."""
        for num_neighborhoods in [1, 2, 4]:
            model = CHOMPNN(
                in_dim=setup_chompnn['in_dim'],
                hidden_dim=setup_chompnn['hidden_dim'],
                out_dim=setup_chompnn['out_dim'],
                num_layers=2,
                num_neighborhoods=num_neighborhoods
            )

            # Get available edge indices
            available_edge_indices = setup_chompnn['edge_index_list']

            # Create edge_index_list with duplicates if needed
            if len(available_edge_indices) >= num_neighborhoods:
                # We have enough, just slice
                edge_index_list = available_edge_indices[:num_neighborhoods]
            else:
                # Need to duplicate some indices
                edge_index_list = []
                for i in range(num_neighborhoods):
                    # Use modulo to cycle through available indices
                    idx = i % len(available_edge_indices)
                    edge_index_list.append(available_edge_indices[idx])

            output = model(
                setup_chompnn['x'],
                edge_index_list,
                setup_chompnn['batch']
            )

            assert output.shape == (setup_chompnn['batch_size'], setup_chompnn['out_dim'])

class TestModelIntegration:
    """Integration tests for model components."""

    def test_cmpnn_classification(self):
        """Test CMPNN for graph classification task."""
        torch.manual_seed(42)

        # Setup classification problem
        num_classes = 5
        model = CMPNN(
            in_dim=32,
            hidden_dim=64,
            out_dim=num_classes,
            num_layers=3,
            num_neighborhoods=2
        )

        # Create batch of graphs
        batch_size = 4
        nodes_per_graph = 20
        total_nodes = batch_size * nodes_per_graph

        x = torch.randn(total_nodes, 32)
        batch = torch.cat([torch.full((nodes_per_graph,), i, dtype=torch.long)
                           for i in range(batch_size)])

        # Create neighborhoods
        edge_index_list = []
        for _ in range(2):
            edges = []
            for b in range(batch_size):
                offset = b * nodes_per_graph
                src = torch.randint(0, nodes_per_graph, (50,)) + offset
                dst = torch.randint(0, nodes_per_graph, (50,)) + offset
                edges.append(torch.stack([src, dst]))
            edge_index_list.append(torch.cat(edges, dim=1))

        # Forward pass
        output = model(x, edge_index_list, batch)
        assert output.shape == (batch_size, num_classes)

        # Test with softmax for probabilities
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size))

    def test_chompnn_regression(self):
        """Test CHOMPNN for graph regression task."""
        torch.manual_seed(42)

        # Setup regression problem
        output_dim = 1  # Single value prediction
        model = CHOMPNN(
            in_dim=16,
            hidden_dim=32,
            out_dim=output_dim,
            num_layers=2,
            num_neighborhoods=3
        )

        # Create single graph
        num_nodes = 15
        x = torch.randn(num_nodes, 16)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        # Create neighborhoods
        edge_index_list = [
            torch.randint(0, num_nodes, (2, 25)),
            torch.randint(0, num_nodes, (2, 20)),
            torch.randint(0, num_nodes, (2, 15))
        ]

        # Forward pass
        output = model(x, edge_index_list, batch)
        assert output.shape == (1, output_dim)
        assert torch.isfinite(output).all()

    def test_model_comparison(self):
        """Compare CMPNN and CHOMPNN on same data."""
        torch.manual_seed(42)

        # Same configuration for both models
        config = {
            'in_dim': 16,
            'hidden_dim': 32,
            'out_dim': 3,
            'num_layers': 2,
            'num_neighborhoods': 2
        }

        cmpnn = CMPNN(**config)
        chompnn = CHOMPNN(**config)

        # Same input data
        x = torch.randn(30, config['in_dim'])
        batch = torch.cat([torch.zeros(15, dtype=torch.long),
                           torch.ones(15, dtype=torch.long)])
        edge_index_list = [
            torch.randint(0, 30, (2, 50)),
            torch.randint(0, 30, (2, 40))
        ]

        # Forward passes
        output_cmpnn = cmpnn(x, edge_index_list, batch)
        output_chompnn = chompnn(x, edge_index_list, batch)

        assert output_cmpnn.shape == output_chompnn.shape == (2, config['out_dim'])

        # Outputs should be different (different architectures)
        assert not torch.allclose(output_cmpnn, output_chompnn)

    def test_large_graph(self):
        """Test models on large graphs."""
        torch.manual_seed(42)

        # Large graph setup
        model = CMPNN(
            in_dim=64,
            hidden_dim=128,
            out_dim=10,
            num_layers=2,
            num_neighborhoods=1
        )

        # Large graph
        num_nodes = 1000
        num_edges = 5000
        x = torch.randn(num_nodes, 64)
        batch = torch.zeros(num_nodes, dtype=torch.long)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Forward pass
        output = model(x, [edge_index], batch)
        assert output.shape == (1, 10)
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
