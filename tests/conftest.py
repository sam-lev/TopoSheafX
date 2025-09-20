# tests/conftest.py
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# typing imports removed - using builtins


# Make repo root importable if needed
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


@pytest.fixture(autouse=True)
def _seed_everything():
    """Automatically set seed for all tests."""
    torch.manual_seed(0)


@pytest.fixture
def toy_edges_1():
    """Single neighborhood: 6-node ring."""
    row = torch.arange(6)
    col = (row + 1) % 6
    return torch.stack([row, col], dim=0)


@pytest.fixture
def toy_edges_2():
    """Two neighborhoods: chain + star."""
    e1 = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    e2 = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0]])
    return [e1, e2]


@pytest.fixture
def set_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Reset after test
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def simple_graph() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a simple test graph."""
    # Triangle graph
    edge_index = torch.tensor([
        [0, 1, 2, 0],
        [1, 2, 0, 2]
    ])
    x = torch.randn(3, 8)
    return x, edge_index


@pytest.fixture
def batch_graphs() -> dict:
    """Create batched graphs for testing."""
    # Two graphs: triangle and square
    x = torch.randn(7, 8)  # 3 + 4 nodes

    # Triangle edges (offset 0)
    triangle_edges = torch.tensor([[0, 1, 2], [1, 2, 0]])

    # Square edges (offset 3)
    square_edges = torch.tensor([[3, 4, 5, 6], [4, 5, 6, 3]])

    edge_index = torch.cat([triangle_edges, square_edges], dim=1)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])

    return {
        'x': x,
        'edge_index': edge_index,
        'batch': batch,
        'num_graphs': 2,
        'num_nodes': 7
    }


@pytest.fixture
def higher_order_complex() -> dict:
    """Create a higher-order cell complex for testing."""
    num_nodes = 6
    x = torch.randn(num_nodes, 8)

    # Different neighborhood types
    edge_index_list = [
        # 0-0 adjacency (edges)
        torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
        # 0-1 incidence (node-to-edge)
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
        # 1-2 incidence (edge-to-face)
        torch.tensor([[0, 1], [0, 0]])
    ]

    return {
        'x': x,
        'edge_index_list': edge_index_list,
        'num_nodes': num_nodes,
        'num_neighborhoods': 3
    }


@pytest.fixture
def device_cpu():
    """CPU device for testing."""
    return torch.device('cpu')


@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def tolerance():
    """Default tolerance for numerical comparisons."""
    return {'rtol': 1e-5, 'atol': 1e-5}


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests"
    )


# Parametrized fixtures for common test scenarios
@pytest.fixture(params=[4, 8, 16, 32])
def feature_dim(request):
    """Different feature dimensions to test."""
    return request.param


@pytest.fixture(params=['sum', 'mean', 'max'])
def aggregation_method(request):
    """Different aggregation methods to test."""
    return request.param


@pytest.fixture(params=[1, 2, 3])
def num_layers(request):
    """Different number of layers to test."""
    return request.param
