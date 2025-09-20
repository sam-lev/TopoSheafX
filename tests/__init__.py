# tests/__init__.py
"""
Test package initialization for TopoSheafX.
Sets up test configuration and shared fixtures.
"""

import os
import sys

import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set random seed for reproducibility
torch.manual_seed(42)

# Configure PyTorch for testing
torch.set_num_threads(1)  # Limit threads for consistent testing

# Shared test configuration
TEST_CONFIG = {
    'default_tolerance': {'rtol': 1e-5, 'atol': 1e-5},
    'device': 'cpu',  # Use CPU for GitHub Actions
    'small_graph_size': 10,
    'medium_graph_size': 100,
    'large_graph_size': 1000,
}

# Shared fixtures
@pytest.fixture(scope='session')
def device():
    """Get test device (CPU for CI/CD)."""
    return torch.device(TEST_CONFIG['device'])

@pytest.fixture(scope='session')
def tolerance():
    """Get default numerical tolerance for tests."""
    return TEST_CONFIG['default_tolerance']
