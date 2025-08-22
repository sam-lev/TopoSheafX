# tests/conftest.py
import sys
from pathlib import Path
import torch
import pytest

# Make repo root importable if needed
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)

@pytest.fixture
def toy_edges_1():
    # Single neighborhood: 6-node ring
    row = torch.arange(6)
    col = (row + 1) % 6
    return torch.stack([row, col], dim=0)

@pytest.fixture
def toy_edges_2():
    # Two neighborhoods: chain + star
    e1 = torch.tensor([[0,1,2,3,4],[1,2,3,4,5]])
    e2 = torch.tensor([[1,2,3,4,5],[0,0,0,0,0]])
    return [e1, e2]