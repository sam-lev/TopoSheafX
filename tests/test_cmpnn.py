import torch
from models.copresheaf import CopresheafMap, CopresheafMessagePassing
from models.cmpnn_base import CMPNN, CHOMPNN

def test_copresheaf_map_forward():
    model = CopresheafMap(in_dim=8, out_dim=16)
    h_src = torch.randn(10, 8)
    h_tgt = torch.randn(10, 8)
    out = model(h_src, h_tgt)
    assert out.shape == (10, 16)

def test_copresheaf_message_passing():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    x = torch.randn(3, 8)
    model = CopresheafMessagePassing(
        in_channels=8, out_channels=16, edge_index=edge_index,
        edge_map_factory=lambda in_dim, out_dim: torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, out_dim)
        )
    )
    out = model(x)
    assert out.shape == (3, 16)

def test_cmpnn_forward():
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_index_list = [edge_index]
    batch = torch.tensor([0, 0, 0, 1, 1])
    model = CMPNN(in_dim=8, hidden_dim=16, out_dim=2, num_layers=2)
    out = model(x, edge_index_list, batch)
    assert out.shape == (2, 2)

def test_chompnn_forward():
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_index_list = [edge_index]
    batch = torch.tensor([0, 0, 0, 1, 1])
    model = CHOMPNN(in_dim=8, hidden_dim=16, out_dim=2, num_layers=2)
    out = model(x, edge_index_list, batch)
    assert out.shape == (2, 2)
