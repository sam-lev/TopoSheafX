import torch
import types

from train.trainer import train, evaluate
from models.cmpnn_base import CMPNN

class _TinyDataset(torch.utils.data.Dataset):
    """
    Minimal dataset compatible with train()/evaluate():
    each item must have x, edge_index_list, edge_attr_list (optional), batch, y, num_graphs
    """
    def __init__(self, n_graphs=4, nodes_per_graph=4, in_dim=6, num_neighborhoods=1, num_classes=3):
        self.items = []
        for g in range(n_graphs):
            num_nodes = nodes_per_graph
            x = torch.randn(num_nodes, in_dim)
            # simple ring
            row = torch.arange(num_nodes)
            col = (row + 1) % num_nodes
            e = torch.stack([row, col], dim=0)
            data = types.SimpleNamespace()
            data.x = x
            data.edge_index_list = [e] * num_neighborhoods
            data.edge_attr_list = None
            data.batch = torch.zeros(num_nodes, dtype=torch.long)  # one graph per item
            data.y = torch.randint(0, num_classes, (1,))
            data.num_graphs = 1
            self.items.append(data)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def test_trainer_functions_smoke():
    device = torch.device("cpu")
    in_dim, hidden, out_dim = 6, 8, 3
    model = CMPNN(in_dim, hidden, out_dim, num_layers=1, num_neighborhoods=1).to(device)

    ds = _TinyDataset(n_graphs=6, nodes_per_graph=5, in_dim=in_dim, num_neighborhoods=1, num_classes=out_dim)
    # simple DataLoader replacement: the trainer expects PyG DataLoader but only uses iteration
    loader = ds  # iterable is enough for our smoke test
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = train(model, loader, opt, device)
    assert isinstance(train_loss, float)

    acc = evaluate(model, loader, device)
    assert 0.0 <= acc <= 1.0
