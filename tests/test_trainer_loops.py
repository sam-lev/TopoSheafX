# tests/test_trainer_loops.py
import types

import torch

from toposheafx.nn.model import CMPNN
from train.trainer import evaluate, train


class TinyDataset(torch.utils.data.IterableDataset):
    """
    Minimal iterable yielding items compatible with trainer.py expectations:
      .x, .edge_index_list, .edge_attr_list (None), .batch, .y, .num_graphs
    """

    def __init__(self, n_items=6, nodes_per=6, in_dim=6, num_neighborhoods=1, num_classes=3):
        super().__init__()
        self.n_items = n_items
        self.nodes_per = nodes_per
        self.in_dim = in_dim
        self.num_neighborhoods = num_neighborhoods
        self.num_classes = num_classes

    def __iter__(self):
        for _ in range(self.n_items):
            x = torch.randn(self.nodes_per, self.in_dim)
            row = torch.arange(self.nodes_per)
            col = (row + 1) % self.nodes_per
            e = torch.stack([row, col], dim=0)
            item = types.SimpleNamespace()
            item.x = x
            item.edge_index_list = [e] * self.num_neighborhoods
            item.edge_attr_list = None
            item.batch = torch.zeros(self.nodes_per, dtype=torch.long)
            item.y = torch.randint(0, self.num_classes, (1,))
            item.num_graphs = 1
            yield item


def test_train_and_evaluate_smoke():
    device = torch.device("cpu")
    in_dim, hidden, out_dim = 6, 12, 3
    model = CMPNN(in_dim, hidden, out_dim, num_layers=2, num_neighborhoods=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    ds = TinyDataset(
        n_items=8, nodes_per=5, in_dim=in_dim, num_neighborhoods=1, num_classes=out_dim
    )

    # one epoch
    loss1 = train(model, ds, opt, device)
    acc1 = evaluate(model, ds, device)

    # a second epoch — loss should be finite; accuracy between [0,1]
    loss2 = train(model, ds, opt, device)
    acc2 = evaluate(model, ds, device)

    assert isinstance(loss1, float) and isinstance(loss2, float)
    assert 0.0 <= acc1 <= 1.0 and 0.0 <= acc2 <= 1.0
