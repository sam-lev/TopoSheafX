# trainer.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any

def _move_to_device(obj: Any, device: torch.device):
    if hasattr(obj, "to") and callable(getattr(obj, "to")):
        try:
            return obj.to(device)
        except Exception:
            pass  # fall through to manual handling
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        typ = type(obj)
        return typ(_move_to_device(v, device) for v in obj)
    # support SimpleNamespace-like
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            setattr(obj, k, _move_to_device(v, device))
        return obj
    return obj

def train(model: nn.Module, dataloader, optimizer, device):

    """
    One epoch of supervised training for a cmpnn model.

    Args:
        model (nn.Module): cmpnn model instance.
        dataloader (DataLoader): DataLoader providing batched higher-order graphs.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        device (torch.device): Device to train on.

    Returns:
        float: Average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for data in dataloader:
        data = _move_to_device(data, device)   # ← replace: data = data.to(device)

        # Expecting the batch object to contain:
        # - x: node features
        # - edge_index_list: list of [2, num_edges] tensors
        # - edge_attr_list: optional list of edge attribute tensors
        # - batch: [num_nodes] vector mapping nodes to graphs
        # - y: labels

        optimizer.zero_grad()

        out = model(
            x=data.x,
            edge_index_list=data.edge_index_list,
            batch=data.batch,
            edge_attr_list=getattr(data, 'edge_attr_list', None)
        )

        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

        # weight by how many graphs were in this batch
        total_loss += loss.item() * data.num_graphs
        total_graphs += int(data.num_graphs)

    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model accuracy on a validation or test set.

    Args:
        model (nn.Module): cmpnn model instance.
        dataloader (DataLoader): DataLoader for evaluation.
        device (torch.device): Device to evaluate on.

    Returns:
        float: Accuracy over the entire dataset.
    """
    model.eval()
    total_correct = 0
    total = 0
    total_graphs = 0

    for data in dataloader:
        data = _move_to_device(data, device)

        out = model(
            x=data.x,
            edge_index_list=data.edge_index_list,
            batch=data.batch,
            edge_attr_list=getattr(data, 'edge_attr_list', None)
        )

        # predictions and accuracy
        preds = out.argmax(dim=-1)
        correct = (preds == data.y).sum().item()
        num_g = int(data.num_graphs)

        # total_loss += loss.item() * num_g
        total_correct += correct
        total_graphs += num_g

    # avg_loss = total_loss / max(total_graphs, 1)
    avg_acc = total_correct / max(total_graphs, 1)
    return avg_acc