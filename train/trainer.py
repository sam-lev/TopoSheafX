# trainer.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def train(model, dataloader, optimizer, device):
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

    for data in tqdm(dataloader, desc="Training"):
        data = data.to(device)

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

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(dataloader.dataset)


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

    for data in tqdm(dataloader, desc="Evaluating"):
        data = data.to(device)

        out = model(
            x=data.x,
            edge_index_list=data.edge_index_list,
            batch=data.batch,
            edge_attr_list=getattr(data, 'edge_attr_list', None)
        )

        pred = out.argmax(dim=1)
        total_correct += (pred == data.y).sum().item()
        total += data.num_graphs

    return total_correct / total