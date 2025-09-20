# train_model.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import torch
from torch_geometric.loader import DataLoader

from data.graph_datasets import load_complex_dataset  # You implement this
from toposheafx.nn.model import CMPNN
from train.trainer import evaluate, train


def main(args):
    # Load higher-order combinatorial complex dataset
    dataset = load_complex_dataset(args.dataset)
    num_classes = dataset.num_classes
    input_dim = dataset.num_node_features
    sample = dataset[0]
    edge_attr_dim = (
        sample.edge_attr_list[0].shape[-1] if sample.edge_attr_list[0] is not None else 0
    )
    num_neighborhoods = len(sample.edge_index_list)

    # Split dataset
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    split1 = int(0.8 * len(dataset))
    split2 = int(0.9 * len(dataset))
    train_dataset = dataset[:split1]
    val_dataset = dataset[split1:split2]
    test_dataset = dataset[split2:]

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model setup
    model = CMPNN(
        in_dim=input_dim,
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        num_layers=args.num_layers,
        edge_attr_dim=edge_attr_dim,
        num_neighborhoods=num_neighborhoods,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, args.device)
        val_acc = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch:03d}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}")

    # Final test accuracy
    test_acc = evaluate(model, test_loader, args.device)
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset to load")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    main(args)
