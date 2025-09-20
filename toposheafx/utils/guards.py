# guards.py
import torch


def _check_index(index, num_nodes, name="index"):
    if index.dtype != torch.long:
        raise TypeError(f"{name} must be torch.long (got {index.dtype}).")

    # Handle empty index tensor
    if index.numel() == 0:
        return  # Empty index is valid, nothing to check

    if index.min().item() < 0 or index.max().item() >= num_nodes:
        raise ValueError(f"{name} has values outside [0, {num_nodes - 1}].")


def _same_device(*tensors):
    devs = [t.device for t in tensors if isinstance(t, torch.Tensor)]
    if len(set(devs)) > 1:
        raise RuntimeError(f"All tensors must be on same device (got {set(devs)}).")
