import torch


def load_cls(weight):
    """Load YOLOv5 model."""
    return torch.load(weight)