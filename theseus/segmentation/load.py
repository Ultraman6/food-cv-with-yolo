import torch


def load_seg(weight):
    """Load YOLOv5 model."""
    return torch.load(weight)