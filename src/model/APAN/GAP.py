import torch.nn as nn


class GAP(nn.Module):
    """
    Standard Global Average Pooling: (B, C, H, W) -> (B, C)
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(x).flatten(1)
