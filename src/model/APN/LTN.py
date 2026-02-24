import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CFG

# NOTE: In MSPA, the "LTN" block in the diagram corresponds to a
# Local Attention Network (LAM-based) rather than a spatial transformer.


class _ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=CFG["ltn"]["channel_attention_reduction"]):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx = self.mlp(self.max_pool(x))
        return self.sigmoid(avg + mx)


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size=CFG["ltn"]["spatial_attention_kernel_size"]):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        att = self.conv(torch.cat([avg, mx], dim=1))
        return self.sigmoid(att)


class LAMBlock(nn.Module):
    """
    Local Attention Module (LAM): two 3x3 convs + channel & spatial attention.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = _ChannelAttention(out_channels)
        self.sa = _SpatialAttention()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class LTN(nn.Module):
    """
    Local Attention Network (LTN in MSPA diagram):
    - Split feature map into 4 local feature maps (2x2 grid)
    - Apply two LAM stages (each stage has 4 blocks)
    - Recompose into full feature map
    """
    def __init__(self, in_channels, mid_channels=CFG["apn"]["ltn_mid_channels"], out_channels=CFG["apn"]["ltn_out_channels"]):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # Two comparable LAM stages, each with num_local_parts blocks
        _n: int = CFG["ltn"]["num_local_parts"]
        self.lam_stage1 = nn.ModuleList([LAMBlock(mid_channels, mid_channels) for _ in range(_n)])
        self.lam_stage2 = nn.ModuleList([LAMBlock(mid_channels, out_channels) for _ in range(_n)])

    def _split_2x2(self, x):
        _, _, h, w = x.shape
        h_mid = h // 2
        w_mid = w // 2
        return [
            x[:, :, :h_mid, :w_mid],
            x[:, :, :h_mid, w_mid:],
            x[:, :, h_mid:, :w_mid],
            x[:, :, h_mid:, w_mid:],
        ]

    def _merge_2x2(self, parts):
        top = torch.cat([parts[0], parts[1]], dim=3)
        bottom = torch.cat([parts[2], parts[3]], dim=3)
        return torch.cat([top, bottom], dim=2)

    def forward(self, x):
        x = self.reduce(x)
        parts = self._split_2x2(x)
        parts = [self.lam_stage1[i](p) for i, p in enumerate(parts)]
        parts = [self.lam_stage2[i](p) for i, p in enumerate(parts)]
        x = self._merge_2x2(parts)
        return x
