import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CFG


class PAFM(nn.Module):
    """
    Pyramid conv branches + global context (GAP -> 1x1 -> upsample),
    fuse into attention map, then pixel-wise multiply with features.

    Pyramid kernel sizes and post-processing kernel are read from
    ``CFG["pafm"]["pyramid_kernel_sizes"]``. The largest kernel is reused
    as the post-processing convolution (as in the MSPA paper).
    """
    def __init__(self, in_channels: int, proj_channels: int = CFG["pafm"]["proj_channels"]):
        super().__init__()
        self.proj_channels = proj_channels

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
        )

        _ks: list[int] = list(CFG["pafm"]["pyramid_kernel_sizes"])
        _pk: int = CFG["pafm"]["post_conv_kernel_size"]
        self.conv3 = nn.Conv2d(proj_channels, proj_channels, kernel_size=_ks[0], padding=_ks[0] // 2, bias=False)
        self.conv5 = nn.Conv2d(proj_channels, proj_channels, kernel_size=_ks[1], padding=_ks[1] // 2, bias=False)
        self.conv7 = nn.Conv2d(proj_channels, proj_channels, kernel_size=_ks[2], padding=_ks[2] // 2, bias=False)
        # Post-processing conv after each pyramid branch (as described in MSPA PAFM)
        self.post7_3 = nn.Conv2d(proj_channels, proj_channels, kernel_size=_pk, padding=_pk // 2, bias=False)
        self.post7_5 = nn.Conv2d(proj_channels, proj_channels, kernel_size=_pk, padding=_pk // 2, bias=False)
        self.post7_7 = nn.Conv2d(proj_channels, proj_channels, kernel_size=_pk, padding=_pk // 2, bias=False)
        self.bn_pyr = nn.BatchNorm2d(proj_channels)

        self.gc = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(proj_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # pyramid path
        p = self.proj(x)
        p3 = self.post7_3(self.conv3(p))
        p5 = self.post7_5(self.conv5(p))
        p7 = self.post7_7(self.conv7(p))
        p_sum = p3 + p5 + p7
        p_sum = F.relu(self.bn_pyr(p_sum), inplace=True)

        # global context path
        gc = F.adaptive_avg_pool2d(x, 1)   # (B,C,1,1)
        gc = self.gc(gc)                   # (B,proj,1,1)
        gc = F.interpolate(gc, size=(H, W), mode="bilinear", align_corners=False)

        # fuse -> attention
        att = self.fuse(p_sum + gc)
        att = torch.sigmoid(att)           # attention gate in [0,1]

        return x * att
