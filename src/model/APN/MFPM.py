import torch.nn as nn
import torch.nn.functional as F

from src.config import CFG


class MFPM(nn.Module):
    def __init__(self, in_channels, decoder_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.in_proj = nn.ModuleList()
        if decoder_channels is None:
            decoder_channels = list(CFG["mfpm"]["decoder_channels"])
        self._use_decoder = len(decoder_channels) > 0

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        _dr: list[int] = list(CFG["mfpm"]["dilation_rates"])
        self.dconv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=_dr[0], dilation=_dr[0]),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.dconv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=_dr[1], dilation=_dr[1]),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.dconv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=_dr[2], dilation=_dr[2]),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Paper Eq. (7): sum multi-scale + global context, then 1x1 conv
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.ModuleList()
        if self._use_decoder:
            prev = in_channels
            for i, ch in enumerate(decoder_channels):
                k = 1 if i == len(decoder_channels) - 1 else 3
                pad = 0 if k == 1 else 1
                conv = nn.Conv2d(prev, ch, kernel_size=k, padding=pad, bias=False)
                if i == len(decoder_channels) - 1:
                    block = nn.Sequential(conv)
                else:
                    block = nn.Sequential(
                        conv,
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True),
                    )
                self.decoder.append(block)
                prev = ch
            self.out_channels = decoder_channels[-1]
        else:
            self.out_channels = in_channels

    def _fuse_inputs(self, x):
        # Accept either a single feature map or a list/tuple of feature maps.
        if not isinstance(x, (list, tuple)):
            return x
        feats = list(x)
        if len(feats) == 0:
            raise ValueError("MFPM received an empty feature list.")
        target_h = max(f.shape[2] for f in feats)
        target_w = max(f.shape[3] for f in feats)
        fused = None
        for i, feat in enumerate(feats):
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            if i >= len(self.in_proj) or self.in_proj[i].in_channels != feat.shape[1]:
                proj = nn.Conv2d(feat.shape[1], self.in_channels, kernel_size=1, bias=False)
                proj = proj.to(device=feat.device, dtype=feat.dtype)
                self.in_proj.append(proj)
            else:
                # Ensure projection matches current device/dtype
                self.in_proj[i] = self.in_proj[i].to(device=feat.device, dtype=feat.dtype)
            feat = self.in_proj[i](feat)
            fused = feat if fused is None else fused + feat
        return fused

    def forward(self, x):
        x = self._fuse_inputs(x)
        f1 = self.conv1x1(x)
        f2 = self.dconv2(x)
        f3 = self.dconv3(x)
        f4 = self.dconv4(x)

        f5 = self.global_context(x)
        f5 = F.interpolate(f5, size=x.shape[2:], mode="bilinear", align_corners=False)

        fused = f1 + f2 + f3 + f4 + f5
        x = self.fusion(fused)
        if self._use_decoder:
            _, _, H, W = x.shape
            for block in self.decoder:
                x = block(x)
                x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x
