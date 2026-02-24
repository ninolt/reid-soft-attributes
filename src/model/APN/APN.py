import torch.nn as nn

from src.config import CFG
from src.model.APAN.GAP import GAP

from .LTN import LTN
from .MFPM import MFPM


class APN(nn.Module):
    """
    Appearance Pyramid Network (APN).
    MFPM -> LTN -> GAP -> BNNeck -> FC -> ID logits.
    Uses BNNeck (Bag of Tricks for ReID) for better performance.
    """
    def __init__(
        self,
        in_channels: int = CFG["apn"]["in_channels"],
        num_classes: int = CFG["iqa"]["num_classes"],
        feature_dim: int = CFG["apn"]["feature_dim"],
        dropout_rate: float = CFG["model"]["dropout_rate"],
    ):
        super().__init__()
        self.mfpm = MFPM(in_channels)
        self.ltn = LTN(self.mfpm.out_channels, mid_channels=CFG["apn"]["ltn_mid_channels"], out_channels=CFG["apn"]["ltn_out_channels"])
        self.gap = GAP()
        
        # Reduce from ltn_out_channels (LTN output) to feature_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(CFG["apn"]["ltn_out_channels"], feature_dim),
            nn.BatchNorm1d(feature_dim),
        )
        
        # BNNeck - critical for ReID (Bag of Tricks paper)
        # Feature for triplet loss (before BN) vs classifier (after BN)
        self.bn_neck = nn.BatchNorm1d(feature_dim)
        self.bn_neck.bias.requires_grad_(False)  # No bias in BNNeck
        
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)  # No bias with BNNeck
        nn.init.normal_(self.classifier.weight, std=CFG["model"]["classifier_weight_std"])

    def forward(self, x, is_train: bool = True):
        # x should be layer3 feature map (1024 channels) or list of feature maps
        if isinstance(x, (list, tuple)):
            # Take the last one (highest resolution with most channels) - typically layer3
            x = x[-1] if len(x) > 0 else x
        
        x = self.mfpm(x)
        x = self.ltn(x)
        pooled = self.gap(x)  # (B, 1024)
        feat = self.bottleneck(pooled)  # (B, feature_dim) - for triplet loss
        feat_bn = self.bn_neck(feat)  # (B, feature_dim) - for classifier
        feat_bn = self.dropout(feat_bn)
        
        if is_train:
            logits = self.classifier(feat_bn)
            return logits, feat  # Return pre-BN features for triplet loss
        else:
            return None, feat  # Inference: return features for matching
        
