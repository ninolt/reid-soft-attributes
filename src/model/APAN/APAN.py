import torch.nn as nn

from src.config import CFG

from .GAP import GAP
from .LSTM import LSTM
from .PAFM import PAFM


class APAN(nn.Module):
    """
    Attribute Pyramid Attention Network (APAN).
    PAFM -> ConvLSTM -> GAP -> BNNeck -> FC -> attribute logits.
    Uses BNNeck for better feature learning.
    """
    def __init__(
        self,
        in_channels: int = CFG["model"]["apan_in_channels"],
        feature_dim: int = CFG["apan"]["feature_dim"],
        lstm_hidden: int = CFG["apan"]["lstm_hidden"],
        dropout_rate: float = CFG["model"]["dropout_rate"],
    ):
        super().__init__()
        self.pafm = PAFM(in_channels, proj_channels=CFG["pafm"]["proj_channels"])
        self.lstm = LSTM(in_channels, lstm_hidden)
        self.gap = GAP()

        # Project LSTM output to feature dimension
        self.bottleneck = nn.Sequential(
            nn.Linear(lstm_hidden, feature_dim),
            nn.BatchNorm1d(feature_dim),
        )
        
        # BNNeck for attribute classification
        self.bn_neck = nn.BatchNorm1d(feature_dim)
        self.bn_neck.bias.requires_grad_(False)
        
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Separate heads for binary and multi-class attributes
        self.binary_keys: list[str] = list(CFG["apan"]["binary_keys"])
        self.multi_keys: dict[str, int] = dict(CFG["apan"]["multi_keys"])
        
        # Calculate total outputs
        total_binary = len(self.binary_keys)
        total_multi = sum(self.multi_keys.values())
        self.classifier = nn.Linear(feature_dim, total_binary + total_multi, bias=False)
        nn.init.normal_(self.classifier.weight, std=CFG["model"]["classifier_weight_std"])

    def forward(self, x):
        x = self.pafm(x)
        b, _, h, w = x.size()
        state = self.lstm.init_state(b, (h, w), x.device, x.dtype)
        x, _ = self.lstm(x, state)
        pooled = self.gap(x)
        feat = self.bottleneck(pooled)  # Features for fusion
        feat_bn = self.bn_neck(feat)
        feat_bn = self.dropout(feat_bn)
        
        logits = self.classifier(feat_bn)

        outputs_attr = {}
        idx = 0
        for k in self.binary_keys:
            outputs_attr[k] = logits[:, idx]
            idx += 1
        for k, ncls in self.multi_keys.items():
            outputs_attr[k] = logits[:, idx:idx + ncls]
            idx += ncls

        return outputs_attr, feat  # Return pre-BN features for fusion
